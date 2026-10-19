import numpy as np

from src.core.clock_network import SimNetwork, VirtualClock
from src.core.raft_node import RaftNode, Role
from src.ml.fast_loop import FastLoopNodeEngine, SplitFailureClassifier
from src.ml.slow_loop import EncoderReport, SlowFederatedAggregator


def make_cluster(size=3):
    clock = VirtualClock()
    network = SimNetwork(clock, 31)
    ids = [str(index) for index in range(size)]
    nodes = {node_id: RaftNode(node_id, [p for p in ids if p != node_id],
                               clock, network, np.random.default_rng(50 + index),
                               heartbeat_interval_ms=40)
             for index, node_id in enumerate(ids)}
    for node in nodes.values():
        node.start()
    clock.advance_to(500)
    leader = next(node for node in nodes.values() if node.role is Role.LEADER)
    return clock, leader


def reports_from(weights, count=5):
    return [EncoderReport(str(index),
                          {name: value + index * 0.001 for name, value in weights.items()},
                          0.9, 0.0, 100, np.array([0.2, 0.1]))
            for index in range(count)]


def test_trimmed_mean_rejects_corrupted_peer():
    model = SplitFailureClassifier(seed=3)
    reports = reports_from(model.get_encoder_weights())
    reports[-1].encoder_weights = {
        name: np.full_like(value, 1e6) for name, value in reports[-1].encoder_weights.items()}
    aggregated = SlowFederatedAggregator().aggregate(reports)
    assert all(np.max(np.abs(value)) < 10 for value in aggregated.values())


def test_drift_downweighting_and_raft_log_commit():
    clock, leader = make_cluster()
    model = SplitFailureClassifier(seed=4)
    reports = reports_from(model.get_encoder_weights())
    reports[0].drift_score = 1.0
    aggregator = SlowFederatedAggregator()
    assert aggregator.aggregation_weight(reports[0]) < aggregator.aggregation_weight(reports[1])
    aggregator.aggregate(reports, raft_leader=leader)
    proposed_index = len(leader.log) - 1
    clock.advance_to(650)
    assert leader.commit_index >= proposed_index
    payload = leader.log[proposed_index].value
    assert payload["type"] == "encoder_version"
    assert payload["checksum"]


def test_warm_start_cold_node():
    clock, leader = make_cluster()
    donor = SplitFailureClassifier(seed=42)
    rng = np.random.default_rng(7)
    candidates = rng.normal(size=(2_000, 10))
    sample = next(x for x in candidates if np.argmax(donor.predict_proba(x)) == 1)
    reports = reports_from(donor.get_encoder_weights())
    aggregator = SlowFederatedAggregator()
    aggregator.aggregate(reports, raft_leader=leader)
    clock.advance_to(650)

    cold = FastLoopNodeEngine(SplitFailureClassifier(seed=42))
    cold.classifier.set_encoder_weights({
        name: np.zeros_like(value) for name, value in cold.classifier.get_encoder_weights().items()})
    assert np.argmax(cold.classifier.predict_proba(sample)) != 1
    version = aggregator.warm_start_node(cold)
    assert version.startswith("encoder-")
    assert cold.trained_labels == 0
    assert np.argmax(cold.classifier.predict_proba(sample)) == 1

import numpy as np

from src.core.clock_network import SimNetwork, VirtualClock
from src.core.types import FaultEvent, FaultType
from src.ml.features import TelemetryExtractor
from src.ml.self_labeler import FailureLabel, LabelQualityEvaluator, ProtocolSelfLabeler


def feature_vector(local_lag=0.0):
    extractor = TelemetryExtractor(heartbeat_interval_ms=100)
    for index in range(20):
        extractor.observe("target", index * 100, rtt_ms=20 + index % 3,
                          witnesses_alive=2, witnesses_queried=3,
                          local_health_lag_ms=local_lag)
    result = extractor.vector("target")
    assert result.shape == (10,)
    assert np.all((0 <= result) & (result <= 1))
    return result


def test_link_failure_self_label():
    clock = VirtualClock()
    network = SimNetwork(clock, 42)
    network.add_fault(FaultEvent(0, 600, FaultType.LINK_DROP, ("observer", "target")))
    labeler = ProtocolSelfLabeler()
    labeler.suspect(0, "target", feature_vector(), known_boot_id=4, known_log_index=8)
    clock.advance_to(600)
    result = labeler.observe_direct(clock.now_ms, "target", boot_id=4, log_index=8)
    assert result.label is FailureLabel.LINK_FAILURE
    assert result.confidence_weight >= 0.8


def test_crash_recovery_and_permanent_crash_labels():
    labeler = ProtocolSelfLabeler(horizon_ms=2_000)
    labeler.suspect(0, "restarted", feature_vector(), known_boot_id=2)
    recovered = labeler.observe_direct(500, "restarted", boot_id=3, log_index=0)
    assert recovered.label is FailureLabel.CRASH
    assert recovered.confidence_weight == 1.0

    labeler.suspect(0, "dead", feature_vector(), known_boot_id=1,
                    expected_witnesses=3)
    for witness in ("w1", "w2", "w3"):
        labeler.observe_witness(1_900, "dead", witness, True, False, None)
    expired = labeler.expire(2_001)
    assert expired[0].label is FailureLabel.CRASH
    assert expired[0] in labeler.training_labels


def test_unresolved_local_isolation():
    labeler = ProtocolSelfLabeler(horizon_ms=2_000)
    labeler.suspect(0, "leader", feature_vector(), known_boot_id=0,
                    expected_witnesses=3)
    result = labeler.expire(2_001)[0]
    assert result.label is FailureLabel.UNRESOLVED
    assert result not in labeler.training_labels
    assert not labeler.training_labels


def test_healthy_slow_label_uses_local_health_signal():
    labeler = ProtocolSelfLabeler()
    labeler.suspect(0, "target", feature_vector(50), 0, heartbeat_interval_ms=100)
    result = labeler.observe_direct(130, "target", 0, -1,
                                    heartbeat_delay_ms=130,
                                    local_health_lag_ms=50)
    assert result.label is FailureLabel.HEALTHY_SLOW


def test_protocol_label_noise_rate():
    clock = VirtualClock()
    network = SimNetwork(clock, 42)
    labeler = ProtocolSelfLabeler()
    vector = feature_vector()
    for index in range(200):
        started = index * 3_000.0
        peer = f"peer-{index}"
        if index % 2 == 0:
            network.add_fault(FaultEvent(started, started + 600, FaultType.LINK_DROP,
                                         ("observer", peer)))
            labeler.suspect(started, peer, vector, known_boot_id=0)
            labeler.observe_direct(started + 600, peer, 0, -1)
        else:
            network.add_fault(FaultEvent(started, started + 500, FaultType.NODE_CRASH,
                                         (peer,)))
            labeler.suspect(started, peer, vector, known_boot_id=0)
            labeler.observe_direct(started + 500, peer, 1, -1)
    quality = LabelQualityEvaluator().evaluate(labeler.training_labels, network)
    assert quality.evaluated == 200
    assert quality.agreement > 0.90
    assert quality.noise_rate < 0.10

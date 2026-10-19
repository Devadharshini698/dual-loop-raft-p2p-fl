import numpy as np

from src.core.clock_network import SimNetwork, VirtualClock
from src.core.raft_node import RaftNode, Role
from src.core.types import FaultEvent, FaultType


def cluster(size=5, seed=42):
    clock = VirtualClock()
    network = SimNetwork(clock, seed)
    ids = [str(index) for index in range(1, size + 1)]
    nodes = {
        node_id: RaftNode(node_id, [peer for peer in ids if peer != node_id],
                          clock, network, np.random.default_rng(seed + int(node_id)),
                          heartbeat_interval_ms=40)
        for node_id in ids
    }
    for node in nodes.values():
        node.start()
    return clock, network, nodes


def leaders(nodes, network):
    return [node for node in nodes.values()
            if node.role is Role.LEADER and node.node_id not in network.crashed]


def test_leader_election_succeeds_within_500ms():
    clock, network, nodes = cluster()
    clock.advance_to(500)
    assert len(leaders(nodes, network)) == 1


def test_leader_crash_triggers_new_term():
    clock, network, nodes = cluster()
    clock.advance_to(500)
    first = leaders(nodes, network)[0]
    first_term = first.current_term
    network.crash_node(first.node_id)
    clock.advance_to(1_000)
    replacement = leaders(nodes, network)
    assert len(replacement) == 1
    assert replacement[0].node_id != first.node_id
    assert replacement[0].current_term > first_term


def test_witnesses_see_leader_during_asymmetric_drop():
    clock, network, nodes = cluster()
    clock.advance_to(500)
    leader = leaders(nodes, network)[0]
    follower = next(node for node in nodes.values() if node is not leader)
    network.add_fault(FaultEvent(500, 800, FaultType.ASYMMETRIC_LOSS,
                                 (leader.node_id, follower.node_id),
                                 {"loss_ratio": 1.0}))
    clock.advance_to(610)
    follower.query_witnesses(leader.node_id, k=3)
    clock.advance_to(640)
    assert follower.query_witnesses(leader.node_id, k=3)


def test_majority_commit_replicates_lightweight_entry():
    clock, network, nodes = cluster()
    clock.advance_to(500)
    leader = leaders(nodes, network)[0]
    index = leader.propose({"model_version": "sha256:abc", "weights": [1, 2, 3]})
    clock.advance_to(600)
    assert leader.commit_index >= index
    committed = [node.log[index] for node in nodes.values() if node.commit_index >= index]
    assert len(committed) >= 3
    assert all(entry == committed[0] for entry in committed)

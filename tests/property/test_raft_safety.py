import numpy as np
import pytest

from src.core.clock_network import SimNetwork, VirtualClock
from src.core.raft_node import RaftNode, Role


class WildPlanner:
    def __init__(self, rng):
        self.rng = rng

    def get_timeout_decision(self, node, leader_id):
        return float(self.rng.uniform(20, 2_000))


class FixedPlanner:
    def __init__(self, timeout_ms):
        self.timeout_ms = timeout_ms

    def get_timeout_decision(self, node, leader_id):
        return self.timeout_ms


@pytest.mark.parametrize("timeout_ms", [1, 5_000])
def test_extreme_timeout_advice_cannot_break_election_safety(timeout_ms):
    clock = VirtualClock()
    network = SimNetwork(clock, 7)
    ids = [str(index) for index in range(5)]
    nodes = [RaftNode(node_id, [p for p in ids if p != node_id], clock, network,
                      np.random.default_rng(index), FixedPlanner(timeout_ms), 40)
             for index, node_id in enumerate(ids)]
    for node in nodes:
        node.start()
    leaders_by_term = {}
    for _ in range(500):
        clock.advance_to(clock.now_ms + 1)
        for node in nodes:
            if node.role is Role.LEADER:
                assert leaders_by_term.setdefault(node.current_term, node.node_id) == node.node_id


def test_election_and_log_safety_under_deterministic_chaos():
    clock = VirtualClock()
    network = SimNetwork(clock, 42)
    network.default_link.loss_ratio = 0.2
    network.default_link.sigma_ms = 8
    ids = [str(index) for index in range(5)]
    nodes = {}
    for index, node_id in enumerate(ids):
        rng = np.random.default_rng(1_000 + index)
        nodes[node_id] = RaftNode(node_id, [p for p in ids if p != node_id],
                                  clock, network, rng, WildPlanner(rng), 40)
        nodes[node_id].start()
    chaos = np.random.default_rng(42)
    leaders_by_term = {}
    proposed_terms = set()
    for tick in range(10_000):
        if tick in set(range(500, 9_501, 1_000)):
            active = [n for n in nodes.values()
                      if n.role is Role.LEADER and n.node_id not in network.crashed]
            if active:
                victim = active[0].node_id
                network.crash_node(victim)
                clock.call_later(100, lambda v=victim: network.reboot_node(v))
        if chaos.random() < 0.002:
            active = [n for n in nodes.values()
                      if n.role is Role.LEADER and n.node_id not in network.crashed]
            if active and active[0].current_term not in proposed_terms:
                active[0].propose(f"term-{active[0].current_term}")
                proposed_terms.add(active[0].current_term)
        clock.advance_to(clock.now_ms + 1)
        for node in nodes.values():
            if node.role is Role.LEADER and node.node_id not in network.crashed:
                previous = leaders_by_term.setdefault(node.current_term, node.node_id)
                assert previous == node.node_id
    committed = {}
    for node in nodes.values():
        for index in range(node.commit_index + 1):
            value = node.log[index]
            assert committed.setdefault(index, value) == value

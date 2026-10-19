from src.core.clock_network import SimNetwork, VirtualClock
from src.core.types import FaultEvent, FaultType, Message, MessageType


def replay(seed: int):
    clock = VirtualClock()
    network = SimNetwork(clock, seed)
    network.set_link("A", "B", mean_ms=8, sigma_ms=3, loss_ratio=0.2)
    delivered = []
    network.register("B", lambda message: delivered.append((clock.now_ms, message.log_index)))
    for index in range(1_000):
        network.send(Message("A", "B", 1, MessageType.HEARTBEAT, log_index=index))
    clock.run()
    records = [(r.delivery_at, r.dropped, r.reason) for r in network.records]
    return delivered, records


def test_deterministic_replay():
    first = replay(42)
    second = replay(42)
    assert first == second
    assert len(first[0]) + sum(record[1] for record in first[1]) == 1_000


def test_asymmetric_link_and_crash_incarnation():
    clock = VirtualClock()
    network = SimNetwork(clock, 42)
    received = []
    for node in ("A", "B", "C"):
        network.register(node, lambda message, n=node: received.append((n, message.sender_id)))
    network.add_fault(FaultEvent(0, 100, FaultType.ASYMMETRIC_LOSS,
                                 ("A", "B"), {"loss_ratio": 1.0}))
    assert not network.send(Message("A", "B", 0, MessageType.HEARTBEAT))
    assert network.send(Message("B", "C", 0, MessageType.HEARTBEAT))
    assert network.send(Message("C", "A", 0, MessageType.HEARTBEAT))
    clock.run()
    assert set(received) == {("C", "B"), ("A", "C")}
    network.add_fault(FaultEvent(110, 120, FaultType.NODE_CRASH, ("B",)))
    clock.advance_to(110)
    assert "B" in network.crashed
    clock.advance_to(120)
    assert network.boot_ids["B"] == 1
    assert "B" not in network.crashed


def test_drift_schedule_shift():
    clock = VirtualClock()
    network = SimNetwork(clock, 42)
    network.set_link("A", "B", 5, 1, 0)
    network.add_fault(FaultEvent(50, 100, FaultType.DRIFT_SHIFT, ("A", "B"),
                                 {"mean_ms": 30, "sigma_ms": 4, "loss_ratio": 0.25}))
    clock.advance_to(49.999)
    assert network.links[("A", "B")].mean_ms == 5
    clock.advance_to(50)
    profile = network.links[("A", "B")]
    assert (profile.mean_ms, profile.sigma_ms, profile.loss_ratio) == (30, 4, 0.25)

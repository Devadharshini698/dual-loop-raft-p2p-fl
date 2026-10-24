"""Validated configuration for a physical Raft/Flower laboratory cluster."""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class NodeSpec:
    node_id: str
    host: str
    raft_port: int
    flower_partition: int
    data_interface: str
    management_interface: str = ""
    ssh_user: str = ""

    @property
    def raft_address(self) -> tuple[str, int]:
        return self.host, self.raft_port

    @property
    def ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.host}" if self.ssh_user else self.host


@dataclass(slots=True, frozen=True)
class LabConfig:
    cluster_id: str
    nodes: tuple[NodeSpec, ...]
    heartbeat_interval_ms: float = 100.0
    state_root: str = "state"
    evidence_root: str = "runs"
    controller_host: str = "127.0.0.1"
    flower_superlink_port: int = 8000

    def node(self, node_id: str) -> NodeSpec:
        matches = [node for node in self.nodes if node.node_id == node_id]
        if not matches:
            raise KeyError(f"unknown node: {node_id}")
        return matches[0]

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.cluster_id):
            raise ValueError("cluster_id must be a safe non-empty identifier")
        if len(self.nodes) < 3 or len({node.node_id for node in self.nodes}) != len(self.nodes):
            raise ValueError("the physical cluster needs at least three uniquely named nodes")
        addresses = [node.raft_address for node in self.nodes]
        partitions = [node.flower_partition for node in self.nodes]
        if len(set(addresses)) != len(addresses) or len(set(partitions)) != len(partitions):
            raise ValueError("Raft addresses and Flower partitions must be unique")
        if self.heartbeat_interval_ms <= 0:
            raise ValueError("heartbeat_interval_ms must be positive")
        if not 1 <= self.flower_superlink_port <= 65535:
            raise ValueError("invalid Flower SuperLink port")
        for node in self.nodes:
            _validate_node(node)


def _validate_node(node: NodeSpec) -> None:
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", node.node_id)
            or not 1 <= node.raft_port <= 65535):
        raise ValueError("invalid node id or Raft port")
    try:
        address = ipaddress.ip_address(node.host)
        if address.version != 4:
            raise ValueError("the UDP adapter currently requires IPv4 hosts")
    except ValueError:
        labels = node.host.split(".")
        if not all(part and len(part) <= 63 and part[0].isalnum()
                   and part[-1].isalnum() and part.replace("-", "").isalnum()
                   for part in labels):
            raise ValueError(f"invalid host: {node.host}") from None
    if node.flower_partition < 0:
        raise ValueError("Flower partitions must be non-negative")
    if node.ssh_user and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", node.ssh_user):
        raise ValueError("invalid SSH user")
    if (not re.fullmatch(r"[A-Za-z0-9_.:-]+", node.data_interface)
            or node.data_interface == node.management_interface):
        raise ValueError("data_interface must be set and separate from management_interface")


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("run_id must be a safe non-empty identifier")
    return run_id


def load_lab_config(path: str | Path) -> LabConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = tuple(NodeSpec(**item) for item in data.pop("nodes"))
    config = LabConfig(nodes=nodes, **data)
    config.validate()
    return config

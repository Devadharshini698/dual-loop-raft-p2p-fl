"""Drift-aware federated aggregation for shared encoder parameters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from src.core.raft_node import RaftNode
from src.ml.fast_loop import FastLoopNodeEngine


@dataclass(slots=True)
class EncoderReport:
    node_id: str
    encoder_weights: dict[str, np.ndarray]
    mean_confidence: float
    drift_score: float
    sample_count: int
    regime_signature: np.ndarray


@dataclass(slots=True)
class EncoderVersion:
    version_id: str
    checksum: str
    encoder_weights: dict[str, np.ndarray]
    log_index: int


class SlowFederatedAggregator:
    """Aggregate shared encoders and publish accepted versions through Raft."""

    def __init__(self, trim_ratio: float = 0.2, regime_radius: float = 0.25) -> None:
        if not 0 <= trim_ratio < 0.5 or regime_radius <= 0:
            raise ValueError("invalid aggregation settings")
        self.trim_ratio = float(trim_ratio)
        self.regime_radius = float(regime_radius)
        self.latest_version: EncoderVersion | None = None
        self._raft_leader: RaftNode | None = None
        self._version_counter = 0

    @staticmethod
    def aggregation_weight(report: EncoderReport) -> float:
        quality = (report.mean_confidence, report.drift_score)
        if (report.sample_count <= 0 or not all(np.isfinite(quality))
                or not 0 < report.mean_confidence <= 1
                or not 0 <= report.drift_score <= 1):
            raise ValueError("invalid report quality fields")
        drift_factor = max(0.05, 1.0 - 0.8 * report.drift_score)
        return report.sample_count * report.mean_confidence * drift_factor

    def aggregate(self, reports: list[EncoderReport], raft_leader: RaftNode | None = None,
                  target_regime: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if not reports:
            raise ValueError("at least one encoder report is required")
        selected = self._select_regime(reports, target_regime)
        self._validate_reports(selected)
        report_weights = np.asarray([self.aggregation_weight(item) for item in selected])
        aggregated = {
            name: self._weighted_trimmed_parameter(
                [item.encoder_weights[name] for item in selected], report_weights)
            for name in selected[0].encoder_weights
        }
        if raft_leader is not None:
            self._commit_version(aggregated, raft_leader)
        return {name: value.copy() for name, value in aggregated.items()}

    def warm_start_node(self, fast_engine: FastLoopNodeEngine) -> str:
        self._refresh_committed_version()
        if self.latest_version is None:
            raise RuntimeError("no committed encoder version is available")
        fast_engine.classifier.set_encoder_weights(
            self.latest_version.encoder_weights)
        return self.latest_version.version_id

    def _select_regime(self, reports: list[EncoderReport],
                       target_regime: np.ndarray | None) -> list[EncoderReport]:
        if target_regime is None or len(reports) < 6:
            return reports
        target = np.asarray(target_regime, dtype=float)
        if not target.size or not np.all(np.isfinite(target)):
            raise ValueError("target regime must be a finite non-empty vector")
        matches = [report for report in reports
                   if np.asarray(report.regime_signature).shape == target.shape
                   and np.linalg.norm(np.asarray(report.regime_signature) - target)
                   <= self.regime_radius]
        return matches if len(matches) >= 6 else reports

    @staticmethod
    def _validate_reports(reports: list[EncoderReport]) -> None:
        if not reports or len({report.node_id for report in reports}) != len(reports):
            raise ValueError("encoder reports require unique node identifiers")
        keys = set(reports[0].encoder_weights)
        shapes = {name: np.asarray(value).shape
                  for name, value in reports[0].encoder_weights.items()}
        if keys != {"W1", "b1", "W2", "b2"}:
            raise ValueError("reports must contain shared encoder parameters only")
        for report in reports:
            SlowFederatedAggregator.aggregation_weight(report)
            signature = np.asarray(report.regime_signature, dtype=float)
            if not signature.size or not np.all(np.isfinite(signature)):
                raise ValueError("regime signatures must be finite and non-empty")
            if set(report.encoder_weights) != keys:
                raise ValueError("encoder parameter keys differ")
            for name, value in report.encoder_weights.items():
                array = np.asarray(value, dtype=float)
                if array.shape != shapes[name] or not np.all(np.isfinite(array)):
                    raise ValueError(f"invalid encoder parameter {name}")

    def _weighted_trimmed_parameter(self, parameters: list[np.ndarray],
                                    weights: np.ndarray) -> np.ndarray:
        stacked = np.stack(parameters).astype(float, copy=False)
        flat = stacked.reshape(len(stacked), -1)
        order = np.argsort(flat, axis=0)
        sorted_values = np.take_along_axis(flat, order, axis=0)
        expanded_weights = np.broadcast_to(weights[:, None], flat.shape)
        sorted_weights = np.take_along_axis(expanded_weights, order, axis=0)
        trim = min(int(len(stacked) * self.trim_ratio), (len(stacked) - 1) // 2)
        stop = len(stacked) - trim if trim else len(stacked)
        kept_values, kept_weights = sorted_values[trim:stop], sorted_weights[trim:stop]
        denominator = kept_weights.sum(axis=0)
        if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
            raise ValueError("retained aggregation weight must be positive")
        result = (kept_values * kept_weights).sum(axis=0) / denominator
        return result.reshape(stacked.shape[1:])

    def _commit_version(self, weights: dict[str, np.ndarray], leader: RaftNode) -> None:
        checksum = self._checksum(weights)
        existing = sum(isinstance(entry.value, dict)
                       and entry.value.get("type") == "encoder_version"
                       for entry in leader.log)
        self._version_counter = max(self._version_counter, existing) + 1
        version_id = f"encoder-{self._version_counter:06d}-{checksum[:12]}"
        serial = {name: value.tolist() for name, value in weights.items()}
        payload = {"type": "encoder_version", "version_id": version_id,
                   "checksum": checksum, "encoder_weights": serial}
        index = leader.propose(payload)
        self._raft_leader = leader
        self.latest_version = EncoderVersion(version_id, checksum,
                                             self._copy_weights(weights), index)

    def _refresh_committed_version(self) -> None:
        leader = self._raft_leader
        if leader is None or leader.commit_index < 0:
            self.latest_version = None
            return
        committed = leader.log[:leader.commit_index + 1]
        for index in range(len(committed) - 1, -1, -1):
            value = committed[index].value
            if isinstance(value, dict) and value.get("type") == "encoder_version":
                weights = {name: np.asarray(array, dtype=float)
                           for name, array in value["encoder_weights"].items()}
                if self._checksum(weights) != value["checksum"]:
                    raise ValueError("committed encoder checksum mismatch")
                self.latest_version = EncoderVersion(
                    value["version_id"], value["checksum"], weights, index)
                return
        self.latest_version = None

    @staticmethod
    def _checksum(weights: dict[str, np.ndarray]) -> str:
        digest = hashlib.sha256()
        for name in sorted(weights):
            array = np.ascontiguousarray(weights[name], dtype=np.float64)
            digest.update(name.encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _copy_weights(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: np.asarray(value, dtype=float).copy()
                for name, value in weights.items()}

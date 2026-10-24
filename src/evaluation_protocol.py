"""Protocol-derived training labels for deterministic RQ1 evaluation."""

from __future__ import annotations

import numpy as np

from src.core.clock_network import SimNetwork, VirtualClock
from src.core.types import FaultEvent, FaultType
from src.ml.self_labeler import LabelQualityEvaluator, ProtocolSelfLabeler


def derive_protocol_dataset(features: np.ndarray, truth: np.ndarray,
                            seed: int = 42) -> tuple:
    """Resolve synthetic fault evidence through the production self-labeller."""
    vectors = np.asarray(features, dtype=float)
    labels = np.asarray(truth, dtype=int)
    if (vectors.ndim != 2 or vectors.shape[1] != 10
            or labels.shape != (len(vectors),)):
        raise ValueError("expected aligned feature and label arrays")
    network = SimNetwork(VirtualClock(), seed)
    labeler = ProtocolSelfLabeler()
    for index, (vector, label) in enumerate(zip(vectors, labels, strict=True)):
        started, peer = index * 3_000.0, f"rq1-peer-{index}"
        fault_type, duration = _fault(label)
        network.add_fault(FaultEvent(started, started + duration, fault_type,
                                     ("observer", peer)))
        labeler.suspect(started, peer, vector, known_boot_id=1)
        if index % 20 == 0:
            labeler.expire(started + labeler.horizon_ms)
        else:
            _resolve(labeler, started, peer, label)
    resolved = labeler.training_labels
    quality = LabelQualityEvaluator().evaluate(resolved, network)
    derived_x = np.stack([item.features for item in resolved])
    derived_y = np.asarray([int(item.label) for item in resolved])
    confidence = np.asarray([item.confidence_weight for item in resolved])
    coverage = len(resolved) / len(vectors)
    return derived_x, derived_y, confidence, quality.noise_rate, coverage


def _fault(label: int) -> tuple[FaultType, float]:
    mapping = {0: (FaultType.LINK_DROP, 600.0),
               1: (FaultType.NODE_CRASH, 500.0),
               2: (FaultType.LINK_DELAY, 130.0)}
    if int(label) not in mapping:
        raise ValueError("failure labels must be in [0, 2]")
    return mapping[int(label)]


def _resolve(labeler: ProtocolSelfLabeler, started: float,
             peer: str, label: int) -> None:
    if label == 0:
        labeler.observe_direct(started + 600, peer, 1, -1)
    elif label == 1:
        labeler.observe_direct(started + 500, peer, 2, -1)
    else:
        labeler.observe_direct(started + 130, peer, 1, -1,
                               heartbeat_delay_ms=130,
                               local_health_lag_ms=50)

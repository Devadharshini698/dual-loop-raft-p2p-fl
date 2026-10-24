"""Optional real-machine adapters for the deterministic research core."""

from .config import LabConfig, NodeSpec, load_lab_config
from .runtime import PhysicalNodeRuntime

__all__ = ["LabConfig", "NodeSpec", "PhysicalNodeRuntime", "load_lab_config"]

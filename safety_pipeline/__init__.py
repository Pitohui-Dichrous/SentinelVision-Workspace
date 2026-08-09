"""Pure-Python safety reasoning components for SentinelVision.

The package deliberately has no Qt, OpenCV, Torch, or network dependency so
the safety logic can be replayed and unit-tested independently from inference.
"""

from .config import (
    ConfigError,
    PPEConflictConfig,
    RiskConfig,
    SafetyPipelineConfig,
    TemporalConfig,
    TrackingConfig,
    config_fingerprint,
    load_safety_pipeline_config,
    load_safety_pipeline_config_checked,
)
from .clock import PipelineClock
from .pipeline import PPETemporalPipeline
from .ppe_conflict import PPEConflictResolver
from .journal import JsonlEventJournal
from .policy import ppe_evidence_enabled, stable_class_visible
from .spatial import filter_monitoring_domain
from .types import PPEState, RiskState

__all__ = [
    "ConfigError",
    "PPEConflictConfig",
    "PipelineClock",
    "PPEConflictResolver",
    "JsonlEventJournal",
    "PPEState",
    "PPETemporalPipeline",
    "RiskConfig",
    "RiskState",
    "SafetyPipelineConfig",
    "TemporalConfig",
    "TrackingConfig",
    "config_fingerprint",
    "filter_monitoring_domain",
    "load_safety_pipeline_config",
    "load_safety_pipeline_config_checked",
    "ppe_evidence_enabled",
    "stable_class_visible",
]

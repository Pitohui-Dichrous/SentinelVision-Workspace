"""Versioned configuration for the optional temporal safety pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


LOGGER = logging.getLogger(__name__)
SUPPORTED_MODES = {"baseline", "ppe_temporal"}


class ConfigError(ValueError):
    """Raised when a versioned safety-pipeline configuration is invalid."""


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError("%s must be a mapping" % field_name)
    return value


def _text(mapping: Mapping[str, Any], key: str, default: str) -> str:
    value = mapping.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("%s must be a non-empty string" % key)
    return value.strip()


def _integer(mapping: Mapping[str, Any], key: str, default: int, minimum: int) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError("%s must be an integer >= %d" % (key, minimum))
    return value


def _number(
    mapping: Mapping[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: Optional[float] = None,
) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("%s must be numeric" % key)
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError("%s must be finite" % key)
    if result < minimum or (maximum is not None and result > maximum):
        if maximum is None:
            raise ConfigError("%s must be >= %s" % (key, minimum))
        raise ConfigError("%s must be between %s and %s" % (key, minimum, maximum))
    return result


@dataclass(frozen=True)
class PPEConflictConfig:
    protected_class_id: str = "hat"
    unprotected_class_id: str = "person"
    uncertain_class_id: str = "ppe_uncertain"
    min_iou: float = 0.60
    score_margin: float = 0.20
    max_center_distance_ratio: float = 0.35
    min_area_ratio: float = 0.50

    @classmethod
    def from_mapping(cls, value: Any) -> "PPEConflictConfig":
        data = _mapping(value, "ppe.conflict")
        result = cls(
            protected_class_id=_text(data, "protected_class_id", cls.protected_class_id),
            unprotected_class_id=_text(data, "unprotected_class_id", cls.unprotected_class_id),
            uncertain_class_id=_text(data, "uncertain_class_id", cls.uncertain_class_id),
            min_iou=_number(data, "min_iou", cls.min_iou, 0.0, 1.0),
            score_margin=_number(data, "score_margin", cls.score_margin, 0.0, 1.0),
            max_center_distance_ratio=_number(
                data, "max_center_distance_ratio", cls.max_center_distance_ratio, 0.0, 2.0
            ),
            min_area_ratio=_number(data, "min_area_ratio", cls.min_area_ratio, 0.0, 1.0),
        )
        class_ids = {
            result.protected_class_id,
            result.unprotected_class_id,
            result.uncertain_class_id,
        }
        if len(class_ids) != 3:
            raise ConfigError("PPE protected, unprotected, and uncertain class IDs must differ")
        return result


@dataclass(frozen=True)
class TrackingConfig:
    iou_threshold: float = 0.30
    max_lost_frames: int = 6
    min_hits: int = 2
    smoothing_alpha: float = 0.70

    @classmethod
    def from_mapping(cls, value: Any) -> "TrackingConfig":
        data = _mapping(value, "ppe.tracking")
        return cls(
            iou_threshold=_number(data, "iou_threshold", cls.iou_threshold, 0.0, 1.0),
            max_lost_frames=_integer(data, "max_lost_frames", cls.max_lost_frames, 0),
            min_hits=_integer(data, "min_hits", cls.min_hits, 1),
            smoothing_alpha=_number(data, "smoothing_alpha", cls.smoothing_alpha, 0.0, 1.0),
        )


@dataclass(frozen=True)
class TemporalConfig:
    window_size: int = 8
    min_votes: int = 6
    ema_alpha: float = 0.35
    score_margin: float = 0.12
    min_stable_frames: int = 1
    recovery_frames: int = 3
    missing_tolerance: int = 3

    @classmethod
    def from_mapping(cls, value: Any) -> "TemporalConfig":
        data = _mapping(value, "ppe.temporal")
        result = cls(
            window_size=_integer(data, "window_size", cls.window_size, 1),
            min_votes=_integer(data, "min_votes", cls.min_votes, 1),
            ema_alpha=_number(data, "ema_alpha", cls.ema_alpha, 0.0, 1.0),
            score_margin=_number(data, "score_margin", cls.score_margin, 0.0, 1.0),
            min_stable_frames=_integer(data, "min_stable_frames", cls.min_stable_frames, 1),
            recovery_frames=_integer(data, "recovery_frames", cls.recovery_frames, 1),
            missing_tolerance=_integer(data, "missing_tolerance", cls.missing_tolerance, 0),
        )
        if result.min_votes > result.window_size:
            raise ConfigError("ppe.temporal.min_votes cannot exceed window_size")
        return result


@dataclass(frozen=True)
class RiskConfig:
    suspect_frames: int = 1
    confirm_frames: int = 2
    recovery_frames: int = 4
    cooldown_seconds: float = 10.0

    @classmethod
    def from_mapping(cls, value: Any) -> "RiskConfig":
        data = _mapping(value, "ppe.risk")
        return cls(
            suspect_frames=_integer(data, "suspect_frames", cls.suspect_frames, 1),
            confirm_frames=_integer(data, "confirm_frames", cls.confirm_frames, 1),
            recovery_frames=_integer(data, "recovery_frames", cls.recovery_frames, 1),
            cooldown_seconds=_number(data, "cooldown_seconds", cls.cooldown_seconds, 0.0),
        )


@dataclass(frozen=True)
class SafetyPipelineConfig:
    schema_version: int = 1
    default_mode: str = "baseline"
    conflict: PPEConflictConfig = PPEConflictConfig()
    tracking: TrackingConfig = TrackingConfig()
    temporal: TemporalConfig = TemporalConfig()
    risk: RiskConfig = RiskConfig()

    @classmethod
    def from_mapping(cls, value: Any) -> "SafetyPipelineConfig":
        data = _mapping(value, "root")
        schema_version = _integer(data, "schema_version", cls.schema_version, 1)
        if schema_version != 1:
            raise ConfigError("unsupported safety pipeline schema_version: %s" % schema_version)
        default_mode = _text(data, "default_mode", cls.default_mode)
        if default_mode not in SUPPORTED_MODES:
            raise ConfigError("default_mode must be baseline or ppe_temporal")
        ppe = _mapping(data.get("ppe", {}), "ppe")
        return cls(
            schema_version=schema_version,
            default_mode=default_mode,
            conflict=PPEConflictConfig.from_mapping(ppe.get("conflict")),
            tracking=TrackingConfig.from_mapping(ppe.get("tracking")),
            temporal=TemporalConfig.from_mapping(ppe.get("temporal")),
            risk=RiskConfig.from_mapping(ppe.get("risk")),
        )

    def with_mode(self, mode: str) -> "SafetyPipelineConfig":
        if mode not in SUPPORTED_MODES:
            raise ConfigError("unsupported safety pipeline mode: %s" % mode)
        return replace(self, default_mode=mode)


def load_safety_pipeline_config(
    path: Path,
    logger: Optional[logging.Logger] = None,
) -> SafetyPipelineConfig:
    """Load validated defaults, falling back to baseline on any error.

    A bad optional research configuration must never prevent the detector from
    starting.  The diagnostic is logged and legacy behavior remains active.
    """

    return load_safety_pipeline_config_checked(path, logger)[0]


def load_safety_pipeline_config_checked(
    path: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[SafetyPipelineConfig, bool]:
    """Load config and report whether the tracked file passed validation."""

    active_logger = logger or LOGGER
    try:
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        return SafetyPipelineConfig.from_mapping(document), True
    except Exception as exc:
        active_logger.warning(
            "Safety pipeline config %s is unavailable or invalid; using baseline defaults: %s",
            path,
            exc,
        )
        return SafetyPipelineConfig(), False


def config_fingerprint(config: SafetyPipelineConfig) -> str:
    payload = json.dumps(asdict(config), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

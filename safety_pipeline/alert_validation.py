"""Generic, dependency-free temporal validation for safety alerts.

The gate deliberately separates *association* from *alert evidence*.  A caller
may report low-confidence, missing, or predicted updates to keep lifecycle and
presence accounting accurate, but only a real, confirmed, high-confidence
observation can advance confirmation.  Keys are isolated, so evidence from two
tracks or two spatial fire events can never be combined.

Timestamps belong to the caller's monotonic pipeline clock.  Invalid or
regressing timestamps and invalid confidence values fail safe: they return a
non-emitting decision without mutating event evidence.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Hashable, Mapping, Optional, Protocol, Tuple


_EPSILON = 1e-9


class AlertValidationConfigLike(Protocol):
    """Structural contract implemented by ``AlertValidationConfig``."""

    min_high_confidence_hits: int
    min_duration_seconds: float
    window_seconds: float
    max_gap_seconds: float
    min_presence_ratio: float
    min_ema_confidence: float
    ema_alpha: float
    cooldown_seconds: float
    max_active_events: int


class AlertEvidenceState(str, Enum):
    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class AlertValidationDecision:
    key: Optional[Hashable]
    timestamp: Optional[float]
    state: AlertEvidenceState
    emitted: bool
    confirmed: bool
    actual_hits: int
    duration: float
    presence_ratio: float
    ema: float
    reason: str
    cooldown_remaining: float = 0.0
    gap_reset: bool = False

    @property
    def ema_confidence(self) -> float:
        return self.ema

    def as_dict(self) -> Mapping[str, object]:
        return {
            "key": self.key,
            "timestamp": self.timestamp,
            "state": self.state.value,
            "emitted": self.emitted,
            "confirmed": self.confirmed,
            "actual_hits": self.actual_hits,
            "duration": self.duration,
            "presence_ratio": self.presence_ratio,
            "ema": self.ema,
            "reason": self.reason,
            "cooldown_remaining": self.cooldown_remaining,
            "gap_reset": self.gap_reset,
        }


@dataclass(frozen=True)
class AlertValidationMetrics:
    dropped: int
    expired: int
    suppressed: int
    emitted: int
    active_events: int
    confirmed_events: int

    def as_dict(self) -> Mapping[str, int]:
        return {
            "dropped": self.dropped,
            "expired": self.expired,
            "suppressed": self.suppressed,
            "emitted": self.emitted,
            "active_events": self.active_events,
            "confirmed_events": self.confirmed_events,
        }


@dataclass(frozen=True)
class _Policy:
    min_high_confidence_hits: int
    min_duration_seconds: float
    window_seconds: float
    max_gap_seconds: float
    min_presence_ratio: float
    min_ema_confidence: float
    ema_alpha: float
    cooldown_seconds: float
    max_active_events: int

    @classmethod
    def from_config(cls, config: AlertValidationConfigLike) -> "_Policy":
        min_hits = _positive_integer(
            getattr(config, "min_high_confidence_hits", None),
            "min_high_confidence_hits",
        )
        max_active = _positive_integer(
            getattr(config, "max_active_events", None),
            "max_active_events",
        )
        min_duration = _finite_number(
            getattr(config, "min_duration_seconds", None),
            "min_duration_seconds",
            minimum=0.0,
        )
        window = _finite_number(
            getattr(config, "window_seconds", None),
            "window_seconds",
            minimum=0.0,
            strictly_positive=True,
        )
        if min_duration > window + _EPSILON:
            raise ValueError("min_duration_seconds cannot exceed window_seconds")
        return cls(
            min_high_confidence_hits=min_hits,
            min_duration_seconds=min_duration,
            window_seconds=window,
            max_gap_seconds=_finite_number(
                getattr(config, "max_gap_seconds", None),
                "max_gap_seconds",
                minimum=0.0,
            ),
            min_presence_ratio=_finite_number(
                getattr(config, "min_presence_ratio", None),
                "min_presence_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            min_ema_confidence=_finite_number(
                getattr(config, "min_ema_confidence", None),
                "min_ema_confidence",
                minimum=0.0,
                maximum=1.0,
            ),
            ema_alpha=_finite_number(
                getattr(config, "ema_alpha", None),
                "ema_alpha",
                minimum=0.0,
                maximum=1.0,
            ),
            cooldown_seconds=_finite_number(
                getattr(config, "cooldown_seconds", None),
                "cooldown_seconds",
                minimum=0.0,
            ),
            max_active_events=max_active,
        )


@dataclass
class _EventEvidence:
    sequence: int
    created_at: float
    last_update_at: float
    samples: Deque[Tuple[float, bool]] = field(default_factory=deque)
    actual_observations: Deque[Tuple[float, float]] = field(default_factory=deque)
    last_actual_at: Optional[float] = None
    episode_confirmed: bool = False
    last_emitted_at: Optional[float] = None


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be a positive integer" % field_name)
    return value


def _finite_number(
    value: object,
    field_name: str,
    *,
    minimum: float,
    maximum: Optional[float] = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be numeric" % field_name)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % field_name)
    if strictly_positive and result <= minimum:
        raise ValueError("%s must be > %s" % (field_name, minimum))
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError("%s is outside its supported range" % field_name)
    return result


def _valid_timestamp(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _valid_confidence(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        return None
    return result


class AlertEvidenceGate:
    """Validate alert evidence independently for every hashable event key."""

    def __init__(self, config: AlertValidationConfigLike):
        self.policy = _Policy.from_config(config)
        self._events: Dict[Hashable, _EventEvidence] = {}
        self._next_sequence = 1
        self._dropped = 0
        self._expired = 0
        self._suppressed = 0
        self._emitted = 0

    @property
    def metrics(self) -> AlertValidationMetrics:
        return AlertValidationMetrics(
            dropped=self._dropped,
            expired=self._expired,
            suppressed=self._suppressed,
            emitted=self._emitted,
            active_events=len(self._events),
            confirmed_events=sum(event.episode_confirmed for event in self._events.values()),
        )

    @property
    def active_count(self) -> int:
        return len(self._events)

    def reset(self) -> None:
        self._events.clear()
        self._next_sequence = 1
        self._dropped = 0
        self._expired = 0
        self._suppressed = 0
        self._emitted = 0

    def observe(
        self,
        key: Hashable,
        timestamp: float,
        confidence: float,
        *,
        confirmed: bool = True,
        high_confidence: bool = True,
    ) -> AlertValidationDecision:
        """Submit a real detector observation.

        The defaults match the production call site: upstream tracking has
        confirmed the object and upstream class policy has classified the
        confidence as high.  Passing either flag as false records a negative
        opportunity for an existing candidate but never creates evidence.
        """

        return self.update(
            key,
            timestamp,
            confidence,
            actual=True,
            confirmed=confirmed,
            high_confidence=high_confidence,
            predicted=False,
        )

    def missing(
        self,
        key: Hashable,
        timestamp: float,
        *,
        predicted: bool = False,
    ) -> AlertValidationDecision:
        """Record an inference opportunity without real alert evidence."""

        return self.update(
            key,
            timestamp,
            None,
            actual=False,
            confirmed=False,
            high_confidence=False,
            predicted=predicted,
        )

    def update(
        self,
        key: Hashable,
        timestamp: float,
        confidence: Optional[float],
        *,
        actual: bool,
        confirmed: bool,
        high_confidence: bool,
        predicted: bool = False,
    ) -> AlertValidationDecision:
        numeric_timestamp = _valid_timestamp(timestamp)
        if numeric_timestamp is None:
            self._dropped += 1
            return self._empty_decision(key, None, "invalid_timestamp")
        if not self._is_hashable(key):
            self._dropped += 1
            return self._empty_decision(None, numeric_timestamp, "invalid_key")
        if not all(isinstance(flag, bool) for flag in (actual, confirmed, high_confidence, predicted)):
            self._dropped += 1
            return self._empty_decision(key, numeric_timestamp, "invalid_observation_flags")
        numeric_confidence = None
        if actual:
            numeric_confidence = _valid_confidence(confidence)
            if numeric_confidence is None:
                self._dropped += 1
                return self._decision_for(key, numeric_timestamp, "invalid_confidence")

        event = self._events.get(key)
        if event is not None and numeric_timestamp + _EPSILON < event.last_update_at:
            self._dropped += 1
            return self._decision_for(key, numeric_timestamp, "timestamp_regression")

        positive = bool(actual and confirmed and high_confidence and not predicted)
        if event is None and not positive:
            if predicted:
                return self._empty_decision(key, numeric_timestamp, "predicted_ignored")
            if not actual:
                return self._empty_decision(key, numeric_timestamp, "missing_ignored")
            if not confirmed:
                return self._empty_decision(key, numeric_timestamp, "unconfirmed_ignored")
            return self._empty_decision(key, numeric_timestamp, "low_confidence_ignored")

        if event is None:
            self.cleanup(numeric_timestamp)
            if not self._make_capacity(numeric_timestamp):
                self._dropped += 1
                return self._empty_decision(key, numeric_timestamp, "capacity_rejected")
            event = _EventEvidence(
                sequence=self._next_sequence,
                created_at=numeric_timestamp,
                last_update_at=numeric_timestamp,
            )
            self._next_sequence += 1
            self._events[key] = event

        gap_reset = False
        if (
            event.last_actual_at is not None
            and numeric_timestamp - event.last_actual_at > self.policy.max_gap_seconds + _EPSILON
        ):
            self._reset_episode(event)
            gap_reset = True

        event.last_update_at = numeric_timestamp
        if not positive:
            if gap_reset:
                return self._decision_for(key, numeric_timestamp, "gap_reset", gap_reset=True)
            event.samples.append((numeric_timestamp, False))
            self._prune(event, numeric_timestamp)
            if predicted:
                reason = "predicted_ignored"
            elif not actual:
                reason = "missing_ignored"
            elif not confirmed:
                reason = "unconfirmed_ignored"
            else:
                reason = "low_confidence_ignored"
            return self._decision_for(key, numeric_timestamp, reason)

        event.samples.append((numeric_timestamp, True))
        event.actual_observations.append((numeric_timestamp, float(numeric_confidence)))
        event.last_actual_at = numeric_timestamp
        self._prune(event, numeric_timestamp)

        if gap_reset:
            return self._decision_for(key, numeric_timestamp, "gap_reset", gap_reset=True)

        actual_hits, duration, presence_ratio, ema = self._statistics(event)
        if actual_hits < self.policy.min_high_confidence_hits:
            return self._decision_for(key, numeric_timestamp, "insufficient_hits")
        if duration + _EPSILON < self.policy.min_duration_seconds:
            return self._decision_for(key, numeric_timestamp, "insufficient_duration")
        if presence_ratio + _EPSILON < self.policy.min_presence_ratio:
            return self._decision_for(key, numeric_timestamp, "insufficient_presence")
        if ema + _EPSILON < self.policy.min_ema_confidence:
            return self._decision_for(key, numeric_timestamp, "insufficient_ema")

        event.episode_confirmed = True
        cooldown_remaining = self._cooldown_remaining(event, numeric_timestamp)
        if cooldown_remaining > _EPSILON:
            self._suppressed += 1
            return self._decision_for(
                key,
                numeric_timestamp,
                "cooldown_active",
                cooldown_remaining=cooldown_remaining,
            )

        event.last_emitted_at = numeric_timestamp
        self._emitted += 1
        return self._decision_for(key, numeric_timestamp, "emitted", emitted=True)

    def cleanup(self, timestamp: float) -> Tuple[Hashable, ...]:
        """Expire inactive records while preserving active cooldown memory."""

        numeric_timestamp = _valid_timestamp(timestamp)
        if numeric_timestamp is None:
            self._dropped += 1
            return ()
        expired = []
        for key, event in tuple(self._events.items()):
            if numeric_timestamp + _EPSILON < event.last_update_at:
                continue
            inactive = numeric_timestamp - event.last_update_at > self.policy.max_gap_seconds + _EPSILON
            cooldown_finished = self._cooldown_remaining(event, numeric_timestamp) <= _EPSILON
            if inactive and (event.last_emitted_at is None or cooldown_finished):
                if not event.episode_confirmed and event.actual_observations:
                    self._suppressed += 1
                expired.append(key)
                del self._events[key]
        self._expired += len(expired)
        return tuple(expired)

    @staticmethod
    def _is_hashable(key: object) -> bool:
        try:
            hash(key)
            return True
        except (TypeError, ValueError):
            return False

    def _make_capacity(self, timestamp: float) -> bool:
        if len(self._events) < self.policy.max_active_events:
            return True
        candidates = [
            (key, event)
            for key, event in self._events.items()
            if not self._protected(event, timestamp)
        ]
        if not candidates:
            return False
        key, event = min(candidates, key=lambda item: (item[1].last_update_at, item[1].sequence))
        if event.actual_observations:
            self._suppressed += 1
        del self._events[key]
        self._dropped += 1
        return True

    def _protected(self, event: _EventEvidence, timestamp: float) -> bool:
        return event.episode_confirmed or self._cooldown_remaining(event, timestamp) > _EPSILON

    def _cooldown_remaining(self, event: _EventEvidence, timestamp: float) -> float:
        if event.last_emitted_at is None:
            return 0.0
        return max(0.0, event.last_emitted_at + self.policy.cooldown_seconds - timestamp)

    def _reset_episode(self, event: _EventEvidence) -> None:
        if not event.episode_confirmed and event.actual_observations:
            self._suppressed += 1
        event.samples.clear()
        event.actual_observations.clear()
        event.last_actual_at = None
        event.episode_confirmed = False

    def _prune(self, event: _EventEvidence, timestamp: float) -> None:
        cutoff = timestamp - self.policy.window_seconds
        while event.samples and event.samples[0][0] + _EPSILON < cutoff:
            event.samples.popleft()
        while event.actual_observations and event.actual_observations[0][0] + _EPSILON < cutoff:
            event.actual_observations.popleft()

    def _statistics(self, event: _EventEvidence) -> Tuple[int, float, float, float]:
        observations = tuple(event.actual_observations)
        actual_hits = len(observations)
        duration = observations[-1][0] - observations[0][0] if actual_hits > 1 else 0.0
        presence_ratio = actual_hits / len(event.samples) if event.samples else 0.0
        ema = 0.0
        if observations:
            ema = observations[0][1]
            alpha = self.policy.ema_alpha
            for _, confidence in observations[1:]:
                ema = alpha * confidence + (1.0 - alpha) * ema
        return actual_hits, max(0.0, duration), presence_ratio, max(0.0, min(1.0, ema))

    def _state(self, event: _EventEvidence, timestamp: float) -> AlertEvidenceState:
        if self._cooldown_remaining(event, timestamp) > _EPSILON:
            return AlertEvidenceState.COOLDOWN
        if event.episode_confirmed:
            return AlertEvidenceState.CONFIRMED
        if event.actual_observations:
            return AlertEvidenceState.CANDIDATE
        return AlertEvidenceState.IDLE

    def _decision_for(
        self,
        key: Hashable,
        timestamp: float,
        reason: str,
        *,
        emitted: bool = False,
        cooldown_remaining: Optional[float] = None,
        gap_reset: bool = False,
    ) -> AlertValidationDecision:
        event = self._events.get(key)
        if event is None:
            return self._empty_decision(key, timestamp, reason, gap_reset=gap_reset)
        actual_hits, duration, presence_ratio, ema = self._statistics(event)
        remaining = (
            self._cooldown_remaining(event, timestamp)
            if cooldown_remaining is None
            else cooldown_remaining
        )
        return AlertValidationDecision(
            key=key,
            timestamp=timestamp,
            state=self._state(event, timestamp),
            emitted=emitted,
            confirmed=event.episode_confirmed,
            actual_hits=actual_hits,
            duration=duration,
            presence_ratio=presence_ratio,
            ema=ema,
            reason=reason,
            cooldown_remaining=max(0.0, remaining),
            gap_reset=gap_reset,
        )

    @staticmethod
    def _empty_decision(
        key: Optional[Hashable],
        timestamp: Optional[float],
        reason: str,
        *,
        gap_reset: bool = False,
    ) -> AlertValidationDecision:
        return AlertValidationDecision(
            key=key,
            timestamp=timestamp,
            state=AlertEvidenceState.IDLE,
            emitted=False,
            confirmed=False,
            actual_hits=0,
            duration=0.0,
            presence_ratio=0.0,
            ema=0.0,
            reason=reason,
            gap_reset=gap_reset,
        )


__all__ = [
    "AlertEvidenceGate",
    "AlertEvidenceState",
    "AlertValidationConfigLike",
    "AlertValidationDecision",
    "AlertValidationMetrics",
]

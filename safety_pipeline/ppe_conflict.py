"""Mutually-exclusive PPE class conflict resolution.

The resolver keeps both raw detections as evidence while returning one logical
head observation for rendering and temporal reasoning.  It intentionally runs
after canonical class mapping and before tracking.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .config import PPEConflictConfig
from .matching import maximum_cardinality_weight_matching
from .types import Box, HeadObservation, PPEState, ResolutionFrame, ResolvedDetection


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter_x1, inter_y1 = max(a[0], b[0]), max(a[1], b[1])
    inter_x2, inter_y2 = min(a[2], b[2]), min(a[3], b[3])
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _geometry_compatible(a: Sequence[float], b: Sequence[float], config: PPEConflictConfig) -> bool:
    area_a, area_b = _area(a), _area(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return False
    area_ratio = min(area_a, area_b) / max(area_a, area_b)
    if area_ratio < config.min_area_ratio:
        return False
    center_a = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    center_b = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    center_distance = math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
    reference_diagonal = max(math.sqrt(area_a), math.sqrt(area_b), 1e-9)
    return center_distance / reference_diagonal <= config.max_center_distance_ratio


def _source_ids(*detections: Any) -> Tuple[str, ...]:
    result: List[str] = []
    for detection in detections:
        for model_id in tuple(getattr(detection, "source_model_ids", ()) or ()):
            model_id = str(model_id)
            if model_id not in result:
                result.append(model_id)
    return tuple(result)


def _weighted_box(first: Any, second: Any) -> Box:
    first_weight = max(0.0, float(first.confidence))
    second_weight = max(0.0, float(second.confidence))
    total = first_weight + second_weight
    if total <= 1e-12:
        return tuple(float(value) for value in first.box)  # type: ignore[return-value]
    return tuple(
        (float(a) * first_weight + float(b) * second_weight) / total
        for a, b in zip(first.box, second.box)
    )  # type: ignore[return-value]


class PPEConflictResolver:
    """Resolve configured protected/unprotected detections one-to-one."""

    def __init__(self, config: PPEConflictConfig):
        self.config = config

    def _single(self, detection: Any) -> ResolvedDetection:
        protected = detection.class_id == self.config.protected_class_id
        state = PPEState.HELMET if protected else PPEState.NO_HELMET
        score = float(detection.confidence)
        observation = HeadObservation(
            box=tuple(float(value) for value in detection.box),
            state=state,
            helmet_score=score if protected else None,
            nohelmet_score=None if protected else score,
            confidence=score,
            raw_detections=(detection,),
            source_model_ids=_source_ids(detection),
            resolution="single",
        )
        return ResolvedDetection(
            box=observation.box,
            confidence=observation.confidence,
            class_id=detection.class_id,
            source_model_ids=observation.source_model_ids,
            head_observation=observation,
        )

    def _pair(self, protected: Any, unprotected: Any) -> ResolvedDetection:
        helmet_score = float(protected.confidence)
        nohelmet_score = float(unprotected.confidence)
        delta = helmet_score - nohelmet_score
        # The configured margin is strict: equality remains UNCERTAIN.  The
        # epsilon prevents binary float representation from changing policy.
        if delta > self.config.score_margin + 1e-12:
            state = PPEState.HELMET
            class_id = self.config.protected_class_id
            confidence = helmet_score
            resolution = "conflict_helmet"
        elif -delta > self.config.score_margin + 1e-12:
            state = PPEState.NO_HELMET
            class_id = self.config.unprotected_class_id
            confidence = nohelmet_score
            resolution = "conflict_no_helmet"
        else:
            state = PPEState.UNCERTAIN
            class_id = self.config.uncertain_class_id
            confidence = max(helmet_score, nohelmet_score)
            resolution = "conflict_uncertain"
        observation = HeadObservation(
            box=_weighted_box(protected, unprotected),
            state=state,
            helmet_score=helmet_score,
            nohelmet_score=nohelmet_score,
            confidence=confidence,
            raw_detections=(protected, unprotected),
            source_model_ids=_source_ids(protected, unprotected),
            resolution=resolution,
        )
        return ResolvedDetection(
            box=observation.box,
            confidence=confidence,
            class_id=class_id,
            source_model_ids=observation.source_model_ids,
            head_observation=observation,
        )

    def resolve(self, detections: Iterable[Any], enabled: bool = True) -> ResolutionFrame:
        items = tuple(detections)
        ppe_indexes = tuple(
            index
            for index, detection in enumerate(items)
            if getattr(detection, "class_id", None)
            in (self.config.protected_class_id, self.config.unprotected_class_id)
        )
        if not enabled:
            return ResolutionFrame(
                detections=items,
                input_count=len(items),
                output_count=len(items),
                ppe_raw_count=len(ppe_indexes),
                head_count=len(ppe_indexes),
                conflicts_resolved=0,
                uncertain_count=0,
            )

        protected_indexes = [
            index for index in ppe_indexes if items[index].class_id == self.config.protected_class_id
        ]
        unprotected_indexes = [
            index for index in ppe_indexes if items[index].class_id == self.config.unprotected_class_id
        ]
        candidate_weights: Dict[Tuple[int, int], float] = {}
        for protected_index in protected_indexes:
            protected = items[protected_index]
            for unprotected_index in unprotected_indexes:
                unprotected = items[unprotected_index]
                overlap = box_iou(protected.box, unprotected.box)
                if overlap + 1e-12 < self.config.min_iou:
                    continue
                if not _geometry_compatible(protected.box, unprotected.box, self.config):
                    continue
                combined_score = float(protected.confidence) + float(unprotected.confidence)
                # Cardinality is optimized first by the matcher.  The scale
                # makes total IoU dominate confidence as the secondary goal.
                candidate_weights[(protected_index, unprotected_index)] = overlap * 1_000_000.0 + combined_score

        matched = maximum_cardinality_weight_matching(
            protected_indexes,
            unprotected_indexes,
            candidate_weights,
        )

        pair_by_anchor: Dict[int, ResolvedDetection] = {}
        skipped = set()
        for protected_index, unprotected_index in matched.items():
            anchor = min(protected_index, unprotected_index)
            pair_by_anchor[anchor] = self._pair(items[protected_index], items[unprotected_index])
            skipped.update((protected_index, unprotected_index))

        output: List[Any] = []
        head_count = 0
        uncertain_count = 0
        for index, detection in enumerate(items):
            if index in pair_by_anchor:
                resolved = pair_by_anchor[index]
                output.append(resolved)
                head_count += 1
                if resolved.head_observation.state == PPEState.UNCERTAIN:
                    uncertain_count += 1
                continue
            if index in skipped:
                continue
            if index in ppe_indexes:
                output.append(self._single(detection))
                head_count += 1
            else:
                output.append(detection)

        return ResolutionFrame(
            detections=tuple(output),
            input_count=len(items),
            output_count=len(output),
            ppe_raw_count=len(ppe_indexes),
            head_count=head_count,
            conflicts_resolved=len(matched),
            uncertain_count=uncertain_count,
        )

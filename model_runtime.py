"""Lazy YOLOv5 model loading and multi-model inference for SentinelVision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from model_catalog import CatalogSnapshot, ModelSpec
from project_paths import YOLO_REPO_DIR


@dataclass(frozen=True)
class Detection:
    box: Tuple[float, float, float, float]
    confidence: float
    class_id: str
    source_model_ids: Tuple[str, ...]


@dataclass(frozen=True)
class SelectionResult:
    requested_ids: Tuple[str, ...]
    active_ids: Tuple[str, ...]
    failures: Mapping[str, str]


StatusCallback = Callable[[str, str, str], None]


def _model_names(model) -> Tuple[str, ...]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        try:
            names = [names[index] for index in sorted(names)]
        except (KeyError, TypeError):
            names = list(names.values())
    if not isinstance(names, (list, tuple)):
        return ()
    return tuple(str(name) for name in names)


def _extract_rows(result) -> np.ndarray:
    try:
        return result.xyxy[0].detach().cpu().numpy()
    except Exception:
        return result.pandas().xyxy[0].to_numpy()


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter_x1, inter_y1 = max(a[0], b[0]), max(a[1], b[1])
    inter_x2, inter_y2 = min(a[2], b[2]), min(a[3], b[3])
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def cross_model_nms(detections: Iterable[Detection], iou_threshold: float) -> List[Detection]:
    """Suppress duplicate boxes after all selected models have run.

    Suppression is grouped by canonical class.  If two models provide the same
    box, the retained detection records both source model IDs.
    """
    grouped: Dict[str, List[Detection]] = {}
    for detection in detections:
        grouped.setdefault(detection.class_id, []).append(detection)

    kept_all: List[Detection] = []
    for class_detections in grouped.values():
        remaining = sorted(class_detections, key=lambda item: item.confidence, reverse=True)
        while remaining:
            best = remaining.pop(0)
            providers = list(best.source_model_ids)
            survivors: List[Detection] = []
            for candidate in remaining:
                if _iou(best.box, candidate.box) > iou_threshold:
                    for model_id in candidate.source_model_ids:
                        if model_id not in providers:
                            providers.append(model_id)
                else:
                    survivors.append(candidate)
            kept_all.append(Detection(
                box=best.box,
                confidence=best.confidence,
                class_id=best.class_id,
                source_model_ids=tuple(providers),
            ))
            remaining = survivors
    return kept_all


class ModelRuntime:
    """Owns model objects inside the video worker thread."""

    def __init__(self, snapshot: CatalogSnapshot, device: Optional[str] = None):
        self.snapshot = snapshot
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded: Dict[str, object] = {}
        self.loaded_specs: Dict[str, ModelSpec] = {}
        self.loaded_fingerprints: Dict[str, tuple] = {}
        self.load_failures: Dict[str, str] = {}
        self.inference_failures: Dict[str, str] = {}

    @staticmethod
    def _runtime_fingerprint(spec: ModelSpec) -> tuple:
        """Identify both deployed bytes and metadata that affects inference."""
        try:
            stat = spec.weight_path.stat()
            file_state = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            file_state = (None, None)
        return (
            str(spec.weight_path),
            spec.selectable,
            spec.registered_sha256,
            spec.registered_bytes,
            tuple(spec.expected_classes),
            tuple(sorted(spec.class_map.items())),
            file_state,
        )

    def update_catalog(self, snapshot: CatalogSnapshot) -> None:
        self.snapshot = snapshot
        current = {spec.model_id: spec for spec in snapshot.models}
        stale = []
        for model_id, old_spec in self.loaded_specs.items():
            new_spec = current.get(model_id)
            if (
                new_spec is None
                or not new_spec.selectable
                or self.loaded_fingerprints.get(model_id) != self._runtime_fingerprint(new_spec)
            ):
                stale.append(model_id)
        self._unload(stale)

    def _unload(self, model_ids: Iterable[str]) -> None:
        changed = False
        for model_id in tuple(model_ids):
            if model_id in self.loaded:
                del self.loaded[model_id]
                self.loaded_specs.pop(model_id, None)
                self.loaded_fingerprints.pop(model_id, None)
                changed = True
        if changed and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_all(self) -> None:
        self._unload(tuple(self.loaded))

    def _load_one(self, spec: ModelSpec):
        model = torch.hub.load(
            repo_or_dir=str(YOLO_REPO_DIR),
            model="custom",
            path=str(spec.weight_path),
            source="local",
        ).to(self.device).eval()
        observed = _model_names(model)
        if spec.expected_classes and observed != spec.expected_classes:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise ValueError(
                "类别顺序不一致：登记=%s，权重=%s"
                % (list(spec.expected_classes), list(observed))
            )
        return model

    def apply_selection(
        self,
        requested_ids: Iterable[str],
        status_callback: Optional[StatusCallback] = None,
    ) -> SelectionResult:
        requested = tuple(dict.fromkeys(str(model_id) for model_id in requested_ids))
        specs = {spec.model_id: spec for spec in self.snapshot.models}

        # Release models the user no longer wants before allocating new CUDA
        # memory.  This is important on a 12 GB laptop GPU.
        removed = tuple(model_id for model_id in tuple(self.loaded) if model_id not in requested)
        self._unload(removed)
        if status_callback:
            for model_id in removed:
                status_callback(model_id, "inactive", "可用")

        failures: Dict[str, str] = {}
        for model_id in requested:
            spec = specs.get(model_id)
            if spec is None:
                failures[model_id] = "模型不在本次 RESULTS 扫描结果中"
                continue
            if not spec.selectable:
                failures[model_id] = spec.issue or "模型不可用"
                continue
            if model_id in self.loaded:
                if status_callback:
                    status_callback(model_id, "active", "已启用")
                continue
            if status_callback:
                status_callback(model_id, "loading", "正在加载")
            try:
                model = self._load_one(spec)
                self.loaded[model_id] = model
                self.loaded_specs[model_id] = spec
                self.loaded_fingerprints[model_id] = self._runtime_fingerprint(spec)
                self.load_failures.pop(model_id, None)
                if status_callback:
                    status_callback(model_id, "active", "已启用")
            except Exception as exc:
                message = str(exc).strip() or exc.__class__.__name__
                self.load_failures[model_id] = message
                failures[model_id] = message
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if status_callback:
                    status_callback(model_id, "failed", message)

        active = tuple(model_id for model_id in requested if model_id in self.loaded)
        return SelectionResult(requested_ids=requested, active_ids=active, failures=failures)

    def infer(self, image, conf: float, iou: float, image_size: int) -> List[Detection]:
        detections: List[Detection] = []
        failed_model_ids: List[str] = []
        self.inference_failures = {}
        for model_id, model in tuple(self.loaded.items()):
            try:
                spec = self.loaded_specs[model_id]
                model.conf = conf
                model.iou = iou
                with torch.no_grad():
                    result = model(image, size=image_size)
                rows = _extract_rows(result)
                names = _model_names(model)
                for row in rows:
                    x1, y1, x2, y2, confidence, raw_index = row[:6]
                    class_index = int(raw_index)
                    if class_index < 0 or class_index >= len(names):
                        continue
                    raw_name = names[class_index]
                    canonical = spec.class_map.get(raw_name, "%s::%s" % (model_id, raw_name))
                    detections.append(Detection(
                        box=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(confidence),
                        class_id=canonical,
                        source_model_ids=(model_id,),
                    ))
            except Exception as exc:
                message = str(exc).strip() or exc.__class__.__name__
                self.inference_failures[model_id] = message
                self.load_failures[model_id] = message
                failed_model_ids.append(model_id)
        if failed_model_ids:
            self._unload(failed_model_ids)
        return cross_model_nms(detections, iou_threshold=float(iou))


__all__ = ["Detection", "ModelRuntime", "SelectionResult", "cross_model_nms"]

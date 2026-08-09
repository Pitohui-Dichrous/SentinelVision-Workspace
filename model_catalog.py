"""Dynamic deployment-model discovery for SentinelVision.

Only files below ``RESULTS`` are considered deployable.  Training outputs are
deliberately outside this trust boundary and are never scanned here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from project_paths import RESULTS_DIR


ACCEPTED_STATUSES = {"approved", "deployed", "legacy_in_use", "active"}
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".runtime",
    "_archive",
    "archive",
    "archives",
    "_backup",
    "backup",
    "backups",
    "disabled",
    "_staging",
}


@dataclass(frozen=True)
class ClassProfile:
    class_id: str
    display_zh: str
    display_en: str
    severity: Optional[str] = None
    alert_enabled: bool = False
    color_rgb: Tuple[int, int, int] = (96, 165, 250)
    enabled_by_default: bool = True

    def display_name(self, language: str) -> str:
        return self.display_zh if language == "zh" else self.display_en


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    weight_path: Path
    relative_weight_path: str
    source: str
    status: str
    selectable: bool
    expected_classes: Tuple[str, ...] = ()
    class_map: Mapping[str, str] = field(default_factory=dict)
    default_selected: bool = False
    allow_class_overlap: bool = False
    registered_sha256: Optional[str] = None
    registered_bytes: Optional[int] = None
    issue: Optional[str] = None

    @property
    def canonical_classes(self) -> Tuple[str, ...]:
        return tuple(self.class_map.get(raw_name, raw_name) for raw_name in self.expected_classes)


@dataclass(frozen=True)
class CatalogSnapshot:
    models: Tuple[ModelSpec, ...]
    class_profiles: Mapping[str, ClassProfile]
    registry_path: Path
    warnings: Tuple[str, ...] = ()

    def model_by_id(self, model_id: str) -> Optional[ModelSpec]:
        return next((model for model in self.models if model.model_id == model_id), None)

    @property
    def selectable_models(self) -> Tuple[ModelSpec, ...]:
        return tuple(model for model in self.models if model.selectable)

    def selected_class_ids(self, model_ids: Iterable[str]) -> Tuple[str, ...]:
        wanted = set(model_ids)
        result: List[str] = []
        for model in self.models:
            if model.model_id not in wanted:
                continue
            for class_id in model.canonical_classes:
                if class_id not in result:
                    result.append(class_id)
        return tuple(result)

    def selection_conflicts(self, model_ids: Iterable[str]) -> Mapping[str, Tuple[str, ...]]:
        """Return canonical classes supplied by overlapping, non-ensemble models."""
        wanted = set(model_ids)
        providers: Dict[str, List[ModelSpec]] = {}
        for model in self.models:
            if model.model_id not in wanted:
                continue
            for class_id in model.canonical_classes:
                class_providers = providers.setdefault(class_id, [])
                if all(provider.model_id != model.model_id for provider in class_providers):
                    class_providers.append(model)

        conflicts: Dict[str, Tuple[str, ...]] = {}
        for class_id, models in providers.items():
            if len(models) < 2:
                continue
            if all(model.allow_class_overlap for model in models):
                continue
            conflicts[class_id] = tuple(model.model_id for model in models)
        return conflicts


DEFAULT_CLASS_PROFILES: Mapping[str, ClassProfile] = {
    "fire": ClassProfile(
        class_id="fire",
        display_zh="火焰",
        display_en="Fire",
        severity="critical",
        alert_enabled=True,
        color_rgb=(244, 63, 94),
    ),
    # The historic checkpoint uses the label ``person`` for a person without
    # a helmet.  Keep that meaning for backward compatibility.
    "person": ClassProfile(
        class_id="person",
        display_zh="未戴安全帽",
        display_en="NO HELMET",
        severity="warning",
        alert_enabled=True,
        color_rgb=(56, 189, 248),
    ),
    "hat": ClassProfile(
        class_id="hat",
        display_zh="已戴安全帽",
        display_en="Helmet",
        severity=None,
        alert_enabled=False,
        color_rgb=(234, 179, 8),
    ),
    # Logical output produced only by the optional PPE temporal pipeline.  It
    # is not a model class and must never raise an alert by itself.
    "ppe_uncertain": ClassProfile(
        class_id="ppe_uncertain",
        display_zh="PPE 待确认",
        display_en="PPE Uncertain",
        severity=None,
        alert_enabled=False,
        color_rgb=(148, 163, 184),
        enabled_by_default=True,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    return cleaned or "model"


def _parse_color(value, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            channels = tuple(max(0, min(255, int(channel))) for channel in value)
            return channels  # type: ignore[return-value]
        except (TypeError, ValueError):
            pass
    return fallback


def _generated_color(name: str) -> Tuple[int, int, int]:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    # Keep generated colors bright enough for both themes and video frames.
    return tuple(80 + (channel % 156) for channel in digest[:3])  # type: ignore[return-value]


def read_checkpoint_classes(path: Path) -> Tuple[str, ...]:
    """Read YOLOv5 class names for a newly discovered, unregistered model."""
    import torch

    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        checkpoint = torch.load(str(path), map_location="cpu")

    model = checkpoint
    if isinstance(checkpoint, dict):
        model = checkpoint.get("ema") or checkpoint.get("model") or checkpoint
    names = getattr(model, "names", None)
    if names is None and isinstance(checkpoint, dict):
        names = checkpoint.get("names")
    if isinstance(names, dict):
        try:
            names = [names[index] for index in sorted(names)]
        except (KeyError, TypeError):
            names = list(names.values())
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError("权重中没有可读取的类别名称")
    result = tuple(str(name).strip() for name in names)
    if any(not name for name in result) or len(set(result)) != len(result):
        raise ValueError("权重类别名称为空或重复")
    return result


class ModelCatalog:
    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = Path(results_dir).resolve()
        self.registry_path = self.results_dir / "model_registry.yaml"

    def _inside_results(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.results_dir)
            return True
        except (OSError, ValueError):
            return False

    def _read_registry(self) -> Tuple[dict, List[str]]:
        if not self.registry_path.is_file():
            return {"schema_version": 2, "models": {}}, ["缺少 RESULTS/model_registry.yaml；仅显示自动发现模型。"]
        try:
            with self.registry_path.open("r", encoding="utf-8-sig") as handle:
                document = yaml.safe_load(handle) or {}
            if not isinstance(document, dict) or not isinstance(document.get("models", {}), dict):
                raise ValueError("models 必须是映射")
            return document, []
        except Exception as exc:
            return {"schema_version": 2, "models": {}}, ["模型登记文件读取失败：%s" % exc]

    def _profiles_from_registry(self, registry: Mapping) -> Dict[str, ClassProfile]:
        profiles = dict(DEFAULT_CLASS_PROFILES)
        raw_profiles = registry.get("class_profiles", {})
        if not isinstance(raw_profiles, dict):
            return profiles
        for class_id, raw in raw_profiles.items():
            if not isinstance(raw, dict):
                continue
            class_id = str(class_id)
            previous = profiles.get(class_id)
            display = raw.get("display", {}) if isinstance(raw.get("display", {}), dict) else {}
            severity = raw.get("severity", previous.severity if previous else None)
            alert_enabled = bool(raw.get("alert_enabled", severity is not None))
            profiles[class_id] = ClassProfile(
                class_id=class_id,
                display_zh=str(display.get("zh", previous.display_zh if previous else class_id)),
                display_en=str(display.get("en", previous.display_en if previous else class_id)),
                severity=str(severity) if severity not in (None, "", "none") else None,
                alert_enabled=alert_enabled,
                color_rgb=_parse_color(raw.get("color"), previous.color_rgb if previous else _generated_color(class_id)),
                enabled_by_default=bool(raw.get("enabled_by_default", True)),
            )
        return profiles

    def _registered_specs(self, registry: Mapping, verify_hashes: bool) -> Tuple[List[ModelSpec], set]:
        specs: List[ModelSpec] = []
        registered_paths = set()
        raw_models = registry.get("models", {})
        for raw_model_id, record in raw_models.items():
            model_id = _slug(str(raw_model_id))
            if not isinstance(record, dict):
                specs.append(ModelSpec(
                    model_id=model_id,
                    display_name=str(raw_model_id),
                    weight_path=self.results_dir / "__invalid__",
                    relative_weight_path="",
                    source="registry",
                    status="blocked",
                    selectable=False,
                    issue="登记项格式错误",
                ))
                continue

            relative = str(record.get("deployment_path", "")).replace("\\", "/").strip()
            path = (self.results_dir / relative).resolve() if relative else self.results_dir / "__invalid__"
            if relative:
                registered_paths.add(path)
            display_name = str(record.get("display_name") or raw_model_id).strip()
            expected_classes = tuple(str(item) for item in (record.get("expected_classes") or ()))
            raw_map = record.get("class_map", {})
            class_map = {
                raw_name: str(raw_map.get(raw_name, raw_name)) if isinstance(raw_map, dict) else raw_name
                for raw_name in expected_classes
            }
            artifact = record.get("artifact", {}) if isinstance(record.get("artifact", {}), dict) else {}
            expected_hash = str(artifact.get("sha256", "")).lower() or None
            try:
                expected_bytes = int(artifact["bytes"]) if "bytes" in artifact else None
            except (TypeError, ValueError):
                expected_bytes = None

            issue = None
            if not relative:
                issue = "缺少 deployment_path"
            elif not self._inside_results(path):
                issue = "部署路径逃出 RESULTS，已拒绝"
            elif not path.is_file():
                issue = "权重文件不存在"
            elif path.name.lower() == "last.pt":
                issue = "last.pt 不能作为部署权重"
            elif str(record.get("status", "legacy_in_use")).lower() not in ACCEPTED_STATUSES:
                issue = "模型尚未批准部署"
            elif expected_bytes is not None and path.stat().st_size != expected_bytes:
                issue = "文件大小与登记不一致"
            elif verify_hashes and expected_hash and sha256_file(path) != expected_hash:
                issue = "SHA-256 与登记不一致"

            specs.append(ModelSpec(
                model_id=model_id,
                display_name=display_name,
                weight_path=path,
                relative_weight_path=relative,
                source="registry",
                status="ready" if issue is None else "blocked",
                selectable=issue is None,
                expected_classes=expected_classes,
                class_map=class_map,
                default_selected=bool(record.get("default_selected", model_id == "combined")),
                allow_class_overlap=bool(record.get("allow_class_overlap", False)),
                registered_sha256=expected_hash,
                registered_bytes=expected_bytes,
                issue=issue,
            ))
        return specs, registered_paths

    def _discovery_candidates(self) -> Tuple[Path, ...]:
        candidates = set(self.results_dir.glob("*.pt"))
        candidates.update(self.results_dir.rglob("best.pt"))
        valid: List[Path] = []
        for path in candidates:
            try:
                relative = path.relative_to(self.results_dir)
            except ValueError:
                continue
            if any(part.lower() in IGNORED_DIRECTORY_NAMES or part.startswith(".") for part in relative.parts[:-1]):
                continue
            if path.is_file() and path.name.lower() != "last.pt" and self._inside_results(path):
                valid.append(path.resolve())
        return tuple(sorted(valid, key=lambda item: str(item).lower()))

    def scan(self, verify_hashes: bool = True, inspect_new_models: bool = True) -> CatalogSnapshot:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        registry, warnings = self._read_registry()
        profiles = self._profiles_from_registry(registry)
        specs, registered_paths = self._registered_specs(registry, verify_hashes=verify_hashes)
        used_ids = {spec.model_id for spec in specs}

        for path in self._discovery_candidates():
            if path in registered_paths:
                continue
            relative = path.relative_to(self.results_dir).as_posix()
            if path.parent.name.lower() == "weights":
                suggested_name = path.parent.parent.name
            elif path.name.lower() == "best.pt":
                suggested_name = path.parent.name
            else:
                suggested_name = path.stem
            model_id = _slug(suggested_name)
            if model_id in used_ids:
                model_id = "%s_%s" % (model_id, hashlib.sha256(relative.encode("utf-8")).hexdigest()[:8])
            used_ids.add(model_id)

            issue = None
            classes: Tuple[str, ...] = ()
            if inspect_new_models:
                try:
                    classes = read_checkpoint_classes(path)
                except Exception as exc:
                    issue = "无法读取模型类别：%s" % exc
            else:
                issue = "尚未执行模型技术检查"

            # Unregistered class names are namespaced.  A reviewer can map
            # them to shared semantics later in model_registry.yaml.
            class_map = {raw_name: "%s::%s" % (model_id, raw_name) for raw_name in classes}
            for raw_name, canonical in class_map.items():
                if canonical not in profiles:
                    profiles[canonical] = ClassProfile(
                        class_id=canonical,
                        display_zh=raw_name,
                        display_en=raw_name,
                        severity=None,
                        alert_enabled=False,
                        color_rgb=_generated_color(canonical),
                    )
            specs.append(ModelSpec(
                model_id=model_id,
                display_name=suggested_name.replace("_", " ").strip() or model_id,
                weight_path=path,
                relative_weight_path=relative,
                source="discovered",
                status="new" if issue is None else "blocked",
                selectable=issue is None,
                expected_classes=classes,
                class_map=class_map,
                default_selected=False,
                allow_class_overlap=False,
                issue=issue or "人工放入的新模型；未登记类别默认不告警",
            ))

        specs.sort(key=lambda model: (model.source != "registry", model.display_name.lower(), model.model_id))
        return CatalogSnapshot(
            models=tuple(specs),
            class_profiles=profiles,
            registry_path=self.registry_path,
            warnings=tuple(warnings),
        )


__all__ = [
    "CatalogSnapshot",
    "ClassProfile",
    "ModelCatalog",
    "ModelSpec",
    "read_checkpoint_classes",
    "sha256_file",
]

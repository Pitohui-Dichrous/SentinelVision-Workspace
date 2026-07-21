"""Human-gated candidate-model deployment into the RESULTS trust boundary."""

from __future__ import annotations

import copy
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import yaml

from model_catalog import read_checkpoint_classes, sha256_file
from project_paths import PROJECT_ROOT, RESULTS_DIR


REQUIRED_REVIEW_ITEMS = (
    "metrics_reviewed",
    "class_order_reviewed",
    "failure_samples_reviewed",
    "real_media_reviewed",
)


@dataclass(frozen=True)
class CandidateInfo:
    path: Path
    classes: Tuple[str, ...]
    bytes: int
    sha256: str


@dataclass(frozen=True)
class DeploymentResult:
    model_id: str
    deployment_directory: Path
    weight_path: Path
    archive_directory: Optional[Path]


def safe_model_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    if not cleaned:
        raise ValueError("模型 ID 不能为空")
    return cleaned


def inspect_candidate(path) -> CandidateInfo:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.suffix.lower() != ".pt":
        raise FileNotFoundError("候选权重不存在或不是 .pt：%s" % candidate)
    classes = read_checkpoint_classes(candidate)
    return CandidateInfo(
        path=candidate,
        classes=classes,
        bytes=candidate.stat().st_size,
        sha256=sha256_file(candidate),
    )


class DeploymentManager:
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = Path(project_root).resolve()
        self.results_dir = (self.project_root / "RESULTS").resolve()
        self.registry_path = self.results_dir / "model_registry.yaml"
        self.archive_root = self.project_root / "MODEL_ARCHIVE"

    def _candidate_provenance(self, candidate_path: Path) -> Dict[str, str]:
        """Record an audit hint without persisting a removable-drive letter."""
        resolved = Path(candidate_path).resolve()
        try:
            relative = resolved.relative_to(self.project_root).as_posix()
        except ValueError:
            return {"candidate_path": resolved.name, "candidate_location": "external"}
        return {"candidate_path": relative, "candidate_location": "workspace"}

    def _read_registry(self):
        if self.registry_path.is_file():
            with self.registry_path.open("r", encoding="utf-8-sig") as handle:
                registry = yaml.safe_load(handle) or {}
        else:
            registry = {}
        if not isinstance(registry, dict):
            raise ValueError("model_registry.yaml 顶层必须是映射")
        registry["schema_version"] = 2
        if not isinstance(registry.get("models"), dict):
            registry["models"] = {}
        if not isinstance(registry.get("class_profiles"), dict):
            registry["class_profiles"] = {}
        if not isinstance(registry.get("policy"), dict):
            registry["policy"] = {
                "deployment_root_only": True,
                "auto_discover": True,
                "auto_select_new_models": False,
            }
        return registry

    def _target_directory(self, registry, model_id):
        record = registry["models"].get(model_id)
        if isinstance(record, dict) and record.get("deployment_path"):
            existing_weight = (self.results_dir / str(record["deployment_path"])).resolve()
            try:
                existing_weight.relative_to(self.results_dir)
            except ValueError:
                raise ValueError("现有登记路径逃出 RESULTS")
            if existing_weight.parent.name.lower() == "weights":
                return existing_weight.parent.parent
        return self.results_dir / model_id

    @staticmethod
    def _normalise_class_settings(
        info: CandidateInfo,
        settings: Mapping,
        existing_profiles: Mapping,
    ) -> Tuple[Dict, Dict]:
        class_map = {}
        profiles = {}
        canonical_sources = {}
        for raw_name in info.classes:
            raw = settings.get(raw_name, {}) if isinstance(settings, Mapping) else {}
            if not isinstance(raw, Mapping):
                raw = {}
            canonical = str(raw.get("canonical_id") or raw_name).strip().lower()
            if not re.fullmatch(r"[a-z0-9][a-z0-9_.:-]*", canonical):
                raise ValueError("类别 %s 的统一类别 ID 只能使用小写英文、数字及 _ . : -" % raw_name)
            if canonical in canonical_sources:
                raise ValueError(
                    "同一模型的类别 %s 与 %s 不能都映射为 %s"
                    % (canonical_sources[canonical], raw_name, canonical)
                )
            canonical_sources[canonical] = raw_name
            severity = raw.get("severity")
            if severity in ("", "none", "None"):
                severity = None
            if severity not in (None, "info", "warning", "critical"):
                raise ValueError("类别 %s 的告警等级无效" % raw_name)
            alert_enabled = bool(raw.get("alert_enabled", severity is not None))
            if alert_enabled and severity is None:
                raise ValueError("类别 %s 已启用告警，但没有选择告警等级" % raw_name)
            color = str(raw.get("color") or "#60A5FA")
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                raise ValueError("类别 %s 的颜色必须是 #RRGGBB" % raw_name)
            class_map[raw_name] = canonical
            # Profiles are global for a canonical class.  A new model may
            # reuse that semantic ID, but it must never silently alter alert
            # behavior already used by deployed models.
            if canonical not in existing_profiles:
                profiles[canonical] = {
                    "display": {
                        "zh": str(raw.get("display_zh") or raw_name),
                        "en": str(raw.get("display_en") or raw_name),
                    },
                    "severity": severity,
                    "alert_enabled": alert_enabled,
                    "color": color,
                    "enabled_by_default": bool(raw.get("enabled_by_default", True)),
                }
        return class_map, profiles

    def deploy(
        self,
        candidate: CandidateInfo,
        model_id: str,
        display_name: str,
        class_settings: Mapping,
        review: Mapping,
        approved: bool,
        replace_existing: bool = False,
        reviewer: str = "user",
    ) -> DeploymentResult:
        model_id = safe_model_id(model_id)
        if not approved:
            raise PermissionError("未收到最终部署确认")
        missing_reviews = [item for item in REQUIRED_REVIEW_ITEMS if not bool(review.get(item))]
        if missing_reviews:
            raise PermissionError("人工审核清单未完成：%s" % ", ".join(missing_reviews))

        self.results_dir.mkdir(parents=True, exist_ok=True)
        registry = self._read_registry()
        existing_record = copy.deepcopy(registry["models"].get(model_id))
        target_directory = self._target_directory(registry, model_id)
        if target_directory.exists() and not replace_existing:
            raise FileExistsError("模型 %s 已存在；替换必须再次明确确认" % model_id)

        class_map, profiles = self._normalise_class_settings(
            candidate,
            class_settings,
            registry.get("class_profiles", {}),
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        staging_directory = self.results_dir / "_STAGING" / (model_id + "_" + timestamp)
        if staging_directory.exists():
            raise FileExistsError(staging_directory)
        staging_weight = staging_directory / "weights" / "best.pt"
        staging_weight.parent.mkdir(parents=True, exist_ok=False)
        shutil.copy2(str(candidate.path), str(staging_weight))
        if staging_weight.stat().st_size != candidate.bytes or sha256_file(staging_weight) != candidate.sha256:
            shutil.rmtree(str(staging_directory), ignore_errors=True)
            raise IOError("暂存权重复制校验失败")

        approved_at = datetime.now(timezone.utc).isoformat()
        review_record = {
            "approved": True,
            "approved_at": approved_at,
            "approved_by": reviewer,
            **{item: bool(review.get(item)) for item in REQUIRED_REVIEW_ITEMS},
            "notes": str(review.get("notes") or ""),
        }
        deployment_manifest = {
            "schema_version": 2,
            "id": model_id,
            "display_name": display_name.strip() or model_id,
            "status": "approved",
            "weight": "weights/best.pt",
            "expected_classes": list(candidate.classes),
            "class_map": class_map,
            "artifact": {"bytes": candidate.bytes, "sha256": candidate.sha256},
            "review": review_record,
        }
        with (staging_directory / "deployment.yaml").open("w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(deployment_manifest, handle, allow_unicode=True, sort_keys=False)

        archive_directory = None
        moved_new_model = False
        try:
            if target_directory.exists():
                archive_directory = self.archive_root / model_id / timestamp
                archive_directory.parent.mkdir(parents=True, exist_ok=True)
                if archive_directory.exists():
                    raise FileExistsError(archive_directory)
                shutil.move(str(target_directory), str(archive_directory))
                if existing_record is not None:
                    with (archive_directory / "registry_record.yaml").open("w", encoding="utf-8", newline="\n") as handle:
                        yaml.safe_dump(existing_record, handle, allow_unicode=True, sort_keys=False)

            target_directory.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging_directory), str(target_directory))
            moved_new_model = True

            relative_weight = (target_directory / "weights" / "best.pt").relative_to(self.results_dir).as_posix()
            registry["class_profiles"].update(profiles)
            registry["models"][model_id] = {
                "display_name": display_name.strip() or model_id,
                "deployment_path": relative_weight,
                "format": "yolov5_pt",
                "expected_classes": list(candidate.classes),
                "class_map": class_map,
                "default_selected": False,
                "allow_class_overlap": False,
                "status": "approved",
                "artifact": {"bytes": candidate.bytes, "sha256": candidate.sha256},
                "technical_check": {
                    "checked_at": approved_at,
                    "checkpoint_metadata": "passed",
                    "observed_classes": list(candidate.classes),
                },
                "provenance": self._candidate_provenance(candidate.path),
                "review": review_record,
            }
            temporary_registry = self.registry_path.with_suffix(".yaml.tmp")
            with temporary_registry.open("w", encoding="utf-8", newline="\n") as handle:
                yaml.safe_dump(registry, handle, allow_unicode=True, sort_keys=False)
            temporary_registry.replace(self.registry_path)
        except Exception:
            # Best-effort rollback keeps the last approved deployment usable.
            if moved_new_model and target_directory.exists():
                failed_directory = self.results_dir / "_STAGING" / (model_id + "_failed_" + timestamp)
                failed_directory.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target_directory), str(failed_directory))
            if archive_directory is not None and archive_directory.exists() and not target_directory.exists():
                shutil.move(str(archive_directory), str(target_directory))
            raise
        finally:
            if staging_directory.exists():
                shutil.rmtree(str(staging_directory), ignore_errors=True)

        return DeploymentResult(
            model_id=model_id,
            deployment_directory=target_directory,
            weight_path=target_directory / "weights" / "best.pt",
            archive_directory=archive_directory,
        )


__all__ = [
    "CandidateInfo",
    "DeploymentManager",
    "DeploymentResult",
    "REQUIRED_REVIEW_ITEMS",
    "inspect_candidate",
    "safe_model_id",
]

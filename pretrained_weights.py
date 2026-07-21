"""Curated, offline YOLOv5 v7.0 detection checkpoints.

Only official object-detection checkpoints that work with the root ``train.py``
workflow are exposed here.  Segmentation, classification, YOLOv3 and
experimental YAML files intentionally stay out of the beginner training UI.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from project_paths import PRETRAINED_WEIGHTS_DIR


@dataclass(frozen=True)
class WeightPreset:
    key: str
    filename: str
    display_name: str
    family: str
    image_size: int
    size_bytes: int
    sha256: str
    speed: str
    accuracy: str
    memory: str
    recommendation: str

    @property
    def path(self) -> Path:
        return PRETRAINED_WEIGHTS_DIR / self.filename

    @property
    def is_p6(self) -> bool:
        return self.family == "P6"


PRESETS: Tuple[WeightPreset, ...] = (
    WeightPreset("n", "yolov5n.pt", "YOLOv5n · Nano", "P5", 640, 4062133,
                 "4F180CF23BA0717ADA0BADD6C685026D73D48F184D00FC159C2641284B2AC0A3",
                 "最快", "基础", "最低", "快速试验、小型数据集或低延迟场景"),
    WeightPreset("s", "yolov5s.pt", "YOLOv5s · Small", "P5", 640, 14808437,
                 "8B3B748C1E592DDD8868022E8732FDE20025197328490623CC16C6F24D0782EE",
                 "快", "均衡", "较低", "默认推荐，适合第一次训练和参数摸索"),
    WeightPreset("m", "yolov5m.pt", "YOLOv5m · Medium", "P5", 640, 42806829,
                 "61D933360BA5A7733A36764996C800287D973889D875227F5BEEDD2473A97A56",
                 "中等", "较高", "中等", "速度与精度的进阶平衡"),
    WeightPreset("l", "yolov5l.pt", "YOLOv5l · Large", "P5", 640, 93622629,
                 "2F603B7354C25454D1270663A14D8DDC1EEA98E5EEBC1D84CE0C6E3150FA155F",
                 "较慢", "高", "较高", "精度优先；建议先用 s/m 验证数据集"),
    WeightPreset("x", "yolov5x.pt", "YOLOv5x · XLarge", "P5", 640, 174114333,
                 "9F27A794FA0308E2606F90565164571D6F0A0BA18A3CE2E5E5D323B71C157859",
                 "最慢", "最高", "高", "高成本精度实验；训练时间最长"),
    WeightPreset("n6", "yolov5n6.pt", "YOLOv5n6 · Nano P6", "P6", 1280, 7190229,
                 "496C05A2B991DA15ACC6C4408C63DA287F2513A46C6756618776D4DD71170781",
                 "较快", "大图基础", "中等", "高级：超高清画面与小目标，建议 1280px"),
    WeightPreset("s6", "yolov5s6.pt", "YOLOv5s6 · Small P6", "P6", 1280, 25978837,
                 "95BDA9019ADA63F37338308D0C81A66047A6FBABA061ABAFF271BA9334CF2A6F",
                 "中等", "大图均衡", "较高", "高级：1280px 大图训练的推荐起点"),
    WeightPreset("m6", "yolov5m6.pt", "YOLOv5m6 · Medium P6", "P6", 1280, 72308357,
                 "7AFE7FA0F29A8351467200A2A5A33C7996D1155ED6636C7EDD8CA70B14951C64",
                 "较慢", "大图较高", "高", "高级：优先在 RTX 4090 上训练"),
    WeightPreset("l6", "yolov5l6.pt", "YOLOv5l6 · Large P6", "P6", 1280, 154521589,
                 "B357B77F646B0190912985937C44E2923441369D5FDA69BA86DA2AA919212618",
                 "慢", "大图高", "很高", "高级：需要较大显存，自动 batch 仍可能很小"),
    WeightPreset("x6", "yolov5x6.pt", "YOLOv5x6 · XLarge P6", "P6", 1280, 282720357,
                 "257C991B216D625427DC4D9C7C76D6C6D93CF3732797538DA34BDC02F97BFEF0",
                 "最慢", "大图最高", "极高", "高级实验；RTX 4080 Laptop 不推荐"),
)

PRESETS_BY_FILENAME = {preset.filename.lower(): preset for preset in PRESETS}


def preset_for_path(path) -> Optional[WeightPreset]:
    candidate = Path(path)
    preset = PRESETS_BY_FILENAME.get(candidate.name.lower())
    if preset is None:
        return None
    try:
        if candidate.resolve().parent != PRETRAINED_WEIGHTS_DIR.resolve():
            return None
    except OSError:
        return None
    return preset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_preset(preset: WeightPreset, full_hash: bool = True) -> Tuple[bool, str]:
    path = preset.path
    if not path.is_file():
        return False, "缺少官方预训练权重：%s" % path
    if path.stat().st_size != preset.size_bytes:
        return False, "%s 文件大小不正确，可能下载不完整" % preset.filename
    if full_hash and sha256_file(path) != preset.sha256:
        return False, "%s 的 SHA-256 不匹配，已拒绝使用" % preset.filename
    return True, "官方 YOLOv5 v7.0 权重校验通过"


def verify_catalogued_path(path, full_hash: bool = True) -> Tuple[bool, str]:
    """Validate known bundled weights; leave trusted custom local weights alone."""
    preset = preset_for_path(path)
    if preset is None:
        return True, "自定义本地权重"
    return verify_preset(preset, full_hash=full_hash)


def validate_training_weight(path) -> Tuple[bool, str]:
    """Reject segmentation/classification or malformed custom checkpoints early."""
    candidate = Path(path)
    preset = preset_for_path(candidate)
    if preset is not None:
        return verify_preset(preset, full_hash=True)
    if not candidate.is_file():
        return False, "权重文件不存在：%s" % candidate
    try:
        import torch

        checkpoint = torch.load(str(candidate), map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint 不是 YOLOv5 字典格式")
        # train.py transfers architecture and state from ckpt['model']; inspect
        # that exact object rather than accepting a possibly unrelated EMA.
        model = checkpoint.get("model")
        layers = getattr(model, "model", None)
        if model is None or layers is None or len(layers) == 0:
            raise ValueError("checkpoint 中没有可训练的 YOLOv5 检测模型")
        if not isinstance(getattr(model, "yaml", None), dict):
            raise ValueError("checkpoint 中缺少 YOLOv5 model.yaml 结构")
        head_name = type(layers[-1]).__name__
        if head_name != "Detect":
            raise ValueError("任务类型为 %s，不是目标检测 Detect" % head_name)
        names = getattr(model, "names", None)
        class_count = len(names) if isinstance(names, (list, tuple, dict)) else "未知"
        return True, "自定义 YOLOv5 目标检测权重检查通过（%s 类）" % class_count
    except Exception as exc:
        return False, "自定义权重与当前目标检测训练流程不兼容：%s" % exc


__all__ = [
    "PRESETS", "PRESETS_BY_FILENAME", "WeightPreset", "preset_for_path",
    "sha256_file", "validate_training_weight", "verify_catalogued_path", "verify_preset",
]

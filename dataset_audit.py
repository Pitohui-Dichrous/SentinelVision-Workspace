"""Audit YOLO datasets before training without changing any data files."""

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import yaml

from project_paths import PROJECT_ROOT


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def resolve_path(value, base=PROJECT_ROOT):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def images_from_source(source, dataset_root=PROJECT_ROOT):
    """Expand one YOLO directory or image-list source into absolute paths."""
    path = resolve_path(source, dataset_root)
    if path.is_dir():
        return {
            item.resolve()
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        }, []

    if path.is_file() and path.suffix.lower() == ".txt":
        images = set()
        missing = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                image = Path(raw).expanduser()
                if not image.is_absolute():
                    # YOLO list entries beginning with ./ are relative to the
                    # list file.  Other relative entries are rooted at the
                    # dataset's YAML ``path``.
                    if raw.startswith("./") or raw.startswith(".\\"):
                        image = path.parent / raw[2:]
                    else:
                        image = Path(dataset_root) / image
                image = image.resolve()
                images.add(image)
                if not image.is_file():
                    missing.append(image)
        return images, missing

    return set(), [path]


def expand_split(value, dataset_root=PROJECT_ROOT):
    sources = value if isinstance(value, list) else [value]
    images = set()
    missing = []
    for source in sources:
        source_images, source_missing = images_from_source(source, dataset_root)
        images.update(source_images)
        missing.extend(source_missing)
    return images, missing


def label_path_for(image_path):
    parts = list(image_path.parts)
    image_directory_index = None
    for index, part in enumerate(parts):
        if part.lower() == "images":
            image_directory_index = index
    if image_directory_index is not None:
        parts[image_directory_index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def inspect_labels(images, class_count, verify_images=True):
    classes = Counter()
    missing_labels = []
    empty_labels = 0
    malformed = []
    invalid_classes = []
    invalid_coordinates = []
    boundary_crossings = []
    corrupt_images = []

    image_module = None
    if verify_images:
        from PIL import Image
        image_module = Image

    for image in sorted(images):
        if image_module is not None:
            try:
                with image_module.open(image) as opened:
                    opened.verify()
            except Exception as exc:
                corrupt_images.append((image, str(exc)))
        label_path = label_path_for(image)
        if not label_path.is_file():
            missing_labels.append(label_path)
            continue

        nonempty_lines = 0
        with label_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                text = line.strip()
                if not text:
                    continue
                nonempty_lines += 1
                fields = text.split()
                if len(fields) != 5:
                    malformed.append((label_path, line_number, "应有 5 列"))
                    continue
                try:
                    raw_class, x, y, width, height = [float(value) for value in fields]
                except ValueError:
                    malformed.append((label_path, line_number, "存在非数字内容"))
                    continue

                class_id = int(raw_class)
                if raw_class != class_id or class_id < 0 or class_id >= class_count:
                    invalid_classes.append((label_path, line_number, raw_class))
                else:
                    classes[class_id] += 1

                valid_components = 0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1
                if not valid_components:
                    invalid_coordinates.append((label_path, line_number, (x, y, width, height)))
                elif (
                    x - width / 2 < 0 or x + width / 2 > 1
                    or y - height / 2 < 0 or y + height / 2 > 1
                ):
                    boundary_crossings.append((label_path, line_number, (x, y, width, height)))

        if nonempty_lines == 0:
            empty_labels += 1

    return {
        "classes": classes,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "malformed": malformed,
        "invalid_classes": invalid_classes,
        "invalid_coordinates": invalid_coordinates,
        "boundary_crossings": boundary_crossings,
        "corrupt_images": corrupt_images,
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_overlaps(split_images):
    """Find byte-identical images stored under different split paths."""
    size_buckets = {}
    for split_name, images in split_images.items():
        for path in images:
            try:
                size = path.stat().st_size
            except OSError:
                continue
            size_buckets.setdefault(size, []).append((split_name, path))

    overlaps = []
    for entries in size_buckets.values():
        if len({split for split, _path in entries}) < 2:
            continue
        digest_buckets = {}
        for split_name, path in entries:
            try:
                digest = _file_sha256(path)
            except OSError:
                continue
            digest_buckets.setdefault(digest, []).append((split_name, path))
        for digest_entries in digest_buckets.values():
            for left_index, (left_split, left_path) in enumerate(digest_entries):
                for right_split, right_path in digest_entries[left_index + 1:]:
                    if left_split == right_split or left_path == right_path:
                        continue
                    overlaps.append((left_split, right_split, left_path, right_path))
    return overlaps


def print_examples(title, values, limit=5):
    if not values:
        return
    print("  %s（前 %d 项）:" % (title, min(limit, len(values))))
    for value in values[:limit]:
        print("    - %s" % (value,))


def audit_dataset(yaml_path, verify_images=True, verify_split_content=True):
    config_path = resolve_path(yaml_path)
    print("\n数据集: %s" % config_path)
    print("-" * 72)
    if not config_path.is_file():
        print("[错误] 配置文件不存在")
        return 1

    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle) or {}

    dataset_root = PROJECT_ROOT
    configured_root = config.get("path")
    if configured_root:
        raw_root = Path(str(configured_root)).expanduser()
        if raw_root.is_absolute():
            dataset_root = raw_root.resolve()
        else:
            # Match YOLOv5: ``path`` is normally relative to the repository.
            project_candidate = (PROJECT_ROOT / raw_root).resolve()
            yaml_candidate = (config_path.parent / raw_root).resolve()
            dataset_root = project_candidate if project_candidate.exists() else yaml_candidate
        print("数据集根目录: %s" % dataset_root)
        if not dataset_root.exists():
            print("[错误] path 指定的数据集根目录不存在")
            return 1

    errors = 0
    class_count = config.get("nc")
    names = config.get("names")
    if not isinstance(class_count, int) or class_count <= 0:
        print("[错误] nc 必须是正整数")
        return 1
    if not isinstance(names, (list, dict)) or len(names) != class_count:
        print("[错误] names 的数量必须与 nc=%d 一致" % class_count)
        errors += 1

    split_images = {}
    for split_name in ("train", "val", "test"):
        if split_name not in config:
            if split_name in ("train", "val"):
                print("[错误] 缺少 %s 数据源" % split_name)
                errors += 1
            continue
        images, missing_images = expand_split(config[split_name], dataset_root)
        split_images[split_name] = images
        print("%s: %d 张图片" % (split_name, len(images)))
        if missing_images:
            print("[错误] %s 中有 %d 个图片或数据源不存在" % (split_name, len(missing_images)))
            print_examples("缺失路径", missing_images)
            errors += 1
        if not images:
            print("[错误] %s 没有可用图片" % split_name)
            errors += 1

        report = inspect_labels(images, class_count, verify_images=verify_images)
        readable_classes = {str(key): value for key, value in sorted(report["classes"].items())}
        print("  合法标注框: %s" % readable_classes)
        print("  空标注图片: %d，缺少标注文件: %d" % (report["empty_labels"], len(report["missing_labels"])))

        fatal_groups = (
            ("损坏或无法解码的图片", report["corrupt_images"]),
            ("格式错误", report["malformed"]),
            ("超出 nc 范围的类别", report["invalid_classes"]),
            ("非法归一化坐标", report["invalid_coordinates"]),
        )
        for title, values in fatal_groups:
            if values:
                print("[错误] %s: %d" % (title, len(values)))
                print_examples(title, values)
                errors += 1

        if report["missing_labels"]:
            print("[提示] 缺少 txt 会被 YOLOv5 当作无目标背景图，请确认这是有意的。")
        if report["boundary_crossings"]:
            print("[提示] %d 个框因标注取整略微越过图像边界；YOLOv5 可裁剪处理，建议后续清洗。" % len(report["boundary_crossings"]))
            print_examples("越界框", report["boundary_crossings"])

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if left not in split_images or right not in split_images:
            continue
        overlap = split_images[left].intersection(split_images[right])
        if overlap:
            print("[错误] %s 与 %s 重复 %d 张图片，会导致评估数据泄漏。" % (left, right, len(overlap)))
            print_examples("重复图片", sorted(overlap))
            errors += 1

    if verify_split_content:
        duplicate_content = content_overlaps(split_images)
        if duplicate_content:
            print("[错误] 不同 split 中有 %d 对内容完全相同的图片，会导致评估数据泄漏。" % len(duplicate_content))
            print_examples("同内容图片", duplicate_content)
            errors += 1

    if errors:
        print("结论: [不可训练] 发现 %d 类必须修复的问题。" % errors)
        return 1
    print("结论: [通过] 未发现会使训练或验证失真的结构问题。")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="训练前检查 YOLO 数据集")
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["data/helmet_fire.yaml", "data/fire.yaml"],
        help="要检查的一个或多个 YAML 配置",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = [audit_dataset(path) for path in args.datasets]
    return 1 if any(results) else 0


if __name__ == "__main__":
    sys.exit(main())

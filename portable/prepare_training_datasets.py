"""Materialize consistent training views without altering preserved sources.

The historic fire source mixes class id 2 in its train split and class id 0 in
its validation split.  This script creates two explicit views:

* combined_legacy_v1: person=0, hat=1, fire=2
* fire_only_legacy_v1: fire=0

Images are hard-linked when possible and copied as a fallback.  Labels are
rewritten only in the generated DATASETS directory.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "DATASETS"
HELMET_ROOT = PROJECT_ROOT / "data" / "helmetdata"
FIRE_ROOT = PROJECT_ROOT / "data" / "firedata" / "VOCdevkit"
_IMAGE_DIGEST_CACHE = {}


def read_image_list(path):
    images = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                if raw.startswith("./") or raw.startswith(".\\"):
                    candidate = path.parent / raw[2:]
                else:
                    candidate = PROJECT_ROOT / candidate
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            images.append(candidate)
    return images


def source_label(image):
    parts = list(image.parts)
    image_index = None
    for index, part in enumerate(parts):
        if part.lower() == "images":
            image_index = index
    if image_index is None:
        raise ValueError("image path has no images directory: %s" % image)
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(str(source), str(destination))


def converted_labels(source, mode):
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError("%s:%d does not have five columns" % (source, line_number))
            try:
                raw_class = float(fields[0])
            except ValueError:
                raise ValueError("%s:%d contains a non-numeric class" % (source, line_number))
            class_id = int(raw_class)
            if raw_class != class_id:
                raise ValueError("%s:%d class id must be an integer" % (source, line_number))
            if mode == "fire_as_2":
                class_id = 2
            elif mode == "fire_as_0":
                class_id = 0
            elif mode == "identity":
                if class_id not in (0, 1, 2):
                    raise ValueError("%s:%d invalid class %s" % (source, line_number, class_id))
            else:
                raise ValueError("unknown conversion mode: %s" % mode)
            rows.append("%d %s" % (class_id, " ".join(fields[1:])))
    return "\n".join(rows) + ("\n" if rows else "")


def unique_name(prefix, image):
    try:
        portable_source = image.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix().casefold()
    except ValueError:
        raise ValueError("legacy source must remain inside the workspace: %s" % image)
    digest = hashlib.sha1(portable_source.encode("utf-8")).hexdigest()[:10]
    return "%s_%s_%s%s" % (prefix, image.stem, digest, image.suffix.lower())


def image_digest(path):
    resolved = path.resolve()
    cached = _IMAGE_DIGEST_CACHE.get(resolved)
    if cached:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _IMAGE_DIGEST_CACHE[resolved] = value
    return value


def deduplicate_sources(sources, priority):
    """Keep one byte-identical image, preferring evaluation over training."""
    seen = {}
    filtered = {}
    removed = {split: 0 for split in sources}
    for split in priority:
        groups = []
        for prefix, images, conversion in sources.get(split, ()):
            kept = []
            for image in images:
                digest = image_digest(image)
                if digest in seen:
                    removed[split] += 1
                    continue
                seen[digest] = (split, image)
                kept.append(image)
            groups.append((prefix, kept, conversion))
        filtered[split] = tuple(groups)
    return filtered, removed


def add_images(dataset_root, split, prefix, images, conversion):
    output_images = dataset_root / "images" / split
    output_labels = dataset_root / "labels" / split
    count = 0
    for image in images:
        name = unique_name(prefix, image)
        destination_image = output_images / name
        destination_label = output_labels / (Path(name).stem + ".txt")
        link_or_copy(image, destination_image)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        label_text = converted_labels(source_label(image), conversion)
        with destination_label.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(label_text)
        count += 1
    return count


def write_dataset_yaml(root, published_name, names, include_test):
    lines = [
        "path: %s" % (Path("DATASETS") / published_name).as_posix(),
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")
    lines.extend([
        "nc: %d" % len(names),
        "names: [%s]" % ", ".join(names),
    ])
    with (root / "dataset.yaml").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def prepare_combined(root, published_name):
    sources = {
        "train": (
            ("helmet", read_image_list(HELMET_ROOT / "train_yolo.txt"), "identity"),
            ("fire", read_image_list(FIRE_ROOT / "train.txt"), "fire_as_2"),
        ),
        "val": (
            ("helmet", read_image_list(HELMET_ROOT / "val_yolo.txt"), "identity"),
            ("fire", read_image_list(FIRE_ROOT / "val.txt"), "fire_as_2"),
        ),
        "test": (
            ("helmet", read_image_list(HELMET_ROOT / "test_yolo.txt"), "identity"),
        ),
    }
    sources, removed = deduplicate_sources(sources, ("test", "val", "train"))
    counts = {}
    for split, groups in sources.items():
        counts[split] = sum(add_images(root, split, prefix, images, conversion) for prefix, images, conversion in groups)
    counts["duplicates_removed"] = removed
    write_dataset_yaml(root, published_name, ["person", "hat", "fire"], include_test=True)
    return root, counts


def prepare_fire_only(root, published_name):
    sources = {
        "train": (("fire", read_image_list(FIRE_ROOT / "train.txt"), "fire_as_0"),),
        "val": (("fire", read_image_list(FIRE_ROOT / "val.txt"), "fire_as_0"),),
    }
    sources, removed = deduplicate_sources(sources, ("val", "train"))
    counts = {
        split: sum(add_images(root, split, prefix, images, conversion) for prefix, images, conversion in groups)
        for split, groups in sources.items()
    }
    counts["duplicates_removed"] = removed
    write_dataset_yaml(root, published_name, ["fire"], include_test=False)
    return root, counts


def publish_generated_view(name, prepare):
    """Build from scratch, audit, then replace only the named generated view."""
    final_root = DATASETS_ROOT / name
    staging_root = DATASETS_ROOT / ("_BUILDING_" + name)
    backup_root = DATASETS_ROOT / ("_PREVIOUS_" + name)
    datasets_root = DATASETS_ROOT.resolve()
    if any(path.resolve().parent != datasets_root for path in (final_root, staging_root, backup_root)):
        raise RuntimeError("generated dataset path escaped DATASETS")

    # Recover the last complete view if a previous run stopped between the
    # two directory renames.
    if backup_root.exists() and not final_root.exists():
        backup_root.rename(final_root)
    if staging_root.exists():
        shutil.rmtree(str(staging_root))
    if backup_root.exists():
        shutil.rmtree(str(backup_root))

    root, counts = prepare(staging_root, name)
    if final_root.exists():
        final_root.rename(backup_root)
    try:
        staging_root.rename(final_root)
        audit = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "dataset_audit.py"), str(final_root / "dataset.yaml")],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if audit.returncode != 0:
            raise RuntimeError("generated dataset did not pass its final audit")
    except Exception:
        if final_root.exists():
            shutil.rmtree(str(final_root))
        if backup_root.exists():
            backup_root.rename(final_root)
        raise
    if backup_root.exists():
        shutil.rmtree(str(backup_root))
    return final_root, counts


def main():
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, prepare in (("combined_legacy_v1", prepare_combined), ("fire_only_legacy_v1", prepare_fire_only)):
        root, counts = publish_generated_view(name, prepare)
        results[name] = {"path": root.relative_to(PROJECT_ROOT).as_posix(), "counts": counts}
        print("%s: %s" % (name, counts))
    manifest = {
        "schema_version": 1,
        "source_policy": "generated views; preserved source images and labels were not modified",
        "datasets": results,
    }
    with (DATASETS_ROOT / "legacy_views_manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

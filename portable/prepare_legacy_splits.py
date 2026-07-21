"""Build YOLO image lists from the preserved VOC ImageSets split files."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "helmetdata"
SOURCE_ROOT = DATASET_ROOT / "ImageSets" / "Main"


def main():
    for split in ("train", "val", "test"):
        source = SOURCE_ROOT / (split + ".txt")
        destination = DATASET_ROOT / (split + "_yolo.txt")
        identifiers = [line.strip() for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        rows = []
        missing = []
        for identifier in identifiers:
            image = DATASET_ROOT / "images" / (identifier + ".jpg")
            if not image.is_file():
                missing.append(image)
            rows.append("./images/%s.jpg" % identifier)
        if missing:
            raise FileNotFoundError("%s: %d images are missing" % (split, len(missing)))
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(rows) + "\n")
        print("%s: %d -> %s" % (split, len(rows), destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

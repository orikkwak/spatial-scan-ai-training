from pathlib import Path
import json

import cv2

PROJECT_ROOT = Path.cwd()
RAW_ROOT = PROJECT_ROOT / "datasets" / "raw" / "structured3d"
PROCESSED_ROOT = PROJECT_ROOT / "datasets" / "processed" / "structured3d"

OUTPUT_INDEX = PROCESSED_ROOT / "index.jsonl"


RGB_CANDIDATES = [
    "rgb_rawlight.png",
    "rgb.png",
    "raw.png",
]

MASK_CANDIDATES = [
    "semantic.png",
    "semantic_raw.png",
]


def find_pairs():
    pairs = []
    total_candidates = 0
    skipped = 0

    for rgb_path in RAW_ROOT.rglob("*.png"):
        if rgb_path.name not in RGB_CANDIDATES:
            continue

        total_candidates += 1
        folder = rgb_path.parent

        mask_path = None
        for mask_name in MASK_CANDIDATES:
            candidate = folder / mask_name
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            skipped += 1
            continue

        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        if image is None or mask is None:
            skipped += 1
            continue

        pairs.append(
            {
                "image_path": str(rgb_path),
                "mask_path": str(mask_path),
                "source": "structured3d",
            }
        )

    return pairs, total_candidates, skipped


def main():
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    pairs, total_candidates, skipped = find_pairs()

    if not pairs:
        print("[WARNING] No image-mask pairs found.")
        print("Structured3D 압축 구조를 먼저 확인해야 함.")
        return

    with OUTPUT_INDEX.open("w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] Total candidates: {total_candidates}")
    print(f"[OK] Valid pairs: {len(pairs)}")
    print(f"[OK] Skipped samples: {skipped}")
    print(f"[OK] Saved to {OUTPUT_INDEX}")


if __name__ == "__main__":
    main()

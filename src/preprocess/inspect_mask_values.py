from pathlib import Path
import json
import cv2
import numpy as np
from collections import Counter


INDEX_PATH = Path(
    r"C:\spatial-scan\ai-training\datasets\processed\structured3d\index.jsonl"
)


def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}")

    counter = Counter()

    with INDEX_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()[:20]

    for line in lines:
        item = json.loads(line)
        mask_path = item["mask_path"]

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            print(f"[SKIP] Cannot read: {mask_path}")
            continue

        if len(mask.shape) == 2:
            values = np.unique(mask)
            for value in values:
                counter[int(value)] += 1

        else:
            pixels = mask.reshape(-1, mask.shape[-1])
            unique_colors = np.unique(pixels, axis=0)

            for color in unique_colors:
                counter[tuple(int(x) for x in color)] += 1

    print("[MASK VALUES / COLORS]")
    for key, count in counter.most_common(50):
        print(key, count)


if __name__ == "__main__":
    main()

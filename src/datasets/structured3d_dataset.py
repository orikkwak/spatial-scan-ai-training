from pathlib import Path
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Structured3DSegmentationDataset(Dataset):
    def __init__(
        self,
        index_path: str,
        image_size=(512, 256),
        num_classes: int = 40,
    ):
        self.index_path = Path(index_path)
        self.image_size = image_size  # (width, height)
        self.num_classes = num_classes
        self.color_to_class = {}

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        self.items = []

        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                self.items.append(json.loads(line))

        if not self.items:
            raise ValueError(f"Dataset index is empty: {self.index_path}")

        self._build_global_color_mapping()

    def __len__(self):
        return len(self.items)

    def _load_image(self, path: str) -> torch.Tensor:
        image_path = Path(path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LINEAR,
        )

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image).float()

    def _encode_color_mask(self, mask: np.ndarray) -> np.ndarray:
        b = mask[:, :, 0].astype(np.int64)
        g = mask[:, :, 1].astype(np.int64)
        r = mask[:, :, 2].astype(np.int64)
        return r * 256 * 256 + g * 256 + b

    def _build_global_color_mapping(self):
        values = set()

        for item in self.items:
            mask = cv2.imread(item["mask_path"], cv2.IMREAD_UNCHANGED)

            if mask is None or len(mask.shape) == 2:
                continue

            values.update(np.unique(self._encode_color_mask(mask)).tolist())

        for value in sorted(values):
            if value == 0:
                self.color_to_class[value] = 0
            else:
                class_id = 1 + ((len(self.color_to_class) - 1) % (self.num_classes - 1))
                self.color_to_class[value] = class_id

    def _load_mask(self, path: str) -> torch.Tensor:
        mask_path = Path(path)

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")

        mask = cv2.resize(
            mask,
            self.image_size,
            interpolation=cv2.INTER_NEAREST,
        )

        if len(mask.shape) == 2:
            mask = mask.astype(np.int64)
        else:
            encoded = self._encode_color_mask(mask)
            class_mask = np.zeros_like(encoded, dtype=np.int64)

            for value, class_id in self.color_to_class.items():
                class_mask[encoded == value] = class_id

            mask = class_mask

        mask = np.clip(mask, 0, self.num_classes - 1)

        return torch.from_numpy(mask).long()

    def __getitem__(self, idx: int):
        item = self.items[idx]

        image_path = item["image_path"]
        mask_path = item["mask_path"]

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        return {
            "image": image,
            "mask": mask,
            "image_path": image_path,
            "mask_path": mask_path,
        }

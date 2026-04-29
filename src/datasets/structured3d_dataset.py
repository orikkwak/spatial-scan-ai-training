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

        # cv2.resize size order = (width, height)
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LINEAR,
        )

        image = image.astype(np.float32) / 255.0

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image).float()

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

        # case 1: semantic.png가 이미 class id 형태인 경우
        if len(mask.shape) == 2:
            mask = mask.astype(np.int64)

        # case 2: semantic.png가 BGR/RGB color mask 형태인 경우
        else:
            # OpenCV는 BGR로 읽음
            b = mask[:, :, 0].astype(np.int64)
            g = mask[:, :, 1].astype(np.int64)
            r = mask[:, :, 2].astype(np.int64)

            encoded = r * 256 * 256 + g * 256 + b
            unique_values = np.unique(encoded)

            # 임시 매핑:
            # 색상값을 class id로 바꿈.
            # 나중에 정확한 Structured3D semantic color table 기준으로 교체해야 함.
            value_to_class = {
                value: idx % self.num_classes for idx, value in enumerate(unique_values)
            }

            class_mask = np.zeros_like(encoded, dtype=np.int64)

            for value, class_id in value_to_class.items():
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

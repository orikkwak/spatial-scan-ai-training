# Structured3D Panorama 00 Report

## Dataset

- Panorama archive: present, stable, extracted with partial errors.
- Annotation archive: present, stable, extracted successfully.
- Corrupted archive subitems: 36.
- Index total candidates: 3345.
- Valid readable pairs: 3339.
- Skipped unreadable samples: 6.

## Mask Inspection

- Mask inspection completed after index rebuild.
- Observed RGB semantic mask colors include 24 unique sampled values.

## Training

- Device: cuda.
- GPU: NVIDIA GeForce RTX 5080.
- Dataset size: 3339.
- Epochs: 10.
- Final train loss: 1.7709.
- Final validation loss: 1.8071.
- Final validation pixel accuracy: 0.3276.
- Best checkpoint: checkpoints/structured3d/best.pt.
- Last checkpoint: checkpoints/structured3d/last.pt.

## Notes

- Initial training run failed on unreadable mask `scene_00013/2D_rendering/99/panorama/simple/semantic.png`.
- The index builder now verifies image and mask readability before writing `index.jsonl`.

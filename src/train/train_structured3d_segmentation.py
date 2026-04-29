from pathlib import Path
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.datasets.structured3d_dataset import Structured3DSegmentationDataset
from src.models.simple_unet import SimpleUNet


PROJECT_ROOT = Path.cwd()
INDEX_PATH = PROJECT_ROOT / "datasets" / "processed" / "structured3d" / "index.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "structured3d"
METRICS_REPORT = PROJECT_ROOT / "reports" / "training_metrics_report.md"

IMAGE_SIZE = (512, 256)
NUM_CLASSES = 8
BATCH_SIZE = 2
EPOCHS = int(os.environ.get("SPATIAL_TRAIN_EPOCHS", "10"))
LEARNING_RATE = 1e-4
NUM_WORKERS = 0


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. CPU fallback is prohibited.")

    return torch.device("cuda")


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()

    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def calculate_miou(preds, masks, num_classes):
    ious = []

    for class_id in range(num_classes):
        pred_mask = preds == class_id
        target_mask = masks == class_id
        union = (pred_mask | target_mask).sum().item()

        if union == 0:
            continue

        intersection = (pred_mask & target_mask).sum().item()
        ious.append(intersection / union)

    return sum(ious) / max(len(ious), 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    miou_total = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        preds = torch.argmax(logits, dim=1)

        correct += (preds == masks).sum().item()
        total += masks.numel()
        total_loss += loss.item()
        miou_total += calculate_miou(preds, masks, NUM_CLASSES)

    avg_loss = total_loss / len(loader)
    pixel_acc = correct / max(total, 1)
    miou = miou_total / len(loader)

    return avg_loss, pixel_acc, miou


def save_metrics_report(metrics):
    METRICS_REPORT.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Training Metrics Report",
        "",
        f"- epoch: {metrics['epoch']}",
        f"- train_loss: {metrics['train_loss']:.4f}",
        f"- val_loss: {metrics['val_loss']:.4f}",
        f"- pixel_accuracy: {metrics['val_acc']:.4f}",
        f"- mIoU: {metrics['miou']:.4f}",
    ]

    METRICS_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    seed_everything()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[INFO] Device: {device}")

    dataset = Structured3DSegmentationDataset(
        index_path=INDEX_PATH,
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
    )

    print(f"[INFO] Dataset size: {len(dataset)}")

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = SimpleUNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    start_epoch = 1
    last_path = CHECKPOINT_DIR / "last.pt"

    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        if checkpoint.get("num_classes") == NUM_CLASSES:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_val_loss = checkpoint.get("val_loss", best_val_loss)
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"[INFO] Resuming from epoch {start_epoch}")
        else:
            print("[INFO] Existing checkpoint has incompatible class count; starting fresh")

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
        )

        val_loss, val_acc, miou = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"miou={miou:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "miou": miou,
                "num_classes": NUM_CLASSES,
                "image_size": IMAGE_SIZE,
            },
            last_path,
        )

        save_metrics_report(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "miou": miou,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_path = CHECKPOINT_DIR / "best.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "miou": miou,
                    "num_classes": NUM_CLASSES,
                    "image_size": IMAGE_SIZE,
                },
                best_path,
            )

            print(f"[SAVE] Best checkpoint saved: {best_path}")


if __name__ == "__main__":
    main()

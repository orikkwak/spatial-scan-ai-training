from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.datasets.structured3d_dataset import Structured3DSegmentationDataset
from src.models.simple_unet import SimpleUNet


INDEX_PATH = r"C:\spatial-scan\ai-training\datasets\processed\structured3d\index.jsonl"
CHECKPOINT_DIR = Path(r"C:\spatial-scan\ai-training\checkpoints\structured3d")

IMAGE_SIZE = (512, 256)
NUM_CLASSES = 40
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 0


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()

    total_loss = 0.0

    progress = tqdm(loader, desc=f"Train Epoch {epoch}")

    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Valid Epoch {epoch}")

    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        preds = torch.argmax(logits, dim=1)

        correct += (preds == masks).sum().item()
        total += masks.numel()

        total_loss += loss.item()

        pixel_acc = correct / max(total, 1)
        progress.set_postfix(loss=loss.item(), acc=pixel_acc)

    avg_loss = total_loss / len(loader)
    pixel_acc = correct / max(total, 1)

    return avg_loss, pixel_acc


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

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        last_path = CHECKPOINT_DIR / "last.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_classes": NUM_CLASSES,
                "image_size": IMAGE_SIZE,
            },
            last_path,
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
                    "num_classes": NUM_CLASSES,
                    "image_size": IMAGE_SIZE,
                },
                best_path,
            )

            print(f"[SAVE] Best checkpoint saved: {best_path}")


if __name__ == "__main__":
    main()

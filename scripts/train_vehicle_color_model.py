#!/usr/bin/env python
"""Train the MobileNetV3-small vehicle-color classifier.

Trains the model defined in ``udfs.vehicle_color_model.build_color_model`` on the
8-color *Vehicle Color Recognition* dataset (Chen et al.) and writes a
``state_dict`` checkpoint that :func:`udfs.vehicle_color_model.predict_color_from_crop`
can load at inference time.

Dataset layout
--------------
Point ``--data-dir`` at an ``ImageFolder``-style directory with one subfolder per
color. Either provide pre-split ``train/`` and ``val/`` roots, or a single root
that this script splits ``--val-split`` off of::

    <data-dir>/
      black/  *.jpg
      blue/   *.jpg
      cyan/   *.jpg
      gray/   *.jpg
      green/  *.jpg
      red/    *.jpg
      white/  *.jpg
      yellow/ *.jpg

The folder names must match ``udfs.vehicle_color_model.CHEN_COLOR_CLASSES``
exactly; ``ImageFolder`` sorts classes alphabetically, which is the canonical
label order the inference path relies on.

Usage
-----
::

    PYTHONPATH=src:. python scripts/train_vehicle_color_model.py \
        --data-dir data/vehicle_color --epochs 15 \
        --output models/vehicle_color_mobilenet.pt

This box is CPU-only; run on a CUDA machine (``--device cuda``) for real speed.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make ``udfs`` importable when run as a plain script from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from udfs.vehicle_color_model import CHEN_COLOR_CLASSES, build_color_model


def _build_transforms(train: bool):
    from torchvision import transforms

    base = [
        transforms.Resize((224, 224)),
    ]
    if train:
        base += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ]
    return transforms.Compose(base)


def _load_datasets(data_dir: str, val_split: float):
    """Return (train_ds, val_ds), validating the class vocabulary."""
    import torch
    from torchvision.datasets import ImageFolder

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_ds = ImageFolder(train_dir, transform=_build_transforms(train=True))
        val_ds = ImageFolder(val_dir, transform=_build_transforms(train=False))
        _validate_classes(train_ds.classes)
        _validate_classes(val_ds.classes)
        return train_ds, val_ds

    # Single-root: split deterministically.
    full = ImageFolder(data_dir, transform=_build_transforms(train=True))
    _validate_classes(full.classes)
    val_size = max(1, int(len(full) * val_split))
    train_size = len(full) - val_size
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        full, [train_size, val_size], generator=generator
    )
    # The val subset should not be augmented; rebuild it with eval transforms.
    val_base = ImageFolder(data_dir, transform=_build_transforms(train=False))
    val_ds = torch.utils.data.Subset(val_base, val_ds.indices)
    return train_ds, val_ds


def _validate_classes(classes: list[str]) -> None:
    if tuple(classes) != CHEN_COLOR_CLASSES:
        raise SystemExit(
            "Dataset classes do not match the expected color vocabulary.\n"
            f"  expected (alphabetical): {list(CHEN_COLOR_CLASSES)}\n"
            f"  found:                   {list(classes)}\n"
            "Rename the color subfolders so they match exactly."
        )


def _evaluate(model, loader, device) -> float:
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)
    return correct / total if total else 0.0


def train(args: argparse.Namespace) -> None:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds = _load_datasets(args.data_dir, args.val_split)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    model = build_color_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * images.size(0)

        train_loss = running / len(train_ds)
        val_acc = _evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:>2}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  saved checkpoint (val_acc={val_acc:.4f}) -> {args.output}")

    print(f"Best val_acc={best_acc:.4f}; checkpoint at {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="ImageFolder root.")
    parser.add_argument(
        "--output",
        default=os.path.join("models", "vehicle_color_mobilenet.pt"),
        help="Where to write the state_dict checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device", default=None, help="cuda / cpu (default: auto-detect)."
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()

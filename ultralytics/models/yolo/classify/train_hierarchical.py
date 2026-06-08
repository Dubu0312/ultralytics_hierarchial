#!/usr/bin/env python3
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
Hierarchical YOLO Classification Training Script.

This script trains a YOLO classification model with hierarchical multi-head outputs
(order, family, genus, species) similar to the ViT hierarchical approach.

Data layout:
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg

CSV (IDs):
  mapping.csv with columns: class_name, ordor, familia, genus
    - ordor, familia, genus are INT IDs you provide.
    - species ID is derived from class_name across train+val.

Usage:
  python train_hierarchical.py --data /path/to/dataset --mapping_csv /path/to/mapping.csv --epochs 100
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import Any, Dict, List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ultralytics import YOLO
from ultralytics.nn.modules.head import HierarchicalClassify
from ultralytics.nn.modules.conv import Conv

# =========================
# DEFAULT CONFIG
# =========================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Loss weights for each level
LAMBDA = {
    "ordor": 1.0,
    "familia": 0.9,
    "genus": 0.8,
    "species": 0.7,
}


# =========================
# Utilities
# =========================
def list_images(root: str) -> Dict[str, List[str]]:
    """List all images in subdirectories of root."""
    if not os.path.isdir(root):
        return {}
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    out = {}
    for c in classes:
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths += glob.glob(os.path.join(root, c, "**", ext), recursive=True)
        if paths:
            out[c] = paths
    return out


def build_species_ids(train_root: str, val_root: str) -> Dict[str, int]:
    """Build species ID mapping from train and val folders."""
    names = set(list_images(train_root).keys()) | set(list_images(val_root).keys())
    if not names:
        raise SystemExit("No classes found in train/ or val/.")
    return {name: i for i, name in enumerate(sorted(names))}


def df_from_folder_plus_csv(root: str, map_df: pd.DataFrame, species2id: Dict[str, int]) -> pd.DataFrame:
    """Build DataFrame with image paths and hierarchical labels."""
    rows = []
    class_to_row = {r["class_name"]: r for _, r in map_df.iterrows()}
    class_images = list_images(root)
    for cls, paths in class_images.items():
        if cls not in class_to_row:
            raise ValueError(f"class_name '{cls}' not found in mapping CSV")
        r = class_to_row[cls]
        o = int(r["ordor"])
        f = int(r["familia"])
        g = int(r["genus"])
        s = species2id[cls]
        for p in paths:
            rows.append({"image_path": p, "ordor": o, "familia": f, "genus": g, "species": s})
    return pd.DataFrame(rows)


def build_parents(map_df: pd.DataFrame, species2id: Dict[str, int]):
    """Build parent lookup arrays for hierarchical structure."""
    n_ord = int(map_df["ordor"].max()) + 1
    n_fam = int(map_df["familia"].max()) + 1
    n_gen = int(map_df["genus"].max()) + 1
    n_spe = len(species2id)

    familia_parent = torch.full((n_fam,), -1, dtype=torch.long)
    genus_parent = torch.full((n_gen,), -1, dtype=torch.long)
    species_parent = torch.full((n_spe,), -1, dtype=torch.long)

    for _, r in map_df.iterrows():
        o, f, g = int(r["ordor"]), int(r["familia"]), int(r["genus"])
        familia_parent[f] = o
        genus_parent[g] = f
        species_parent[species2id[r["class_name"]]] = g

    return familia_parent, genus_parent, species_parent


# =========================
# Dataset
# =========================
class HierarchicalDataset(Dataset):
    """Dataset for hierarchical classification."""

    def __init__(self, df: pd.DataFrame, imgsz: int = 224, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.imgsz = imgsz
        self.augment = augment

        # Augmentation transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(imgsz, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((imgsz, imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        x = self.transform(img)
        y = {
            "ordor": torch.tensor(int(r["ordor"]), dtype=torch.long),
            "familia": torch.tensor(int(r["familia"]), dtype=torch.long),
            "genus": torch.tensor(int(r["genus"]), dtype=torch.long),
            "species": torch.tensor(int(r["species"]), dtype=torch.long),
        }
        return x, y


# =========================
# Model
# =========================
class YOLOHierarchicalClassifier(nn.Module):
    """YOLO-based hierarchical classifier with 4 heads."""

    def __init__(self, base_model_path: str, n_ordor: int, n_familia: int, n_genus: int, n_species: int):
        super().__init__()

        # Load YOLO classification model as backbone
        yolo_model = YOLO(base_model_path)
        self.backbone = yolo_model.model.model[:-1]  # Remove the original classification head

        # Get the output channels from the backbone
        # Run a dummy forward to get the channel size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            for layer in self.backbone:
                dummy = layer(dummy)
            c1 = dummy.shape[1]

        # Feature processing
        c_ = 1280  # Feature dimension
        self.conv = Conv(c1, c_, k=1, s=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)

        # Create 4 classification heads
        self.head_ordor = nn.Linear(c_, n_ordor)
        self.head_familia = nn.Linear(c_, n_familia)
        self.head_genus = nn.Linear(c_, n_genus)
        self.head_species = nn.Linear(c_, n_species)

    def forward(self, x):
        # Forward through backbone
        for layer in self.backbone:
            x = layer(x)

        # Feature processing
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)

        # 4 classification heads
        return (
            self.head_ordor(x),
            self.head_familia(x),
            self.head_genus(x),
            self.head_species(x),
        )


# =========================
# Loss & Metrics
# =========================
def multitask_ce(logits, targets, lambdas):
    """Compute multi-task cross-entropy loss."""
    lo, lf, lg, ls = logits
    Lo = F.cross_entropy(lo, targets["ordor"])
    Lf = F.cross_entropy(lf, targets["familia"])
    Lg = F.cross_entropy(lg, targets["genus"])
    Ls = F.cross_entropy(ls, targets["species"])
    total = lambdas["ordor"] * Lo + lambdas["familia"] * Lf + lambdas["genus"] * Lg + lambdas["species"] * Ls
    return total, (Lo.item(), Lf.item(), Lg.item(), Ls.item())


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on validation set."""
    model.eval()
    o_pred, o_true = [], []
    f_pred, f_true = [], []
    g_pred, g_true = [], []
    s_pred, s_true = [], []

    for x, y in loader:
        x = x.to(device)
        y = {k: v.to(device) for k, v in y.items()}
        lo, lf, lg, ls = model(x)
        o_pred += lo.argmax(1).cpu().tolist()
        o_true += y["ordor"].cpu().tolist()
        f_pred += lf.argmax(1).cpu().tolist()
        f_true += y["familia"].cpu().tolist()
        g_pred += lg.argmax(1).cpu().tolist()
        g_true += y["genus"].cpu().tolist()
        s_pred += ls.argmax(1).cpu().tolist()
        s_true += y["species"].cpu().tolist()

    acc_o = accuracy_score(o_true, o_pred)
    acc_f = accuracy_score(f_true, f_pred)
    acc_g = accuracy_score(g_true, g_pred)
    acc_s = accuracy_score(s_true, s_pred)
    hier = sum(
        (a == b) and (c == d) and (e == f) and (g == h)
        for a, b, c, d, e, f, g, h in zip(o_pred, o_true, f_pred, f_true, g_pred, g_true, s_pred, s_true)
    ) / len(o_true)
    return {"ordor": acc_o, "familia": acc_f, "genus": acc_g, "species": acc_s, "hier": hier}


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Train hierarchical YOLO classification model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset directory (containing train/ and val/)")
    parser.add_argument("--mapping_csv", type=str, required=True, help="Path to mapping CSV file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size")
    parser.add_argument("--model", type=str, default="yolov8s-cls.pt", help="Base YOLO classification model")
    parser.add_argument("--output", type=str, default="outputs_yolo_hierarchical", help="Output directory")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--freeze-epochs", type=int, default=3, help="Number of epochs to freeze backbone")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (0 to disable)")

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths
    train_root = os.path.join(args.data, "train")
    val_root = os.path.join(args.data, "val")

    # Read mapping CSV
    print(f"Loading mapping CSV: {args.mapping_csv}")
    map_df = pd.read_csv(args.mapping_csv)
    cols = set(map_df.columns)
    if "order" in cols and "ordor" not in cols:
        map_df = map_df.rename(columns={"order": "ordor"})
    assert set(["class_name", "ordor", "familia", "genus"]).issubset(map_df.columns), \
        "mapping.csv must have columns: class_name, ordor, familia, genus"

    # Build species IDs
    species2id = build_species_ids(train_root, val_root)
    json.dump(species2id, open(os.path.join(args.output, "species_to_id.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Found {len(species2id)} species classes")

    # Build DataFrames
    df_train = df_from_folder_plus_csv(train_root, map_df, species2id)
    df_val = df_from_folder_plus_csv(val_root, map_df, species2id)
    df_train.to_csv(os.path.join(args.output, "train_expanded.csv"), index=False)
    df_val.to_csv(os.path.join(args.output, "val_expanded.csv"), index=False)
    print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}")

    # Class counts
    n_ord = int(map_df["ordor"].max()) + 1
    n_fam = int(map_df["familia"].max()) + 1
    n_gen = int(map_df["genus"].max()) + 1
    n_spe = len(species2id)
    print(f"Classes - Order: {n_ord}, Family: {n_fam}, Genus: {n_gen}, Species: {n_spe}")

    # Build parent mappings
    familia_parent, genus_parent, species_parent = build_parents(map_df, species2id)

    # Datasets and loaders
    train_ds = HierarchicalDataset(df_train, imgsz=args.imgsz, augment=True)
    val_ds = HierarchicalDataset(df_val, imgsz=args.imgsz, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Model
    print(f"Loading base model: {args.model}")
    model = YOLOHierarchicalClassifier(args.model, n_ord, n_fam, n_gen, n_spe).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Freeze backbone for initial epochs
    def set_backbone_grad(flag: bool):
        for p in model.backbone.parameters():
            p.requires_grad = flag

    set_backbone_grad(False)
    print(f"Backbone frozen for first {args.freeze_epochs} epochs")
    if args.patience > 0:
        print(f"Early stopping patience: {args.patience} epochs")

    # Training loop
    best_hier = -1.0
    patience_counter = 0
    print("\nStarting training...")
    print("=" * 100)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if epoch == args.freeze_epochs + 1:
            set_backbone_grad(True)
            print(f"Epoch {epoch}: Backbone unfrozen")

        total_running = 0.0
        Lo = Lf = Lg = Ls = 0.0

        for x, y in train_dl:
            x = x.to(device)
            y = {k: v.to(device) for k, v in y.items()}

            logits = model(x)
            loss, parts = multitask_ce(logits, y, LAMBDA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_running += loss.item()
            Lo += parts[0]
            Lf += parts[1]
            Lg += parts[2]
            Ls += parts[3]

        n_batches = max(1, len(train_dl))
        train_loss = total_running / n_batches
        Lo /= n_batches
        Lf /= n_batches
        Lg /= n_batches
        Ls /= n_batches

        # Validation
        metrics = evaluate(model, val_dl, device)
        print(
            f"Epoch {epoch:03d} | total={train_loss:.4f} "
            f"(Lo={Lo:.3f}, Lf={Lf:.3f}, Lg={Lg:.3f}, Ls={Ls:.3f}) | "
            f"val_acc - ordor={metrics['ordor']:.3f} familia={metrics['familia']:.3f} "
            f"genus={metrics['genus']:.3f} species={metrics['species']:.3f} hier={metrics['hier']:.3f}"
        )

        # Save best model
        if metrics["hier"] > best_hier:
            best_hier = metrics["hier"]
            patience_counter = 0  # Reset patience counter
            ckpt = {
                "model_state": model.state_dict(),
                "base_model": args.model,
                "species2id": species2id,
                "familia_parent": familia_parent.tolist(),
                "genus_parent": genus_parent.tolist(),
                "species_parent": species_parent.tolist(),
                "lambdas": LAMBDA,
                "n_classes": {"ordor": n_ord, "familia": n_fam, "genus": n_gen, "species": n_spe},
            }
            torch.save(ckpt, os.path.join(args.output, "best_model.pt"))
            print(f"  -> Saved best checkpoint (hier={best_hier:.3f})")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {args.patience} epochs without improvement.")
                break

    print("=" * 100)
    print(f"Training completed at epoch {epoch}! Best hierarchical accuracy: {best_hier:.3f}")
    print(f"Model saved to: {os.path.join(args.output, 'best_model.pt')}")


if __name__ == "__main__":
    main()

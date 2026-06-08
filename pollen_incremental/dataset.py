"""Hierarchical image dataset for incremental sessions.

Re-implemented from train_hierarchical_masked.py:128-167 with three differences:
    1. Class IDs come from a TaxonomyState (append-only) instead of being
       re-derived from folder alphabetic order — this is the critical fix
       that makes incremental learning safe (see taxonomy.py docstring).
    2. Supports loading from a list of (path, species_id) tuples — needed for
       replay (paths come from exemplar memory).
    3. Knows about `species2id` so it can translate folder name → species ID.

Same image-level augmentation as the base trainer.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _train_transform(imgsz: int = 224) -> transforms.Compose:
    """Match train_hierarchical_masked.py augmentation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(imgsz, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _eval_transform(imgsz: int = 224) -> transforms.Compose:
    """No augmentation."""
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def list_images(split_root: Path | str, species_filter: set[str] | None = None) -> dict[str, list[str]]:
    """List image paths per species folder under split_root/<species>/*.{jpg,...}.

    Args:
        split_root: e.g. /home/dubu/.../streams/session_2/train
        species_filter: if given, only return species in this set.

    Returns:
        dict {species_name: [path, ...]}.
    """
    split_root = Path(split_root)
    if not split_root.is_dir():
        return {}
    out: dict[str, list[str]] = {}
    for cls_dir in sorted(split_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        if species_filter is not None and cls_dir.name not in species_filter:
            continue
        paths: list[str] = []
        for ext in IMG_EXTS:
            paths.extend(str(p) for p in cls_dir.glob(f"**/*{ext}"))
        if paths:
            out[cls_dir.name] = sorted(paths)
    return out


def build_dataframe(
    split_root: Path | str,
    mapping_df: pd.DataFrame,
    species2id: dict[str, int],
    species_filter: set[str] | None = None,
) -> pd.DataFrame:
    """Build a flat dataframe of (image_path, ordor, familia, genus, species) rows.

    Used to feed `HierarchicalImageDataset`. Skips species not present in
    `species2id` (with a warning would be nice but caller does that).

    Args:
        split_root: dir containing species subfolders.
        mapping_df: must have columns class_name, ordor, familia, genus.
        species2id: name → ID map from TaxonomyState.
        species_filter: optional set of species names to keep.

    Returns:
        DataFrame with one row per image.
    """
    if "order" in mapping_df.columns and "ordor" not in mapping_df.columns:
        mapping_df = mapping_df.rename(columns={"order": "ordor"})

    class_to_row = {r["class_name"]: r for _, r in mapping_df.iterrows()}
    rows = []
    for cls, paths in list_images(split_root, species_filter).items():
        if cls not in class_to_row or cls not in species2id:
            continue
        r = class_to_row[cls]
        o, f, g = int(r["ordor"]), int(r["familia"]), int(r["genus"])
        s = species2id[cls]
        for p in paths:
            rows.append({
                "image_path": p, "ordor": o, "familia": f, "genus": g, "species": s,
            })
    return pd.DataFrame(rows)


class HierarchicalImageDataset(Dataset):
    """Loads images + 4-level labels from a flat dataframe.

    The dataframe must have columns: image_path, ordor, familia, genus, species
    (4 integer IDs already resolved). Build via `build_dataframe()` for a
    session split, or by concatenating rows from exemplar memory for replay.
    """

    def __init__(self, df: pd.DataFrame, imgsz: int = 224, augment: bool = False) -> None:
        self.df = df.reset_index(drop=True)
        self.imgsz = imgsz
        self.transform = _train_transform(imgsz) if augment else _eval_transform(imgsz)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        r = self.df.iloc[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        x = self.transform(img)
        y = {
            "ordor":   torch.tensor(int(r["ordor"]),   dtype=torch.long),
            "familia": torch.tensor(int(r["familia"]), dtype=torch.long),
            "genus":   torch.tensor(int(r["genus"]),   dtype=torch.long),
            "species": torch.tensor(int(r["species"]), dtype=torch.long),
        }
        return x, y


def build_replay_dataframe(
    memory_items: Iterable[tuple[str, int]],
    species_parent: list[int],
    genus_parent: list[int],
    familia_parent: list[int],
) -> pd.DataFrame:
    """Convert (image_path, species_id) pairs from ExemplarMemory into a full
    4-level dataframe by deriving parents from the taxonomy state.

    Args:
        memory_items: iterable of (path, species_id) tuples.
        species_parent / genus_parent / familia_parent: parent lookup lists.

    Returns:
        DataFrame ready to feed HierarchicalImageDataset.
    """
    rows = []
    for path, sid in memory_items:
        g = species_parent[sid]
        f = genus_parent[g] if g >= 0 else -1
        o = familia_parent[f] if f >= 0 else -1
        rows.append({"image_path": path, "ordor": o, "familia": f, "genus": g, "species": sid})
    return pd.DataFrame(rows)

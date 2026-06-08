#!/usr/bin/env python3
"""
prepare_cascade_datasets.py — Create 3 datasets (order/family/genus) by re-grouping
images from the existing species-level dataset (dataset-dongvan-train).

Each new dataset reorganizes images by parent level for training a flat classifier
per level. Output structure:

  /home/dubu/manh/dongvan-yolo/
  ├── dataset-dongvan-train/       (existing, species-level)
  ├── dataset-dongvan-order/       (new)
  │   ├── train/ord_1/  ord_2/  ord_3/  ord_4/  ord_5/
  │   ├── val/
  │   └── test/
  ├── dataset-dongvan-family/      (new)
  │   ├── train/fam_1/ ... fam_9/
  │   └── ...
  └── dataset-dongvan-genus/       (new)
      ├── train/gen_1/ ... gen_15/
      └── ...
"""

import os
import shutil
from pathlib import Path
import pandas as pd

SRC_ROOT = Path("/home/dubu/manh/dongvan-yolo/dataset-dongvan-train")
DST_BASE = Path("/home/dubu/manh/dongvan-yolo")
MAPPING_CSV = Path("/home/dubu/manh/dongvan-yolo/pollen_dong_van.csv")
SPLITS = ["train", "val", "test"]
LEVELS = {
    "order":  ("ordor",   "ord"),    # CSV column, folder prefix
    "family": ("familia", "fam"),
    "genus":  ("genus",   "gen"),
}


def main():
    # Read mapping
    map_df = pd.read_csv(MAPPING_CSV)
    if "order" in map_df.columns and "ordor" not in map_df.columns:
        map_df = map_df.rename(columns={"order": "ordor"})

    # Build species → (order, family, genus)
    species_to_levels = {}
    for _, r in map_df.iterrows():
        species_to_levels[r["class_name"]] = {
            "order":  int(r["ordor"]),
            "family": int(r["familia"]),
            "genus":  int(r["genus"]),
        }

    # For each new dataset
    for level_name, (csv_col, prefix) in LEVELS.items():
        dst_dataset = DST_BASE / f"dataset-dongvan-{level_name}"
        print(f"\n=== Creating {dst_dataset} ===")

        copied_count = {split: 0 for split in SPLITS}
        for split in SPLITS:
            src_split = SRC_ROOT / split
            if not src_split.exists():
                print(f"  [SKIP] {src_split} not found")
                continue

            # Iterate species folders in this split
            for species_dir in sorted(src_split.iterdir()):
                if not species_dir.is_dir():
                    continue
                species = species_dir.name
                if species not in species_to_levels:
                    print(f"  [WARN] {species} not in mapping CSV, skipped")
                    continue

                level_id = species_to_levels[species][level_name]
                dst_class = dst_dataset / split / f"{prefix}_{level_id}"
                dst_class.mkdir(parents=True, exist_ok=True)

                # Copy all images (use shutil.copy2 to preserve timestamps)
                for img in species_dir.iterdir():
                    if img.is_file() and img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        # Prefix with species name to avoid filename collisions
                        # (e.g. cler_1/Image-1.jpg and cler/Image-1.jpg)
                        dst_file = dst_class / f"{species}__{img.name}"
                        shutil.copy2(img, dst_file)
                        copied_count[split] += 1

        # Report
        for split in SPLITS:
            n_classes = len(list((dst_dataset / split).iterdir())) if (dst_dataset / split).exists() else 0
            print(f"  {split}: {copied_count[split]} images in {n_classes} classes")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

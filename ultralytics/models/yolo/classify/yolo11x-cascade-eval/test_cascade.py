#!/usr/bin/env python3
"""
test_cascade.py — Evaluate 4 independent flat YOLO classifiers (order/family/genus/species)
on the pollen test set.

For each test image, runs all 4 models independently → 4 predictions → compute per-level
accuracy + HierAcc. Outputs match the format of test_yolo.py / test_flat.py for direct
comparison with multi-head pipelines.

Usage (from this directory):
  python test_cascade.py
  python test_cascade.py --consistency-decode  # mask child predictions by predicted parent

Default paths assume the 4 trained checkpoints are at:
  ../yolo11x-flat-order/weights/best.pt
  ../yolo11x-flat-family/weights/best.pt
  ../yolo11x-flat-genus/weights/best.pt
  ../yolo11x-flat-default/weights/best.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from ultralytics import YOLO


IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def list_test_images(root: str) -> Dict[str, List[str]]:
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    out = {}
    for c in classes:
        paths = []
        for ext in IMG_EXTS:
            paths += glob.glob(os.path.join(root, c, "**", ext), recursive=True)
        if paths:
            out[c] = paths
    return out


def build_test_df(root: str, map_df: pd.DataFrame, species2id: Dict[str, int]) -> pd.DataFrame:
    rows = []
    class_to_row = {r["class_name"]: r for _, r in map_df.iterrows()}
    for cls, paths in list_test_images(root).items():
        if cls not in class_to_row or cls not in species2id:
            continue
        r = class_to_row[cls]
        for p in paths:
            rows.append({
                "image_path": p, "class_name": cls,
                "ordor": int(r["ordor"]), "familia": int(r["familia"]),
                "genus": int(r["genus"]), "species": species2id[cls],
            })
    return pd.DataFrame(rows)


def parse_id_from_class_name(name: str) -> int:
    """Extract integer ID from class folder name (e.g. 'ord_3' → 3, 'fam_5' → 5)."""
    m = re.search(r"_(\d+)$", name)
    if m is None:
        raise ValueError(f"Cannot parse ID from class name: {name}")
    return int(m.group(1))


def predict_batch(model, image_paths: List[str], imgsz: int, batch_size: int) -> List[int]:
    """Run YOLO.predict on a list of paths, return list of predicted class IDs.

    For order/family/genus models, the class name is e.g. 'ord_3' → we extract 3.
    For species model, the class name is the species name → caller must map via species2id.
    Returns the raw top-1 class index from model.names (not the parsed ID).
    """
    preds = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        results = model.predict(batch, imgsz=imgsz, verbose=False)
        for r in results:
            preds.append(int(r.probs.top1))
    return preds


def build_parent_arrays(map_df: pd.DataFrame, species2id: Dict[str, int]):
    n_fam = int(map_df["familia"].max()) + 1
    n_gen = int(map_df["genus"].max()) + 1
    n_spe = len(species2id)
    familia_parent = np.full(n_fam, -1, dtype=np.int64)
    genus_parent = np.full(n_gen, -1, dtype=np.int64)
    species_parent = np.full(n_spe, -1, dtype=np.int64)
    for _, r in map_df.iterrows():
        o, f, g = int(r["ordor"]), int(r["familia"]), int(r["genus"])
        familia_parent[f] = o
        genus_parent[g] = f
        species_parent[species2id[r["class_name"]]] = g
    return familia_parent, genus_parent, species_parent


def _plot_cm(cm, names, title, png_path, dpi=150, annotate=True):
    n = len(names)
    size = max(6, min(36, 0.6 * n))
    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=14); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if annotate and n <= 30:
        is_float = np.issubdtype(cm.dtype, np.floating)
        for i in range(n):
            for j in range(n):
                v = cm[i, j]
                if v > 0:
                    ax.text(j, i, f"{v:.2f}" if is_float else str(int(v)),
                            ha="center", va="center", fontsize=9)
    fig.tight_layout(); plt.savefig(png_path, dpi=dpi); plt.close(fig)


def save_cm(head_name, y_true, y_pred, names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    labels = sorted(set(y_true) | set(y_pred))
    idx_map = {lab: i for i, lab in enumerate(labels)}
    yt = [idx_map[i] for i in y_true]; yp = [idx_map[i] for i in y_pred]
    label_names = [names.get(i, str(i)) for i in labels]
    cm = confusion_matrix(yt, yp, labels=list(range(len(labels))))
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(
        os.path.join(out_dir, f"{head_name}_confmat.csv"), encoding="utf-8")
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, rs, where=rs != 0)
    pd.DataFrame(cm_norm, index=label_names, columns=label_names).to_csv(
        os.path.join(out_dir, f"{head_name}_confmat_norm.csv"), encoding="utf-8")
    _plot_cm(cm, label_names, f"{head_name.capitalize()} CM",
             os.path.join(out_dir, f"{head_name}_confmat.png"))
    _plot_cm(cm_norm, label_names, f"{head_name.capitalize()} CM (normalized)",
             os.path.join(out_dir, f"{head_name}_confmat_norm.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_order", default="../yolo11x-flat-order/weights/best.pt")
    ap.add_argument("--ckpt_family", default="../yolo11x-flat-family/weights/best.pt")
    ap.add_argument("--ckpt_genus", default="../yolo11x-flat-genus/weights/best.pt")
    ap.add_argument("--ckpt_species", default="../yolo11x-flat-default/weights/best.pt")
    ap.add_argument("--root_test", default="/home/dubu/manh/dongvan-yolo/dataset-dongvan-train/test")
    ap.add_argument("--mapping_csv", default="/home/dubu/manh/dongvan-yolo/pollen_dong_van.csv")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--consistency_decode", action="store_true",
                    help="Mask child predictions by predicted parent (top-down)")
    ap.add_argument("--out_csv", default="predictions_cascade.csv")
    ap.add_argument("--confmat_dir", default="cm_cascade")
    args = ap.parse_args()

    # Load 4 models
    print("Loading models...")
    m_order = YOLO(args.ckpt_order)
    m_family = YOLO(args.ckpt_family)
    m_genus = YOLO(args.ckpt_genus)
    m_species = YOLO(args.ckpt_species)

    # Build maps from each model's class index → real ID
    # For order/family/genus models, model.names is e.g. {0: 'ord_1', 1: 'ord_2', ...}
    # We need: model class idx → real ordor/familia/genus ID
    order_idx_to_id = {idx: parse_id_from_class_name(name) for idx, name in m_order.names.items()}
    family_idx_to_id = {idx: parse_id_from_class_name(name) for idx, name in m_family.names.items()}
    genus_idx_to_id = {idx: parse_id_from_class_name(name) for idx, name in m_genus.names.items()}
    species_idx_to_name = m_species.names  # {idx: 'abel', 'cler_1', ...}

    # Species ID = model class idx for species model (since it was trained on species folders directly)
    species2id = {v: int(k) for k, v in species_idx_to_name.items()}
    print(f"  Order classes:   {len(order_idx_to_id)}  → real IDs: {sorted(order_idx_to_id.values())}")
    print(f"  Family classes:  {len(family_idx_to_id)} → real IDs: {sorted(family_idx_to_id.values())}")
    print(f"  Genus classes:   {len(genus_idx_to_id)}  → real IDs: {sorted(genus_idx_to_id.values())}")
    print(f"  Species classes: {len(species2id)}")

    # Mapping CSV + test df
    map_df = pd.read_csv(args.mapping_csv)
    if "order" in map_df.columns and "ordor" not in map_df.columns:
        map_df = map_df.rename(columns={"order": "ordor"})
    df = build_test_df(args.root_test, map_df, species2id)
    print(f"\nTest samples: {len(df)}")

    image_paths = df["image_path"].tolist()

    # Run 4 models
    print("\nRunning 4 models...")
    print("  [1/4] Order...")
    order_preds_idx = predict_batch(m_order, image_paths, args.imgsz, args.batch_size)
    print("  [2/4] Family...")
    family_preds_idx = predict_batch(m_family, image_paths, args.imgsz, args.batch_size)
    print("  [3/4] Genus...")
    genus_preds_idx = predict_batch(m_genus, image_paths, args.imgsz, args.batch_size)
    print("  [4/4] Species...")
    species_preds = predict_batch(m_species, image_paths, args.imgsz, args.batch_size)

    # Convert order/family/genus model indices to real taxonomy IDs
    o_preds = [order_idx_to_id[i] for i in order_preds_idx]
    f_preds = [family_idx_to_id[i] for i in family_preds_idx]
    g_preds = [genus_idx_to_id[i] for i in genus_preds_idx]
    s_preds = species_preds

    # GT
    o_true = df["ordor"].astype(int).tolist()
    f_true = df["familia"].astype(int).tolist()
    g_true = df["genus"].astype(int).tolist()
    s_true = df["species"].astype(int).tolist()

    # Optional: consistency decoding (mask child predictions by predicted parent)
    if args.consistency_decode:
        print("\n[Consistency decoding] Re-deriving family/genus/species from predicted parent (top-down)")
        familia_parent, genus_parent, species_parent = build_parent_arrays(map_df, species2id)
        # For each sample: derive everything from predicted ORDER, falling back to original prediction
        # if its parent isn't the predicted parent.
        # Strategy: use original prediction if it's consistent with predicted parent; else fall back to
        # the most-likely class among valid (consistent) classes. Since we don't have logits here
        # (just argmax), we fall back to using the model's top-1 directly — i.e., we only OVERRIDE
        # a child if it's inconsistent with predicted parent, replacing it with... what?
        # For a cascade w/o logits, simplest is: keep predictions as-is and just measure HierAcc.
        # True consistency decoding requires logits. Skip for now (warning).
        print("  WARN: cascade without logits cannot do true masked decoding. Keeping raw predictions.")

    # Metrics
    def per_level(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return acc, f1m, f1w

    acc_o, f1m_o, f1w_o = per_level(o_true, o_preds)
    acc_f, f1m_f, f1w_f = per_level(f_true, f_preds)
    acc_g, f1m_g, f1w_g = per_level(g_true, g_preds)
    acc_s, f1m_s, f1w_s = per_level(s_true, s_preds)
    hier = sum(int(a == b and c == d and e == f and g == h)
               for a, b, c, d, e, f, g, h in
               zip(o_preds, o_true, f_preds, f_true, g_preds, g_true, s_preds, s_true)) / len(s_true)

    # Consistency check: how many predictions are internally consistent with taxonomy?
    familia_parent, genus_parent, species_parent = build_parent_arrays(map_df, species2id)
    consistent_count = 0
    for op, fp, gp, sp in zip(o_preds, f_preds, g_preds, s_preds):
        # Check: parent of f == op, parent of g == fp, parent of s == gp
        ok = (0 <= fp < len(familia_parent) and familia_parent[fp] == op
              and 0 <= gp < len(genus_parent) and genus_parent[gp] == fp
              and 0 <= sp < len(species_parent) and species_parent[sp] == gp)
        if ok:
            consistent_count += 1
    consistency_rate = consistent_count / len(s_true)

    print("\n" + "=" * 100)
    print(f"[CASCADE — 4 independent yolo11x-cls models]")
    print(f"  ordor:   acc={acc_o:.3f}, f1_macro={f1m_o:.3f}, f1_weighted={f1w_o:.3f}")
    print(f"  familia: acc={acc_f:.3f}, f1_macro={f1m_f:.3f}, f1_weighted={f1w_f:.3f}")
    print(f"  genus:   acc={acc_g:.3f}, f1_macro={f1m_g:.3f}, f1_weighted={f1w_g:.3f}")
    print(f"  species: acc={acc_s:.3f}, f1_macro={f1m_s:.3f}, f1_weighted={f1w_s:.3f}")
    print(f"  hierarchical accuracy:    {hier:.3f}")
    print(f"  taxonomy consistency rate: {consistency_rate:.3f}  "
          f"({consistent_count}/{len(s_true)} predictions form valid path in tree)")
    print("=" * 100)

    # Save predictions table
    rows = []
    for i in range(len(df)):
        rows.append({
            "image_path": df["image_path"].iloc[i],
            "class_name": df["class_name"].iloc[i],
            "true_ordor": o_true[i], "true_familia": f_true[i],
            "true_genus": g_true[i], "true_species": s_true[i],
            "pred_ordor": o_preds[i], "pred_familia": f_preds[i],
            "pred_genus": g_preds[i], "pred_species": s_preds[i],
        })
    pd.DataFrame(rows).to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"\nSaved predictions: {args.out_csv}")

    summary = pd.DataFrame([
        {"head": "ordor", "accuracy": acc_o, "f1_macro": f1m_o, "f1_weighted": f1w_o},
        {"head": "familia", "accuracy": acc_f, "f1_macro": f1m_f, "f1_weighted": f1w_f},
        {"head": "genus", "accuracy": acc_g, "f1_macro": f1m_g, "f1_weighted": f1w_g},
        {"head": "species", "accuracy": acc_s, "f1_macro": f1m_s, "f1_weighted": f1w_s},
        {"head": "hierarchical", "accuracy": hier, "f1_macro": None, "f1_weighted": None},
        {"head": "consistency_rate", "accuracy": consistency_rate, "f1_macro": None, "f1_weighted": None},
    ])
    summary.to_csv("metrics_summary_cascade.csv", index=False, encoding="utf-8")
    print("Saved metrics_summary_cascade.csv")

    # CMs
    if args.confmat_dir:
        print(f"\nGenerating confusion matrices: {args.confmat_dir}")
        id_to_species = {v: k for k, v in species2id.items()}
        save_cm("species", s_true, s_preds, id_to_species, args.confmat_dir)
        save_cm("ordor", o_true, o_preds, {i: f"ord:{i}" for i in set(o_true) | set(o_preds)}, args.confmat_dir)
        save_cm("familia", f_true, f_preds, {i: f"fam:{i}" for i in set(f_true) | set(f_preds)}, args.confmat_dir)
        save_cm("genus", g_true, g_preds, {i: f"gen:{i}" for i in set(g_true) | set(g_preds)}, args.confmat_dir)
        print("Done.")


if __name__ == "__main__":
    main()

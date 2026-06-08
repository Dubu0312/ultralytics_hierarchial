#!/usr/bin/env python3
"""
test.py
Evaluate a saved hierarchical ViT checkpoint on a test set and (optionally)
export confusion matrices per head (raw + row-normalized), with YOLO-like
blue colormap on white background. Also computes F1 scores.

New:
- Larger, configurable figure size per label (--cm-figscale) and dpi (--cm-dpi)
- Control annotation with --cm-annotate-up-to (max labels to draw numbers)
- Optional tiling for very large label sets (--cm-chunk-size)
"""

import os, glob, json, argparse
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, ViTModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Utilities (mirrors train)
# -------------------------
IMG_EXTS = ("*.jpg","*.jpeg","*.png","*.bmp")

def list_images(root: str) -> Dict[str, List[str]]:
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    out = {}
    for c in classes:
        paths = []
        for ext in IMG_EXTS:
            paths += glob.glob(os.path.join(root, c, "**", ext), recursive=True)
        if paths:
            out[c] = paths
    return out

def df_from_folder_plus_csv_for_test(
    root: str,
    map_df: pd.DataFrame,
    species2id: Dict[str, int],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    rows = []
    class_to_row = {r["class_name"]: r for _, r in map_df.iterrows()}
    class_images = list_images(root)

    classes_missing_in_csv = []
    unseen_species = []
    for cls, paths in class_images.items():
        if cls not in class_to_row:
            classes_missing_in_csv.append(cls)
            continue
        if cls not in species2id:
            unseen_species.append(cls)
            continue

        r = class_to_row[cls]
        o = int(r["ordor"]); f = int(r["familia"]); g = int(r["genus"])
        s = species2id[cls]
        for p in paths:
            rows.append({
                "image_path": p,
                "class_name": cls,
                "ordor": o,
                "familia": f,
                "genus": g,
                "species": s
            })

    return pd.DataFrame(rows), classes_missing_in_csv, unseen_species

# -------------------------
# Dataset
# -------------------------
class PollenTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_processor):
        self.df = df.reset_index(drop=True)
        self.image_processor = image_processor
        self.tf = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        img = self.tf(img)
        enc = self.image_processor(images=img, return_tensors="pt")
        x = enc["pixel_values"].squeeze(0)
        y = {
            "ordor":   torch.tensor(int(r["ordor"]),   dtype=torch.long),
            "familia": torch.tensor(int(r["familia"]), dtype=torch.long),
            "genus":   torch.tensor(int(r["genus"]),   dtype=torch.long),
            "species": torch.tensor(int(r["species"]), dtype=torch.long),
        }
        meta = {
            "image_path": r["image_path"],
            "class_name": r["class_name"],
        }
        return x, y, meta

# -------------------------
# Model (must match train)
# -------------------------
class ViTMultiHead(nn.Module):
    def __init__(self, model_name: str, n_ordor: int, n_familia: int, n_genus: int, n_species: int):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(model_name)
        hid = self.backbone.config.hidden_size
        self.head_ordor   = nn.Linear(hid, n_ordor)
        self.head_familia = nn.Linear(hid, n_familia)
        self.head_genus   = nn.Linear(hid, n_genus)
        self.head_species = nn.Linear(hid, n_species)

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        feat = out.last_hidden_state[:, 0, :]
        return (
            self.head_ordor(feat),
            self.head_familia(feat),
            self.head_genus(feat),
            self.head_species(feat),
        )

# -------------------------
# Masked (hierarchy-constrained) decoding
# -------------------------
def masked_decode(
    lo: torch.Tensor, lf: torch.Tensor, lg: torch.Tensor, ls: torch.Tensor,
    familia_parent: torch.Tensor, genus_parent: torch.Tensor, species_parent: torch.Tensor
):
    o = lo.argmax(1)

    B, F = lf.shape
    lf_masked = torch.full_like(lf, -1e9)
    for b in range(B):
        valid_f = torch.where(familia_parent == o[b])[0]
        lf_masked[b, valid_f] = lf[b, valid_f]
    f = lf_masked.argmax(1)

    B, G = lg.shape
    lg_masked = torch.full_like(lg, -1e9)
    for b in range(B):
        valid_g = torch.where(genus_parent == f[b])[0]
        lg_masked[b, valid_g] = lg[b, valid_g]
    g = lg_masked.argmax(1)

    B, S = ls.shape
    ls_masked = torch.full_like(ls, -1e9)
    for b in range(B):
        valid_s = torch.where(species_parent == g[b])[0]
        ls_masked[b, valid_s] = ls[b, valid_s]
    s = ls_masked.argmax(1)

    return o, f, g, s

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def run_eval(model, loader, device, masked: bool,
             familia_parent=None, genus_parent=None, species_parent=None):
    model.eval()

    o_pred, o_true = [], []
    f_pred, f_true = [], []
    g_pred, g_true = [], []
    s_pred, s_true = [], []
    rows = []

    for x, y, meta in loader:
        x = x.to(device)
        y = {k: v.to(device) for k, v in y.items()}
        lo, lf, lg, ls = model(x)

        if masked:
            o_hat, f_hat, g_hat, s_hat = masked_decode(
                lo, lf, lg, ls,
                familia_parent.to(device),
                genus_parent.to(device),
                species_parent.to(device)
            )
        else:
            o_hat = lo.argmax(1)
            f_hat = lf.argmax(1)
            g_hat = lg.argmax(1)
            s_hat = ls.argmax(1)

        o_pred += o_hat.cpu().tolist(); o_true += y["ordor"].cpu().tolist()
        f_pred += f_hat.cpu().tolist(); f_true += y["familia"].cpu().tolist()
        g_pred += g_hat.cpu().tolist(); g_true += y["genus"].cpu().tolist()
        s_pred += s_hat.cpu().tolist(); s_true += y["species"].cpu().tolist()

        for i in range(x.size(0)):
            rows.append({
                "image_path": meta["image_path"][i],
                "class_name": meta["class_name"][i],
                "true_ordor":   int(y["ordor"][i].cpu()),
                "true_familia": int(y["familia"][i].cpu()),
                "true_genus":   int(y["genus"][i].cpu()),
                "true_species": int(y["species"][i].cpu()),
                "pred_ordor":   int(o_hat[i].cpu()),
                "pred_familia": int(f_hat[i].cpu()),
                "pred_genus":   int(g_hat[i].cpu()),
                "pred_species": int(s_hat[i].cpu()),
            })

    # Accuracies
    acc_o = accuracy_score(o_true, o_pred)
    acc_f = accuracy_score(f_true, f_pred)
    acc_g = accuracy_score(g_true, g_pred)
    acc_s = accuracy_score(s_true, s_pred)

    # F1s (macro + weighted)
    f1_o_macro = f1_score(o_true, o_pred, average="macro", zero_division=0)
    f1_f_macro = f1_score(f_true, f_pred, average="macro", zero_division=0)
    f1_g_macro = f1_score(g_true, g_pred, average="macro", zero_division=0)
    f1_s_macro = f1_score(s_true, s_pred, average="macro", zero_division=0)

    f1_o_weighted = f1_score(o_true, o_pred, average="weighted", zero_division=0)
    f1_f_weighted = f1_score(f_true, f_pred, average="weighted", zero_division=0)
    f1_g_weighted = f1_score(g_true, g_pred, average="weighted", zero_division=0)
    f1_s_weighted = f1_score(s_true, s_pred, average="weighted", zero_division=0)

    hier = sum((a==b) and (c==d) and (e==f) and (g==h)
               for a,b,c,d,e,f,g,h in zip(o_pred,o_true,f_pred,f_true,g_pred,g_true,s_pred,s_true)) / len(o_true)

    metrics = {
        "ordor":   {"acc": acc_o, "f1_macro": f1_o_macro, "f1_weighted": f1_o_weighted},
        "familia": {"acc": acc_f, "f1_macro": f1_f_macro, "f1_weighted": f1_f_weighted},
        "genus":   {"acc": acc_g, "f1_macro": f1_g_macro, "f1_weighted": f1_g_weighted},
        "species": {"acc": acc_s, "f1_macro": f1_s_macro, "f1_weighted": f1_s_weighted},
        "hier": hier
    }
    preds_truth = {
        "ordor": (o_pred, o_true),
        "familia": (f_pred, f_true),
        "genus": (g_pred, g_true),
        "species": (s_pred, s_true),
    }
    return metrics, rows, preds_truth

# -------------------------
# Label-name helpers & CMs
# -------------------------
def load_id_name_map(path: str):
    df = pd.read_csv(path)
    ord_map = {}
    fam_map = {}
    gen_map = {}
    if "ordor_ID" in df.columns and "ordor_name" in df.columns:
        for _, r in df[["ordor_ID","ordor_name"]].dropna().drop_duplicates().iterrows():
            ord_map[int(r["ordor_ID"])] = str(r["ordor_name"])
    if "familia_ID" in df.columns and "familia_name" in df.columns:
        for _, r in df[["familia_ID","familia_name"]].dropna().drop_duplicates().iterrows():
            fam_map[int(r["familia_ID"])] = str(r["familia_name"])
    if "genus_ID" in df.columns and "genus_name" in df.columns:
        for _, r in df[["genus_ID","genus_name"]].dropna().drop_duplicates().iterrows():
            gen_map[int(r["genus_ID"])] = str(r["genus_name"])
    return ord_map, fam_map, gen_map

def build_label_lists(preds_truth, species2id, id_name_maps, df_test):
    def uniq_ids(head_key):
        p, t = preds_truth[head_key]
        return sorted(set(p) | set(t))

    ord_ids = uniq_ids("ordor")
    fam_ids = uniq_ids("familia")
    gen_ids = uniq_ids("genus")
    spe_ids = uniq_ids("species")

    ord_map, fam_map, gen_map = id_name_maps
    inv_species = {v:k for k,v in species2id.items()}

    ord_names = [ord_map.get(i, f"ordor:{i}") for i in ord_ids]
    fam_names = [fam_map.get(i, f"familia:{i}") for i in fam_ids]
    gen_names = [gen_map.get(i, f"genus:{i}") for i in gen_ids]
    spe_names = [inv_species.get(i, f"species:{i}") for i in spe_ids]

    label_ids = {"ordor": ord_ids, "familia": fam_ids, "genus": gen_ids, "species": spe_ids}
    label_names = {"ordor": ord_names, "familia": fam_names, "genus": gen_names, "species": spe_names}
    return label_ids, label_names

def _plot_cm(cm, names, title, png_path, figscale_per_label=0.42, dpi=300, annotate=True):
    """
    Plot confusion matrix with YOLO-like styling:
      - Blue colormap ('Blues')
      - White figure and axes backgrounds
    base_font: base font size for all text (tick labels, title, annotations)
    """
    base_font = 18
    h_in = max(6, min(36, figscale_per_label * len(names)))
    w_in = max(6, min(36, figscale_per_label * len(names)))
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=base_font + 4)
    ax.set_xlabel("Predicted", fontsize=base_font)
    ax.set_ylabel("True", fontsize=base_font)

    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=base_font)
    ax.set_yticklabels(names, fontsize=base_font)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_facecolor('white')
    cbar.ax.tick_params(labelsize=base_font)

    if annotate:
        is_float = np.issubdtype(cm.dtype, np.floating)
        font_size = base_font - 2 if len(names) <= 30 else max(5, base_font - int(0.08 * len(names)))
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                if val > 0:
                    txt = f"{val:.2f}" if is_float else str(int(val))
                    ax.text(j, i, txt, ha="center", va="center", fontsize=font_size, color="black")

    fig.tight_layout()
    plt.savefig(png_path, dpi=dpi)
    plt.close(fig)

def _chunk_indices(n, chunk_size):
    idx = list(range(n))
    return [idx[i:i+chunk_size] for i in range(0, n, chunk_size)]

def _save_cm_block(cm, names, head_name, suffix, out_dir, figscale, dpi, annotate):
    title = f"{head_name.capitalize()} Confusion Matrix{suffix}"
    fname = f"{head_name}_confmat{suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    _plot_cm(cm, names, title, os.path.join(out_dir, fname),
             figscale_per_label=figscale, dpi=dpi, annotate=annotate)

def save_confmats_and_perclass(head_name, y_true, y_pred, label_ids, label_names, out_dir,
                               figscale=0.42, dpi=300, annotate_up_to=30, chunk_size=None):
    """
    Saves:
      - Raw CM: PNG + CSV (tiled if chunk_size is set)
      - Row-normalized CM: PNG + CSV (tiled if chunk_size is set)
      - Per-class accuracy & F1 CSV
    figscale: inches per label; increase to get more space per cell
    dpi: output PNG resolution
    annotate_up_to: if total labels <= this, draw numbers inside cells
    chunk_size: if set (e.g., 50), tile the matrix across both axes into blocks
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = label_ids[head_name]
    names = label_names[head_name]

    idx_map = {lab:i for i, lab in enumerate(labels)}
    y_true_idx = [idx_map[i] for i in y_true]
    y_pred_idx = [idx_map[i] for i in y_pred]

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(labels))))
    raw_csv = os.path.join(out_dir, f"{head_name}_confmat.csv")
    pd.DataFrame(cm, index=names, columns=names).to_csv(raw_csv, encoding="utf-8")

    # Row-normalized
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, where=row_sums!=0)
    norm_csv = os.path.join(out_dir, f"{head_name}_confmat_norm.csv")
    pd.DataFrame(cm_norm, index=names, columns=names).to_csv(norm_csv, encoding="utf-8")

    # Per-class metrics
    per_class_acc = np.nan_to_num(np.diag(cm_norm))
    report = classification_report(y_true_idx, y_pred_idx,
                                   labels=list(range(len(labels))),
                                   output_dict=True, zero_division=0)
    per_class_rows = []
    for i, name in enumerate(names):
        f1 = report.get(str(i), {}).get("f1-score", 0.0)
        support = int(report.get(str(i), {}).get("support", 0))
        per_class_rows.append({"class": name, "accuracy": per_class_acc[i], "f1": f1, "support": support})
    pd.DataFrame(per_class_rows).to_csv(os.path.join(out_dir, f"{head_name}_per_class_acc_f1.csv"),
                                        index=False, encoding="utf-8")

    # Decide annotation
    annotate = len(names) <= annotate_up_to

    # Plot (tiling if requested)
    if chunk_size and len(names) > chunk_size:
        rows_chunks = _chunk_indices(len(names), chunk_size)
        cols_chunks = _chunk_indices(len(names), chunk_size)
        for ri, r_idx in enumerate(rows_chunks):
            for ci, c_idx in enumerate(cols_chunks):
                sub_cm = cm[np.ix_(r_idx, c_idx)]
                sub_names_r = [names[i] for i in r_idx]
                sub_names_c = [names[i] for i in c_idx]
                # Raw block
                _plot_cm(sub_cm, sub_names_c,  # x labels = columns
                         f"{head_name.capitalize()} Confusion Matrix (raw) [rows {ri+1}/{len(rows_chunks)}, cols {ci+1}/{len(cols_chunks)}]",
                         os.path.join(out_dir, f"{head_name}_confmat_raw_r{ri+1}_c{ci+1}.png"),
                         figscale_per_label=figscale, dpi=dpi, annotate=annotate)
                # Norm block
                with np.errstate(divide='ignore', invalid='ignore'):
                    rs = sub_cm.sum(axis=1, keepdims=True)
                    sub_norm = np.divide(sub_cm, rs, where=rs!=0)
                _plot_cm(sub_norm, sub_names_c,
                         f"{head_name.capitalize()} Confusion Matrix (row-normalized) [rows {ri+1}/{len(rows_chunks)}, cols {ci+1}/{len(cols_chunks)}]",
                         os.path.join(out_dir, f"{head_name}_confmat_norm_r{ri+1}_c{ci+1}.png"),
                         figscale_per_label=figscale, dpi=dpi, annotate=annotate)
        # Also save “full” overview without annotations (for quick glance)
        _plot_cm(cm, names,
                 f"{head_name.capitalize()} Confusion Matrix (overview, no annotations)",
                 os.path.join(out_dir, f"{head_name}_confmat_overview.png"),
                 figscale_per_label=figscale, dpi=dpi, annotate=False)
        _plot_cm(cm_norm, names,
                 f"{head_name.capitalize()} Confusion Matrix (row-normalized overview, no annotations)",
                 os.path.join(out_dir, f"{head_name}_confmat_norm_overview.png"),
                 figscale_per_label=figscale, dpi=dpi, annotate=False)
    else:
        _plot_cm(cm, names, f"{head_name.capitalize()} Confusion Matrix",
                 os.path.join(out_dir, f"{head_name}_confmat.png"),
                 figscale_per_label=figscale, dpi=dpi, annotate=annotate)
        _plot_cm(cm_norm, names, f"{head_name.capitalize()} Confusion Matrix (Row-normalized)",
                 os.path.join(out_dir, f"{head_name}_confmat_norm.png"),
                 figscale_per_label=figscale, dpi=dpi, annotate=annotate)

# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, help="Path to outputs_vit_4heads_min/best_model.pt", default="outputs_vit_4heads_min/best_model.pt")
    # p.add_argument("--root_test", type=str, default="data_crop_46/test")
    p.add_argument("--root_test", type=str, default="/home/dubu/manh/lab/dongvan-vit/dataset-dongvan-train/test")
    # p.add_argument("--root_test", type=str, default="/home/dubu/datasets/lab/data_crop_200_46/test")
    p.add_argument("--mapping_csv", type=str, default="pollen_dong_van.csv")
    p.add_argument("--id_name_csv", type=str, default="hier_name.csv", help="Optional ID↔name mapping CSV")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--masked", action="store_true", help="Enable hierarchy-constrained decoding")
    p.add_argument("--out_csv", type=str, default="test_predictions.csv")
    p.add_argument("--confmat_dir", type=str, default="cm_out_test", help="If set, save per-head confusion matrices here")

    # NEW: CM sizing/tiling knobs
    p.add_argument("--cm-figscale", type=float, default=0.80, help="Inches per label for CM figure size (both axes). Increase for more space.")
    p.add_argument("--cm-dpi", type=int, default=100, help="CM image DPI")
    p.add_argument("--cm-annotate-up-to", type=int, default=60, help="Annotate cells only if #labels <= this")
    p.add_argument("--cm-chunk-size", type=int, default=None, help="Tile CM into blocks of this many labels on each axis (e.g., 50)")

    return p.parse_args()

def main():
    args = parse_args()
    assert os.path.isdir(args.root_test), f"Test folder not found: {args.root_test}"
    assert os.path.isfile(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    assert os.path.isfile(args.mapping_csv), f"mapping_csv not found: {args.mapping_csv}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt["model_name"]
    species2id = ckpt["species2id"]
    ncls = ckpt["n_classes"]
    familia_parent = torch.tensor(ckpt["familia_parent"], dtype=torch.long)
    genus_parent   = torch.tensor(ckpt["genus_parent"],   dtype=torch.long)
    species_parent = torch.tensor(ckpt["species_parent"], dtype=torch.long)

    # CSV (ensure columns)
    map_df = pd.read_csv(args.mapping_csv)
    if "order" in map_df.columns and "ordor" not in map_df.columns:
        map_df = map_df.rename(columns={"order":"ordor"})
    assert set(["class_name","ordor","familia","genus"]).issubset(map_df.columns), \
        "mapping.csv must have columns: class_name, ordor, familia, genus"

    # Build test dataframe
    df_test, miss_csv, unseen = df_from_folder_plus_csv_for_test(args.root_test, map_df, species2id)
    if miss_csv:
        print(f"[WARN] {len(miss_csv)} classes in test not found in mapping.csv (skipped): {sorted(miss_csv)[:10]}{' ...' if len(miss_csv)>10 else ''}")
    if unseen:
        print(f"[WARN] {len(unseen)} classes in test are unseen species (not in ckpt species2id) (skipped): {sorted(unseen)[:10]}{' ...' if len(unseen)>10 else ''}")
    if df_test.empty:
        raise SystemExit("No valid test images after filtering. Check mapping.csv and species coverage.")

    # Processor & loader
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    test_ds = PollenTestDataset(df_test, processor)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    # Model
    model = ViTMultiHead(model_name, ncls["ordor"], ncls["familia"], ncls["genus"], ncls["species"])
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(device)

    # Run eval
    metrics, rows, preds_truth = run_eval(
        model, test_dl, device, masked=args.masked,
        familia_parent=familia_parent, genus_parent=genus_parent, species_parent=species_parent
    )

    # Print metrics
    print(f"[TEST] masked={args.masked} | "
          f"ordor: acc={metrics['ordor']['acc']:.3f}, f1_macro={metrics['ordor']['f1_macro']:.3f}, f1_weighted={metrics['ordor']['f1_weighted']:.3f} | "
          f"familia: acc={metrics['familia']['acc']:.3f}, f1_macro={metrics['familia']['f1_macro']:.3f}, f1_weighted={metrics['familia']['f1_weighted']:.3f} | "
          f"genus: acc={metrics['genus']['acc']:.3f}, f1_macro={metrics['genus']['f1_macro']:.3f}, f1_weighted={metrics['genus']['f1_weighted']:.3f} | "
          f"species: acc={metrics['species']['acc']:.3f}, f1_macro={metrics['species']['f1_macro']:.3f}, f1_weighted={metrics['species']['f1_weighted']:.3f} | "
          f"hier={metrics['hier']:.3f}"
    )

    # Save predictions table
    pd.DataFrame(rows).to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Saved detailed predictions to: {args.out_csv}")

    # Save metrics summary CSV
    metrics_rows = []
    for head in ["ordor","familia","genus","species"]:
        metrics_rows.append({
            "head": head,
            "accuracy": metrics[head]["acc"],
            "f1_macro": metrics[head]["f1_macro"],
            "f1_weighted": metrics[head]["f1_weighted"],
        })
    metrics_rows.append({"head":"hierarchical", "accuracy": metrics["hier"], "f1_macro": None, "f1_weighted": None})
    pd.DataFrame(metrics_rows).to_csv("metrics_summary.csv", index=False, encoding="utf-8")
    print("Saved metrics_summary.csv")

    # Confusion matrices (optional)
    if args.confmat_dir:
        if args.id_name_csv and os.path.isfile(args.id_name_csv):
            ord_map, fam_map, gen_map = load_id_name_map(args.id_name_csv)
        else:
            ord_map, fam_map, gen_map = {}, {}, {}

        label_ids, label_names = build_label_lists(
            preds_truth=preds_truth,
            species2id=species2id,
            id_name_maps=(ord_map, fam_map, gen_map),
            df_test=df_test,
        )

        for head in ["ordor", "familia", "genus", "species"]:
            y_pred, y_true = preds_truth[head]
            save_confmats_and_perclass(
                head, y_true, y_pred, label_ids, label_names, args.confmat_dir,
                figscale=args.cm_figscale, dpi=args.cm_dpi,
                annotate_up_to=args.cm_annotate_up_to,
                chunk_size=args.cm_chunk_size
            )

if __name__ == "__main__":
    main()
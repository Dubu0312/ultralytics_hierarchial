#!/usr/bin/env python3
"""Compute R[i][j] HierAcc matrix + FM + BWT for all method variants.

Reads checkpoints from:
    pollen_incremental/sessions/                — V1 (replay + KD, β=0.1)
    pollen_incremental/sessions/naive/          — naive finetune (no replay, no KD)
    pollen_incremental/sessions/simplecil/      — train-free, frozen prototype
    pollen_incremental/sessions/cache_refit/    — joint-training upper bound

Saves:
    pollen_incremental/results/all_baselines.json    — full R + metrics per method
    pollen_incremental/results/all_baselines.csv     — flat table for spreadsheets
    stdout: human-readable comparison table

Usage:
    python scripts/eval_all_baselines.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pollen_incremental.checkpoint import IncrementalCheckpoint
from pollen_incremental.dataset import HierarchicalImageDataset, build_dataframe
from pollen_incremental.eval_incremental import evaluate
from pollen_incremental.masked_ops import to_parent_tensors
from pollen_incremental.model import YOLOHierarchicalClassifier
from pollen_incremental.stream import load_stream


DEFAULT_MANIFEST = Path("/home/dubu/manh/lab/ultralytics/data/stream_split.json")
DEFAULT_RESULTS = Path("/home/dubu/manh/lab/ultralytics/pollen_incremental/results")
DEFAULT_SESSIONS_ROOT = Path("/home/dubu/manh/lab/ultralytics/pollen_incremental/sessions")

METHODS = {
    "V1":          "",            # sessions/ directly
    "naive":       "naive",
    "SimpleCIL":   "simplecil",
    "CacheRefit":  "cache_refit",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--sessions-root", type=Path, default=DEFAULT_SESSIONS_ROOT)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--batch-size", type=int, default=16)
    return ap.parse_args()


def build_model(ckpt: IncrementalCheckpoint, device: torch.device) -> YOLOHierarchicalClassifier:
    m = YOLOHierarchicalClassifier(
        ckpt.base_model,
        n_ordor=ckpt.tax_state.n_ordor, n_familia=ckpt.tax_state.n_familia,
        n_genus=ckpt.tax_state.n_genus, n_species=ckpt.tax_state.n_species,
    )
    m.load_state_dict(ckpt.model_state, strict=True)
    return m.to(device)


def compute_R_for_method(
    method_name: str,
    sessions_dir: Path,
    stream,
    test_root: Path,
    device: torch.device,
    batch_size: int,
) -> dict:
    """Returns dict with R[i][j], FM, BWT, cumulative HierAcc, per-session test sizes."""
    if not sessions_dir.is_dir():
        print(f"  [SKIP] {method_name}: {sessions_dir} not found")
        return None

    # Load all 6 checkpoints
    ckpts = {}
    for sid in range(len(stream)):
        name = "session_0_base.pt" if sid == 0 else f"session_{sid}.pt"
        p = sessions_dir / name
        if not p.is_file():
            print(f"  [SKIP] {method_name}: missing {p}")
            return None
        ckpts[sid] = IncrementalCheckpoint.load(p)

    # R[i][j] = HierAcc of model-i on test split containing ONLY species
    # added at session j.
    R = {}
    n_test = {}
    for i in range(len(stream)):
        m = build_model(ckpts[i], device)
        fp, gp, sp = to_parent_tensors(ckpts[i].tax_state, device=device)
        map_df_i = pd.read_csv(f"data/mapping_session_{i}.csv")
        R[i] = {}
        for j in range(i + 1):
            species_j = set(stream.sessions[j].new_species)
            df_j = build_dataframe(test_root, map_df_i, ckpts[i].tax_state.species2id,
                                   species_filter=species_j)
            if len(df_j) == 0:
                continue
            loader = DataLoader(
                HierarchicalImageDataset(df_j, augment=False),
                batch_size=batch_size, shuffle=False, num_workers=2,
            )
            r = evaluate(m, loader, device, fp, gp, sp, decode="masked")
            R[i][j] = r.hier_acc
            if i == len(stream) - 1:
                n_test[j] = len(df_j)
        del m
        torch.cuda.empty_cache()

    # Metrics
    last = len(stream) - 1
    fm_terms = []
    for j in range(last):
        if not all(j in R[i] for i in range(j, last + 1)):
            continue
        best = max(R[i][j] for i in range(j, last + 1))
        fm_terms.append(best - R[last][j])
    fm = sum(fm_terms) / max(1, len(fm_terms))
    bwt_terms = [R[last][j] - R[j][j] for j in range(last) if j in R[j] and j in R[last]]
    bwt = sum(bwt_terms) / max(1, len(bwt_terms))

    cum_correct = 0.0; cum_total = 0
    for j in range(last + 1):
        if j not in n_test or j not in R[last]:
            continue
        cum_correct += R[last][j] * n_test[j]
        cum_total += n_test[j]
    cum_acc = cum_correct / max(1, cum_total)

    return {
        "method": method_name,
        "R": R,
        "FM": fm,
        "BWT": bwt,
        "cumulative_HierAcc_final": cum_acc,
        "n_test_per_session": n_test,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = load_stream(args.manifest)
    test_root = Path("/home/dubu/manh/dongvan-yolo/streams/session_5/test")  # all 22 species

    args.results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for method_name, subdir in METHODS.items():
        sessions_dir = args.sessions_root / subdir if subdir else args.sessions_root
        print(f"\n=== Evaluating {method_name} ({sessions_dir.name or 'root'}) ===")
        res = compute_R_for_method(method_name, sessions_dir, stream, test_root,
                                   device, args.batch_size)
        if res is None:
            continue
        all_results[method_name] = res
        print(f"  Final cumulative HierAcc: {res['cumulative_HierAcc_final']:.3f}")
        print(f"  FM: {res['FM']:.4f}  BWT: {res['BWT']:+.4f}")

    # Save JSON
    json_path = args.results_dir / "all_baselines.json"
    # JSON serialization: convert int keys to str
    def _ser(o):
        if isinstance(o, dict):
            return {str(k): _ser(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_ser(x) for x in o]
        return o
    with open(json_path, "w") as f:
        json.dump(_ser(all_results), f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Flat CSV: method, FM, BWT, cum_acc, plus R[i][j] columns
    rows = []
    for name, res in all_results.items():
        row = {
            "method": name,
            "cum_HierAcc_final": res["cumulative_HierAcc_final"],
            "FM": res["FM"],
            "BWT": res["BWT"],
        }
        for i in range(len(stream)):
            for j in range(i + 1):
                row[f"R[{i}][{j}]"] = res["R"][i].get(j)
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = args.results_dir / "all_baselines.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Pretty-print comparison
    print("\n" + "=" * 80)
    print(f"{'Method':<14}{'HierAcc(cum)':<14}{'FM (↓)':<10}{'BWT':<10}")
    print("-" * 80)
    for name, res in all_results.items():
        print(f"{name:<14}{res['cumulative_HierAcc_final']:<14.3f}"
              f"{res['FM']:<10.4f}{res['BWT']:<+10.4f}")
    print("=" * 80)

    # Per-method R matrix
    for name, res in all_results.items():
        print(f"\n--- {name} R[i][j] matrix ---")
        header = "      " + "".join(f"  S{j}    " for j in range(len(stream)))
        print(header)
        for i in range(len(stream)):
            row = f"  S{i}: "
            for j in range(len(stream)):
                v = res["R"][i].get(j)
                row += f"  {v:.3f}" if v is not None else "    -  "
            print(row)


if __name__ == "__main__":
    main()

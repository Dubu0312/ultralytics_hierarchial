#!/usr/bin/env python3
"""Convert the baseline yolo11x-trained-masked checkpoint into the incremental
session-0 format.

The base checkpoint (produced by train_hierarchical_masked.py) stores:
    model_state, base_model, species2id, *_parent, lambdas, n_classes,
    masked_training

The incremental format wraps these as IncrementalCheckpoint with:
    - TaxonomyState (built from base species2id + parent arrays)
    - session = 0
    - ExemplarMemory (empty for now — populated AFTER S₀ in
      `scripts/run_session.py` when we have backbone features ready)
    - incremental_cfg (snapshot of base hyperparams for reproducibility)

This is a ONE-TIME conversion. The base trainer doesn't need to know about
the new format. After running this, S₀_base.pt is the "starting position"
for all incremental sessions S₁, S₂, …

Run from repo root:
    python scripts/build_base_session.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from pollen_incremental.checkpoint import IncrementalCheckpoint
from pollen_incremental.exemplar_memory import ExemplarMemory
from pollen_incremental.taxonomy import TaxonomyState

DEFAULT_BASE_CKPT = Path("/home/dubu/manh/lab/ultralytics/ultralytics/models/yolo/classify/"
                         "yolo11x-trained-masked/best_model.pt")
DEFAULT_MAPPING_CSV = Path("/home/dubu/manh/dongvan-yolo/pollen_dong_van.csv")
DEFAULT_OUT_PATH = Path("/home/dubu/manh/lab/ultralytics/pollen_incremental/sessions/session_0_base.pt")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE_CKPT,
                    help="Path to the base trainer's checkpoint (best_model.pt).")
    ap.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV,
                    help="The same pollen_dong_van.csv used during base training.")
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH,
                    help="Where to write the converted session-0 checkpoint.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not args.base_ckpt.is_file():
        raise SystemExit(f"Base checkpoint not found: {args.base_ckpt}")
    if not args.mapping_csv.is_file():
        raise SystemExit(f"Mapping CSV not found: {args.mapping_csv}")

    print(f"Loading base checkpoint: {args.base_ckpt}")
    base = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    for required in ("model_state", "base_model", "species2id",
                     "familia_parent", "genus_parent", "species_parent",
                     "lambdas", "n_classes"):
        if required not in base:
            raise SystemExit(f"Base checkpoint missing required key: {required!r}")

    print(f"  base_model: {base['base_model']}")
    print(f"  n_classes:  {base['n_classes']}")
    print(f"  #species:   {len(base['species2id'])}")

    # Re-build the TaxonomyState. The base checkpoint already stores parent
    # arrays, so we don't need to recompute from the CSV — but we read the
    # CSV anyway to populate ordor/familia/genus sets (the base format only
    # has parent arrays, which is enough for masked_decode but ambiguous for
    # extend_state which wants explicit "which IDs are taken").
    print(f"Reading mapping CSV: {args.mapping_csv}")
    map_df = pd.read_csv(args.mapping_csv)
    if "order" in map_df.columns and "ordor" not in map_df.columns:
        map_df = map_df.rename(columns={"order": "ordor"})

    # Build sets from BASE-known species only (defensive — in case CSV has extras)
    base_species = set(base["species2id"].keys())
    sub_df = map_df[map_df["class_name"].isin(base_species)]
    ordor_set = {int(x) for x in sub_df["ordor"].unique()}
    familia_set = {int(x) for x in sub_df["familia"].unique()}
    genus_set = {int(x) for x in sub_df["genus"].unique()}

    tax_state = TaxonomyState(
        species2id=dict(base["species2id"]),
        ordor_set=ordor_set,
        familia_set=familia_set,
        genus_set=genus_set,
        familia_parent=list(base["familia_parent"]),
        genus_parent=list(base["genus_parent"]),
        species_parent=list(base["species_parent"]),
    )

    # Sanity: TaxonomyState sizes must match base n_classes
    computed = tax_state.n_classes()
    expected = base["n_classes"]
    for level in ("ordor", "familia", "genus", "species"):
        if computed[level] != expected[level]:
            raise SystemExit(
                f"Size mismatch at level {level}: "
                f"computed {computed[level]} vs base {expected[level]}"
            )
    print(f"  TaxonomyState built — sizes match: {computed}")

    # Empty memory for now — V1's run_session.py will populate it after S₀
    # using the frozen backbone to compute prototypes for all 22 base species.
    memory = ExemplarMemory(budget_per_class=20, mode="per_class")

    ckpt = IncrementalCheckpoint(
        session=0,
        base_model=base["base_model"],
        model_state=base["model_state"],
        tax_state=tax_state,
        exemplar_memory=memory,
        lambdas=dict(base["lambdas"]),
        incremental_cfg={
            "source": "converted from train_hierarchical_masked.py best_model.pt",
            "base_ckpt_path": str(args.base_ckpt),
            "masked_training": bool(base.get("masked_training", True)),
            "memory_populated": False,
            "memory_note": ("Memory is empty — run scripts/populate_base_memory.py "
                            "or let train_incremental on S1 do it inline."),
        },
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt.save(args.out_path)
    print(f"\n✓ Wrote session-0 checkpoint: {args.out_path}")
    print(f"  Size: {args.out_path.stat().st_size / (1024 ** 2):.1f} MB")
    print()
    print("Next steps:")
    print("  1. Populate base exemplar memory (run later with the frozen backbone).")
    print("  2. Run scripts/run_session.py to train S₁.")


if __name__ == "__main__":
    main()

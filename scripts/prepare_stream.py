#!/usr/bin/env python3
"""Create the incremental learning stream by partitioning the 22 Dong Van species.

Produces, for sessions S₀ (base) through S₅:
    - data/stream_split.json              — manifest of which species belong to each session
    - data/mapping_session_{k}.csv        — cumulative mapping CSV (base + incremental thru Sₖ)
    - data/streams/session_{k}/{train,val,test}/<class_name>/  — symlinks to original images

Stream design (deliberate, see HIERARCHICAL_INCREMENTAL.md §5.4):

    S₀ base   : 17 species          — bootstrap backbone + masked-CE head
    S₁  +1    : chro_2              — NEW LEAF under existing genus 5
    S₂  +1    : abel                — NEW BRANCH (pulls in order 2, family 2, genus 3)
    S₃  +1    : ruel_2              — NEW LEAF under existing genus 12
    S₄  +1    : bras                — NEW BRANCH (pulls in order 4, family 8, genus 13)
    S₅  +1    : ipom_1              — NEW LEAF under existing genus 14

After S₅ the cumulative set matches the full base dataset (22 species, 6/10/16 orders/
families/genera). The choice mixes "new leaf" and "new branch" cases to stress-test
both head-expansion paths.

Idempotent: re-running overwrites JSON + CSVs; symlinks are re-created if missing.

Usage:
    python scripts/prepare_stream.py                                  # default paths
    python scripts/prepare_stream.py --src-root /custom/dataset \\
                                     --src-csv  /custom/mapping.csv  \\
                                     --out-dir  /custom/output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_SRC_ROOT = Path("/home/dubu/manh/dongvan-yolo/dataset-dongvan-train")
DEFAULT_SRC_CSV = Path("/home/dubu/manh/dongvan-yolo/pollen_dong_van.csv")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_STREAM_ROOT = Path("/home/dubu/manh/dongvan-yolo/streams")

SPLITS = ("train", "val", "test")

# Stream definition: each session adds these species (cumulative).
# Tag each new species with the case it represents (for analysis/logging).
STREAM_SCHEDULE: list[dict] = [
    {
        "session": 0,
        "label": "base",
        # 17 species (everything except the 5 incremental ones).
        "new_species": [
            "ager", "bide", "chro", "chro_1", "cler", "cler_1", "clin", "ipom",
            "leuc", "ruel", "ruel_1", "sige", "sige_1", "teco", "tith", "tore", "trid",
        ],
        "case": "base",
    },
    {"session": 1, "label": "S1_chro_2", "new_species": ["chro_2"],  "case": "new_leaf"},
    {"session": 2, "label": "S2_abel",   "new_species": ["abel"],    "case": "new_branch"},  # +order2 +family2 +genus3
    {"session": 3, "label": "S3_ruel_2", "new_species": ["ruel_2"],  "case": "new_leaf"},
    {"session": 4, "label": "S4_bras",   "new_species": ["bras"],    "case": "new_branch"},  # +order4 +family8 +genus13
    {"session": 5, "label": "S5_ipom_1", "new_species": ["ipom_1"],  "case": "new_leaf"},
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT,
                    help="Root of the original dataset (must contain train/, val/, test/)")
    ap.add_argument("--src-csv", type=Path, default=DEFAULT_SRC_CSV,
                    help="Original mapping CSV (class_name, ordor, familia, genus)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Where to write stream_split.json + mapping_session_*.csv")
    ap.add_argument("--stream-root", type=Path, default=DEFAULT_STREAM_ROOT,
                    help="Where to create the per-session symlink trees")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan + counts but do not create any files")
    return ap.parse_args()


def validate_schedule_against_csv(schedule: list[dict], df: pd.DataFrame) -> None:
    """Ensure every species in the schedule exists in the source CSV, and that
    every CSV species ends up in exactly one session."""
    all_csv_species = set(df["class_name"])
    all_scheduled = []
    for entry in schedule:
        all_scheduled.extend(entry["new_species"])

    scheduled_set = set(all_scheduled)

    # No duplicates within the schedule
    if len(all_scheduled) != len(scheduled_set):
        from collections import Counter
        dups = [n for n, c in Counter(all_scheduled).items() if c > 1]
        raise ValueError(f"Duplicate species in schedule: {dups}")

    # Every scheduled species must exist in CSV
    missing_in_csv = scheduled_set - all_csv_species
    if missing_in_csv:
        raise ValueError(f"Scheduled species not in CSV: {sorted(missing_in_csv)}")

    # Every CSV species must be scheduled
    missing_in_schedule = all_csv_species - scheduled_set
    if missing_in_schedule:
        raise ValueError(f"CSV species not scheduled: {sorted(missing_in_schedule)}")


def count_images(root: Path, species_set: set[str]) -> dict[str, dict[str, int]]:
    """For each split, count images per species. Returns {split: {species: count}}."""
    out: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in species_set:
                continue
            n = sum(1 for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in
                    (".jpg", ".jpeg", ".png", ".bmp"))
            out[split][cls_dir.name] = n
    return out


def build_session_state(schedule: list[dict]) -> list[dict]:
    """Compute, for each session, the cumulative set of species seen so far and
    the per-session new IDs that would be assigned by extend_state().

    This is metadata only — it does NOT actually call extend_state(), but it
    matches its append-only contract so the JSON is a self-contained manifest.
    """
    cumulative_species: list[str] = []  # ordered list (insertion order = ID order)
    out: list[dict] = []
    for entry in schedule:
        before = list(cumulative_species)
        new = entry["new_species"]
        cumulative_species.extend(new)
        # ID = position in cumulative_species after this session
        new_species_ids = {name: cumulative_species.index(name) for name in new}
        out.append({
            "session": entry["session"],
            "label": entry["label"],
            "case": entry["case"],
            "new_species": new,
            "new_species_ids": new_species_ids,
            "cumulative_species": list(cumulative_species),
            "cumulative_count": len(cumulative_species),
            "delta_count": len(new),
            "species_before": before,
        })
    return out


def write_cumulative_csvs(
    schedule_state: list[dict],
    src_df: pd.DataFrame,
    out_dir: Path,
) -> dict[int, Path]:
    """Write per-session cumulative mapping CSVs.

    Each CSV at session k contains rows for every species visible up to and
    including Sₖ — this is what gets fed to extend_state() during training.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    name_to_row = {r["class_name"]: r for _, r in src_df.iterrows()}

    for entry in schedule_state:
        rows = [name_to_row[n] for n in entry["cumulative_species"]]
        df = pd.DataFrame(rows)[["class_name", "ordor", "familia", "genus"]]
        path = out_dir / f"mapping_session_{entry['session']}.csv"
        df.to_csv(path, index=False)
        paths[entry["session"]] = path
    return paths


def create_session_symlinks(
    schedule_state: list[dict],
    src_root: Path,
    stream_root: Path,
) -> dict[int, Path]:
    """For each session, create stream_root/session_{k}/{train,val,test}/<class>/
    where <class> is a symlink to the original dataset's class directory.

    Each session's directory contains only the species CUMULATIVELY visible
    at that point — this is what the trainer/evaluator iterates over.

    Idempotent: existing symlinks are skipped; broken symlinks are recreated.
    """
    stream_root.mkdir(parents=True, exist_ok=True)
    out: dict[int, Path] = {}
    for entry in schedule_state:
        sess_dir = stream_root / f"session_{entry['session']}"
        for split in SPLITS:
            sd = sess_dir / split
            sd.mkdir(parents=True, exist_ok=True)
            for cls in entry["cumulative_species"]:
                src = src_root / split / cls
                if not src.exists():
                    continue
                dst = sd / cls
                # Replace if dst exists but isn't pointing at src
                if dst.is_symlink():
                    if dst.resolve() == src.resolve():
                        continue
                    dst.unlink()
                elif dst.exists():
                    # A real directory exists — refuse to clobber
                    raise FileExistsError(
                        f"{dst} exists and is not a symlink; refusing to overwrite"
                    )
                dst.symlink_to(src)
        out[entry["session"]] = sess_dir
    return out


def main() -> None:
    args = parse_args()

    print(f"Source dataset: {args.src_root}")
    print(f"Source CSV:     {args.src_csv}")
    print(f"Output dir:     {args.out_dir}")
    print(f"Stream root:    {args.stream_root}")
    print()

    if not args.src_csv.is_file():
        raise SystemExit(f"Mapping CSV not found: {args.src_csv}")
    if not args.src_root.is_dir():
        raise SystemExit(f"Source dataset not found: {args.src_root}")

    src_df = pd.read_csv(args.src_csv)
    if "order" in src_df.columns and "ordor" not in src_df.columns:
        src_df = src_df.rename(columns={"order": "ordor"})

    validate_schedule_against_csv(STREAM_SCHEDULE, src_df)
    schedule_state = build_session_state(STREAM_SCHEDULE)

    # Report image counts per session for sanity
    print("=== Per-session plan ===")
    print(f"{'Session':<10}{'Label':<14}{'Case':<14}{'+Species':<22}{'Cum.':<6}")
    for entry in schedule_state:
        new_str = ",".join(entry["new_species"])[:20] or "—"
        print(f"S{entry['session']:<9}{entry['label']:<14}{entry['case']:<14}"
              f"{new_str:<22}{entry['cumulative_count']:<6}")
    print()

    # Image counts (cumulative train images only, for quick sanity)
    print("=== Train image counts per session (cumulative) ===")
    for entry in schedule_state:
        counts = count_images(args.src_root, set(entry["cumulative_species"]))
        n_train = sum(counts["train"].values())
        n_val = sum(counts["val"].values())
        n_test = sum(counts["test"].values())
        print(f"S{entry['session']}: train={n_train:>5}  val={n_val:>4}  test={n_test:>4}")
    print()

    if args.dry_run:
        print("[dry-run] No files written.")
        return

    # 1) Write the master manifest JSON
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "src_root": str(args.src_root),
        "src_csv": str(args.src_csv),
        "stream_root": str(args.stream_root),
        "sessions": schedule_state,
    }
    manifest_path = args.out_dir / "stream_split.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote manifest: {manifest_path}")

    # 2) Cumulative mapping CSVs
    csv_paths = write_cumulative_csvs(schedule_state, src_df, args.out_dir)
    for sid, p in csv_paths.items():
        print(f"Wrote mapping CSV for S{sid}: {p}")

    # 3) Per-session symlink trees
    print()
    sess_dirs = create_session_symlinks(schedule_state, args.src_root, args.stream_root)
    for sid, p in sess_dirs.items():
        # Count actual symlinks per session as verification
        n_classes_train = sum(
            1 for _ in (p / "train").iterdir() if _.is_symlink() or _.is_dir()
        ) if (p / "train").exists() else 0
        print(f"Session S{sid} symlinks → {p} ({n_classes_train} class dirs in train)")

    print("\nDone.")


if __name__ == "__main__":
    main()

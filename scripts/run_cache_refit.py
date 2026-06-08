#!/usr/bin/env python3
"""Run cache-refit upper bound across the whole stream.

Each session: cache backbone features for ALL cumulative training images,
then re-init heads and train from scratch on the cached features. Not CL —
this is the joint-training upper bound.

Usage:
    python scripts/run_cache_refit.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pollen_incremental.cache_refit import run_cache_refit_session
from pollen_incremental.stream import load_stream


DEFAULT_MANIFEST = Path("/home/dubu/manh/lab/ultralytics/data/stream_split.json")
DEFAULT_SOURCE_S0 = Path("/home/dubu/manh/lab/ultralytics/pollen_incremental/sessions/session_0_base.pt")
DEFAULT_OUTDIR = Path("/home/dubu/manh/lab/ultralytics/pollen_incremental/sessions/cache_refit")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--source-s0", type=Path, default=DEFAULT_SOURCE_S0)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch-size-cache", type=int, default=16)
    ap.add_argument("--batch-size-refit", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50,
                    help="Refit epochs per session (head-only, on cached features — fast).")
    ap.add_argument("--lr", type=float, default=1e-3)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    stream = load_stream(args.manifest)

    dst_s0 = args.outdir / "session_0_base.pt"
    if not dst_s0.is_file():
        shutil.copy2(args.source_s0, dst_s0)
        print(f"Copied S0: {args.source_s0} → {dst_s0}\n")

    prev = dst_s0
    for sid in range(1, len(stream)):
        session = stream[sid]
        out = args.outdir / f"session_{sid}.pt"
        print(f"\n=== CacheRefit S{sid} ({session.label}) ===")
        run_cache_refit_session(
            prev_ckpt_path=prev,
            mapping_csv_new=session.mapping_csv,
            session_data_dir=session.data_dir,
            out_ckpt_path=out,
            imgsz=args.imgsz,
            batch_size_cache=args.batch_size_cache,
            batch_size_refit=args.batch_size_refit,
            epochs=args.epochs,
            lr=args.lr,
        )
        prev = out

    print(f"\n✓ All cache-refit sessions complete: {args.outdir}")


if __name__ == "__main__":
    main()

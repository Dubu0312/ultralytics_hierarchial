#!/usr/bin/env python3
"""Run SimpleCIL (train-free, frozen-prototype) across the whole stream.

Unlike run_session.py which does one session at a time, this script runs ALL
6 sessions (S0 stays as-is, S1..S5 re-imprint heads using the frozen backbone)
because SimpleCIL is so cheap there's no point in per-session CLI.

Usage:
    python scripts/run_simplecil.py
    python scripts/run_simplecil.py --weight-scale 10.0   # sharper softmax
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pollen_incremental.simplecil import run_simplecil_session
from pollen_incremental.stream import load_stream


DEFAULT_MANIFEST = Path("/home/dubu/manh/lab/ultralytics/data/stream_split.json")
DEFAULT_SOURCE_S0 = Path("/home/dubu/manh/lab/ultralytics/pollen_incremental/sessions/session_0_base.pt")
DEFAULT_OUTDIR = Path("/home/dubu/manh/lab/ultralytics/pollen_incremental/sessions/simplecil")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--source-s0", type=Path, default=DEFAULT_SOURCE_S0,
                    help="V1 session_0_base.pt to start from (we'll copy it).")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--weight-scale", type=float, default=1.0,
                    help="Multiplier on prototype magnitudes. 1.0 = pure cosine.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    stream = load_stream(args.manifest)

    # Copy S0 (SimpleCIL also starts from the same trained base — that's
    # consistent with how V1 starts).
    dst_s0 = args.outdir / "session_0_base.pt"
    if not dst_s0.is_file():
        shutil.copy2(args.source_s0, dst_s0)
        print(f"Copied S0: {args.source_s0} → {dst_s0}\n")

    # Run S1..S5 with SimpleCIL re-imprinting
    prev = dst_s0
    for sid in range(1, len(stream)):
        session = stream[sid]
        out = args.outdir / f"session_{sid}.pt"
        print(f"\n=== SimpleCIL S{sid} ({session.label}) ===")
        run_simplecil_session(
            prev_ckpt_path=prev,
            mapping_csv_new=session.mapping_csv,
            session_data_dir=session.data_dir,
            out_ckpt_path=out,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            weight_scale=args.weight_scale,
        )
        prev = out

    print(f"\n✓ All SimpleCIL sessions complete: {args.outdir}")


if __name__ == "__main__":
    main()

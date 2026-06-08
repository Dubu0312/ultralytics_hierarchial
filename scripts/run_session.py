#!/usr/bin/env python3
"""CLI to run one incremental session.

Reads the stream manifest, picks the requested session, calls run_session().

Usage:
    # Run S1 from the converted base
    python scripts/run_session.py --session 1

    # Run S2 from the S1 checkpoint
    python scripts/run_session.py --session 2

    # Override epochs / β / etc.
    python scripts/run_session.py --session 1 --epochs 10 --beta-kd 0.5

    # Smoke test (1 epoch, no backbone training, useful for debugging)
    python scripts/run_session.py --session 1 --epochs 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pollen_incremental.checkpoint import IncrementalCheckpoint
from pollen_incremental.stream import load_stream
from pollen_incremental.train_incremental import SessionConfig, run_session


DEFAULT_MANIFEST = Path("/home/dubu/manh/lab/ultralytics/data/stream_split.json")
DEFAULT_SESSIONS_DIR = Path("/home/dubu/manh/lab/ultralytics/pollen_incremental/sessions")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=int, required=True,
                    help="Session ID to run (1, 2, …). Session 0 must already exist (run build_base_session.py).")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help="Path to stream_split.json.")
    ap.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR,
                    help="Where to read previous + write current checkpoints.")
    ap.add_argument("--prev-ckpt", type=Path, default=None,
                    help="Override the path to the previous session checkpoint.")
    ap.add_argument("--out-ckpt", type=Path, default=None,
                    help="Override the output checkpoint path.")
    # Hyperparam overrides
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--beta-kd", type=float, default=None)
    ap.add_argument("--temperature-kd", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--budget-per-class", type=int, default=None)
    ap.add_argument("--no-populate-base-memory", action="store_true",
                    help="Skip populating memory for base species (default: populate on session 1).")
    ap.add_argument("--no-replay", action="store_true",
                    help="Disable exemplar replay. For naive-finetune baseline.")
    ap.add_argument("--no-kd", action="store_true",
                    help="Disable LwF knowledge distillation. For naive-finetune baseline.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.session < 1:
        raise SystemExit("--session must be >= 1 (session 0 is built by build_base_session.py)")

    stream = load_stream(args.manifest)
    if args.session >= len(stream):
        raise SystemExit(f"Session {args.session} not found in stream (max: {len(stream) - 1})")
    session = stream[args.session]

    prev_ckpt = args.prev_ckpt or (args.sessions_dir / f"session_{args.session - 1}_base.pt"
                                    if args.session == 1
                                    else args.sessions_dir / f"session_{args.session - 1}.pt")
    out_ckpt = args.out_ckpt or (args.sessions_dir / f"session_{args.session}.pt")

    if not prev_ckpt.is_file():
        raise SystemExit(
            f"Previous checkpoint not found: {prev_ckpt}\n"
            f"(Did you run scripts/build_base_session.py first?)"
        )

    # Build config from defaults + CLI overrides
    cfg = SessionConfig()
    for attr in ("epochs", "lr", "batch_size", "beta_kd", "temperature_kd",
                 "seed", "budget_per_class"):
        cli_name = attr.replace("_", "-").replace("batch-size", "batch-size")
        val = getattr(args, attr.replace("-", "_"), None)
        if val is not None:
            setattr(cfg, attr, val)

    populate_base = (args.session == 1) and (not args.no_populate_base_memory)

    print(f"=== Running session {session.session} ({session.label}) ===")
    print(f"Case:           {session.case}")
    print(f"New species:    {session.new_species}")
    print(f"Data dir:       {session.data_dir}")
    print(f"Mapping CSV:    {session.mapping_csv}")
    print(f"Prev ckpt:      {prev_ckpt}")
    print(f"Out ckpt:       {out_ckpt}")
    print(f"Cfg:            {cfg.as_dict()}")
    print(f"Populate base memory: {populate_base}")
    print()

    metrics = run_session(
        prev_ckpt_path=prev_ckpt,
        mapping_csv_new=session.mapping_csv,
        session_data_dir=session.data_dir,
        out_ckpt_path=out_ckpt,
        cfg=cfg,
        populate_base_memory=populate_base,
        use_replay=not args.no_replay,
        use_kd=not args.no_kd,
    )

    print()
    print(f"=== Session {metrics.session_id} done ===")
    print(f"Delta: {metrics.delta.as_dict()}")
    print(f"Best val HierAcc: {metrics.best_val_hier:.3f} (epoch {metrics.best_epoch})")
    print(f"Elapsed: {metrics.elapsed_sec:.1f}s")


if __name__ == "__main__":
    main()

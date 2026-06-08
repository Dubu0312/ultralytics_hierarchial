# CLAUDE.md

Guidance for Claude/agents working in this repo.

## What this repo is

A fork of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) modified for **hierarchical taxonomy classification** (Order → Family → Genus → Species). One image, four simultaneous label predictions. Built for pollen classification.

The upstream Ultralytics codebase is intact; the hierarchical work is additive.

## Read this before touching hierarchical code

There are **two parallel implementations** of hierarchical classification and they do not connect to each other:

### Path A — Standalone trainers (THE ONES THAT WORK)

Two parallel scripts, identical except for the loss function:

- [ultralytics/models/yolo/classify/train_hierarchical.py](ultralytics/models/yolo/classify/train_hierarchical.py) — **baseline**, independent CE per level (`multitask_ce`)
- [ultralytics/models/yolo/classify/train_hierarchical_masked.py](ultralytics/models/yolo/classify/train_hierarchical_masked.py) — **masked variant**, child logits masked by GT parent before CE (`multitask_ce_masked`)

Shared properties of both:
- Bypass Ultralytics' Trainer entirely
- Define their own `YOLOHierarchicalClassifier`: load a YOLO classify model, slice off the last layer (`yolo_model.model.model[:-1]`), add Conv+Pool+Dropout, then 4 `nn.Linear` heads
- Own `HierarchicalDataset`, own training loop, own eval, own checkpoint format
- Read `mapping.csv` with columns `class_name, ordor, familia, genus`; species IDs derived from folder names
- Default level names: **`ordor / familia / genus / species`** (note the typo "ordor")
- Eval uses **independent argmax** in both — keeps per-epoch numbers comparable between the two scripts

Masked checkpoint additionally has field `"masked_training": True`.

### Path B — Framework integration (INCOMPLETE, dead from user perspective)
- [`HierarchicalClassify`](ultralytics/nn/modules/head.py#L815) — multi-head classifier module, `nn.ModuleDict` of `nn.Linear`
- [`HierarchicalClassificationLoss`](ultralytics/utils/loss.py#L1245) — weighted sum of per-level cross-entropy
- [`ClassificationModel.init_criterion`](ultralytics/nn/tasks.py#L690) — switches to hierarchical loss when `self.hierarchical=True`
- **Missing**: no Trainer subclass, no dataset that yields dict targets, nothing ever sets `self.hierarchical=True`
- Default level names in the example: **`order / family / genus / species`** (no typo) — inconsistent with Path A

If a user asks to "train hierarchical", they mean Path A (one of the two scripts). If they ask to "finish the integration", they mean wiring Path B together (likely renaming Path B keys to match Path A so the upstream `HierarchicalClassificationLoss.lambdas` default doesn't `KeyError`).

### Hierarchy-consistency: where it lives

| Stage | Implementation | Location |
|---|---|---|
| Inference-time masked decoding | Mask child logits by **predicted** parent, top-down argmax | [test_yolo.py:176-203](ultralytics/models/yolo/classify/yolov8s-trained/test/test_yolo.py#L176) `masked_decode` (use with `--masked` flag) |
| Training-time masked CE | Mask child logits by **ground-truth** parent, then CE | [train_hierarchical_masked.py:211](ultralytics/models/yolo/classify/train_hierarchical_masked.py#L211) `multitask_ce_masked` |

The two training scripts × `--masked` inference flag give a 4-cell ablation grid. See [README.md](README.md) for the user-facing table and [HIERARCHICAL.md](HIERARCHICAL.md) for the academic writeup (Vietnamese — Section 5 has the result tables with placeholders to fill in).

## Repository layout (hierarchical-relevant)

```
.
├── README.md                                    # User-facing, covers Path A
├── HIERARCHICAL.md                              # Technical writeup (Vietnamese) — for paper/thesis/slides
├── CLAUDE.md                                    # This file
├── test_hierarchical.py                         # Smoke test for Path B head (forward shape only)
├── ultralytics/
│   ├── nn/
│   │   ├── modules/
│   │   │   ├── head.py                          # Path B: HierarchicalClassify (line ~815)
│   │   │   └── __init__.py                      # Exports HierarchicalClassify
│   │   └── tasks.py                             # Path B switch in init_criterion
│   ├── utils/loss.py                            # Path B: HierarchicalClassificationLoss (line ~1245)
│   └── models/yolo/classify/
│       ├── train_hierarchical.py                # Path A: baseline trainer (independent CE)
│       ├── train_hierarchical_masked.py         # Path A: masked CE trainer (hierarchy-consistent)
│       ├── yolov8s-trained/                     # Trained yolov8s checkpoint + test outputs (gitignored?)
│       └── yolo11x-trained/                     # Trained yolo11x checkpoint + test outputs
└── (rest of upstream Ultralytics, unmodified)
```

## Running things

### Train (Path A)

```bash
# Baseline (independent CE)
python ultralytics/models/yolo/classify/train_hierarchical.py \
    --data /path/to/dataset --mapping_csv /path/to/mapping.csv \
    --model yolov8s-cls.pt --epochs 100 --output outputs_hierarchical

# Masked CE (hierarchy-consistent training)
python ultralytics/models/yolo/classify/train_hierarchical_masked.py \
    --data /path/to/dataset --mapping_csv /path/to/mapping.csv \
    --model yolov8s-cls.pt --epochs 100 --output outputs_hierarchical_masked
```

Both scripts have **identical CLI**. Use whichever python interpreter has Ultralytics installed — in this env: `/home/dubu/miniconda3/envs/lab/bin/python`.

Dataset: `train/<class>/*.jpg` and `val/<class>/*.jpg`. CSV needs `class_name, ordor, familia, genus` (script auto-renames `order` → `ordor`).

### Smoke test (Path B head)

```bash
python test_hierarchical.py
```

Only checks `HierarchicalClassify` forward returns the expected dict shapes. Does **not** exercise Path A.

## Conventions worth knowing

- **Level naming**: Path A uses `ordor / familia / genus / species`. Don't "fix" the typo `ordor` without also updating the trainer, CSV column expectations, and saved checkpoints — it's load-bearing across the standalone path.
- **Loss weighting**: `LAMBDA` constant near the top of [train_hierarchical.py:52](ultralytics/models/yolo/classify/train_hierarchical.py#L52). Higher weight on higher taxonomic levels (Order > Family > Genus > Species), because coarse-level errors are considered more serious.
- **Early stopping** watches **hierarchical accuracy** (all four levels correct on the same sample), not species accuracy.

## Don't do this

- Don't add a `train_hierarchical.py` at the repo root — there used to be one that just re-imported a non-existent `HierarchicalClassificationTrainer` class. It was deleted; don't resurrect it.
- Don't merge Path A and Path B without explicit ask. Two paths exist because Path A was the pragmatic standalone solution while Path B was a partial attempt to do it "the Ultralytics way." Merging them is a real piece of work, not a cleanup.
- Don't merge `train_hierarchical.py` and `train_hierarchical_masked.py` into a single script with a `--masked` flag, and don't extract a shared `hierarchical_utils.py` between them. User explicitly chose the duplicated-script approach so the baseline file stays frozen as a reproducibility anchor. Confirm before refactoring this.
- Don't reformat or restructure upstream Ultralytics files beyond the minimal additions already there. This is a fork, not a rewrite — keeping upstream diffs small matters for future rebases.

## When upstream changes

This is a fork of Ultralytics. If pulling upstream:

- `ultralytics/nn/modules/head.py`, `ultralytics/utils/loss.py`, `ultralytics/nn/tasks.py`, `ultralytics/nn/modules/__init__.py` all have local edits — expect conflicts there
- The new file `ultralytics/models/yolo/classify/train_hierarchical.py` is unique to this fork and won't conflict

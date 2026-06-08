"""Vòng huấn luyện 1 session incremental.

V1 implementation. Combines the four ingredients from HIERARCHICAL_INCREMENTAL.md §0:
    1. Frozen backbone (feature-level forgetting prevention)
    2. Dynamic head expansion + weight imprinting (room for new classes, init from
       prototypes so they're useful immediately)
    3. Exemplar replay + herding (rehearsal of old classes)
    4. Knowledge distillation (soft anchor to teacher's old-class predictions)

Entry point:
    run_session(prev_ckpt_path, mapping_csv_new, new_train_dir, ...)

Outputs:
    - Saves the new IncrementalCheckpoint to `out_ckpt_path`.
    - Returns a dict with training metrics (per-epoch HierAcc on val).

This module does training only; it doesn't pick the data — the caller
(scripts/run_session.py) is responsible for resolving the session's data
paths from the stream manifest.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from pollen_incremental.checkpoint import IncrementalCheckpoint
from pollen_incremental.dataset import (
    HierarchicalImageDataset,
    build_dataframe,
    build_replay_dataframe,
    list_images,
)
from pollen_incremental.distill import multihead_kd_loss
from pollen_incremental.eval_incremental import evaluate, format_eval_result
from pollen_incremental.exemplar_memory import ExemplarMemory
from pollen_incremental.masked_ops import multitask_ce_masked, to_parent_tensors
from pollen_incremental.model import YOLOHierarchicalClassifier
from pollen_incremental.model_expand import (
    compute_prototype,
    compute_prototypes_per_class,
    expand_linear,
)
from pollen_incremental.taxonomy import TaxonomyDelta, TaxonomyState, extend_state


@dataclass
class SessionConfig:
    """Hyperparameters for one incremental session."""

    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.05
    batch_size: int = 32
    imgsz: int = 224
    workers: int = 4
    beta_kd: float = 0.1           # KD weight (LwF strength) — tuned: 1.0 over-anchors species head,
                                   # blocks intra-genus plasticity. 0.1 keeps no-forgetting AND lets
                                   # new same-genus species be learned. See ablation in
                                   # HIERARCHICAL_INCREMENTAL.md §5.4.
    temperature_kd: float = 2.0
    seed: int = 42
    freeze_backbone: bool = True   # V1 default per Section 3.2
    eval_every: int = 1            # epochs between val evals
    log_train_every: int = 1       # epochs between train-loss prints
    # Memory: after the session, herd `budget_per_class` exemplars for the NEW species
    # (and re-herd existing species if backbone changed — V1 doesn't unfreeze so we skip).
    budget_per_class: int = 20

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class SessionMetrics:
    """Returned from run_session() for logging / aggregation."""

    session_id: int
    delta: TaxonomyDelta
    epochs_run: int
    train_losses: list[float] = field(default_factory=list)  # per-epoch
    val_hier_per_epoch: list[float] = field(default_factory=list)
    best_val_hier: float = 0.0
    best_epoch: int = 0
    final_val_result_dict: dict | None = None
    elapsed_sec: float = 0.0


# =========================
# Helpers
# =========================


def _set_seed(seed: int) -> None:
    import random as _r
    _r.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_species_image_lists(
    new_train_dir: Path,
    new_species_names: list[str],
) -> dict[str, list[str]]:
    """For each new species, list all its train images under new_train_dir/<species>/."""
    all_imgs = list_images(new_train_dir, species_filter=set(new_species_names))
    missing = [n for n in new_species_names if n not in all_imgs or not all_imgs[n]]
    if missing:
        raise FileNotFoundError(
            f"New species missing training images under {new_train_dir}: {missing}"
        )
    return {n: all_imgs[n] for n in new_species_names}


def _compute_new_prototypes(
    model: YOLOHierarchicalClassifier,
    images_by_species_name: dict[str, list[str]],
    new_species_ids: dict[str, int],
    imgsz: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """For each new species (in the order they were just appended to species2id),
    compute one 1280-d L2-normalized prototype.

    Returns:
        (n_new_species, 1280) tensor on `device`.
    """
    from PIL import Image
    from torchvision import transforms

    # Use eval transform — clean features for prototype.
    tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _batches(paths: list[str]):
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = [tf(Image.open(p).convert("RGB")) for p in batch_paths]
            yield torch.stack(imgs)

    # Order by ascending species ID (matches the order in which extend_state added them)
    ordered_names = sorted(new_species_ids.keys(), key=lambda n: new_species_ids[n])
    by_class = {new_species_ids[n]: _batches(images_by_species_name[n]) for n in ordered_names}
    return compute_prototypes_per_class(model.forward_features, by_class, device=device)


def _populate_memory_for_species(
    model: YOLOHierarchicalClassifier,
    memory: ExemplarMemory,
    species_id: int,
    image_paths: list[str],
    n_classes_now: int,
    imgsz: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Run backbone on all images of one species, then store herding-selected exemplars."""
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
            feats = model.forward_features(imgs)  # (B, 1280)
            all_feats.append(feats.cpu())
    all_feats = torch.cat(all_feats, dim=0)
    memory.add_class(species_id, all_feats, image_paths, n_classes_now=n_classes_now)


# =========================
# Main entry
# =========================


def run_session(
    prev_ckpt_path: Path | str,
    mapping_csv_new: Path | str,
    session_data_dir: Path | str,
    out_ckpt_path: Path | str,
    cfg: SessionConfig | None = None,
    populate_base_memory: bool = False,
    populate_old_species_memory: bool = False,
    use_replay: bool = True,
    use_kd: bool = True,
) -> SessionMetrics:
    """Train one incremental session.

    Args:
        prev_ckpt_path: previous session's IncrementalCheckpoint (.pt).
        mapping_csv_new: cumulative mapping CSV (base + everything up to this session).
        session_data_dir: directory with train/val/test subdirs containing
            symlinks to images for cumulative species.
        out_ckpt_path: where to save the new IncrementalCheckpoint.
        cfg: hyperparameters; None → defaults (SessionConfig()).
        populate_base_memory: if True AND prev checkpoint has empty memory
            (typical for session 1, after build_base_session.py), populate
            memory for all base species using the frozen backbone before
            training. This is the iCaRL "post-base-session" step.
        populate_old_species_memory: if True, also re-herd memory for old species
            this session (useful if backbone gets unfrozen). V1 default: False.
        use_replay: if False, the training set contains ONLY new-species data
            (no exemplar replay). Used for the naive-finetune baseline.
        use_kd: if False, the loss is just masked CE — no LwF distillation.
            Used for the naive-finetune baseline.

    Returns:
        SessionMetrics with per-epoch training info.
    """
    cfg = cfg or SessionConfig()
    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prev_ckpt_path = Path(prev_ckpt_path)
    session_data_dir = Path(session_data_dir)
    out_ckpt_path = Path(out_ckpt_path)
    mapping_csv_new = Path(mapping_csv_new)

    t_start = time.time()

    # 1. Load previous-session checkpoint
    print(f"[Session] Loading previous: {prev_ckpt_path}")
    prev = IncrementalCheckpoint.load(prev_ckpt_path)
    print(f"  prev session: {prev.session}, #classes: {prev.n_classes()}")

    # 2. Build model from previous state (with OLD head sizes)
    model = YOLOHierarchicalClassifier(
        prev.base_model,
        n_ordor=prev.tax_state.n_ordor,
        n_familia=prev.tax_state.n_familia,
        n_genus=prev.tax_state.n_genus,
        n_species=prev.tax_state.n_species,
    )
    model.load_state_dict(prev.model_state, strict=True)
    model.to(device)

    if cfg.freeze_backbone:
        model.freeze_backbone(True)
        print(f"  Backbone frozen.")

    # 3. Extend taxonomy with new session's CSV
    map_df = pd.read_csv(mapping_csv_new)
    new_state, delta = extend_state(prev.tax_state, map_df)
    print(f"  Delta (new IDs added): {delta.as_dict()}")

    # Compute HEAD-SIZE delta (the actual #columns to add to each Linear head).
    # This differs from `delta` (which counts new IDs added to each level's set):
    # ID convention is `n_class = max_id + 1`, so adding a class with an ID
    # that fits in an already-allocated slot (e.g. familia 2 when max was 9)
    # does NOT grow the head — it just fills a previously-sentinel slot.
    n_old = prev.n_classes()
    head_delta = {
        "order":   new_state.n_ordor   - n_old["ordor"],
        "family":  new_state.n_familia - n_old["familia"],
        "genus":   new_state.n_genus   - n_old["genus"],
        "species": new_state.n_species - n_old["species"],
    }
    print(f"  Head delta (columns to add): {head_delta}  (new total: {new_state.n_classes()})")
    n_old_dict = {
        "ordor": n_old["ordor"], "familia": n_old["familia"],
        "genus": n_old["genus"], "species": n_old["species"],
    }

    # 4. Identify the NEW species names (everything in new_state.species2id that
    # wasn't in prev). Sorted by new ID for prototype ordering.
    new_species_names = [
        name for name, sid in sorted(new_state.species2id.items(), key=lambda kv: kv[1])
        if name not in prev.tax_state.species2id
    ]
    print(f"  New species: {new_species_names}")

    # 5. Snapshot teacher BEFORE expanding head (so teacher has the OLD head sizes)
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 6. Compute prototypes for new species and expand all heads
    new_species_ids = {name: new_state.species2id[name] - n_old["species"]
                       for name in new_species_names}
    # ^ Indices INTO the protos tensor; we'll re-map below to the full species ID.
    # Actually, compute_prototypes_per_class returns rows in dict-key insertion order.
    # We want the rows in NEW species ID order:
    new_species_ids_actual = {name: new_state.species2id[name] for name in new_species_names}

    if delta.species > 0:
        new_imgs_by_name = _new_species_image_lists(
            session_data_dir / "train", new_species_names
        )
        protos_species = _compute_new_prototypes(
            model, new_imgs_by_name, new_species_ids_actual,
            imgsz=cfg.imgsz, batch_size=cfg.batch_size, device=device,
        )
    else:
        protos_species = None

    # Expand each head by the HEAD-SIZE delta (see comment above for why
    # this differs from `delta`). Prototypes are only computed for species
    # because that's the only level with image-level labels per class — for
    # order/family/genus we use zero-init (the head will be tuned during training).
    model.head_ordor   = expand_linear(model.head_ordor,   head_delta["order"])
    model.head_familia = expand_linear(model.head_familia, head_delta["family"])
    model.head_genus   = expand_linear(model.head_genus,   head_delta["genus"])
    model.head_species = expand_linear(
        model.head_species, head_delta["species"], proto=protos_species,
    )
    model.to(device)
    print(f"  Heads expanded.")

    # 7. Populate memory for OLD species if asked (typically at S1 because S0
    # memory was empty after build_base_session.py)
    memory = copy.deepcopy(prev.exemplar_memory)
    if populate_base_memory and memory.n_classes() == 0:
        print(f"  Populating base memory for {prev.tax_state.n_species} old species...")
        all_train = list_images(session_data_dir / "train",
                                species_filter=set(prev.tax_state.species2id.keys()))
        n_total = prev.tax_state.n_species + delta.species
        for sid_name, sid in sorted(prev.tax_state.species2id.items(), key=lambda kv: kv[1]):
            if sid_name not in all_train:
                continue
            _populate_memory_for_species(
                model, memory, sid, all_train[sid_name],
                n_classes_now=n_total, imgsz=cfg.imgsz, batch_size=cfg.batch_size,
                device=device,
            )
        print(f"  Memory now has {memory.n_classes()} classes, {memory.n_exemplars()} exemplars.")

    # 8. Build datasets: NEW species data + REPLAY exemplars (no old-species fresh data)
    new_train_df = build_dataframe(
        session_data_dir / "train", map_df, new_state.species2id,
        species_filter=set(new_species_names),
    )
    if len(new_train_df) == 0:
        raise RuntimeError(
            f"No images found for new species {new_species_names} under {session_data_dir / 'train'}"
        )
    new_train_ds = HierarchicalImageDataset(new_train_df, imgsz=cfg.imgsz, augment=True)

    if use_replay:
        replay_df = build_replay_dataframe(
            memory.all_items(),
            new_state.species_parent, new_state.genus_parent, new_state.familia_parent,
        )
        if len(replay_df) > 0:
            replay_ds = HierarchicalImageDataset(replay_df, imgsz=cfg.imgsz, augment=True)
            train_ds = ConcatDataset([new_train_ds, replay_ds])
            print(f"  Train: {len(new_train_ds)} new + {len(replay_ds)} replay = {len(train_ds)} total")
        else:
            train_ds = new_train_ds
            print(f"  Train: {len(train_ds)} (NO REPLAY — memory empty)")
    else:
        train_ds = new_train_ds
        print(f"  Train: {len(train_ds)} new only (--no-replay: naive baseline)")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=True,
    )

    val_df = build_dataframe(
        session_data_dir / "val", map_df, new_state.species2id,
    )
    val_ds = HierarchicalImageDataset(val_df, imgsz=cfg.imgsz, augment=False)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True,
    )
    print(f"  Val: {len(val_ds)} images")

    # 9. Parent tensors for masked CE / masked decode (after extend)
    fp, gp, sp_t = to_parent_tensors(new_state, device=device)

    # 10. Optimizer (only trainable parameters — i.e. heads if backbone frozen)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Trainable parameters: {n_trainable:,}")
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * max(1, len(train_loader)),
    )

    # 11. Training loop
    metrics = SessionMetrics(session_id=prev.session + 1, delta=delta, epochs_run=0)
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if cfg.freeze_backbone:
            model.backbone.eval()  # backbone in eval mode regardless of model.train()

        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = {k: v.to(device) for k, v in y.items()}

            logits = model(x)
            ce_total, _ = multitask_ce_masked(logits, y, prev.lambdas, fp, gp, sp_t)

            if use_kd and cfg.beta_kd > 0 and n_old["species"] > 0:
                with torch.no_grad():
                    teacher_out = teacher.forward_dict(x)
                student_out = model.forward_dict(x)
                kd = multihead_kd_loss(student_out, teacher_out, n_old_dict,
                                       temperature=cfg.temperature_kd)
                loss = ce_total + cfg.beta_kd * kd
            else:
                loss = ce_total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        metrics.train_losses.append(avg_loss)

        # Validation
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            val_result = evaluate(model, val_loader, device, fp, gp, sp_t, decode="masked")
            metrics.val_hier_per_epoch.append(val_result.hier_acc)
            if val_result.hier_acc > metrics.best_val_hier:
                metrics.best_val_hier = val_result.hier_acc
                metrics.best_epoch = epoch
                # Snapshot the best model state (CPU to save VRAM)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch % cfg.log_train_every == 0 or epoch == cfg.epochs:
                print(f"  [E{epoch:03d}/{cfg.epochs:03d}] loss={avg_loss:.4f}  "
                      f"{format_eval_result('val', val_result)}")

        metrics.epochs_run = epoch

    # 12. Use the best-by-val model state for the saved checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate(model, val_loader, device, fp, gp, sp_t, decode="masked")
    metrics.final_val_result_dict = final_val.as_dict()
    print(f"  {format_eval_result('FINAL val (best by HierAcc)', final_val)}")

    # 13. Update memory: add exemplars for the NEW species
    if delta.species > 0:
        print(f"  Adding new species to memory...")
        for name in new_species_names:
            sid = new_state.species2id[name]
            _populate_memory_for_species(
                model, memory, sid, new_imgs_by_name[name],
                n_classes_now=new_state.n_species,
                imgsz=cfg.imgsz, batch_size=cfg.batch_size, device=device,
            )
        memory.rebalance(n_classes_now=new_state.n_species)
        print(f"  Memory: {memory.n_classes()} classes, {memory.n_exemplars()} exemplars")

    # 14. Save
    new_ckpt = IncrementalCheckpoint(
        session=prev.session + 1,
        base_model=prev.base_model,
        model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        tax_state=new_state,
        exemplar_memory=memory,
        lambdas=dict(prev.lambdas),
        incremental_cfg={
            **cfg.as_dict(),
            "prev_ckpt": str(prev_ckpt_path),
            "mapping_csv": str(mapping_csv_new),
            "delta": delta.as_dict(),
            "best_val_hier": metrics.best_val_hier,
            "best_epoch": metrics.best_epoch,
        },
    )
    new_ckpt.save(out_ckpt_path)
    metrics.elapsed_sec = time.time() - t_start
    print(f"\n✓ Wrote session-{new_ckpt.session} checkpoint: {out_ckpt_path}")
    print(f"  Elapsed: {metrics.elapsed_sec:.1f}s")
    return metrics

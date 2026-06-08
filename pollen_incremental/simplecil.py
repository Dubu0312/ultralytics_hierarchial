"""SimpleCIL baseline — train-free, frozen-prototype classifier.

Reference: Zhou, D.-W. et al. "Revisiting Class-Incremental Learning with
Pre-Trained Models: Generalizability and Adaptivity are All You Need."
IJCV 2024 (arXiv:2303.07338).

Recipe per session:
    1. Load prev checkpoint (frozen backbone + 4 heads).
    2. For each NEW species: forward all training images through backbone,
       compute L2-normalized mean feature → that IS the new row of the
       species head's weight matrix (no scaling — pure cosine classifier).
    3. (Optional) Re-imprint OLD species too, so all rows are on the same
       cosine scale. V1's expand_linear scales prototypes by mean(||old rows||),
       which is fine when we keep training; for SimpleCIL we want pure cosine.
    4. NO training, NO KD, NO replay. Just save the checkpoint.

This is the strongest *train-free* baseline in modern PTM-CIL literature.
If V1 (with replay + KD + head training) cannot beat SimpleCIL, we have
no value-add over the simple frozen-prototype approach.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from pollen_incremental.checkpoint import IncrementalCheckpoint
from pollen_incremental.dataset import list_images
from pollen_incremental.exemplar_memory import ExemplarMemory
from pollen_incremental.model import YOLOHierarchicalClassifier
from pollen_incremental.taxonomy import extend_state


def _eval_transform(imgsz: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def _class_prototype(
    feature_extractor,
    image_paths: list[str],
    imgsz: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute one L2-normalized mean-feature prototype for a single class."""
    tf = _eval_transform(imgsz)
    feats = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch]).to(device)
        f = feature_extractor(imgs)
        feats.append(F.normalize(f, dim=1))
    all_feats = torch.cat(feats, dim=0)
    mu = all_feats.mean(dim=0)
    return F.normalize(mu, dim=0)


def _imprint_all_species(
    model: YOLOHierarchicalClassifier,
    species2id: dict[str, int],
    train_root: Path,
    imgsz: int,
    batch_size: int,
    device: torch.device,
    weight_scale: float = 1.0,
) -> None:
    """Replace EVERY row of head_species with the L2-normalized prototype of that class.

    Args:
        weight_scale: multiplier on prototypes. Default 1.0 = pure cosine
            classifier. Set higher to amplify logits (sharper softmax).
    """
    model.eval()
    imgs_by_class = list_images(train_root, species_filter=set(species2id.keys()))

    new_weight = torch.zeros_like(model.head_species.weight.data)
    for name, sid in species2id.items():
        if name not in imgs_by_class or not imgs_by_class[name]:
            print(f"    [WARN] no images for {name} — leaving row {sid} unchanged")
            continue
        proto = _class_prototype(
            model.forward_features, imgs_by_class[name],
            imgsz=imgsz, batch_size=batch_size, device=device,
        )
        new_weight[sid] = proto * weight_scale

    model.head_species.weight.data.copy_(new_weight)
    if model.head_species.bias is not None:
        model.head_species.bias.data.zero_()


def _imprint_internal_heads_by_majority(
    model: YOLOHierarchicalClassifier,
    species2id: dict[str, int],
    species_parent: list[int],
    genus_parent: list[int],
    familia_parent: list[int],
    train_root: Path,
    imgsz: int,
    batch_size: int,
    device: torch.device,
    weight_scale: float = 1.0,
) -> None:
    """For order/family/genus heads, set each row = L2-norm mean of all images
    belonging to that node (i.e. union of descendant-species images).

    This gives a coherent multi-head cosine classifier at every level — without
    that, only the species head is meaningful and order/family/genus would be
    random.
    """
    model.eval()
    imgs_by_species = list_images(train_root, species_filter=set(species2id.keys()))

    # Build species_id → image paths
    paths_by_sid = {species2id[name]: imgs_by_species.get(name, []) for name in species2id}

    # For each internal level, aggregate images by parent ID
    def _imprint_one(head: nn.Linear, child_to_parent_for_species: list[int]):
        """For each parent ID active at this level (= an index of `head`), gather
        all SPECIES image paths whose ancestor at that level equals this parent ID."""
        n_out = head.out_features
        new_w = torch.zeros_like(head.weight.data)
        for parent_id in range(n_out):
            paths = []
            for sid, sid_paths in paths_by_sid.items():
                # Compute the ancestor at this level
                g = species_parent[sid]
                if g < 0:
                    continue
                if child_to_parent_for_species is species_parent:
                    ancestor = g  # genus
                elif child_to_parent_for_species is genus_parent:
                    ancestor = genus_parent[g] if g >= 0 else -1  # familia
                else:
                    f = genus_parent[g] if g >= 0 else -1
                    ancestor = familia_parent[f] if f >= 0 else -1  # ordor
                if ancestor == parent_id:
                    paths.extend(sid_paths)
            if not paths:
                continue
            proto = _class_prototype(
                model.forward_features, paths,
                imgsz=imgsz, batch_size=batch_size, device=device,
            )
            new_w[parent_id] = proto * weight_scale
        head.weight.data.copy_(new_w)
        if head.bias is not None:
            head.bias.data.zero_()

    _imprint_one(model.head_genus, species_parent)     # genus = parent of species
    _imprint_one(model.head_familia, genus_parent)     # familia = parent of genus
    _imprint_one(model.head_ordor, familia_parent)     # ordor = parent of familia


def run_simplecil_session(
    prev_ckpt_path: Path | str,
    mapping_csv_new: Path | str,
    session_data_dir: Path | str,
    out_ckpt_path: Path | str,
    imgsz: int = 224,
    batch_size: int = 16,
    weight_scale: float = 1.0,
) -> dict:
    """Run one SimpleCIL session: no training, just re-imprint all heads
    using the frozen backbone over CUMULATIVE training data.

    Args:
        prev_ckpt_path: previous session's IncrementalCheckpoint (.pt).
        mapping_csv_new: cumulative mapping CSV.
        session_data_dir: dir with train/val/test subdirs (cumulative species).
        out_ckpt_path: where to save the new SimpleCIL checkpoint.
        imgsz / batch_size: inference settings.
        weight_scale: multiplier on prototypes (default 1.0 = pure cosine).

    Returns:
        Dict with timing + delta info.
    """
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prev_ckpt_path = Path(prev_ckpt_path)
    session_data_dir = Path(session_data_dir)
    out_ckpt_path = Path(out_ckpt_path)

    print(f"[SimpleCIL] Loading: {prev_ckpt_path}")
    prev = IncrementalCheckpoint.load(prev_ckpt_path)

    map_df = pd.read_csv(mapping_csv_new)
    new_state, delta = extend_state(prev.tax_state, map_df)
    print(f"  Delta: {delta.as_dict()}  (new total: {new_state.n_classes()})")

    head_delta = {
        "order":   new_state.n_ordor   - prev.tax_state.n_ordor,
        "family":  new_state.n_familia - prev.tax_state.n_familia,
        "genus":   new_state.n_genus   - prev.tax_state.n_genus,
        "species": new_state.n_species - prev.tax_state.n_species,
    }

    # Build model with NEW head sizes (we'll fully overwrite weights anyway)
    model = YOLOHierarchicalClassifier(
        prev.base_model,
        n_ordor=new_state.n_ordor, n_familia=new_state.n_familia,
        n_genus=new_state.n_genus, n_species=new_state.n_species,
    )
    # Load only backbone + conv + (old head shapes) — we'll rebuild heads
    # by re-imprinting, so old head weights don't matter.
    # Copy backbone state from prev.
    prev_model = YOLOHierarchicalClassifier(
        prev.base_model,
        n_ordor=prev.tax_state.n_ordor, n_familia=prev.tax_state.n_familia,
        n_genus=prev.tax_state.n_genus, n_species=prev.tax_state.n_species,
    )
    prev_model.load_state_dict(prev.model_state, strict=True)
    # Backbone weights
    model.backbone.load_state_dict(prev_model.backbone.state_dict())
    model.conv.load_state_dict(prev_model.conv.state_dict())
    del prev_model

    model.freeze_backbone(True)
    model.to(device)

    print(f"  Imprinting all {new_state.n_species} species + internal heads "
          f"(scale={weight_scale})...")
    train_root = session_data_dir / "train"
    _imprint_all_species(model, new_state.species2id, train_root,
                         imgsz=imgsz, batch_size=batch_size, device=device,
                         weight_scale=weight_scale)
    _imprint_internal_heads_by_majority(
        model, new_state.species2id,
        new_state.species_parent, new_state.genus_parent, new_state.familia_parent,
        train_root, imgsz=imgsz, batch_size=batch_size, device=device,
        weight_scale=weight_scale,
    )

    # SimpleCIL has no exemplar memory (train-free)
    empty_mem = ExemplarMemory(budget_per_class=20, mode="per_class")
    ckpt = IncrementalCheckpoint(
        session=prev.session + 1,
        base_model=prev.base_model,
        model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        tax_state=new_state,
        exemplar_memory=empty_mem,
        lambdas=dict(prev.lambdas),
        incremental_cfg={
            "method": "SimpleCIL",
            "weight_scale": weight_scale,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "delta": delta.as_dict(),
            "head_delta": head_delta,
        },
    )
    ckpt.save(out_ckpt_path)
    elapsed = time.time() - t0
    print(f"✓ Wrote {out_ckpt_path} (elapsed {elapsed:.1f}s)")
    return {"session": prev.session + 1, "delta": delta.as_dict(), "elapsed_sec": elapsed}

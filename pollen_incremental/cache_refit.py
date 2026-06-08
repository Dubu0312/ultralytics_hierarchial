"""Cache-refit upper bound — not continual learning, just the joint-training trick.

Recipe:
    1. Cache backbone features for EVERY image of EVERY species seen so far
       (~2237 × 1280 float = ~11 MB at S5 — trivial).
    2. After each session, refit the 4 heads from scratch using full masked CE
       on the cached features. Backbone is frozen so this is "free" in the CL
       sense: no gradient through 28M backbone params, just 1M head params on
       1280-d vectors.
    3. NOT a CL method — uses ALL old training data, violates the "limited
       replay budget" constraint. Serves as the upper bound: if frozen backbone
       is good enough, this is the ceiling V1/SimpleCIL try to approach.

If V1 (with replay+KD) approaches this upper bound, we've extracted essentially
all the info that's extractable given the frozen backbone — further gains
would require unfreezing.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from pollen_incremental.checkpoint import IncrementalCheckpoint
from pollen_incremental.dataset import build_dataframe
from pollen_incremental.exemplar_memory import ExemplarMemory
from pollen_incremental.masked_ops import multitask_ce_masked, to_parent_tensors
from pollen_incremental.model import YOLOHierarchicalClassifier
from pollen_incremental.taxonomy import extend_state


def _eval_transform(imgsz: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def _cache_features(
    model: YOLOHierarchicalClassifier,
    df: pd.DataFrame,
    imgsz: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """For each image in df, run backbone → return (N, 1280) feature tensor +
    per-level label tensors.

    No augmentation (clean features). Stored on CPU to save VRAM during refit.
    """
    tf = _eval_transform(imgsz)
    model.eval()
    feats_list: list[torch.Tensor] = []
    o, f, g, s = [], [], [], []
    for i in range(0, len(df), batch_size):
        sub = df.iloc[i:i + batch_size]
        imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in sub["image_path"]]).to(device)
        feats = model.forward_features(imgs).cpu()
        feats_list.append(feats)
        o.extend(sub["ordor"].astype(int).tolist())
        f.extend(sub["familia"].astype(int).tolist())
        g.extend(sub["genus"].astype(int).tolist())
        s.extend(sub["species"].astype(int).tolist())
    return torch.cat(feats_list, dim=0), {
        "ordor": torch.tensor(o, dtype=torch.long),
        "familia": torch.tensor(f, dtype=torch.long),
        "genus": torch.tensor(g, dtype=torch.long),
        "species": torch.tensor(s, dtype=torch.long),
    }


def _refit_heads(
    model: YOLOHierarchicalClassifier,
    feats: torch.Tensor,
    labels: dict[str, torch.Tensor],
    parents: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    lambdas: dict[str, float],
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> float:
    """Train all 4 heads from scratch (re-initialized) on cached features.

    Returns the final loss for logging.
    """
    # Re-init heads (cache-refit semantics: full retrain of heads, not warm-start)
    for head in (model.head_ordor, model.head_familia, model.head_genus, model.head_species):
        nn.init.kaiming_uniform_(head.weight, a=5 ** 0.5)
        if head.bias is not None:
            nn.init.zeros_(head.bias)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)

    feats_dev = feats.to(device)
    labels_dev = {k: v.to(device) for k, v in labels.items()}
    n = feats_dev.size(0)
    fp, gp, sp = parents

    last_loss = float("inf")
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = feats_dev[idx]
            y = {k: v[idx] for k, v in labels_dev.items()}
            logits = (
                model.head_ordor(x),
                model.head_familia(x),
                model.head_genus(x),
                model.head_species(x),
            )
            loss, _ = multitask_ce_masked(logits, y, lambdas, fp, gp, sp)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); nb += 1
        last_loss = epoch_loss / max(1, nb)
    return last_loss


def run_cache_refit_session(
    prev_ckpt_path: Path | str,
    mapping_csv_new: Path | str,
    session_data_dir: Path | str,
    out_ckpt_path: Path | str,
    imgsz: int = 224,
    batch_size_cache: int = 16,
    batch_size_refit: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
) -> dict:
    """Run one cache-refit session.

    1. Extend taxonomy.
    2. Cache features for ALL train images (cumulative species).
    3. Re-init heads, train on cached features with masked CE (full retrain).
    4. Save checkpoint.

    No exemplar memory (we used full training data — not a CL method).
    """
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prev_ckpt_path = Path(prev_ckpt_path)
    session_data_dir = Path(session_data_dir)
    out_ckpt_path = Path(out_ckpt_path)

    print(f"[CacheRefit] Loading: {prev_ckpt_path}")
    prev = IncrementalCheckpoint.load(prev_ckpt_path)

    map_df = pd.read_csv(mapping_csv_new)
    new_state, delta = extend_state(prev.tax_state, map_df)
    print(f"  Delta: {delta.as_dict()}  (new total: {new_state.n_classes()})")

    # Build model with NEW head sizes
    model = YOLOHierarchicalClassifier(
        prev.base_model,
        n_ordor=new_state.n_ordor, n_familia=new_state.n_familia,
        n_genus=new_state.n_genus, n_species=new_state.n_species,
    )
    # Copy backbone from prev
    prev_model = YOLOHierarchicalClassifier(
        prev.base_model,
        n_ordor=prev.tax_state.n_ordor, n_familia=prev.tax_state.n_familia,
        n_genus=prev.tax_state.n_genus, n_species=prev.tax_state.n_species,
    )
    prev_model.load_state_dict(prev.model_state, strict=True)
    model.backbone.load_state_dict(prev_model.backbone.state_dict())
    model.conv.load_state_dict(prev_model.conv.state_dict())
    del prev_model
    model.freeze_backbone(True)
    model.to(device)

    # Build full cumulative train dataframe
    train_df = build_dataframe(
        session_data_dir / "train", map_df, new_state.species2id,
    )
    print(f"  Caching features for {len(train_df)} train images...")

    t_cache0 = time.time()
    feats, labels = _cache_features(model, train_df, imgsz, batch_size_cache, device)
    print(f"  Cached {feats.shape[0]} × {feats.shape[1]} features in {time.time()-t_cache0:.1f}s "
          f"({feats.element_size() * feats.nelement() / 1e6:.1f} MB)")

    # Refit heads
    parents = to_parent_tensors(new_state, device=device)
    print(f"  Refitting heads ({epochs} epochs, lr={lr})...")
    t_refit0 = time.time()
    final_loss = _refit_heads(model, feats, labels, parents,
                              prev.lambdas, epochs, lr, batch_size_refit, device)
    print(f"  Refit done in {time.time()-t_refit0:.1f}s, final loss = {final_loss:.4f}")

    # Save
    empty_mem = ExemplarMemory(budget_per_class=20, mode="per_class")
    ckpt = IncrementalCheckpoint(
        session=prev.session + 1,
        base_model=prev.base_model,
        model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        tax_state=new_state,
        exemplar_memory=empty_mem,
        lambdas=dict(prev.lambdas),
        incremental_cfg={
            "method": "CacheRefit",
            "epochs": epochs,
            "lr": lr,
            "batch_size_refit": batch_size_refit,
            "delta": delta.as_dict(),
            "final_loss": final_loss,
            "n_train_used": len(train_df),
        },
    )
    ckpt.save(out_ckpt_path)
    elapsed = time.time() - t0
    print(f"✓ Wrote {out_ckpt_path} (elapsed {elapsed:.1f}s)")
    return {"session": prev.session + 1, "elapsed_sec": elapsed, "final_loss": final_loss}

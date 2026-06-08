"""Multi-head YOLO classifier for hierarchical taxonomy classification.

Re-implemented here (instead of imported from train_hierarchical_masked.py) so
the incremental codebase is self-contained — we own the head expansion contract
and don't want any drift if the base trainer is ever modified.

Architecture is bit-for-bit identical to YOLOHierarchicalClassifier in
train_hierarchical_masked.py:170-216:

    image (3, 224, 224)
       → YOLO backbone (with last classify head stripped)
       → Conv 1×1 (in_channels → 1280)
       → AdaptiveAvgPool2d(1) → flatten
       → Dropout
       → 4 parallel Linear heads: order / family / genus / species

The base trainer uses attribute names `head_ordor / head_familia / head_genus
/ head_species` (note typo 'ordor'). We keep the same names so existing
checkpoints load via state_dict without renaming.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv


class YOLOHierarchicalClassifier(nn.Module):
    """Shared YOLO backbone + 4 parallel Linear heads.

    Args:
        base_model_path: path or alias of the pretrained YOLO classify model
            (e.g. "yolo11x-cls.pt"). The backbone is everything except the
            original final classify layer.
        n_ordor / n_familia / n_genus / n_species: number of output classes
            at each level (use TaxonomyState.n_classes()).
        dropout: head dropout probability (matches base default 0.0).
        feature_dim: bottleneck channel size after Conv1×1 (matches base 1280).
    """

    def __init__(
        self,
        base_model_path: str,
        n_ordor: int,
        n_familia: int,
        n_genus: int,
        n_species: int,
        dropout: float = 0.0,
        feature_dim: int = 1280,
    ) -> None:
        super().__init__()

        yolo_model = YOLO(base_model_path)
        self.backbone = yolo_model.model.model[:-1]  # strip original classify head

        # Discover backbone output channels by running a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            for layer in self.backbone:
                dummy = layer(dummy)
            c1 = dummy.shape[1]

        self.conv = Conv(c1, feature_dim, k=1, s=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout, inplace=True)

        self.head_ordor = nn.Linear(feature_dim, n_ordor)
        self.head_familia = nn.Linear(feature_dim, n_familia)
        self.head_genus = nn.Linear(feature_dim, n_genus)
        self.head_species = nn.Linear(feature_dim, n_species)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the shared 1280-d feature vector for a batch of images.

        Useful for: prototype computation, NCM classification, exemplar feature
        caching. Always runs through backbone + Conv1×1 + pool + dropout.
        """
        for layer in self.backbone:
            x = layer(x)
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.drop(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits at all 4 levels: (order, family, genus, species)."""
        feat = self.forward_features(x)
        return (
            self.head_ordor(feat),
            self.head_familia(feat),
            self.head_genus(feat),
            self.head_species(feat),
        )

    def forward_dict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Same as forward() but returns a dict — convenient for KD loss."""
        lo, lf, lg, ls = self.forward(x)
        return {"ordor": lo, "familia": lf, "genus": lg, "species": ls}

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze (or unfreeze) the YOLO backbone for incremental training."""
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.backbone.eval()

    def head_modules(self) -> dict[str, nn.Linear]:
        """Return all 4 head Linear modules in a dict for easy iteration."""
        return {
            "ordor": self.head_ordor,
            "familia": self.head_familia,
            "genus": self.head_genus,
            "species": self.head_species,
        }

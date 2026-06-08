# YOLO Hierarchical Classification for Pollen Taxonomy

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

**A modified Ultralytics YOLO framework with hierarchical multi-head classification for pollen taxonomy**

</div>

---

## Overview

This repository extends the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework to support **hierarchical classification** with multiple output heads. It's designed for taxonomic classification tasks where predictions need to be made at multiple levels simultaneously (e.g., Order → Family → Genus → Species).

### Key Features

- **Multi-head Classification**: Separate classification heads for each taxonomic level
- **Hierarchical Loss**: Weighted cross-entropy loss with configurable weights per level
- **YOLO Backbone**: Leverage powerful YOLO feature extractors (YOLOv8, YOLO11, etc.)
- **Flexible Architecture**: Easy to adapt for different hierarchical structures

## Architecture

```
                    ┌─────────────────┐
                    │   Input Image   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  YOLO Backbone  │
                    │  (Feature Ext.) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Conv + Pool   │
                    │   (1280 dim)    │
                    └────────┬────────┘
                             │
           ┌─────────┬───────┴───────┬─────────┐
           │         │               │         │
    ┌──────▼──────┐ ┌▼─────────┐ ┌───▼────┐ ┌──▼─────┐
    │ Order Head  │ │Family Head│ │Genus   │ │Species │
    │ (N classes) │ │(M classes)│ │Head    │ │Head    │
    └─────────────┘ └──────────┘ └────────┘ └────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/yolo-hierarchical-classification.git
cd yolo-hierarchical-classification

# Install dependencies
pip install -e .

# Additional dependencies for training
pip install pandas scikit-learn
```

## Dataset Structure

### Directory Layout

```
dataset/
├── train/
│   ├── class_name_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── class_name_2/
│   │   └── ...
│   └── ...
└── val/
    ├── class_name_1/
    │   └── ...
    └── ...
```

### Mapping CSV

Create a `mapping.csv` file that maps each class to its taxonomic hierarchy:

```csv
class_name,ordor,familia,genus
Pollen_Species_A,0,0,0
Pollen_Species_B,0,0,1
Pollen_Species_C,0,1,2
Pollen_Species_D,1,2,3
...
```

- `class_name`: Folder name in train/val directories
- `ordor`: Order ID (integer, 0-indexed)
- `familia`: Family ID (integer, 0-indexed)
- `genus`: Genus ID (integer, 0-indexed)
- Species ID is automatically derived from folder names

## Usage

### Training

Two training scripts are provided — identical model, dataset, optimizer, and eval, differing only in the loss function:

| Script | Loss | When to use |
|---|---|---|
| `train_hierarchical.py` | Independent CE per level | Baseline |
| `train_hierarchical_masked.py` | Masked CE — child logits masked by GT parent before CE | Enforces taxonomy consistency during training |

#### Baseline (independent CE)

```bash
python ultralytics/models/yolo/classify/train_hierarchical.py \
    --data /path/to/dataset \
    --mapping_csv /path/to/mapping.csv \
    --model yolov8s-cls.pt \
    --epochs 100 \
    --batch-size 32 \
    --imgsz 224 \
    --lr 1e-3 \
    --freeze-epochs 3 \
    --patience 20 \
    --output outputs_hierarchical
```

#### Masked CE variant (hierarchy-consistent training)

```bash
python ultralytics/models/yolo/classify/train_hierarchical_masked.py \
    --data /path/to/dataset \
    --mapping_csv /path/to/mapping.csv \
    --model yolov8s-cls.pt \
    --epochs 100 \
    --output outputs_hierarchical_masked
```

Same CLI as baseline. At each step, family/genus/species logits are masked to only those whose **ground-truth parent** matches the sample's GT parent label, then cross-entropy is computed. Validation still uses independent argmax so metrics are directly comparable with the baseline.

Checkpoint from this variant has an extra field `"masked_training": True`.

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to dataset directory |
| `--mapping_csv` | required | Path to taxonomy mapping CSV |
| `--model` | `yolov8s-cls.pt` | Base YOLO classification model |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--imgsz` | 224 | Input image size |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 0.05 | Weight decay |
| `--freeze-epochs` | 3 | Epochs to freeze backbone |
| `--patience` | 20 | Early stopping patience (0 to disable) |
| `--output` | `outputs_yolo_hierarchical` | Output directory |

### Loss Weights

The hierarchical loss combines cross-entropy losses from each level with configurable weights:

```python
LAMBDA = {
    "ordor": 1.0,    # Order level weight
    "familia": 0.9,  # Family level weight
    "genus": 0.8,    # Genus level weight
    "species": 0.7,  # Species level weight
}
```

Total loss: `L = λ_o × L_order + λ_f × L_family + λ_g × L_genus + λ_s × L_species`

## Output

After training, the following files are saved to the output directory:

```
outputs_hierarchical/
├── best_model.pt          # Best checkpoint (by hierarchical accuracy)
├── species_to_id.json     # Species name to ID mapping
├── train_expanded.csv     # Training data with all labels
└── val_expanded.csv       # Validation data with all labels
```

### Checkpoint Contents

```python
checkpoint = {
    "model_state": model.state_dict(),
    "base_model": "yolov8s-cls.pt",
    "species2id": {...},
    "familia_parent": [...],
    "genus_parent": [...],
    "species_parent": [...],
    "lambdas": {...},
    "n_classes": {"ordor": N, "familia": M, "genus": K, "species": L}
}
```

## Inference

```python
import torch
from ultralytics import YOLO

# Load checkpoint
ckpt = torch.load("outputs_hierarchical/best_model.pt")

# Recreate model
from ultralytics.models.yolo.classify.train_hierarchical import YOLOHierarchicalClassifier

n_classes = ckpt["n_classes"]
model = YOLOHierarchicalClassifier(
    base_model_path=ckpt["base_model"],
    n_ordor=n_classes["ordor"],
    n_familia=n_classes["familia"],
    n_genus=n_classes["genus"],
    n_species=n_classes["species"]
)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Inference
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("test_image.jpg").convert("RGB")
x = transform(img).unsqueeze(0)

with torch.no_grad():
    order_logits, family_logits, genus_logits, species_logits = model(x)
    order_pred = order_logits.argmax(1).item()
    family_pred = family_logits.argmax(1).item()
    genus_pred = genus_logits.argmax(1).item()
    species_pred = species_logits.argmax(1).item()

print(f"Order: {order_pred}, Family: {family_pred}, Genus: {genus_pred}, Species: {species_pred}")
```

## Modified Files

This fork modifies the following files from the original Ultralytics repository:

| File | Changes |
|------|---------|
| `ultralytics/nn/modules/head.py` | Added `HierarchicalClassify` class |
| `ultralytics/nn/modules/__init__.py` | Export `HierarchicalClassify` |
| `ultralytics/nn/tasks.py` | Import hierarchical modules |
| `ultralytics/utils/loss.py` | Added `HierarchicalClassificationLoss` |

New files:
- `ultralytics/models/yolo/classify/train_hierarchical.py` — Training script (baseline, independent CE)
- `ultralytics/models/yolo/classify/train_hierarchical_masked.py` — Training script (masked CE variant)

## Metrics

The training script reports the following metrics:

- **Order Accuracy**: Classification accuracy at the Order level
- **Family Accuracy**: Classification accuracy at the Family level
- **Genus Accuracy**: Classification accuracy at the Genus level
- **Species Accuracy**: Classification accuracy at the Species level
- **Hierarchical Accuracy**: Percentage of samples where ALL levels are correctly predicted

### Hierarchy-consistent decoding at inference

[`yolo*-trained/test/test_yolo.py`](ultralytics/models/yolo/classify/yolov8s-trained/test/test_yolo.py) supports `--masked` flag that decodes top-down: predict order → mask families to those whose parent matches → argmax → repeat for genus and species. Combine with the masked-trained checkpoint for full hierarchy consistency at both train and inference time.

### Ablation table

The two training scripts × `--masked` inference flag give 4 configurations. Run them on the same test set for the ablation table:

| Training | Inference | Command |
|---|---|---|
| Independent CE | Independent argmax | `train_hierarchical.py` → `test_yolo.py` |
| Independent CE | Masked decoding | `train_hierarchical.py` → `test_yolo.py --masked` |
| Masked CE | Independent argmax | `train_hierarchical_masked.py` → `test_yolo.py` |
| Masked CE | Masked decoding | `train_hierarchical_masked.py` → `test_yolo.py --masked` |

## Requirements

- Python >= 3.8
- PyTorch >= 1.8
- ultralytics (this repo)
- pandas
- scikit-learn
- PIL/Pillow
- torchvision

## Citation

If you use this work, please cite:

```bibtex
@software{yolo_hierarchical_pollen,
  title = {YOLO Hierarchical Classification for Pollen Taxonomy},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-username/yolo-hierarchical-classification}
}
```

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO framework
- Based on hierarchical classification approaches from ViT literature

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ for pollen classification research</sub>
</div>

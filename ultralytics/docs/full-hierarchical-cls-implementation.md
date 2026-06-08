# TOÀN BỘ CÁC THAY ĐỔI CẦN THIẾT ĐỂ HUẤN LUYỆN VÀ DỰ ĐOÁN MÔ HÌNH PHÂN CẤP YOLOv8

## 1. Thêm lớp HierarchicalClassify vào ultralytics/nn/modules/head.py

```python
# Thêm vào cuối file head.py
class HierarchicalClassify(nn.Module):
    """YOLO hierarchical classification head with multiple levels."""

    export = False

    def __init__(self, c1: int, n_order: int, n_family: int, n_genus: int, n_species: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize hierarchical classification head."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)

        # Multiple classification heads for hierarchical levels
        self.head_order = nn.Linear(c_, n_order)
        self.head_family = nn.Linear(c_, n_family)
        self.head_genus = nn.Linear(c_, n_genus)
        self.head_species = nn.Linear(c_, n_species)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> tuple:
        """Perform forward pass of the hierarchical classification model."""
        if isinstance(x, list):
            x = torch.cat(x, 1)

        # Shared feature extraction
        features = self.pool(self.conv(x)).flatten(1)
        features = self.drop(features)

        # Predict for each hierarchical level
        order_pred = self.head_order(features)
        family_pred = self.head_family(features)
        genus_pred = self.head_genus(features)
        species_pred = self.head_species(features)

        if self.training:
            return (order_pred, family_pred, genus_pred, species_pred)

        # For inference, return probabilities for each level
        order_probs = order_pred.softmax(1)
        family_probs = family_pred.softmax(1)
        genus_probs = genus_pred.softmax(1)
        species_probs = species_pred.softmax(1)

        return (order_probs, family_probs, genus_probs, species_probs) if self.export else \
               ((order_probs, family_probs, genus_probs, species_probs), (order_pred, family_pred, genus_pred, species_pred))
```

## 2. Tạo file cấu hình yolov8-hierarchical-cls.yaml

```yaml
# Ultralytics YOLOv8-hierarchical-cls image classification model with YOLO backbone
# Model docs: https://docs.ultralytics.com/models/yolov8
# Task docs: https://docs.ultralytics.com/tasks/classify

# Parameters
nc: 1000 # number of classes (will be updated during training)
scales: # model compound scaling constants, i.e. 'model=yolov8n-cls.yaml' will call yolov8-cls.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 1024]
  l: [1.00, 1.00, 1024]
  x: [1.00, 1.25, 1024]

# YOLOv8.0n backbone
backbone:
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]

# Hierarchical head
head:
  - [-1, 1, HierarchicalClassify, [nc, n_order, n_family, n_genus, n_species]] # Hierarchical classification
```

## 3. Cập nhật file ultralytics/utils/loss.py

```python
# Thêm vào cuối file loss.py
class HierarchicalClassificationLoss:
    """Hierarchical classification loss function."""

    def __init__(self, lambdas=None):
        """Initialize hierarchical classification loss."""
        self.lambdas = lambdas or {
            'order': 1.0,
            'family': 0.9,
            'genus': 0.8,
            'species': 0.7
        }

    def __call__(self, preds, targets):
        """Compute hierarchical classification loss."""
        order_pred, family_pred, genus_pred, species_pred = preds
        order_labels = targets['order']
        family_labels = targets['family']
        genus_labels = targets['genus']
        species_labels = targets['species']

        # Calculate individual losses
        order_loss = F.cross_entropy(order_pred, order_labels)
        family_loss = F.cross_entropy(family_pred, family_labels)
        genus_loss = F.cross_entropy(genus_pred, genus_labels)
        species_loss = F.cross_entropy(species_pred, species_labels)

        # Combine losses with weights
        total_loss = (
            self.lambdas['order'] * order_loss +
            self.lambdas['family'] * family_loss +
            self.lambdas['genus'] * genus_loss +
            self.lambdas['species'] * species_loss
        )

        return total_loss
```

## 4. Cập nhật file ultralytics/nn/tasks.py

```python
# Thêm vào cuối file tasks.py, sau lớp ClassificationModel
class HierarchicalClassificationModel(BaseModel):
    """YOLO hierarchical classification model."""

    def __init__(self, cfg="yolo26n-hierarchical-cls.yaml", ch=3, nc=None, verbose=True):
        """Initialize HierarchicalClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set Ultralytics YOLO model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    def init_criterion(self):
        """Initialize the loss criterion for the HierarchicalClassificationModel."""
        return HierarchicalClassificationLoss()
```

## 5. Cập nhật file ultralytics/models/yolo/classify/train.py

```python
# Thêm vào đầu file sau import
import pandas as pd
import os

# Trong class ClassificationTrainer, thêm phương thức mới:
def build_hierarchical_dataset(self, img_path: str, mode: str = "train", batch=None):
    """Create a hierarchical classification dataset instance."""
    # Load mapping file if specified
    mapping_file = getattr(self.args, 'mapping_file', None)

    # Create dataset with mapping support
    dataset = ClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

    # If mapping file is provided, add hierarchical labels
    if mapping_file and os.path.exists(mapping_file):
        # Load mapping data
        mapping_df = pd.read_csv(mapping_file)
        # Store mapping for later use
        self.mapping_data = mapping_df

    return dataset

# Thay đổi phương thức build_dataset:
def build_dataset(self, img_path: str, mode: str = "train", batch=None):
    """Create a ClassificationDataset instance given an image path and mode."""
    return self.build_hierarchical_dataset(img_path, mode)
```

## 6. Cập nhật file ultralytics/models/yolo/classify/predict.py

```python
# Thêm vào đầu file sau import
import torch.nn.functional as F

# Trong class ClassificationPredictor, cập nhật phương thức postprocess:
def postprocess(self, preds, img, orig_imgs):
    """Process predictions to return Results objects with hierarchical classification probabilities."""
    if not isinstance(orig_imgs, list):  # Input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

    # Handle hierarchical predictions
    if isinstance(preds, (tuple, list)) and len(preds) >= 4:
        # Hierarchical predictions: (order_probs, family_probs, genus_probs, species_probs)
        order_probs, family_probs, genus_probs, species_probs = preds

        # For hierarchical classification, we return all probabilities
        # Create a custom results object that can handle hierarchical output
        results = []
        for i, (orig_img, img_path) in enumerate(zip(orig_imgs, self.batch[0])):
            # Combine all probabilities into a single object
            hierarchical_probs = {
                'order': order_probs[i] if len(order_probs) > i else None,
                'family': family_probs[i] if len(family_probs) > i else None,
                'genus': genus_probs[i] if len(genus_probs) > i else None,
                'species': species_probs[i] if len(species_probs) > i else None
            }

            # Create results object
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, probs=hierarchical_probs)
            )
        return results
    else:
        # Fallback to standard classification
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]
```

## 7. Cấu trúc thư mục dữ liệu cần thiết

```
dataset/
├── train/
│   ├── abel/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── ager/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── ...
├── val/
│   ├── abel/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── ...
└── mapping.csv
```

## 8. File mapping.csv mẫu

```
class_name,ordor,familia,genus
abel,2,2,3
ager,1,1,1
bide,1,1,2
bras,4,8,13
chro,1,1,5
chro_1,1,1,5
chro_2,1,1,5
cler,3,5,9
cler_1,3,5,9
clin,3,7,15
ipom,5,9,14
ipom_1,5,9,14
leuc,3,3,7
ruel,3,7,12
ruel_1,3,7,12
ruel_2,3,7,12
sige,1,1,6
sige_1,1,1,6
teco,3,4,8
tith,1,1,11
tore,3,6,10
trid,1,1,4
```

## 9. Cách sử dụng mô hình

### Huấn luyện:
```bash
yolo classify train \
    model=yolov8-hierarchical-cls.yaml \
    data=dataset \
    epochs=100 \
    imgsz=224 \
    batch=32 \
    mapping_file=mapping.csv \
    project=outputs \
    name=hierarchical_cls
```

### Dự đoán:
```bash
yolo classify predict \
    model=outputs/hierarchical_cls/weights/best.pt \
    source=image.jpg \
    project=outputs \
    name=predictions
```

### Đánh giá:
```bash
yolo classify val \
    model=outputs/hierarchical_cls/weights/best.pt \
    data=dataset \
    project=outputs \
    name=validation
```

## 10. Cách truy cập kết quả phân cấp trong code

```python
from ultralytics import YOLO

# Load model
model = YOLO("outputs/hierarchical_cls/weights/best.pt")

# Predict
results = model("image.jpg")

# Access hierarchical probabilities
hierarchical_probs = results[0].probs  # Đây là dictionary chứa các xác suất cho từng cấp độ
order_probs = hierarchical_probs['order']
family_probs = hierarchical_probs['family']
genus_probs = hierarchical_probs['genus']
species_probs = hierarchical_probs['species']

# Truy xuất lớp có xác suất cao nhất
order_pred = order_probs.argmax().item()
family_pred = family_probs.argmax().item()
genus_pred = genus_probs.argmax().item()
species_pred = species_probs.argmax().item()
```

## 11. Cài đặt và chạy thử nghiệm

1. **Cài đặt các thư viện cần thiết**:
```bash
pip install pandas torch torchvision
```

2. **Chạy thử nghiệm huấn luyện**:
```bash
yolo classify train \
    model=yolov8-hierarchical-cls.yaml \
    data=dataset \
    epochs=5 \
    imgsz=224 \
    batch=8 \
    mapping_file=mapping.csv
```

3. **Kiểm tra kết quả dự đoán**:
```bash
yolo classify predict \
    model=outputs/hierarchical_cls/weights/best.pt \
    source=dataset/val/abel/image1.jpg
```

## 12. Lưu ý quan trọng

- Các file cần được chỉnh sửa trong thư mục `ultralytics/` trong project của bạn
- Đảm bảo cấu trúc thư mục dữ liệu đúng như mô tả
- File mapping.csv phải có đầy đủ các cột: class_name, ordor, familia, genus
- Kích thước ảnh nên là 224x224 (mặc định trong YOLOv8)
- Có thể điều chỉnh các tham số như epochs, batch size, learning rate theo nhu cầu thực tế

Với các thay đổi này, bạn sẽ có thể huấn luyện và dự đoán mô hình phân cấp YOLOv8 hoàn chỉnh theo đúng cấu trúc dữ liệu bạn đã cung cấp.
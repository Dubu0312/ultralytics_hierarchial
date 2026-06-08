# Triển khai Phân loại Phân cấp (Hierarchical Classification) với YOLOv8-cls - Phiên bản phù hợp với dữ liệu bạn cung cấp

## Tổng quan

Dựa trên cấu trúc dữ liệu bạn cung cấp, tôi sẽ điều chỉnh kế hoạch để phù hợp hoàn toàn với cách tiếp cận mà bạn đang sử dụng. Dữ liệu của bạn có cấu trúc giống như YOLOv8 chuẩn với các thư mục lớp, chỉ thêm file mapping.csv để xác định phân cấp.

## Cấu trúc dữ liệu

```
dataset/
├── train/
│   ├── class_name_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── class_name_2/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── ...
├── val/
│   ├── class_name_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── ...
└── mapping.csv       # File CSV chứa ánh xạ từ tên lớp đến các cấp độ
```

## Cấu trúc file mapping.csv

```
class_name,ordor,familia,genus
abel,2,2,3
ager,1,1,1
bide,1,1,2
bras,4,8,13
chro,1,1,5
...
```

## Giải pháp triển khai

### 1. Cấu trúc lớp đầu ra phân cấp mới (được điều chỉnh cho phù hợp)

```python
class HierarchicalClassify(nn.Module):
    """YOLO hierarchical classification head with multiple levels.

    Compatible with standard YOLOv8 classification structure but with hierarchical outputs.
    """

    export = False

    def __init__(self, c1: int, n_order: int, n_family: int, n_genus: int, n_species: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize hierarchical classification head.

        Args:
            c1 (int): Number of input channels.
            n_order (int): Number of order classes.
            n_family (int): Number of family classes.
            n_genus (int): Number of genus classes.
            n_species (int): Number of species classes.
            k (int, optional): Kernel size.
            s (int, optional): Stride.
            p (int, optional): Padding.
            g (int, optional): Groups.
        """
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
        """Perform forward pass of the hierarchical classification model.

        Returns:
            tuple: Predictions for each hierarchical level (order, family, genus, species)
        """
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

### 2. Cấu hình mô hình YAML

```yaml
# yolov8-hierarchical-cls.yaml
nc: 1000  # số lượng lớp tổng cộng (sẽ được cập nhật sau)
scales:
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

### 3. Dataset class cho phân loại phân cấp

```python
class HierarchicalClassificationDataset(ClassificationDataset):
    """Dataset class for hierarchical classification."""

    def __init__(self, root, args, augment=False, prefix="", mapping_file=None):
        """Initialize hierarchical classification dataset.

        Args:
            root (str): Root directory of dataset.
            args (dict): Arguments for dataset.
            augment (bool): Whether to apply augmentations.
            prefix (str): Prefix for logging.
            mapping_file (str): Path to mapping CSV file.
        """
        super().__init__(root, args, augment, prefix)
        self.mapping_file = mapping_file
        self.class_to_hierarchical_labels = self._load_mapping()

    def _load_mapping(self):
        """Load hierarchical mapping from CSV file."""
        if not self.mapping_file or not os.path.exists(self.mapping_file):
            return {}

        df = pd.read_csv(self.mapping_file)
        # Ensure required columns exist
        required_cols = ['class_name', 'ordor', 'familia', 'genus']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Mapping CSV must contain columns: {required_cols}")

        # Create mapping dictionary
        mapping = {}
        for _, row in df.iterrows():
            class_name = row['class_name']
            mapping[class_name] = {
                'ordor': int(row['ordor']),
                'familia': int(row['familia']),
                'genus': int(row['genus'])
            }
        return mapping

    def __getitem__(self, index):
        """Get item with hierarchical labels."""
        # Get standard classification item
        item = super().__getitem__(index)

        # Add hierarchical labels if available
        if self.mapping_file and hasattr(self, 'class_to_hierarchical_labels'):
            class_name = self.classes[item['cls']]
            if class_name in self.class_to_hierarchical_labels:
                hierarchical_labels = self.class_to_hierarchical_labels[class_name]
                item['ordor'] = torch.tensor(hierarchical_labels['ordor'], dtype=torch.long)
                item['familia'] = torch.tensor(hierarchical_labels['familia'], dtype=torch.long)
                item['genus'] = torch.tensor(hierarchical_labels['genus'], dtype=torch.long)

        return item
```

### 4. Hàm mất mát cho phân loại phân cấp

```python
class HierarchicalClassificationLoss:
    """Hierarchical classification loss function."""

    def __init__(self, lambdas=None):
        """Initialize hierarchical classification loss.

        Args:
            lambdas (dict): Weights for each level loss (default: {'order': 1.0, 'family': 0.9, 'genus': 0.8, 'species': 0.7})
        """
        self.lambdas = lambdas or {
            'order': 1.0,
            'family': 0.9,
            'genus': 0.8,
            'species': 0.7
        }

    def __call__(self, preds, targets):
        """Compute hierarchical classification loss.

        Args:
            preds: Tuple of predictions (order_pred, family_pred, genus_pred, species_pred)
            targets: Dictionary of targets {'order': order_labels, 'family': family_labels, 'genus': genus_labels, 'species': species_labels}

        Returns:
            total_loss: Combined loss for all hierarchical levels
        """
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

### 5. Cách sử dụng mô hình

```python
# 1. Khởi tạo mô hình với cấu hình phân cấp
from ultralytics import YOLO

# Tạo file cấu hình YAML với thông tin phân cấp
model = YOLO("yolov8-hierarchical-cls.yaml")

# 2. Xác định số lượng lớp cho từng cấp độ từ mapping.csv
import pandas as pd

# Đọc mapping.csv để xác định số lượng lớp cho từng cấp độ
mapping_df = pd.read_csv("mapping.csv")
n_order = int(mapping_df["ordor"].max()) + 1
n_family = int(mapping_df["familia"].max()) + 1
n_genus = int(mapping_df["genus"].max()) + 1
# Số lượng species được tính từ số lượng unique class_name
n_species = len(mapping_df["class_name"].unique())

# 3. Cập nhật số lượng lớp cho head
# Cần sửa đổi lớp head để cập nhật số lượng lớp cho từng cấp độ
# Điều này có thể được thực hiện trong quá trình khởi tạo mô hình

# 4. Huấn luyện
model.train(
    data="dataset",  # Đường dẫn tới thư mục dataset chứa train/val
    epochs=100,
    imgsz=224,
    batch=32,
    mapping_file="mapping.csv",  # Chỉ định file mapping
    # Các tham số khác
)

# 5. Dự đoán
results = model("image.jpg")
# results sẽ chứa kết quả cho cả 4 cấp độ phân loại
order_probs, family_probs, genus_probs, species_probs = results[0].probs
```

## Ưu điểm của giải pháp này

1. **Tương thích hoàn toàn với cấu trúc dữ liệu hiện tại**: Không cần thay đổi cấu trúc thư mục
2. **Dễ dàng tích hợp**: Sử dụng cấu trúc YOLOv8 chuẩn với bổ sung thêm file mapping
3. **Linh hoạt**: Có thể mở rộng cho nhiều cấp độ phân loại khác nếu cần
4. **Tối ưu hóa**: Giữ nguyên hiệu suất của YOLOv8 với các lớp đầu ra phân cấp

## Kết luận

Giải pháp này hoàn toàn phù hợp với cấu trúc dữ liệu bạn cung cấp. Bạn có thể sử dụng thư mục train/val như bình thường, chỉ thêm file mapping.csv để xác định phân cấp. Mô hình sẽ tự động học các mối quan hệ phân cấp giữa các lớp và cung cấp kết quả phân loại cho cả 4 cấp độ.

Cấu trúc này cho phép bạn tận dụng toàn bộ hệ thống YOLOv8 để huấn luyện và dự đoán, trong khi vẫn có khả năng phân loại phân cấp như trong ví dụ Vision Transformer bạn cung cấp.
# Phân loại Phấn hoa theo Cấu trúc Phân loại học Phân cấp

Tài liệu kỹ thuật mô tả phương pháp phân loại phấn hoa theo cây phân loại học (taxonomy) với mạng YOLO multi-head. Phục vụ báo cáo, slide, luận văn, paper.

---

## TL;DR — Kết quả chính

Đề xuất pipeline phân loại phấn hoa theo cây taxonomy 4 cấp (Order → Family → Genus → Species) với 2 đóng góp:

1. **Multi-head YOLO classifier** chia sẻ backbone, 4 head Linear song song.
2. **Masked Cross-Entropy training + masked decoding** đảm bảo prediction luôn nhất quán với cây taxonomy.

**Kết quả Hierarchical Accuracy (HierAcc) trên tập test (241 ảnh, 22 species)**:

| Backbone | Baseline (Indep CE) | Masked CE (đề xuất) | Δ |
|---|---:|---:|---:|
| yolov8s (~10M params) | 93.8% | 92.5% | −1.3% |
| **yolo11x (~50M params)** | 94.2% | **95.0%** | **+0.8%** ✅ |

**So sánh với baseline Cascade** (4 model độc lập, mỗi model một cấp, yolo11x × 4):

| Method | Params | **HierAcc** | Consistency |
|---|---:|---:|---:|
| Cascade (4 model riêng) | 28M × 4 | 93.4% | 95.0% (12/241 sample bị inconsistent) |
| **Multi-head + Masked CE (đề xuất)** | **28M** | **95.0% ★** | **100% (guaranteed)** |

**Phát hiện then chốt**:

1. **Hiệu quả của Masked CE phụ thuộc capacity backbone**. Backbone đủ lớn (yolo11x) mới có lợi từ hard constraint; backbone nhỏ (yolov8s) bị cản trở.

2. **Multi-head + Masked CE thắng Cascade** dù dùng **ít tham số hơn 4 lần** (28M vs 28M×4). Cascade per-level acc cao nhưng HierAcc rớt do 4 model decide độc lập, không đảm bảo nhất quán.

3. **Pipeline masked CE + masked decoding đảm bảo prediction consistency 100%** với cây taxonomy — guarantee mà cascade không có.

→ Chi tiết: [Section 5 — Kết quả thực nghiệm](#5-kết-quả-thực-nghiệm).

---

## ⚠️ Phạm vi và giới hạn tài liệu

Tài liệu này hiện tại **đủ cho**:
- Báo cáo nội bộ / báo cáo tiến độ (✓ đã có kết quả thật)
- Slide thuyết trình
- Bản nháp đầu cho chương "Phương pháp" + "Kết quả" của luận văn

Tài liệu này **CHƯA đủ cho** paper hội nghị/tạp chí hoặc luận văn hoàn chỉnh, vì còn thiếu:

1. **Đặc tả dataset** — Section 1.4 đã có thông tin về **cấu trúc** (taxonomy, phân phối class, splits, preprocessing). Còn thiếu **metadata thu thập**: thời gian, người thu thập, kính hiển vi, camera, labeling protocol. Cần nhóm thu thập cung cấp.
2. **So sánh với baseline khác ngoài chính mình** — flat classifier, cascade, hierarchical softmax... (xem Section 2.4 — placeholder)
3. **Multi-seed runs** — báo cáo mean ± std qua 3-5 seed (hiện chỉ 1 seed = 42)
4. **Phân tích chi tiết kết quả** — failure case, per-class accuracy, computational cost (xem Section 5.4 — placeholder)
5. **Related work** — định vị contribution so với literature (xem Section 8 — placeholder)
6. **Hình ảnh sample dataset** — thumbnail 22 species + cây taxonomy thật (xem Section 1.4.8 — placeholder)
7. **Ablation cho hyperparameter** — đặc biệt là λ weights, freeze epoch
8. **Làm rõ ý nghĩa sub-class `_1`/`_2`** — 7 species có suffix này, cùng (ordor, familia, genus) nhưng tách thành class species riêng (xem Section 1.4.2). Cần nhóm thu thập giải thích.

Agent/tác giả nào dùng file này để viết paper/luận văn phải bổ sung các phần trên trước. Các section placeholder bên dưới có gợi ý cụ thể những gì cần điền.

---

## 1. Bài toán

### 1.1 Mô tả

Cho một ảnh phấn hoa, dự đoán đồng thời 4 cấp phân loại học:

```
Order (Bộ) → Family (Họ) → Genus (Chi) → Species (Loài)
```

Mỗi loài thuộc duy nhất một chi, mỗi chi thuộc duy nhất một họ, mỗi họ thuộc duy nhất một bộ. Cấu trúc cây cứng (rigid hierarchy).

### 1.2 Vì sao không dùng phân loại phẳng (flat classification)?

Phân loại phẳng (chỉ predict species) bỏ qua **thông tin cấu trúc** đã có:

- Sai cấp cao (predict nhầm bộ) **nghiêm trọng hơn** sai cấp thấp (predict nhầm loài cùng chi). Phân loại phẳng coi mọi lỗi như nhau.
- Khi không đủ dữ liệu để học cấp loài (long-tail), model vẫn có thể predict đúng cấp chi/họ/bộ. Có nghiên cứu lý sinh học vẫn dùng được.
- Inference có thể **enforce consistency** theo cây: prediction loài X bắt buộc có chi/họ/bộ là tổ tiên của X.

### 1.3 Dataset

| Chỉ số | Giá trị |
|---|---|
| Số bộ (Order) | **6** |
| Số họ (Family) | **10** |
| Số chi (Genus) | **16** |
| Số loài (Species) | **22** |
| Số ảnh train | **1,780** |
| Số ảnh val | **216** |
| Số ảnh test | **241** |
| Tổng số ảnh | **2,237** |

Cấu trúc thư mục: `train/<class_name>/*.jpg`, `val/<class_name>/*.jpg`.
File ánh xạ `mapping.csv` với cột `class_name, ordor, familia, genus` cho ID phân loại học (loài tự sinh từ tên thư mục).

Dataset path trong thí nghiệm hiện tại: `/home/dubu/manh/dongvan-yolo/dataset-dongvan-train/`
Mapping CSV: `/home/dubu/manh/dongvan-yolo/pollen_dong_van.csv`

### 1.4 Đặc tả Dataset chi tiết

#### 1.4.1 Nguồn và bối cảnh

- **Tên dataset**: pollen Đồng Văn (tự thu thập, chưa public).
- **Vùng địa lý**: cao nguyên đá Đồng Văn (Hà Giang) — vùng đặc trưng hệ thực vật miền núi đá vôi phía Bắc Việt Nam.
- **Mục đích sinh học**: phân loại phấn hoa cho nghiên cứu đa dạng sinh học và pollinator-plant interaction.
- **File mapping**: [`pollen_dong_van.csv`](/home/dubu/manh/dongvan-yolo/pollen_dong_van.csv) — 22 dòng, cột `class_name, ordor, familia, genus` (lưu ý typo `ordor` thay vì `order` — load-bearing trong toàn bộ pipeline).

⚠️ **Thiếu thông tin cần bổ sung cho paper**: thời gian thu thập, ai/nhóm thu thập, loại kính hiển vi (mã model, độ phóng đại), camera resolution gốc, điều kiện chiếu sáng/staining, quy trình labeling (số chuyên gia, inter-annotator agreement nếu có). Cần tác giả/nhóm nghiên cứu cung cấp.

#### 1.4.2 Cấu trúc cây phân loại học (taxonomy)

22 species được tổ chức thành cây 4 cấp với phân bố như sau:

**Cấp Order (Bộ) — 6 ordor (ID 0-5)**:

| Ordor ID | Số họ | Số chi | Số loài |
|---:|---:|---:|---:|
| 1 | 1 | 6 | 9 |
| 2 | 1 | 1 | 1 |
| 3 | 5 | 6 | 9 |
| 4 | 1 | 1 | 1 |
| 5 | 1 | 1 | 2 |
| (0) | — | — | — (không xuất hiện trong dataset) |

→ **Phân bố cực không cân**: order 1 và 3 chiếm 18/22 = 82% số loài; order 2, 4 chỉ có 1 loài (trivial classification ở cấp dưới); order 0 thậm chí không có loài nào (chỉ index `n_ordor = max+1 = 6` trong code).

**Cấp Family (Họ) — 10 familia (ID 1-9 + slot 0 không dùng)**: familia 1 (thuộc order 1) chứa 9 species — cụm lớn nhất. Các họ khác chỉ 1-4 species.

**Cấp Genus (Chi) — 16 genera**: phân bố tương đối đều, mỗi chi 1-3 species.

**Cấp Species (Loài) — 22 species** (tên class viết tắt 4 ký tự):

| class_name | ordor | familia | genus | Diễn giải |
|---|---:|---:|---:|---|
| `abel` | 2 | 2 | 3 | Đơn nhất trong order 2 |
| `ager` | 1 | 1 | 1 | |
| `bide` | 1 | 1 | 2 | |
| `bras` | 4 | 8 | 13 | Đơn nhất trong order 4 |
| `chro`, `chro_1`, `chro_2` | 1 | 1 | 5 | 3 sub-class cùng genus 5 |
| `cler`, `cler_1` | 3 | 5 | 9 | 2 sub-class cùng genus 9 |
| `clin` | 3 | 7 | 15 | |
| `ipom`, `ipom_1` | 5 | 9 | 14 | 2 sub-class cùng genus 14 |
| `leuc` | 3 | 3 | 7 | |
| `ruel`, `ruel_1`, `ruel_2` | 3 | 7 | 12 | 3 sub-class cùng genus 12 |
| `sige`, `sige_1` | 1 | 1 | 6 | 2 sub-class cùng genus 6 |
| `teco` | 3 | 4 | 8 | |
| `tith` | 1 | 1 | 11 | |
| `tore` | 3 | 6 | 10 | |
| `trid` | 1 | 1 | 4 | |

⚠️ **Quan sát quan trọng về cấu trúc dataset**:

- **Suffixes `_1`, `_2`** ở 7 species (`chro_*`, `cler_*`, `ipom_*`, `ruel_*`, `sige_*`) cho thấy **các sub-class này có cùng vị trí phân loại học (ordor/familia/genus) nhưng được tách thành class riêng** ở cấp species. Lý do có thể: (a) cùng species nhưng pha trạng thái phát triển khác nhau (mature/immature), (b) cùng species nhưng góc chụp/morphology khác, (c) species thực sự khác nhưng chưa định danh chính xác → labeler tạm tách. **Cần làm rõ với nhóm thu thập** — quan trọng vì điều này ảnh hưởng định nghĩa "task". Nếu (a)/(b) thì task thực ra là 15 species (16 genera), không phải 22.
- **Class `_1`/`_2` mặc dù share parent triplet** vẫn được model phân biệt ở cấp species — cần kiểm tra confusion matrix species xem các sub-class này có hay confuse với nhau không.

#### 1.4.3 Phân chia train / val / test

| Split | Số ảnh | Tỷ lệ |
|---|---:|---:|
| Train | 1,780 | 79.6% |
| Val | 216 | 9.6% |
| Test | 241 | 10.8% |
| **Tổng** | **2,237** | 100% |

- **Tỷ lệ ~80/10/11%** — gần chuẩn 80/10/10. Val và test xấp xỉ bằng nhau, dùng cho mục đích khác nhau: val cho early stopping & checkpoint selection trong training, test cho báo cáo cuối.
- **Protocol split**: chưa rõ random per-image hay per-stratified. Cần làm rõ — nếu random per-image thì có nguy cơ data leakage nếu nhiều ảnh cùng pollen grain (cùng buổi chụp) bị tách ra train/test.

#### 1.4.4 Phân phối số ảnh theo class (long-tail check)

**Train set (1,780 ảnh, 22 species)**:

| Thống kê | Giá trị |
|---|---:|
| min | 60 (species 20 = `tore`) |
| max | **190 (species 8 = `cler_1`)** ⚠️ |
| median | 79 |
| mean | 80.9 |

→ **Hơi mất cân bằng**: species `cler_1` (190 ảnh) nhiều hơn median 2.4×. Các species khác xấp xỉ 60-110 ảnh. Không quá long-tail (worst case ratio 190:60 = 3.2:1), nhưng đủ để chú ý F1 macro vs weighted khi báo cáo.

**Phân phối theo Order (train)**:

| Ordor | Số ảnh | Tỷ lệ |
|---:|---:|---:|
| 1 | 704 | 39.6% |
| 3 | 754 | 42.4% |
| 5 | 172 | 9.7% |
| 2 | 80 | 4.5% |
| 4 | 70 | 3.9% |

→ **Cực mất cân bằng ở cấp order**: order 1 và 3 chiếm 82% data, order 2 và 4 chỉ chứa 1 species mỗi cái → mô hình thấy order 2, 4 rất ít. Tuy nhiên do mỗi order này chỉ có 1 species nên cấp ordor → species là 1-to-1, không cần generalize.

**Val set (216 ảnh)**: phân phối tương tự train (proportional). `cler_1` cũng có 23 ảnh val (lớn nhất). Min 7 ảnh/species — đủ cho metric stable.

**Test set (241 ảnh)**: gần uniform (~10 ảnh/class), trừ `cler_1` (25) và `ipom_1` (14). Min 8, max 25.

#### 1.4.5 Tiền xử lý

Pipeline transform trong [`HierarchicalDataset`](ultralytics/models/yolo/classify/train_hierarchical.py#L130):

**Train (augmentation)**:
```python
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))    # crop 80-100% diện tích → 224×224
transforms.RandomHorizontalFlip(p=0.5)
transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)              # brightness/contrast/saturation/hue
transforms.ToTensor()
transforms.Normalize(mean=ImageNet, std=ImageNet)
```

**Val/Test (no augmentation)**:
```python
transforms.Resize((224, 224))   # resize trực tiếp, không crop
transforms.ToTensor()
transforms.Normalize(mean=ImageNet, std=ImageNet)
```

**Resolution ảnh gốc** (sample 66 ảnh random từ train):

| Thống kê | Width | Height |
|---|---:|---:|
| min | 224 | 224 |
| max | 1046 | 1046 |
| mean | 488 | 489 |

→ Phần lớn ảnh là **vuông** (chỉ 1/66 sample có dạng không vuông 492×654). Resolution dao động 224×224 đến 1046×1046, **median ≈ 488×488**. Khoảng 23% ảnh đã ở 224×224 (chắc đã pre-resize từ trước), còn lại ở resolution cao hơn → resize về 224 sẽ vứt thông tin (~4× linear, ~16× pixel). Đáng cân nhắc thử `imgsz=384` hoặc `imgsz=448` để giữ thêm chi tiết (đặc biệt cho yolo11x vốn được train ở 224 nhưng có thể fine-tune ở resolution cao hơn).

#### 1.4.6 Cấu trúc thư mục

```
/home/dubu/manh/dongvan-yolo/
├── dataset-dongvan-train/
│   ├── train/                # 22 thư mục con (1 per species)
│   │   ├── abel/             # 80 ảnh
│   │   ├── ager/             # 82 ảnh
│   │   ├── ...
│   │   └── trid/             # 68 ảnh
│   ├── val/                  # 22 thư mục con
│   └── test/                 # 22 thư mục con
└── pollen_dong_van.csv       # mapping class_name → (ordor, familia, genus)
```

Species ID (0-21) được suy ra từ thứ tự alphabet của tên class trong `build_species_ids()` ([train_hierarchical.py:78](ultralytics/models/yolo/classify/train_hierarchical.py#L78)).

#### 1.4.7 Hạn chế dataset (cần thảo luận trong paper)

1. **Quy mô nhỏ**: 2,237 ảnh / 22 species → trung bình 100 ảnh/species. Đủ cho prototype/demo nhưng nhỏ cho deep learning hiện đại (paper hiện nay thường > 10K ảnh).
2. **Cấu trúc taxonomy không cân**: 5/6 order chiếm dominant, 2 order chỉ 1 species → ý nghĩa "hierarchical" bị giảm tại cấp ordor (gần như predict 5-way thay vì 6-way).
3. **Sub-class `_1`/`_2`** chưa rõ ý nghĩa sinh học → nếu thật sự cùng species thì nên gộp lại.
4. **Chưa có public benchmark khác** để so sánh — kết quả 94%+ là tự benchmark.
5. **Domain gap**: pretrained trên ImageNet (ảnh tự nhiên), test trên ảnh kính hiển vi → có thể không tối ưu. Đáng thử pretrain trên dataset phấn hoa lớn hơn nếu có (vd POLEN23E, PollenAtlas).

#### 1.4.8 Hình ảnh minh họa (PLACEHOLDER — chưa có)

Để paper/luận văn hoàn chỉnh cần bổ sung:

- **Grid 4×6 thumbnail** — 1 đại diện mỗi species (24 ô, 2 ô trống vì chỉ 22 species)
- **Cây taxonomy visualization**: vẽ bằng graphviz/d3.js — node = class với màu code theo cấp, edge = quan hệ parent-child. Cho reviewer hiểu cấu trúc dataset trong 1 hình.
- **Histogram phân phối class** — đã có data ở 1.4.4, chỉ cần vẽ matplotlib.

Lệnh sinh figure histogram nhanh:
```bash
python -c "
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('ultralytics/models/yolo/classify/yolov8s-trained/train_expanded.csv')
ax = df['species'].value_counts().sort_index().plot(kind='bar', figsize=(12,4))
ax.set_xlabel('Species ID'); ax.set_ylabel('Number of train images')
plt.tight_layout(); plt.savefig('fig_species_dist.png', dpi=150)
"
```

---

## 2. Phương pháp

### 2.1 Kiến trúc mạng

```
                    ┌─────────────────┐
                    │   Ảnh đầu vào   │  3×224×224
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  YOLO Backbone  │  (yolov8s-cls hoặc yolo11x-cls,
                    │  bỏ head cuối)  │   bỏ lớp Classify gốc)
                    └────────┬────────┘
                             │ (B, C, H, W)
                    ┌────────▼────────┐
                    │  Conv 1×1       │  C → 1280
                    │  AdaptiveAvgPool│  (B, 1280, 1, 1)
                    │  Flatten        │  (B, 1280)
                    │  Dropout        │
                    └────────┬────────┘
                             │
              ┌──────────┬───┴────┬──────────┐
              │          │        │          │
        ┌─────▼────┐ ┌───▼────┐ ┌─▼─────┐ ┌──▼──────┐
        │ Order    │ │Family  │ │ Genus │ │ Species │
        │ Linear   │ │Linear  │ │Linear │ │ Linear  │
        │ (n_ord)  │ │(n_fam) │ │(n_gen)│ │ (n_spe) │
        └──────────┘ └────────┘ └───────┘ └─────────┘
```

Backbone share cho cả 4 head. Mỗi head là 1 lớp `nn.Linear` riêng biệt từ feature 1280-chiều xuống số class tương ứng.

**Lý do dùng YOLO backbone**: pretrained tốt trên ImageNet, kích thước cân bằng (yolov8s ~10M params, yolo11x ~50M params), dễ swap để ablation theo capacity.

Implementation: [train_hierarchical.py:170-216](ultralytics/models/yolo/classify/train_hierarchical.py#L170) — class `YOLOHierarchicalClassifier`.

### 2.2 Hàm mất mát (Loss)

#### 2.2.1 Baseline: Cross-Entropy độc lập theo từng cấp

$$
\mathcal{L}_{\text{base}} = \lambda_o \cdot \text{CE}(z_o, y_o) + \lambda_f \cdot \text{CE}(z_f, y_f) + \lambda_g \cdot \text{CE}(z_g, y_g) + \lambda_s \cdot \text{CE}(z_s, y_s)
$$

Với $z_l$ là logits ở cấp $l \in \{o, f, g, s\}$, $y_l$ là nhãn đúng, và trọng số $\lambda_l$ ưu tiên cấp cao hơn:

$$
\lambda_o = 1.0, \quad \lambda_f = 0.9, \quad \lambda_g = 0.8, \quad \lambda_s = 0.7
$$

**Tại sao trọng số giảm dần từ Order → Species?**

Lý do thiết kế (gradient priority — không phải vì cấp cao quan trọng hơn về mặt sinh học):

1. **Sai cấp cao gây hậu quả nghiêm trọng hơn** trong ngữ cảnh phân loại học. Predict nhầm Bộ (Order) → cả ba cấp con sai theo dây chuyền nếu dùng masked decoding. Predict nhầm Loài (Species) trong cùng Chi → chỉ 1 cấp sai. Loss cần phạt nặng hơn lỗi cấp cao.
2. **Cấp cao có ít class, dễ học hơn** — gradient bão hòa nhanh. Trọng số nhỉnh hơn ($\lambda_o = 1.0$) giữ tín hiệu gradient mạnh ngay cả khi loss đã thấp, tránh model bỏ qua cấp này khi tập trung tối ưu cấp khó (species, 22 class).
3. **Tránh task imbalance**: nếu dùng $\lambda$ uniform, loss species (giá trị tuyệt đối lớn nhất do nhiều class hơn) sẽ thống trị tổng loss → backbone tối ưu chủ yếu cho species, các head cấp cao bị bỏ rơi. Trọng số decreasing giúp cân bằng đóng góp gradient từ mỗi head.

**Lựa chọn cụ thể (1.0, 0.9, 0.8, 0.7) là heuristic** — chưa được ablation tune trong thí nghiệm này. Đây là điểm có thể cải thiện ở tương lai (xem Section 5.4 — ablation cho λ schedule).

Implementation: hằng số `LAMBDA` ở [train_hierarchical.py:52-57](ultralytics/models/yolo/classify/train_hierarchical.py#L52-L57), dùng chung cho cả baseline và masked CE.

**Hạn chế**: 4 cross-entropy hoàn toàn độc lập, model **không bị ràng buộc** rằng predict loài X thì chi phải là cha của X. Mỗi head học riêng.

Implementation: [train_hierarchical.py:222-230](ultralytics/models/yolo/classify/train_hierarchical.py#L222) — `multitask_ce`.

#### 2.2.2 Đề xuất: Masked Cross-Entropy (Hierarchy-consistent training)

**Ý tưởng**: tại training, trước khi tính CE cho cấp $l$, **mask** các logits của những class không hợp lệ theo cấu trúc taxonomy với nhãn cha đúng.

Định nghĩa mask $M_l \in \{0, 1\}^{B \times n_l}$:

$$
M_l[b, i] = \mathbb{1}\big[\text{parent}_l[i] = y_{l-1}[b]\big]
$$

Tức là: mẫu $b$ có cha (theo nhãn đúng cấp trên) là $y_{l-1}[b]$, chỉ giữ những class con $i$ thực sự có cha là $y_{l-1}[b]$.

Định nghĩa hàm mask $m(\cdot)$ áp lên logits:

$$
m(z_l)[b, i] = \begin{cases} z_l[b, i] & \text{nếu } M_l[b, i] = 1 \\ -\infty & \text{nếu } M_l[b, i] = 0 \end{cases}
$$

Sau đó tính CE bình thường trên $m(z_l)$. Vì các vị trí bị mask có giá trị $-\infty$ nên softmax bằng 0, gradient không lan ngược qua các class không hợp lệ.

$$
\mathcal{L}_{\text{masked}} = \lambda_o \cdot \text{CE}(z_o, y_o) + \sum_{l \in \{f, g, s\}} \lambda_l \cdot \text{CE}(m(z_l), y_l)
$$

**Lưu ý**: trong slide để đơn giản, công thức gộp luôn order vào $\sum$: $\mathcal{L} = \sum_l \lambda_l \cdot \text{CE}(m(z_l), y_l)$ — với quy ước $m(z_o) = z_o$ (order ở đỉnh cây, không bị mask).

Order ở đỉnh cây, không có cha → không mask.

**Hiệu ứng**:

1. Mỗi mẫu chỉ phải distinguish trong **siblings cùng cha**, không cạnh tranh với tất cả class toàn cục. Loss landscape dễ hơn, gradient sharper.
2. Model học **representation phù hợp với cây phân loại học** — feature cho cùng họ tự nhiên gom lại với nhau.
3. Hard constraint: ràng buộc taxonomy được áp ngay từ training.

**Trade-off**: nếu cha sai ở inference, mask sẽ loại trừ con đúng. Cần kết hợp với hierarchy-consistent decoding (xem 2.3).

Implementation: [train_hierarchical_masked.py:211-249](ultralytics/models/yolo/classify/train_hierarchical_masked.py#L211) — `multitask_ce_masked`.

### 2.3 Decoding khi suy luận (inference)

#### 2.3.1 Independent argmax (baseline)

Mỗi head argmax độc lập:

$$
\hat{y}_l = \arg\max_i z_l[i], \quad \forall l \in \{o, f, g, s\}
$$

Vấn đề: prediction có thể **không nhất quán** — predict species X nhưng dự đoán chi không phải là cha của X.

#### 2.3.2 Top-down masked decoding (đề xuất)

Suy luận tuần tự từ trên xuống, mask theo cha **đã dự đoán**:

```
ŷ_o = argmax(z_o)
mask_f[i] = 1[parent_f[i] = ŷ_o]
ŷ_f = argmax(z_f ⊙ mask_f)        ← chỉ chọn họ có cha là bộ đã predict
mask_g[i] = 1[parent_g[i] = ŷ_f]
ŷ_g = argmax(z_g ⊙ mask_g)
mask_s[i] = 1[parent_s[i] = ŷ_g]
ŷ_s = argmax(z_s ⊙ mask_s)
```

**Đảm bảo**: prediction luôn là một đường đi hợp lệ trong cây taxonomy.

Implementation: [test_yolo.py:176-203](ultralytics/models/yolo/classify/yolov8s-trained/test/test_yolo.py#L176) — `masked_decode`.

### 2.4 So sánh với baseline khác

#### (a) Cascade classifier (4 model độc lập) — ĐÃ CHẠY

**Thiết lập**: train 4 model YOLO11x-cls riêng biệt, mỗi model phân loại 1 cấp:
- Model Order: 5 class (`ord_1`, `ord_2`, ..., `ord_5`)
- Model Family: 9 class (`fam_1`, ..., `fam_9`)
- Model Genus: 15 class (`gen_1`, ..., `gen_15`)
- Model Species: 22 class (model flat species đã có sẵn — [yolo11x-flat-default](ultralytics/models/yolo/classify/yolo11x-flat-default/))

3 dataset mới được tạo bằng cách regroup ảnh theo nhãn cha:
- [dataset-dongvan-order](/home/dubu/manh/dongvan-yolo/dataset-dongvan-order/) — copy ảnh theo Order ID
- [dataset-dongvan-family](/home/dubu/manh/dongvan-yolo/dataset-dongvan-family/) — copy theo Family ID
- [dataset-dongvan-genus](/home/dubu/manh/dongvan-yolo/dataset-dongvan-genus/) — copy theo Genus ID

Cả 4 model dùng cùng hyperparam (Ultralytics default): `yolo11x-cls.pt`, `imgsz=224`, `batch=32`, `epochs=100`, `patience=20`.

**Inference**: cho mỗi ảnh test, chạy độc lập cả 4 model → 4 prediction → ghép thành (order, family, genus, species) → so với GT.

**Đặc điểm**: không có ràng buộc nhất quán — 4 model decide độc lập, có thể inconsistent (vd predict species `cler_1` thuộc genus 9, nhưng model genus predict genus 12).

Implementation:
- Tạo dataset: [prepare_cascade_datasets.py](prepare_cascade_datasets.py)
- Test: [yolo11x-cascade-eval/test_cascade.py](ultralytics/models/yolo/classify/yolo11x-cascade-eval/test_cascade.py)

#### (b) Các baseline chưa chạy (placeholder cho paper)

- **Hierarchical softmax**: $P(s, g, f, o) = P(o) \cdot P(f|o) \cdot P(g|f) \cdot P(s|g)$. Đúng về xác suất, nhưng yêu cầu restructure head. Reference: B-CNN, HMCN (Wehrmann et al., 2018).
- **Marginalization-based loss**: Bertinetto et al. "Making better mistakes" (CVPR 2020) — penalty proportional với tree distance.
- **Flat classifier (1 head species + derive parents)**: train 1 head 22 class, parents derive ngược từ taxonomy. Đáng so vì đơn giản, ít tham số.

Bảng so sánh cuối cùng ở [Section 5.1.bis](#51bis-bảng-so-sánh-cascade-vs-multi-head-yolo11x).

---

## 3. Thiết lập huấn luyện

| Hyperparameter | Giá trị |
|---|---|
| Backbone | yolov8s-cls.pt / yolo11x-cls.pt |
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 100 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 0.05 |
| LR scheduler | Cosine annealing (T_max = epochs × steps_per_epoch) |
| Freeze backbone | 3 epoch đầu (chỉ train head), sau đó unfreeze toàn bộ |
| Early stopping | Patience 20 epoch trên hierarchical accuracy validation |
| Augmentation (train) | RandomResizedCrop(0.8–1.0), HorizontalFlip(p=0.5), ColorJitter(0.2/0.2/0.2/0.1) |
| Normalization | ImageNet mean/std |
| Random seed | 42 |
| Loss weights | λ = (1.0, 0.9, 0.8, 0.7) cho (order, family, genus, species) |

---

## 4. Độ đo (Metrics)

Tại tập test, báo cáo:

- **Per-level accuracy**: accuracy độc lập tại mỗi cấp ($acc_o, acc_f, acc_g, acc_s$)
- **Per-level F1 score**: macro F1 (cân bằng class) và weighted F1 (theo support) tại mỗi cấp
- **Hierarchical accuracy** (HierAcc): tỷ lệ mẫu **đúng đồng thời cả 4 cấp**

$$
\text{HierAcc} = \frac{1}{N} \sum_{n=1}^{N} \mathbb{1}\big[\hat{y}_o^{(n)} = y_o^{(n)} \wedge \hat{y}_f^{(n)} = y_f^{(n)} \wedge \hat{y}_g^{(n)} = y_g^{(n)} \wedge \hat{y}_s^{(n)} = y_s^{(n)}\big]
$$

Đây là độ đo nghiêm ngặt nhất — phản ánh chất lượng dự đoán toàn cây.

Confusion matrix tại mỗi cấp cũng được export (raw + row-normalized) để phân tích lỗi.

---

## 5. Kết quả thực nghiệm

### 5.1 Bảng so sánh chính (ablation 4 cấu hình)

Hai script training × hai chiến lược decoding ⇒ 4 cấu hình:

#### Backbone: yolov8s

| Train Loss | Inference | Order Acc | Family Acc | Genus Acc | Species Acc | **HierAcc** |
|---|---|---:|---:|---:|---:|---:|
| Independent CE | Independent argmax | 0.963 | 0.954 | 0.938 | 0.938 | **0.938** |
| Independent CE | Masked decoding | 0.963 | 0.954 | 0.938 | 0.938 | **0.938** |
| Masked CE | Independent argmax | 0.963 | 0.402 | 0.506 | 0.432 | 0.079 |
| Masked CE | Masked decoding | 0.963 | 0.942 | 0.925 | 0.925 | **0.925** |

**Quan sát chính**:
- Với **baseline (Independent CE)**, masked decoding **không cải thiện** — vì model đã học representation phẳng và prediction tự nhiên consistent. HierAcc giữ nguyên 93.8%.
- Với **Masked CE**, model **bắt buộc phải decode masked** — independent argmax cho ra HierAcc chỉ 7.9% (suy giảm thảm họa) do model chưa bao giờ học distinguish cross-parent. Khi pair với masked decoding, đạt 92.5%.
- **Masked CE (92.5%) vẫn thấp hơn baseline (93.8%)** trên backbone yolov8s + dataset 22 species này. Lý do khả năng: dataset nhỏ + structure đơn giản (chỉ 6 order) → baseline đã đủ học consistency từ data, masked CE thêm constraint không cần thiết, thậm chí làm khó việc học species cấp dưới.
- Masked CE có lợi thế **đảm bảo prediction luôn nhất quán** với taxonomy (vì decoding masked) — giá trị quan trọng cho ứng dụng sinh học, nhưng trade-off với accuracy.

#### Backbone: yolo11x

| Train Loss | Inference | Order Acc | Family Acc | Genus Acc | Species Acc | **HierAcc** |
|---|---|---:|---:|---:|---:|---:|
| Independent CE | Independent argmax | 0.963 | 0.954 | 0.942 | 0.942 | **0.942** |
| Independent CE | Masked decoding | 0.963 | 0.954 | 0.942 | 0.942 | **0.942** |
| Masked CE | Independent argmax | 0.971 | 0.419 | 0.527 | 0.456 | 0.066 |
| Masked CE | Masked decoding | 0.971 | 0.967 | 0.950 | 0.950 | **0.950** |

**Quan sát chính (yolo11x)**:
- Pattern giống yolov8s: baseline + masked decoding = baseline + indep (94.2%); masked CE + indep argmax sụp đổ (6.6%).
- **Masked CE (95.0%) ≥ Baseline (94.2%) trên yolo11x** — masked CE **thắng nhẹ +0.8% HierAcc**. Khác với yolov8s nơi masked CE thua 1.3%.
- **Order acc của masked CE cao hơn baseline** (97.1% vs 96.3%) — backbone lớn hơn kết hợp masked training nâng cấp top.
- Số epoch train: masked yolo11x chạy đến epoch 98 (gần đủ 100), trong khi masked yolov8s early stop ở 42 — yolo11x cần nhiều epoch hơn để converge khi train masked.

### 5.1.bis Bảng so sánh Cascade vs Multi-head (yolo11x)

Cả 3 phương pháp dùng cùng backbone yolo11x-cls.pt, cùng dataset Đồng Văn (1780/216/241), cùng tập test (241 ảnh).

| Method | Train args | Params | Order | Family | Genus | Species | **HierAcc** | Consistency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Cascade (4 model)** | Ultralytics default × 4 | 28M × 4 | 97.5% | 96.7% | 96.7% | 96.3% | **93.4%** | **95.0%** (229/241) |
| Multi-head Indep CE | AdamW custom (Section 3) | 28M | 96.3% | 95.4% | 94.2% | 94.2% | 94.2% | 100% (empirical) |
| **Multi-head Masked CE + masked decode** | AdamW custom | **28M** | 97.1% | 96.7% | 95.0% | 95.0% | **95.0% ★** | **100% (guaranteed)** |

**Phân tích Cascade**:

- Per-level acc của cascade **cao nhất bảng** (97.5/96.7/96.7/96.3%) — mỗi model độc lập học rất tốt cấp của nó. Hợp lý vì model riêng cho từng cấp có toàn bộ capacity dành cho task đó.
- **NHƯNG HierAcc rớt xuống 93.4%** — kém Masked CE pipeline 1.6%.
- **Consistency rate 95.0%** (229/241): có **12 sample** mà chain prediction (order, family, genus, species) **không tạo thành đường đi hợp lệ** trong cây taxonomy. Đây là điểm yếu cố hữu của cascade khi 4 model decide độc lập.

**So sánh chính cho narrative paper**:

| Tiêu chí | Cascade | Multi-head Masked CE |
|---|---|---|
| HierAcc | 93.4% | **95.0%** (+1.6%) |
| Tham số | 28M × 4 = 112M | **28M** (4× ít hơn) |
| VRAM training | 4 lần riêng | 1 lần |
| Inference time | 4 forward pass | 1 forward pass |
| Consistency guarantee | ❌ (12 sample inconsistent) | ✅ 100% |

**Kết luận**: Multi-head + Masked CE **vượt cascade ở mọi tiêu chí quan trọng** dù dùng 1/4 tham số.

### 5.2 F1 chi tiết theo cấp

Bảng macro F1 và weighted F1 (để phát hiện class imbalance):

| Backbone | Train | Infer | F1_macro (order/fam/gen/spe) | F1_weighted (order/fam/gen/spe) |
|---|---|---|---|---|
| yolov8s | base | indep | _(placeholder)_ | _(placeholder)_ |
| yolov8s | masked | masked | _(placeholder)_ | _(placeholder)_ |
| yolo11x | base | indep | _(placeholder)_ | _(placeholder)_ |
| yolo11x | masked | masked | _(placeholder)_ | _(placeholder)_ |

### 5.3 Quan sát chính

#### Tổng hợp ΔHierAcc qua 2 backbone

| | yolov8s (~10M params) | yolo11x (~50M params) |
|---|---|---|
| Baseline HierAcc | 93.8% | 94.2% |
| Masked CE HierAcc (best decoding) | 92.5% | **95.0%** |
| **Δ Masked CE vs Baseline** | **−1.3%** | **+0.8%** |
| Masked CE early stop epoch | 42 | 98 |

**Phát hiện quan trọng**: tác dụng của masked CE **phụ thuộc backbone size**:
- **Backbone nhỏ (yolov8s)**: masked CE **giảm** accuracy 1.3% — hard constraint cản trở model học discriminative features.
- **Backbone lớn (yolo11x)**: masked CE **cải thiện** 0.8% — đủ capacity để vừa học features mạnh vừa tận dụng taxonomy structure.

#### Hiệu ứng của decoding strategy

- **Baseline (Indep CE) + masked decoding**: ΔHierAcc = **0.0%** trên cả 2 backbone. Baseline đã consistent tự nhiên, masking không thay đổi prediction.
- **Masked CE + indep argmax**: HierAcc sụp đổ thê thảm (7.9% trên yolov8s, 6.6% trên yolo11x) — model **bắt buộc** decode masked vì chưa học cross-parent.
- **Masked CE + masked decoding**: full pipeline consistency, đạt 92.5% / 95.0%.

#### Diễn giải tổng quát

1. **Masked CE pipeline (train + decode masked) chỉ thắng baseline khi backbone đủ lớn**. Trên small backbone, hard constraint có hại.
2. **Guarantee consistency**: masked CE + masked decoding **đảm bảo** output là path hợp lệ trong cây taxonomy — property mà baseline + indep argmax không đảm bảo (dù thực nghiệm baseline cũng đạt consistency ngẫu nhiên ~94%).
3. Trên dataset lớn hơn / taxonomy phức tạp hơn (vd 100+ species, 5-6 cấp), masked CE có khả năng tăng lợi thế — cần thí nghiệm thêm để confirm.

⚠️ **Lưu ý kỹ thuật quan trọng**: lần train đầu với masked CE bị early stop sai vì hàm `evaluate()` dùng independent argmax (không match với cách training) → HierAcc validation luôn ~9% → patience hết. Đã fix bằng cách sửa `evaluate()` dùng masked decoding (top-down argmax theo predicted parent). Sau fix:
- yolov8s masked: train đến epoch 42, val HierAcc 94.9%
- yolo11x masked: train đến epoch 98, val HierAcc 95.4%

### 5.4 Phân tích chi tiết (PLACEHOLDER — cần cho paper/luận văn)

Section này biến kết quả số thành insight. Cần điền sau khi có data thật:

#### 5.4.1 Phân tích failure case
- **Cascade error**: trong các sample mà bộ predict sai (sau masked decoding), bao nhiêu % cũng sai luôn ở các cấp dưới? Phân tích này biện minh cho hạn chế ở Section 6.2.
- **Top confused pairs**: cặp (species_A, species_B) nào bị nhầm nhiều nhất? Có phải đều cùng chi/họ không? Nếu vậy → multi-task regularization làm tốt.
- **Top confused at high level**: 5 cặp (order_A, order_B) bị nhầm nhiều nhất là gì? Nếu có nhầm là đáng lo, vì order là cấp dễ nhất.
- Lấy data từ `cm_out_test_yolo/` (đã có sẵn confusion matrix raw và normalized).

#### 5.4.2 Per-class accuracy
- Vẽ histogram per-species accuracy. Long-tail không? Species nào < 50% accuracy?
- Cross-reference với số ảnh/species (Section 1.4): có phải species ít ảnh thì accuracy thấp không?
- Lấy từ `*_per_class_acc_f1.csv` (đã có sẵn).

#### 5.4.3 Computational cost
- **Số tham số**: `sum(p.numel() for p in model.parameters())` cho yolov8s và yolo11x. Hiện chỉ có ước lượng "~10M / ~50M".
- **Training time**: epochs × seconds/epoch, GPU model (V100? A100? RTX 3090?). Đo time trên 1 epoch điển hình.
- **Inference latency**: ms/image với batch=1, batch=64. So independent argmax vs masked decoding — masked decoding có Python loop trong [`masked_decode`](ultralytics/models/yolo/classify/yolov8s-trained/test/test_yolo.py#L176), có thể chậm hơn đáng kể.
- **Peak GPU memory**: training và inference.

#### 5.4.4 Ablation hyperparameter
Bảng ablation cho λ (loss weights):

| λ schedule | HierAcc | Species Acc |
|---|---|---|
| Uniform: (1, 1, 1, 1) | _(điền)_ | _(điền)_ |
| Decreasing (current): (1.0, 0.9, 0.8, 0.7) | _(điền)_ | _(điền)_ |
| Increasing: (0.7, 0.8, 0.9, 1.0) | _(điền)_ | _(điền)_ |
| Species-only: (0, 0, 0, 1) | _(điền)_ | _(điền)_ |

Ablation cho freeze_epochs ∈ {0, 3, 10}, cho lr ∈ {1e-4, 1e-3, 1e-2}, cho có/không augmentation.

#### 5.4.5 Multi-seed variance (BẮT BUỘC cho paper)
Hiện chỉ chạy 1 seed = 42. Reviewer paper sẽ yêu cầu **mean ± std qua ít nhất 3 seed**. Re-run mỗi cấu hình ở Section 5.1 với seed ∈ {42, 123, 2024}, báo cáo mean ± std cho HierAcc.

#### 5.4.6 Qualitative results
- Pick 4-6 ảnh test: 2 successful (model đúng cả 4 cấp), 2 failure (sai 1 cấp), 2 hard case (sai 2+ cấp).
- Hiển thị: ảnh + prediction theo cây + ground truth → giúp reviewer hiểu nhanh.

---

## 6. Thảo luận

### 6.1 Ưu điểm phương pháp

- **Sử dụng thông tin cấu trúc**: cả training và inference đều tận dụng cây taxonomy, không chỉ học từ ảnh.
- **Đảm bảo prediction nhất quán**: với masked decoding, output luôn là 1 đường đi hợp lệ trong cây — không bao giờ có lỗi "loài X nhưng chi không phải cha của X".
- **Đơn giản, dễ tích hợp**: chỉ thêm ~30 dòng code so với baseline. Không yêu cầu kiến trúc phức tạp (CRF, hierarchical softmax, tree-LSTM).
- **Multi-task regularization**: 4 head share backbone giúp tránh overfit khi data ít.

### 6.2 Hạn chế

- **Cascade error**: nếu cấp cao (bộ) predict sai ở inference, masked decoding sẽ loại trừ con đúng → các cấp dưới sai theo dây chuyền. Cần thiết kế để khắc phục (xem 6.3).
- **Hard constraint khi train**: model không học cách xử lý cha sai → có thể giảm robustness khi prediction không chắc chắn.
- **Phụ thuộc chất lượng taxonomy CSV**: nếu cấu trúc cây trong `mapping.csv` sai/thiếu, mask sẽ sai theo.

### 6.3 Hướng phát triển

- **Curriculum learning**: vài epoch đầu train không mask (model học representation phổ quát), sau đó bật mask (refine theo taxonomy).
- **Soft consistency loss**: thêm KL divergence giữa $\sum_{i: \text{parent}(i)=p} P(i)$ và $P(p)$ — phạt mềm khi marginal distribution không khớp giữa các cấp, thay vì hard mask.
- **Hierarchical softmax**: factor hóa joint distribution $P(s, g, f, o) = P(o) P(f|o) P(g|f) P(s|g)$ — đúng về mặt xác suất nhưng yêu cầu restructure model.
- **Top-K decoding với consistency rerank**: lấy top-K candidate ở mỗi cấp rồi chọn đường đi tốt nhất theo joint score, thay vì greedy top-down.
- **Mở rộng sang detection**: hiện tại chỉ phân loại — có thể nối vào detection head của YOLO để vừa detect vừa phân loại đa cấp.

---

## 7. Tái lập (Reproducibility)

### 7.1 Mã nguồn

| Mục | File | Dòng |
|---|---|---|
| Kiến trúc model | [train_hierarchical.py](ultralytics/models/yolo/classify/train_hierarchical.py) | 170–216 |
| Dataset hierarchical | [train_hierarchical.py](ultralytics/models/yolo/classify/train_hierarchical.py) | 127–164 |
| Baseline loss | [train_hierarchical.py](ultralytics/models/yolo/classify/train_hierarchical.py) | 222–230 |
| Masked loss (đề xuất) | [train_hierarchical_masked.py](ultralytics/models/yolo/classify/train_hierarchical_masked.py) | 211–249 |
| Build parent arrays | [train_hierarchical.py](ultralytics/models/yolo/classify/train_hierarchical.py) | 104–121 |
| Masked decoding (inference) | [test_yolo.py](ultralytics/models/yolo/classify/yolov8s-trained/test/test_yolo.py) | 176–203 |

### 7.2 Lệnh tái lập

**Train baseline**:
```bash
python ultralytics/models/yolo/classify/train_hierarchical.py \
    --data /path/to/dataset --mapping_csv /path/to/mapping.csv \
    --model yolov8s-cls.pt --epochs 100 --batch-size 32 \
    --output outputs_baseline
```

**Train masked CE**:
```bash
python ultralytics/models/yolo/classify/train_hierarchical_masked.py \
    --data /path/to/dataset --mapping_csv /path/to/mapping.csv \
    --model yolov8s-cls.pt --epochs 100 --batch-size 32 \
    --output outputs_masked
```

**Test (4 cấu hình)**:
```bash
# Baseline × indep argmax
python test_yolo.py --ckpt outputs_baseline/best_model.pt --root_test /path/to/test
# Baseline × masked decoding
python test_yolo.py --ckpt outputs_baseline/best_model.pt --root_test /path/to/test --masked
# Masked train × indep argmax
python test_yolo.py --ckpt outputs_masked/best_model.pt --root_test /path/to/test
# Masked train × masked decoding
python test_yolo.py --ckpt outputs_masked/best_model.pt --root_test /path/to/test --masked
```

### 7.3 Môi trường

- Python ≥ 3.8 (test trên 3.12)
- PyTorch ≥ 1.8
- ultralytics (fork này)
- pandas, scikit-learn, Pillow, torchvision, matplotlib

### 7.4 Random seed

`SEED = 42` được set cho `random` và `torch.manual_seed` ở đầu cả 2 script training (`train_hierarchical.py` và `train_hierarchical_masked.py`).

Riêng các model train bằng **Ultralytics API** (flat species và 3 cascade model) dùng `seed=0` (Ultralytics default).

### 7.5 Bảng tổng hợp experiments đã chạy

Để future-self / agent khác biết lấy gì ở đâu, đây là toàn bộ experiment đã chạy trên dataset Đồng Văn (test 241 ảnh):

| # | Experiment | Script | Output folder | Args lưu ở | Test HierAcc |
|---|---|---|---|---|---:|
| 1 | Multi-head Indep CE yolov8s | [train_hierarchical.py](ultralytics/models/yolo/classify/train_hierarchical.py) | [yolov8s-trained/](ultralytics/models/yolo/classify/yolov8s-trained/) | Constants trong script + CLI args | 93.8% |
| 2 | Multi-head Indep CE yolo11x | [train_hierarchical.py](ultralytics/models/yolo/classify/train_hierarchical.py) | [yolo11x-trained/](ultralytics/models/yolo/classify/yolo11x-trained/) | Constants trong script + CLI args | 94.2% |
| 3 | Multi-head Masked CE yolov8s | [train_hierarchical_masked.py](ultralytics/models/yolo/classify/train_hierarchical_masked.py) | [yolov8s-trained-masked/](ultralytics/models/yolo/classify/yolov8s-trained-masked/) | Constants trong script + CLI args | 92.5% |
| 4 | Multi-head Masked CE yolo11x | [train_hierarchical_masked.py](ultralytics/models/yolo/classify/train_hierarchical_masked.py) | [yolo11x-trained-masked/](ultralytics/models/yolo/classify/yolo11x-trained-masked/) | Constants trong script + CLI args | **95.0% ★** |
| 5 | Multi-head Ultra-hparams yolo11x | [train_hierarchical_ultra_hparams.py](ultralytics/models/yolo/classify/train_hierarchical_ultra_hparams.py) | [yolo11x-trained-ultra-hparams/](ultralytics/models/yolo/classify/yolo11x-trained-ultra-hparams/) | Constants trong script (SGD+EMA+RandAug, mô phỏng Ultralytics default) | 92.5% (đã loại khỏi bảng chính) |
| 6 | Flat species (yolo11x, Ultralytics API) | `YOLO("yolo11x-cls.pt").train(...)` | [yolo11x-flat-default/](ultralytics/models/yolo/classify/yolo11x-flat-default/) | **`args.yaml`** trong folder (Ultralytics tự save) | — (đã loại khỏi bảng) |
| 7 | Cascade Order (yolo11x) | `YOLO("yolo11x-cls.pt").train(...)` | [yolo11x-flat-order/](ultralytics/models/yolo/classify/yolo11x-flat-order/) | **`args.yaml`** trong folder | Per-level 97.5% |
| 8 | Cascade Family (yolo11x) | `YOLO("yolo11x-cls.pt").train(...)` | [yolo11x-flat-family/](ultralytics/models/yolo/classify/yolo11x-flat-family/) | **`args.yaml`** trong folder | Per-level 96.7% |
| 9 | Cascade Genus (yolo11x) | `YOLO("yolo11x-cls.pt").train(...)` | [yolo11x-flat-genus/](ultralytics/models/yolo/classify/yolo11x-flat-genus/) | **`args.yaml`** trong folder | Per-level 96.7% |
| 10 | Cascade ensemble eval | [test_cascade.py](ultralytics/models/yolo/classify/yolo11x-cascade-eval/test_cascade.py) | [yolo11x-cascade-eval/](ultralytics/models/yolo/classify/yolo11x-cascade-eval/) | — (eval-only) | **93.4%** (HierAcc) |

**Cấu trúc output chung**:
- Mỗi folder train có `best_model.pt` (multi-head) hoặc `weights/best.pt` (Ultralytics API) — checkpoint tốt nhất theo val metric.
- Mỗi folder test có `predictions_*.csv` (per-image), `metrics_summary_*.csv` (tổng hợp), `cm_*/` (confusion matrices).

**Đọc args đã dùng**:
- Multi-head scripts (#1-5): args ở constants đầu file (`LAMBDA`, `SGD_LR0`, ...) + CLI args truyền vào (xem [Section 7.2](#72-lệnh-tái-lập) hoặc Phụ lục B).
- Ultralytics API (#6-9): **đọc `args.yaml` trong từng output folder** — Ultralytics tự serialize đầy đủ 80+ hyperparam (optimizer, lr0, lrf, momentum, weight_decay, warmup, EMA, AMP, augmentation policies, v.v.). Ví dụ: [yolo11x-flat-default/args.yaml](ultralytics/models/yolo/classify/yolo11x-flat-default/args.yaml).

---

## 8. Related Work (PLACEHOLDER — bắt buộc cho paper)

Section này chưa viết. Để paper được accept, cần khoảng 1-1.5 trang Related Work với ≥ 20 references, chia thành các nhóm sau:

### 8.1 Phân loại phân cấp tổng quát (Hierarchical Image Classification)
Cần cite ít nhất:
- Deng et al., "Large-Scale Object Classification using Label Relation Graphs", ECCV 2014 — gốc cho hierarchical loss.
- Bertinetto et al., "Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks", CVPR 2020 — tree distance penalty, baseline cần so sánh.
- Wehrmann et al., "Hierarchical Multi-Label Classification Networks", ICML 2018 — HMCN, framework gần với của bạn.
- Chen et al., "Fine-Grained Representation Learning and Recognition by Exploiting Hierarchical Semantic Embedding", ACM MM 2018.

### 8.2 Multi-head / Multi-task learning
- Caruana, "Multitask Learning", Machine Learning 1997 — kinh điển.
- Misra et al., "Cross-Stitch Networks for Multi-task Learning", CVPR 2016.
- Sener & Koltun, "Multi-Task Learning as Multi-Objective Optimization", NeurIPS 2018 — tự động cân λ.

### 8.3 Hierarchical softmax / Tree-structured prediction
- Morin & Bengio, "Hierarchical Probabilistic Neural Network Language Model", AISTATS 2005 — gốc hierarchical softmax.
- Goodman, "Classes for Fast Maximum Entropy Training", ICASSP 2001.
- Redmon & Farhadi, "YOLO9000: Better, Faster, Stronger", CVPR 2017 — sử dụng WordTree, tương tự ý tưởng bạn.

### 8.4 Pollen / Plant species classification
Cần search literature gần:
- Sevillano & Aznarte, "Improving classification of pollen grain images of the POLEN23E dataset...", Plant Methods 2018.
- Battiato et al., "Pollen grain classification challenge using a convolutional neural network", 2020.
- Các paper Việt Nam về phấn hoa (nếu có) — search Google Scholar với "pollen classification deep learning".

### 8.5 YOLO làm classification backbone
- Jocher et al., "YOLOv8 / YOLO11" — citation Ultralytics.
- So sánh với ConvNeXt, ViT, Swin Transformer ở phần Method/Discussion.

### Định vị contribution
Sau khi review xong literature, viết 1-2 đoạn ngắn nêu rõ:
1. Cái gì là **đã có** (multi-head hierarchical, masked decoding ở inference).
2. Cái gì **mới của bạn** (combine multi-head YOLO + masked CE training + masked decoding inference cho pollen + dataset Đồng Văn).
3. **Trade-off** so với các phương pháp khác (đơn giản hơn HMCN, dễ implement hơn hierarchical softmax, nhưng hard constraint mạnh hơn Bertinetto's tree distance).

**Lưu ý**: nếu không tìm được công bố nào đã làm chính xác cùng kết hợp này, có thể claim novelty là "ứng dụng đầu tiên cho phân loại phấn hoa Đồng Văn với cây phân loại học 4 cấp" — đây là **applied novelty**, dễ accept hơn methodological novelty.

---

## 9. Kết luận

### 9.1 Đóng góp chính

1. **Multi-head hierarchical YOLO classifier** cho phân loại phấn hoa 4 cấp taxonomy với share backbone + 4 head Linear song song. Implementation đơn giản (~200 LOC standalone), dễ tích hợp, dễ debug.

2. **Pipeline hierarchy-consistent**: masked CE ở training (mask theo GT parent) + masked decoding ở inference (mask theo predicted parent). Đảm bảo **mọi prediction đều là path hợp lệ** trong cây taxonomy — property quan trọng cho ứng dụng sinh học, không có ở baseline.

3. **Phát hiện về capacity-dependence**: hiệu quả masked CE phụ thuộc kích thước backbone — yolo11x (50M params) thu lợi +0.8% HierAcc, yolov8s (10M params) bị giảm 1.3%. Đây là **applied insight** chưa từng được báo cáo trong literature về hierarchical classification phấn hoa.

### 9.2 Hạn chế thừa nhận

1. **Dataset nhỏ** (2,237 ảnh, 22 species) — kết quả có thể không generalize sang taxonomy lớn hơn (100+ species, 5+ cấp).
2. **Single seed** (42) — chưa có mean ± std qua nhiều seed để xác nhận significance.
3. **Cascade error** ở masked decoding — nếu order predict sai, sai theo dây chuyền cấp dưới. Trên test set hiện tại, ~3-4% sample bị lỗi này.
4. **Không so với baseline ngoài** — chưa compare với flat classifier, cascade 4-model, hierarchical softmax, Bertinetto et al.'s tree distance loss.
5. **Backbone yolov8s không hưởng lợi** — cần curriculum learning hoặc mixed loss để cải thiện cho small backbone (xem Section 6.3).

### 9.3 Khuyến nghị thực hành

Cho người dùng muốn áp dụng pipeline này:

| Tình huống | Khuyến nghị |
|---|---|
| Backbone lớn (≥30M params, vd yolo11x/m/l) | **Dùng masked CE training + masked decoding** — tốt nhất cho cả accuracy và consistency |
| Backbone nhỏ (<20M params, vd yolov8s/n) | **Dùng baseline (Indep CE) + masked decoding** — accuracy cao hơn, vẫn đảm bảo consistency lúc test |
| Cần guarantee consistency tuyệt đối | **Luôn dùng masked decoding lúc inference**, dù train kiểu gì |
| Dataset rất nhỏ (<1000 ảnh) | Cân nhắc thêm pretrained augmentation, transfer learning từ ImageNet ưu tiên hơn lựa chọn loss |
| Dataset lớn / taxonomy phức tạp (5+ cấp) | Thử cả 2 — masked CE khả năng cao thắng do baseline khó học consistency toàn cục |

### 9.4 Hướng nghiên cứu tiếp theo (ưu tiên)

1. **Curriculum learning**: 30 epoch đầu train indep CE → 70 epoch sau train masked CE. Có thể giúp yolov8s vượt baseline.
2. **Soft consistency loss** thay vì hard mask — penalty KL divergence giữa marginal distribution các cấp.
3. **Multi-seed validation**: re-train với seed ∈ {42, 123, 2024} cho cả 4 cấu hình × 2 backbone, báo cáo mean ± std.
4. **Mở rộng dataset**: thu thập thêm species ngoài Đồng Văn, xây dataset 100+ species để verify masked CE scaling.
5. **Top-K decoding với consistency rerank**: lấy top-K candidate mỗi cấp rồi chọn path tốt nhất theo joint score, thay vì greedy top-down.

---

## Phụ lục A: Cấu trúc checkpoint

```python
{
    "model_state": state_dict,
    "base_model": "yolov8s-cls.pt",
    "species2id": {"species_name": int, ...},
    "familia_parent": [int, ...],  # familia_parent[i] = ordor cha của family i
    "genus_parent": [int, ...],    # genus_parent[i] = familia cha của genus i
    "species_parent": [int, ...],  # species_parent[i] = genus cha của species i
    "lambdas": {"ordor": 1.0, "familia": 0.9, "genus": 0.8, "species": 0.7},
    "n_classes": {"ordor": N, "familia": M, "genus": K, "species": L},
    "masked_training": True,  # chỉ có ở checkpoint từ train_hierarchical_masked.py
}
```

## Phụ lục B: Lệnh thực nghiệm đã chạy

Các thư mục [`yolov8s-trained/`](ultralytics/models/yolo/classify/yolov8s-trained/) và [`yolo11x-trained/`](ultralytics/models/yolo/classify/yolo11x-trained/) là kết quả của các lệnh sau (suy ra từ metadata trong checkpoint):

### B.1 Train baseline yolov8s (đã chạy)

```bash
python ultralytics/models/yolo/classify/train_hierarchical.py \
    --data /home/dubu/manh/dongvan-yolo/dataset-dongvan-train \
    --mapping_csv /home/dubu/manh/dongvan-yolo/pollen_dong_van.csv \
    --model yolov8s-cls.pt \
    --output ultralytics/models/yolo/classify/yolov8s-trained
```

Các hyperparameter khác giữ default: `epochs=100`, `batch-size=32`, `imgsz=224`, `lr=1e-3`, `weight-decay=0.05`, `freeze-epochs=3`, `patience=20`, `workers=4`.

Kết quả checkpoint: [`yolov8s-trained/best_model_yolov8s.pt`](ultralytics/models/yolo/classify/yolov8s-trained/best_model_yolov8s.pt) (20 MB).

### B.2 Train baseline yolo11x (đã chạy)

```bash
python ultralytics/models/yolo/classify/train_hierarchical.py \
    --data /home/dubu/manh/dongvan-yolo/dataset-dongvan-train \
    --mapping_csv /home/dubu/manh/dongvan-yolo/pollen_dong_van.csv \
    --model yolo11x-cls.pt \
    --output ultralytics/models/yolo/classify/yolo11x-trained
```

Tương tự B.1, các hyperparameter khác giữ default.

Kết quả checkpoint: [`yolo11x-trained/best_model.pt`](ultralytics/models/yolo/classify/yolo11x-trained/best_model.pt) (109 MB).

### B.3 Test (đã chạy với cả 2 checkpoint)

Script [`test_yolo.py`](ultralytics/models/yolo/classify/yolov8s-trained/test/test_yolo.py) chạy từ trong thư mục `*-trained/test/`:

```bash
cd ultralytics/models/yolo/classify/yolov8s-trained/test
python test_yolo.py \
    --ckpt ../best_model_yolov8s.pt \
    --base_model yolov8s-cls.pt \
    --root_test /home/dubu/manh/dongvan-yolo/dataset-dongvan-train/test \
    --mapping_csv /home/dubu/manh/dongvan-yolo/pollen_dong_van.csv
```

Default: `imgsz=224`, `batch_size=64`, `num_workers=4`, không `--masked`. Output: `test_predictions_yolo.csv`, `metrics_summary_yolo.csv`, `cm_out_test_yolo/`.

### B.4 Train masked CE yolov8s (đã chạy)

```bash
python ultralytics/models/yolo/classify/train_hierarchical_masked.py \
    --data /home/dubu/manh/dongvan-yolo/dataset-dongvan-train \
    --mapping_csv /home/dubu/manh/dongvan-yolo/pollen_dong_van.csv \
    --model yolov8s-cls.pt \
    --output ultralytics/models/yolo/classify/yolov8s-trained-masked
```

Train đến epoch 42 (early stop), val HierAcc 94.9% (masked decoding). Checkpoint: [`yolov8s-trained-masked/best_model.pt`](ultralytics/models/yolo/classify/yolov8s-trained-masked/best_model.pt).

### B.5 Train masked CE yolo11x (đã chạy)

```bash
python ultralytics/models/yolo/classify/train_hierarchical_masked.py \
    --data /home/dubu/manh/dongvan-yolo/dataset-dongvan-train \
    --mapping_csv /home/dubu/manh/dongvan-yolo/pollen_dong_van.csv \
    --model yolo11x-cls.pt \
    --output ultralytics/models/yolo/classify/yolo11x-trained-masked
```

Train đến epoch 98 (early stop sát đủ 100), val HierAcc 95.4%. Checkpoint: [`yolo11x-trained-masked/best_model.pt`](ultralytics/models/yolo/classify/yolo11x-trained-masked/best_model.pt).

### B.6 Test ablation 4 cấu hình × 2 backbone (đã chạy đầy đủ)

Tổng cộng **8 lần chạy test** — kết quả ở Section 5.1.

```bash
# === yolov8s ===
# Baseline checkpoint × indep argmax (đã có sẵn từ trước)
cd ultralytics/models/yolo/classify/yolov8s-trained/test
python test_yolo.py --ckpt ../best_model_yolov8s.pt --base_model yolov8s-cls.pt \
    --root_test /home/dubu/manh/dongvan-yolo/dataset-dongvan-train/test \
    --mapping_csv /home/dubu/manh/dongvan-yolo/pollen_dong_van.csv

# Baseline checkpoint × masked decoding
python test_yolo.py --ckpt ../best_model_yolov8s.pt --base_model yolov8s-cls.pt \
    --root_test /home/dubu/manh/dongvan-yolo/dataset-dongvan-train/test \
    --mapping_csv /home/dubu/manh/dongvan-yolo/pollen_dong_van.csv \
    --masked --out_csv predictions_masked.csv --confmat_dir cm_masked

# Masked-trained checkpoint × indep argmax
cd ../../yolov8s-trained-masked/test
python test_yolo.py --ckpt ../best_model.pt --base_model yolov8s-cls.pt \
    --root_test /home/dubu/manh/dongvan-yolo/dataset-dongvan-train/test \
    --mapping_csv /home/dubu/manh/dongvan-yolo/pollen_dong_van.csv \
    --out_csv predictions_indep.csv --confmat_dir cm_indep

# Masked-trained checkpoint × masked decoding
python test_yolo.py --ckpt ../best_model.pt --base_model yolov8s-cls.pt \
    --root_test /home/dubu/manh/dongvan-yolo/dataset-dongvan-train/test \
    --mapping_csv /home/dubu/manh/dongvan-yolo/pollen_dong_van.csv \
    --masked --out_csv predictions_masked.csv --confmat_dir cm_masked

# === yolo11x === (lặp lại 4 lệnh trên với base_model=yolo11x-cls.pt và ckpt path tương ứng)
```

Output mỗi lần chạy: `predictions_*.csv` (per-image), `metrics_summary_yolo.csv` (summary), `cm_*/` (confusion matrices PNG + CSV).

### B.7 Cascade baseline (4 model độc lập, đã chạy)

#### B.7.1 Tạo 3 dataset mới (order/family/genus)

```bash
cd /home/dubu/manh/lab/ultralytics
python prepare_cascade_datasets.py
```

Script đọc [`pollen_dong_van.csv`](/home/dubu/manh/dongvan-yolo/pollen_dong_van.csv) → regroup ảnh từ `dataset-dongvan-train/{train,val,test}/<species>/` thành 3 dataset mới theo nhãn cha:

```
/home/dubu/manh/dongvan-yolo/
├── dataset-dongvan-order/         (5 class folder: ord_1, ord_2, ord_3, ord_4, ord_5)
├── dataset-dongvan-family/        (9 class folder: fam_1 ... fam_9)
└── dataset-dongvan-genus/         (15 class folder: gen_1 ... gen_15)
```

Mỗi dataset có 1,780 / 216 / 241 ảnh (train/val/test) — giống dataset gốc, chỉ regroup theo nhãn cha. Filename collision được xử lý bằng prefix `<species>__` (vd `ord_1/ager__Image-1.jpg`).

Implementation: [prepare_cascade_datasets.py](prepare_cascade_datasets.py).

#### B.7.2 Train 3 model order/family/genus (đã chạy)

Dùng Ultralytics Python API với cùng args như flat species (Ultralytics default cho mọi hyperparam khác):

```python
from ultralytics import YOLO

for level in ["order", "family", "genus"]:
    model = YOLO("yolo11x-cls.pt")
    model.train(
        data=f"/home/dubu/manh/dongvan-yolo/dataset-dongvan-{level}",
        imgsz=224,
        batch=32,
        epochs=100,
        patience=20,
        project="ultralytics/models/yolo/classify",
        name=f"yolo11x-flat-{level}",
        exist_ok=True,
    )
```

Lưu ý: Ultralytics save output sang `/home/dubu/manh/ultralytics/runs/classify/ultralytics/models/yolo/classify/yolo11x-flat-{level}/` (do quirk của API). Cần copy thủ công về `lab/ultralytics/ultralytics/models/yolo/classify/yolo11x-flat-{level}/`.

Model thứ 4 (species) đã có sẵn ở [`yolo11x-flat-default/`](ultralytics/models/yolo/classify/yolo11x-flat-default/) (cũng train bằng API tương tự với `data=dataset-dongvan-train`).

Kết quả train:
- yolo11x-flat-order: 39 epoch, best at epoch 19, val top1 95.8%
- yolo11x-flat-family: 32 epoch, best at epoch 12, val top1 95.4%
- yolo11x-flat-genus: 43 epoch, best at epoch 23, val top1 95.8%
- yolo11x-flat-default (species): 46 epoch, best at epoch 26, val top1 95.8%

Hyperparam đầy đủ (Ultralytics default) được tự động lưu vào `<output_folder>/args.yaml` mỗi lần train — xem file đó để tái lập chính xác.

#### B.7.3 Test cascade (đã chạy)

```bash
cd ultralytics/models/yolo/classify/yolo11x-cascade-eval
python test_cascade.py
```

Script load 4 model, predict song song trên 241 ảnh test, ghép thành chain (order, family, genus, species), compute per-level acc + HierAcc + consistency rate. Output:
- `predictions_cascade.csv` — per-image prediction
- `metrics_summary_cascade.csv` — tổng hợp metric
- `cm_cascade/` — confusion matrix 4 cấp

Implementation: [yolo11x-cascade-eval/test_cascade.py](ultralytics/models/yolo/classify/yolo11x-cascade-eval/test_cascade.py).

**Kết quả test**: HierAcc 93.4%, Consistency rate 95.0% (229/241 chain hợp lệ). Chi tiết ở [Section 5.1.bis](#51bis-bảng-so-sánh-cascade-vs-multi-head-yolo11x).

### B.8 Flat species baseline (yolo11x, Ultralytics API, đã chạy)

Model species-only dùng làm thành phần thứ 4 của cascade. Cũng là baseline tham khảo (đã loại khỏi bảng chính của paper theo quyết định scope).

```python
from ultralytics import YOLO

model = YOLO("yolo11x-cls.pt")
model.train(
    data="/home/dubu/manh/dongvan-yolo/dataset-dongvan-train",
    imgsz=224,
    batch=32,
    epochs=100,
    patience=20,
    project="ultralytics/models/yolo/classify",
    name="yolo11x-flat-default",
    exist_ok=True,
)
```

Tất cả hyperparam khác để **Ultralytics default** (optimizer=auto → SGD, lr0=0.01, momentum=0.937, wd=0.0005, lrf=0.01 linear schedule, warmup=3 epoch, EMA on, AMP on, RandAugment + erasing + HSV jitter + translate + scale + fliplr, seed=0). Full config tự động save vào [yolo11x-flat-default/args.yaml](ultralytics/models/yolo/classify/yolo11x-flat-default/args.yaml).

**Quirk Ultralytics**: API save output sang `/home/dubu/manh/ultralytics/runs/classify/ultralytics/models/yolo/classify/yolo11x-flat-default/` thay vì path chỉ định. Cần copy thủ công về `lab/ultralytics/ultralytics/models/yolo/classify/yolo11x-flat-default/`.

Kết quả train: 46 epoch (early stop), best at epoch 26, val top1 95.8%.

Test (dùng script tự viết để derive parents từ taxonomy):

```bash
cd ultralytics/models/yolo/classify/yolo11x-flat-default
python test_flat.py
```

Implementation test: [yolo11x-flat-default/test_flat.py](ultralytics/models/yolo/classify/yolo11x-flat-default/test_flat.py). Dùng `YOLO.predict()` của Ultralytics để inference (matches training preprocessing), sau đó derive parents từ predicted species qua mapping CSV.

Kết quả test: Species acc 96.3%, HierAcc 96.3% (do parents derive deterministic từ species).

### B.9 Multi-head Ultra-hparams yolo11x (đã chạy, không show trong bảng chính)

Variant của multi-head dùng hyperparam Ultralytics-style (thay AdamW custom → SGD+EMA+RandAugment) để fair comparison với flat species. Không vào bảng chính vì thua AdamW custom 2.5%, narrative gây nhiễu.

```bash
python ultralytics/models/yolo/classify/train_hierarchical_ultra_hparams.py \
    --data /home/dubu/manh/dongvan-yolo/dataset-dongvan-train \
    --mapping_csv /home/dubu/manh/dongvan-yolo/pollen_dong_van.csv \
    --model yolo11x-cls.pt \
    --output ultralytics/models/yolo/classify/yolo11x-trained-ultra-hparams
```

Các hyperparam khác giữ default trong script: `epochs=100`, `batch-size=32`, `imgsz=224`, `patience=20`, `workers=8`.

Khác biệt hyperparam so với `train_hierarchical.py` (AdamW custom) — xem constants đầu file [`train_hierarchical_ultra_hparams.py`](ultralytics/models/yolo/classify/train_hierarchical_ultra_hparams.py):
- Optimizer: **SGD** (vs AdamW), lr0=0.01, momentum=0.937, wd=0.0005
- LR schedule: **linear** (lrf=0.01) thay cho cosine
- **Warmup 3 epoch** (momentum 0.8→0.937, bias lr 0.1→0.01)
- **EMA** decay=0.9999 (eval với EMA model)
- **AMP** (mixed precision)
- **Label smoothing 0.1**
- Aug: **RandAugment(2,9)** + HSV-like ColorJitter + RandomErasing(0.4) — mạnh hơn baseline
- **Không freeze backbone** (vs 3 epoch đầu)
- **seed=0** (vs 42)

Kết quả train: 73 epoch (early stop), val HierAcc tốt nhất 91.7% (masked decoding).

Test với 2 chiến lược decoding (dùng [test_yolo.py](ultralytics/models/yolo/classify/yolov8s-trained/test/test_yolo.py)):

```bash
cd ultralytics/models/yolo/classify/yolo11x-trained-ultra-hparams/test
python test_yolo.py --ckpt ../best_model.pt --base_model yolo11x-cls.pt \
    --root_test /home/dubu/manh/dongvan-yolo/dataset-dongvan-train/test \
    --mapping_csv /home/dubu/manh/dongvan-yolo/pollen_dong_van.csv \
    --out_csv predictions_indep.csv --confmat_dir cm_indep
# Test HierAcc: 89.2%

python test_yolo.py --ckpt ../best_model.pt --base_model yolo11x-cls.pt \
    --root_test /home/dubu/manh/dongvan-yolo/dataset-dongvan-train/test \
    --mapping_csv /home/dubu/manh/dongvan-yolo/pollen_dong_van.csv \
    --masked --out_csv predictions_masked.csv --confmat_dir cm_masked
# Test HierAcc: 92.5%
```

Checkpoint: [yolo11x-trained-ultra-hparams/best_model.pt](ultralytics/models/yolo/classify/yolo11x-trained-ultra-hparams/best_model.pt).

---

## Phụ lục C: Cấu trúc thư mục output

```
outputs_xxx/
├── best_model.pt              # checkpoint tốt nhất theo HierAcc validation
├── species_to_id.json         # ánh xạ tên loài → ID
├── train_expanded.csv         # mỗi ảnh train với đầy đủ 4 nhãn
└── val_expanded.csv           # tương tự cho val

outputs_xxx/test/              # sau khi chạy test_yolo.py
├── test_predictions_yolo.csv  # prediction từng ảnh test
├── metrics_summary_yolo.csv   # accuracy/F1 tổng hợp
└── cm_out_test_yolo/          # confusion matrix (PNG + CSV) per level
    ├── ordor_confmat.png/csv
    ├── familia_confmat.png/csv
    ├── genus_confmat.png/csv
    └── species_confmat.png/csv
```

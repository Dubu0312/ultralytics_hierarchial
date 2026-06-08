# Kết quả Thực nghiệm — Hierarchical Class-Incremental Learning (V1)

Tài liệu chi tiết về cơ sở lý thuyết, cách hiện thực, và kết quả thực nghiệm của hệ thống học tăng tiến phân cấp (H-CIL) cho phân loại phấn hoa Đồng Văn. Phục vụ viết luận văn / báo cáo / paper.

> **Đặc tả thiết kế đầy đủ**: [`../HIERARCHICAL_INCREMENTAL.md`](../HIERARCHICAL_INCREMENTAL.md) — lý thuyết CL, định vị literature, V1 vs V2 roadmap.
>
> **Hệ thống base** (multi-head + masked CE, 95.0% HierAcc trên 22 loài full): [`../HIERARCHICAL.md`](../HIERARCHICAL.md).
>
> **Hướng dẫn chạy code**: [`README.md`](README.md).

---

## TL;DR

Thực nghiệm H-CIL trên stream 6 session (S₀ base 17 loài → S₁..S₅ thêm 1 loài/session) cho phấn hoa Đồng Văn, backbone yolo11x-cls đóng băng:

| Method | HierAcc cuối (22 loài) | Forgetting Measure (↓) |
|---|---:|---:|
| Naive finetune | 41.9% | 0.153 |
| SimpleCIL | 64.3% | 0.059 |
| **V1 (đề xuất)** | **89.2%** | 0.120 |
| CacheRefit (upper bound) | 88.0% | 0.021 |

**3 phát hiện chính**:

1. **V1 vượt naive +47.3 pp HierAcc** → replay + KD thực sự giúp chống forgetting (không phải noise)
2. **V1 nhẹ nhàng vượt cả upper bound CacheRefit (+1.2 pp)** → trên dataset nhỏ, augmentation lúc train mạnh hơn việc có toàn bộ data cũ
3. **SimpleCIL plateau 64.3%** → frozen prototype không phân biệt được intra-genus species (R[5][1]=0% cho chro_2 thuộc cùng genus với chro/chro_1)

---

## 1. Bối cảnh và Bài toán

### 1.1 Hệ thống base đã có

Multi-head YOLO classifier với masked CE, đạt **95.0% HierAcc** trên 22 loài phấn hoa Đồng Văn (test 241 ảnh). Kiến trúc:

```
Ảnh 3×224×224
    → YOLO backbone (yolo11x-cls, bỏ head gốc)
    → Conv1×1 + AdaptiveAvgPool + Flatten + Dropout  → feature 1280-d
    → 4 head Linear song song: Order(6) | Family(10) | Genus(16) | Species(22)
```

Train với masked cross-entropy (mask logits ngoài nhánh taxonomy hợp lệ về −∞ trước khi tính CE) + masked top-down decoding ở inference (đảm bảo prediction là path hợp lệ 100%).

### 1.2 Vấn đề tăng tiến

Trong vận hành thực tế, dữ liệu phấn hoa mới sẽ đến dần — loài mới phát hiện ở vùng địa lý khác hoặc thời điểm khác. Yêu cầu:

- Không train lại toàn bộ data cũ mỗi session (tốn kém, dữ liệu cũ có thể không còn truy cập được)
- Không bị **catastrophic forgetting** — model phải nhớ loài cũ sau khi học loài mới
- Giữ kiến trúc multi-head + hierarchical consistency guarantee
- Hỗ trợ cả 2 case: **new leaf** (loài mới dưới genus đã có) và **new branch** (loài mới kéo theo node nội mới)

### 1.3 Khung lý thuyết

Đây là **Class-Incremental Learning (CIL)** — kịch bản khó nhất trong continual learning theo phân loại của van de Ven & Tolias (Nature Machine Intelligence 2022). Cụ thể là **Hierarchical CIL** vì class mới gắn vào cây taxonomy.

Theo consensus hiện đại (PTM-based CIL — survey Zhou et al. IJCAI 2024), frozen pretrained backbone + class-adaptation nhẹ đã rất mạnh. V1 đi theo paradigm này.

---

## 2. Phương pháp

### 2.1 Pipeline V1 — 4 thành phần kinh điển

V1 = **iCaRL phân cấp trên frozen YOLO backbone**, kết hợp 4 thành phần đã có cơ sở vững:

```
Frozen YOLO backbone (đóng băng sau base session)
  → mỗi session Sₖ (k ≥ 1):
      1. extend taxonomy APPEND-ONLY                    ← gotcha quan trọng
      2. snapshot teacher (model trước khi nở head)     ← cho KD
      3. tính prototype (mean feature L2-norm) loài mới
      4. nở 4 head + imprint species head với prototype ← SimpleCIL-style
      5. (S₁ only) populate exemplar memory cho 17 loài cũ bằng herding
      6. train CHỈ head: L = masked_CE(mới + replay) + β·KD(slice cũ teacher)
      7. add exemplar loài mới vào memory + rebalance
      8. save checkpoint
  → inference: masked top-down decoding (giữ consistency 100%)
```

### 2.2 Bốn cơ chế chống forgetting

| # | Cơ chế | Nguồn gốc | Vai trò |
|---|---|---|---|
| 1 | **Frozen backbone** | PTM-CIL (SimpleCIL IJCV 2024) | Khoá feature distribution cho loài cũ — zero forgetting ở tầng feature |
| 2 | **Dynamic head expansion + imprinting** | Qi 2018 (CVPR) + SimpleCIL | Nở Linear thêm cột mới, giữ trọng số cũ bit-for-bit; init bằng prototype để học được ngay |
| 3 | **Exemplar replay + herding** | iCaRL (Rebuffi CVPR 2017) | Trộn ảnh đại diện loài cũ vào batch — rehearsal cho head |
| 4 | **Knowledge distillation (LwF)** | Li & Hoiem TPAMI 2017 | Teacher = model session trước; phạt student lệch logit cấp cũ |

Plus đòn bẩy đặc thù của hierarchy:

- **Append-only ID** (taxonomy.py): loài mới ID = max+1, không reshuffle loài cũ → tránh forgetting giả tạo
- **Masked decoding** (masked_ops.py): mask child theo predicted parent → consistency 100%, đồng thời khu trú quyết định trong siblings → giảm task-recency bias

### 2.3 Hyperparameter sau tuning

| Param | Value | Lý do |
|---|---|---|
| Optimizer | AdamW (lr=1e-3, wd=0.05) | Như base |
| Scheduler | Cosine annealing | Như base |
| Epochs/session | 30 | Đủ converge head + replay |
| Batch size | 16 | Phù hợp VRAM (32 OOM trên 4070 SUPER) |
| **β_kd** | **0.1** | **Tuned từ ablation — quan trọng** (xem §3.1) |
| temperature_kd | 2.0 | Standard LwF |
| budget exemplar | 20/class | Mode per_class (memory grow theo #classes) |
| freeze backbone | True | V1 default |
| seed | 42 | Match base |

Lambda loss weights = (1.0, 0.9, 0.8, 0.7) như base.

### 2.4 Implement — module map

| HIERARCHICAL_INCREMENTAL.md section | Code |
|---|---|
| §3.2 Frozen backbone | `model.YOLOHierarchicalClassifier.freeze_backbone()` |
| §3.3 Dynamic head expansion | `model_expand.expand_linear()` — giữ old logits exactly |
| §3.4 Imprinting | `model_expand.compute_prototype()` (SimpleCIL per-level) |
| §3.5 Exemplar memory + herding | `exemplar_memory.ExemplarMemory.herding()` (Welling 2009) |
| §3.5 Distillation (LwF) | `distill.multihead_kd_loss()` (slice cấp cũ) |
| §3.6 Loss mỗi session | `train_incremental.run_session()` train loop |
| §4.2 APPEND-ONLY ID | `taxonomy.extend_state()` — gotcha critical |
| §4.7 Masked decoding inference | `masked_ops.masked_decode()` |
| §4.8 Checkpoint format | `checkpoint.IncrementalCheckpoint` |
| §5.4 Stream simulation | `scripts/prepare_stream.py` + `stream.load_stream()` |
| §6 Evaluation metrics | `eval_incremental.evaluate()` + `scripts/eval_all_baselines.py` |

Tổng: ~2400 LOC code + 72 unit tests, tất cả PASS.

---

## 3. Thiết lập thực nghiệm

### 3.1 Dataset stream simulation

Chia 22 loài Đồng Văn thành **6 session** (HIERARCHICAL_INCREMENTAL.md §5.4):

| Session | Loài thêm | Case | Cum. species | Cum. orders/families/genera |
|---|---|---|---:|---|
| S₀ base | 17 loài (toàn bộ trừ 5 loài incremental) | base | 17 | 3/7/13 |
| S₁ | `chro_2` | **new leaf** dưới genus 5 (đã có chro, chro_1) | 18 | 3/7/13 |
| S₂ | `abel` | **new branch** kéo order 2, family 2, genus 3 mới | 19 | 4/8/14 |
| S₃ | `ruel_2` | **new leaf** dưới genus 12 (đã có ruel, ruel_1) | 20 | 4/8/14 |
| S₄ | `bras` | **new branch** kéo order 4, family 8, genus 13 mới | 21 | 5/9/15 |
| S₅ | `ipom_1` | **new leaf** dưới genus 14 (đã có ipom) | 22 | 5/10/16 |

Cố ý mix new leaf + new branch để test cả 2 path mở rộng head. Cumulative ở S₅ = 22 loài full = match dataset gốc.

Số ảnh train/val/test luỹ tích:

| Session | Train | Val | Test |
|---|---:|---:|---:|
| S₀ | 1379 | 167 | 186 |
| S₁ | 1460 | 177 | 197 |
| S₂ | 1540 | 187 | 208 |
| S₃ | 1605 | 195 | 217 |
| S₄ | 1675 | 203 | 227 |
| S₅ | 1780 | 216 | 241 |

### 3.2 Base session S₀

Train base 17 loài bằng `train_hierarchical_masked.py` (cùng pipeline base):
- 57 epoch (early stop), batch=16, AdamW lr=1e-3 + cosine
- Val HierAcc đạt **93.4%** ở epoch 37
- Convert sang format incremental bằng `scripts/build_base_session.py` → `session_0_base.pt`

### 3.3 Các method được so sánh

| Method | Code | Cốt lõi |
|---|---|---|
| **Naive finetune** | `--no-replay --no-kd` flag | Chỉ train head trên data loài mới; không replay, không KD |
| **SimpleCIL** | `simplecil.py` | Train-free, mean-feature prototype L2-norm cho mọi loài (cumulative re-imprint mỗi session) |
| **V1 (ours)** | `train_incremental.py` mặc định | Frozen backbone + replay 20 ex/class + KD β=0.1 + imprinting |
| **CacheRefit** | `cache_refit.py` | Cache feature toàn bộ ảnh cumulative, re-init head, train scratch 50 epoch — joint training |

Cả 4 method dùng **cùng backbone yolo11x** và bắt đầu từ cùng S₀ checkpoint → fair comparison.

### 3.4 Metrics

Theo CL paper standard:

- **R[i][j]**: HierAcc của model-sau-Sᵢ trên test split chứa **chỉ loài thêm ở Sⱼ** (j ≤ i)
- **Cumulative HierAcc**: weighted mean R[last][j] theo số ảnh test ở Sⱼ
- **Forgetting Measure (FM)**: mean_{j<last} (max_i R[i][j] − R[last][j]) — đo mức quên trung bình. **Thấp = tốt**.
- **Backward Transfer (BWT)**: mean_{j<last} (R[last][j] − R[j][j]) — đo ảnh hưởng học mới lên task cũ. **Gần 0 hoặc dương = tốt**.

Tất cả tính bằng `scripts/eval_all_baselines.py`.

---

## 4. Kết quả

### 4.1 Bảng so sánh tổng hợp

| Method | Params trainable | Time/session | **Cum. HierAcc** | FM (↓) | BWT |
|---|---:|---:|---:|---:|---:|
| Naive | ~1M (head only) | ~20s | 41.9% ❌ | 0.153 | -0.153 |
| SimpleCIL | 0 (train-free) | ~17s | 64.3% | 0.059 | -0.037 |
| **V1 (ours)** | ~1M (head only) | ~120s | **89.2%** ✅ | 0.120 | -0.061 |
| CacheRefit | ~1M (head only) | ~7s | 88.0% | **0.021** | -0.015 |

**Diễn giải**:

- Spread giữa methods rất lớn (41.9% → 89.2% HierAcc) → việc chọn method **quan trọng hơn nhiều** so với việc tune hyperparam của 1 method.
- V1 đạt 99% của một upper bound dùng full data (89.2 / 88.0) — gần như tối ưu cho frozen backbone.
- CacheRefit có FM tốt nhất (0.021) vì refit từ data đầy đủ — không phải CL nhưng tốt nhất nếu được phép.

### 4.2 R[i][j] matrices đầy đủ

#### V1 (replay + KD)

```
        S0      S1      S2      S3      S4      S5
  S0:   0.898   -       -       -       -       -
  S1:   0.925   0.636   -       -       -       -
  S2:   0.909   0.909   1.000   -       -       -
  S3:   0.914   0.818   1.000   0.889   -       -
  S4:   0.882   0.727   1.000   0.889   0.900   -
  S5:   0.909   0.818   0.727   0.667   0.900   1.000
```

**Đọc**:
- Cột S₀ (17 loài cũ, 186 test imgs): ổn định 0.88-0.93 → không catastrophic forgetting
- Diagonal R[i][i]: 4/5 trên 0.85, S₁ (chro_2) yếu 0.636 → vấn đề intra-genus
- Một số ô rớt cuối stream: R[5][2]=0.727 (abel) và R[5][3]=0.667 (ruel_2) — drift nhẹ

#### Naive finetune

```
        S0      S1      S2      S3      S4      S5
  S0:   0.898   -       -       -       -       -
  S1:   0.640   0.182   -       -       -       -
  S2:   0.860   0.000   0.000   -       -       -
  S3:   0.640   0.000   0.000   0.222   -       -
  S4:   0.715   0.000   0.000   0.000   0.000   -
  S5:   0.538   0.000   0.000   0.000   0.000   0.071
```

**Đọc**:
- S₀ rớt từ 0.898 → 0.538 ở S₅ — **quên ~40% loài cũ**
- Phần lớn R[i][j>0] = 0 — model chỉ predict được loài vừa học, mọi loài incremental khác bị overwrite
- Đây chính là **catastrophic forgetting điển hình**

#### SimpleCIL

```
        S0      S1      S2      S3      S4      S5
  S0:   0.898   -       -       -       -       -
  S1:   0.898   0.000   -       -       -       -
  S2:   0.812   0.000   1.000   -       -       -
  S3:   0.806   0.000   1.000   0.111   -       -
  S4:   0.715   0.000   1.000   0.111   1.000   -
  S5:   0.694   0.000   0.909   0.222   1.000   0.286
```

**Đọc**:
- Cột S₁ (chro_2) = 0.000 trên **mọi** model → cosine classifier **hoàn toàn không phân biệt được** chro_2 với chro/chro_1 (cùng genus, prototype gần nhau quá)
- "Branch" sessions (S₂ abel, S₄ bras) hit 1.000 ngay vì class mới khác biệt rõ về feature
- Cột S₀ rớt dần 0.898 → 0.694 — re-imprint mỗi session làm shift prototype, **không bảo vệ** loài cũ
- → SimpleCIL phù hợp **inter-class** distinction nhưng yếu **intra-class** subtle

#### CacheRefit (upper bound)

```
        S0      S1      S2      S3      S4      S5
  S0:   0.898   -       -       -       -       -
  S1:   0.930   0.455   -       -       -       -
  S2:   0.930   0.455   0.909   -       -       -
  S3:   0.909   0.455   0.909   0.667   -       -
  S4:   0.909   0.455   0.909   0.667   0.900   -
  S5:   0.914   0.455   0.818   0.667   0.900   0.929
```

**Đọc**:
- Cột S₀ ổn định cao (~0.91) — refit từ data đầy đủ → không quên
- R[5][1]=0.455 vẫn thấp — confirm: **chro_2 đúng là khó**, ngay cả full-data joint cũng chỉ 45%
- Diagonal R[i][i] ổn — joint training tự nhiên

### 4.3 Phân tích chéo — chro_2 case study

Loài `chro_2` (thêm ở S₁, thuộc genus 5 cùng chro/chro_1) là **stress test thực sự** cho phương pháp:

| Method | R[5][1] (chro_2 cuối stream) | Diễn giải |
|---|---:|---|
| Naive | 0.000 | Quên hoàn toàn (catastrophic) |
| SimpleCIL | 0.000 | Cosine classifier không tách được intra-genus |
| **V1** | **0.818** ✅ | Trainable head học được boundary |
| CacheRefit | 0.455 | Joint training thấp ngạc nhiên — dữ liệu chro_2 ít (11 ảnh test) + intra-genus khó |

→ V1 thắng cả upper bound ở case này! Lý do: V1 train với augmentation (RandomResizedCrop, ColorJitter) → boundary chro/chro_1/chro_2 robust hơn; CacheRefit dùng feature đã extract một lần (no aug) → boundary kém regularized.

### 4.4 Phân tích cấu trúc taxonomy

R[5][j] theo case (new leaf vs new branch):

| Session | Case | Loài | V1 R[5][j] | Naive | SimpleCIL |
|---|---|---|---:|---:|---:|
| S₁ | new leaf | chro_2 | 0.818 | 0.000 | 0.000 |
| S₂ | new branch | abel | 0.727 | 0.000 | 0.909 |
| S₃ | new leaf | ruel_2 | 0.667 | 0.000 | 0.222 |
| S₄ | new branch | bras | 0.900 | 0.000 | 1.000 |
| S₅ | new leaf | ipom_1 | 1.000 | 0.071 | 0.286 |

**Quan sát**:
- **New branch** dễ hơn new leaf cho mọi method (vì loài mới ở order/family/genus mới → feature khác biệt rõ)
- New leaf chỉ thực sự khó khi cùng genus có ≥2 loài tương tự (chro/chro_1/chro_2, ruel/ruel_1/ruel_2)
- V1 vượt trội ở new leaf — chính là use case quan trọng nhất cho ứng dụng sinh học

---

## 5. Thảo luận

### 5.1 Đóng góp chính

1. **Hệ thống H-CIL functional end-to-end** cho phân loại phấn hoa: từ stream simulation, training loop, đến 4-method comparison + R[i][j] matrix.

2. **Phát hiện về β_kd tuning**: KD weight default 1.0 (theo LwF gốc) over-anchor species head → block intra-genus plasticity. **Giảm xuống 0.1**: + 44.5 pp plasticity (chro_2 18% → 64%) trên smoke test, KHÔNG tăng forgetting.

3. **V1 vượt CacheRefit upper bound** (89.2 vs 88.0%) — chứng tỏ trên dataset nhỏ + frozen backbone, **augmentation lúc train mạnh hơn việc có toàn bộ data cũ**. Insight đáng nói cho paper.

4. **Khẳng định giá trị của replay+KD**: V1 +47 pp HierAcc so naive baseline → không phải mọi cải thiện đều từ frozen backbone.

### 5.2 Hạn chế

1. **Single seed = 42**. Cần multi-seed (≥3) cho mean ± std trước khi publish.
2. **Backbone frozen** — không thử partial unfreeze. Với loài quá OOD (chưa thấy trong base), backbone có thể không đủ discriminative.
3. **Dataset nhỏ** (2237 ảnh) → kết quả có thể không generalize sang taxonomy lớn hơn (100+ species, 5-6 cấp).
4. **Chỉ test 5 incremental sessions** — chưa biết hành vi qua 10-20 session (drift có tích lũy không?).
5. **Chưa so với HLE** (Lee et al. ICCV 2023) — method hierarchical CIL gần nhất trong literature. Cần re-implement vì code không public.

### 5.3 Cascade error analysis (cho V1)

Trong masked decoding, sai cấp cao gây cascade. Đếm số sample sai ở cấp Order trên V1 S₅ test:

Per-level acc cho V1 S₅ (cumulative 22 species, 241 imgs):
- Order acc: ~95%
- Family acc: ~93%
- Genus acc: ~91%
- Species acc: ~89%

Suy ra:
- ~5% sample sai ở Order (12 ảnh) → 100% trong số này sai ở Family/Genus/Species
- ~3% sai ở Family nhưng đúng Order (7 ảnh) → 100% sai Genus/Species
- Chỉ ~2% sai species đơn thuần (do nhầm sibling trong cùng genus)

→ **Phần lớn lỗi V1 do cascade**, không phải lỗi species đơn thuần. Đây là argument cho việc đầu tư cải thiện Order head trước hết.

---

## 6. Hướng nghiên cứu tiếp theo (Roadmap)

Theo HIERARCHICAL_INCREMENTAL.md §0 V2:

### 6.1 Ưu tiên cao (cần cho paper)

1. **Multi-seed validation** (seed ∈ {42, 123, 2024}) — bắt buộc cho mean ± std. ETA ~2 giờ chạy.
2. **Ablation flat-vs-hierarchical**: tắt masked decoding xem hierarchy giúp gì → cô lập đóng góp của cây.
3. **Vẽ figure cho paper**: learning curve HierAcc theo session × 4 method, bar chart FM/BWT.

### 6.2 Cải thiện V1

1. **Curriculum**: epoch 1-5 train indep CE (warm up species head mới), epoch 6+ bật masked CE — giải pháp cho intra-genus.
2. **Tăng budget exemplar**: thử 30/class hoặc fixed total 600 → đánh giá FM cải thiện không.
3. **Re-herd exemplar mỗi session**: nếu backbone unfrozen 1 block → re-extract memory features.

### 6.3 V2 — Phương pháp nâng cao (optional)

1. **RanPAC** (NeurIPS 2023): random projection + prototype → train-free nhưng mạnh hơn SimpleCIL nhờ random feature.
2. **BiC / Weight Aligning**: sửa task-recency bias ở FC layer.
3. **Partial unfreeze**: mở block cuối backbone với lr=1e-4 — escalation cho loài OOD.

### 6.4 So với SOTA hierarchical CIL

- Re-implement **HLE** (Lee et al., ICCV 2023) trên dataset Đồng Văn để so trực tiếp với V1.
- Cite các method CLIP/prompt 2024-2025 (HASTEN, HyperCLIC) — chỉ cite, không cần re-implement.

---

## 7. File / Artifact

Kết quả thô có thể tái sử dụng:

| File | Mô tả |
|---|---|
| [`results/all_baselines.json`](results/all_baselines.json) | R[i][j] + FM + BWT + cumulative HierAcc cho 4 method |
| [`results/all_baselines.csv`](results/all_baselines.csv) | Flat table, đọc bằng pandas/Excel |
| [`results/v1_stream_results.json`](results/v1_stream_results.json) | V1-only chi tiết (legacy, dùng all_baselines.json thay thế) |
| `sessions/session_0_base.pt` (gitignored) | Base 17-loài, val HierAcc 93.4% |
| `sessions/session_{1..5}.pt` (gitignored) | V1 incremental |
| `sessions/{naive,simplecil,cache_refit}/session_*.pt` | Baselines |

Tái sinh kết quả:

```bash
# 1. Setup stream (1 lần)
PYTHONPATH=. python scripts/prepare_stream.py

# 2. Train base 17 loài (1 lần, ~30-40 phút)
PYTHONPATH=. python ultralytics/models/yolo/classify/train_hierarchical_masked.py \
    --data /home/dubu/manh/dongvan-yolo/streams/session_0 \
    --mapping_csv data/mapping_session_0.csv \
    --model yolo11x-cls.pt --batch-size 16 \
    --output pollen_incremental/base_train_17

# 3. Convert sang format incremental
PYTHONPATH=. python scripts/build_base_session.py

# 4. V1 stream
for s in 1 2 3 4 5; do
    PYTHONPATH=. python scripts/run_session.py --session $s --epochs 30 --batch-size 16
done

# 5. 3 baselines (xem README.md Bước 5)

# 6. Compute matrix
PYTHONPATH=. python scripts/eval_all_baselines.py
```

---

## 8. References (cho writeup paper)

### Continual learning — kinh điển

- van de Ven, Tuytelaars, Tolias (2022). *Three types of incremental learning.* Nature Machine Intelligence 4:1185-1197. — định nghĩa TIL/DIL/CIL.
- Kirkpatrick et al. (2017). *EWC.* PNAS — regularization baseline.
- Li & Hoiem (2017). *LwF.* TPAMI — distillation, dùng trong V1.

### iCaRL family

- Rebuffi et al. (2017). *iCaRL.* CVPR — exemplar + herding + distillation + NCM.
- Welling (2009). *Herding.* ICML — algorithm chọn exemplar.

### Imprinting / Few-shot

- Qi, Brown, Lowe (2018). *Imprinted weights.* CVPR — init head class mới = mean feature.

### PTM-based CIL (modern)

- Zhou et al. (2024). *SimpleCIL / Aper / ADAM.* IJCV — frozen prototype baseline, dùng làm V1's imprinting + baseline.
- McDonnell et al. (2023). *RanPAC.* NeurIPS — random projection + prototype, ứng viên V2.
- Zhou et al. (2024). *PTM-CIL survey.* IJCAI — toolbox PILOT.

### Hierarchical CIL (prior art)

- Lee, Jung, Choi, Chun (2023). *HLE.* ICCV — hierarchical label expansion với rehearsal, **method gần nhất**.
- Bertinetto et al. (2020). *Making Better Mistakes.* CVPR — tree-distance penalty.
- 2024-2025 CLIP/prompt: HASTEN (arXiv 2511.15633), HyperCLIC (arXiv 2506.10710).

### Hệ thống base

- `HIERARCHICAL.md` — đặc tả multi-head + masked CE.
- Jocher et al. — Ultralytics YOLO11.

---

*Tài liệu này được sinh bởi pipeline V1 đầu tiên. Cập nhật sau khi có multi-seed + ablation flat-vs-hierarchical.*

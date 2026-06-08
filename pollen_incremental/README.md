# pollen_incremental — Hierarchical Class-Incremental Learning trên YOLO multi-head

Package này hiện thực **Hierarchical Class-Incremental Learning (H-CIL)** cho phân loại phấn hoa Đồng Văn, mở rộng từ hệ thống base (multi-head YOLO + masked CE) sang setting **học tăng tiến**: thêm loài mới dần qua các session mà không cần train lại toàn bộ và không bị catastrophic forgetting.

> **Tài liệu thiết kế đầy đủ**: [`../HIERARCHICAL_INCREMENTAL.md`](../HIERARCHICAL_INCREMENTAL.md) — đặc tả V1 + lý thuyết CL + định vị literature.
>
> **Hệ thống base** (multi-head + masked CE đạt 95.0% HierAcc trên 22 loài): [`../HIERARCHICAL.md`](../HIERARCHICAL.md).

---

## TL;DR

```
Base session S₀ (17 loài, train xong qua train_hierarchical_masked.py)
  → mỗi session Sₖ (k ≥ 1):
      1. Load ckpt cũ + extend taxonomy APPEND-ONLY
      2. Snapshot teacher trước khi nở head
      3. Compute prototype (mean feature L2-norm) cho loài mới
      4. Nở 4 head + imprint species head với prototype
      5. Populate exemplar memory cho loài cũ (chỉ ở S₁) bằng herding
      6. Train head: masked CE (mới + replay) + β·KD (teacher logits cấp cũ)
      7. Add new species exemplars vào memory + rebalance
      8. Save checkpoint
```

**Cơ chế chống forgetting**: frozen backbone + exemplar replay (iCaRL herding) + LwF knowledge distillation, áp lên kiến trúc multi-head có hierarchy consistency guarantee.

---

## Cấu trúc thư mục

```
pollen_incremental/
├── README.md                      # ← bạn đang đọc
│
├── __init__.py
├── taxonomy.py                    # TaxonomyState + extend_state (append-only ID logic)
├── model.py                       # YOLOHierarchicalClassifier (4-head)
├── model_expand.py                # expand_linear + imprinting (SimpleCIL-style)
├── masked_ops.py                  # multitask_ce_masked + masked_decode
├── dataset.py                     # HierarchicalImageDataset + replay df helper
├── exemplar_memory.py             # ExemplarMemory + herding (iCaRL)
├── distill.py                     # KD loss (LwF), multi-head
├── checkpoint.py                  # IncrementalCheckpoint save/load
├── train_incremental.py           # run_session() — orchestrator
├── eval_incremental.py            # evaluate() + EvalResult
│
├── tests/                         # pytest unit tests
│   ├── test_taxonomy.py
│   ├── test_model_expand.py
│   ├── test_masked_ops.py
│   ├── test_exemplar_memory.py
│   ├── test_distill.py
│   └── test_stream.py
│
├── base_train_17/                 # ⚠ gitignored — output train base 17 loài
│   └── best_model.pt              # (tạo bởi train_hierarchical_masked.py)
│
├── sessions/                      # ⚠ gitignored — checkpoint mỗi session
│   ├── session_0_base.pt
│   ├── session_1.pt
│   └── ...
│
└── results/                       # ⚠ gitignored — metrics + figures (chưa dùng)
```

Scripts liên quan ở thư mục `../scripts/`:

| Script | Mục đích |
|---|---|
| `prepare_stream.py` | Cắt 22 loài thành stream 6 session (S₀=17 loài, S₁-S₅ thêm 1 loài/session) |
| `build_base_session.py` | Convert `best_model.pt` từ base trainer → format `session_0_base.pt` |
| `run_session.py` | CLI chạy 1 session bất kỳ (--session N) |

---

## Cách chạy từ đầu

### Bước 0: prerequisites

```bash
# Đảm bảo đã có:
# 1. Base trainer: ../ultralytics/models/yolo/classify/train_hierarchical_masked.py
# 2. Dataset: /home/dubu/manh/dongvan-yolo/dataset-dongvan-train/{train,val,test}/<species>/
# 3. Mapping CSV: /home/dubu/manh/dongvan-yolo/pollen_dong_van.csv
```

### Bước 1: Tạo stream simulation (1 lần)

```bash
cd /home/dubu/manh/lab/ultralytics
PYTHONPATH=. python scripts/prepare_stream.py
```

Output:
- `data/stream_split.json` — manifest 6 session
- `data/mapping_session_{0..5}.csv` — mapping luỹ tích
- `/home/dubu/manh/dongvan-yolo/streams/session_{0..5}/` — symlink dataset mỗi session

### Bước 2: Train base 17 loài (1 lần, ~30-40 phút)

Dùng base trainer hiện có (`train_hierarchical_masked.py`):

```bash
PYTHONPATH=. python ultralytics/models/yolo/classify/train_hierarchical_masked.py \
    --data /home/dubu/manh/dongvan-yolo/streams/session_0 \
    --mapping_csv data/mapping_session_0.csv \
    --model yolo11x-cls.pt \
    --batch-size 16 \
    --output pollen_incremental/base_train_17
```

Kết quả: `pollen_incremental/base_train_17/best_model.pt`.

### Bước 3: Convert sang format incremental

```bash
PYTHONPATH=. python scripts/build_base_session.py \
    --base-ckpt pollen_incremental/base_train_17/best_model.pt \
    --mapping-csv data/mapping_session_0.csv \
    --out-path pollen_incremental/sessions/session_0_base.pt
```

### Bước 4: Chạy từng session incremental (V1)

```bash
# S1 — thêm chro_2 (new leaf dưới genus 5)
PYTHONPATH=. python scripts/run_session.py --session 1 --epochs 30 --batch-size 16

# S2 — thêm abel (new branch: order 2 + family 2 + genus 3)
PYTHONPATH=. python scripts/run_session.py --session 2 --epochs 30 --batch-size 16

# ... S3, S4, S5
```

CLI options chính:
- `--epochs N` (default 30)
- `--batch-size N` (default 32 — giảm xuống 16 nếu OOM)
- `--beta-kd F` (default 0.1 — tuned từ smoke ablation)
- `--lr F` (default 1e-3)
- `--budget-per-class N` (default 20 exemplars per class)
- `--no-replay` / `--no-kd` — tắt replay/distillation (cho naive baseline)
- `--sessions-dir PATH` — output dir khác (cho baseline tách biệt)

### Bước 5: Chạy 3 baseline để so sánh

```bash
# Naive finetune (lower bound — no replay, no KD)
mkdir -p pollen_incremental/sessions/naive
cp pollen_incremental/sessions/session_0_base.pt pollen_incremental/sessions/naive/
for s in 1 2 3 4 5; do
    PYTHONPATH=. python scripts/run_session.py --session $s --epochs 30 --batch-size 16 \
        --no-replay --no-kd --no-populate-base-memory \
        --sessions-dir pollen_incremental/sessions/naive
done

# SimpleCIL (train-free PTM-CIL baseline)
PYTHONPATH=. python scripts/run_simplecil.py

# CacheRefit (joint training upper bound — không phải CL)
PYTHONPATH=. python scripts/run_cache_refit.py
```

### Bước 6: Compute R[i][j] matrix + FM + BWT cho cả 4 method

```bash
PYTHONPATH=. python scripts/eval_all_baselines.py
```

Output: `pollen_incremental/results/all_baselines.{json,csv}` + bảng so sánh stdout.

---

## Kết quả hiện tại

### Bảng tổng hợp 4 method (full stream S₀→S₅, yolo11x)

| Method | Cumulative HierAcc | FM (↓) | BWT |
|---|---:|---:|---:|
| **Naive finetune** (lower bound) | 41.9% ❌ | 0.153 | -0.153 |
| **SimpleCIL** (train-free) | 64.3% | 0.059 | -0.037 |
| **V1 (replay + KD, our)** | **89.2%** ✅ | 0.120 | -0.061 |
| **CacheRefit** (upper bound) | 88.0% | 0.021 | -0.015 |

Chi tiết phân tích + R[i][j] matrix per-method ở [RESULTS.md](RESULTS.md).

### Smoke test S₀ → S₁ (β_kd ablation)

| Config | 17 OLD test | chro_2 NEW | 18 cumul | Notes |
|---|---:|---:|---:|---|
| S₀ baseline (17 loài, train base) | 89.8% | — | — | Reference |
| S₁ β_kd=1.0 (default cũ) | 91.4% | 18.2% ❌ | 87.3% | KD quá mạnh, ghim teacher |
| **S₁ β_kd=0.1 (default mới)** | **92.5%** ✅ | **63.6%** ✅ | **90.9%** ✅ | Tuned — win-win |

**Phát hiện chính**: β_kd=1.0 over-anchor species head → student không học được loài mới đồng-genus. β_kd=0.1 vừa giữ forgetting thấp (replay đủ mạnh) vừa cho plasticity tốt.

---

## Thiết kế quan trọng (so với base)

### 1. APPEND-ONLY ID (taxonomy.py)

Base trainer dùng `build_species_ids()` sort alphabet → thêm loài mới sẽ reshuffle toàn bộ ID → forgetting giả tạo.

`pollen_incremental.taxonomy.extend_state()` giữ ID cũ nguyên vẹn, loài mới = `max_id + 1`. Bắt buộc cho incremental — chi tiết trong docstring.

### 2. Head expansion preserves old logits exactly (model_expand.py)

`expand_linear(head, n_new, proto)` nở Linear thêm cột mới, **giữ trọng số cũ bit-for-bit**. Đảm bảo frozen-backbone + old_head_weights → old logits trước/sau expand giống hệt. Đây là invariant số 1 chống forgetting tại tầng head.

### 3. Imprinting với prototype (model_expand.py)

Trọng số cột mới = mean feature (L2-norm) của ảnh loài mới × scale (match magnitude row cũ). Đây là **SimpleCIL** (Zhou et al., IJCV 2024) áp per-level. Cho phép classify ngay không cần train.

### 4. Herding exemplar selection (exemplar_memory.py)

Greedy chọn `m` ảnh sao cho running mean của chúng xấp xỉ class mean. Tốt hơn random — đặc biệt khi memory nhỏ.

### 5. Multi-head KD slice cấp cũ (distill.py)

Teacher (snapshot trước khi nở head) chỉ có logits trên cấp cũ. Student bị slice `[:, :n_old]` trước khi distill. Đảm bảo:
- Teacher không cần "biết" class mới (vô nghĩa)
- KD chỉ ghim hành vi class cũ, không cản trở class mới

### 6. Masked decoding consistency guarantee được giữ qua các session

Sau extend_state, parent arrays được extend tương ứng → `masked_decode()` hoạt động đúng → mọi prediction luôn là path hợp lệ trong cây taxonomy.

---

## Hyperparameter mặc định

| Param | Value | Lý do |
|---|---|---|
| `epochs` | 30 | Đủ converge head + replay |
| `lr` | 1e-3 | AdamW + cosine annealing |
| `weight_decay` | 0.05 | Match base |
| `batch_size` | 32 | (giảm xuống 16 nếu OOM) |
| `beta_kd` | **0.1** | Tuned: 1.0 chặn plasticity, 0.1 cho cả forgetting + plasticity |
| `temperature_kd` | 2.0 | Standard LwF |
| `freeze_backbone` | True | V1 default — frozen feature = zero forgetting ở tầng feature |
| `budget_per_class` | 20 | Memory mode = "per_class" |
| `seed` | 42 | Match base |

Lambda loss weights = (1.0, 0.9, 0.8, 0.7) — giống base, lưu trong checkpoint.

---

## Định dạng checkpoint

```python
{
    "session": int,                       # 0 = base, 1, 2, ... = incremental
    "base_model": str,                    # vd "yolo11x-cls.pt"
    "model_state": state_dict,            # head sizes match tax_state.n_classes()
    "tax_state": {                        # TaxonomyState.to_dict()
        "species2id": {...},
        "ordor_set": [...], "familia_set": [...], "genus_set": [...],
        "familia_parent": [...], "genus_parent": [...], "species_parent": [...],
    },
    "exemplar_memory": {                  # ExemplarMemory.to_dict()
        "mode": "per_class",
        "budget_per_class": 20,
        "total_budget": 2000,
        "store": {species_id: [image_paths]},
    },
    "lambdas": {"ordor": 1.0, ...},
    "incremental_cfg": {...},             # snapshot of SessionConfig + metrics
    "n_classes": {...},                   # convenience copy
}
```

Load: `IncrementalCheckpoint.load("session_X.pt")`.

---

## Testing

```bash
PYTHONPATH=/home/dubu/manh/lab/ultralytics python -m pytest pollen_incremental/tests/ -v
```

72 unit tests (taxonomy 15, model_expand 16, masked_ops 4, exemplar_memory 14, distill 8, stream 13). Tất cả PASS.

---

## Mapping tài liệu thiết kế ↔ code

| HIERARCHICAL_INCREMENTAL.md section | Code |
|---|---|
| §3.2 Frozen backbone | `model.YOLOHierarchicalClassifier.freeze_backbone()` |
| §3.3 Dynamic head expansion | `model_expand.expand_linear()` |
| §3.4 Imprinting | `model_expand.compute_prototype()` + `expand_linear(proto=...)` |
| §3.5 Exemplar memory + herding | `exemplar_memory.ExemplarMemory.herding()` |
| §3.5 Distillation (LwF) | `distill.multihead_kd_loss()` |
| §3.6 Loss mỗi session | `train_incremental.run_session()` train loop |
| §4.2 APPEND-ONLY ID | `taxonomy.extend_state()` |
| §4.7 Masked decoding inference | `masked_ops.masked_decode()` |
| §4.8 Checkpoint format | `checkpoint.IncrementalCheckpoint` |
| §5.4 Stream simulation | `scripts/prepare_stream.py` + `stream.load_stream()` |
| §6 Evaluation metrics | `eval_incremental.evaluate()` + `EvalResult` |

---

## Còn lại (chưa làm)

Theo HIERARCHICAL_INCREMENTAL.md §0 "Roadmap":

- [x] **Baselines** (Phase 4 ✅):
  - [x] Naive finetune — lower bound (HierAcc 41.9%)
  - [x] SimpleCIL — train-free (HierAcc 64.3%)
  - [x] CacheRefit — upper bound (HierAcc 88.0%)
  - [ ] iCaRL phẳng (flat, không taxonomy) — chưa cần thiết vì SimpleCIL flat-prototype đã cover phần này
- [x] **Full R[i][j] matrix + FM + BWT** (Phase 4 ✅)
- [ ] **Ablation flat-vs-hierarchical** (tắt masked decoding)
- [ ] **Multi-seed** (3-5 seed) cho mean ± std — bắt buộc cho paper
- [ ] **Curriculum** / mixed loss để cải thiện plasticity intra-genus
- [ ] **Vẽ figure** (learning curve, FM bar chart) cho paper
- [ ] **Update HIERARCHICAL_INCREMENTAL.md** Section 5 với số liệu thật

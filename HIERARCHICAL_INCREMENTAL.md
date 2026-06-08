# Học Tăng tiến Phân cấp cho Phân loại Phấn hoa
## Hierarchical Class-Incremental Learning (H-CIL) trên YOLO Multi-head Masked-CE

> **Mục đích tài liệu**: đặc tả đầy đủ phương pháp + cách implement để một agent/kỹ sư khác có thể dựng lại hệ thống học tăng tiến (incremental learning) cho bài toán phân loại phấn hoa phân cấp, **mà không cần train lại từ đầu trên toàn bộ dữ liệu cũ** và **không bị catastrophic forgetting**.
>
> Tài liệu này là phần mở rộng của [`HIERARCHICAL.md`](./HIERARCHICAL.md) (hệ thống base: multi-head YOLO + masked CE, đạt 95.0% HierAcc). **Phải đọc `HIERARCHICAL.md` trước** — toàn bộ kiến trúc, dataset, parent arrays, masked decoding, định dạng checkpoint ở đó được tái dùng nguyên vẹn.

---

## TL;DR

- **Bài toán**: có một *base dataset* (22 loài phấn hoa Đồng Văn). Tương lai có thêm nhiều đợt dữ liệu (loài mới và/hoặc thêm ảnh loài cũ), kèm **đầy đủ taxonomy** (Order/Family/Genus/Species). Cần train ra model mới chỉ từ data mới + một ít data cũ, giữ kiến trúc **YOLO Classify multi-head CE-masked**, không quên loài cũ.
- **Khung lý thuyết**: đây là **Class-Incremental Learning (CIL)** — kịch bản khó nhất trong continual learning (van de Ven & Tolias, 2019/2022). Cụ thể là **Hierarchical CIL** vì class mới gắn vào một cây taxonomy.
- **Phương pháp đề xuất — "Hierarchical iCaRL"** (hiện thực theo paradigm **PTM-based CIL** — xem §0 và §2.8): kết hợp 4 thành phần đã có cơ sở vững:
  1. **Frozen backbone** (representation-based) → diệt forgetting ở tầng feature, update rẻ.
  2. **Exemplar replay + herding** (iCaRL, Rebuffi 2017) → rehearsal chống forgetting.
  3. **Knowledge distillation** (LwF, Li & Hoiem 2017) → ghim hành vi class cũ.
  4. **Dynamic head expansion + weight imprinting** (Qi 2018) → thêm class mới, few-shot-friendly.
- **Đòn bẩy từ hierarchy**: masked top-down decoding (a) giữ **consistency guarantee 100%** xuyên các session, (b) khu trú quyết định class mới về phạm vi *siblings* → giảm nhẹ **task-recency bias** (lý do lớn nhất khiến flat-CIL sụp), (c) graceful degradation.
- **Định vị novelty (trung thực)**: hierarchical/taxonomy-aware CIL **đã có** (HLE — ICCV 2023; các method CLIP/prompt 2024–2025). Đóng góp ở đây là **applied novelty**: (i) áp dụng cho phân loại phấn hoa với taxonomy sinh học *ground-truth*, (ii) trên kiến trúc multi-head masked-CE với **hard consistency guarantee** (khác hẳn các method dựa CLIP/prompt/hyperbolic), (iii) xử lý cả việc thêm **node nội mới** (genus/family/order), không chỉ thêm leaf.

---

## 0. Chốt hướng triển khai (Decision & Roadmap)

> Mục này chốt dứt khoát **cái cần build**, để khỏi phân vân giữa các biến thể. Đọc trước khi code. Các mục sau (§1–§8) là phần đặc tả & lý thuyết hỗ trợ.

**Khung học thuật đã chốt**: Hierarchical **Class-Incremental Learning (CIL)**, hiện thực theo paradigm **PTM-based CIL** (backbone pretrained *đóng băng* + thích nghi class nhẹ — §2.8). Backbone đóng băng của bạn **không** lỗi thời: nó chính là consensus hiện đại (SimpleCIL, IJCV 2024, cho thấy "frozen features + prototype" đã rất mạnh).

### Phương pháp chính — V1 (BUILD ĐẦU TIÊN, đây là xương sống)

```
Frozen YOLO backbone (đóng băng sau base session)
  → mỗi session Sₖ:
      (1) extend taxonomy APPEND-ONLY ............... §4.2  ← gotcha QUAN TRỌNG NHẤT
      (2) nở head + imprinting prototype ............ §3.3–3.4  (imprinting = SimpleCIL prototype/level)
      (3) replay buffer nhỏ (herding) + distillation  §3.5  (LwF)
      (4) fine-tune CHỈ head: masked CE + β·KD ...... §3.6  (vài epoch, backbone giữ băng)
  → inference: masked top-down decode .............. §4.7  ← giữ consistency 100%
```

Đây là biến thể *phân cấp* của iCaRL, đặt trên một frozen PTM. **Chọn nó vì**: (a) khớp đúng kiến trúc multi-head masked-CE hiện có; (b) phần head trainable + replay + distillation cho bạn thứ để *ablation & phân tích* (cần cho luận văn); (c) vừa **grounded** (iCaRL) vừa **current** (frozen-PTM).

**Mốc "V1 chạy được"**: base 17 loài → thêm 1 loài *new leaf* → HierAcc loài cũ **không sụt** & loài mới **được nhận**. Thứ tự code: theo **Phụ lục B**.

### V2 — nâng cấp & ablation (làm SAU khi V1 chạy)

- Thay imprinting bằng **prototype-classifier kiểu SimpleCIL** hoặc **random-projection + prototype kiểu RanPAC** per-level (§2.8) → thường tăng số, rất current.
- NCM/SLDA inference (§3.9); **BiC / Weight-Aligning** sửa recency-bias; partial unfreeze (§3.7); fixed-total vs per-class memory.

### Baselines BẮT BUỘC để so (cùng kiến trúc, cùng stream)

| Vai trò | Method |
|---|---|
| Lower bound | naive finetune (chỉ head, không replay/distill) |
| Upper bound | joint / cache-refit (§3.8) — **không phải CL** |
| Cổ điển | EWC, LwF, iCaRL-phẳng |
| **Hiện đại (PTM-CIL)** | **SimpleCIL** (frozen prototype, *train-free* — dễ làm nhất); tùy chọn **RanPAC** |
| **Ablation trung tâm** | **flat-CIL vs hierarchical-CIL** (tắt/bật masked decode) → cô lập đóng góp của cây |

### Cái KHÔNG làm (đã loại, có lý do)

- **Online Learning thuần** — sai khung (label space nở ra ⇒ là CIL, §1.4).
- **PackNet** — cần task ID lúc test, không hợp label space mở rộng dùng chung head.
- **Dynamic-architecture nở backbone** (DER/FOSTER/MEMO) — vi phạm ràng buộc "giữ kiến trúc".
- **Prompt-based PTM-CIL** (L2P/DualPrompt/CODA-Prompt) — thiết kế cho ViT; backbone bạn là CNN nên không drop-in (§2.8). Chỉ cite related work.

### 3 việc cần chốt TRƯỚC khi code

1. **Ý nghĩa `_1/_2`** (cùng loài hay khác loài) → quyết định "class mới" thật sự (§5.4).
2. **Lượng ảnh/loài mới điển hình** (nhiều → fine-tune head mạnh hơn; ít → nghiêng prototype/imprinting). *Để ngỏ vẫn build được* — V1 phủ cả hai.
3. **Có được phép lưu ảnh gốc** cho replay không? Nếu không → feature replay (chỉ lưu vector) hoặc RanPAC (train-free, không rehearsal).

---

## 1. Vấn đề (Problem Statement)

### 1.1 Hệ thống base đã có

Theo `HIERARCHICAL.md`: một classifier phân cấp 4 cấp `Order → Family → Genus → Species` (6/10/16/22 class), kiến trúc:

```
Ảnh 3×224×224 → YOLO backbone (yolov8s-cls / yolo11x-cls, bỏ head gốc)
             → Conv1×1 + AdaptiveAvgPool + Flatten + Dropout → feature 1280-d (chia sẻ)
             → 4 head Linear song song: Order(6) | Family(10) | Genus(16) | Species(22)
```

Train bằng **masked CE** (mask logit ngoài nhánh taxonomy hợp lệ về −∞ trước khi tính CE), suy luận bằng **masked top-down decoding** → mọi prediction là một path hợp lệ trong cây. Kết quả tốt nhất: yolo11x + masked CE = **95.0% HierAcc** trên 241 ảnh test.

### 1.2 Yêu cầu tăng tiến

Trong vận hành thực tế:

- **Base session** `S₀`: dữ liệu hiện có (22 loài).
- Các **incremental session** `S₁, S₂, …`: mỗi đợt là một dataset mới, có thể chứa:
  - **Loài mới (new leaf)** dưới một genus đã có (ví dụ thêm 1 species vào genus 5, cạnh `chro/chro_1/chro_2`). Chỉ nở `species head` +1.
  - **Loài mới kéo theo node nội mới (new branch)**: một genus/family/order mới. Nở nhiều head, thay đổi topology cây.
  - **Thêm ảnh cho loài cũ (data-incremental)**: không thêm class, chỉ bổ sung mẫu.
- Mỗi đợt **đầy đủ taxonomy** (biết chính xác Order/Family/Genus/Species của từng ảnh).

### 1.3 Ràng buộc (constraints)

| Ràng buộc | Diễn giải |
|---|---|
| **Giữ kiến trúc** | Vẫn YOLO multi-head CE-masked; chỉ được nở chiều output head + thêm số hạng loss. |
| **Không retrain toàn bộ** | Không train lại trên *toàn bộ* data cũ mỗi session. Chỉ data mới + một buffer nhỏ. |
| **Không catastrophic forgetting** | HierAcc trên loài cũ không được tụt đáng kể sau khi học loài mới. |
| **Giữ consistency** | Prediction luôn là path hợp lệ trong cây (đã có nhờ masked decoding). |

### 1.4 Vì sao là CIL chứ không phải Online Learning

Lõi bài toán là **label space nở ra** (class mới thêm dần) → thuộc về Continual Learning / CIL. Online Learning cổ điển (regret minimization, label space cố định, single-pass streaming) **không** mô tả đúng bài toán này. Nếu sau này cần ràng buộc "online/streaming, single-pass", xem biến thể OCL ở §3.9 (SLDA per-level) — nhưng đó là tùy chọn, không bắt buộc.

> **Lưu ý lượng dữ liệu mới**: để ngỏ — có thể nhiều hoặc ít ảnh/loài mới. CIL phủ cả hai. Khi loài mới *ít ảnh*, bài toán đặc biệt hóa thành **Few-Shot CIL (FSCIL)** và thành phần imprinting/prototype (§3.4) trở nên quan trọng hơn; khi *nhiều ảnh*, có thể fine-tune head mạnh hơn hoặc mở một phần backbone (§3.7). Cùng một phương pháp, chỉ đổi trọng tâm.

---

## 2. Cơ sở lý thuyết (Theoretical Basis)

### 2.1 Catastrophic forgetting & stability–plasticity

Khi cập nhật mạng cho class mới, gradient ghi đè trọng số mã hóa class cũ → **catastrophic forgetting**. Mọi phương pháp CL là một cách cân bằng **stability** (giữ kiến thức cũ) ↔ **plasticity** (học cái mới).

### 2.2 Ba kịch bản CL và tại sao CIL khó nhất

Theo van de Ven & Tolias:
- **Task-IL**: có task ID lúc test → dễ nhất (mask theo task).
- **Domain-IL**: cùng label, đổi phân phối input.
- **Class-IL (CIL)**: class mới thêm dần, **không** task ID lúc test, phải phân biệt giữa *tất cả* class đã học → khó nhất.

CIL khó vì: (1) không có task ID để thu hẹp lựa chọn; (2) output head nở dần; (3) **task-recency bias** — FC layer thiên về class học gần đây, dìm logit class cũ. Thực nghiệm (van de Ven & Tolias; cả survey của nhóm bạn) cho thấy ở CIL: **regularization thuần (EWC) sụp; replay là gần như bắt buộc.**

### 2.3 Năm họ phương pháp CL và cái ta dùng

| Họ | Ý tưởng | Dùng trong H-CIL? |
|---|---|---|
| Regularization | Phạt thay đổi tham số quan trọng (EWC) / giữ output cũ (LwF) | **LwF (distillation): có**, làm add-on. EWC: chỉ làm baseline. |
| **Replay** | Lưu & ôn lại mẫu cũ (ER, iCaRL) | **Có — trục chính.** Survey gọi ER là "versatile nhất". |
| Optimization | Chiếu gradient tránh xung đột (GEM/A-GEM) | Không (nặng); có thể nhắc trong related work. |
| **Architecture** | Mở rộng/cô lập cấu trúc | **Có** — head expansion (nhẹ). PackNet *không* hợp (cần task ID). |
| **Representation** | Feature mạnh, bền (pretrained, frozen) | **Có — trụ cột.** Frozen backbone. |

### 2.4 iCaRL — nền tảng phương pháp

iCaRL (Rebuffi et al., 2017) = **exemplar memory (rehearsal) + herding (chọn exemplar) + knowledge distillation + nearest-mean-of-exemplars (NCM) classification**. Đây là method CIL kinh điển nhất; phương án của ta là một biến thể *phân cấp* của nó.

### 2.5 Weight imprinting

Qi et al. (2018): khởi tạo trọng số class mới = **mean feature (L2-normalized)** của ảnh class đó. Cho phép classify class mới ngay cả khi rất ít ảnh (few-shot) → khớp với thực tế loài phấn hoa mới thường khan dữ liệu.

### 2.6 Hierarchy là đòn bẩy, không phải gánh nặng

1. **Known parent path**: nhà sinh học cho biết loài mới thuộc Order/Family/Genus nào → không phải "khám phá" vị trí như CIL phẳng.
2. **Masked decoding khu trú quyết định**: mỗi cấp chỉ chọn trong *siblings* của parent đã dự đoán. Điều này tạo ra một dạng *task-localization có cấu trúc, không cần task ID* → giảm nhẹ chính **task-recency bias** (lệch scale logit giữa class cũ–mới trên *toàn* label space ít ảnh hưởng hơn nhiều so với softmax phẳng 22+k chiều).
3. **Consistency guarantee** được bảo toàn miễn phí qua các session (chỉ cần extend parent arrays).
4. **Graceful degradation**: sai cấp Species vẫn có thể đúng Genus/Family/Order.

### 2.7 Định vị so với literature (trung thực)

Hierarchical/taxonomy-aware CIL **đã tồn tại** (xem §8): HLE (Lee et al., ICCV 2023) làm "hierarchical label expansion" với rehearsal + hierarchy-aware pseudo-labeling; các method CLIP/prompt 2024–2025 dùng cây ngữ nghĩa để giảm forgetting. **Không được claim "phương pháp đầu tiên".** Đóng góp ở đây nằm ở:
- **Applied**: phân loại phấn hoa với taxonomy sinh học ground-truth (không phải cây ngữ nghĩa do LLM/expert suy ra).
- **Kiến trúc**: multi-head masked-CE với **hard consistency guarantee** — khác hẳn dòng CLIP/prompt/hyperbolic.
- **Phạm vi tăng tiến**: thêm cả **node nội mới** (genus/family/order), không chỉ leaf.

### 2.8 SOTA hiện đại: PTM-based CIL — vì sao frozen-backbone là ĐÚNG hướng

Trọng tâm field đã dịch từ "train from scratch" (regime mà iCaRL/EWC/LwF 2016–2019 sinh ra để giải) sang **PTM-based CIL**: dùng backbone *pretrained đóng băng* + thích nghi class nhẹ. Hai dòng chính:

- **Prototype/analytic trên feature đóng băng** *(liên quan trực tiếp — dùng được cho CNN/YOLO)*:
  - **SimpleCIL + Aper/ADAM** (Zhou et al., IJCV 2024): chỉ đặt trọng số classifier = embedding trung bình mỗi class trên PTM đóng băng đã vượt SOTA, *kể cả không tinh chỉnh downstream*. **Bước imprinting của ta (§3.4) chính là cái này, áp per-level.**
  - **RanPAC** (McDonnell et al., NeurIPS 2023): chèn random projection đóng băng (kèm phi tuyến) + tích luỹ class-prototype; *training-free*, *không cần rehearsal*, giảm error mạnh trên 7 benchmark. → ứng viên nâng cấp V2.
- **Prompt/adapter trên ViT** *(KHÔNG dùng trực tiếp — backbone ta là CNN)*: L2P, DualPrompt, CODA-Prompt, S-Prompt, CAPrompt. Học prompt để suy task ID rồi inject vào ViT; SOTA trên benchmark ViT nhưng không drop-in cho YOLO. Chỉ để cite related work.
- **Dynamic architecture** (DER, FOSTER, MEMO): mạnh nhưng nở backbone → **vi phạm** ràng buộc "giữ kiến trúc".

**Hệ quả cho ta**: lựa chọn frozen-YOLO-backbone *trùng* với consensus PTM-CIL hiện đại — không lỗi thời dù dùng rehearsal kiểu iCaRL làm thành phần. Nên đóng khung rõ trong luận văn là **"PTM-based CIL"** và so với SimpleCIL/RanPAC. **Khác biệt của ta so với toàn bộ dòng PTM-CIL kia: chúng đều *phẳng*, không có taxonomy + hard consistency guarantee** — đó là chỗ đứng riêng.

---

## 3. Workflow & Kiến trúc Phương pháp

### 3.1 Sơ đồ tổng quan

```
                    ┌──────────────── SESSION Sₖ (k ≥ 1) ────────────────┐
 Base ckpt Sₖ₋₁ ──► │  1. Load model + species2id + parent arrays + memory │
 (model, memory)    │  2. teacher ← snapshot(model)  [đóng băng, eval]     │
                    │  3. Đọc mapping CSV mới → tính DELTA id mới          │
                    │  4. Trích feature loài mới (backbone đóng băng)      │
                    │  5. Nở head theo DELTA + imprinting prototype        │
                    │  6. Extend parent arrays                             │
                    │  7. Train set = data mới + exemplar replay           │
                    │  8. Fine-tune HEAD (backbone frozen):                │
                    │        L = masked_CE(mới+replay) + β·KD(class cũ)     │
                    │  9. Cập nhật exemplar memory (herding)               │
                    │ 10. (tùy chọn) tính lại NCM prototype                │
                    │ 11. Lưu ckpt Sₖ (đã extend)                          │
                    └──────────────────────────────────────────────────────┘
 Inference: masked_top_down_decode với parent arrays đã extend
            (mặc định dùng logit head; tùy chọn NCM)
```

### 3.2 Frozen backbone — trụ cột chống forgetting

Sau base session, **đóng băng toàn bộ backbone YOLO** (`requires_grad=False`). Hệ quả:
- Feature `f(x)` cố định ⇒ phân phối feature của loài cũ không trôi ⇒ gần như **0 forgetting ở tầng feature**.
- Mỗi session chỉ train 4 head Linear nhỏ ⇒ rất rẻ, không backprop qua ~50M params.
- Cho phép **cache feature** (xem §3.8) nếu muốn.

Chỉ mở backbone khi gặp loài *quá OOD* (escalation §3.7).

### 3.3 Dynamic head expansion

Khi session có class mới ở cấp `l`, nở `Linear(1280, Nₗ) → Linear(1280, Nₗ + Δₗ)`, **copy nguyên trọng số cũ**, init hàng mới bằng imprinting. (Code §4.2.)

### 3.4 Imprinting khởi tạo class mới

Với mỗi loài mới `c`: forward toàn bộ ảnh của `c` qua backbone đóng băng → lấy mean feature `μ_c`, L2-normalize → gán vào hàng trọng số mới của species head (và prototype cho các node nội mới tương ứng). Few-shot vẫn classify được ngay.

> **Liên hệ SOTA (§2.8)**: bước imprinting này *chính là* prototype-classifier kiểu **SimpleCIL** (Zhou et al., IJCV 2024) áp per-level. Một biến thể "train-free hoàn toàn" (chỉ prototype, **không** fine-tune head) là một **baseline hiện đại rất mạnh** — nên implement để so. Nâng cấp V2: chèn **random projection đóng băng kiểu RanPAC** trước khi tính prototype để tăng linear separability.

### 3.5 Exemplar memory + herding + distillation

- **Memory**: giữ một tập ảnh đại diện mỗi loài *đã học*. Hai chế độ ngân sách:
  - *Fixed-per-class*: `m` ảnh/loài (vd m=20).
  - *Fixed-total* (iCaRL gốc): tổng `K` ảnh, `m = K // n_classes` (giảm dần khi class tăng).
- **Herding** (iCaRL): chọn exemplar sao cho mean của chúng xấp xỉ mean feature của class (greedy). Tốt hơn random.
- **Hierarchy-aware (đề xuất)**: ưu tiên ngân sách cho *siblings* của loài mới đến (vì masked decoding chỉ so trong siblings) → buffer nhỏ mà hiệu quả.
- **Distillation (LwF)**: teacher = model trước session; với mỗi head, phạt KL giữa softmax(logit class-cũ) của student và teacher (trên data đi qua = mới + replay). Ghim hành vi cũ. (Code §4.4.)

### 3.6 Loss mỗi session

```
L_total = L_masked_CE(logits, labels)                 # masked CE như base, trên (data mới + replay)
        + β · Σ_{level∈{O,F,G,S}} KD(z_level[:, :n_old_level], teacher)   # distillation trên slice class-cũ
```

`masked_CE` giữ nguyên (`multitask_ce_masked` từ `train_hierarchical_masked.py`) ⇒ **consistency guarantee không đổi**. `β` (vd 0.5–2.0) cân bằng plasticity/stability; T (temperature KD) ~2.

### 3.7 Escalation (khi frozen feature không tách được loài mới)

Nếu validate thấy loài mới lẫn siblings ở cấp species (frozen feature không đủ):
1. Unfreeze **block cuối** của backbone, LR thấp (vd 1e-4), 1–2 block thôi.
2. Tăng vai trò replay + distillation (để bảo vệ cũ khi backbone đổi).
3. **Re-extract cache** feature cũ sau session (vì feature đã thay đổi → cache cũ stale).

### 3.8 (Tùy chọn) "Cheap upper-bound" — KHÔNG phải CL

Vì backbone đóng băng, có thể **cache feature toàn bộ ảnh cũ một lần** (~2.237×1280 float ≈ 45MB) rồi **refit head trên (feature cũ cached + mới)** → bằng joint-training phần head, **0 forgetting**, chi phí cực thấp. Đây **không phải** continual learning (dùng toàn bộ data cũ) → trong tài liệu này nó đóng vai **upper bound baseline** (§7), không phải phương pháp chính.

### 3.9 (Tùy chọn) Biến thể OCL thuần — SLDA per-level

Nếu cần "update tức thì, single-pass, không backprop": chạy **Deep SLDA / NCM per-level** trên feature đóng băng + masked decoding. Forgetting-free by construction, trần accuracy thấp hơn. Dùng làm baseline online hoặc lựa chọn production cực nhẹ.

---

## 4. Cách Implement (Implementation)

### 4.1 Tổ chức mã nguồn

Tái dùng (không sửa logic):
- `train_hierarchical.py` → `YOLOHierarchicalClassifier` (L170–216), `HierarchicalDataset` (L127–164), `build_parent_arrays` (L104–121).
- `train_hierarchical_masked.py` → `multitask_ce_masked` (L211–249).
- `test_yolo.py` → `masked_decode` (L176–203).

Thêm mới (đề xuất đặt trong `ultralytics/models/yolo/classify/incremental/`):

```
incremental/
├── taxonomy.py          # rebuild/extend species2id + parent arrays (APPEND-ONLY)
├── model_expand.py      # nở head + imprinting
├── exemplar_memory.py   # ExemplarMemory + herding
├── distill.py           # distillation loss
├── train_incremental.py # vòng lặp 1 session
└── eval_incremental.py  # metrics CIL theo session (AA/FM/BWT/HierAcc/consistency)
```

### 4.2 ⚠️ Yêu cầu cứng: ID phải APPEND-ONLY

Trong base, **Species ID suy từ thứ tự alphabet tên class** (`build_species_ids`, `train_hierarchical.py:78`). Nếu thêm loài mới rồi sort lại alphabet, **toàn bộ ID cũ bị đánh số lại → vỡ ánh xạ head ↔ class → forgetting giả tạo.**

Bắt buộc chuyển sang **append-only**:
- Persist `species2id` trong checkpoint; loài cũ giữ nguyên ID; loài mới nhận `max_id + 1, +2, …`.
- Tương tự `ordor/familia/genus` ID lấy từ mapping CSV: các CSV qua các session **phải giữ nguyên ID cũ**, chỉ cấp ID mới (max+1) cho node mới.

```python
# incremental/taxonomy.py
import pandas as pd

def extend_taxonomy(prev_state, mapping_csv_new):
    """
    prev_state: dict từ checkpoint cũ chứa species2id, ordor/familia/genus maps, parent arrays.
    Trả về state mới + DELTA (số id mới mỗi cấp) — APPEND-ONLY.
    """
    df = pd.read_csv(mapping_csv_new)  # cột: class_name, ordor, familia, genus  (giữ typo 'ordor')
    sp2id   = dict(prev_state["species2id"])
    # map id cấp cao: giữ nguyên cũ, cấp mới = max+1
    def next_id(existing):  # existing: set/dict các id đang dùng
        return (max(existing) + 1) if existing else 0

    ordor_ids  = set(prev_state["ordor_set"])
    fam_ids    = set(prev_state["familia_set"])
    gen_ids    = set(prev_state["genus_set"])

    delta = {"order": 0, "family": 0, "genus": 0, "species": 0}
    # parent arrays cũ (list), sẽ append
    fam_parent = list(prev_state["familia_parent"])   # fam_parent[familia_id] = ordor cha
    gen_parent = list(prev_state["genus_parent"])     # gen_parent[genus_id]   = familia cha
    sp_parent  = list(prev_state["species_parent"])   # sp_parent[species_id]  = genus cha

    for _, r in df.iterrows():
        o, f, g, name = int(r["ordor"]), int(r["familia"]), int(r["genus"]), r["class_name"]
        # các node nội: ID đã do CSV cấp; chỉ cần đảm bảo parent arrays đủ dài
        if o not in ordor_ids: ordor_ids.add(o); delta["order"] += 1
        if f not in fam_ids:
            fam_ids.add(f); delta["family"] += 1
            _ensure_len(fam_parent, f); fam_parent[f] = o
        if g not in gen_ids:
            gen_ids.add(g); delta["genus"] += 1
            _ensure_len(gen_parent, g); gen_parent[g] = f
        if name not in sp2id:                      # LOÀI MỚI → cấp species id mới
            sid = len(sp2id); sp2id[name] = sid; delta["species"] += 1
            _ensure_len(sp_parent, sid); sp_parent[sid] = g

    new_state = {
        "species2id": sp2id,
        "ordor_set": sorted(ordor_ids), "familia_set": sorted(fam_ids), "genus_set": sorted(gen_ids),
        "familia_parent": fam_parent, "genus_parent": gen_parent, "species_parent": sp_parent,
        "n_classes": {"order": _n(ordor_ids), "family": len(fam_parent),
                      "genus": len(gen_parent), "species": len(sp2id)},
    }
    return new_state, delta

def _ensure_len(lst, idx, fill=-1):
    while len(lst) <= idx: lst.append(fill)

def _n(idset):  # n_ordor = max+1 theo convention base (kể cả id 0 không dùng)
    return max(idset) + 1
```

### 4.3 Nở head + imprinting

```python
# incremental/model_expand.py
import torch, torch.nn as nn

@torch.no_grad()
def expand_linear(head: nn.Linear, n_new: int, proto: torch.Tensor | None = None) -> nn.Linear:
    """Nở Linear thêm n_new output, giữ nguyên trọng số cũ. proto: (n_new, in_features) đã L2-norm."""
    if n_new <= 0: return head
    in_f, old_out = head.in_features, head.out_features
    new = nn.Linear(in_f, old_out + n_new, bias=head.bias is not None).to(head.weight.device)
    new.weight[:old_out].copy_(head.weight)
    if head.bias is not None: new.bias[:old_out].copy_(head.bias)
    if proto is not None:                       # imprinting
        scale = head.weight.norm(dim=1).mean()  # khớp độ lớn với hàng cũ
        new.weight[old_out:].copy_(proto * scale)
        if head.bias is not None: new.bias[old_out:].zero_()
    return new

@torch.no_grad()
def class_prototypes(backbone, images_loader, device) -> torch.Tensor:
    """Mean feature (L2-norm) cho 1 loài mới. images_loader: chỉ ảnh của loài đó."""
    feats = []
    for x in images_loader:
        f = backbone(x.to(device))             # (B, 1280) — backbone đã frozen+eval
        feats.append(torch.nn.functional.normalize(f, dim=1))
    mu = torch.cat(feats).mean(0)
    return torch.nn.functional.normalize(mu, dim=0)

def expand_model(model, delta, protos: dict):
    """protos: {'species': Tensor(Δs,1280), 'genus': ..., ...} đã L2-norm. Nở 4 head."""
    model.head_order   = expand_linear(model.head_order,   delta["order"],   protos.get("order"))
    model.head_family  = expand_linear(model.head_family,  delta["family"],  protos.get("family"))
    model.head_genus   = expand_linear(model.head_genus,   delta["genus"],   protos.get("genus"))
    model.head_species = expand_linear(model.head_species, delta["species"], protos.get("species"))
    return model
```
*(Tên thuộc tính head — `head_order…` — chỉnh theo `YOLOHierarchicalClassifier` thực tế.)*

### 4.4 Exemplar memory + herding

```python
# incremental/exemplar_memory.py
import torch, torch.nn.functional as F

class ExemplarMemory:
    def __init__(self, budget_per_class=20, mode="per_class", total_budget=2000):
        self.mode, self.m, self.K = mode, budget_per_class, total_budget
        self.store = {}            # species_id -> list[str] đường dẫn ảnh

    def _budget(self, n_classes):
        return self.m if self.mode == "per_class" else max(1, self.K // max(1, n_classes))

    @torch.no_grad()
    def herding(self, feats: torch.Tensor, paths: list, m: int) -> list:
        feats = F.normalize(feats, dim=1)
        mean = F.normalize(feats.mean(0), dim=0)
        chosen, run = [], torch.zeros_like(mean)
        for k in range(min(m, len(paths))):
            cand = (run + feats) / (k + 1)               # (N, D)
            d = (mean - cand).norm(dim=1)
            for j in chosen: d[j] = float("inf")
            j = int(d.argmin()); chosen.append(j); run += feats[j]
        return [paths[j] for j in chosen]

    def add_class(self, sid, feats, paths, n_classes_now):
        self.store[sid] = self.herding(feats, paths, self._budget(n_classes_now))

    def rebalance(self, n_classes_now):                  # fixed-total: cắt bớt khi class tăng
        if self.mode != "total": return
        m = self._budget(n_classes_now)
        for sid in self.store: self.store[sid] = self.store[sid][:m]

    def all_items(self):                                 # -> list[(path, sid)] để dựng replay loader
        return [(p, sid) for sid, ps in self.store.items() for p in ps]
```

### 4.5 Distillation loss

```python
# incremental/distill.py
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, T=2.0):
    """KD trên slice class-cũ của MỘT head. logits: (B, n_old)."""
    p_t = F.softmax(teacher_logits / T, dim=1)
    logp_s = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(logp_s, p_t, reduction="batchmean") * (T * T)

def multihead_kd(student_out, teacher_out, n_old: dict, T=2.0):
    """student_out/teacher_out: dict level->logits. n_old: số class cũ mỗi cấp (slice đầu)."""
    loss = 0.0
    for lv in ("order", "family", "genus", "species"):
        s = student_out[lv][:, :n_old[lv]]
        t = teacher_out[lv][:, :n_old[lv]]
        loss = loss + kd_loss(s, t, T)
    return loss
```

### 4.6 Vòng lặp một session

```python
# incremental/train_incremental.py  (rút gọn)
import copy, torch
from torch.utils.data import DataLoader, ConcatDataset

def run_session(ckpt_in, mapping_csv_new, new_data_root, ckpt_out,
                E=30, lr=1e-3, beta=1.0, T=2.0, freeze_backbone=True):
    ckpt = torch.load(ckpt_in, map_location="cpu")
    model = build_model_from_ckpt(ckpt)                       # YOLOHierarchicalClassifier + load state
    state, delta = extend_taxonomy(ckpt["tax_state"], mapping_csv_new)
    n_old = {lv: ckpt["n_classes"][lv] for lv in ckpt["n_classes"]}

    if freeze_backbone:
        for p in model.backbone.parameters(): p.requires_grad = False
    model.backbone.eval()

    teacher = copy.deepcopy(model).eval()                     # snapshot TRƯỚC khi nở head
    for p in teacher.parameters(): p.requires_grad = False

    # prototypes cho class mới (imprinting) — từ ảnh loài mới của session này
    protos = compute_all_new_protos(model.backbone, new_data_root, state, delta)
    model = expand_model(model, delta, protos)                # nở 4 head

    # parent arrays mới (tensor) cho masked CE / masked decode
    parents = to_parent_tensors(state)

    # dataset: mới + replay
    ds_new   = HierarchicalDataset(new_data_root, state, split="train", augment=True)
    ds_replay= ReplayDataset(memory.all_items(), state)       # dùng exemplar đã lưu trong ckpt
    loader   = DataLoader(ConcatDataset([ds_new, ds_replay]), batch_size=32, shuffle=True)

    params = [p for p in model.parameters() if p.requires_grad]   # chỉ head (+ block mở nếu escalate)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E*len(loader))

    for ep in range(E):
        for x, y in loader:                                   # y: dict {order,family,genus,species}
            x = x.to(dev); y = {k: v.to(dev) for k, v in y.items()}
            with torch.no_grad(): feat = model.backbone(x) if freeze_backbone else None
            out = model.heads_forward(feat if freeze_backbone else x)   # dict logits
            L_ce = multitask_ce_masked(out, y, parents, LAMBDA)         # masked CE (tái dùng)
            with torch.no_grad(): t_out = teacher.forward_logits(x)
            L_kd = multihead_kd(out, t_out, n_old, T)
            loss = L_ce + beta * L_kd
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    # cập nhật memory cho loài mới (herding) + rebalance
    update_memory_after_session(model.backbone, memory, new_data_root, state, delta)

    save_incremental_ckpt(ckpt_out, model, state, memory, meta={"session": ckpt["session"]+1,
                          "beta": beta, "T": T, "lr": lr})
```

### 4.7 Suy luận

Giữ nguyên `masked_decode` (`test_yolo.py:176–203`) với **parent arrays đã extend**. Hai chế độ:
- **(mặc định)** logit head + masked top-down decode.
- **(tùy chọn, chống recency-bias)** thay logit head bằng **NCM**: tính khoảng cách feature tới prototype mỗi class (mean của exemplars), rồi vẫn masked top-down decode theo cây.

### 4.8 Định dạng checkpoint mở rộng

Bổ sung vào checkpoint base (Phụ lục A của `HIERARCHICAL.md`):

```python
{
    # ... các field cũ: model_state, base_model, lambdas, masked_training ...
    "tax_state": {                       # taxonomy APPEND-ONLY
        "species2id": {...}, "ordor_set": [...], "familia_set": [...], "genus_set": [...],
        "familia_parent": [...], "genus_parent": [...], "species_parent": [...],
    },
    "n_classes": {"order": .., "family": .., "genus": .., "species": ..},
    "session": int,                      # 0 = base, 1,2,... = incremental
    "exemplar_memory": {sid: [paths]},   # hoặc lưu feature vector nếu không giữ được ảnh
    "incremental_cfg": {"beta": .., "T": .., "budget": .., "mode": ".."},
}
```

### 4.9 Hyperparameter gợi ý (session tăng tiến)

| Tham số | Giá trị khởi điểm | Ghi chú |
|---|---|---|
| `E` (epoch/session) | 20–40 | Ít hơn base (chỉ train head). |
| `lr` | 1e-3 (head) / 1e-4 (nếu mở backbone) | AdamW, wd=0.05, cosine. |
| `β` (KD) | 0.5–2.0 | ↑ stability, ↓ plasticity. Ablation. |
| `T` (KD temp) | 2.0 | |
| budget exemplar | 20/class hoặc total 2000 | iCaRL fixed-total nếu lo storage. |
| `LAMBDA` | (1.0, 0.9, 0.8, 0.7) | giữ như base. |
| seed | 42 (+123, 2024 cho multi-seed) | |

---

## 5. Dataset

### 5.1 Base (tóm tắt từ `HIERARCHICAL.md`)

pollen Đồng Văn: 22 species / 16 genus / 10 family / 6 order; 1.780 train / 216 val / 241 test. Thư mục `train|val|test/<class_name>/*.jpg`. Mapping `pollen_dong_van.csv` cột `class_name, ordor, familia, genus` (**giữ typo `ordor`** — load-bearing). Mất cân bằng Order (order 1+3 ≈ 82%). 7 loài có suffix `_1/_2` cùng triplet cha.

### 5.2 Định dạng dataset tăng tiến

Mỗi session `Sₖ` là một dataset cùng convention:
```
session_k/
├── train/<class_name>/*.jpg     # loài mới và/hoặc thêm ảnh loài cũ
├── val/<class_name>/*.jpg
└── (test gộp chung — xem 5.4)
mapping_session_k.csv            # class_name, ordor, familia, genus
```
**Quy tắc CSV (bắt buộc, xem §4.2)**: giữ nguyên (class_name → ordor/familia/genus ID) của các node cũ; chỉ cấp ID mới (max+1) cho node mới. Tệp mapping nên là **luỹ tích** (chứa cả class cũ) để dễ rebuild.

### 5.3 Ba kiểu dữ liệu mới — xử lý thống nhất

| Kiểu | DELTA | Hành động |
|---|---|---|
| Loài mới, genus cũ | species +1 | Nở species head, imprint, append `species_parent`. |
| Loài mới, kéo node nội mới | species +1 và genus/family/order +Δ | Nở các head tương ứng, append nhiều parent arrays. |
| Thêm ảnh loài cũ (data-incremental) | 0 | Không nở head; thêm ảnh vào train + **refresh exemplar** loài đó (chống drift nếu điều kiện chụp đổi). |

Không cần nhánh logic riêng — pipeline §4.6 tự xử lý theo DELTA.

### 5.4 Mô phỏng "stream" cho thực nghiệm (từ chính 22 loài)

Để đánh giá khi chưa có data tương lai thật, cắt 22 loài thành các session:
- **Base `S₀`**: ~17 loài (giữ nhiều ảnh, đủ học backbone tốt).
- **Incremental `S₁…S₅`**: thêm 1 loài/session, **cố ý chọn** vài loài là *new leaf* (vd thêm `chro_2` khi base chỉ có `chro/chro_1`) và vài loài *new branch* (loài kéo theo genus/family/order mới như `bras` ở order 4, `abel` ở order 2) để cover cả hai case.
- **Test cố định**: gộp test của *tất cả* loài đã học tới session đó (đo cả cũ lẫn mới). Giữ test base 241 ảnh + test loài mới riêng.
- Few-shot regime (tùy chọn): giới hạn K ảnh/loài mới (vd K∈{5,10}) để mô phỏng FSCIL.

> Trước khi cắt stream, **chốt với nhóm thu thập ý nghĩa `_1/_2`**: nếu chúng là cùng loài (chỉ khác pha/góc chụp) thì "loài mới" ≠ 22 mà là 15 → ảnh hưởng định nghĩa task và cách cắt session.

---

## 6. Cách Đánh giá (Evaluation)

### 6.1 Ma trận hiệu năng

Sau khi học session `i`, đo accuracy trên test của session `j` (j ≤ i) → `R[i][j]`. Từ đó tính các chỉ số CL chuẩn.

### 6.2 Chỉ số (đo trên cây, không chỉ flat)

| Chỉ số | Định nghĩa | Mục tiêu |
|---|---|---|
| **HierAcc theo session** | tỉ lệ đúng *đồng thời cả 4 cấp*, đo trên test luỹ tích sau mỗi session | giữ cao |
| **Per-level AA** | accuracy trung bình mỗi cấp (Order/Family/Genus/Species) qua các task đã học | giữ cao |
| **Forgetting Measure (FM)** | `mean_j (max_i R[i][j] − R[last][j])` | thấp |
| **BWT** | ảnh hưởng học mới lên task cũ; âm = quên | gần 0 / dương |
| **Consistency rate** | % prediction là path hợp lệ trong cây | 100% (kỳ vọng, do masked decode) |
| **New-class acc (plasticity)** | accuracy trên loài *vừa thêm* | cao |
| **Old-class retention (stability)** | accuracy trên loài cũ sau khi thêm mới | cao |
| **Per-class acc** | phát hiện loài nào sụp; cross-ref số ảnh/loài | phân tích |

### 6.3 Baselines & ablation (cho luận văn)

- **Lower bound** — *naive finetune*: chỉ train head trên data mới, **không** replay/distill. Kỳ vọng forgetting nặng (chứng minh vấn đề).
- **Upper bound** — *joint / cache-refit* (§3.8): refit head trên toàn bộ feature cũ+mới. Trần lý thuyết (không phải CL).
- **So với method trong survey** trên đúng setup: **EWC** (kỳ vọng sụp ở CIL), **LwF** (distill-only), **ER/iCaRL phẳng** (không hierarchy).
- **So với baseline hiện đại (PTM-CIL, §2.8)**: **SimpleCIL** (frozen prototype per-level, *không train* — dễ implement nhất, nối thẳng consensus); tùy chọn **RanPAC** (random-projection + prototype, train-free). Cho thấy phương án của bạn vừa grounded vừa current.
- **Ablation then chốt** — *flat-CIL vs hierarchical-CIL*: cùng phương pháp (replay+distill+expand), **tắt/bật masked decoding + masked CE**, để **cô lập đóng góp của cây**. Đây là thí nghiệm trung tâm chứng minh "hierarchy giúp gì" cho continual learning.
- **Ablation phụ**: β (KD weight), budget exemplar, herding vs random, frozen vs partial-unfreeze, fixed-per-class vs fixed-total memory.

### 6.4 Giao thức báo cáo

- Multi-seed (≥3: 42/123/2024), báo cáo **mean ± std** cho HierAcc cuối + FM.
- Vẽ **đường HierAcc theo session** (x = session, y = HierAcc luỹ tích) cho từng method → hình chính của chương.
- Bảng cuối: method × {HierAcc cuối, FM, BWT, #params thêm, consistency}.

### 6.5 Khung eval (rút gọn)

```python
# incremental/eval_incremental.py
def evaluate_session(model, parents, test_loaders_by_session, current_sid):
    R = {}
    for j, loader in test_loaders_by_session.items():
        if j > current_sid: continue
        hier, perlevel, valid = run_masked_decode_eval(model, parents, loader)
        R[j] = {"hieracc": hier, "perlevel": perlevel, "consistency": valid}
    return R

def forgetting_measure(R_history):           # R_history[i][j]['hieracc']
    fm = []
    last = max(R_history)
    for j in R_history[last]:
        best = max(R_history[i][j]["hieracc"] for i in R_history if j in R_history[i])
        fm.append(best - R_history[last][j]["hieracc"])
    return sum(fm)/len(fm)
```

---

## 7. Tóm tắt: phương pháp thỏa ràng buộc thế nào

| Ràng buộc (§1.3) | Cơ chế đáp ứng |
|---|---|
| Giữ kiến trúc YOLO multi-head masked-CE | Chỉ nở chiều output head + thêm số hạng KD; loss masked-CE & masked decode nguyên vẹn. |
| Không retrain toàn bộ data cũ | Train trên data mới + exemplar buffer nhỏ; backbone đóng băng. |
| Không catastrophic forgetting | Frozen features + replay (herding) + distillation (LwF) + lớp bảo vệ từ hierarchy. |
| Giữ consistency | Masked top-down decode + parent arrays append-only → path hợp lệ 100%. |

---

## 8. References

**Continual learning — scenario & nền tảng**
1. van de Ven, G. M., & Tolias, A. S. (2019). *Three scenarios for continual learning.* arXiv:1904.07734. (Bản tạp chí: van de Ven, Tuytelaars, Tolias, *Three types of incremental learning*, Nature Machine Intelligence 4:1185–1197, 2022.) — định nghĩa TIL/DIL/CIL; chứng minh EWC sụp ở CIL, replay gần như bắt buộc.
2. Kirkpatrick, J., et al. (2017). *Overcoming catastrophic forgetting in neural networks (EWC).* PNAS 114(13):3521–3526. — baseline regularization.
3. Li, Z., & Hoiem, D. (2017). *Learning without Forgetting (LwF).* TPAMI 40(12):2935–2947 (ECCV 2016; arXiv:1606.09282). — distillation; dùng làm thành phần KD.

**Class-incremental — replay & bias correction**
4. Rebuffi, S.-A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). *iCaRL: Incremental Classifier and Representation Learning.* CVPR 2017:5533–5542 (arXiv:1611.07725). — **nền chính**: exemplar + herding + distillation + NCM.
5. Welling, M. (2009). *Herding Dynamical Weights to Learn.* ICML 2009. — thuật toán herding chọn exemplar.
6. Wu, Y., et al. (2019). *Large Scale Incremental Learning (BiC).* CVPR 2019 (arXiv:1905.13260). — sửa task-recency bias.
7. Zhao, B., Xiao, X., Gan, G., Zhang, B., & Xia, S.-T. (2020). *Maintaining Discrimination and Fairness in Class Incremental Learning (Weight Aligning).* CVPR 2020. — sửa bias ở FC.

**Few-shot CIL (chế độ ít dữ liệu) & imprinting**
8. Tao, X., Hong, X., Chang, X., Dong, S., Wei, X., & Gong, Y. (2020). *Few-Shot Class-Incremental Learning.* CVPR 2020:12183–12192. — protocol FSCIL.
9. Zhang, C., Song, N., Lin, G., Zheng, Y., Pan, P., & Xu, Y. (2021). *Few-Shot Incremental Learning with Continually Evolved Classifiers (CEC).* CVPR 2021 (arXiv:2104.03047).
10. Qi, H., Brown, M., & Lowe, D. G. (2018). *Low-Shot Learning with Imprinted Weights.* CVPR 2018. — **imprinting khởi tạo class mới**.
11. Gidaris, S., & Komodakis, N. (2018). *Dynamic Few-Shot Visual Learning without Forgetting.* CVPR 2018. — cosine classifier + few-shot không quên.
12. Mensink, T., Verbeek, J., Perronnin, F., & Csurka, G. (2013). *Distance-Based Image Classification: Generalizing to New Classes at Near-Zero Cost (NCM).* TPAMI. — nền NCM.

**Online / streaming CL (biến thể OCL §3.9)**
13. Hayes, T. L., & Kanan, C. (2020). *Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis (SLDA).* CVPR Workshops 2020.
14. Lopez-Paz, D., & Ranzato, M. (2017). *Gradient Episodic Memory (GEM).* NeurIPS 2017. (A-GEM: Chaudhry et al., ICLR 2019.)

**Hierarchical / taxonomy-aware CIL — PRIOR ART để định vị novelty (đọc kỹ §2.7)**
15. Lee, B. H., Jung, O., Choi, J., & Chun, S. Y. (2023). *Online Continual Learning on Hierarchical Label Expansion (HLE).* ICCV 2023. — **gần nhất**: CIL phân cấp nhiều cấp, rehearsal + hierarchy-aware pseudo-labeling; mở rộng nhãn từ thô → mịn.
16. *Leveraging Hierarchical Taxonomies in Prompt-based Continual Learning.* (2024) arXiv:2410.04327. — cây taxonomy giảm forgetting (prompt-based).
17. *Hierarchical Semantic Tree Anchoring for CLIP-Based Class-Incremental Learning (HASTEN).* (2025) arXiv:2511.15633.
18. *Continual Hyperbolic Learning of Instances and Classes (HyperCLIC).* (2025) arXiv:2506.10710. — phân cấp trong không gian hyperbolic, distillation, "hierarchically better mistakes".
19. Bertinetto, L., et al. (2020). *Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks.* CVPR 2020. — tree-distance penalty (so sánh ở Method/Discussion).

**Hệ thống base & backbone**
20. [`HIERARCHICAL.md`](./HIERARCHICAL.md) — đặc tả hệ multi-head YOLO + masked CE (kiến trúc, dataset, masked decoding, checkpoint).
21. Jocher, G., et al. — Ultralytics YOLOv8 / YOLO11 (backbone classify).

**PTM-based CIL — SOTA hiện đại (§0, §2.8)**
22. Zhou, D.-W., Sun, H.-L., Ning, J., Ye, H.-J., & Zhan, D.-C. (2024). *Continual Learning with Pre-Trained Models: A Survey.* IJCAI 2024. — survey PTM-CIL.
23. Zhou, D.-W., Wang, Q.-W., Qi, Z.-H., Ye, H.-J., Zhan, D.-C., & Liu, Z. (2024). *Class-Incremental Learning: A Survey.* IEEE TPAMI 2024. — survey CIL tổng quát; kèm toolbox **PyCIL / PILOT** (rất hữu ích để dựng baseline).
24. Zhou, D.-W., Cai, Z.-W., Ye, H.-J., Zhan, D.-C., & Liu, Z. (2024). *Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need* **(SimpleCIL / Aper / ADAM)**. IJCV 2024 (arXiv:2303.07338). — frozen prototype baseline vượt SOTA; **nền cho bước imprinting của ta**.
25. McDonnell, M. D., Gong, D., Parvaneh, A., Abbasnejad, E., & van den Hengel, A. (2023). *RanPAC: Random Projections and Pre-trained Models for Continual Learning.* NeurIPS 2023 (arXiv:2307.02251). — frozen + random projection + prototype; train-free, không rehearsal. Ứng viên nâng cấp V2.
26. Wang, Z., et al. (2022). *Learning to Prompt for Continual Learning (L2P).* CVPR 2022; *DualPrompt*, ECCV 2022; Smith, J. S., et al., *CODA-Prompt*, CVPR 2023. — prompt-based PTM-CIL (ViT; chỉ cite, **không** drop-in cho CNN/YOLO).
27. Yan, S., Xie, J., & He, X. (2021). *DER: Dynamically Expandable Representation for Class Incremental Learning.* CVPR 2021. (FOSTER — Wang et al., ECCV 2022; MEMO — Zhou et al., ICLR 2023.) — dynamic-architecture CIL (nở backbone; **mâu thuẫn** ràng buộc của ta — liệt kê để loại trừ có lý do).

> **Lưu ý cho người viết paper/luận văn**: các citation trên đã được xác minh ở mức hội nghị/tạp chí + năm. Trước khi nộp, đối chiếu lại số trang/DOI và bổ sung các method FSCIL mới hơn (FACT — Zhou et al. CVPR 2022; SAVC; NC-FSCIL…) nếu cần làm dày Related Work. Đặc biệt **đọc HLE (ref 15) kỹ** để diff hóa đóng góp.

---

## Phụ lục A: Glossary

- **HierAcc**: hierarchical accuracy — đúng đồng thời cả 4 cấp.
- **CIL / FSCIL**: Class-Incremental / Few-Shot Class-Incremental Learning.
- **Exemplar / herding**: mẫu cũ lưu để replay / thuật toán chọn mẫu của iCaRL.
- **Imprinting**: khởi tạo trọng số class mới = mean feature L2-norm.
- **KD**: knowledge distillation (LwF).
- **DELTA**: số class mới thêm ở mỗi cấp trong một session.
- **Recency bias**: FC layer thiên về class học gần đây.

## Phụ lục B: Thứ tự triển khai gợi ý (cho agent)

1. `taxonomy.py` — extend append-only + DELTA. Test: thêm 1 loài, kiểm ID cũ không đổi, parent arrays đúng.
2. `model_expand.py` — nở head + imprinting. Test: logit loài cũ trước/sau khi nở giống nhau (frozen feature).
3. `exemplar_memory.py` — herding. Test: mean của exemplar xấp xỉ class mean.
4. `distill.py` — KD per-head trên slice cũ.
5. `train_incremental.py` — ráp lại; chạy 1 session đồ chơi (base 17 loài → +1 loài).
6. `eval_incremental.py` — R[i][j], FM/BWT/HierAcc/consistency theo session.
7. Chạy lower/upper bound + ablation flat-vs-hierarchical.

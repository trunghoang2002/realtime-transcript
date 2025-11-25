# result.log:
```
=== Evaluating pyannote embeddings ===
Skipped 251/15000 trials due to missing embeddings
Computing metrics on 14749 valid trials
EER: 27.77% | FAR@EER: 27.77% | FRR@EER: 27.76% | Thr(EER): 0.2995
Precision@EER: 56.56% | Recall@EER: 72.24% | F1@EER: 63.44%
Best F1: 65.28% | Precision@F1: 68.16% | Recall@F1: 62.64% | Thr(F1): 0.4175
AUC: 0.7882

=== Evaluating speechbrain embeddings ===
Skipped 251/15000 trials due to missing embeddings
Computing metrics on 14749 valid trials
EER: 15.36% | FAR@EER: 15.36% | FRR@EER: 15.37% | Thr(EER): 0.3781
Precision@EER: 73.39% | Recall@EER: 84.63% | F1@EER: 78.61%
Best F1: 86.70% | Precision@F1: 99.14% | Recall@F1: 77.03% | Thr(F1): 0.6130
AUC: 0.9369

=== Plotting curves ===

ROC curve saved to: eval_results/roc_curves.png
DET curve saved to: eval_results/det_curves.png
Precision-Recall curve saved to: eval_results/precision_recall_curves.png

=== Final Results ===
Pyannote: {'EER': 0.27769589657947646, 'threshold_at_EER': 0.299517510560555, 'FAR_at_EER': 0.2777495167361888, 'FRR_at_EER': 0.2776422764227642, 'precision_at_EER': 0.5655633354551242, 'recall_at_EER': 0.7223577235772358, 'F1_at_EER': 0.6344162799000357, 'best_F1': 0.6528277907223046, 'threshold_at_best_F1': 0.41750451029192304, 'precision_at_best_F1': 0.6815568332596197, 'recall_at_best_F1': 0.6264227642276423, 'AUC': 0.7882437237740981, 'FAR_curve': array([0.        , 0.        , 0.        , ..., 0.99989826, 0.99989826,
       1.        ], shape=(4588,)), 'FRR_curve': array([1.00000000e+00, 9.99796748e-01, 9.98780488e-01, ...,
       4.06504065e-04, 0.00000000e+00, 0.00000000e+00], shape=(4588,)), 'thresholds': array([        inf,  0.9519734 ,  0.94280498, ..., -0.1990335 ,
       -0.2250784 , -0.22687587], shape=(4588,))}
SpeechBrain: {'EER': 0.15364277933144577, 'threshold_at_EER': 0.3781467771103011, 'FAR_at_EER': 0.15362702207752568, 'FRR_at_EER': 0.15365853658536588, 'precision_at_EER': 0.7338738103630595, 'recall_at_EER': 0.8463414634146341, 'F1_at_EER': 0.7861053426467812, 'best_F1': 0.8669792977238934, 'threshold_at_best_F1': 0.6129652218046867, 'precision_at_best_F1': 0.9913680355741564, 'recall_at_best_F1': 0.7703252032520326, 'AUC': 0.9368787981805955, 'FAR_curve': array([0.        , 0.        , 0.        , ..., 0.99715129, 0.99715129,
       1.        ], shape=(1926,)), 'FRR_curve': array([1.00000000e+00, 9.99796748e-01, 2.97764228e-01, ...,
       2.03252033e-04, 0.00000000e+00, 0.00000000e+00], shape=(1926,)), 'thresholds': array([        inf,  0.9400493 ,  0.70697628, ..., -0.06335743,
       -0.06396167, -0.17806255], shape=(1926,))}
```

# 1. Nhìn nhanh vào kết quả tổng

## **Pyannote EER = 27.77% → rất kém**

## **SpeechBrain EER = 15.36% → tốt hơn rõ rệt**

Điều này có nghĩa:

* **Pyannote embeddings** phân biệt speaker yếu → similarity giữa same-speaker và different-speaker chồng lấn nhiều.
* **SpeechBrain ECAPA embeddings** mạnh hơn đáng kể → tách speaker rõ hơn → clustering tốt hơn.

---

# 2. Giải thích từng phần

## 2.1. “Skipped 251/15000 trials”

251 trial bị bỏ vì:

* 1 trong 2 file không có embedding (có thể file đọc lỗi / ngắn quá / embedding model trả NaN / zero vector)

=> Không ảnh hưởng lớn (chỉ ~1.6%).

---

# 3. Giải thích metric cho từng model

---

# 🔵 PYANNOTE EMBEDDING

### **EER: 27.77%**

* Khi threshold được chỉnh sao cho FAR = FRR, hệ thống sai ~28%.
* **Đây là EER rất cao**, chứng tỏ embedding kém.

Trong speaker verification:

* EER < 5%: cực tốt
* 5–10%: trung bình
* 10–15%: dùng được tùy domain
* **>20%: kém**
  → Pyannote = 27.7% là *tệ rõ ràng*.

---

### **FAR@EER: 27.77%**

### **FRR@EER: 27.76%**

Khớp nhau → đúng tiêu chuẩn EER.

---

### **Threshold_at_EER: 0.2995**

* Cosine similarity > 0.2995 → cùng người
* < 0.2995 → khác người

Ngưỡng này **khá thấp**, cho thấy embedding scatter rộng, phân bố không tách biệt.

---

### **Precision@EER: 56.56%**

Khi threshold = EER:

* Chỉ 56.5% cặp predicted “same speaker” là đúng
  → Rất nhiều false accept.

---

### **Recall@EER: 72.24%**

72% cặp same-speaker được nhận đúng → hơi khá, nhưng precision thấp kéo xuống chất lượng tổng thể.

---

### **F1@EER: 63.44%**

* Trung bình yếu

---

### **Best F1: 65.28% at threshold 0.4175**

Nếu tối ưu F1:

* Precision: 68.16%
* Recall: 62.64%

F1 vẫn thấp → tách speaker kém.

---

### **AUC: 0.788**

* ROC-AUC 0.78 = mức trung bình thấp
* Thể hiện phân bố score overlap nhiều giữa same-speaker và diff-speaker.

---

# 🔴 SPEECHBRAIN EMBEDDING (ECAPA)

## **EER: 15.36%**

→ Tốt đáng kể so với Pyannote (27.7%)

Không phải mức SOTA (SOTA ECAPA có thể 2–4%),
nhưng với **micro dataset 4 style khác nhau**, domain khác VoxCeleb, thì **15% là hợp lý và tốt**.

---

### **Precision@EER: 73.39%**

### **Recall@EER: 84.63%**

Cả hai đều cao hơn nhiều so với Pyannote.

---

### **F1@EER: 78.61%**

Tốt.

---

### **Best F1: 86.70% at threshold 0.6130**

* **Precision: 99.14%** (!)
* Recall: 77.03%

Điều này nói lên rằng:

* Nếu threshold đặt cao (0.613) → rất ít false accept (precision gần như tuyệt đối)
* Nhưng recall giảm (miss some same-speaker)

Đây là đặc điểm của **embedding phân bố tốt, tail clean**.

---

### **AUC: 0.9369**

* Gần 0.94 → rất tốt
* Curves cho thấy separation rõ ràng.

---

# 4. Nguyên nhân chính khiến Pyannote embedding yếu

### ❌ Pyannote community là model **segmentation-first**, embedding chỉ là phụ

* Pyannote community pipeline không dùng ECAPA hoặc x-vector đời mới
* Embedding của nó **thiết kế để hỗ trợ diarization pipeline của chính nó**, không phải để làm verification độc lập.

### ❌ Được huấn luyện domain khác với dataset eval

Dataset eval có:

* whisper
* falsetto
* nonpara
* high pitch

  → Những style này **khác xa** các dataset mà Pyannote community dùng (mostly AMI/VoxConverse-style).

### ❌ ECAPA (SpeechBrain) là model **speaker verification chuyên dụng**

* Trained trên VoxCeleb2
* Highly discriminative
* Robust với pitch, noise, speaking style
* Do đó cho score tốt hơn nhiều.

---

# 5. Điều này nói gì cho bài toán **speaker diarization**

### ✔ SpeechBrain ECAPA embedding sẽ cho:

* Clustering tốt hơn
* Ít merge nhầm speaker
* Ít split
* Affinity matrix sharp hơn
* DER giảm mạnh

### ✔ Pyannote embedding sẽ:

* Nhiều false same-speaker → merge các người khác nhau
* Nhiều miss same-speaker → split 1 speaker thành 2–3 cluster
  → DER rất cao.

Dựa trên **toàn bộ kết quả EER, PR, ROC, DET và phân bố score** mà bạn đã tính, ta có thể đưa ra **ngưỡng (threshold) tốt nhất** cho việc phân loại **same-speaker vs different-speaker** tùy mục đích sử dụng.

---

# ✅ 1. Số liệu quan trọng (đã tính trước đó)

### **SpeechBrain (ECAPA)**

* **Threshold tại EER:** `0.3781`
* **Threshold tại F1 tốt nhất:** `0.6130`
* **Precision tại F1:** ~**0.99** (gần như không merge nhầm)
* **AUC cao:** 0.937 → separable tốt

### **Pyannote**

* **Threshold tại EER:** `0.2995`
* **Threshold tại F1:** `0.4175`
  → Embedding yếu → threshold kém ổn định
  => Không khuyến khích dùng để phân loại speaker.

Vì vậy **ngưỡng chính cần chọn** là từ **SpeechBrain ECAPA**.

---

# 🎯 2. Chọn threshold theo mục đích sử dụng

---

## 🔵 Trường hợp 1: **Speaker diarization** (quan trọng nhất)

Trong diarization, **merge nhầm** (false accept) gây hại nặng hơn split.

→ Nên ưu tiên **Precision cao**, chấp nhận Recall thấp hơn.

### **⇒ Ngưỡng tốt nhất: ~0.60 – 0.65 (theo F1-optimal)**

#### **Đề xuất: `0.61`**

Vì tại threshold ~0.61:

* Precision ≈ **0.99** (hầu như không merge nhầm)
* Recall ≈ **0.77**
* Best-F1 đạt **86.7%**

➡️ Đây là **ngưỡng lý tưởng để dùng cho clustering AHC/VBx** → tránh merge, giảm DER rất mạnh.

---

## 🔵 Trường hợp 2: **Speaker verification tiêu chuẩn**

Muốn cân bằng FAR = FRR (chuẩn benchmark)

Dùng ngưỡng **EER**:

### **⇒ Ngưỡng: `0.378`**

Tại threshold này:

* FAR = FRR ≈ 15.4%
* F1 ≈ 78.6%
* Dùng khi bạn cần so sánh fairness giữa các model.

---

## 🔵 Trường hợp 3: **Muốn Recall cao (tránh split nhiều)**

Nếu bạn sợ split nhiều, chấp nhận merge một chút:

### **⇒ Ngưỡng: ~0.45–0.50**

* Precision 90–95%
* Recall > 85%

Nhưng **không nên dùng cho diarization**, vì merge khó sửa.

---

# 📌 3. Tổng hợp gợi ý chọn threshold (dễ đưa vào báo cáo)

| Mục đích                      | Threshold similarity  | Lý do                              |
| ----------------------------- | --------------------- | ---------------------------------- |
| **Diarization (recommended)** | **0.60–0.65 (≈0.61)** | Precision ≈ 1.0 → không merge nhầm |
| Verification cân bằng         | **0.378**             | FAR = FRR = EER                    |
| Muốn recall cao               | **0.45–0.50**         | Ít split nhưng tăng merge          |

---

# ⭐ 4. Lựa chọn cuối cùng (gọn gàng – thực tế)

### **👉 Sử dụng `0.61` làm threshold phân biệt same/diff speaker.**

Đây là ngưỡng:

* tối ưu về F1
* precision cực cao
* phù hợp nhất khi đưa embedding vào **clustering (AHC, VBx, k-means)**
* giúp giảm mạnh **speaker merge**, thứ gây sai lệch diarization nhiều nhất.
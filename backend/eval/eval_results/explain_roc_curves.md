![DET Curves](roc_curves.png)

# ✅ 1. ROC Curve là gì?

ROC curve biểu diễn:

* **FAR (False Acceptance Rate)** trên trục X
* **TPR (True Positive Rate = 1 − FRR)** trên trục Y

ROC càng **cong lên phía góc trái trên** → model càng mạnh.

Đường chéo đen (AUC = 0.5) = random (không phân biệt được speaker).

---

# ✅ 2. Nhìn vào biểu đồ ta thấy

### ✔ Đường đỏ (SpeechBrain) luôn nằm **cao hơn** đường xanh (Pyannote) trên toàn bộ trục X

→ **SpeechBrain tốt hơn hoàn toàn ở mọi threshold**.

### ✔ AUC:

Đây là diện tích dưới đường ROC curve, đo khả năng phân biệt same-speaker và different-speaker của model.

ROC curve vẽ:

FAR (False Accept Rate)

TPR (True Positive Rate)

- AUC = 1.0 → phân biệt hoàn hảo
- AUC = 0.5 → random (không phân biệt được)
- AUC < 0.5 → tệ hơn random

Ý nghĩa:

- AUC cao → score distribution của two classes (same vs diff) tách biệt tốt
- AUC thấp → score chồng lấn, khó phân loại

* Pyannote: **0.788**
* SpeechBrain: **0.937**

SpeechBrain vượt Pyannote **gần 0.15 absolute**, một khoảng cách rất lớn.

### ✔ Điểm EER (chấm tròn màu đỏ & xanh)

* SpeechBrain dừng ở TPR ~0.85 tại FAR ~0.15 → **EER = 15.36%**
* Pyannote dừng ở TPR ~0.72 tại FAR ~0.28 → **EER = 27.77%**

→ Chênh lệch rõ ràng: Pyannote sai nhiều hơn gần gấp đôi.

---

# ✅ 3. Phân tích hình học ROC (giúp hiểu model)

### 🔴 SpeechBrain ROC:

* Bật lên rất nhanh → nghĩa là **FAR nhỏ nhưng TPR đã rất cao**
* Đường cong áp sát phía trên → ưu thế mạnh ở toàn miền

Điều này chứng minh:

* Embedding **separable tốt** (same-speaker score >> diff-speaker)
* False accept **ít**
* False reject **ít**
* Threshold rộng → ổn định cho clustering

---

### 🔵 Pyannote ROC:

* Bị kéo xuống → TPR thấp hơn ở mọi FAR
* Đường cong **ít cong**, gần hơn với random
  → embedding yếu, score overlap lớn

---

# ✅ 4. Ý nghĩa thực tiễn cho Speaker Diarization

**Với ROC như này, kết luận rất rõ:**

### ✔ SpeechBrain embedding sẽ:

* Ít merge (FAR thấp)
* Ít split (FRR thấp → TPR cao)
* Ma trận similarity sạch hơn
* AHC/VBx clustering ổn định hơn
* DER giảm đáng kể

### ❌ Pyannote embedding trong dạng “standalone”:

* FAR cao → merge nhiều speaker
* FRR cũng cao → split nhiều
* Dẫn đến DER cao trừ khi dùng cùng toàn bộ pipeline Pyannote đã tuning sẵn.

---

# 🔍 5. Giải thích chính xác cho EER point trên ROC

**EER point** là điểm trên ROC nơi:

* FAR = FRR
* Trên ROC: TPR = 1 − FRR
  → EER point xuất hiện tại nơi đường cong gần đường chéo đen

Trong hình có thể thấy:

* Pyannote EER point thấp hơn và lệch phải hơn
* SpeechBrain EER point cao hơn và lệch trái hơn

→ SpeechBrain tốt hơn rõ rệt.
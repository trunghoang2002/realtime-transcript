![PR Curves](precision_recall_curves.png)

# 1. Precision–Recall Curve là gì ?

* **Precision**: tỉ lệ dự đoán “same speaker” là đúng.
* **Recall**: tỉ lệ bắt đúng tất cả cặp same-speaker.
* PR curve rất quan trọng khi dữ liệu **mất cân bằng** (diff-speaker >> same-speaker) và khi ta muốn kiểm soát trade-off merge vs split trong diarization.

---

# 2. Tổng quan đồ thị

| Model             | Màu đường | PR-AUC | Best F1 | Precision@best F1 | Recall@best F1 |
| ----------------- | --------- | ------ | ------- | ----------------- | -------------- |
| Pyannote          | Xanh lam  | 0.727  | 65.28   | 68.16             | 62.64          |
| SpeechBrain ECAPA | Đỏ        | 0.924  | 86.70   | **99.14**         | 77.03          |
| NeMo Titanet      | Xanh lá   | 0.929  | 87.25   | 97.56             | 78.90          |
| NeMo ECAPA TDNN   | Vàng      | 0.928  | 87.27   | 97.56             | 78.94          |

* Ba đường **SpeechBrain + NeMo** bám sát trần Precision≈1 đến khi Recall ~0.8–0.9 → gần như không merge nhầm.
* **Pyannote** tụt dốc: Precision rơi xuống ~0.7 khi Recall đạt 0.8 → score overlap lớn.

---

# 3. Phân tích từng đường cong

### 🔵 Pyannote
* Precision giảm đều khi tăng Recall; vùng Precision>0.9 chỉ tồn tại ở Recall <0.2.
* Điều này cho thấy false accept tăng rất nhanh, khó duy trì cả precision và recall cao.

### 🟥 SpeechBrain ECAPA
* Giữ Precision ≈ 1 cho tới khi Recall ~0.8 rồi mới giảm.
* Tạo “vùng threshold an toàn” rộng: bạn có thể thay đổi ngưỡng khá nhiều mà precision vẫn >0.97.

### 🟢 NeMo Titanet & 🟡 NeMo ECAPA
* Hai đường gần như trùng nhau, nằm trên/áp sát đường đỏ ở đoạn cuối.
* Ở Recall ~0.85 vẫn giữ Precision >0.95 → phù hợp cho cả tasks cần recall cao (speaker search, active learning).

---

# 4. Best F1 points (các chấm tròn)

* **Pyannote**: F1 65.28% (Precision 0.68 / Recall 0.63) → chỉ đủ làm baseline.
* **SpeechBrain**: F1 86.70% (Precision 0.99 / Recall 0.77) → lý tưởng cho diarization ưu tiên tránh merge.
* **NeMo Titanet & ECAPA**: F1 ≈87.3% (Precision 0.975 / Recall ~0.79) → trade-off cân bằng hơn, tăng recall ~2% so với SpeechBrain trong khi precision vẫn rất cao.

---

# 5. Hàm ý thực tế

* **Chọn embedding**: Titanet hoặc NeMo ECAPA nếu bạn có GPU; SpeechBrain là lựa chọn nhẹ nhưng sát nút về hiệu năng; Pyannote chỉ nên dùng trong pipeline gốc của họ.
* **Tuning threshold**:
  * Dành cho diarization (ưu tiên precision): đặt cosine threshold ~0.59 (NeMo) hoặc 0.61 (SpeechBrain) tương ứng điểm best F1 → hầu như không merge.
  * Cần recall cao hơn (speaker search): có thể hạ threshold cho NeMo tới khi Precision ~0.95 (theo đoạn đuôi đường cong) để lấy Recall >0.9.
* **Clustering pipelines**: PR-AUC >0.92 giúp affinity matrix sắc nét, VBx/AHC ổn định hơn; Pyannote cần thêm heuristic để hạn chế merge.

---

# 6. Kết luận

* Khoảng cách PR-AUC giữa nhóm ECAPA/Titanet (~0.93) và Pyannote (~0.73) trùng khớp với bảng `result.log`, chứng minh lợi thế rõ ràng.
* Điểm best F1 thể hiện Precision ≥97% cho tất cả embedding ECAPA/Titanet, trong khi Pyannote chỉ 68%.
* Khi báo cáo, kết hợp đồ thị này với `explain_result.md` để giải thích vì sao ta chọn embedding NeMo/SpeechBrain cho production diarization.


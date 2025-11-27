![DET Curves](det_curves.png)

# 1. DET Curve là gì?

* Trục X: **False Acceptance Rate (FAR)**.
* Trục Y: **False Rejection Rate (FRR)**.
* DET sử dụng thang log-normal giúp phóng đại sự khác biệt ở vùng lỗi thấp.
* Điểm giao với đường nét đứt FAR = FRR chính là **Equal Error Rate (EER)**.

---

# 2. Tổng quan đồ thị

| Model             | Màu đường | EER (%) | FAR tại EER (%)  | FRR tại EER (%)  |
| ----------------- | --------- | ------- | ---------------- | ---------------- |
| Pyannote          | Xanh lam  | 27.77   | 27.77            | 27.76            |
| SpeechBrain ECAPA | Đỏ        | 15.36   | 15.36            | 15.37            |
| NeMo Titanet      | Xanh lá   | 14.65   | 14.65            | 14.65            |
| NeMo ECAPA TDNN   | Vàng      | 14.95   | 14.95            | 14.96            |

* Các đường **NeMo Titanet / NeMo ECAPA / SpeechBrain** nằm sát nhau ở vùng thấp trái → lỗi đều quanh 15%.
* **Pyannote** nằm hẳn phía trên bên phải → cần FAR cao hơn mới đạt cùng FRR → biểu hiện của embedding yếu.

---

# 3. Giải thích hình dạng từng đường

### 🔵 Pyannote
* Đường cong dốc chậm và treo cao: khi giảm FAR thì FRR vẫn >20%.
* Thậm chí ở FAR 10% vẫn còn FRR >25%, chứng tỏ score overlap lớn giữa same/diff speaker.

### 🟥 SpeechBrain ECAPA
* Nhanh chóng hạ xuống vùng FAR/FRR <20%.
* Đường cong khá mượt, song vẫn cao hơn hai model NeMo ở đoạn 5–15% FAR.

### 🟢 NeMo Titanet & 🟡 NeMo ECAPA
* Gần như trùng nhau và là đường thấp nhất trên toàn miền.
* Ở FAR 10% chúng chỉ có FRR ~13–14%, tiếp tục giảm khi FAR nhích lên → tốt nhất cho trade-off.

---

# 4. Các điểm EER (chấm màu)

* **Pyannote**: chấm xanh nằm ở FAR ≈ 28%, FRR ≈ 28% → lỗi gần gấp đôi so với nhóm còn lại.
* **SpeechBrain**: chấm đỏ tại FAR ≈ 15% / FRR ≈ 15% → giảm 12 điểm phần trăm so với Pyannote.
* **NeMo Titanet (xanh lá)** và **NeMo ECAPA (vàng)**: chấm nằm thấp nhất (~14.7%) → hiện đang dẫn đầu.

Nhìn ngang qua đường nét đứt FAR = FRR có thể thấy rõ thứ tự: Titanet ≈ NeMo ECAPA < SpeechBrain ≪ Pyannote.

---

# 5. Hàm ý thực tế

* **Chọn embedding**: nếu muốn EER thấp nhất, dùng NeMo Titanet hoặc NeMo ECAPA; SpeechBrain là lựa chọn nhẹ mà vẫn giữ EER ~15%.
* **Tuning threshold**:
  * Diarization ưu tiên tránh merge → đặt cosine threshold tương đương các điểm EER/Best-F1 đã ghi trong `result.log` (NeMo ~0.59, SpeechBrain ~0.61) để giữ FAR ~15% hoặc thấp hơn.
  * Nếu cần giảm FRR thêm, chấp nhận FAR nhỉnh hơn: đọc đoạn cuối đường cong (FAR 20% → FRR ~10% cho NeMo).
* **Pipeline Pyannote**: chỉ nên dùng embedding này khi chạy full pipeline của họ (có PLDA, re-scoring). Nếu dùng standalone, DET cho thấy sẽ có cả merge và split cao, kéo DER lên mạnh.

---

# 6. Cách dùng hình trong báo cáo

* Hình DET cho thấy rõ “khoảng cách an toàn” giữa nhóm ECAPA/Titanet và Pyannote ở tất cả vùng FAR.
* Đính kèm bảng EER ở trên + trích đường nét đứt để giải thích lý do lựa chọn embedding cuối cùng cho production.
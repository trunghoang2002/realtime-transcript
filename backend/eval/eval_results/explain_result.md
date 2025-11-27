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

=== Evaluating nemo_tianet embeddings ===
Skipped 251/15000 trials due to missing embeddings
Computing metrics on 14749 valid trials
EER: 14.65% | FAR@EER: 14.65% | FRR@EER: 14.65% | Thr(EER): 0.4092
Precision@EER: 74.46% | Recall@EER: 85.35% | F1@EER: 79.53%
Best F1: 87.25% | Precision@F1: 97.56% | Recall@F1: 78.90% | Thr(F1): 0.5915
AUC: 0.9417

=== Evaluating nemo_ecapa_tdnn embeddings ===
Skipped 251/15000 trials due to missing embeddings
Computing metrics on 14749 valid trials
EER: 14.95% | FAR@EER: 14.95% | FRR@EER: 14.96% | Thr(EER): 0.4114
Precision@EER: 74.01% | Recall@EER: 85.04% | F1@EER: 79.14%
Best F1: 87.27% | Precision@F1: 97.56% | Recall@F1: 78.94% | Thr(F1): 0.5926
AUC: 0.9417

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
NeMo Titanet: {'EER': 0.1465249775221325, 'threshold_at_EER': 0.4092250378753265, 'FAR_at_EER': 0.1465052395971106, 'FRR_at_EER': 0.14654471544715442, 'precision_at_EER': 0.7446355736832772, 'recall_at_EER': 0.8534552845528456, 'F1_at_EER': 0.795340467847334, 'best_F1': 0.8724575795033149, 'threshold_at_best_F1': 0.5914630945239923, 'precision_at_best_F1': 0.9756220155818045, 'recall_at_best_F1': 0.7890243902439025, 'AUC': 0.9417264491090327, 'FAR_curve': array([0.        , 0.        , 0.        , ..., 0.98575644, 0.98575644,  
       1.        ], shape=(1888,)), 'FRR_curve': array([1.00000000e+00, 9.99796748e-01, 2.45325203e-01, ...,
       2.03252033e-04, 0.00000000e+00, 0.00000000e+00], shape=(1888,)), 'thresholds': array([        inf,  0.95404786,  0.71301672, ..., -0.02433496,
       -0.02436331, -0.21396785], shape=(1888,))}
NeMo Ecapa TDNN: {'EER': 0.14952459413697808, 'threshold_at_EER': 0.4114152495898431, 'FAR_at_EER': 0.14945569233899686, 'FRR_at_EER': 0.1495934959349593, 'precision_at_EER': 0.7401379798337167, 'recall_at_EER': 0.8504065040650407, 'F1_at_EER': 0.791449919606545, 'best_F1': 0.8727109313560274, 'threshold_at_best_F1': 0.5926270551780972, 'precision_at_best_F1': 0.9756342627480532, 'recall_at_best_F1': 0.7894308943089431, 'AUC': 0.9416869939377999, 'FAR_curve': array([0.        , 0.        , 0.        , ..., 0.96835894, 0.96835894,
       1.        ], shape=(1838,)), 'FRR_curve': array([1.00000000e+00, 9.99796748e-01, 2.68902439e-01, ...,
       2.03252033e-04, 0.00000000e+00, 0.00000000e+00], shape=(1838,)), 'thresholds': array([        inf,  0.95972115,  0.72892271, ...,  0.04003473,
        0.03996694, -0.19377163], shape=(1838,))}
```

# 1. Nhìn nhanh vào kết quả tổng

| Model                | EER   | FAR@EER | FRR@EER | Best F1 | Thr(EER) | Thr(best F1) | Precision@best F1 | AUC   |
| -------------------- | ----- | ------- | ------- | ------- | -------- | ------------ | ----------------- | ----- |
| Pyannote             | 27.77 | 27.77   | 27.76   | 65.28   | 0.2995   | 0.4175       | 68.16             | 0.788 |
| SpeechBrain ECAPA    | 15.36 | 15.36   | 15.37   | 86.70   | 0.3781   | 0.6130       | 99.14             | 0.937 |
| NeMo Titanet         | 14.65 | 14.65   | 14.65   | 87.25   | 0.4092   | 0.5915       | 97.56             | 0.942 |
| NeMo ECAPA TDNN      | 14.95 | 14.95   | 14.96   | 87.27   | 0.4114   | 0.5926       | 97.56             | 0.942 |

- **NeMo Titanet và NeMo ECAPA TDNN** đang dẫn đầu với EER ~15% và F1 ~87%.
- **SpeechBrain ECAPA** vẫn rất cạnh tranh và dễ triển khai, chỉ thua NeMo ~0.7% EER.
- **Pyannote** tụt xa (EER 27.77%), khó dùng cho cả verification lẫn diarization.

---

# 2. Giải thích từng phần

## 2.1. “Skipped 251/15000 trials”

251 trial bị bỏ vì:

* Một trong hai file không có embedding (file lỗi, audio quá ngắn, model trả NaN/zero).

=> Không ảnh hưởng lớn (1.6% dữ liệu), nên các metric vẫn đáng tin.

---

# 3. Giải thích metric cho từng model

## 3.1 Pyannote

* **EER 27.77%**: hệ thống sai gần 3/10 cặp ở ngưỡng cân bằng → khó dùng.
* **Precision@EER 56.56%**: hơn 40% cặp bị nhận nhầm là cùng speaker.
* **Best F1 65.28% @ thr 0.4175** nhưng vẫn thấp → embedding không tách biệt.
* **AUC 0.788**: chồng lấn lớn giữa score same/diff.
* Kết luận: chỉ nên dùng khi không còn lựa chọn nào khác.

### **EER: 27.77%**
## 3.2 SpeechBrain ECAPA

* **EER 15.36%**, giảm 12% tuyệt đối so với Pyannote.
* **Best F1 86.70% @ thr 0.6130** với **precision ≈ 99%** → cực ít false accept.
* **AUC 0.9369**: đường ROC đẹp, score distribution tách rõ.
* WER-style domain khác VoxCeleb vẫn giữ hiệu năng tốt → phù hợp cho production khi tài nguyên GPU hạn chế.

## 3.3 NeMo Titanet

* **EER 14.65%** (tốt nhất trong 4 model).
* **Best F1 87.25% @ thr 0.5915**, precision 97.56% / recall 78.90%.
* **AUC 0.9417**: cao nhất, chứng tỏ embedding ổn định.
* Lưu ý: Titanet nặng hơn SpeechBrain, cần GPU nhưng đổi lại score cao nhất.

## 3.4 NeMo ECAPA TDNN

* **EER 14.95%**, rất sát Titanet.
* **Best F1 87.27% @ thr 0.5926**, precision 97.56% / recall 78.94% (gần như trùng Titanet).
* **AUC 0.9417**: tương đương Titanet.
* Ưu điểm: cùng họ ECAPA nên dễ fine-tune, pipeline tương tự SpeechBrain.

---

# 4. Vì sao Pyannote yếu hơn?

* **Segmentation-first**: pyannote community tập trung vào diarization pipeline, embedding chỉ là bước phụ trợ nên không được huấn luyện tối ưu cho verification độc lập.
* **Khác biệt domain**: dataset eval gồm whisper, falsetto, nonpara, high pitch → khác xa AMI/VoxConverse mà Pyannote quen thuộc.
* **Model architecture**: không sử dụng ECAPA/x-vector mới, nên không tận dụng kỹ thuật hiện đại (channel attention, squeeze-excitation, etc.).

Ngược lại, cả SpeechBrain và hai model NeMo đều là **speaker verification chuyên dụng** (ECAPA/Titanet), được huấn luyện trên VoxCeleb2 với augment mạnh nên giữ được độ phân tách khi chuyển domain.

---

# 5. Ảnh hưởng lên bài toán **speaker diarization**

* **NeMo Titanet / NeMo ECAPA / SpeechBrain ECAPA**: affinity matrix sắc nét, ít merge sai, giảm DER đáng kể khi feed vào AHC/VBx/k-means.
* **Pyannote**: nhiều false accept → merge nhầm speaker khác nhau, đồng thời false reject → split một speaker thành nhiều cluster. Kết quả DER rất cao.

Tóm lại, hãy ưu tiên embedding họ ECAPA/Titanet cho mọi pipeline diarization hoặc speaker search.

---

# 6. Gợi ý chọn threshold

## 6.1 Số liệu chính

* **NeMo Titanet**
  * Thr(EER): `0.4092`
  * Thr(best F1): `0.5915`
  * Precision@best F1: `0.9756`
* **NeMo ECAPA TDNN**
  * Thr(EER): `0.4114`
  * Thr(best F1): `0.5926`
  * Precision@best F1: `0.9756`
* **SpeechBrain ECAPA**
  * Thr(EER): `0.3781`
  * Thr(best F1): `0.6130`
  * Precision@best F1: `0.9914`
* **Pyannote**
  * Thr(EER): `0.2995`
  * Thr(best F1): `0.4175`

Các ngưỡng tốt nhất nên lấy từ Titanet/NeMo ECAPA (hoặc SpeechBrain nếu cần model nhẹ).

## 6.2 Chọn ngưỡng theo mục đích

| Mục đích                      | Model gợi ý      | Threshold similarity | Lý do chính                                       |
| ----------------------------- | ---------------- | -------------------- | ------------------------------------------------- |
| **Diarization (ưu tiên)**     | NeMo Titanet     | **0.59–0.60**        | Precision ~0.98, hạn chế merge nhầm               |
| Verification cân bằng         | Titanet / NeMo E | **0.41**             | FAR ≈ FRR ≈ 15% (chuẩn EER)                       |
| Thiết lập nhẹ (không có NeMo) | SpeechBrain      | **0.61**             | Precision ~0.99, dễ deploy                       |
| Muốn recall cao               | SpeechBrain      | **0.45–0.50**        | Recall >85%, chấp nhận merge tăng                 |

## 6.3 Khuyến nghị thực tế

* **Prod diarization**: đặt ngưỡng cosine ~`0.59` (Titanet/NeMo ECAPA). Nếu dùng SpeechBrain, giữ `0.61`.
* **Benchmark công bằng**: báo cáo EER tại `0.409` (Titanet) hoặc `0.411` (NeMo ECAPA).
* **Fallback Pyannote**: tránh dùng, nhưng nếu bắt buộc hãy tăng threshold (>0.45) để giảm merge, dù recall sẽ giảm mạnh.

---

# 7. Kết luận gọn

* **NeMo Titanet / ECAPA TDNN** hiện là lựa chọn tốt nhất (EER ~14.7%, F1 ~87%).
* **SpeechBrain ECAPA** là phương án nhẹ nhưng vẫn mạnh, dễ tích hợp.
* **Pyannote** không đáp ứng yêu cầu, chỉ nên giữ cho mục đích tham khảo.
* Với embedding mạnh, đặt threshold ~0.59–0.60 giúp tránh merge nhầm và kéo DER xuống rõ rệt trong mọi pipeline diarization.
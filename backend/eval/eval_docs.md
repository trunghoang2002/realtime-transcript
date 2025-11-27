# 📊 Evaluation Scripts Documentation

Hệ thống đánh giá chất lượng cho 2 tác vụ chính:
1. **ASR Evaluation** (`eval_asr.py`): Đánh giá chất lượng nhận dạng giọng nói (Speech Recognition)
2. **Diarization Evaluation** (`eval_diarization.py`): Đánh giá chất lượng phân biệt người nói (Speaker Verification)

---

# 🎤 Part 1: ASR Evaluation

## 📋 Mục đích

Script `eval_asr.py` được sử dụng để:
- Đánh giá độ chính xác của các model ASR (Whisper, SenseVoice, Gemini, VLLM)
- Đo lường hiệu suất xử lý thời gian thực (Real-Time Factor - RTF)
- So sánh hiệu suất giữa các model khác nhau
- Đánh giá với/không có VAD filtering
- Hỗ trợ checkpoint để tiếp tục evaluation khi bị gián đoạn

## 🏗️ Kiến trúc hệ thống

### Base Class: `BaseASR`
Abstract class định nghĩa interface chung cho tất cả ASR models:
- `_transcribe_no_vad(audio)`: Transcribe không dùng VAD
- `_transcribe_with_vad(audio)`: Transcribe có dùng VAD
- `transcribe(file_path, vad_filter)`: Public API

### Supported ASR Models

#### 1. **WhisperJA** (Faster Whisper)
```python
transcriber = WhisperJA(
    model_name="large-v3",  # tiny, base, small, medium, large, large-v1, large-v2, large-v3, turbo
    device="cuda",
    compute_type="float16",
    beam_size=5,
    best_of=5,
    temperature=0.0
)
```

**Features:**
- Hỗ trợ multiple model sizes
- Tối ưu với faster-whisper (CTranslate2)
- VAD filtering với Silero VAD
- Beam search và temperature control

#### 2. **SenseVoiceJA** (FunASR)
```python
transcriber = SenseVoiceJA(
    model_name="iic/SenseVoiceSmall",
    device="cuda",
    max_single_segment_time=30000,
    batch_size_s=20,
    use_itn=False
)
```

**Features:**
- Optimized cho tiếng Nhật
- Batch processing
- Inverse Text Normalization (ITN)

#### 3. **GeminiASR** (Google Gemini)
```python
transcriber = GeminiASR(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name="gemini-2.5-pro"  # gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro
)
```

**Features:**
- Cloud-based ASR
- Hỗ trợ multiple languages
- Automatic retry với exponential backoff
- Temperature=0.0 cho reproducibility

#### 4. **VllmASR** (vLLM Server)
```python
transcriber = VllmASR(
    base_url="https://your-vllm-server.com/v1",
    api_key="EMPTY",
    model_name="Qwen/Qwen2-Audio-7B-Instruct",
    prompt="Transcribe the audio into text."
)
```

**Features:**
- OpenAI-compatible API
- Custom prompt engineering
- Base64 audio encoding
- Exponential backoff retry

## 📊 Metrics

### 1. WER (Word Error Rate)
- **Định nghĩa**: Tỷ lệ lỗi từ
- **Công thức**: `WER = (S + D + I) / N`
  - S: Số từ bị thay thế (Substitutions)
  - D: Số từ bị xóa (Deletions)
  - I: Số từ bị thêm vào (Insertions)
  - N: Tổng số từ trong ground truth
- **Giá trị**: 0.0 = hoàn hảo, càng cao càng kém
- **Tokenization**: Sử dụng Sudachi tokenizer cho tiếng Nhật

### 2. CER (Character Error Rate)
- **Định nghĩa**: Tỷ lệ lỗi ký tự
- **Công thức**: `CER = (S + D + I) / N` (tính theo ký tự)
- **Giá trị**: 0.0 = hoàn hảo, càng cao càng kém
- **Hữu ích cho**: Ngôn ngữ không có khoảng trắng (tiếng Nhật, tiếng Trung)

### 3. RTF (Real-Time Factor)
- **Định nghĩa**: Tỷ lệ giữa thời gian xử lý và độ dài audio
- **Công thức**: `RTF = processing_time / audio_duration`
- **Giá trị**: 
  - RTF < 1.0: Xử lý nhanh hơn realtime (tốt)
  - RTF = 1.0: Xử lý đúng bằng realtime
  - RTF > 1.0: Xử lý chậm hơn realtime (không đủ cho realtime)

## 🔄 Text Normalization

Trước khi tính WER/CER, text được normalize:

```python
def eval_score(ground_truth, prediction):
    # 1. Lowercase
    ground_truth = ground_truth.lower()
    prediction = prediction.lower()
    
    # 2. Remove punctuation (Japanese)
    pattern = r"[\p{P}～~＋＝＄|]+"
    ground_truth = re.sub(pattern, "", ground_truth)
    prediction = re.sub(pattern, "", prediction)
    
    # 3. Remove tags
    prediction = prediction.replace("<[^>]*>", "")
    
    # 4. Remove English/spaces
    ground_truth = re.sub(r"[A-Za-z\s]+", "", ground_truth).strip()
    prediction = re.sub(r"[A-Za-z\s]+", "", prediction).strip()
    
    # 5. Calculate metrics with Japanese tokenizer
    wer_score = wer(ground_truth, prediction, 
                    reference_transform=wer_ja, 
                    hypothesis_transform=wer_ja)
    cer_score = cer(ground_truth, prediction)
    return wer_score, cer_score
```

## 💾 Checkpoint Mechanism

Script hỗ trợ checkpoint để tiếp tục evaluation khi bị gián đoạn:

```python
results_file = "eval_results_checkpoint.csv"
```

**Features:**
- Tự động lưu kết quả sau mỗi test case
- Resume từ file checkpoint nếu script bị dừng
- Track completed files để không xử lý lại

**CSV Format:**
```csv
file_path,ground_truth,prediction,wer_score,cer_score,rtf,audio_duration,processing_time
```

## 🚀 Cách sử dụng

### 1. Chuẩn bị dataset
Dataset CSV với format:
```csv
...,file_path,ground_truth
...,audio/file1.wav,これはテストです
...,audio/file2.wav,音声認識の評価
```

### 2. Cấu hình transcriber
Uncomment model bạn muốn test trong `eval_asr.py`:

```python
# Option 1: Whisper
transcriber = WhisperJA(
    model_name="large-v1",
    device="cuda",
    compute_type="float16"
)

# Option 2: SenseVoice
# transcriber = SenseVoiceJA(...)

# Option 3: Gemini
# transcriber = GeminiASR(...)

# Option 4: VLLM
# transcriber = VllmASR(...)
```

### 3. Chạy evaluation
```bash
cd backend/eval
python eval_asr.py
```

### 4. Xem kết quả
```
Total test cases: 400
Remaining: 350
Processing: 100%|████████████| 400/400
==================================================
FINAL RESULTS
==================================================
Total evaluated: 400 test cases
WER: 0.1234
CER: 0.0567
RTF: 0.4567 (mean)
RTF: 0.4123 (median)
RTF: 0.2100 (min) | 0.9800 (max)
```

---

# 👥 Part 2: Diarization Evaluation

## 📋 Mục đích

Script `eval_diarization.py` được sử dụng để:
- Đánh giá chất lượng speaker verification/diarization
- So sánh multiple embeddings từ nhiều models: Pyannote, SpeechBrain, NeMo (TitaNet, ECAPA-TDNN)
- Hỗ trợ multiple instances của cùng model type (ví dụ: 2 NeMo models với configs khác nhau)
- Tính toán multiple metrics: EER, FAR, FRR, Precision, Recall, F1, AUC
- Visualize ROC curves, DET curves, Precision-Recall curves
- Tìm optimal threshold cho từng metric

## 🔧 Dependencies

```bash
pip install scikit-learn
pip install matplotlib
pip install tqdm
pip install numpy
```

## 🏗️ Kiến trúc hệ thống

### Pipeline
```python
from fusion_diarization import RealtimeSpeakerDiarization

# Fusion pipeline với multiple model instances
pipeline = RealtimeSpeakerDiarization(
    models=[
        ('nemo', 'nemo_titanet'),
        ('nemo', 'nemo_ecapa_tdnn'),
        ('pyannote', 'pyan_community'),
        ('speechbrain', 'sb_default')
    ],
    model_configs={
        'nemo_titanet': {'pretrained_speaker_model': 'titanet_large'},
        'nemo_ecapa_tdnn': {'pretrained_speaker_model': 'ecapa_tdnn'},
        'pyan_community': {
            'model_name': "pyannote/speaker-diarization-community-1",
            'token': os.getenv("HF_TOKEN")
        },
        'sb_default': {}
    }
)
```

Pipeline extract 4 loại embeddings từ multiple model instances:
- **NeMo TitaNet embeddings**: From NVIDIA NeMo TitaNet Large
- **NeMo ECAPA-TDNN embeddings**: From NVIDIA NeMo ECAPA-TDNN
- **Pyannote embeddings**: From pyannote.audio
- **SpeechBrain embeddings**: From SpeechBrain ECAPA-TDNN

**Lưu ý**: Bạn có thể sử dụng nhiều instances của cùng model type với configs khác nhau. Ví dụ: 2 NeMo models với pretrained models khác nhau.

### Evaluation Process

1. **List speakers and utterances**: Scan dataset folder
2. **Build trials**: Tạo genuine pairs (cùng speaker) và impostor pairs (khác speaker)
3. **Extract embeddings**: Extract 1 lần cho tất cả files
4. **Compute scores**: Tính cosine similarity cho từng trial
5. **Calculate metrics**: EER, FAR, FRR, Precision, Recall, F1, AUC
6. **Visualize**: Vẽ và lưu ROC, DET, PR curves

## 📊 Metrics

### 1. EER (Equal Error Rate)
- **Định nghĩa**: Điểm mà FAR = FRR
- **Giá trị**: Càng thấp càng tốt (0% = hoàn hảo)
- **Ý nghĩa**: Cân bằng giữa false accept và false reject

### 2. FAR (False Acceptance Rate)
- **Định nghĩa**: Tỷ lệ impostor pairs bị accept nhầm
- **Công thức**: `FAR = FP / (FP + TN)`
- **Giá trị**: Càng thấp càng tốt

### 3. FRR (False Rejection Rate)
- **Định nghĩa**: Tỷ lệ genuine pairs bị reject nhầm
- **Công thức**: `FRR = FN / (FN + TP)`
- **Giá trị**: Càng thấp càng tốt

### 4. Precision
- **Công thức**: `Precision = TP / (TP + FP)`
- **Giá trị**: 0.0 - 1.0 (càng cao càng tốt)

### 5. Recall (Sensitivity, TPR)
- **Công thức**: `Recall = TP / (TP + FN)`
- **Giá trị**: 0.0 - 1.0 (càng cao càng tốt)

### 6. F1 Score
- **Công thức**: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
- **Giá trị**: 0.0 - 1.0 (càng cao càng tốt)

### 7. AUC (Area Under ROC Curve)
- **Định nghĩa**: Diện tích dưới ROC curve
- **Giá trị**: 0.0 - 1.0 (1.0 = perfect classifier)

## 📁 Dataset Structure

Dataset cần tuân theo cấu trúc:
```
dataset/
├── speaker_001/
│   ├── falset10/
│   │   └── wav24kHz16bit/
│   │       ├── file1.wav
│   │       └── file2.wav
│   ├── nonpara30/
│   │   └── wav24kHz16bit/
│   ├── parallel100/
│   │   └── wav24kHz16bit/
│   └── whisper10/
│       └── wav24kHz16bit/
├── speaker_002/
│   └── ...
```

**Yêu cầu:**
- Mỗi speaker có ít nhất 2 utterances
- 4 loại folder: `falset10`, `nonpara30`, `parallel100`, `whisper10`
- Audio files trong `wav24kHz16bit/`

## 🎯 Trial Generation

### Genuine Pairs (label=1)
- Cặp 2 utterances khác nhau của cùng 1 speaker
- Max pairs per speaker: `max_genuine_per_spk=50`

### Impostor Pairs (label=0)
- Cặp utterances từ 2 speakers khác nhau
- Pairs per speaker: `impostor_per_spk=100`

### Example
```python
trials = build_trials(
    spk2utts,
    max_genuine_per_spk=50,  # Max 50 genuine pairs mỗi speaker
    impostor_per_spk=100     # 100 impostor pairs mỗi speaker
)
# Output: [(path1, path2, label), ...]
```

## 🎨 Visualization

Script tự động tạo 3 loại biểu đồ trong folder `eval_results/`:

### 1. ROC Curve (`roc_curves.png`)
- **Trục X**: False Positive Rate (FAR)
- **Trục Y**: True Positive Rate (1 - FRR)
- **Features**:
  - So sánh tất cả models (Pyannote, SpeechBrain, NeMo variants)
  - Hiển thị AUC score cho mỗi model
  - Đánh dấu điểm EER cho mỗi curve
  - Đường baseline (random classifier)
- **Color coding**:
  - Blue: Pyannote
  - Red: SpeechBrain  
  - Green: NeMo TitaNet
  - Yellow: NeMo ECAPA-TDNN

### 2. DET Curve (`det_curves.png`)
- **Trục X**: False Acceptance Rate (%)
- **Trục Y**: False Rejection Rate (%)
- **Features**:
  - Dễ nhìn hơn cho speaker verification
  - Đánh dấu điểm EER cho mỗi model
  - Đường chéo FAR=FRR (EER line)
- **Color coding**: Giống ROC curves
  - Blue: Pyannote | Red: SpeechBrain
  - Green: NeMo TitaNet | Yellow: NeMo ECAPA-TDNN

### 3. Precision-Recall Curve (`precision_recall_curves.png`)
- **Trục X**: Recall
- **Trục Y**: Precision
- **Features**:
  - Hiển thị PR AUC cho mỗi model
  - Đánh dấu điểm Best F1 cho mỗi curve
  - So sánh tất cả embeddings
- **Color coding**: Giống ROC/DET curves
  - Blue: Pyannote | Red: SpeechBrain
  - Green: NeMo TitaNet | Yellow: NeMo ECAPA-TDNN

## 🎯 Multiple Model Instances

Script hỗ trợ đánh giá multiple instances của cùng model type với configs khác nhau:

### Ví dụ: Multiple NeMo Models
```python
pipeline = RealtimeSpeakerDiarization(
    models=[
        ('nemo', 'nemo_titanet'),        # NeMo với TitaNet Large
        ('nemo', 'nemo_ecapa_tdnn'),     # NeMo với ECAPA-TDNN
        ('pyannote', 'pyan_community'),   # Pyannote
        ('speechbrain', 'sb_default')     # SpeechBrain
    ],
    model_configs={
        'nemo_titanet': {'pretrained_speaker_model': 'titanet_large'},
        'nemo_ecapa_tdnn': {'pretrained_speaker_model': 'ecapa_tdnn'},
        'pyan_community': {
            'model_name': "pyannote/speaker-diarization-community-1",
            'token': os.getenv("HF_TOKEN")
        },
        'sb_default': {}
    }
)
```

### Lợi ích:
- **So sánh variants**: So sánh hiệu suất giữa các pretrained models khác nhau
- **Tối ưu selection**: Chọn model config tốt nhất cho dataset cụ thể
- **Ensemble insights**: Hiểu được đóng góp của từng model trong fusion
- **Efficient extraction**: Extract tất cả embeddings chỉ 1 lần

### Caching System:
Script có caching mechanism để tối ưu tốc độ:
- **Auto cache**: Tự động lưu embeddings sau khi extract
- **Cache key**: Dựa trên MD5 hash của danh sách files
- **Reusable**: Cache có thể tái sử dụng cho nhiều lần chạy
- **Clear cache**: Sử dụng `clear_cache()` để xóa cache cũ

```python
# Sử dụng cache (mặc định)
results = evaluate_dataset("dataset/jvs_ver1", use_cache=True)

# Force re-extract (không dùng cache)
results = evaluate_dataset("dataset/jvs_ver1", use_cache=False)

# Xóa tất cả cache
clear_cache()
```

## 🚀 Cách sử dụng

### 1. Chuẩn bị dataset
Organize audio files theo cấu trúc folder như trên.

### 2. Cấu hình pipeline
Trong `eval_diarization.py`, cấu hình fusion pipeline:
```python
pipeline = RealtimeSpeakerDiarization(
    models=[
        ('nemo', 'nemo_titanet'),
        ('nemo', 'nemo_ecapa_tdnn'),
        ('pyannote', 'pyan_community'),
        ('speechbrain', 'sb_default')
    ],
    model_configs={
        'nemo_titanet': {'pretrained_speaker_model': 'titanet_large'},
        'nemo_ecapa_tdnn': {'pretrained_speaker_model': 'ecapa_tdnn'},
        'pyan_community': {
            'model_name': "pyannote/speaker-diarization-community-1",
            'token': os.getenv("HF_TOKEN")
        },
        'sb_default': {}
    }
)
```

### 3. Cấu hình evaluation (optional)
Có thể điều chỉnh các tham số:
```python
# Số lượng trials
max_genuine_per_spk = 50
impostor_per_spk = 100

# Output directory cho biểu đồ
output_dir = "eval_results"

# Dataset path
dataset_path = "dataset/jvs_ver1"

# Cache settings
use_cache = True  # False để force re-extract
```

### 4. Chạy evaluation
```bash
cd backend/eval
python eval_diarization.py
```

### 5. Xem kết quả

**Console Output:**
```
Found 100 speakers usable.
Total trials: 15000
Extracting embeddings: 100%|████████| 500/500

=== Evaluating pyannote embeddings ===
Computing metrics on 14850 valid trials
EER: 5.23% | FAR@EER: 5.21% | FRR@EER: 5.25% | Thr(EER): 0.6234
Precision@EER: 94.77% | Recall@EER: 94.75% | F1@EER: 94.76%
Best F1: 96.12% | Precision@F1: 95.89% | Recall@F1: 96.35% | Thr(F1): 0.6789
AUC: 0.9876

=== Evaluating speechbrain embeddings ===
Computing metrics on 14850 valid trials
EER: 4.87% | FAR@EER: 4.85% | FRR@EER: 4.89% | Thr(EER): 0.7123
Precision@EER: 95.13% | Recall@EER: 95.11% | F1@EER: 95.12%
Best F1: 96.87% | Precision@F1: 96.45% | Recall@F1: 97.29% | Thr(F1): 0.7456
AUC: 0.9912

=== Evaluating nemo_titanet embeddings ===
Computing metrics on 14850 valid trials
EER: 3.45% | FAR@EER: 3.42% | FRR@EER: 3.48% | Thr(EER): 0.7456
Precision@EER: 96.55% | Recall@EER: 96.52% | F1@EER: 96.53%
Best F1: 97.89% | Precision@F1: 97.65% | Recall@F1: 98.13% | Thr(F1): 0.7890
AUC: 0.9945

=== Evaluating nemo_ecapa_tdnn embeddings ===
Computing metrics on 14850 valid trials
EER: 4.12% | FAR@EER: 4.10% | FRR@EER: 4.14% | Thr(EER): 0.7234
Precision@EER: 95.88% | Recall@EER: 95.86% | F1@EER: 95.87%
Best F1: 97.34% | Precision@F1: 97.12% | Recall@F1: 97.56% | Thr(F1): 0.7567
AUC: 0.9928

=== Plotting curves ===
ROC curve saved to: eval_results/roc_curves.png
DET curve saved to: eval_results/det_curves.png
Precision-Recall curve saved to: eval_results/precision_recall_curves.png
```

**Generated Files:**
- `eval_results/roc_curves.png` - So sánh ROC curves của tất cả models
- `eval_results/det_curves.png` - So sánh DET curves của tất cả models  
- `eval_results/precision_recall_curves.png` - So sánh PR curves của tất cả models
- `eval_cache/embeddings_cache_*.pkl` - Cached embeddings (auto-generated)

**Final Results Output:**
```python
=== Final Results ===
Pyannote: {'EER': 0.0523, 'AUC': 0.9876, ...}
SpeechBrain: {'EER': 0.0487, 'AUC': 0.9912, ...}
NeMo Titanet: {'EER': 0.0345, 'AUC': 0.9945, ...}
NeMo Ecapa TDNN: {'EER': 0.0412, 'AUC': 0.9928, ...}
```

## 🛡️ Error Handling

Script xử lý robust với các edge cases:

### 1. Embedding Extraction Failures
```python
# Skip nếu:
- result is None
- embeddings is None or empty
- embeddings toàn bằng 0
- embeddings toàn là NaN
```

### 2. Score Computation Failures
```python
# Skip trial nếu:
- File không có trong cache
- Embedding type không tồn tại
- Embedding là None
- Embedding toàn 0 hoặc NaN
- Cosine score là NaN hoặc inf
```

### 3. Metrics Calculation
```python
# Return None nếu:
- Không có valid trials
- zero_division=0 trong precision_recall_fscore_support
```

## 🔍 Giải thích kết quả

### EER thấp (< 5%)
- ✅ Chất lượng speaker verification xuất sắc
- System phân biệt speakers rất chính xác

### EER trung bình (5% - 10%)
- ⚠️ Chất lượng ở mức chấp nhận được
- Có thể cần fine-tune threshold

### EER cao (> 10%)
- ❌ Chất lượng kém
- Cần cải thiện model hoặc features

### F1 Score
- **Best F1 > 95%**: Xuất sắc
- **Best F1 = 90-95%**: Tốt
- **Best F1 < 90%**: Cần cải thiện

### AUC Score
- **AUC > 0.95**: Xuất sắc
- **AUC = 0.90-0.95**: Tốt
- **AUC = 0.80-0.90**: Trung bình
- **AUC < 0.80**: Kém

## ⚡ Optimization Features

### 1. Single Embedding Extraction
Extract embeddings **chỉ 1 lần** cho mỗi file:
```python
# Extract tất cả embeddings từ các model instances cùng lúc
result, _ = pipeline._extract_embeddings(file_path, max_speakers=1)
emb_cache[file_path] = {
    "pyannote": result["pyan_community_embeddings"][0],
    "speechbrain": result["sb_default_embeddings"][0],
    "nemo_titanet": result["nemo_tianet_embeddings"][0],
    "nemo_ecapa_tdnn": result["nemo_ecapa_tdnn_embeddings"][0]
}
```

### 2. Efficient Trial Processing
```python
# Chỉ extract unique files
all_files = set()
for p1, p2, _ in trials:
    all_files.add(p1)
    all_files.add(p2)
```

### 3. Progress Tracking
```python
# Progress bar cho embedding extraction
for fpath in tqdm(list(all_files), desc="Extracting embeddings"):
    ...
```

---

# 💡 Tips & Best Practices

## ASR Evaluation
1. **Checkpoint regularly**: Script tự động save checkpoint, đừng xóa file
2. **VAD filtering**: Test cả 2 modes (with/without VAD)
3. **RTF analysis**: Monitor RTF để đảm bảo realtime performance
4. **Batch processing**: Sử dụng batch cho API-based models (Gemini, VLLM)
5. **Error handling**: Script có retry mechanism cho API calls

## Diarization Evaluation
1. **Dataset quality**: Đảm bảo audio quality tốt và speakers đủ diverse
2. **Trial balance**: Cân bằng số genuine và impostor pairs
3. **Threshold selection**: 
   - Dùng threshold tại EER cho balanced performance
   - Dùng threshold tại Best F1 cho maximum accuracy
4. **Visualization**: Xem curves để understand model behavior
5. **Embedding comparison**: So sánh giữa các models (Pyannote, SpeechBrain, NeMo variants) để chọn best model
6. **Multiple instances**: Sử dụng multiple instances của cùng model type với configs khác nhau để đánh giá và so sánh
7. **Cache management**: Script có caching mechanism để tránh re-extract embeddings

---

# 🔗 Tham khảo

## ASR
- [jiwer documentation](https://github.com/jitsi/jiwer)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [WER explanation](https://en.wikipedia.org/wiki/Word_error_rate)

## Speaker Verification
- [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [ROC Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [DET Curve](https://en.wikipedia.org/wiki/Detection_error_tradeoff)
- [EER explanation](https://www.sciencedirect.com/topics/computer-science/equal-error-rate)

## Models
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [SpeechBrain](https://github.com/speechbrain/speechbrain)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
  - [TitaNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/models.html#titanet)
  - [ECAPA-TDNN](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/models.html#ecapa-tdnn)
- [Google Gemini](https://ai.google.dev/)
- [vLLM](https://github.com/vllm-project/vllm)

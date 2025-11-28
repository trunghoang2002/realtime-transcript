# Realtime Speaker Diarization Pipeline

## Tổng quan

`RealtimeSpeakerDiarization` là pipeline custom để xử lý speaker diarization realtime với khả năng **duy trì context embedding của speakers** qua các lần gọi hàm liên tiếp.

## Các tính năng chính

### 1. **Speaker Memory Persistence**
- Lưu trữ embedding của mỗi speaker đã được phát hiện
- Tự động match speakers mới với speakers đã biết dựa trên similarity
- Cập nhật embeddings theo thời gian với moving average

### 2. **Adaptive Embedding Update**
- Embeddings được cập nhật liên tục để phản ánh sự thay đổi của giọng nói (noise, cảm xúc, etc.)
- Sử dụng exponential moving average: `new = α * current + (1-α) * old`

### 3. **Speaker Identity Consistency**
- Đảm bảo cùng một speaker có cùng ID xuyên suốt conversation
- Tránh việc gán nhầm speaker khi xử lý từng chunk riêng lẻ

## Cách sử dụng

### Khởi tạo Pipeline

```python
from test2 import RealtimeSpeakerDiarization

pipeline = RealtimeSpeakerDiarization(
    model_name="pyannote/speaker-diarization-community-1",
    token="YOUR_HF_TOKEN",
    similarity_threshold=0.7,      # Ngưỡng để match speaker (0.0-1.0)
    embedding_update_weight=0.3    # Tỷ lệ update embedding mới (0.0-1.0)
)

# Chuyển sang GPU
pipeline.to(torch.device("cuda"))
```

### Tham số quan trọng

#### `similarity_threshold` (default: 0.7)
- Ngưỡng similarity để quyết định speaker mới có phải là speaker cũ
- **Cao (0.8-0.9)**: Strict hơn, ít nhầm lẫn nhưng có thể tạo nhiều speaker IDs cho cùng người
- **Thấp (0.5-0.6)**: Relaxed hơn, có thể gộp nhiều người thành một speaker

#### `embedding_update_weight` (default: 0.3)
- Tỷ lệ giữa embedding mới và embedding cũ khi update
- **Cao (0.5-0.7)**: Adapt nhanh với thay đổi giọng nói, nhưng kém ổn định
- **Thấp (0.1-0.3)**: Ổn định hơn, nhưng chậm adapt với thay đổi

### Xử lý Audio Chunks

```python
# Chunk 1
output1 = pipeline(
    "path/to/audio_chunk1.wav",
    min_speakers=1,
    max_speakers=5,
    use_memory=True  # Enable realtime mode
)

# In kết quả
for turn, _, speaker in output1.speaker_diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}s - {turn.end:.1f}s: {speaker}")

# Chunk 2 - speakers sẽ được match với chunk 1
output2 = pipeline(
    "path/to/audio_chunk2.wav",
    min_speakers=1,
    max_speakers=5,
    use_memory=True
)
```

### Kiểm tra Speaker Memory

```python
info = pipeline.get_speaker_info()
print(f"Số speakers đã biết: {info['num_speakers']}")
print(f"Speakers: {info['speakers']}")
print(f"Số lần xuất hiện: {info['speaker_counts']}")
print(f"Đã xử lý {info['total_chunks']} chunks")
```

Output ví dụ:
```
Số speakers đã biết: 3
Speakers: ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']
Số lần xuất hiện: {'SPEAKER_00': 5, 'SPEAKER_01': 3, 'SPEAKER_02': 2}
Đã xử lý 10 chunks
```

### Reset Context

Khi bắt đầu conversation mới:

```python
pipeline.reset_context()
# Bây giờ speaker IDs sẽ bắt đầu lại từ SPEAKER_00
```

### Batch Mode (không dùng memory)

Nếu muốn xử lý như pipeline bình thường:

```python
output = pipeline(
    "path/to/audio.wav",
    use_memory=False  # Disable realtime mode
)
```

## Workflow Realtime

```
┌─────────────────────────────────────────────────────────────┐
│                    Audio Stream                              │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ Chia thành chunks
               ▼
        ┌──────────────┐
        │   Chunk 1    │──► Pipeline ──► SPEAKER_00, SPEAKER_01
        └──────────────┘                       │
                                               │ Lưu embeddings
                                               ▼
        ┌──────────────┐               ┌─────────────────┐
        │   Chunk 2    │──► Pipeline ─►│ Speaker Memory  │
        └──────────────┘      │        │ SPEAKER_00: [e0]│
                              │        │ SPEAKER_01: [e1]│
                              │        └─────────────────┘
                              │ Match với memory
                              ▼
                    SPEAKER_00 (matched!)
                    SPEAKER_02 (new!)
                              │
                              │ Update embeddings
                              ▼
                       ┌─────────────────┐
                       │ Speaker Memory  │
                       │  SPEAKER_00: [e0']│ ← updated
                       │  SPEAKER_01: [e1] │
                       │  SPEAKER_02: [e2] │ ← new
                       └─────────────────┘
```

## Cơ chế hoạt động chi tiết

### 1. **Speaker Matching Algorithm**

Khi có embeddings mới từ chunk hiện tại:

```python
# Bước 1: Tính similarity với tất cả speakers trong memory
similarities = cosine_similarity(new_embeddings, memory_embeddings)

# Bước 2: Greedy matching
for each new_speaker:
    best_match = argmax(similarities[new_speaker])
    if similarity > threshold and not already_used:
        # Match với speaker cũ
        match_with_existing_speaker()
        update_embedding()
    else:
        # Tạo speaker mới
        create_new_speaker()
```

### 2. **Embedding Update Strategy**

```python
# Moving average để cập nhật embedding
updated_embedding = α * new_embedding + (1 - α) * old_embedding

# Normalize cho cosine similarity
updated_embedding = updated_embedding / norm(updated_embedding)
```

### 3. **Consistency Guarantees**

- **Monotonic Speaker IDs**: Speaker IDs luôn tăng dần (SPEAKER_00, 01, 02, ...)
- **No Conflicts**: Một speaker trong chunk chỉ match với tối đa một speaker trong memory
- **History Tracking**: Lưu lịch sử tất cả các chunks đã xử lý

## Use Cases

### 1. **Video Conference Transcription**
```python
pipeline = RealtimeSpeakerDiarization(similarity_threshold=0.75)

for audio_chunk in video_conference_stream:
    output = pipeline(audio_chunk, use_memory=True)
    for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
        transcript = transcribe(audio_chunk[turn])
        print(f"{speaker}: {transcript}")
```

### 2. **Phone Call Analysis**
```python
pipeline = RealtimeSpeakerDiarization(
    similarity_threshold=0.6,  # Relaxed vì phone quality thấp
    embedding_update_weight=0.4  # Adapt nhanh với noise
)

for call_segment in phone_call:
    output = pipeline(call_segment, min_speakers=2, max_speakers=2)
    # Phân biệt caller vs callee
```

### 3. **Podcast Segmentation**
```python
pipeline = RealtimeSpeakerDiarization(similarity_threshold=0.8)

for minute in podcast:
    output = pipeline(minute, use_memory=True)
    # Consistent speaker IDs throughout podcast
```

## Hiệu năng & Tối ưu

### Memory Usage
- Mỗi speaker: ~512 floats (embedding dimension)
- History: O(num_chunks)
- Tổng: ~2-5 MB cho 10 speakers, 1000 chunks

### Processing Speed
- Tương tự pipeline gốc (~1-2x realtime)
- Overhead matching: ~1-10ms per chunk (negligible)

### Tips
1. **GPU**: Luôn dùng GPU nếu có thể
2. **Batch size**: Tăng `embedding_batch_size` cho chunks dài
3. **Threshold tuning**: Test với audio sample trước khi deploy

## Troubleshooting

### Vấn đề: Quá nhiều speaker IDs cho cùng một người

**Nguyên nhân**: `similarity_threshold` quá cao

**Giải pháp**:
```python
pipeline = RealtimeSpeakerDiarization(
    similarity_threshold=0.6  # Giảm xuống
)
```

### Vấn đề: Nhiều người bị gộp thành một speaker

**Nguyên nhân**: `similarity_threshold` quá thấp

**Giải pháp**:
```python
pipeline = RealtimeSpeakerDiarization(
    similarity_threshold=0.8  # Tăng lên
)
```

### Vấn đề: Speaker IDs không ổn định giữa các chunks

**Nguyên nhân**: `embedding_update_weight` quá cao

**Giải pháp**:
```python
pipeline = RealtimeSpeakerDiarization(
    embedding_update_weight=0.2  # Giảm xuống
)
```

## API Reference

### Class: `RealtimeSpeakerDiarization`

#### Methods

##### `__init__(model_name, token, similarity_threshold, embedding_update_weight, **kwargs)`
Khởi tạo pipeline

##### `apply(file, use_memory=True, **kwargs)`
Xử lý audio file

##### `apply_realtime(file, use_memory=True, hook=None, **kwargs)`
Xử lý với realtime mode (được gọi bởi `apply`)

##### `reset_context()`
Reset toàn bộ speaker memory

##### `get_speaker_info() -> Dict`
Lấy thông tin về speakers hiện tại

**Returns:**
```python
{
    'speakers': ['SPEAKER_00', 'SPEAKER_01'],
    'speaker_counts': {'SPEAKER_00': 5, 'SPEAKER_01': 3},
    'total_chunks': 8,
    'num_speakers': 2
}
```

#### Attributes

- `speaker_memory: Dict[str, np.ndarray]` - Embedding của mỗi speaker
- `speaker_counts: Dict[str, int]` - Số lần xuất hiện
- `speaker_history: List[Dict]` - Lịch sử xử lý
- `similarity_threshold: float` - Ngưỡng similarity
- `embedding_update_weight: float` - Tỷ lệ update

## License

MIT License (same as pyannote.audio)


# 🚀 Quick Start Guide

## Cài đặt nhanh (5 phút)

### Bước 1: Install dependencies

```bash
cd /home/hoang/realtime-transcript/backend/pyanote
pip install -r requirements.txt
```

### Bước 2: Chạy demo đầu tiên

```bash
python test2.py
```

Bạn sẽ thấy output như:

```
============================================================
VÍ DỤ 1: Xử lý audio chunk đầu tiên
============================================================

📊 Kết quả chunk 1:
  ⏱️  6.6s → 7.1s | 🎤 SPEAKER_00
  ⏱️  8.2s → 9.5s | 🎤 SPEAKER_01

💾 Speaker Memory: {
    'speakers': ['SPEAKER_00', 'SPEAKER_01'],
    'num_speakers': 2,
    'total_chunks': 1
}
```

## 🎯 Use Case 1: Xử lý audio chunks liên tiếp

```python
from test2 import RealtimeSpeakerDiarization
import torch

# Khởi tạo pipeline
pipeline = RealtimeSpeakerDiarization(
    model_name="pyannote/speaker-diarization-community-1",
    token="YOUR_HF_TOKEN",  # Lấy từ https://huggingface.co/settings/tokens
    similarity_threshold=0.7,
    embedding_update_weight=0.3
)

pipeline.to(torch.device("cuda"))  # Hoặc "cpu" nếu không có GPU

# Xử lý chunk 1
output1 = pipeline("audio_chunk1.wav", use_memory=True)
for turn, _, speaker in output1.speaker_diarization.itertracks(yield_label=True):
    print(f"{speaker}: {turn.start:.1f}s - {turn.end:.1f}s")

# Xử lý chunk 2 - speakers sẽ consistent với chunk 1
output2 = pipeline("audio_chunk2.wav", use_memory=True)
for turn, _, speaker in output2.speaker_diarization.itertracks(yield_label=True):
    print(f"{speaker}: {turn.start:.1f}s - {turn.end:.1f}s")

# Xem thông tin speakers
print(pipeline.get_speaker_info())
```

## 🎯 Use Case 2: Xử lý audio dài

```python
from realtime_example import AudioStreamProcessor

processor = AudioStreamProcessor(hf_token="YOUR_HF_TOKEN")

# Tự động chia thành chunks và xử lý
results = processor.process_long_audio(
    audio_path="long_audio.wav",
    chunk_duration=30.0,  # 30 giây mỗi chunk
    overlap=1.0,          # 1 giây overlap
    min_speakers=2,
    max_speakers=5
)

# Kết quả với timestamps đã được adjust
for result in results:
    print(f"Chunk {result['chunk_id']}: {result['speakers']}")
```

## 🎯 Use Case 3: Realtime streaming

```python
import numpy as np
from realtime_example import AudioStreamProcessor

processor = AudioStreamProcessor(hf_token="YOUR_HF_TOKEN")

# Simulate realtime audio stream
for audio_chunk in your_audio_stream():  # Generator hoặc queue
    # audio_chunk: numpy array, shape (samples,)
    
    result = processor.process_audio_chunk(
        audio_chunk,
        sample_rate=16000,
        min_speakers=1,
        max_speakers=10
    )
    
    # Process results
    for segment in result['segments']:
        speaker = segment['speaker']
        start = segment['start']
        end = segment['end']
        
        # Gửi tới UI, database, v.v.
        print(f"{speaker} spoke from {start}s to {end}s")

# Reset khi bắt đầu conversation mới
processor.reset()
```

## ⚙️ Tuning Parameters

### Khi nào adjust `similarity_threshold`?

**Giảm threshold (0.6-0.7)** nếu:
- ❌ Cùng một người bị chia thành nhiều speaker IDs
- ❌ Audio quality thấp (noise, compression)
- ❌ Giọng nói thay đổi nhiều (cảm xúc, volume)

**Tăng threshold (0.8-0.9)** nếu:
- ❌ Nhiều người bị gộp thành một speaker
- ❌ Giọng nói rất giống nhau
- ❌ Audio quality cao, giọng ổn định

### Khi nào adjust `embedding_update_weight`?

**Tăng weight (0.4-0.6)** nếu:
- ❌ Giọng nói thay đổi nhanh (noise levels, distance to mic)
- ❌ Conversation dài, giọng có thể thay đổi
- ❌ Cần adapt nhanh với thay đổi

**Giảm weight (0.1-0.2)** nếu:
- ❌ Speaker IDs không ổn định giữa chunks
- ❌ Giọng nói ổn định, không đổi nhiều
- ❌ Conversation ngắn

## 🔧 Common Configurations

### Video Conference (Zoom, Teams)
```python
pipeline = RealtimeSpeakerDiarization(
    similarity_threshold=0.75,      # Balanced
    embedding_update_weight=0.35    # Adapt to varying audio quality
)
```

### Phone Calls
```python
pipeline = RealtimeSpeakerDiarization(
    similarity_threshold=0.65,      # Relaxed (phone quality)
    embedding_update_weight=0.4     # Fast adapt to noise
)
```

### Podcast / High Quality Audio
```python
pipeline = RealtimeSpeakerDiarization(
    similarity_threshold=0.8,       # Strict (clear voices)
    embedding_update_weight=0.25    # Stable (consistent quality)
)
```

### Meeting Room Recording
```python
pipeline = RealtimeSpeakerDiarization(
    similarity_threshold=0.7,       # Balanced
    embedding_update_weight=0.3     # Standard
)
```

## 📊 Performance Tips

### 1. Use GPU
```python
import torch
pipeline.to(torch.device("cuda"))  # ~5-10x faster
```

### 2. Adjust batch sizes
```python
pipeline = RealtimeSpeakerDiarization(
    segmentation_batch_size=4,    # Larger = faster (needs more VRAM)
    embedding_batch_size=8
)
```

### 3. Optimal chunk duration
- **Too short** (< 5s): Poor embeddings, unstable
- **Too long** (> 60s): Slow processing, delays
- **Recommended**: 15-30 seconds

### 4. Use overlap
```python
processor.process_long_audio(
    chunk_duration=30.0,
    overlap=2.0  # 2s overlap helps continuity
)
```

## 🐛 Troubleshooting

### Error: "CUDA out of memory"
```python
# Solution 1: Reduce batch size
pipeline.segmentation_batch_size = 1
pipeline.embedding_batch_size = 1

# Solution 2: Use CPU
pipeline.to(torch.device("cpu"))

# Solution 3: Shorter chunks
chunk_duration = 15.0  # instead of 30.0
```

### Warning: "Too many speakers detected"
```python
# Limit max speakers
output = pipeline(
    audio,
    min_speakers=2,
    max_speakers=5  # Reasonable upper bound
)
```

### Issue: "Inconsistent speaker IDs"
```python
# More stable configuration
pipeline = RealtimeSpeakerDiarization(
    similarity_threshold=0.75,      # Higher = more strict matching
    embedding_update_weight=0.2     # Lower = more stable
)
```

## 📚 Next Steps

1. **Read full docs**: `README_REALTIME.md`
2. **See examples**: `realtime_example.py`
3. **WebSocket integration**: `websocket_server.py`
4. **Technical details**: `SOLUTION_OVERVIEW.md`

## 💡 Pro Tips

### Tip 1: Monitor speaker info
```python
info = pipeline.get_speaker_info()
if info['num_speakers'] > expected_speakers:
    # Adjust similarity_threshold
    pipeline.similarity_threshold = 0.8
```

### Tip 2: Reset giữa conversations
```python
# Between different calls/meetings
pipeline.reset_context()
```

### Tip 3: Save/load speaker memory
```python
import pickle

# Save
memory = {
    'speaker_memory': pipeline.speaker_memory,
    'speaker_counts': pipeline.speaker_counts,
}
with open('speakers.pkl', 'wb') as f:
    pickle.dump(memory, f)

# Load
with open('speakers.pkl', 'rb') as f:
    memory = pickle.load(f)
    pipeline.speaker_memory = memory['speaker_memory']
    pipeline.speaker_counts = memory['speaker_counts']
```

### Tip 4: Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log mỗi chunk
result = pipeline(audio)
logger.info(f"Processed chunk with {len(result.speakers)} speakers")
logger.info(f"Total known speakers: {pipeline.get_speaker_info()['num_speakers']}")
```

## 🎉 You're Ready!

Bây giờ bạn có thể:
- ✅ Xử lý audio chunks với persistent speaker IDs
- ✅ Track speakers xuyên suốt conversation
- ✅ Tune parameters cho use case của bạn
- ✅ Integrate vào production system

**Happy diarizing! 🎤✨**


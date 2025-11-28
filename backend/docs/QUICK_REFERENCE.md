# 🚀 Quick Reference Card

## Basic Usage

```python
from test2 import RealtimeSpeakerDiarization
import torch

# Initialize
pipeline = RealtimeSpeakerDiarization(
    token="YOUR_HF_TOKEN",
    similarity_threshold=0.6,
    embedding_update_weight=0.3
)
pipeline.to(torch.device("cuda"))

# Process chunks
for audio in stream:
    output = pipeline(audio, use_memory=True)
    for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
        print(f"{speaker}: {turn.start}-{turn.end}")
```

## Two-Tier Matching Flow

```
New Embedding
    ↓
Tier 1: EMA Matching
    ├─ similarity ≥ 0.6 → ✅ Match (FAST)
    └─ similarity < 0.6 → Tier 2
                            ↓
                    Cluster Centroid
                        ├─ similarity ≥ 0.6 → ✅ Match (ROBUST)
                        └─ similarity < 0.6 → 🆕 New Speaker
```

## Key Parameters

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `similarity_threshold` | 0.7 | 0.5-0.9 | Match threshold (both tiers) |
| `embedding_update_weight` | 0.3 | 0.1-0.7 | EMA update rate |
| `max_cluster_size` | 20 | 5-50 | Embeddings per speaker |

## Tuning Guide

### Problem: Too many speaker IDs

```python
# Lower threshold
pipeline.similarity_threshold = 0.6
```

### Problem: Multiple speakers merged

```python
# Higher threshold
pipeline.similarity_threshold = 0.8
```

### Problem: IDs unstable

```python
# Lower EMA weight + larger cluster
pipeline.embedding_update_weight = 0.2
pipeline.max_cluster_size = 30
```

## Speaker Info

```python
info = pipeline.get_speaker_info()

# v2.0 output:
{
    'speakers': ['SPEAKER_00', 'SPEAKER_01'],
    'speaker_counts': {'SPEAKER_00': 5, 'SPEAKER_01': 3},
    'cluster_sizes': {'SPEAKER_00': 12, 'SPEAKER_01': 8},  # v2.0
    'num_speakers': 2,
    'total_chunks': 8
}
```

## Methods

```python
# Process with memory
output = pipeline(audio, use_memory=True, min_speakers=1, max_speakers=5)

# Reset for new conversation
pipeline.reset_context()

# Check speakers
info = pipeline.get_speaker_info()
```

## Output Structure

```python
output.speaker_diarization           # With overlaps
output.exclusive_speaker_diarization # No overlaps
output.speaker_embeddings            # (N, 512) array
```

## Quick Configs

### Video Conference
```python
similarity_threshold=0.75
embedding_update_weight=0.35
```

### Phone Call
```python
similarity_threshold=0.65
embedding_update_weight=0.4
```

### Podcast
```python
similarity_threshold=0.8
embedding_update_weight=0.25
```

## Performance

| Metric | Value |
|--------|-------|
| Speed | ~1.8x realtime |
| Memory/speaker | ~42KB |
| Tier 1 overhead | <1ms |
| Tier 2 overhead | ~3-5ms |

## Common Issues

| Symptom | Fix |
|---------|-----|
| CUDA OOM | `pipeline.to(torch.device("cpu"))` |
| Too slow | Reduce batch size or use GPU |
| False matches | Increase threshold |
| False non-matches | Decrease threshold, increase cluster size |

## Logging

Tier 1/2 logs are printed automatically:

```
[TIER 1] Label: SPEAKER_00
  Best EMA similarity: 0.652
  ❌ EMA not matched
  [TIER 2] Cluster centroid: 0.718
  ✅ Matched via cluster!
```

## Files

- `test2.py` - Core implementation
- `TWO_TIER_MATCHING.md` - Algorithm details
- `README.md` - Full documentation
- `QUICKSTART.md` - Tutorial

## Installation

```bash
pip install -r requirements.txt
python test_installation.py  # Verify
python test2.py              # Run example
```

## Key Concepts

**EMA Embedding**: Fast, smooth average
- Pro: Fast matching
- Con: May drift

**Cluster**: Collection of embeddings
- Pro: Captures variations
- Con: More memory

**Two-Tier**: Best of both worlds!
- Fast path via EMA (typical)
- Robust path via cluster (fallback)

---

**v2.0** | Two-Tier Matching | See `TWO_TIER_MATCHING.md` for details


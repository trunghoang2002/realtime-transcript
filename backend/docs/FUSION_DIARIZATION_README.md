# Fusion Speaker Diarization

Multi-model speaker diarization system that combines embeddings from 2 or more models (SpeechBrain, Pyannote, NeMo) using various fusion strategies.

## Features

- ✅ **Multiple Model Support**: Combine 2 or more models (SpeechBrain, Pyannote, NeMo)
- ✅ **7 Fusion Strategies**: concatenate, normalized_average, weighted_average, score_level, product, max_pool, learned_concat
- ✅ **Session Management**: Track speakers independently across multiple conversations
- ✅ **Memory Persistence**: Maintain speaker embeddings across audio chunks
- ✅ **Flexible Weighting**: Custom weights for each model in fusion

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- `speechbrain`
- `pyannote.audio`
- `nemo_toolkit[asr]`
- `torch`
- `numpy`
- `scipy`

## Quick Start

### Example 1: SpeechBrain + Pyannote

```python
from fusion_diarization import RealtimeSpeakerDiarization

pipeline = RealtimeSpeakerDiarization(
    models=['speechbrain', 'pyannote'],
    fusion_method="score_level",
    fusion_weights=[0.4, 0.6],  # 40% SB, 60% Pyannote
    model_configs={
        'pyannote': {
            'model_name': "pyannote/speaker-diarization-3.1",
            'token': "your_hf_token"
        }
    }
)

# Set session
pipeline.set_session("meeting_1")

# Process audio
output = pipeline(
    "audio.wav",
    num_speakers=2,
    use_memory=True,
    session_id="meeting_1"
)

print(output['speaker_labels'])
```

### Example 2: SpeechBrain + NeMo

```python
pipeline = RealtimeSpeakerDiarization(
    models=['speechbrain', 'nemo'],
    fusion_method="weighted_average",
    fusion_weights=[0.5, 0.5],
    model_configs={
        'nemo': {
            'output_dir': "nemo_temp_dir",
            'domain_type': "telephonic",
            'pretrained_speaker_model': "titanet_large"
        }
    }
)

pipeline.set_session("meeting_2")
output = pipeline("audio.wav", num_speakers=2, use_memory=True, session_id="meeting_2")
```

### Example 3: Pyannote + NeMo

```python
pipeline = RealtimeSpeakerDiarization(
    models=['pyannote', 'nemo'],
    fusion_method="normalized_average",
    model_configs={
        'pyannote': {
            'model_name': "pyannote/speaker-diarization-3.1",
            'token': "your_hf_token"
        },
        'nemo': {
            'output_dir': "nemo_temp_dir",
            'domain_type': "telephonic"
        }
    }
)

pipeline.set_session("meeting_3")
output = pipeline("audio.wav", num_speakers=2, use_memory=True, session_id="meeting_3")
```

### Example 4: All Three Models

```python
pipeline = RealtimeSpeakerDiarization(
    models=['speechbrain', 'pyannote', 'nemo'],
    fusion_method="score_level",
    fusion_weights=[0.3, 0.4, 0.3],  # 30% SB, 40% Pyannote, 30% NeMo
    model_configs={
        'pyannote': {
            'model_name': "pyannote/speaker-diarization-3.1",
            'token': "your_hf_token"
        },
        'nemo': {
            'output_dir': "nemo_temp_dir",
            'domain_type': "telephonic"
        }
    }
)

pipeline.set_session("meeting_4")

# Process multiple chunks
output1 = pipeline("chunk1.wav", num_speakers=2, use_memory=True, session_id="meeting_4")
output2 = pipeline("chunk2.wav", num_speakers=2, use_memory=True, session_id="meeting_4")
output3 = pipeline("chunk3.wav", num_speakers=2, use_memory=True, session_id="meeting_4")
```

## Fusion Strategies

### 1. Concatenate
Simple concatenation of embeddings: `E = [E1 ; E2 ; ...]`

```python
fusion_method="concatenate"
```

### 2. Normalized Average
Average of normalized embeddings: `E = (normalize(E1) + normalize(E2) + ...) / n`

```python
fusion_method="normalized_average"
```

### 3. Weighted Average
Weighted average with custom weights: `E = α*normalize(E1) + β*normalize(E2) + ...`

**Note**: Weights must sum to 1.0

```python
fusion_method="weighted_average"
fusion_weights=[0.3, 0.4, 0.3]  # Must sum to 1.0
```

### 4. Score Level
Compute similarities separately then combine: `final_score = α*s1 + β*s2 + ...`

**Note**: Weights must sum to 1.0

```python
fusion_method="score_level"
fusion_weights=[0.4, 0.6]  # Must sum to 1.0
```

### 5. Product
Element-wise product: `E = normalize(E1) ⊙ normalize(E2) ⊙ ...`

```python
fusion_method="product"
```

### 6. Max Pool
Max pooling: `E = max(normalize(E1), normalize(E2), ...)`

```python
fusion_method="max_pool"
```

### 7. Learned Concat
Weighted concatenation: `E = [w1*E1 ; w2*E2 ; ...]`

```python
fusion_method="learned_concat"
fusion_weights=[1.0, 1.5, 0.8]  # Can be any positive values
```

## Model Configurations

### SpeechBrain
```python
model_configs={
    'speechbrain': {
        'model_name': "speechbrain/spkrec-ecapa-voxceleb"
    }
}
```

### Pyannote
```python
model_configs={
    'pyannote': {
        'model_name': "pyannote/speaker-diarization-3.1",
        'token': "your_hf_token",  # Required
        'cache_dir': "./cache"
    }
}
```

### NeMo
```python
model_configs={
    'nemo': {
        'output_dir': "nemo_temp_dir",
        'domain_type': "telephonic",  # or "meeting", "general"
        'pretrained_speaker_model': "titanet_large"  # or "ecapa_tdnn"
    }
}
```

## Session Management

```python
# Create/switch session
pipeline.set_session("meeting_1")

# Get current session
current = pipeline.get_current_session_id()

# List all sessions
sessions = pipeline.list_sessions()

# Get speaker info
info = pipeline.get_speaker_info("meeting_1")
print(info['speakers'])        # ['SPEAKER_00', 'SPEAKER_01']
print(info['speaker_counts'])  # {SPEAKER_00: 5, SPEAKER_01: 3}
print(info['num_speakers'])    # 2

# Reset session (clear memory but keep session)
pipeline.reset_session("meeting_1")

# Delete session
pipeline.delete_session("meeting_1")

# Get all sessions info
all_info = pipeline.get_all_sessions_info()
```

## Advanced Parameters

```python
pipeline = RealtimeSpeakerDiarization(
    models=['speechbrain', 'pyannote'],
    fusion_method="score_level",
    fusion_weights=[0.5, 0.5],
    
    # Embedding alignment
    dimension_alignment="max",  # "min", "max", or "pad_zero"
    
    # Speaker matching
    similarity_threshold=0.7,              # Minimum similarity to match
    min_similarity_gap=0.3,                # Minimum gap for distinctive match
    init_similarity_threshold=0.4,         # Lower threshold for 2nd chunk
    
    # Embedding update
    embedding_update_weight=0.3,           # 30% new, 70% old
    skip_update_short_audio=True,          # Skip update for short audio
    min_duration_for_update=2.0,           # Min duration (seconds) to update
    
    # Model configs
    model_configs={...}
)
```

## Output Format

```python
output = pipeline("audio.wav", ...)

print(output['speaker_labels'])           # ['SPEAKER_00', 'SPEAKER_01']
print(output['speaker_embeddings'])       # Fused embeddings (n_speakers, dim)
print(output['speechbrain_embeddings'])   # SpeechBrain embeddings
print(output['pyannote_embeddings'])      # Pyannote embeddings
print(output['nemo_embeddings'])          # NeMo embeddings (if used)
```

## Running Examples

```bash
# Run all examples
python fusion_diarization_examples.py

# Or import specific examples
from fusion_diarization_examples import (
    example_speechbrain_pyannote,
    example_speechbrain_nemo,
    example_pyannote_nemo,
    example_all_three_models,
    test_all_fusion_methods
)

example_speechbrain_pyannote()
```

## Performance Tips

1. **Score-level fusion**: Best for diverse models with different strengths
2. **Weighted average**: Fast and effective when models have similar embeddings
3. **Concatenate**: Preserves all information but increases dimensionality
4. **Use appropriate weights**: Give higher weight to more accurate models

## Troubleshooting

### Weight validation error
```
ValueError: fusion_weights must sum to 1.0 for score_level
```
**Solution**: Ensure weights sum to 1.0 for `weighted_average` and `score_level` methods.

### Model initialization failed
```
⚠️ Failed to initialize pyannote: Invalid token
```
**Solution**: Set valid HuggingFace token for Pyannote in environment or config.

### NaN similarity warnings
```
⚠️ Warning: NaN in fused_score!
```
**Solution**: This usually indicates zero embeddings. Check audio quality and duration.

## Citation

If you use this fusion system, please cite the original models:

- **SpeechBrain**: Ravanelli et al., 2021
- **Pyannote**: Bredin et al., 2020
- **NeMo**: Kuchaiev et al., 2019

## License

MIT License


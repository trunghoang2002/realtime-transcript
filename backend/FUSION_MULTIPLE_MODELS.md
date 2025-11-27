# Fusion Diarization: Multiple Model Instances

## Overview

The Fusion Diarization system now supports using **multiple instances of the same model type** with different configurations. This allows you to combine the strengths of different model variants or configurations.

## Use Cases

1. **Multiple Pyannote versions**: Combine `pyannote/speaker-diarization-3.1` and `pyannote/speaker-diarization-community-1`
2. **Different SpeechBrain models**: Use different pretrained models or configurations
3. **Various NeMo models**: Combine `titanet_large` with `speakernet` or other variants

## Syntax

### Basic Format (Backward Compatible)

```python
# Simple list of model types (backward compatible)
models = ['speechbrain', 'pyannote', 'nemo']
```

### New Format: Multiple Instances

```python
# List of tuples: (model_type, model_id)
models = [
    ('pyannote', 'pyan_v3'),           # Pyannote v3.1
    ('pyannote', 'pyan_community'),    # Pyannote community
    ('speechbrain', 'sb_default')      # SpeechBrain default
]
```

- `model_type`: One of `'speechbrain'`, `'pyannote'`, or `'nemo'`
- `model_id`: Unique identifier for this instance (used as key in `model_configs`)

## Complete Example

```python
from fusion_diarization import RealtimeSpeakerDiarization
import os

# Initialize with multiple Pyannote instances
pipeline = RealtimeSpeakerDiarization(
    models=[
        ('pyannote', 'pyan_v3'),
        ('pyannote', 'pyan_community'),
        ('speechbrain', 'sb_default')
    ],
    fusion_method="score_level",
    fusion_weights=[0.3, 0.4, 0.3],  # Must sum to 1.0
    model_configs={
        'pyan_v3': {
            'model_name': 'pyannote/speaker-diarization-3.1',
            'token': os.getenv('HF_TOKEN')
        },
        'pyan_community': {
            'model_name': 'pyannote/speaker-diarization-community-1',
            'token': os.getenv('HF_TOKEN')
        },
        'sb_default': {}  # Use default SpeechBrain config
    },
    similarity_threshold=0.7,
    embedding_update_weight=0.3,
    min_similarity_gap=0.3
)

# Use it
pipeline.set_session("my_session")
output = pipeline(
    "audio.wav",
    num_speakers=2,
    use_memory=True,
    session_id="my_session"
)

print(f"Speaker labels: {output['speaker_labels']}")
print(f"Session info: {pipeline.get_speaker_info()}")
```

## Example: Multiple NeMo Models

```python
pipeline = RealtimeSpeakerDiarization(
    models=[
        ('nemo', 'nemo_titanet'),
        ('nemo', 'nemo_ecapa'),
        ('speechbrain', 'sb')
    ],
    fusion_method="weighted_average",
    fusion_weights=[0.4, 0.3, 0.3],
    model_configs={
        'nemo_titanet': {
            'pretrained_speaker_model': 'titanet_large'
        },
        'nemo_ecapa': {
            'pretrained_speaker_model': 'ecapa_tdnn'
        },
        'sb': {}
    }
)
```

## Example: Multiple SpeechBrain Instances

```python
pipeline = RealtimeSpeakerDiarization(
    models=[
        ('speechbrain', 'sb_resnet'),
        ('speechbrain', 'sb_ecapa'),
        ('pyannote', 'pyan')
    ],
    fusion_method="normalized_average",
    model_configs={
        'sb_resnet': {
            'embedding_model': 'speechbrain/spkrec-resnet-voxceleb'
        },
        'sb_ecapa': {
            'embedding_model': 'speechbrain/spkrec-ecapa-voxceleb'
        },
        'pyan': {
            'token': os.getenv('HF_TOKEN')
        }
    }
)
```

## Key Points

1. **Unique Model IDs**: Each model instance must have a unique `model_id`
2. **Backward Compatible**: Old code using simple strings still works
3. **Fusion Weights**: Number of weights must match number of model instances
4. **Model Configs**: Use `model_id` as key in `model_configs` dictionary
5. **Display Format**: Models are displayed as `type(id)` in logs, e.g., `pyannote(pyan_v3)`

## Benefits

- **Better Accuracy**: Combine strengths of different model versions
- **Robustness**: Reduce errors from individual model weaknesses
- **Flexibility**: Test different configurations without changing code structure
- **Experimentation**: Easy to compare fusion strategies with various model combinations

## Fusion Methods

All fusion methods work with multiple instances:
- `concatenate`: Concatenate all embeddings
- `normalized_average`: Average of normalized embeddings
- `weighted_average`: Weighted average (weights must sum to 1.0)
- `score_level`: Fuse similarity scores (weights must sum to 1.0)
- `product`: Element-wise product
- `max_pool`: Element-wise maximum
- `learned_concat`: Weighted concatenation

## Troubleshooting

### Duplicate Model IDs Error
```
ValueError: Duplicate model_ids found: [...]. Each model instance must have a unique ID.
```
**Solution**: Ensure each tuple has a unique second element (model_id).

### Wrong Number of Weights
```
ValueError: fusion_weights length (2) must match number of models (3)
```
**Solution**: Provide exactly one weight per model instance.

### Model Config Not Found
If a model doesn't find its config, it uses default settings. To ensure configs are applied:
- Use the same `model_id` in both the `models` list and `model_configs` dict


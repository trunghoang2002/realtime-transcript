# Max Speakers Constraint Handling

## 📋 Tổng quan

Cải tiến logic để **respect max_speakers constraint** khi matching speakers. Khi đã đạt số lượng speakers tối đa, hệ thống sẽ **force-assign** vào speaker có similarity cao nhất thay vì tạo speaker mới.

## 🎯 Vấn đề

### Behavior cũ (v2.0)

```python
# Có 2 speakers: SPEAKER_00, SPEAKER_01
# User set max_speakers=2

New embedding arrives:
  ├─ Tier 1: similarity with SPEAKER_00 = 0.47 ❌ (< 0.6)
  ├─ Tier 1: similarity with SPEAKER_01 = 0.32 ❌ (< 0.6)
  ├─ Tier 2: cluster similarity with SPEAKER_00 = 0.47 ❌
  └─ Tier 2: cluster similarity with SPEAKER_01 = 0.32 ❌

Result: Creates SPEAKER_02 ❌  # WRONG! Violates max_speakers=2
```

**Problem**: Hệ thống tạo speaker thứ 3 mặc dù user chỉ muốn tối đa 2 speakers!

## ✅ Giải pháp (v2.1)

### Behavior mới

```python
# Có 2 speakers: SPEAKER_00, SPEAKER_01
# User set max_speakers=2

New embedding arrives:
  ├─ Tier 1: similarity with SPEAKER_00 = 0.47 ❌ (< 0.6)
  ├─ Tier 1: similarity with SPEAKER_01 = 0.32 ❌ (< 0.6)
  ├─ Tier 2: cluster similarity with SPEAKER_00 = 0.47 ❌
  └─ Tier 2: cluster similarity with SPEAKER_01 = 0.32 ❌

Check constraint:
  ├─ Current speakers: 2
  └─ Max speakers: 2 → LIMIT REACHED!

Force-assign:
  ├─ Find best similarity (max of EMA and cluster): 0.47
  ├─ Assign to SPEAKER_00 ✅
  └─ Update EMA and cluster

Result: Assigned to SPEAKER_00 ✅  # CORRECT! Respects max_speakers=2
```

## 🔬 Algorithm Details

### Flow Diagram

```
New Embedding → Tier 1 → Match? → Yes → ✅ Assign
                   ↓
                   No
                   ↓
            Tier 2 → Match? → Yes → ✅ Assign
                      ↓
                      No
                      ↓
              Check Constraint
                      ↓
        ┌─────────────┴─────────────┐
        │                           │
    num < max?               num ≥ max?
        │                           │
        ↓                           ↓
  Create New Speaker      Force-Assign to Best
        🆕                          🔀
```

### Pseudocode

```python
def match_speaker(new_embedding, max_speakers=None):
    # Tier 1: EMA matching
    best_ema_match, best_ema_sim = find_best_ema_match(new_embedding)
    if best_ema_sim >= threshold:
        return assign(best_ema_match)
    
    # Tier 2: Cluster centroid matching
    best_cluster_match, best_cluster_sim = find_best_cluster_match(new_embedding)
    if best_cluster_sim >= threshold:
        return assign(best_cluster_match)
    
    # No match - check constraint
    current_num_speakers = len(speaker_memory)
    
    if max_speakers is not None and current_num_speakers >= max_speakers:
        # CONSTRAINT REACHED - Force assign
        print(f"⚠️ Max speakers ({max_speakers}) reached!")
        
        # Find absolute best match (even if < threshold)
        best_overall_sim = -1
        best_overall_speaker = None
        
        for speaker_id in speakers:
            ema_sim = similarity(new_embedding, ema[speaker_id])
            cluster_sim = similarity(new_embedding, centroid[speaker_id])
            max_sim = max(ema_sim, cluster_sim)
            
            if max_sim > best_overall_sim:
                best_overall_sim = max_sim
                best_overall_speaker = speaker_id
        
        print(f"🔀 Force-assigned to {best_overall_speaker} (sim: {best_overall_sim})")
        return assign(best_overall_speaker)
    
    else:
        # No constraint - create new speaker
        print(f"🆕 Created new speaker")
        return create_new_speaker(new_embedding)
```

## 📊 Examples

### Example 1: Two Speakers Limit

```python
pipeline = RealtimeSpeakerDiarization(
    similarity_threshold=0.6,
    max_cluster_size=20
)

# Chunk 1: Speaker A
output1 = pipeline(audio1, max_speakers=2)
# → Creates SPEAKER_00

# Chunk 2: Speaker B  
output2 = pipeline(audio2, max_speakers=2)
# → Creates SPEAKER_01

# Chunk 3: Speaker A (but similarity = 0.45 < threshold)
output3 = pipeline(audio3, max_speakers=2)
# Old behavior: Would create SPEAKER_02 ❌
# New behavior: Force-assigns to SPEAKER_00 ✅ (best match)
```

### Example 2: No Limit

```python
# Chunk 3: No max_speakers
output3 = pipeline(audio3)  # No max_speakers specified
# → Creates SPEAKER_02 (normal behavior)
```

### Example 3: Using num_speakers

```python
# Equivalent to max_speakers
output = pipeline(audio, num_speakers=2)
# → Will force-assign if 2 speakers already exist
```

## 🎓 Use Cases

### 1. Two-Person Interview

```python
# Interviewer + Guest = exactly 2 speakers
pipeline = RealtimeSpeakerDiarization(...)

for chunk in interview_audio:
    output = pipeline(chunk, num_speakers=2)  # Hard limit
    # Will never create SPEAKER_02!
```

### 2. Conference with Known Speakers

```python
# Panel of 5 speakers
for chunk in conference_audio:
    output = pipeline(chunk, max_speakers=5)
    # Can have 1-5 speakers, but won't exceed 5
```

### 3. Phone Call (2 speakers)

```python
# Caller + Agent = 2 speakers
for chunk in call_audio:
    output = pipeline(chunk, num_speakers=2)
    # Forces binary classification
```

## ⚙️ Configuration

### Parameters

```python
pipeline(
    audio,
    num_speakers=N,      # Hard limit (if specified, used as max)
    max_speakers=M,      # Soft limit (used if num_speakers not set)
    min_speakers=K       # Minimum (doesn't affect force-assign)
)
```

**Priority**:
1. `num_speakers` (highest) - If set, used as max limit
2. `max_speakers` - Used if num_speakers not set
3. `None` - No constraint, can create unlimited speakers

### Similarity Threshold

Force-assign **ignores threshold**:

```python
similarity_threshold = 0.7

# Scenario: All similarities < 0.7
#   - Without max_speakers: Creates new speaker
#   - With max_speakers (reached): Force-assigns to best (e.g., 0.45)
```

## 📈 Impact on Accuracy

### When it helps

✅ **Two-person conversations**: Prevents spurious 3rd speaker
✅ **Known speaker count**: Enforces domain knowledge
✅ **Noisy audio**: Prevents over-segmentation

### When to be careful

⚠️ **Unknown speaker count**: May force-assign actual new speaker
⚠️ **Very low similarity**: May assign incorrectly

**Recommendation**: Only use `max_speakers` when you have **strong prior knowledge** about speaker count.

## 🔍 Debugging

### Logs

```
[TIER 1] Label: SPEAKER_00
  Best EMA similarity: 0.470 with SPEAKER_00
  Threshold: 0.600
  ❌ EMA not matched, trying cluster centroids...
  [TIER 2] Cluster centroid similarity with SPEAKER_00: 0.470
  [TIER 2] Cluster centroid similarity with SPEAKER_01: 0.317
  ⚠️ Max speakers (2) reached! Force-assigning to best match...
  🔀 Force-assigned to SPEAKER_00 (similarity: 0.470)
  📊 Updated: EMA + added to cluster (size: 8)
```

**Key indicators**:
- `⚠️ Max speakers (N) reached!` - Constraint triggered
- `🔀 Force-assigned to X` - Shows which speaker chosen
- `(similarity: 0.XXX)` - Shows actual similarity (may be < threshold)

## 🧪 Testing

### Test Case 1: Constraint Enforced

```python
pipeline = RealtimeSpeakerDiarization(token="...", similarity_threshold=0.8)

# Create 2 speakers
pipeline(audio_A, max_speakers=2)  # SPEAKER_00
pipeline(audio_B, max_speakers=2)  # SPEAKER_01

# Try to create 3rd (but similarity < 0.8)
pipeline(audio_C, max_speakers=2)  # Should force-assign to SPEAKER_00 or 01

info = pipeline.get_speaker_info()
assert info['num_speakers'] == 2  # ✅ Constraint respected!
```

### Test Case 2: No Constraint

```python
# Same scenario, no max_speakers
pipeline.reset_context()

pipeline(audio_A)  # SPEAKER_00
pipeline(audio_B)  # SPEAKER_01  
pipeline(audio_C)  # SPEAKER_02 (created normally)

info = pipeline.get_speaker_info()
assert info['num_speakers'] == 3  # ✅ No constraint
```

## 🔄 Backward Compatibility

✅ **Fully backward compatible**

```python
# Old code still works
output = pipeline(audio)  # No max_speakers → behaves as before

# New code
output = pipeline(audio, max_speakers=5)  # With constraint
```

## 📝 API Changes

### v2.0 → v2.1

```python
# Method signature updated
def _match_speakers_with_memory(
    self, 
    new_embeddings,
    new_labels,
    max_speakers=None  # NEW parameter
):
    ...

def apply_realtime(
    self,
    file,
    hook=None,
    use_memory=True,
    max_speakers=None,  # NEW parameter
    **kwargs
):
    ...
```

### Behavior Changes

| Scenario | v2.0 | v2.1 |
|----------|------|------|
| No match, no max_speakers | Create new | Create new ✓ |
| No match, below max_speakers | Create new | Create new ✓ |
| No match, at max_speakers | Create new ❌ | Force-assign ✅ |

## 🎉 Summary

### What Changed

- ✅ Added `max_speakers` parameter to `_match_speakers_with_memory()`
- ✅ Added constraint checking logic
- ✅ Force-assign to best match when limit reached
- ✅ Detailed logging for force-assignments
- ✅ `num_speakers` now acts as hard limit

### Benefits

- 🎯 Respects domain knowledge (known speaker count)
- 🛡️ Prevents over-segmentation in constrained scenarios
- 📊 Better accuracy for two-person conversations
- 🔧 More control over speaker creation

### Usage

```python
# Two-person interview
output = pipeline(audio, num_speakers=2)

# Conference (up to 5 speakers)
output = pipeline(audio, max_speakers=5)

# Unknown count (no constraint)
output = pipeline(audio)  # Works as before
```

**Perfect for**: Interviews, phone calls, panels with known speaker count!


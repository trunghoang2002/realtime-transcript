# Patch SpeechBrain compatibility issue với huggingface_hub
import fix_speechbrain  # Phải import TRƯỚC speechbrain
from speechbrain.inference import EncoderClassifier
import torch
import numpy as np
from typing import Optional, Dict, List, Callable, Union
from scipy.spatial.distance import cdist
from get_audio import decode_audio

class RealtimeSpeakerDiarization():
    """Realtime Speaker Diarization Pipeline with persistent speaker embeddings"""
    
    def __init__(self, similarity_threshold=0.7,  # threshold để match speaker
                 embedding_update_weight=0.3,  # trọng số cập nhật embedding mới
                 min_similarity_gap=0.3,  # gap tối thiểu để match (nếu nổi bật)
                 preloaded_model=None,  # Optional: dùng model đã preload để tránh load lại
                 *args, **kwargs):

        # Context persistence cho realtime
        self.speaker_memory: Dict[str, np.ndarray] = {}  # {speaker_id: EMA embedding}
        self.speaker_embedding_clusters: Dict[str, List[np.ndarray]] = {}  # {speaker_id: [embeddings]}
        self.speaker_counts: Dict[str, int] = {}  # số lần xuất hiện của mỗi speaker
        self.speaker_history: List[Dict] = []  # lịch sử diarization
        self.similarity_threshold = similarity_threshold
        self.embedding_update_weight = embedding_update_weight
        self.min_similarity_gap = min_similarity_gap  # Gap threshold cho distinctive matching
        self.max_cluster_size = 20  # Giới hạn số embeddings trong cluster
        self.total_chunks_processed = 0
        self.next_speaker_id = 0
        self.embedding_metric = "cosine"

        # Load speaker embedding model (hoặc dùng preloaded)
        if preloaded_model is not None:
            self.spk_model = preloaded_model
        else:
            self.spk_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
    
    def reset_context(self):
        """Reset toàn bộ context - dùng khi bắt đầu conversation mới"""
        self.speaker_memory.clear()
        self.speaker_embedding_clusters.clear()
        self.speaker_counts.clear()
        self.speaker_history.clear()
        self.total_chunks_processed = 0
        self.next_speaker_id = 0
        
    def _match_speakers_with_memory(self, new_embeddings: np.ndarray, 
                                    new_labels: List[str],
                                    max_speakers: Optional[int] = None) -> Dict[str, str]:
        """
        Match speakers mới với speakers đã biết trong memory
        Sử dụng 2-tier matching: EMA embedding (fast) và cluster centroid (robust)
        
        Parameters
        ----------
        new_embeddings : (num_speakers, dimension) array
            Embeddings của speakers trong chunk hiện tại
        new_labels : List[str]
            Labels tạm thời của speakers mới
        max_speakers : int, optional
            Maximum number of speakers allowed. If reached, force-assign to best match.
            
        Returns
        -------
        mapping : Dict[str, str]
            Mapping từ label tạm thời -> speaker_id đã biết hoặc mới
        """
        if len(self.speaker_memory) == 0:
            # Chưa có speaker nào trong memory
            mapping = {}
            for i, label in enumerate(new_labels):
                print(f"Creating new speaker: {label} with id: {self.next_speaker_id:02d}")
                speaker_id = f"SPEAKER_{self.next_speaker_id:02d}"
                self.next_speaker_id += 1
                mapping[label] = speaker_id
                
                # Initialize EMA embedding và cluster
                self.speaker_memory[speaker_id] = new_embeddings[i].copy()
                self.speaker_embedding_clusters[speaker_id] = [new_embeddings[i].copy()]
                self.speaker_counts[speaker_id] = 1
            return mapping
        
        # Có speakers trong memory - tính similarity
        memory_speaker_ids = list(self.speaker_memory.keys())
        memory_embeddings = np.array([self.speaker_memory[sid] for sid in memory_speaker_ids])
        
        # Tính cosine similarity hoặc euclidean distance
        if self.embedding_metric == "cosine":
            # Cosine similarity (1 - cosine distance)
            distances = cdist(new_embeddings, memory_embeddings, metric='cosine')
            similarities_ema = 1 - distances
        else:
            # Euclidean distance -> similarity
            distances = cdist(new_embeddings, memory_embeddings, metric='euclidean')
            similarities_ema = 1 / (1 + distances)
        
        mapping = {}
        used_memory_speakers = set()
        
        # Greedy matching với 2-tier approach
        for i, label in enumerate(new_labels):
            new_embedding = new_embeddings[i]
            
            # Tier 1: So sánh với EMA embeddings (fast path)
            best_match_idx = np.argmax(similarities_ema[i])
            best_similarity_ema = similarities_ema[i, best_match_idx]
            best_speaker_id = memory_speaker_ids[best_match_idx]
            
            # Find second best for gap analysis
            sorted_indices = np.argsort(similarities_ema[i])[::-1]  # Descending order
            second_best_similarity_ema = similarities_ema[i, sorted_indices[1]] if len(sorted_indices) > 1 else -1
            similarity_gap_ema = best_similarity_ema - second_best_similarity_ema
            
            print(f"\n[TIER 1] Label: {label}")
            print(f"  Best EMA similarity: {best_similarity_ema:.3f} with {best_speaker_id}")
            print(f"  Second best similarity: {second_best_similarity_ema:.3f}")
            print(f"  Gap: {similarity_gap_ema:.3f}")
            print(f"  Threshold: {self.similarity_threshold}")
            
            matched_speaker_id = None
            
            # Check matching conditions
            if best_speaker_id not in used_memory_speakers:
                if best_similarity_ema >= self.similarity_threshold:
                    # Match via threshold
                    matched_speaker_id = best_speaker_id
                    print(f"  ✅ Matched via EMA (threshold)!")
                elif similarity_gap_ema > self.min_similarity_gap and second_best_similarity_ema > 0:
                    # Match via significant gap
                    matched_speaker_id = best_speaker_id
                    print(f"  ✅ Matched via EMA (significant gap > {self.min_similarity_gap})!")
                
            if matched_speaker_id is None:
                # Tier 2: So sánh với cluster centroids (robust path)
                print(f"  ❌ EMA not matched, trying cluster centroids...")
                
                cluster_similarities = []
                cluster_speaker_ids = []
                
                for speaker_id in memory_speaker_ids:
                    if speaker_id in used_memory_speakers:
                        continue
                    
                    # Tính centroid của cluster
                    cluster = self.speaker_embedding_clusters[speaker_id]
                    if len(cluster) > 0:
                        cluster_array = np.array(cluster)
                        centroid = np.mean(cluster_array, axis=0)
                        
                        # Normalize centroid nếu dùng cosine
                        if self.embedding_metric == "cosine":
                            centroid = centroid / np.linalg.norm(centroid)
                        
                        # Tính similarity với centroid
                        if self.embedding_metric == "cosine":
                            cluster_similarity = 1 - cdist([new_embedding], [centroid], metric='cosine')[0, 0]
                        else:
                            cluster_similarity = 1 / (1 + cdist([new_embedding], [centroid], metric='euclidean')[0, 0])
                        
                        print(f"  [TIER 2] Cluster centroid similarity with {speaker_id}: {cluster_similarity:.3f}")
                        
                        cluster_similarities.append(cluster_similarity)
                        cluster_speaker_ids.append(speaker_id)
                
                # Find best and second best for gap analysis
                if len(cluster_similarities) > 0:
                    sorted_cluster_indices = np.argsort(cluster_similarities)[::-1]
                    best_cluster_idx = sorted_cluster_indices[0]
                    best_cluster_similarity = cluster_similarities[best_cluster_idx]
                    best_cluster_speaker_id = cluster_speaker_ids[best_cluster_idx]
                    
                    second_best_cluster_similarity = (
                        cluster_similarities[sorted_cluster_indices[1]] 
                        if len(sorted_cluster_indices) > 1 
                        else -1
                    )
                    cluster_similarity_gap = best_cluster_similarity - second_best_cluster_similarity
                    
                    print(f"  [TIER 2] Best: {best_cluster_similarity:.3f}, Second: {second_best_cluster_similarity:.3f}, Gap: {cluster_similarity_gap:.3f}")
                    
                    # Check matching conditions for cluster
                    if best_cluster_similarity >= self.similarity_threshold:
                        matched_speaker_id = best_cluster_speaker_id
                        print(f"  ✅ Matched via cluster centroid (threshold) with {matched_speaker_id}!")
                    elif cluster_similarity_gap > self.min_similarity_gap and second_best_cluster_similarity > 0:
                        matched_speaker_id = best_cluster_speaker_id
                        print(f"  ✅ Matched via cluster centroid (significant gap > {self.min_similarity_gap}) with {matched_speaker_id}!")
            
            # Xử lý kết quả matching
            if matched_speaker_id is not None:
                # Match thành công
                mapping[label] = matched_speaker_id
                used_memory_speakers.add(matched_speaker_id)
                
                # Cập nhật EMA embedding
                old_embedding = self.speaker_memory[matched_speaker_id]
                updated_embedding = (1 - self.embedding_update_weight) * old_embedding + \
                                   self.embedding_update_weight * new_embedding
                # Normalize nếu dùng cosine similarity
                if self.embedding_metric == "cosine":
                    updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
                self.speaker_memory[matched_speaker_id] = updated_embedding
                
                # Add vào cluster (với limit size)
                cluster = self.speaker_embedding_clusters[matched_speaker_id]
                cluster.append(new_embedding.copy())
                # Keep only recent embeddings
                if len(cluster) > self.max_cluster_size:
                    self.speaker_embedding_clusters[matched_speaker_id] = cluster[-self.max_cluster_size:]
                
                self.speaker_counts[matched_speaker_id] += 1
                print(f"  📊 Updated: EMA + added to cluster (size: {len(self.speaker_embedding_clusters[matched_speaker_id])})")
                
            else:
                # Không match được - check max_speakers constraint
                current_num_speakers = len(self.speaker_memory)
                
                if max_speakers is not None and current_num_speakers >= max_speakers:
                    # ĐÃ ĐẠT max_speakers - Force assign vào speaker có similarity cao nhất
                    print(f"  ⚠️  Max speakers ({max_speakers}) reached! Force-assigning to best match...")
                    
                    # Tìm speaker có similarity cao nhất (dù < threshold)
                    best_overall_similarity = -1
                    best_overall_speaker_id = None
                    
                    # Check both EMA and cluster similarities
                    for speaker_idx, speaker_id in enumerate(memory_speaker_ids):
                        if speaker_id in used_memory_speakers:
                            continue
                        
                        # Get EMA similarity
                        ema_sim = similarities_ema[i, speaker_idx]
                        
                        # Get cluster centroid similarity
                        cluster = self.speaker_embedding_clusters[speaker_id]
                        if len(cluster) > 0:
                            cluster_array = np.array(cluster)
                            centroid = np.mean(cluster_array, axis=0)
                            if self.embedding_metric == "cosine":
                                centroid = centroid / np.linalg.norm(centroid)
                            
                            if self.embedding_metric == "cosine":
                                cluster_sim = 1 - cdist([new_embedding], [centroid], metric='cosine')[0, 0]
                            else:
                                cluster_sim = 1 / (1 + cdist([new_embedding], [centroid], metric='euclidean')[0, 0])
                        else:
                            cluster_sim = ema_sim
                        
                        # # Use max of EMA and cluster similarity
                        max_sim = max(ema_sim*1.3, cluster_sim)
                        
                        if max_sim > best_overall_similarity:
                            best_overall_similarity = max_sim
                            best_overall_speaker_id = speaker_id
                    
                    if best_overall_speaker_id is not None:
                        matched_speaker_id = best_overall_speaker_id
                        mapping[label] = matched_speaker_id
                        used_memory_speakers.add(matched_speaker_id)
                        
                        print(f"  🔀 Force-assigned to {matched_speaker_id} (similarity: {best_overall_similarity:.3f})")
                        
                        # Update như bình thường
                        old_embedding = self.speaker_memory[matched_speaker_id]
                        updated_embedding = (1 - self.embedding_update_weight) * old_embedding + \
                                           self.embedding_update_weight * new_embedding
                        if self.embedding_metric == "cosine":
                            updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
                        self.speaker_memory[matched_speaker_id] = updated_embedding
                        
                        cluster = self.speaker_embedding_clusters[matched_speaker_id]
                        cluster.append(new_embedding.copy())
                        if len(cluster) > self.max_cluster_size:
                            self.speaker_embedding_clusters[matched_speaker_id] = cluster[-self.max_cluster_size:]
                        
                        self.speaker_counts[matched_speaker_id] += 1
                        print(f"  📊 Updated: EMA + added to cluster (size: {len(self.speaker_embedding_clusters[matched_speaker_id])})")
                    else:
                        # Fallback: assign to first speaker (shouldn't happen)
                        fallback_speaker_id = memory_speaker_ids[0]
                        mapping[label] = fallback_speaker_id
                        print(f"  ⚠️  Fallback: assigned to {fallback_speaker_id}")
                else:
                    # Chưa đạt max_speakers - Tạo speaker mới
                    speaker_id = f"SPEAKER_{self.next_speaker_id:02d}"
                    self.next_speaker_id += 1
                    mapping[label] = speaker_id
                    
                    # Initialize EMA và cluster
                    self.speaker_memory[speaker_id] = new_embedding.copy()
                    self.speaker_embedding_clusters[speaker_id] = [new_embedding.copy()]
                    self.speaker_counts[speaker_id] = 1
                    print(f"  🆕 Created new speaker: {speaker_id} (current total: {len(self.speaker_memory)})")
                
        return mapping
    
    def apply_realtime(self, file_or_audio_f32: Union[str, np.ndarray], 
                      use_memory: bool = True,
                      max_speakers: Optional[int] = None,
                      **kwargs) -> dict:
        """
        Apply diarization với context memory cho realtime processing
        
        Parameters
        ----------
        file_or_audio_f32 : Union[str, np.ndarray]
            Audio chunk hiện tại hoặc file path
        use_memory : bool
            Có sử dụng speaker memory hay không (False = xử lý như batch bình thường)
        max_speakers : int, optional
            Maximum number of speakers. If reached, will force-assign to best match.
        
        Returns
        -------
        output : dict
            Kết quả diarization với speaker labels đã được map với memory
        """
        # Gọi phương thức apply gốc
        output = {}

        if isinstance(file_or_audio_f32, str):
            audio_f32 = decode_audio(file_or_audio_f32)
        else:
            audio_f32 = file_or_audio_f32

         # === Trích embedding ===
        # SpeechBrain yêu cầu tensor shape (batch, time)
        try:
            tensor = torch.tensor(audio_f32).unsqueeze(0)
            with torch.no_grad():
                emb = self.spk_model.encode_batch(tensor).detach().cpu().numpy().mean(axis=1)[0]
        except Exception:
            return output
    
        # Normalize embedding để ổn định hơn
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        emb_norm = np.expand_dims(emb_norm, axis=0)
        output['speaker_embeddings'] = emb_norm
        output['speaker_labels'] = ['SPEAKER_00']


        if not use_memory or output['speaker_embeddings'] is None:
            return output
        
        # Extract embeddings và labels từ output
        current_embeddings = output['speaker_embeddings']  # (num_speakers, dimension)
        current_labels = output['speaker_labels']
        
        
        # Match với speakers trong memory (pass max_speakers constraint)
        label_mapping = self._match_speakers_with_memory(
            current_embeddings, 
            current_labels,
            max_speakers=max_speakers
        )
        print(f"Label mapping: {label_mapping}")
        
        # Cập nhật embeddings theo thứ tự labels mới
        new_labels_ordered = list(label_mapping.values())
        print(f"New labels ordered: {new_labels_ordered}")
        updated_embeddings = np.array([self.speaker_memory[label] for label in new_labels_ordered])
        
        output['speaker_labels'] = new_labels_ordered
        output['speaker_embeddings'] = updated_embeddings

        # Lưu vào history
        self.speaker_history.append({
            'chunk_id': self.total_chunks_processed,
            'labels': new_labels_ordered,
            'num_speakers': len(new_labels_ordered)
        })
        self.total_chunks_processed += 1
        
        return output
    
    def get_speaker_info(self) -> Dict:
        """Lấy thông tin về các speakers đã biết"""
        cluster_sizes = {
            sid: len(self.speaker_embedding_clusters.get(sid, []))
            for sid in self.speaker_memory.keys()
        }
        return {
            'speakers': list(self.speaker_memory.keys()),
            'speaker_counts': self.speaker_counts.copy(),
            'cluster_sizes': cluster_sizes,
            'total_chunks': self.total_chunks_processed,
            'num_speakers': len(self.speaker_memory)
        }
    
    def __call__(self, file_or_audio_f32: Union[str, np.ndarray], num_speakers=None, min_speakers=None, max_speakers=None, 
              use_memory=True, **kwargs):
        """Override apply để sử dụng realtime mode mặc định"""
        # Determine effective max_speakers
        effective_max_speakers = num_speakers if num_speakers is not None else max_speakers
        
        return self.apply_realtime(
            file_or_audio_f32, 
            use_memory=use_memory, 
            num_speakers=num_speakers, 
            min_speakers=min_speakers, 
            max_speakers=effective_max_speakers,  # Pass effective limit
            **kwargs
        )

# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    # Khởi tạo realtime pipeline
    pipeline = RealtimeSpeakerDiarization(
        similarity_threshold=0.7,  # threshold để match speaker (càng cao càng strict)
        embedding_update_weight=0.3,  # trọng số update embedding (0.3 = 30% mới, 70% cũ)
        min_similarity_gap=0.3  # gap tối thiểu để match nếu nổi bật hơn hẳn
    )

    # Ví dụ 2: Xử lý chunk thứ 2 - speakers sẽ được match với chunk 1
    print("\n" + "=" * 100)
    print("VÍ DỤ 2: Xử lý audio chunk thứ 2 (giữ context)")
    print("=" * 100)

    output2 = pipeline(
        "wav/A1.wav",
        # min_speakers=1,
        # max_speakers=3,
        num_speakers=2,
        use_memory=True
    )

    print("\n📊 Kết quả chunk 2:")
    print(f"  🎤 {output2['speaker_labels']}")

    print(f"\n💾 Speaker Memory (updated): {pipeline.get_speaker_info()}")
    print("=" * 100)
    print("=" * 100)

    output3 = pipeline(
        "wav/B1.wav",
        # min_speakers=1,
        # max_speakers=3,
        num_speakers=2,
        use_memory=True
    )

    print("\n📊 Kết quả chunk 3:")
    print(f"  🎤 {output3['speaker_labels']}")

    print(f"\n💾 Speaker Memory (updated): {pipeline.get_speaker_info()}")
    print("=" * 100)
    print("=" * 100)

    output4 = pipeline(
        "wav/A2.wav",
        # min_speakers=1,
        # max_speakers=3,
        num_speakers=2,
        use_memory=True
    )

    print("\n📊 Kết quả chunk 4:")
    print(f"  🎤 {output4['speaker_labels']}")

    print(f"\n💾 Speaker Memory (updated): {pipeline.get_speaker_info()}")
    print("=" * 100)
    print("=" * 100)

    output5 = pipeline(
        "wav/B2.wav",
        # min_speakers=1,
        # max_speakers=3,
        num_speakers=2,
        use_memory=True
    )

    print("\n📊 Kết quả chunk 5:")
    print(f"  🎤 {output5['speaker_labels']}")

    print(f"\n💾 Speaker Memory (updated): {pipeline.get_speaker_info()}")
    print("=" * 100)
    print("=" * 100)

    output6 = pipeline(
        "wav/A3.wav",
        # min_speakers=1,
        # max_speakers=3,
        num_speakers=2,
        use_memory=True
    )

    print("\n📊 Kết quả chunk 6:")
    print(f"  🎤 {output6['speaker_labels']}")

    print(f"\n💾 Speaker Memory (updated): {pipeline.get_speaker_info()}")
    print("=" * 100)
    print("=" * 100)

    output7 = pipeline(
        "wav/B3.wav",
        # min_speakers=1,
        # max_speakers=3,
        num_speakers=2,
        use_memory=True
        )
    print("\n📊 Kết quả chunk 7:")
    print(f"  🎤 {output7['speaker_labels']}")

    print(f"\n💾 Speaker Memory (updated): {pipeline.get_speaker_info()}")
    print("=" * 100)
    print("=" * 100)

    # Ví dụ 3: Reset context và bắt đầu conversation mới
    print("\n" + "=" * 60)
    print("VÍ DỤ 3: Reset context và bắt đầu lại")
    print("=" * 60)

    pipeline.reset_context()
    print(f"✅ Context đã reset: {pipeline.get_speaker_info()}")

    # Ví dụ 4: Disable memory (xử lý như batch mode bình thường)
    print("\n" + "=" * 60)
    print("VÍ DỤ 4: Xử lý không dùng memory (batch mode)")
    print("=" * 60)

    output9 = pipeline(
        # "/home/hoang/realtime-transcript/backend/eval/TestJ.mp3",
        "wav/A1.wav",
        # min_speakers=1,
        # max_speakers=3,
        num_speakers=2,
        use_memory=False  # Không dùng memory
    )

    print("\n📊 Kết quả batch mode:")
    print(f"  🎤 {output9['speaker_labels']}")
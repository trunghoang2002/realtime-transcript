from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization, DiarizeOutput
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio.utils.signal import binarize
from pyannote.core import Annotation, Segment
import torch
import numpy as np
from typing import Optional, Dict, List, Callable, Union
from scipy.spatial.distance import cdist
import soundfile as sf
import tempfile
import os

class RealtimeSpeakerDiarization(SpeakerDiarization):
    """
    Realtime Speaker Diarization Pipeline with persistent speaker embeddings and session management
    
    This class extends the standard SpeakerDiarization pipeline to support:
    1. Persistent speaker tracking across multiple audio chunks
    2. Multi-session management to handle different conversations independently
    3. 2-tier speaker matching (EMA embeddings + cluster centroids)
    
    Session Management:
    -------------------
    - Each session maintains its own speaker memory and embeddings
    - Sessions are identified by unique session_id strings
    - Multiple sessions can be active simultaneously without interference
    - Use set_session(session_id) to switch between sessions
    - Use get_speaker_info(session_id) to query session state
    
    Example Usage:
    --------------
    ```python
    # Initialize pipeline
    pipeline = RealtimeSpeakerDiarization(token="your_hf_token")
    
    # Process conversation 1
    pipeline.set_session("meeting_1")
    output1 = pipeline("audio1.wav", use_memory=True, session_id="meeting_1")
    
    # Process conversation 2 (independent speakers)
    pipeline.set_session("meeting_2")
    output2 = pipeline("audio2.wav", use_memory=True, session_id="meeting_2")
    
    # Continue conversation 1
    pipeline.set_session("meeting_1")
    output3 = pipeline("audio3.wav", use_memory=True, session_id="meeting_1")
    
    # Check all sessions
    print(pipeline.list_sessions())
    print(pipeline.get_all_sessions_info())
    ```
    """
    
    def __init__(self, model_name="pyannote/speaker-diarization-community-1", 
                 token=None, cache_dir=None, 
                 similarity_threshold=0.7,  # threshold để match speaker
                 embedding_update_weight=0.3,  # trọng số cập nhật embedding mới
                 min_similarity_gap=0.3,  # gap tối thiểu để match (nếu nổi bật)
                 *args, **kwargs):
        super().__init__(
            segmentation={"checkpoint": model_name, "subfolder": "segmentation"},
            embedding={"checkpoint": model_name, "subfolder": "embedding"},
            plda={"checkpoint": model_name, "subfolder": "plda"},
            token=token, cache_dir=cache_dir, *args, **kwargs)

        # Session management
        self.current_session_id: Optional[str] = None
        self.sessions: Dict[str, Dict] = {}  # {session_id: session_data}
        
        # Config parameters
        self.similarity_threshold = similarity_threshold
        self.embedding_update_weight = embedding_update_weight
        self.min_similarity_gap = min_similarity_gap  # Gap threshold cho distinctive matching
        self.max_cluster_size = 20  # Giới hạn số embeddings trong cluster
    
    def _get_session_data(self, session_id: Optional[str] = None) -> Dict:
        """
        Lấy session data. Nếu không có session_id, dùng current_session_id.
        Tự động tạo session mới nếu chưa tồn tại.
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None:
            raise ValueError("No session_id provided and no current session set. Call set_session() first.")
        
        if session_id not in self.sessions:
            # Tạo session mới
            self.sessions[session_id] = {
                'speaker_memory': {},  # {speaker_id: EMA embedding}
                'speaker_embedding_clusters': {},  # {speaker_id: [embeddings]}
                'speaker_counts': {},  # số lần xuất hiện của mỗi speaker
                'speaker_history': [],  # lịch sử diarization
                'total_chunks_processed': 0,
                'next_speaker_id': 0,
                'created_at': None,
                'last_updated': None
            }
            print(f"📝 Created new session: {session_id}")
        
        return self.sessions[session_id]
    
    def set_session(self, session_id: str):
        """
        Chuyển sang session cụ thể. Tự động tạo session mới nếu chưa tồn tại.
        
        Parameters
        ----------
        session_id : str
            ID của session cần chuyển sang
        """
        self.current_session_id = session_id
        self._get_session_data(session_id)  # Đảm bảo session tồn tại
        print(f"✅ Switched to session: {session_id}")
    
    def get_current_session_id(self) -> Optional[str]:
        """Lấy ID của session hiện tại"""
        return self.current_session_id
    
    def list_sessions(self) -> List[str]:
        """Liệt kê tất cả session IDs"""
        return list(self.sessions.keys())
    
    def delete_session(self, session_id: str):
        """
        Xóa một session cụ thể
        
        Parameters
        ----------
        session_id : str
            ID của session cần xóa
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"🗑️  Deleted session: {session_id}")
            
            # Nếu đang ở session bị xóa, reset current_session_id
            if self.current_session_id == session_id:
                self.current_session_id = None
                print("⚠️  Current session was deleted. Please set a new session.")
        else:
            print(f"⚠️  Session not found: {session_id}")
    
    def reset_session(self, session_id: Optional[str] = None):
        """
        Reset một session cụ thể (xóa toàn bộ speaker memory nhưng giữ session)
        
        Parameters
        ----------
        session_id : str, optional
            ID của session cần reset. Nếu None, reset session hiện tại.
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None:
            raise ValueError("No session_id provided and no current session set.")
        
        if session_id in self.sessions:
            self.sessions[session_id] = {
                'speaker_memory': {},
                'speaker_embedding_clusters': {},
                'speaker_counts': {},
                'speaker_history': [],
                'total_chunks_processed': 0,
                'next_speaker_id': 0,
                'created_at': self.sessions[session_id].get('created_at'),
                'last_updated': None
            }
            print(f"🔄 Reset session: {session_id}")
        else:
            print(f"⚠️  Session not found: {session_id}")
        
    def _match_speakers_with_memory(self, new_embeddings: np.ndarray, 
                                    new_labels: List[str],
                                    max_speakers: Optional[int] = None,
                                    session_id: Optional[str] = None) -> Dict[str, str]:
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
        session_id : str, optional
            Session ID to use. If None, uses current session.
            
        Returns
        -------
        mapping : Dict[str, str]
            Mapping từ label tạm thời -> speaker_id đã biết hoặc mới
        """
        session_data = self._get_session_data(session_id)
        speaker_memory = session_data['speaker_memory']
        speaker_embedding_clusters = session_data['speaker_embedding_clusters']
        speaker_counts = session_data['speaker_counts']
        
        if len(speaker_memory) == 0:
            # Chưa có speaker nào trong memory
            mapping = {}
            for i, label in enumerate(new_labels):
                next_id = session_data['next_speaker_id']
                print(f"Creating new speaker: {label} with id: {next_id:02d}")
                speaker_id = f"SPEAKER_{next_id:02d}"
                session_data['next_speaker_id'] += 1
                mapping[label] = speaker_id
                
                # Initialize EMA embedding và cluster
                speaker_memory[speaker_id] = new_embeddings[i].copy()
                speaker_embedding_clusters[speaker_id] = [new_embeddings[i].copy()]
                speaker_counts[speaker_id] = 1
            return mapping
        
        # Có speakers trong memory - tính similarity
        memory_speaker_ids = list(speaker_memory.keys())
        memory_embeddings = np.array([speaker_memory[sid] for sid in memory_speaker_ids])
        
        # Tính cosine similarity hoặc euclidean distance
        if self._embedding.metric == "cosine":
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
                    cluster = speaker_embedding_clusters[speaker_id]
                    if len(cluster) > 0:
                        cluster_array = np.array(cluster)
                        centroid = np.mean(cluster_array, axis=0)
                        
                        # Normalize centroid nếu dùng cosine
                        if self._embedding.metric == "cosine":
                            centroid = centroid / np.linalg.norm(centroid)
                        
                        # Tính similarity với centroid
                        if self._embedding.metric == "cosine":
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
                old_embedding = speaker_memory[matched_speaker_id]
                updated_embedding = (1 - self.embedding_update_weight) * old_embedding + \
                                   self.embedding_update_weight * new_embedding
                # Normalize nếu dùng cosine similarity
                if self._embedding.metric == "cosine":
                    updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
                speaker_memory[matched_speaker_id] = updated_embedding
                
                # Add vào cluster (với limit size)
                cluster = speaker_embedding_clusters[matched_speaker_id]
                cluster.append(new_embedding.copy())
                # Keep only recent embeddings
                if len(cluster) > self.max_cluster_size:
                    speaker_embedding_clusters[matched_speaker_id] = cluster[-self.max_cluster_size:]
                
                speaker_counts[matched_speaker_id] += 1
                print(f"  📊 Updated: EMA + added to cluster (size: {len(speaker_embedding_clusters[matched_speaker_id])})")
                
            else:
                # Không match được - check max_speakers constraint
                current_num_speakers = len(speaker_memory)
                
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
                        cluster = speaker_embedding_clusters[speaker_id]
                        if len(cluster) > 0:
                            cluster_array = np.array(cluster)
                            centroid = np.mean(cluster_array, axis=0)
                            if self._embedding.metric == "cosine":
                                centroid = centroid / np.linalg.norm(centroid)
                            
                            if self._embedding.metric == "cosine":
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
                        old_embedding = speaker_memory[matched_speaker_id]
                        updated_embedding = (1 - self.embedding_update_weight) * old_embedding + \
                                           self.embedding_update_weight * new_embedding
                        if self._embedding.metric == "cosine":
                            updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
                        speaker_memory[matched_speaker_id] = updated_embedding
                        
                        cluster = speaker_embedding_clusters[matched_speaker_id]
                        cluster.append(new_embedding.copy())
                        if len(cluster) > self.max_cluster_size:
                            speaker_embedding_clusters[matched_speaker_id] = cluster[-self.max_cluster_size:]
                        
                        speaker_counts[matched_speaker_id] += 1
                        print(f"  📊 Updated: EMA + added to cluster (size: {len(speaker_embedding_clusters[matched_speaker_id])})")
                    else:
                        # Fallback: assign to first speaker (shouldn't happen)
                        fallback_speaker_id = memory_speaker_ids[0]
                        mapping[label] = fallback_speaker_id
                        print(f"  ⚠️  Fallback: assigned to {fallback_speaker_id}")
                else:
                    # Chưa đạt max_speakers - Tạo speaker mới
                    next_id = session_data['next_speaker_id']
                    speaker_id = f"SPEAKER_{next_id:02d}"
                    session_data['next_speaker_id'] += 1
                    mapping[label] = speaker_id
                    
                    # Initialize EMA và cluster
                    speaker_memory[speaker_id] = new_embedding.copy()
                    speaker_embedding_clusters[speaker_id] = [new_embedding.copy()]
                    speaker_counts[speaker_id] = 1
                    print(f"  🆕 Created new speaker: {speaker_id} (current total: {len(speaker_memory)})")
                
        return mapping
    
    def apply_realtime(self, audio_file: str, 
                      hook: Optional[Callable] = None,
                      use_memory: bool = True,
                      max_speakers: Optional[int] = None,
                      session_id: Optional[str] = None,
                      **kwargs) -> DiarizeOutput:
        """
        Apply diarization với context memory cho realtime processing
        
        Parameters
        ----------
        audio_file : str
            Audio file path hiện tại
        hook : Optional[Callable]
            Progress hook
        use_memory : bool
            Có sử dụng speaker memory hay không (False = xử lý như batch bình thường)
        max_speakers : int, optional
            Maximum number of speakers. If reached, will force-assign to best match.
        session_id : str, optional
            Session ID to use. If None, uses current session. Required if use_memory=True.
        
        Returns
        -------
        output : DiarizeOutput
            Kết quả diarization với speaker labels đã được map với memory
        """
        # Set session if provided
        if use_memory and session_id is not None:
            self.set_session(session_id)
        elif use_memory and self.current_session_id is None:
            raise ValueError("No session_id provided and no current session set. "
                           "Call set_session() or provide session_id parameter.")

        # Gọi phương thức apply gốc
        output = super().apply(audio_file, hook=hook, max_speakers=max_speakers, **kwargs)
        
        if not use_memory or output.speaker_embeddings is None:
            return output
        
        # Extract embeddings và labels từ output
        current_embeddings = output.speaker_embeddings  # (num_speakers, dimension)
        current_labels = list(output.speaker_diarization.labels())
        
        if len(current_labels) == 0:
            return output
        
        # Get session data
        session_data = self._get_session_data()
        
        # Sort labels theo thứ tự thời gian xuất hiện để đảm bảo consistency
        # (SPEAKER_00 nên là người xuất hiện đầu tiên)
        if len(session_data['speaker_memory']) == 0:  # Chỉ sort cho chunk đầu tiên
            label_first_appearance = {}
            for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
                if speaker not in label_first_appearance:
                    label_first_appearance[speaker] = turn.start
            
            # Sort labels theo thời gian xuất hiện
            sorted_labels = sorted(current_labels, key=lambda x: label_first_appearance.get(x, float('inf')))
            
            # Reorder embeddings theo sorted labels
            label_to_idx = {label: i for i, label in enumerate(current_labels)}
            sorted_embeddings = np.array([current_embeddings[label_to_idx[label]] for label in sorted_labels])
            
            current_labels = sorted_labels
            current_embeddings = sorted_embeddings
            
            print(f"Sorted labels by appearance time: {current_labels}")
        else:
            print(f"Current labels: {current_labels}")
        
        # Match với speakers trong memory (pass max_speakers constraint)
        label_mapping = self._match_speakers_with_memory(
            current_embeddings, 
            current_labels,
            max_speakers=max_speakers
        )
        print(f"Label mapping: {label_mapping}")
        
        # Rename labels trong annotation
        diarization = output.speaker_diarization.rename_labels(mapping=label_mapping)
        print(f"Diarization: {diarization}")
        exclusive_diarization = output.exclusive_speaker_diarization.rename_labels(mapping=label_mapping)
        
        # Cập nhật embeddings theo thứ tự labels mới
        new_labels_ordered = list(diarization.labels())
        print(f"New labels ordered: {new_labels_ordered}")
        updated_embeddings = np.array([session_data['speaker_memory'][label] for label in new_labels_ordered])
        
        # Tạo output mới
        new_output = DiarizeOutput(
            speaker_diarization=diarization,
            exclusive_speaker_diarization=exclusive_diarization,
            speaker_embeddings=updated_embeddings
        )
        
        # Lưu vào history
        session_data['speaker_history'].append({
            'chunk_id': session_data['total_chunks_processed'],
            'labels': new_labels_ordered,
            'num_speakers': len(new_labels_ordered)
        })
        session_data['total_chunks_processed'] += 1
        
        return new_output
    
    def get_speaker_info(self, session_id: Optional[str] = None) -> Dict:
        """
        Lấy thông tin về các speakers đã biết trong session
        
        Parameters
        ----------
        session_id : str, optional
            Session ID to get info for. If None, uses current session.
            
        Returns
        -------
        info : Dict
            Dictionary containing speaker information for the session
        """
        if session_id is None and self.current_session_id is None:
            return {
                'session_id': None,
                'speakers': [],
                'speaker_counts': {},
                'cluster_sizes': {},
                'total_chunks': 0,
                'num_speakers': 0
            }
        
        session_data = self._get_session_data(session_id)
        cluster_sizes = {
            sid: len(session_data['speaker_embedding_clusters'].get(sid, []))
            for sid in session_data['speaker_memory'].keys()
        }
        return {
            'session_id': session_id or self.current_session_id,
            'speakers': list(session_data['speaker_memory'].keys()),
            'speaker_counts': session_data['speaker_counts'].copy(),
            'cluster_sizes': cluster_sizes,
            'total_chunks': session_data['total_chunks_processed'],
            'num_speakers': len(session_data['speaker_memory'])
        }
    
    def get_all_sessions_info(self) -> Dict[str, Dict]:
        """
        Lấy thông tin về tất cả các sessions
        
        Returns
        -------
        info : Dict[str, Dict]
            Dictionary mapping session_id to session info
        """
        return {
            session_id: self.get_speaker_info(session_id)
            for session_id in self.sessions.keys()
        }
    
    def apply(self, file_or_audio_f32: Union[str, np.ndarray], hook=None, num_speakers=None, min_speakers=None, max_speakers=None, 
              use_memory=True, session_id=None, **kwargs):
        """Override apply để sử dụng realtime mode mặc định"""
        # Determine effective max_speakers
        effective_max_speakers = num_speakers if num_speakers is not None else max_speakers
        
        return self.apply_realtime(
            file_or_audio_f32, 
            hook=hook, 
            use_memory=use_memory, 
            num_speakers=num_speakers, 
            min_speakers=min_speakers, 
            max_speakers=effective_max_speakers,  # Pass effective limit
            session_id=session_id,
            **kwargs
        )

# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    # Khởi tạo realtime pipeline
    import time
    start_time = time.time()
    from dotenv import load_dotenv
    load_dotenv()
    pipeline = RealtimeSpeakerDiarization(
        model_name="pyannote/speaker-diarization-community-1", 
        token=os.getenv("HF_TOKEN"),
        similarity_threshold=0.7,  # threshold để match speaker (càng cao càng strict)
        embedding_update_weight=0.3,  # trọng số update embedding (0.3 = 30% mới, 70% cũ)
        min_similarity_gap=0.3  # gap tối thiểu để match nếu nổi bật hơn hẳn
    )

    # Send pipeline to GPU (when available)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    end_time = time.time()
    print(f"Time taken to initialize pipeline: {end_time - start_time:.2f} seconds")
    
    # Ví dụ 1: Xử lý một audio chunk với session_id (realtime mode)
    print("=" * 100)
    print("VÍ DỤ 1: Xử lý audio chunk đầu tiên với Session ID")
    print("=" * 100)
    
    # Set session for this conversation
    session_1 = "conversation_1"
    pipeline.set_session(session_1)

    with ProgressHook() as hook:
        output1 = pipeline(
            "/home/hoang/speaker_diarization/wav/TestE_sub.wav",
            hook=hook,
            # min_speakers=1,
            # max_speakers=3,
            num_speakers=2,
            use_memory=True,  # Sử dụng memory để track speakers
            session_id=session_1  # Chỉ định session
        )

    print("\n📊 Kết quả chunk 1:") # 1 thời điểm có thể có nhiều speaker nói
    for turn, _, speaker in output1.speaker_diarization.itertracks(yield_label=True):
        print(f"  ⏱️  {turn.start:.1f}s → {turn.end:.1f}s | 🎤 {speaker}")

    print("\n📊 Kết quả chunk 1 exclusive:") # 1 thời điểm chỉ có 1 speaker nói
    for turn, _, speaker in output1.exclusive_speaker_diarization.itertracks(yield_label=True):
        print(f"exclusive  ⏱️  {turn.start:.1f}s → {turn.end:.1f}s | 🎤 {speaker}")

    print(f"\n💾 Speaker Memory (Session {session_1}): {pipeline.get_speaker_info(session_1)}")
    print(f"\n📋 All Sessions: {pipeline.list_sessions()}")

    # Ví dụ 2: Xử lý một conversation khác với session_id khác
    print("\n" + "=" * 100)
    print("VÍ DỤ 2: Xử lý conversation thứ 2 với session khác")
    print("=" * 100)
    
    session_2 = "conversation_2"
    pipeline.set_session(session_2)
    
    with ProgressHook() as hook:
        output2 = pipeline(
            "/home/hoang/speaker_diarization/wav/A2.wav",
            hook=hook,
            num_speakers=2,
            use_memory=True,
            session_id=session_2
        )
    
    print("\n📊 Kết quả chunk 1 (Session 2):")
    for turn, _, speaker in output2.speaker_diarization.itertracks(yield_label=True):
        print(f"  ⏱️  {turn.start:.1f}s → {turn.end:.1f}s | 🎤 {speaker}")
    
    print(f"\n💾 Speaker Memory (Session {session_2}): {pipeline.get_speaker_info(session_2)}")
    print(f"💾 Speaker Memory (Session {session_1}): {pipeline.get_speaker_info(session_1)}")
    print(f"\n📋 All Sessions: {pipeline.list_sessions()}")
    
    # Ví dụ 3: Quay lại session 1 và xử lý chunk tiếp theo
    print("\n" + "=" * 100)
    print("VÍ DỤ 3: Quay lại Session 1 và xử lý chunk tiếp theo")
    print("=" * 100)
    
    pipeline.set_session(session_1)  # Chuyển về session 1

    with ProgressHook() as hook:
        output3 = pipeline(
            "/home/hoang/speaker_diarization/wav/A1.wav",
            hook=hook,
            num_speakers=2,
            use_memory=True,
            session_id=session_1  # Continue with session 1
        )

    print("\n📊 Kết quả chunk 2 (Session 1 continued):")
    for turn, _, speaker in output3.speaker_diarization.itertracks(yield_label=True):
        print(f"  ⏱️  {turn.start:.1f}s → {turn.end:.1f}s | 🎤 {speaker}")

    print(f"\n💾 Speaker Memory (Session 1 updated): {pipeline.get_speaker_info(session_1)}")
    
    # Ví dụ 4: Demo session management operations
    print("\n" + "=" * 100)
    print("VÍ DỤ 4: Session Management - Reset và Delete")
    print("=" * 100)
    
    # Liệt kê tất cả sessions
    print(f"\n📋 Current sessions: {pipeline.list_sessions()}")
    print(f"📍 Current session ID: {pipeline.get_current_session_id()}")
    
    # Reset một session (xóa speaker memory nhưng giữ session)
    print(f"\n🔄 Resetting session: {session_1}")
    pipeline.reset_session(session_1)
    print(f"💾 Speaker Memory after reset: {pipeline.get_speaker_info(session_1)}")
    
    # Delete một session
    # print(f"\n🗑️  Deleting session: {session_2}")
    # pipeline.delete_session(session_2)
    # print(f"📋 Remaining sessions: {pipeline.list_sessions()}")
    
    # Xem thông tin tất cả sessions
    print(f"\n📊 All sessions info:")
    for session_id, info in pipeline.get_all_sessions_info().items():
        print(f"  Session '{session_id}': {info['num_speakers']} speakers, {info['total_chunks']} chunks")
    
    print("=" * 100)

    with ProgressHook() as hook:
        output3 = pipeline(
            "/home/hoang/speaker_diarization/wav/B1.wav",
            hook=hook,
            # min_speakers=1,
            # max_speakers=3,
            num_speakers=2,
            use_memory=True
        )

    print("\n📊 Kết quả chunk 3:")
    for turn, _, speaker in output3.speaker_diarization.itertracks(yield_label=True):
        print(f"  ⏱️  {turn.start:.1f}s → {turn.end:.1f}s | 🎤 {speaker}")

    print(f"\n💾 Speaker Memory (updated): {pipeline.get_speaker_info()}")
    print("=" * 100)
    print("=" * 100)

    with ProgressHook() as hook:
        output4 = pipeline(
            "/home/hoang/speaker_diarization/wav/A2.wav",
            hook=hook,
            # min_speakers=1,
            # max_speakers=3,
            num_speakers=2,
            use_memory=True
        )

    print("\n📊 Kết quả chunk 4:")
    for turn, _, speaker in output4.speaker_diarization.itertracks(yield_label=True):
        print(f"  ⏱️  {turn.start:.1f}s → {turn.end:.1f}s | 🎤 {speaker}")

    print(f"\n💾 Speaker Memory (updated): {pipeline.get_speaker_info()}")
    print("=" * 100)
    print("=" * 100)

    with ProgressHook() as hook:
        output5 = pipeline(
            "/home/hoang/speaker_diarization/wav/B2.wav",
            hook=hook,
            # min_speakers=1,
            # max_speakers=3,
            num_speakers=2,
            use_memory=True
        )

    print("\n📊 Kết quả chunk 5:")
    for turn, _, speaker in output5.speaker_diarization.itertracks(yield_label=True):
        print(f"  ⏱️  {turn.start:.1f}s → {turn.end:.1f}s | 🎤 {speaker}")

    print(f"\n💾 Speaker Memory (updated): {pipeline.get_speaker_info()}")
    print("=" * 100)
    print("=" * 100)

    with ProgressHook() as hook:
        output6 = pipeline(
            "/home/hoang/speaker_diarization/wav/A3.wav",
            hook=hook,
            # min_speakers=1,
            # max_speakers=3,
            num_speakers=2,
            use_memory=True
        )

    print("\n📊 Kết quả chunk 6:")
    for turn, _, speaker in output6.speaker_diarization.itertracks(yield_label=True):
        print(f"  ⏱️  {turn.start:.1f}s → {turn.end:.1f}s | 🎤 {speaker}")

    print(f"\n💾 Speaker Memory (updated): {pipeline.get_speaker_info()}")
    print("=" * 100)
    print("=" * 100)

    with ProgressHook() as hook:
        output7 = pipeline(
            "/home/hoang/speaker_diarization/wav/B3.wav",
            hook=hook,
            # min_speakers=1,
            # max_speakers=3,
            num_speakers=2,
            use_memory=True
        )
    print("\n📊 Kết quả chunk 7:")
    for turn, _, speaker in output7.speaker_diarization.itertracks(yield_label=True):
        print(f"  ⏱️  {turn.start:.1f}s → {turn.end:.1f}s | 🎤 {speaker}")

    print(f"\n💾 Speaker Memory (updated): {pipeline.get_speaker_info()}")
    print("=" * 100)
    print("=" * 100)

    with ProgressHook() as hook:
        output8 = pipeline(
            "/home/hoang/speaker_diarization/wav/TestE_sub.wav",
            hook=hook,
            # min_speakers=1,
            # max_speakers=3,
            num_speakers=2,
            use_memory=True
        )
    print("\n📊 Kết quả chunk 8:")
    for turn, _, speaker in output8.speaker_diarization.itertracks(yield_label=True):
        print(f"  ⏱️  {turn.start:.1f}s → {turn.end:.1f}s | 🎤 {speaker}")

    print(f"\n💾 Speaker Memory (updated): {pipeline.get_speaker_info()}")
    print("=" * 100)
    print("=" * 100)

    # Ví dụ 3: Reset context và bắt đầu conversation mới
    print("\n" + "=" * 60)
    print("VÍ DỤ 3: Reset context và bắt đầu lại")
    print("=" * 60)

    pipeline.reset_session()
    print(f"✅ Context đã reset: {pipeline.get_speaker_info()}")

    # Ví dụ 4: Disable memory (xử lý như batch mode bình thường)
    print("\n" + "=" * 60)
    print("VÍ DỤ 4: Xử lý không dùng memory (batch mode)")
    print("=" * 60)

    with ProgressHook() as hook:
        output3 = pipeline(
            # "/home/hoang/realtime-transcript/backend/eval/TestJ.mp3",
            "/home/hoang/speaker_diarization/wav/A1.wav",
            hook=hook,
            # min_speakers=1,
            # max_speakers=3,
            num_speakers=2,
            use_memory=False  # Không dùng memory
        )

    print("\n📊 Kết quả batch mode:")
    for turn, _, speaker in output3.speaker_diarization.itertracks(yield_label=True):
        print(f"  ⏱️  {turn.start:.1f}s → {turn.end:.1f}s | 🎤 {speaker}")
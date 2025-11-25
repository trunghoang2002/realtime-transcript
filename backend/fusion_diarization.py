"""
Fusion Speaker Diarization: Combines Pyannote and SpeechBrain embeddings
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Callable
from scipy.spatial.distance import cdist
import soundfile as sf
import tempfile

# Import both diarization systems
from pyanote_diarization import RealtimeSpeakerDiarization as PyannoteRealtime
from speechbrain_diarization import RealtimeSpeakerDiarization as SpeechBrainRealtime


class RealtimeSpeakerDiarization:
    """
    Fusion Speaker Diarization combining Pyannote and SpeechBrain embeddings with session management
    
    Supports multiple fusion strategies:
    - concatenate: E = [E1 ; E2]
    - normalized_average: E = (normalize(E1) + normalize(E2)) / 2
    - weighted_average: E = α*normalize(E1) + (1-α)*normalize(E2)
    - score_level: final_score = α*s1 + (1-α)*s2 (compares similarities separately)
    - product: E = normalize(E1) ⊙ normalize(E2) (element-wise product)
    - max_pool: E = max(normalize(E1), normalize(E2))
    - learned_concat: E = [w1*E1 ; w2*E2] with learnable weights
    
    Note: Pyannote and SpeechBrain have different embedding dimensions (typically 256 vs 192).
    The class handles dimension mismatch using the `dimension_alignment` parameter:
    - "min": Truncate to minimum dimension (default, preserves all info but discards extras)
    - "max" or "pad_zero": Pad with zeros to maximum dimension (keeps all dims but adds zeros)
    
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
    pipeline = RealtimeSpeakerDiarization(fusion_method="score_level")
    
    # Process conversation 1
    pipeline.set_session("meeting_1")
    output1 = pipeline("audio1.wav", use_memory=True, session_id="meeting_1")
    
    # Process conversation 2 (independent speakers)
    pipeline.set_session("meeting_2")
    output2 = pipeline("audio2.wav", use_memory=True, session_id="meeting_2")
    
    # Check all sessions
    print(pipeline.list_sessions())
    print(pipeline.get_all_sessions_info())
    ```
    """
    
    def __init__(self, 
                 fusion_method="normalized_average",  # Fusion strategy
                 fusion_alpha=0.5,  # Weight for weighted/score fusion (0-1)
                 fusion_weights=None,  # Custom weights for learned_concat [w1, w2]
                 dimension_alignment="max",  # How to handle dimension mismatch: "min", "max", "pad_zero"
                 pyannote_config=None,  # Config dict for Pyannote
                 speechbrain_config=None,  # Config dict for SpeechBrain
                 similarity_threshold=0.7,
                 embedding_update_weight=0.3,
                 min_similarity_gap=0.3,
                 skip_update_short_audio=True,  # bật/tắt skip update cho audio ngắn
                 min_duration_for_update=2.0,  # duration tối thiểu (giây) để update embedding
                 init_similarity_threshold=0.4,  # threshold thấp hơn cho chunk thứ 2 sau init
                 use_pyannote=True,  # Enable/disable Pyannote
                 use_speechbrain=True,  # Enable/disable SpeechBrain
                 *args, **kwargs):
        """
        Initialize Fusion Speaker Diarization
        
        Parameters
        ----------
        fusion_method : str
            One of: "concatenate", "normalized_average", "weighted_average", 
                    "score_level", "product", "max_pool", "learned_concat"
        fusion_alpha : float
            Weight for weighted_average (Pyannote) or score_level fusion (0-1)
        fusion_weights : tuple[float, float], optional
            Custom weights for learned_concat: (w1, w2)
        dimension_alignment : str
            How to handle dimension mismatch: "min" (truncate), "max" (zero-pad), "pad_zero"
        pyannote_config : dict, optional
            Configuration for Pyannote (model_name, token, cache_dir, etc.)
        speechbrain_config : dict, optional
            Configuration for SpeechBrain (preloaded_model, etc.)
        use_pyannote : bool
            Whether to use Pyannote embeddings
        use_speechbrain : bool
            Whether to use SpeechBrain embeddings
        """
        self.fusion_method = fusion_method
        self.fusion_alpha = fusion_alpha
        self.fusion_weights = fusion_weights or (1.0, 1.0)
        self.dimension_alignment = dimension_alignment
        self.use_pyannote = use_pyannote
        self.use_speechbrain = use_speechbrain
        
        # Validate: at least one system must be enabled
        if not use_pyannote and not use_speechbrain:
            raise ValueError("At least one diarization system (Pyannote or SpeechBrain) must be enabled")
        
        # Initialize configurations
        pyannote_config = pyannote_config or {}
        speechbrain_config = speechbrain_config or {}
        
        # Shared parameters
        shared_params = {
            'similarity_threshold': similarity_threshold,
            'embedding_update_weight': embedding_update_weight,
            'min_similarity_gap': min_similarity_gap,
            'skip_update_short_audio': skip_update_short_audio,
            'min_duration_for_update': min_duration_for_update,
            'init_similarity_threshold': init_similarity_threshold
        }
        
        # Initialize both diarization systems
        self.pyannote_diarizer = None
        self.speechbrain_diarizer = None
        
        if use_pyannote:
            try:
                self.pyannote_diarizer = PyannoteRealtime(
                    **{**pyannote_config, **shared_params}
                )
                if torch.cuda.is_available():
                    self.pyannote_diarizer.to(torch.device("cuda"))
                self.pyannote_diarizer.set_session("default")
                print("✅ Pyannote diarization initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize Pyannote: {e}")
                self.use_pyannote = False
        
        if use_speechbrain:
            try:
                self.speechbrain_diarizer = SpeechBrainRealtime(
                    **{**speechbrain_config, **shared_params}
                )
                self.speechbrain_diarizer.set_session("default")
                print("✅ SpeechBrain diarization initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize SpeechBrain: {e}")
                self.use_speechbrain = False
        
        # Session management
        self.current_session_id: Optional[str] = None
        self.sessions: Dict[str, Dict] = {}  # {session_id: session_data}
        
        # Parameters
        self.similarity_threshold = similarity_threshold
        self.embedding_update_weight = embedding_update_weight
        self.min_similarity_gap = min_similarity_gap
        self.max_cluster_size = 20
        self.embedding_metric = "cosine"
        self.skip_update_short_audio = skip_update_short_audio  # Skip update nếu audio quá ngắn
        self.min_duration_for_update = min_duration_for_update  # Duration tối thiểu để update
        self.init_similarity_threshold = init_similarity_threshold  # Threshold cho chunk thứ 2 sau init
    
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
                'speaker_memory': {},  # Fused embeddings
                'speaker_embedding_clusters': {},
                'speaker_counts': {},
                'speaker_history': [],
                'total_chunks_processed': 0,
                'next_speaker_id': 0,
                'pyannote_memory': {},  # Pyannote EMA embeddings
                'speechbrain_memory': {},  # SpeechBrain EMA embeddings
                'pyannote_embeddings_cache': {},
                'speechbrain_embeddings_cache': {},
                'created_at': None,
                'last_updated': None
            }
            print(f"📝 Created new fusion session: {session_id}")
        
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
        
        # Set session for underlying diarizers
        if self.pyannote_diarizer:
            self.pyannote_diarizer.set_session(session_id)
        if self.speechbrain_diarizer:
            self.speechbrain_diarizer.set_session(session_id)
        
        print(f"✅ Switched to fusion session: {session_id}")
    
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
            print(f"🗑️  Deleted fusion session: {session_id}")
            
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
                'pyannote_memory': {},
                'speechbrain_memory': {},
                'pyannote_embeddings_cache': {},
                'speechbrain_embeddings_cache': {},
                'created_at': self.sessions[session_id].get('created_at'),
                'last_updated': None
            }
            print(f"🔄 Reset fusion session: {session_id}")
            
            # Reset underlying diarizers
            if self.pyannote_diarizer:
                self.pyannote_diarizer.reset_session(session_id)
            if self.speechbrain_diarizer:
                self.speechbrain_diarizer.reset_session(session_id)
        else:
            print(f"⚠️  Session not found: {session_id}")
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def _align_dimensions(self, emb1: np.ndarray, emb2: np.ndarray) -> tuple:
        """
        Align two embeddings to have the same dimension
        
        Parameters
        ----------
        emb1, emb2 : np.ndarray
            Input embeddings (may have different dimensions)
            
        Returns
        -------
        aligned_emb1, aligned_emb2 : tuple[np.ndarray, np.ndarray]
            Aligned embeddings with same dimension
        """
        dim1, dim2 = len(emb1), len(emb2)
        
        if dim1 == dim2:
            return emb1, emb2
        
        if self.dimension_alignment == "min":
            # Truncate to minimum dimension
            min_dim = min(dim1, dim2)
            return emb1[:min_dim], emb2[:min_dim]
        
        elif self.dimension_alignment in ["max", "pad_zero"]:
            # Pad with zeros to maximum dimension
            max_dim = max(dim1, dim2)
            aligned_emb1 = np.zeros(max_dim)
            aligned_emb2 = np.zeros(max_dim)
            aligned_emb1[:dim1] = emb1
            aligned_emb2[:dim2] = emb2
            return aligned_emb1, aligned_emb2
        
        else:
            # Default: use minimum dimension
            min_dim = min(dim1, dim2)
            return emb1[:min_dim], emb2[:min_dim]
    
    def _cosine(self, a, b):
        """Compute cosine similarity with NaN handling"""
        # Check for zero vectors
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            print(f"⚠️  Warning: Zero vector detected! norm_a={norm_a:.6f}, norm_b={norm_b:.6f}")
            return 0.0  # Return 0 similarity for zero vectors
        
        similarity = 1 - cdist([a], [b], metric="cosine")[0, 0]
        
        # Check for NaN result
        if np.isnan(similarity):
            print(f"⚠️  Warning: NaN similarity detected! Returning 0.0")
            return 0.0
        
        return similarity
    
    def _euclidean(self, a, b):
        """Compute euclidean distance"""
        return cdist([a], [b], metric="euclidean")[0, 0]

    def _fuse_embeddings(self, 
                        pyannote_emb: Optional[np.ndarray], 
                        speechbrain_emb: Optional[np.ndarray]) -> np.ndarray:
        """
        Fuse embeddings from both systems based on fusion_method
        
        Parameters
        ----------
        pyannote_emb : np.ndarray or None
            Pyannote embedding
        speechbrain_emb : np.ndarray or None
            SpeechBrain embedding
            
        Returns
        -------
        fused_emb : np.ndarray
            Fused embedding
        """
        # Handle single-system cases
        if pyannote_emb is None and speechbrain_emb is None:
            raise ValueError("At least one embedding must be provided")
        if pyannote_emb is None or np.all(pyannote_emb == 0):
            return speechbrain_emb
        if speechbrain_emb is None or np.all(speechbrain_emb == 0):
            return pyannote_emb
        
        norm_pyannote = self._normalize_embedding(pyannote_emb)
        norm_speechbrain = self._normalize_embedding(speechbrain_emb)

        # Fusion strategies
        if self.fusion_method == "concatenate":
            # Simple concatenation: E = [E1 ; E2]
            fused = np.concatenate([pyannote_emb, speechbrain_emb])
            
        elif self.fusion_method == "normalized_average":
            # Average of normalized embeddings: E = (norm(E1) + norm(E2)) / 2
            # Handle dimension mismatch using alignment strategy
            aligned_pyannote, aligned_speechbrain = self._align_dimensions(norm_pyannote, norm_speechbrain)
            fused = (aligned_pyannote + aligned_speechbrain) / 2
            fused = self._normalize_embedding(fused)  # Re-normalize
            
        elif self.fusion_method == "weighted_average":
            # Weighted average: E = α*norm(E1) + (1-α)*norm(E2)
            # Handle dimension mismatch using alignment strategy
            aligned_pyannote, aligned_speechbrain = self._align_dimensions(norm_pyannote, norm_speechbrain)
            fused = self.fusion_alpha * aligned_pyannote + (1 - self.fusion_alpha) * aligned_speechbrain
            fused = self._normalize_embedding(fused)
            
        elif self.fusion_method == "product":
            # Element-wise product of normalized embeddings
            # Handle dimension mismatch using alignment strategy
            aligned_pyannote, aligned_speechbrain = self._align_dimensions(norm_pyannote, norm_speechbrain)
            fused = aligned_pyannote * aligned_speechbrain
            fused = self._normalize_embedding(fused)
            
        elif self.fusion_method == "max_pool":
            # Max pooling of normalized embeddings
            # Handle dimension mismatch using alignment strategy
            aligned_pyannote, aligned_speechbrain = self._align_dimensions(norm_pyannote, norm_speechbrain)
            fused = np.maximum(aligned_pyannote, aligned_speechbrain)
            fused = self._normalize_embedding(fused)
            
        elif self.fusion_method == "learned_concat":
            # Weighted concatenation: E = [w1*E1 ; w2*E2]
            w1, w2 = self.fusion_weights
            weighted_pyannote = w1 * pyannote_emb
            weighted_speechbrain = w2 * speechbrain_emb
            fused = np.concatenate([weighted_pyannote, weighted_speechbrain])
            fused = self._normalize_embedding(fused)
            
        elif self.fusion_method == "score_level":
            # Score-level fusion doesn't fuse embeddings
            # Returns concatenation as placeholder (not used for matching)
            # Actual score fusion happens in _fuse_score where individual embeddings are compared
            fused = np.concatenate([norm_pyannote, norm_speechbrain])
            fused = self._normalize_embedding(fused)
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused
    
    def _fuse_score(self, 
                    new_emb_pyannote: np.ndarray,
                    new_emb_speechbrain: np.ndarray,
                    memory_emb_pyannote: np.ndarray,
                    memory_emb_speechbrain: np.ndarray) -> float:
        """
        Compute similarity using score-level fusion
        
        Parameters
        ----------
        new_emb_pyannote : np.ndarray
            New Pyannote embedding
        new_emb_speechbrain : np.ndarray
            New SpeechBrain embedding
        memory_emb_pyannote : np.ndarray
            Memory Pyannote embedding
        memory_emb_speechbrain : np.ndarray
            Memory SpeechBrain embedding
            
        Returns
        -------
        fused_score : float
            Fused similarity score
        """
        # Compute individual similarities
        if self.embedding_metric == "cosine":
            sim_pyannote = self._cosine(new_emb_pyannote, memory_emb_pyannote)
            sim_speechbrain = self._cosine(new_emb_speechbrain, memory_emb_speechbrain)
        else:
            dist_pyannote = self._euclidean(new_emb_pyannote, memory_emb_pyannote)
            dist_speechbrain = self._euclidean(new_emb_speechbrain, memory_emb_speechbrain)
            sim_pyannote = 1 / (1 + dist_pyannote)
            sim_speechbrain = 1 / (1 + dist_speechbrain)
        
        # Weighted fusion of scores
        fused_score = self.fusion_alpha * sim_pyannote + (1 - self.fusion_alpha) * sim_speechbrain
        
        # Check for NaN in final result
        if np.isnan(fused_score):
            print(f"⚠️  Warning: NaN in fused_score! sim_pyannote={sim_pyannote:.3f}, sim_speechbrain={sim_speechbrain:.3f}")
            return 0.0
        
        return fused_score
    
    def _extract_embeddings(self, file_or_audio_f32: Union[str, np.ndarray], max_speakers: Optional[int] = None) -> Dict:
        """
        Extract embeddings from both systems
        
        Returns
        -------
        result : dict
            {
                'pyannote_embeddings': np.ndarray or None,
                'speechbrain_embeddings': np.ndarray or None,
                'pyannote_labels': List[str] or None,
                'speechbrain_labels': List[str] or None
            }
        """
        result = {
            'pyannote_embeddings': None,
            'speechbrain_embeddings': None,
            'pyannote_labels': None,
            'speechbrain_labels': None
        }
        
        # Extract from Pyannote
        pyannote_output = None
        if self.use_pyannote and self.pyannote_diarizer:
            try:
                if isinstance(file_or_audio_f32, np.ndarray):
                    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                        tmp_path = tmp_file.name
                        sf.write(tmp_path, file_or_audio_f32, 16000, format="WAV")
                        pyannote_output = self.pyannote_diarizer(
                            tmp_path,
                            max_speakers=max_speakers,
                            use_memory=False  # We manage memory ourselves
                        )
                else:
                    pyannote_output = self.pyannote_diarizer(
                        file_or_audio_f32,
                        max_speakers=max_speakers,
                        use_memory=False  # We manage memory ourselves
                    )
                if pyannote_output.speaker_embeddings is not None:
                    result['pyannote_embeddings'] = pyannote_output.speaker_embeddings
                    result['pyannote_labels'] = list(pyannote_output.speaker_diarization.labels())
                    print(f"[Pyannote] Extracted {len(result['pyannote_labels'])} speakers: {result['pyannote_labels']}")
            except Exception as e:
                print(f"⚠️  Pyannote extraction failed: {e}")
        
        # Extract from SpeechBrain
        speechbrain_output = None
        if self.use_speechbrain and self.speechbrain_diarizer:
            try:
                speechbrain_output = self.speechbrain_diarizer(
                    file_or_audio_f32,
                    max_speakers=max_speakers,
                    use_memory=False  # We manage memory ourselves
                )
                if speechbrain_output.get('speaker_embeddings') is not None:
                    result['speechbrain_embeddings'] = speechbrain_output['speaker_embeddings']
                    result['speechbrain_labels'] = speechbrain_output.get('speaker_labels', ['SPEAKER_00'])
                    print(f"[SpeechBrain] Extracted {len(result['speechbrain_labels'])} speakers: {result['speechbrain_labels']}")
            except Exception as e:
                print(f"⚠️  SpeechBrain extraction failed: {e}")
        
        return result, pyannote_output, speechbrain_output
    
    def _match_speakers_with_memory(self, 
                                   new_embeddings_pyannote: Optional[np.ndarray],
                                   new_embeddings_speechbrain: Optional[np.ndarray],
                                   new_labels: List[str],
                                   max_speakers: Optional[int] = None,
                                   session_id: Optional[str] = None,
                                   audio_duration: Optional[float] = None) -> Dict[str, str]:
        """
        Match speakers with memory using fused embeddings
        
        Similar to individual systems but uses fused embeddings for matching
        
        Parameters
        ----------
        new_embeddings_pyannote : np.ndarray
            New Pyannote embeddings
        new_embeddings_speechbrain : np.ndarray
            New SpeechBrain embeddings
        new_labels : List[str]
            New labels
        max_speakers : int, optional
            Maximum number of speakers to extract
        session_id : str, optional
            Session ID to use. If None, uses current session.
        audio_duration : float, optional
            Duration of the audio chunk in seconds. Used to determine if embedding should be updated.
        """
        session_data = self._get_session_data(session_id)
        speaker_memory = session_data['speaker_memory']
        speaker_embedding_clusters = session_data['speaker_embedding_clusters']
        speaker_counts = session_data['speaker_counts']
        pyannote_memory = session_data['pyannote_memory']
        speechbrain_memory = session_data['speechbrain_memory']
        
        # Kiểm tra xem có nên update embedding hay không dựa vào duration
        should_update_embedding = True
        if self.skip_update_short_audio and audio_duration is not None:
            if audio_duration < self.min_duration_for_update:
                should_update_embedding = False
                print(f"⏱️  Audio duration ({audio_duration:.2f}s) < {self.min_duration_for_update}s. "
                      f"Skipping embedding update (matching only).")
        
        # Kiểm tra xem có phải chunk thứ 2 sau init không (áp dụng threshold thấp hơn)
        is_second_chunk_after_init = (session_data['total_chunks_processed'] == 1)
        effective_threshold = self.init_similarity_threshold if is_second_chunk_after_init else self.similarity_threshold
        
        if is_second_chunk_after_init:
            print(f"🎯 Second chunk after init - using lower threshold: {effective_threshold:.2f} (normal: {self.similarity_threshold:.2f})")
        
        # Fuse new embeddings
        num_speakers = len(new_labels)
        fused_new_embeddings = []
        
        for i in range(num_speakers):
            pyannote_emb = new_embeddings_pyannote[i] if new_embeddings_pyannote is not None else None
            speechbrain_emb = new_embeddings_speechbrain[i] if new_embeddings_speechbrain is not None else None
            
            if pyannote_emb is None and speechbrain_emb is None:
                continue
            
            # Debug: Check for zero embeddings
            if pyannote_emb is not None:
                pyannote_norm = np.linalg.norm(pyannote_emb)
                if pyannote_norm == 0:
                    print(f"⚠️  Warning: Pyannote embedding {i} is zero vector!")
            
            if speechbrain_emb is not None:
                speechbrain_norm = np.linalg.norm(speechbrain_emb)
                if speechbrain_norm == 0:
                    print(f"⚠️  Warning: SpeechBrain embedding {i} is zero vector!")
                
            fused_emb = self._fuse_embeddings(pyannote_emb, speechbrain_emb)
            
            # Check fused embedding
            fused_norm = np.linalg.norm(fused_emb)
            if fused_norm == 0:
                print(f"⚠️  Warning: Fused embedding {i} is zero vector!")
            
            fused_new_embeddings.append(fused_emb)
        
        if len(fused_new_embeddings) == 0:
            return {}
        
        fused_new_embeddings = np.array(fused_new_embeddings)
        
        # First time: initialize memory
        if len(speaker_memory) == 0:
            mapping = {}
            for i, label in enumerate(new_labels):
                next_id = session_data['next_speaker_id']
                speaker_id = f"SPEAKER_{next_id:02d}"
                session_data['next_speaker_id'] += 1
                mapping[label] = speaker_id
                
                # Initialize memory with fused embedding
                speaker_memory[speaker_id] = fused_new_embeddings[i].copy()
                speaker_embedding_clusters[speaker_id] = [fused_new_embeddings[i].copy()]
                speaker_counts[speaker_id] = 1
                
                # Store individual embeddings for score-level fusion
                if new_embeddings_pyannote is not None and i < len(new_embeddings_pyannote) and not np.all(new_embeddings_pyannote[i] == 0):
                    pyannote_memory[speaker_id] = new_embeddings_pyannote[i].copy()
                if new_embeddings_speechbrain is not None and i < len(new_embeddings_speechbrain) and not np.all(new_embeddings_speechbrain[i] == 0):
                    speechbrain_memory[speaker_id] = new_embeddings_speechbrain[i].copy()
                
                print(f"[Fusion] Creating new speaker: {speaker_id}")
            
            return mapping
        
        # Match with memory
        memory_speaker_ids = list(speaker_memory.keys())
        
        # Debug: Check memory embeddings
        for speaker_id in memory_speaker_ids:
            mem_emb = speaker_memory[speaker_id]
            mem_norm = np.linalg.norm(mem_emb)
            if mem_norm == 0:
                print(f"⚠️  Warning: Memory embedding for {speaker_id} is zero vector!")
        
        # Compute similarities based on fusion method
        if self.fusion_method == "score_level":
            # Score-level fusion: compute similarities separately then combine
            print("[Fusion] Using score-level fusion")
            
            similarities_ema = np.zeros((num_speakers, len(memory_speaker_ids)))
            
            for i in range(num_speakers):
                pyannote_emb_new = new_embeddings_pyannote[i] if new_embeddings_pyannote is not None else None
                speechbrain_emb_new = new_embeddings_speechbrain[i] if new_embeddings_speechbrain is not None else None
                
                for j, speaker_id in enumerate(memory_speaker_ids):
                    pyannote_emb_mem = pyannote_memory.get(speaker_id)
                    speechbrain_emb_mem = speechbrain_memory.get(speaker_id)
                    
                    # Use the _fuse_score helper method for score-level fusion
                    if (pyannote_emb_new is not None and pyannote_emb_mem is not None and 
                        speechbrain_emb_new is not None and speechbrain_emb_mem is not None):
                        # Both embeddings available - use fuse_score
                        fused_sim = self._fuse_score(pyannote_emb_new, speechbrain_emb_new,
                                                     pyannote_emb_mem, speechbrain_emb_mem)
                    elif pyannote_emb_new is not None and pyannote_emb_mem is not None:
                        # Only Pyannote available
                        if self.embedding_metric == "cosine":
                            fused_sim = self._cosine(pyannote_emb_new, pyannote_emb_mem)
                        else:
                            dist = self._euclidean(pyannote_emb_new, pyannote_emb_mem)
                            fused_sim = 1 / (1 + dist)
                    elif speechbrain_emb_new is not None and speechbrain_emb_mem is not None:
                        # Only SpeechBrain available
                        if self.embedding_metric == "cosine":
                            fused_sim = self._cosine(speechbrain_emb_new, speechbrain_emb_mem)
                        else:
                            dist = self._euclidean(speechbrain_emb_new, speechbrain_emb_mem)
                            fused_sim = 1 / (1 + dist)
                    else:
                        fused_sim = 0.0
                    
                    # Additional NaN check
                    if np.isnan(fused_sim):
                        print(f"⚠️  NaN detected at speaker {i}, memory {speaker_id}. Setting to 0.0")
                        fused_sim = 0.0
                    
                    similarities_ema[i, j] = fused_sim
                    
        else:
            # Embedding-level fusion: use fused embeddings
            memory_embeddings = np.array([speaker_memory[sid] for sid in memory_speaker_ids])
            
            if self.embedding_metric == "cosine":
                distances = cdist(fused_new_embeddings, memory_embeddings, metric='cosine')
                similarities_ema = 1 - distances
            else:
                distances = cdist(fused_new_embeddings, memory_embeddings, metric='euclidean')
                similarities_ema = 1 / (1 + distances)
        
        mapping = {}
        used_memory_speakers = set()
        
        # Greedy matching
        for i, label in enumerate(new_labels):
            new_embedding = fused_new_embeddings[i]
            
            # Find best match
            best_match_idx = np.argmax(similarities_ema[i])
            best_similarity_ema = similarities_ema[i, best_match_idx]
            best_speaker_id = memory_speaker_ids[best_match_idx]
            
            # Find second best for gap analysis
            sorted_indices = np.argsort(similarities_ema[i])[::-1]
            second_best_similarity_ema = similarities_ema[i, sorted_indices[1]] if len(sorted_indices) > 1 else -1
            similarity_gap_ema = best_similarity_ema - second_best_similarity_ema
            
            print(f"\n[Fusion] Label: {label}")
            print(f"  Best similarity: {best_similarity_ema:.3f} with {best_speaker_id}")
            print(f"  Gap: {similarity_gap_ema:.3f}")
            print(f"  Threshold: {effective_threshold}")
            
            matched_speaker_id = None
            
            # Check matching conditions
            if best_speaker_id not in used_memory_speakers:
                if best_similarity_ema >= effective_threshold:
                    matched_speaker_id = best_speaker_id
                    print(f"  ✅ Matched via threshold!")
                elif similarity_gap_ema > self.min_similarity_gap and second_best_similarity_ema > 0:
                    matched_speaker_id = best_speaker_id
                    print(f"  ✅ Matched via gap!")
            
            # Handle matching result
            if matched_speaker_id is not None:
                mapping[label] = matched_speaker_id
                used_memory_speakers.add(matched_speaker_id)
                
                # Update EMA embedding (chỉ khi should_update_embedding = True)
                if should_update_embedding:
                    old_embedding = speaker_memory[matched_speaker_id]
                    updated_embedding = (1 - self.embedding_update_weight) * old_embedding + \
                                       self.embedding_update_weight * new_embedding
                    if self.embedding_metric == "cosine":
                        updated_embedding = self._normalize_embedding(updated_embedding)
                    speaker_memory[matched_speaker_id] = updated_embedding
                    
                    # For score-level fusion, also update individual embeddings
                    if self.fusion_method == "score_level":
                        # Update Pyannote embedding
                        if new_embeddings_pyannote is not None and i < len(new_embeddings_pyannote):
                            pyannote_emb_new = new_embeddings_pyannote[i]
                            if matched_speaker_id in pyannote_memory:
                                old_pyannote = pyannote_memory[matched_speaker_id]
                                updated_pyannote = (1 - self.embedding_update_weight) * old_pyannote + \
                                                 self.embedding_update_weight * pyannote_emb_new
                                if self.embedding_metric == "cosine":
                                    updated_pyannote = self._normalize_embedding(updated_pyannote)
                                pyannote_memory[matched_speaker_id] = updated_pyannote
                            else:
                                pyannote_memory[matched_speaker_id] = pyannote_emb_new.copy()
                        
                        # Update SpeechBrain embedding
                        if new_embeddings_speechbrain is not None and i < len(new_embeddings_speechbrain):
                            speechbrain_emb_new = new_embeddings_speechbrain[i]
                            if matched_speaker_id in speechbrain_memory:
                                old_speechbrain = speechbrain_memory[matched_speaker_id]
                                updated_speechbrain = (1 - self.embedding_update_weight) * old_speechbrain + \
                                                    self.embedding_update_weight * speechbrain_emb_new
                                if self.embedding_metric == "cosine":
                                    updated_speechbrain = self._normalize_embedding(updated_speechbrain)
                                speechbrain_memory[matched_speaker_id] = updated_speechbrain
                            else:
                                speechbrain_memory[matched_speaker_id] = speechbrain_emb_new.copy()
                    
                    # Add to cluster
                    cluster = speaker_embedding_clusters[matched_speaker_id]
                    cluster.append(new_embedding.copy())
                    if len(cluster) > self.max_cluster_size:
                        speaker_embedding_clusters[matched_speaker_id] = cluster[-self.max_cluster_size:]
                    
                    speaker_counts[matched_speaker_id] += 1
                    print(f"  📊 Updated speaker {matched_speaker_id}")
                else:
                    # Chỉ count mà không update embedding
                    speaker_counts[matched_speaker_id] += 1
                    print(f"  📊 Matched but skipped update (short audio)")
                
            else:
                # Check max_speakers constraint
                current_num_speakers = len(speaker_memory)
                
                if max_speakers is not None and current_num_speakers >= max_speakers:
                    # Force assign to best match
                    print(f"  ⚠️  Max speakers ({max_speakers}) reached! Force-assigning...")
                    
                    best_overall_idx = np.argmax(similarities_ema[i])
                    best_overall_speaker_id = memory_speaker_ids[best_overall_idx]
                    
                    for speaker_idx, speaker_id in enumerate(memory_speaker_ids):
                        if speaker_id not in used_memory_speakers:
                            best_overall_speaker_id = speaker_id
                            break
                    
                    matched_speaker_id = best_overall_speaker_id
                    mapping[label] = matched_speaker_id
                    used_memory_speakers.add(matched_speaker_id)
                    
                    print(f"  🔀 Force-assigned to {matched_speaker_id}")
                    
                    # Update as normal (chỉ khi should_update_embedding = True)
                    if should_update_embedding:
                        old_embedding = speaker_memory[matched_speaker_id]
                        updated_embedding = (1 - self.embedding_update_weight) * old_embedding + \
                                           self.embedding_update_weight * new_embedding
                        if self.embedding_metric == "cosine":
                            updated_embedding = self._normalize_embedding(updated_embedding)
                        speaker_memory[matched_speaker_id] = updated_embedding
                        
                        # For score-level fusion, also update individual embeddings
                        if self.fusion_method == "score_level":
                            if new_embeddings_pyannote is not None and i < len(new_embeddings_pyannote):
                                pyannote_emb_new = new_embeddings_pyannote[i]
                                if matched_speaker_id in pyannote_memory:
                                    old_pyannote = pyannote_memory[matched_speaker_id]
                                    updated_pyannote = (1 - self.embedding_update_weight) * old_pyannote + \
                                                     self.embedding_update_weight * pyannote_emb_new
                                    if self.embedding_metric == "cosine":
                                        updated_pyannote = self._normalize_embedding(updated_pyannote)
                                    pyannote_memory[matched_speaker_id] = updated_pyannote
                                else:
                                    pyannote_memory[matched_speaker_id] = pyannote_emb_new.copy()
                            
                            if new_embeddings_speechbrain is not None and i < len(new_embeddings_speechbrain):
                                speechbrain_emb_new = new_embeddings_speechbrain[i]
                                if matched_speaker_id in speechbrain_memory:
                                    old_speechbrain = speechbrain_memory[matched_speaker_id]
                                    updated_speechbrain = (1 - self.embedding_update_weight) * old_speechbrain + \
                                                        self.embedding_update_weight * speechbrain_emb_new
                                    if self.embedding_metric == "cosine":
                                        updated_speechbrain = self._normalize_embedding(updated_speechbrain)
                                    speechbrain_memory[matched_speaker_id] = updated_speechbrain
                                else:
                                    speechbrain_memory[matched_speaker_id] = speechbrain_emb_new.copy()
                        
                        cluster = speaker_embedding_clusters[matched_speaker_id]
                        cluster.append(new_embedding.copy())
                        if len(cluster) > self.max_cluster_size:
                            speaker_embedding_clusters[matched_speaker_id] = cluster[-self.max_cluster_size:]
                        
                        speaker_counts[matched_speaker_id] += 1
                    else:
                        # Chỉ count mà không update embedding
                        speaker_counts[matched_speaker_id] += 1
                        print(f"  📊 Force-assigned but skipped update (short audio)")
                else:
                    # Create new speaker
                    next_id = session_data['next_speaker_id']
                    speaker_id = f"SPEAKER_{next_id:02d}"
                    session_data['next_speaker_id'] += 1
                    mapping[label] = speaker_id
                    
                    speaker_memory[speaker_id] = new_embedding.copy()
                    speaker_embedding_clusters[speaker_id] = [new_embedding.copy()]
                    speaker_counts[speaker_id] = 1
                    
                    # Store individual embeddings for score-level fusion
                    if new_embeddings_pyannote is not None and i < len(new_embeddings_pyannote):
                        pyannote_memory[speaker_id] = new_embeddings_pyannote[i].copy()
                    if new_embeddings_speechbrain is not None and i < len(new_embeddings_speechbrain):
                        speechbrain_memory[speaker_id] = new_embeddings_speechbrain[i].copy()
                    
                    print(f"  🆕 Created new speaker: {speaker_id}")
        
        return mapping
    
    def apply_realtime(self, 
                      file_or_audio_f32: Union[str, np.ndarray],
                      use_memory: bool = True,
                      max_speakers: Optional[int] = None,
                      session_id: Optional[str] = None,
                      **kwargs) -> dict:
        """
        Apply fusion diarization with context memory
        
        Parameters
        ----------
        file_or_audio_f32 : Union[str, np.ndarray]
            Audio chunk or file path
        use_memory : bool
            Whether to use speaker memory
        max_speakers : int, optional
            Maximum number of speakers
        session_id : str, optional
            Session ID to use. If None, uses current session. Required if use_memory=True.
            
        Returns
        -------
        output : dict
            Diarization results with fused embeddings
        """
        # Set session if provided
        if use_memory and session_id is not None:
            self.set_session(session_id)
        elif use_memory and self.current_session_id is None:
            raise ValueError("No session_id provided and no current session set. "
                           "Call set_session() or provide session_id parameter.")
        
        # Tính audio duration
        audio_duration = None
        try:
            if isinstance(file_or_audio_f32, dict):
                audio_info = sf.info(file_or_audio_f32['audio'])
                audio_duration = audio_info.duration
            elif isinstance(file_or_audio_f32, str):
                audio_info = sf.info(file_or_audio_f32)
                audio_duration = audio_info.duration
            elif isinstance(file_or_audio_f32, np.ndarray):
                audio_duration = len(file_or_audio_f32) / 16000  # Assuming 16kHz sample rate
            
            if audio_duration is not None:
                print(f"duration: {audio_duration:.2f}s")
            else:
                print(f"⚠️  Could not calculate audio duration: {e}; file_or_audio_f32: {file_or_audio_f32}")
        except Exception as e:
            print(f"⚠️  Could not calculate audio duration: {e}; file_or_audio_f32: {file_or_audio_f32}")
        
        # Extract embeddings from both systems
        extracted, pyannote_output, speechbrain_output = self._extract_embeddings(file_or_audio_f32, max_speakers=max_speakers)
        
        pyannote_embeddings = extracted['pyannote_embeddings']
        speechbrain_embeddings = extracted['speechbrain_embeddings']
        pyannote_labels = extracted['pyannote_labels']
        speechbrain_labels = extracted['speechbrain_labels']
        
        # Check if all pyannote labels are SPEAKER_UNK (invalid embeddings)
        pyannote_all_unknown = False
        if pyannote_labels:
            pyannote_all_unknown = all(label == "SPEAKER_UNK" for label in pyannote_labels)
            if pyannote_all_unknown:
                print("⚠️  All Pyannote speakers are SPEAKER_UNK (invalid embeddings)")
                print("✅ Using SpeechBrain results only (skipping fusion)")
        
        # If pyannote has all SPEAKER_UNK, use only SpeechBrain
        if pyannote_all_unknown and speechbrain_labels and self.use_speechbrain and self.speechbrain_diarizer:
            print("[Fusion] Bypassing fusion - using SpeechBrain only")
            
            # Return speechbrain output in fusion format
            return {
                'speaker_embeddings': speechbrain_output.get('speaker_embeddings'),
                'speaker_labels': speechbrain_output.get('speaker_labels', []),
                'pyannote_embeddings': None,
                'speechbrain_embeddings': speechbrain_output.get('speaker_embeddings')
            }
        
        # Determine labels (prefer the system with more speakers or use a consensus)
        if pyannote_labels and speechbrain_labels:
            # Use labels from system with more detected speakers
            if len(pyannote_labels) >= len(speechbrain_labels):
                current_labels = pyannote_labels
            else:
                current_labels = speechbrain_labels
        elif pyannote_labels:
            current_labels = pyannote_labels
        elif speechbrain_labels:
            current_labels = speechbrain_labels
        else:
            # No speakers detected
            return {
                'speaker_embeddings': None,
                'speaker_labels': [],
                'pyannote_embeddings': None,
                'speechbrain_embeddings': None
            }
        
        # Prepare embeddings for fusion (align with labels)
        # For simplicity, we assume both systems detect the same number of speakers
        # In practice, you might need more sophisticated alignment
        
        num_speakers = len(current_labels)
        
        # Expand embeddings if needed
        if pyannote_embeddings is not None and len(pyannote_embeddings) < num_speakers:
            # Add padding to pyannote embeddings
            pyannote_embeddings = np.pad(pyannote_embeddings, ((0, num_speakers - len(pyannote_embeddings)), (0, 0)), mode='constant')
        if speechbrain_embeddings is not None and len(speechbrain_embeddings) < num_speakers:
            # Add padding to speechbrain embeddings
            speechbrain_embeddings = np.pad(speechbrain_embeddings, ((0, num_speakers - len(speechbrain_embeddings)), (0, 0)), mode='constant')
        
        output = {
            'pyannote_embeddings': pyannote_embeddings,
            'speechbrain_embeddings': speechbrain_embeddings,
            'speaker_labels': current_labels
        }
        
        if not use_memory:
            # Just return fused embeddings without memory matching
            fused_embeddings = []
            for i in range(num_speakers):
                pyannote_emb = pyannote_embeddings[i] if pyannote_embeddings is not None else None
                speechbrain_emb = speechbrain_embeddings[i] if speechbrain_embeddings is not None else None
                fused_emb = self._fuse_embeddings(pyannote_emb, speechbrain_emb)
                fused_embeddings.append(fused_emb)
            
            output['speaker_embeddings'] = np.array(fused_embeddings)
            return output
        
        # Get session data
        session_data = self._get_session_data()
        
        # Match with memory
        label_mapping = self._match_speakers_with_memory(
            pyannote_embeddings,
            speechbrain_embeddings,
            current_labels,
            max_speakers=max_speakers,
            audio_duration=audio_duration
        )
        
        print(f"[Fusion] Label mapping: {label_mapping}")
        
        # Update labels and embeddings
        new_labels_ordered = [label_mapping.get(label, label) for label in current_labels]
        updated_embeddings = np.array([
            session_data['speaker_memory'][label] for label in new_labels_ordered
        ])
        
        output['speaker_labels'] = new_labels_ordered
        output['speaker_embeddings'] = updated_embeddings
        
        # Save to history
        session_data['speaker_history'].append({
            'chunk_id': session_data['total_chunks_processed'],
            'labels': new_labels_ordered,
            'num_speakers': len(new_labels_ordered)
        })
        session_data['total_chunks_processed'] += 1
        
        return output
    
    def get_speaker_info(self, session_id: Optional[str] = None) -> Dict:
        """
        Get information about known speakers in session
        
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
                'num_speakers': 0,
                'fusion_method': self.fusion_method,
                'fusion_alpha': self.fusion_alpha
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
            'num_speakers': len(session_data['speaker_memory']),
            'fusion_method': self.fusion_method,
            'fusion_alpha': self.fusion_alpha
        }
    
    def get_all_sessions_info(self) -> Dict[str, Dict]:
        """
        Get information about all sessions
        
        Returns
        -------
        info : Dict[str, Dict]
            Dictionary mapping session_id to session info
        """
        return {
            session_id: self.get_speaker_info(session_id)
            for session_id in self.sessions.keys()
        }
    
    def __call__(self, file_or_audio_f32: Union[str, np.ndarray], 
                 num_speakers=None, min_speakers=None, max_speakers=None,
                 use_memory=True, session_id=None, **kwargs):
        """Call interface for diarization"""
        effective_max_speakers = num_speakers if num_speakers is not None else max_speakers
        
        return self.apply_realtime(
            file_or_audio_f32,
            use_memory=use_memory,
            max_speakers=effective_max_speakers,
            session_id=session_id,
            **kwargs
        )

# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    # Khởi tạo realtime pipeline
    pipeline = RealtimeSpeakerDiarization(
        fusion_method="score_level",
        fusion_alpha=0.4, # trọng số fusion (0-1), final score = fusion_alpha * score_pyannote + (1 - fusion_alpha) * score_speechbrain
        fusion_weights=[0.5, 0.5], # trọng số cho learned_concat, fusion_embedding = w1 * pyannote_embedding + w2 * speechbrain_embedding
        similarity_threshold=0.7,  # threshold để match speaker (càng cao càng strict)
        embedding_update_weight=0.3,  # trọng số update embedding (0.3 = 30% mới, 70% cũ)
        min_similarity_gap=0.3,  # gap tối thiểu để match nếu nổi bật hơn hẳn
        skip_update_short_audio=True,  # bật tính năng skip update cho audio ngắn
        min_duration_for_update=2.0,  # chỉ update embedding nếu audio >= 2s
        init_similarity_threshold=0.4  # threshold thấp hơn cho chunk thứ 2 sau init
    )

    # Ví dụ 1: Xử lý audio chunk đầu tiên với Session ID
    print("=" * 100)
    print("VÍ DỤ 1: Xử lý audio chunk đầu tiên với Session ID")
    print("=" * 100)
    
    # Set session for this conversation
    session_1 = "conversation_1"
    pipeline.set_session(session_1)

    output1 = pipeline(
        "/home/hoang/speaker_diarization/wav/A1.wav",
        num_speakers=2,
        use_memory=True,
        session_id=session_1
    )

    print("\n📊 Kết quả chunk 1:")
    print(f"  🎤 {output1['speaker_labels']}")

    print(f"\n💾 Speaker Memory (Session {session_1}): {pipeline.get_speaker_info(session_1)}")
    print(f"\n📋 All Sessions: {pipeline.list_sessions()}")

    # Ví dụ 2: Xử lý một conversation khác với session_id khác
    print("\n" + "=" * 100)
    print("VÍ DỤ 2: Xử lý conversation thứ 2 với session khác")
    print("=" * 100)
    
    session_2 = "conversation_2"
    pipeline.set_session(session_2)

    output2 = pipeline(
        "/home/hoang/speaker_diarization/wav/A2.wav",
        num_speakers=2,
        use_memory=True,
        session_id=session_2
    )

    print("\n📊 Kết quả chunk 1 (Session 2):")
    print(f"  🎤 {output2['speaker_labels']}")

    print(f"\n💾 Speaker Memory (Session {session_2}): {pipeline.get_speaker_info(session_2)}")
    print(f"💾 Speaker Memory (Session {session_1}): {pipeline.get_speaker_info(session_1)}")
    print(f"\n📋 All Sessions: {pipeline.list_sessions()}")

    # Ví dụ 3: Quay lại session 1 và xử lý chunk tiếp theo
    print("\n" + "=" * 100)
    print("VÍ DỤ 3: Quay lại Session 1 và xử lý chunk tiếp theo")
    print("=" * 100)
    
    pipeline.set_session(session_1)  # Chuyển về session 1

    output3 = pipeline(
        "/home/hoang/speaker_diarization/wav/B1.wav",
        num_speakers=2,
        use_memory=True,
        session_id=session_1
    )

    print("\n📊 Kết quả chunk 2 (Session 1 continued):")
    print(f"  🎤 {output3['speaker_labels']}")

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
    
    # Xem thông tin tất cả sessions
    print(f"\n📊 All sessions info:")
    for session_id, info in pipeline.get_all_sessions_info().items():
        print(f"  Session '{session_id}': {info['num_speakers']} speakers, {info['total_chunks']} chunks")

    print(f"\n💾 Speaker Memory (updated): {pipeline.get_speaker_info()}")
    print("=" * 100)
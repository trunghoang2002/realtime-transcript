"""
Fusion Speaker Diarization: Combines multiple embedding models (Pyannote, SpeechBrain, NeMo)
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Callable
from scipy.spatial.distance import cdist
import soundfile as sf
import tempfile

# Import all diarization systems
from pyanote_diarization import RealtimeSpeakerDiarization as PyannoteRealtime
from speechbrain_diarization import RealtimeSpeakerDiarization as SpeechBrainRealtime
from nemo_diarization import RealtimeSpeakerDiarization as NemoRealtime


class RealtimeSpeakerDiarization:
    """
    Fusion Speaker Diarization combining multiple embedding models (Pyannote, SpeechBrain, NeMo)
    
    Supports multiple fusion strategies for 2 or more models:
    - concatenate: E = [E1 ; E2 ; ...]
    - normalized_average: E = (normalize(E1) + normalize(E2) + ...) / n
    - weighted_average: E = α*normalize(E1) + β*normalize(E2) + ... (with α + β + ... = 1)
    - score_level: final_score = α*s1 + β*s2 + ... (with α + β + ... = 1)
    - product: E = normalize(E1) ⊙ normalize(E2) ⊙ ... (element-wise product)
    - max_pool: E = max(normalize(E1), normalize(E2), ...)
    - learned_concat: E = [w1*E1 ; w2*E2 ; ...] with learnable weights
    
    Note: Different models have different embedding dimensions.
    The class handles dimension mismatch using the `dimension_alignment` parameter:
    - "min": Truncate to minimum dimension (default, preserves all info but discards extras)
    - "max" or "pad_zero": Pad with zeros to maximum dimension (keeps all dims but adds zeros)
    
    Multiple Instances of Same Model Type:
    -------------------------------------
    - You can use multiple instances of the same model type with different configs
    - Use tuples to specify: (model_type, model_id)
    - model_type: 'speechbrain', 'pyannote', or 'nemo'
    - model_id: unique identifier for this instance
    - Example:
        models=[
            ('nemo', 'nemo_tianet'),
            ('nemo', 'nemo_ecapa_tdnn'),
            ('pyannote', 'pyan_community'),
            ('speechbrain', 'sb_default')
        ]
        model_configs={
            'nemo_tianet': {'pretrained_speaker_model': 'titanet_large'},
            'nemo_ecapa_tdnn': {'pretrained_speaker_model': 'ecapa_tdnn'},
            'pyan_community': {'model_name': 'pyannote/speaker-diarization-community-1'},
            'sb_default': {}
        }
    
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
    # Example 1: Different model types
    pipeline = RealtimeSpeakerDiarization(
        models=['speechbrain', 'pyannote', 'nemo'],
        fusion_method="score_level",
        fusion_weights=[0.3, 0.4, 0.3]  # Must sum to 1 for weighted/score methods
    )
    
    # Example 2: Multiple instances of same model type
    pipeline = RealtimeSpeakerDiarization(
        models=[
            ('nemo', 'nemo_tianet'),
            ('nemo', 'nemo_ecapa_tdnn'),
            ('pyannote', 'pyan_community'),
            ('speechbrain', 'sb_default')
        ],
        fusion_method="score_level",
        fusion_weights=[0.3, 0.4, 0.3],
        model_configs={
            'nemo_tianet': {'pretrained_speaker_model': 'titanet_large'},
            'nemo_ecapa_tdnn': {'pretrained_speaker_model': 'ecapa_tdnn'},
            'pyan_community': {'model_name': 'pyannote/speaker-diarization-community-1'},
            'sb_default': {}
        }
    )
    
    # Process conversation
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
                 models=['speechbrain', 'pyannote'],  # List of models to use
                 fusion_method="normalized_average",  # Fusion strategy
                 fusion_weights=None,  # Weights for weighted_average/score_level/learned_concat [w1, w2, ...]
                 dimension_alignment="max",  # How to handle dimension mismatch: "min", "max", "pad_zero"
                 model_configs=None,  # Dict of config dicts for each model {model_name/model_id: config}
                 similarity_threshold=0.7,
                 embedding_update_weight=0.3,
                 min_similarity_gap=0.3,
                 skip_update_short_audio=True,  # bật/tắt skip update cho audio ngắn
                 min_duration_for_update=2.0,  # duration tối thiểu (giây) để update embedding
                 init_similarity_threshold=0.4,  # threshold thấp hơn cho chunk thứ 2 sau init
                 *args, **kwargs):
        """
        Initialize Fusion Speaker Diarization with multiple models
        
        Parameters
        ----------
        models : list of str or list of tuple
            List of models to use. Can be:
            - List of strings: ['speechbrain', 'pyannote', 'nemo']
            - List of tuples: [('pyannote', 'id1'), ('pyannote', 'id2'), ('speechbrain', 'id3')]
              where each tuple is (model_type, model_id)
        fusion_method : str
            One of: "concatenate", "normalized_average", "weighted_average", 
                    "score_level", "product", "max_pool", "learned_concat"
        fusion_weights : list of float, optional
            Weights for weighted_average/score_level/learned_concat: [w1, w2, ...]
            Must sum to 1.0 for weighted_average and score_level
        dimension_alignment : str
            How to handle dimension mismatch: "min" (truncate), "max" (zero-pad), "pad_zero"
        model_configs : dict, optional
            Configuration for each model {model_id: config_dict}
        """
        # Validate models list
        if not models or len(models) < 2:
            raise ValueError("At least 2 models must be specified for fusion")
        
        # Parse models list - support both string and tuple formats
        valid_model_types = ['speechbrain', 'pyannote', 'nemo']
        self.model_instances = []  # List of (model_type, model_id)
        
        for i, model in enumerate(models):
            if isinstance(model, tuple):
                # Tuple format: (model_type, model_id)
                if len(model) != 2:
                    raise ValueError(f"Model tuple must have exactly 2 elements (model_type, model_id), got {model}")
                model_type, model_id = model
                if model_type not in valid_model_types:
                    raise ValueError(f"Invalid model_type '{model_type}'. Valid types: {valid_model_types}")
                self.model_instances.append((model_type, model_id))
            elif isinstance(model, str):
                # String format: backward compatible
                if model not in valid_model_types:
                    raise ValueError(f"Invalid model '{model}'. Valid models: {valid_model_types}")
                # Use model name as both type and id
                self.model_instances.append((model, model))
            else:
                raise ValueError(f"Model must be string or tuple, got {type(model)}")
        
        # Check for duplicate model_ids
        model_ids = [mid for _, mid in self.model_instances]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError(f"Duplicate model_ids found: {model_ids}. Each model instance must have a unique ID.")
        
        self.num_models = len(self.model_instances)
        self.fusion_method = fusion_method
        self.dimension_alignment = dimension_alignment
        
        # Handle fusion weights
        if fusion_weights is None:
            # Equal weights by default
            self.fusion_weights = [1.0 / self.num_models] * self.num_models
        else:
            if len(fusion_weights) != self.num_models:
                raise ValueError(f"fusion_weights length ({len(fusion_weights)}) must match number of models ({self.num_models})")
            self.fusion_weights = fusion_weights
        
        # Validate weights for methods that require sum=1
        if fusion_method in ['weighted_average', 'score_level']:
            weight_sum = sum(self.fusion_weights)
            if not np.isclose(weight_sum, 1.0, atol=1e-6):
                raise ValueError(f"fusion_weights must sum to 1.0 for {fusion_method}, got {weight_sum}")
        
        # Initialize configurations
        model_configs = model_configs or {}
        
        # Shared parameters for all models
        shared_params = {
            'similarity_threshold': similarity_threshold,
            'embedding_update_weight': embedding_update_weight,
            'min_similarity_gap': min_similarity_gap,
            'skip_update_short_audio': skip_update_short_audio,
            'min_duration_for_update': min_duration_for_update,
            'init_similarity_threshold': init_similarity_threshold
        }
        
        # Initialize all diarization systems
        self.diarizers = {}  # {model_id: diarizer_instance}
        
        for model_type, model_id in self.model_instances:
            config = model_configs.get(model_id, {})
            try:
                if model_type == 'pyannote':
                    diarizer = PyannoteRealtime(**{**config, **shared_params})
                    if torch.cuda.is_available():
                        diarizer.to(torch.device("cuda"))
                    diarizer.set_session("default")
                    self.diarizers[model_id] = diarizer
                    print(f"✅ Pyannote diarization initialized (id: {model_id})")
                    
                elif model_type == 'speechbrain':
                    diarizer = SpeechBrainRealtime(**{**config, **shared_params})
                    diarizer.set_session("default")
                    self.diarizers[model_id] = diarizer
                    print(f"✅ SpeechBrain diarization initialized (id: {model_id})")
                    
                elif model_type == 'nemo':
                    diarizer = NemoRealtime(**{**config, **shared_params})
                    diarizer.set_session("default")
                    self.diarizers[model_id] = diarizer
                    print(f"✅ NeMo diarization initialized (id: {model_id})")
                    
            except Exception as e:
                print(f"⚠️  Failed to initialize {model_type} (id: {model_id}): {e}")
                raise  # Raise error since all specified models must be initialized
        
        # Validate at least one model was successfully initialized
        if not self.diarizers:
            raise ValueError("No diarization models were successfully initialized")
        
        model_display = [f"{mt}({mid})" if mt != mid else mt for mt, mid in self.model_instances]
        print(f"🎯 Fusion with {self.num_models} models: {model_display}")
        print(f"   Method: {fusion_method}")
        print(f"   Weights: {self.fusion_weights}")
        
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
            session_data = {
                'speaker_memory': {},  # Fused embeddings
                'speaker_embedding_clusters': {},
                'speaker_counts': {},
                'speaker_history': [],
                'total_chunks_processed': 0,
                'next_speaker_id': 0,
                'created_at': None,
                'last_updated': None
            }
            
            # Add memory for each model instance
            for model_type, model_id in self.model_instances:
                session_data[f'{model_id}_memory'] = {}  # EMA embeddings per model
                session_data[f'{model_id}_embeddings_cache'] = {}
            
            self.sessions[session_id] = session_data
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
        
        # Set session for all underlying diarizers
        for model_id, diarizer in self.diarizers.items():
            diarizer.set_session(session_id)
        
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
            session_data = {
                'speaker_memory': {},
                'speaker_embedding_clusters': {},
                'speaker_counts': {},
                'speaker_history': [],
                'total_chunks_processed': 0,
                'next_speaker_id': 0,
                'created_at': self.sessions[session_id].get('created_at'),
                'last_updated': None
            }
            
            # Add memory for each model instance
            for model_type, model_id in self.model_instances:
                session_data[f'{model_id}_memory'] = {}
                session_data[f'{model_id}_embeddings_cache'] = {}
            
            self.sessions[session_id] = session_data
            print(f"🔄 Reset fusion session: {session_id}")
            
            # Reset all underlying diarizers
            for model_id, diarizer in self.diarizers.items():
                diarizer.reset_session(session_id)
        else:
            print(f"⚠️  Session not found: {session_id}")
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def _align_dimensions(self, *embeddings: np.ndarray) -> List[np.ndarray]:
        """
        Align multiple embeddings to have the same dimension
        
        Parameters
        ----------
        *embeddings : np.ndarray
            Input embeddings (may have different dimensions)
            
        Returns
        -------
        aligned_embeddings : List[np.ndarray]
            Aligned embeddings with same dimension
        """
        if len(embeddings) == 0:
            return []
        
        # Get all dimensions
        dimensions = [len(emb) for emb in embeddings]
        
        # If all same, return as is
        if len(set(dimensions)) == 1:
            return list(embeddings)
        
        if self.dimension_alignment == "min":
            # Truncate to minimum dimension
            min_dim = min(dimensions)
            return [emb[:min_dim] for emb in embeddings]
        
        elif self.dimension_alignment in ["max", "pad_zero"]:
            # Pad with zeros to maximum dimension
            max_dim = max(dimensions)
            aligned_embeddings = []
            for emb in embeddings:
                aligned_emb = np.zeros(max_dim)
                aligned_emb[:len(emb)] = emb
                aligned_embeddings.append(aligned_emb)
            return aligned_embeddings
        
        else:
            # Default: use minimum dimension
            min_dim = min(dimensions)
            return [emb[:min_dim] for emb in embeddings]
    
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

    def _fuse_embeddings(self, *embeddings: Optional[np.ndarray]) -> np.ndarray:
        """
        Fuse embeddings from multiple models based on fusion_method
        
        Parameters
        ----------
        *embeddings : Optional[np.ndarray]
            Embeddings from each model (in order of self.models)
            
        Returns
        -------
        fused_emb : np.ndarray
            Fused embedding
        """
        # Filter out None and zero embeddings
        valid_embeddings = [
            emb for emb in embeddings 
            if emb is not None and not np.all(emb == 0)
        ]
        
        if len(valid_embeddings) == 0:
            raise ValueError("At least one valid embedding must be provided")
        
        # If only one valid embedding, return it
        if len(valid_embeddings) == 1:
            return valid_embeddings[0]
        
        # Normalize all embeddings
        norm_embeddings = [self._normalize_embedding(emb) for emb in valid_embeddings]

        # Fusion strategies
        if self.fusion_method == "concatenate":
            # Simple concatenation: E = [E1 ; E2 ; ...]
            fused = np.concatenate(valid_embeddings)
            
        elif self.fusion_method == "normalized_average":
            # Average of normalized embeddings: E = (norm(E1) + norm(E2) + ...) / n
            aligned_embeddings = self._align_dimensions(*norm_embeddings)
            fused = np.mean(aligned_embeddings, axis=0)
            fused = self._normalize_embedding(fused)
            
        elif self.fusion_method == "weighted_average":
            # Weighted average: E = α*norm(E1) + β*norm(E2) + ...
            aligned_embeddings = self._align_dimensions(*norm_embeddings)
            # Use weights for valid embeddings only
            valid_weights = self.fusion_weights[:len(valid_embeddings)]
            weight_sum = sum(valid_weights)
            normalized_weights = [w / weight_sum for w in valid_weights]
            
            fused = np.zeros_like(aligned_embeddings[0])
            for i, emb in enumerate(aligned_embeddings):
                fused += normalized_weights[i] * emb
            fused = self._normalize_embedding(fused)
            
        elif self.fusion_method == "product":
            # Element-wise product of normalized embeddings
            aligned_embeddings = self._align_dimensions(*norm_embeddings)
            fused = aligned_embeddings[0]
            for emb in aligned_embeddings[1:]:
                fused = fused * emb
            fused = self._normalize_embedding(fused)
            
        elif self.fusion_method == "max_pool":
            # Max pooling of normalized embeddings
            aligned_embeddings = self._align_dimensions(*norm_embeddings)
            fused = np.maximum.reduce(aligned_embeddings)
            fused = self._normalize_embedding(fused)
            
        elif self.fusion_method == "learned_concat":
            # Weighted concatenation: E = [w1*E1 ; w2*E2 ; ...]
            weighted_embeddings = []
            for i, emb in enumerate(valid_embeddings):
                w = self.fusion_weights[i] if i < len(self.fusion_weights) else 1.0
                weighted_embeddings.append(w * emb)
            fused = np.concatenate(weighted_embeddings)
            fused = self._normalize_embedding(fused)
            
        elif self.fusion_method == "score_level":
            # Score-level fusion doesn't fuse embeddings here
            # Actual score fusion happens in _fuse_score
            # Return concatenation as placeholder
            aligned_embeddings = self._align_dimensions(*norm_embeddings)
            fused = np.concatenate(aligned_embeddings)
            fused = self._normalize_embedding(fused)
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused
    
    def _fuse_score(self, 
                    new_embeddings: List[Optional[np.ndarray]],
                    memory_embeddings: List[Optional[np.ndarray]]) -> float:
        """
        Compute similarity using score-level fusion for multiple models
        
        Parameters
        ----------
        new_embeddings : List[Optional[np.ndarray]]
            New embeddings from each model (in order of self.models)
        memory_embeddings : List[Optional[np.ndarray]]
            Memory embeddings from each model (in order of self.models)
            
        Returns
        -------
        fused_score : float
            Fused similarity score
        """
        # Compute individual similarities
        similarities = []
        valid_weights = []
        
        for i in range(len(new_embeddings)):
            new_emb = new_embeddings[i]
            mem_emb = memory_embeddings[i]
            
            if new_emb is None or mem_emb is None:
                continue
            
            if self.embedding_metric == "cosine":
                sim = self._cosine(new_emb, mem_emb)
            else:
                dist = self._euclidean(new_emb, mem_emb)
                sim = 1 / (1 + dist)
            
            similarities.append(sim)
            valid_weights.append(self.fusion_weights[i] if i < len(self.fusion_weights) else 1.0)
        
        if len(similarities) == 0:
            return 0.0
        
        # Weighted fusion of scores
        weight_sum = sum(valid_weights)
        normalized_weights = [w / weight_sum for w in valid_weights]
        
        fused_score = sum(w * s for w, s in zip(normalized_weights, similarities))
        
        # Check for NaN in final result
        if np.isnan(fused_score):
            print(f"⚠️  Warning: NaN in fused_score! similarities={similarities}")
            return 0.0
        
        return fused_score
    
    def _extract_embeddings(self, file_or_audio_f32: Union[str, np.ndarray], max_speakers: Optional[int] = None) -> Dict:
        """
        Extract embeddings from all models
        
        Returns
        -------
        result : dict
            {
                '<model_name/model_id>_embeddings': np.ndarray or None,
                '<model_name/model_id>_labels': List[str] or None,
                ...
            }
        outputs : dict
            Raw outputs from each model
        """
        result = {}
        outputs = {}
        
        for model_type, model_id in self.model_instances:
            result[f'{model_id}_embeddings'] = None
            result[f'{model_id}_labels'] = None
        
        # Extract from each model
        for model_id, diarizer in self.diarizers.items():
            # Get model type from model_instances
            model_type = None
            for mt, mid in self.model_instances:
                if mid == model_id:
                    model_type = mt
                    break
            
            if model_type is None:
                continue
                
            try:
                if model_type == 'pyannote':
                    # Pyannote requires file path
                    if isinstance(file_or_audio_f32, np.ndarray):
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                            sf.write(tmp_path, file_or_audio_f32, 16000, format="WAV")
                            output = diarizer(
                                tmp_path,
                                max_speakers=max_speakers,
                                use_memory=False
                            )
                        os.unlink(tmp_path)
                    else:
                        output = diarizer(
                            file_or_audio_f32,
                            max_speakers=max_speakers,
                            use_memory=False
                        )
                    
                    if output.speaker_embeddings is not None:
                        result[f'{model_id}_embeddings'] = output.speaker_embeddings
                        result[f'{model_id}_labels'] = list(output.speaker_diarization.labels())
                        print(f"[{model_type.capitalize()}({model_id})] Extracted {len(result[f'{model_id}_labels'])} speakers: {result[f'{model_id}_labels']}")
                    outputs[model_id] = output
                    
                else:  # speechbrain or nemo
                    output = diarizer(
                        file_or_audio_f32,
                        max_speakers=max_speakers,
                        use_memory=False
                    )
                    
                    if output.get('speaker_embeddings') is not None:
                        result[f'{model_id}_embeddings'] = output['speaker_embeddings']
                        result[f'{model_id}_labels'] = output.get('speaker_labels', ['SPEAKER_00'])
                        print(f"[{model_type.capitalize()}({model_id})] Extracted {len(result[f'{model_id}_labels'])} speakers: {result[f'{model_id}_labels']}")
                    outputs[model_id] = output
                    
            except Exception as e:
                print(f"⚠️  {model_type.capitalize()}({model_id}) extraction failed: {e}")
                outputs[model_id] = None
        
        return result, outputs
    
    def _match_speakers_with_memory(self, 
                                   new_embeddings_dict: Dict[str, Optional[np.ndarray]],
                                   new_labels: List[str],
                                   max_speakers: Optional[int] = None,
                                   session_id: Optional[str] = None,
                                   audio_duration: Optional[float] = None) -> Dict[str, str]:
        """
        Match speakers with memory using fused embeddings from multiple models
        
        Parameters
        ----------
        new_embeddings_dict : Dict[str, Optional[np.ndarray]]
            Dictionary of embeddings from each model {model_name/model_id: embeddings}
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
        
        # Get memory for each model instance
        model_memories = {
            model_id: session_data[f'{model_id}_memory']
            for model_type, model_id in self.model_instances
        }
        
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
            # Get embeddings from all model instances for this speaker
            model_embeddings = []
            for model_type, model_id in self.model_instances:
                emb_array = new_embeddings_dict.get(f'{model_id}_embeddings')
                if emb_array is not None and i < len(emb_array):
                    emb = emb_array[i]
                    model_embeddings.append(emb)
                    
                    # Debug: Check for zero embeddings
                    emb_norm = np.linalg.norm(emb)
                    if emb_norm == 0:
                        print(f"⚠️  Warning: {model_type}({model_id}) embedding {i} is zero vector!")
                else:
                    model_embeddings.append(None)
            
            # Check if all embeddings are None
            if all(emb is None for emb in model_embeddings):
                continue
            
            # Fuse embeddings
            fused_emb = self._fuse_embeddings(*model_embeddings)
            
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
                
                # Store individual embeddings for each model instance
                for model_type, model_id in self.model_instances:
                    emb_array = new_embeddings_dict.get(f'{model_id}_embeddings')
                    if emb_array is not None and i < len(emb_array):
                        emb = emb_array[i]
                        if not np.all(emb == 0):
                            model_memories[model_id][speaker_id] = emb.copy()
                
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
                # Get new embeddings from all model instances for speaker i
                new_model_embeddings = []
                for model_type, model_id in self.model_instances:
                    emb_array = new_embeddings_dict.get(f'{model_id}_embeddings')
                    if emb_array is not None and i < len(emb_array):
                        new_model_embeddings.append(emb_array[i])
                    else:
                        new_model_embeddings.append(None)
                
                for j, speaker_id in enumerate(memory_speaker_ids):
                    # Get memory embeddings from all model instances for this speaker
                    mem_model_embeddings = []
                    for model_type, model_id in self.model_instances:
                        mem = model_memories[model_id].get(speaker_id)
                        mem_model_embeddings.append(mem)
                    
                    # Use _fuse_score for score-level fusion
                    fused_sim = self._fuse_score(new_model_embeddings, mem_model_embeddings)
                    
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
                        for model_type, model_id in self.model_instances:
                            emb_array = new_embeddings_dict.get(f'{model_id}_embeddings')
                            if emb_array is not None and i < len(emb_array):
                                new_model_emb = emb_array[i]
                                model_mem = model_memories[model_id]
                                
                                if matched_speaker_id in model_mem:
                                    old_model_emb = model_mem[matched_speaker_id]
                                    updated_model_emb = (1 - self.embedding_update_weight) * old_model_emb + \
                                                       self.embedding_update_weight * new_model_emb
                                    if self.embedding_metric == "cosine":
                                        updated_model_emb = self._normalize_embedding(updated_model_emb)
                                    model_mem[matched_speaker_id] = updated_model_emb
                                else:
                                    model_mem[matched_speaker_id] = new_model_emb.copy()
                    
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
                            for model_type, model_id in self.model_instances:
                                emb_array = new_embeddings_dict.get(f'{model_id}_embeddings')
                                if emb_array is not None and i < len(emb_array):
                                    new_model_emb = emb_array[i]
                                    model_mem = model_memories[model_id]
                                    
                                    if matched_speaker_id in model_mem:
                                        old_model_emb = model_mem[matched_speaker_id]
                                        updated_model_emb = (1 - self.embedding_update_weight) * old_model_emb + \
                                                           self.embedding_update_weight * new_model_emb
                                        if self.embedding_metric == "cosine":
                                            updated_model_emb = self._normalize_embedding(updated_model_emb)
                                        model_mem[matched_speaker_id] = updated_model_emb
                                    else:
                                        model_mem[matched_speaker_id] = new_model_emb.copy()
                        
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
                    
                    # Store individual embeddings for each model instance
                    for model_type, model_id in self.model_instances:
                        emb_array = new_embeddings_dict.get(f'{model_id}_embeddings')
                        if emb_array is not None and i < len(emb_array):
                            model_memories[model_id][speaker_id] = emb_array[i].copy()
                    
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
        
        # Extract embeddings from all models
        extracted, outputs = self._extract_embeddings(file_or_audio_f32, max_speakers=max_speakers)
        
        # Check if any valid embeddings were extracted
        has_valid_embeddings = False
        all_labels = []
        
        for model_type, model_id in self.model_instances:
            labels = extracted.get(f'{model_id}_labels')
            if labels:
                # Check if labels are valid (not all SPEAKER_UNK for pyannote)
                if model_type == 'pyannote':
                    if not all(label == "SPEAKER_UNK" for label in labels):
                        has_valid_embeddings = True
                        all_labels.append((model_id, labels))
                else:
                    has_valid_embeddings = True
                    all_labels.append((model_id, labels))
        
        if not has_valid_embeddings:
            # No valid embeddings from any model
            result = {
                'speaker_embeddings': None,
                'speaker_labels': []
            }
            for model_type, model_id in self.model_instances:
                result[f'{model_id}_embeddings'] = None
            return result
        
        # Determine labels (use the system with more detected speakers)
        if len(all_labels) > 0:
            # Sort by number of speakers (descending)
            all_labels.sort(key=lambda x: len(x[1]), reverse=True)
            current_labels = all_labels[0][1]
        else:
            result = {
                'speaker_embeddings': None,
                'speaker_labels': []
            }
            for model_type, model_id in self.model_instances:
                result[f'{model_id}_embeddings'] = None
            return result
        
        # Prepare embeddings for fusion
        num_speakers = len(current_labels)
        
        # Expand embeddings if needed (pad to match num_speakers)
        for model_type, model_id in self.model_instances:
            embeddings = extracted.get(f'{model_id}_embeddings')
            if embeddings is not None and len(embeddings) < num_speakers:
                # Add padding
                padding = ((0, num_speakers - len(embeddings)), (0, 0))
                extracted[f'{model_id}_embeddings'] = np.pad(embeddings, padding, mode='constant')
        
        output = {'speaker_labels': current_labels}
        for model_type, model_id in self.model_instances:
            output[f'{model_id}_embeddings'] = extracted.get(f'{model_id}_embeddings')
        
        if not use_memory:
            # Just return fused embeddings without memory matching
            fused_embeddings = []
            for i in range(num_speakers):
                # Get embeddings from all model instances for this speaker
                model_embeddings = []
                for model_type, model_id in self.model_instances:
                    emb_array = extracted.get(f'{model_id}_embeddings')
                    if emb_array is not None and i < len(emb_array):
                        model_embeddings.append(emb_array[i])
                    else:
                        model_embeddings.append(None)
                
                fused_emb = self._fuse_embeddings(*model_embeddings)
                fused_embeddings.append(fused_emb)
            
            output['speaker_embeddings'] = np.array(fused_embeddings)
            return output
        
        # Get session data
        session_data = self._get_session_data()
        
        # Match with memory
        label_mapping = self._match_speakers_with_memory(
            extracted,  # Pass the extracted dict
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
                'fusion_weights': self.fusion_weights,
                'models': [f"{mt}({mid})" if mt != mid else mt for mt, mid in self.model_instances]
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
            'fusion_weights': self.fusion_weights,
            'models': [f"{mt}({mid})" if mt != mid else mt for mt, mid in self.model_instances]
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
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # ========================================================================
    # EXAMPLE 1: SpeechBrain + Pyannote (2 models)
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 1: SpeechBrain + Pyannote Fusion")
    print("=" * 80)
    
    pipeline_sb_pyan = RealtimeSpeakerDiarization(
        models=['speechbrain', 'pyannote'],
        fusion_method="score_level",
        fusion_weights=[0.4, 0.6],  # 40% SpeechBrain, 60% Pyannote
        model_configs={
            'pyannote': {
                'model_name': "pyannote/speaker-diarization-community-1",
                'token': os.getenv("HF_TOKEN")
            }
        },
        similarity_threshold=0.7,
        embedding_update_weight=0.3,
        min_similarity_gap=0.3,
        skip_update_short_audio=True,
        min_duration_for_update=2.0,
        init_similarity_threshold=0.4
    )
    
    session_1 = "sb_pyan_conversation"
    pipeline_sb_pyan.set_session(session_1)
    
    output1 = pipeline_sb_pyan(
        "/home/hoang/speaker_diarization/wav/A1.wav",
        num_speakers=2,
        use_memory=True,
        session_id=session_1
    )
    
    print(f"\n📊 Results: {output1['speaker_labels']}")
    print(f"💾 Speaker Memory: {pipeline_sb_pyan.get_speaker_info(session_1)}")
    
    # ========================================================================
    # EXAMPLE 2: SpeechBrain + NeMo (2 models)
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 2: SpeechBrain + NeMo Fusion")
    print("=" * 80)
    
    pipeline_sb_nemo = RealtimeSpeakerDiarization(
        models=['speechbrain', 'nemo'],
        fusion_method="weighted_average",
        fusion_weights=[0.5, 0.5],  # Equal weights
        model_configs={
            'nemo': {
                'pretrained_speaker_model': "titanet_large"
            }
        },
        similarity_threshold=0.7,
        embedding_update_weight=0.3,
        min_similarity_gap=0.3,
        skip_update_short_audio=True,
        min_duration_for_update=2.0,
        init_similarity_threshold=0.4
    )
    
    session_2 = "sb_nemo_conversation"
    pipeline_sb_nemo.set_session(session_2)
    
    output2 = pipeline_sb_nemo(
        "/home/hoang/speaker_diarization/wav/A2.wav",
        num_speakers=2,
        use_memory=True,
        session_id=session_2
    )
    
    print(f"\n📊 Results: {output2['speaker_labels']}")
    print(f"💾 Speaker Memory: {pipeline_sb_nemo.get_speaker_info(session_2)}")
    
    # ========================================================================
    # EXAMPLE 3: Pyannote + NeMo (2 models)
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Pyannote + NeMo Fusion")
    print("=" * 80)
    
    pipeline_pyan_nemo = RealtimeSpeakerDiarization(
        models=['pyannote', 'nemo'],
        fusion_method="normalized_average",
        model_configs={
            'pyannote': {
                'model_name': "pyannote/speaker-diarization-community-1",
                'token': os.getenv("HF_TOKEN")
            },
            'nemo': {
                'pretrained_speaker_model': "titanet_large"
            }
        },
        similarity_threshold=0.7,
        embedding_update_weight=0.3,
        min_similarity_gap=0.3,
        skip_update_short_audio=True,
        min_duration_for_update=2.0,
        init_similarity_threshold=0.4
    )
    
    session_3 = "pyan_nemo_conversation"
    pipeline_pyan_nemo.set_session(session_3)
    
    output3 = pipeline_pyan_nemo(
        "/home/hoang/speaker_diarization/wav/B1.wav",
        num_speakers=2,
        use_memory=True,
        session_id=session_3
    )
    
    print(f"\n📊 Results: {output3['speaker_labels']}")
    print(f"💾 Speaker Memory: {pipeline_pyan_nemo.get_speaker_info(session_3)}")
    
    # ========================================================================
    # EXAMPLE 4: SpeechBrain + Pyannote + NeMo (3 models)
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 4: SpeechBrain + Pyannote + NeMo Fusion (3 models)")
    print("=" * 80)
    
    pipeline_all = RealtimeSpeakerDiarization(
        models=['speechbrain', 'pyannote', 'nemo'],
        fusion_method="score_level",
        fusion_weights=[0.3, 0.4, 0.3],  # 30% SB, 40% Pyannote, 30% NeMo
        model_configs={
            'pyannote': {
                'model_name': "pyannote/speaker-diarization-community-1",
                'token': os.getenv("HF_TOKEN")
            },
            'nemo': {
                'pretrained_speaker_model': "titanet_large"
            }
        },
        similarity_threshold=0.7,
        embedding_update_weight=0.3,
        min_similarity_gap=0.3,
        skip_update_short_audio=True,
        min_duration_for_update=2.0,
        init_similarity_threshold=0.4
    )

    
    session_4 = "all_models_conversation"
    pipeline_all.set_session(session_4)
    
    print("\n--- Processing chunk 1 ---")
    output4_1 = pipeline_all(
        "/home/hoang/speaker_diarization/wav/A1.wav",
        num_speakers=2,
        use_memory=True,
        session_id=session_4
    )
    print(f"📊 Results chunk 1: {output4_1['speaker_labels']}")
    print(f"💾 Speaker Memory: {pipeline_all.get_speaker_info(session_4)}")
    
    print("\n--- Processing chunk 2 ---")
    output4_2 = pipeline_all(
        "/home/hoang/speaker_diarization/wav/B1.wav",
        num_speakers=2,
        use_memory=True,
        session_id=session_4
    )
    print(f"📊 Results chunk 2: {output4_2['speaker_labels']}")
    print(f"💾 Speaker Memory: {pipeline_all.get_speaker_info(session_4)}")
    
    # ========================================================================
    # EXAMPLE 5: Different fusion methods with 3 models
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Testing different fusion methods")
    print("=" * 80)
    
    fusion_methods = [
        "concatenate",
        "normalized_average",
        "weighted_average",
        "product",
        "max_pool",
        "learned_concat"
    ]
    
    for method in fusion_methods:
        print(f"\n--- Testing {method} ---")
        try:
            pipeline_test = RealtimeSpeakerDiarization(
                models=['speechbrain', 'pyannote'],
                fusion_method=method,
                fusion_weights=[0.5, 0.5],
                model_configs={
                    'pyannote': {
                        'model_name': "pyannote/speaker-diarization-community-1",
                        'token': os.getenv("HF_TOKEN")
                    }
                },
                similarity_threshold=0.7
            )
            
            pipeline_test.set_session(f"test_{method}")
            output_test = pipeline_test(
                "/home/hoang/speaker_diarization/wav/A1.wav",
                num_speakers=2,
                use_memory=True,
                session_id=f"test_{method}"
            )
            print(f"✅ {method}: {output_test['speaker_labels']}")
        except Exception as e:
            print(f"❌ {method} failed: {e}")
    
    # ========================================================================
    # EXAMPLE 6: Multiple instances of same model type with different configs
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Multiple Pyannote models with different configs")
    print("=" * 80)
    
    try:
        pipeline_multi_pyan = RealtimeSpeakerDiarization(
            models=[
                ('nemo', 'nemo_tianet'),
                ('nemo', 'nemo_ecapa_tdnn'),
                ('pyannote', 'pyan_community'),
                ('speechbrain', 'sb_default')
            ],
            fusion_method="score_level",
            fusion_weights=[0.3, 0.4, 0.3],  # 30% v3, 40% community, 30% SB
            model_configs={
                'nemo_tianet': {'pretrained_speaker_model': 'titanet_large'},
                'nemo_ecapa_tdnn': {'pretrained_speaker_model': 'ecapa_tdnn'},
                'pyan_community': {
                    'model_name': "pyannote/speaker-diarization-community-1",
                    'token': os.getenv("HF_TOKEN")
                },
                'sb_default': {}
            },
            similarity_threshold=0.7,
            embedding_update_weight=0.3,
            min_similarity_gap=0.3
        )
        
        session_multi = "multi_pyannote_session"
        pipeline_multi_pyan.set_session(session_multi)
        
        output_multi = pipeline_multi_pyan(
            "/home/hoang/speaker_diarization/wav/A1.wav",
            num_speakers=2,
            use_memory=True,
            session_id=session_multi
        )
        
        print(f"\n📊 Results: {output_multi['speaker_labels']}")
        print(f"💾 Speaker Memory: {pipeline_multi_pyan.get_speaker_info(session_multi)}")
    except Exception as e:
        print(f"❌ Example 6 failed: {e}")
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
    
    
    # ========================================================================
    # LEGACY EXAMPLES (kept for compatibility)
    # ========================================================================
    print("\n" + "=" * 80)
    print("LEGACY EXAMPLES")
    print("=" * 80)
    
    # Ví dụ 1: Xử lý audio chunk đầu tiên với Session ID
    print("=" * 100)
    print("VÍ DỤ 1: Xử lý audio chunk đầu tiên với Session ID")
    print("=" * 100)
    
    pipeline_legacy = RealtimeSpeakerDiarization(
        models=['speechbrain', 'pyannote'],
        fusion_method="score_level",
        fusion_weights=[0.4, 0.6],
        model_configs={
            'pyannote': {
                'model_name': "pyannote/speaker-diarization-community-1",
                'token': os.getenv("HF_TOKEN")
            }
        },
        similarity_threshold=0.7,
        embedding_update_weight=0.3,
        min_similarity_gap=0.3,
        skip_update_short_audio=True,
        min_duration_for_update=2.0,
        init_similarity_threshold=0.4
    )
    
    # Set session for this conversation
    session_1 = "conversation_1"
    pipeline_legacy.set_session(session_1)

    output1 = pipeline_legacy(
        "/home/hoang/speaker_diarization/wav/A1.wav",
        num_speakers=2,
        use_memory=True,
        session_id=session_1
    )

    print("\n📊 Kết quả chunk 1:")
    print(f"  🎤 {output1['speaker_labels']}")

    print(f"\n💾 Speaker Memory (Session {session_1}): {pipeline_legacy.get_speaker_info(session_1)}")
    print(f"\n📋 All Sessions: {pipeline_legacy.list_sessions()}")

    # Ví dụ 2: Xử lý một conversation khác với session_id khác
    print("\n" + "=" * 100)
    print("VÍ DỤ 2: Xử lý conversation thứ 2 với session khác")
    print("=" * 100)
    
    session_2 = "conversation_2"
    pipeline_legacy.set_session(session_2)

    output2 = pipeline_legacy(
        "/home/hoang/speaker_diarization/wav/A2.wav",
        num_speakers=2,
        use_memory=True,
        session_id=session_2
    )

    print("\n📊 Kết quả chunk 1 (Session 2):")
    print(f"  🎤 {output2['speaker_labels']}")

    print(f"\n💾 Speaker Memory (Session {session_2}): {pipeline_legacy.get_speaker_info(session_2)}")
    print(f"💾 Speaker Memory (Session {session_1}): {pipeline_legacy.get_speaker_info(session_1)}")
    print(f"\n📋 All Sessions: {pipeline_legacy.list_sessions()}")

    # Ví dụ 3: Quay lại session 1 và xử lý chunk tiếp theo
    print("\n" + "=" * 100)
    print("VÍ DỤ 3: Quay lại Session 1 và xử lý chunk tiếp theo")
    print("=" * 100)
    
    pipeline_legacy.set_session(session_1)  # Chuyển về session 1

    output3 = pipeline_legacy(
        "/home/hoang/speaker_diarization/wav/B1.wav",
        num_speakers=2,
        use_memory=True,
        session_id=session_1
    )

    print("\n📊 Kết quả chunk 2 (Session 1 continued):")
    print(f"  🎤 {output3['speaker_labels']}")

    print(f"\n💾 Speaker Memory (Session 1 updated): {pipeline_legacy.get_speaker_info(session_1)}")
    
    # Ví dụ 4: Demo session management operations
    print("\n" + "=" * 100)
    print("VÍ DỤ 4: Session Management - Reset và Delete")
    print("=" * 100)
    
    # Liệt kê tất cả sessions
    print(f"\n📋 Current sessions: {pipeline_legacy.list_sessions()}")
    print(f"📍 Current session ID: {pipeline_legacy.get_current_session_id()}")
    
    # Reset một session (xóa speaker memory nhưng giữ session)
    print(f"\n🔄 Resetting session: {session_1}")
    pipeline_legacy.reset_session(session_1)
    print(f"💾 Speaker Memory after reset: {pipeline_legacy.get_speaker_info(session_1)}")
    
    # Xem thông tin tất cả sessions
    print(f"\n📊 All sessions info:")
    for session_id, info in pipeline_legacy.get_all_sessions_info().items():
        print(f"  Session '{session_id}': {info['num_speakers']} speakers, {info['total_chunks']} chunks")

    print(f"\n💾 Speaker Memory (updated): {pipeline_legacy.get_speaker_info()}")
    print("=" * 100)
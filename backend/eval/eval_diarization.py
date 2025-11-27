import os
from dotenv import load_dotenv
load_dotenv()
import random
import numpy as np
from tqdm import tqdm
from glob import glob
from itertools import combinations
from sklearn.metrics import roc_curve, precision_recall_fscore_support, auc, precision_recall_curve
import matplotlib.pyplot as plt
import pickle
import hashlib
import sys
sys.path.append("..")
from fusion_diarization import RealtimeSpeakerDiarization

pipeline = RealtimeSpeakerDiarization(
    models=[
        ('nemo', 'nemo_tianet'),
        ('nemo', 'nemo_ecapa_tdnn'),
        ('pyannote', 'pyan_community'),
        ('speechbrain', 'sb_default')
    ],
    model_configs={
        'nemo_tianet': {'pretrained_speaker_model': 'titanet_large'},
        'nemo_ecapa_tdnn': {'pretrained_speaker_model': 'ecapa_tdnn'},
        'pyan_community': {
            'model_name': "pyannote/speaker-diarization-community-1",
            'token': os.getenv("HF_TOKEN")
        },
        'sb_default': {}
    }
)

random.seed(123)
np.random.seed(123)

# ========= 1) Liệt kê dữ liệu =========
def list_speakers_and_utts(root):
    spk2utts = {}
    for spk in sorted(os.listdir(root)):
        spk_dir = os.path.join(root, spk)
        if not os.path.isdir(spk_dir):
            continue

        utts = []
        # 4 loại folder
        for sub in ["falset10", "nonpara30", "parallel100", "whisper10"]:
            wdir = os.path.join(spk_dir, sub, "wav24kHz16bit")
            if os.path.isdir(wdir):
                wavs = glob(os.path.join(wdir, "*.wav"))
                utts.extend(wavs)

        if len(utts) >= 2:
            spk2utts[spk] = sorted(utts)

    return spk2utts

# ========= 2) Xây trial =========
def build_trials(spk2utts, max_genuine_per_spk=50, impostor_per_spk=100):
    """Trả về danh sách (path1, path2, label) với label=1 genuine, 0 impostor."""
    trials = []
    speakers = sorted(spk2utts.keys())

    # Genuine: mọi cặp 2 utt khác nhau trong cùng speaker (hoặc sample bớt)
    for spk in speakers:
        utts = spk2utts[spk]
        pairs = list(combinations(utts, 2))
        random.shuffle(pairs)
        for p in pairs[:max_genuine_per_spk]:
            trials.append((p[0], p[1], 1))

    # Impostor: ghép ngẫu nhiên giữa spk này và spk khác
    for spk in speakers:
        others = [s for s in speakers if s != spk]
        utts_a = spk2utts[spk]
        # Lấy random một utt làm anchor, ghép với utt random của spk khác
        for _ in range(impostor_per_spk):
            ua = random.choice(utts_a)
            spk_b = random.choice(others)
            ub = random.choice(spk2utts[spk_b])
            trials.append((ua, ub, 0))

    random.shuffle(trials)
    return trials

# ========= 3) Cache management =========
def get_cache_key(trials):
    """Tạo unique key từ danh sách trials để identify cache."""
    all_files = sorted(set([p for t in trials for p in (t[0], t[1])]))
    files_str = '|'.join(all_files)
    return hashlib.md5(files_str.encode()).hexdigest()

def save_embedding_cache(emb_cache, cache_file):
    """Lưu embedding cache ra file."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(emb_cache, f)
        print(f"Saved embedding cache to: {cache_file}")
        return True
    except Exception as e:
        print(f"Error saving cache: {e}")
        return False

def load_embedding_cache(cache_file):
    """Load embedding cache từ file."""
    try:
        if not os.path.exists(cache_file):
            return None
        with open(cache_file, 'rb') as f:
            emb_cache = pickle.load(f)
        print(f"Loaded embedding cache from: {cache_file} ({len(emb_cache)} files)")
        return emb_cache
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None

def clear_cache(cache_dir="eval_cache"):
    """Xóa tất cả cache files trong thư mục cache."""
    try:
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"Cleared all cache in: {cache_dir}")
            return True
        else:
            print(f"Cache directory does not exist: {cache_dir}")
            return False
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return False

# ========= 4) Trích embedding =========
def extract_all_embeddings(trials, cache_dir="eval_cache", use_cache=True):
    """Extract embeddings một lần cho tất cả các file, lưu cả 3 loại."""
    
    # Tạo cache key và đường dẫn
    cache_key = get_cache_key(trials)
    cache_file = os.path.join(cache_dir, f"embeddings_cache_{cache_key}.pkl")
    
    # Thử load từ cache nếu use_cache=True
    if use_cache:
        emb_cache = load_embedding_cache(cache_file)
        if emb_cache is not None:
            print("Using cached embeddings, skipping extraction.")
            return emb_cache
        else:
            print("No valid cache found, extracting embeddings...")
    
    emb_cache = {}  # {file_path: {"pyannote": emb, "speechbrain": emb, "nemo_tianet": emb, "nemo_ecapa_tdnn": emb}}
    failed_files = []
    
    # Lấy danh sách tất cả các file unique
    all_files = set()
    for p1, p2, _ in trials:
        all_files.add(p1)
        all_files.add(p2)
    
    # Extract embeddings một lần cho mỗi file
    for fpath in tqdm(list(all_files), desc="Extracting embeddings"):
        try:
            result, _ = pipeline._extract_embeddings(fpath, max_speakers=1)
            
            # Kiểm tra nếu result hoặc embeddings là None/empty
            if result is None:
                failed_files.append(fpath)
                continue
                
            pyannote_emb = result.get("pyan_community_embeddings")
            speechbrain_emb = result.get("sb_default_embeddings")
            nemo_emb_tianet = result.get("nemo_tianet_embeddings")
            nemo_emb_ecapa_tdnn = result.get("nemo_ecapa_tdnn_embeddings")
            
            if (pyannote_emb is None or len(pyannote_emb) == 0 or
                speechbrain_emb is None or len(speechbrain_emb) == 0 or
                nemo_emb_tianet is None or len(nemo_emb_tianet) == 0 or
                nemo_emb_ecapa_tdnn is None or len(nemo_emb_ecapa_tdnn) == 0):
                failed_files.append(fpath)
                continue
            
            # Kiểm tra embedding có toàn 0 hoặc NaN không
            pyannote_vec = pyannote_emb[0]
            speechbrain_vec = speechbrain_emb[0]
            nemo_vec_tianet = nemo_emb_tianet[0]
            nemo_vec_ecapa_tdnn = nemo_emb_ecapa_tdnn[0]
            
            if (np.all(pyannote_vec == 0) or np.all(np.isnan(pyannote_vec)) or
                np.all(speechbrain_vec == 0) or np.all(np.isnan(speechbrain_vec)) or
                np.all(nemo_vec_tianet == 0) or np.all(np.isnan(nemo_vec_tianet)) or
                np.all(nemo_vec_ecapa_tdnn == 0) or np.all(np.isnan(nemo_vec_ecapa_tdnn))):
                failed_files.append(fpath)
                continue
            
            emb_cache[fpath] = {
                "pyannote": pyannote_vec.copy(),
                "speechbrain": speechbrain_vec.copy(),
                "nemo_tianet": nemo_vec_tianet.copy(),
                "nemo_ecapa_tdnn": nemo_vec_ecapa_tdnn.copy()
            }
        except Exception as e:
            print(f"\nError extracting embeddings from {fpath}: {e}")
            failed_files.append(fpath)
            continue
    
    if failed_files:
        print(f"\nWarning: Failed to extract embeddings from {len(failed_files)}/{len(all_files)} files")
    
    # Lưu cache nếu use_cache=True
    if use_cache and len(emb_cache) > 0:
        save_embedding_cache(emb_cache, cache_file)
    
    return emb_cache

# ========= 5) Tính score =========
def compute_scores_from_cache(trials, emb_cache, embedding_type):
    """Tính cosine scores từ cache có sẵn cho một loại embedding cụ thể."""
    scores, labels = [], []
    skipped_trials = 0
    
    for p1, p2, y in trials:
        # Kiểm tra nếu file không có trong cache (do extraction failed)
        if p1 not in emb_cache or p2 not in emb_cache:
            skipped_trials += 1
            continue
        
        # Kiểm tra nếu embedding type không có hoặc là None
        if (embedding_type not in emb_cache[p1] or 
            embedding_type not in emb_cache[p2] or
            emb_cache[p1][embedding_type] is None or 
            emb_cache[p2][embedding_type] is None):
            skipped_trials += 1
            continue
        
        try:
            emb1 = emb_cache[p1][embedding_type]
            emb2 = emb_cache[p2][embedding_type]
            
            # Kiểm tra thêm nếu embedding toàn 0 hoặc NaN (double-check)
            if (np.all(emb1 == 0) or np.all(np.isnan(emb1)) or
                np.all(emb2 == 0) or np.all(np.isnan(emb2))):
                skipped_trials += 1
                continue
            
            s = pipeline._cosine(emb1, emb2)
            
            # Kiểm tra nếu cosine score là NaN hoặc inf
            if np.isnan(s) or np.isinf(s):
                skipped_trials += 1
                continue
            
            scores.append(s)
            labels.append(y)
        except Exception as e:
            print(f"\nError computing cosine similarity: {e}")
            skipped_trials += 1
            continue
    
    if skipped_trials > 0:
        print(f"Skipped {skipped_trials}/{len(trials)} trials due to missing embeddings")
    
    return np.array(scores), np.array(labels)

# ========= 6) FAR/FRR/EER =========
def compute_metrics_at_threshold(scores, labels, threshold):
    """Tính precision, recall, F1 tại một threshold cụ thể."""
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    return float(precision), float(recall), float(f1)

def compute_far_frr_eer(scores, labels):
    # labels: 1=genuine, 0=impostor; scores: higher=more similar
    fpr, tpr, thresholds = roc_curve(labels, scores)  # fpr = FAR theo định nghĩa verification
    fnr = 1 - tpr

    # EER: điểm gần nhất khi FAR ~ FRR
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2.0
    thr_eer = thresholds[idx_eer]

    # FAR/FRR tại ngưỡng EER (xấp xỉ)
    far_at_eer = fpr[idx_eer]
    frr_at_eer = fnr[idx_eer]
    
    # Precision, Recall, F1 tại threshold EER
    precision_at_eer, recall_at_eer, f1_at_eer = compute_metrics_at_threshold(
        scores, labels, thr_eer
    )
    
    # Tìm best F1 score
    best_f1 = 0.0
    best_f1_threshold = thr_eer
    best_f1_precision = precision_at_eer
    best_f1_recall = recall_at_eer
    
    for thr in thresholds:
        precision, recall, f1 = compute_metrics_at_threshold(scores, labels, thr)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = thr
            best_f1_precision = precision
            best_f1_recall = recall
    
    # Tính AUC (Area Under ROC Curve)
    roc_auc = auc(fpr, tpr)
    
    return {
        "EER": float(eer),
        "threshold_at_EER": float(thr_eer),
        "FAR_at_EER": float(far_at_eer),
        "FRR_at_EER": float(frr_at_eer),
        "precision_at_EER": float(precision_at_eer),
        "recall_at_EER": float(recall_at_eer),
        "F1_at_EER": float(f1_at_eer),
        "best_F1": float(best_f1),
        "threshold_at_best_F1": float(best_f1_threshold),
        "precision_at_best_F1": float(best_f1_precision),
        "recall_at_best_F1": float(best_f1_recall),
        "AUC": float(roc_auc),
        "FAR_curve": fpr, "FRR_curve": fnr, "thresholds": thresholds
    }

def plot_roc_curves(results, output_dir="eval_results"):
    """Vẽ và lưu ROC curves cho cả 3 loại embeddings."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    colors = {"pyannote": "blue", "speechbrain": "red", "nemo_tianet": "green", "nemo_ecapa_tdnn": "yellow"}
    
    for emb_type in ["pyannote", "speechbrain", "nemo_tianet", "nemo_ecapa_tdnn"]:
        if results[emb_type] is None:
            continue
            
        metrics = results[emb_type]
        fpr = metrics["FAR_curve"]
        tpr = 1 - metrics["FRR_curve"]  # TPR = 1 - FNR
        auc_score = metrics["AUC"]
        eer = metrics["EER"]
        
        # Vẽ ROC curve
        plt.plot(fpr, tpr, color=colors[emb_type], lw=2,
                label=f'{emb_type.capitalize()} (AUC={auc_score:.3f}, EER={eer*100:.2f}%)')
        
        # Đánh dấu điểm EER
        idx_eer = np.nanargmin(np.abs((1 - tpr) - fpr))
        plt.plot(fpr[idx_eer], tpr[idx_eer], 'o', color=colors[emb_type], 
                markersize=8, label=f'{emb_type.capitalize()} EER point')
    
    # Đường chéo (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)', fontsize=12)
    plt.ylabel('True Positive Rate (1 - FRR)', fontsize=12)
    plt.title('ROC Curves - Speaker Verification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Lưu ảnh
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curve saved to: {roc_path}")
    plt.close()

def plot_det_curves(results, output_dir="eval_results"):
    """Vẽ và lưu DET curves (Detection Error Tradeoff) - log scale."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    colors = {"pyannote": "blue", "speechbrain": "red", "nemo_tianet": "green", "nemo_ecapa_tdnn": "yellow"}
    
    for emb_type in ["pyannote", "speechbrain", "nemo_tianet", "nemo_ecapa_tdnn"]:
        if results[emb_type] is None:
            continue
            
        metrics = results[emb_type]
        far = metrics["FAR_curve"]
        frr = metrics["FRR_curve"]
        eer = metrics["EER"]
        
        # Vẽ DET curve (FAR vs FRR)
        plt.plot(far * 100, frr * 100, color=colors[emb_type], lw=2,
                label=f'{emb_type.capitalize()} (EER={eer*100:.2f}%)')
        
        # Đánh dấu điểm EER
        idx_eer = np.nanargmin(np.abs(frr - far))
        plt.plot(far[idx_eer] * 100, frr[idx_eer] * 100, 'o', 
                color=colors[emb_type], markersize=8)
    
    # Đường chéo (EER line)
    plt.plot([0, 50], [0, 50], 'k--', lw=1, label='EER line (FAR=FRR)')
    
    plt.xlim([0, 50])
    plt.ylim([0, 50])
    plt.xlabel('False Acceptance Rate (%)', fontsize=12)
    plt.ylabel('False Rejection Rate (%)', fontsize=12)
    plt.title('DET Curves - Speaker Verification', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Lưu ảnh
    det_path = os.path.join(output_dir, "det_curves.png")
    plt.savefig(det_path, dpi=300, bbox_inches='tight')
    print(f"DET curve saved to: {det_path}")
    plt.close()

def plot_precision_recall_curves(results, scores_data, output_dir="eval_results"):
    """Vẽ và lưu Precision-Recall curves."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    colors = {"pyannote": "blue", "speechbrain": "red", "nemo_tianet": "green", "nemo_ecapa_tdnn": "yellow"}
    
    for emb_type in ["pyannote", "speechbrain", "nemo_tianet", "nemo_ecapa_tdnn"]:
        if results[emb_type] is None or emb_type not in scores_data:
            continue
            
        scores, labels = scores_data[emb_type]
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        
        metrics = results[emb_type]
        best_f1 = metrics["best_F1"]
        
        # Vẽ PR curve
        plt.plot(recall, precision, color=colors[emb_type], lw=2,
                label=f'{emb_type.capitalize()} (AUC={pr_auc:.3f}, Best F1={best_f1*100:.2f}%)')
        
        # Đánh dấu điểm best F1
        best_prec = metrics["precision_at_best_F1"]
        best_rec = metrics["recall_at_best_F1"]
        plt.plot(best_rec, best_prec, 'o', color=colors[emb_type], 
                markersize=8, label=f'{emb_type.capitalize()} Best F1 point')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Speaker Verification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Lưu ảnh
    pr_path = os.path.join(output_dir, "precision_recall_curves.png")
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curve saved to: {pr_path}")
    plt.close()

# ========= 7) Chạy end-to-end =========
def evaluate_dataset(root, output_dir="eval_results", use_cache=True):
    """Đánh giá cả 3 loại embeddings trong 1 lần chạy.
    
    Args:
        root: Đường dẫn đến dataset
        use_cache: Nếu True, sẽ load embeddings từ cache (nếu có) hoặc lưu cache sau khi extract.
                   Nếu False, sẽ extract lại từ đầu và không lưu cache.
    """
    spk2utts = list_speakers_and_utts(root)
    print(f"Found {len(spk2utts)} speakers usable.")
    trials = build_trials(spk2utts, max_genuine_per_spk=50, impostor_per_spk=100)
    print(f"Total trials: {len(trials)}")
    
    # Extract embeddings một lần cho tất cả files (hoặc load từ cache)
    emb_cache = extract_all_embeddings(trials, use_cache=use_cache)
    
    # Tính metrics cho cả 3 loại embeddings
    results = {}
    scores_data = {}  # Lưu scores để vẽ precision-recall curves
    
    for emb_type in ["pyannote", "speechbrain", "nemo_tianet", "nemo_ecapa_tdnn"]:
        print(f"\n=== Evaluating {emb_type} embeddings ===")
        scores, labels = compute_scores_from_cache(trials, emb_cache, emb_type)
        
        if len(scores) == 0:
            print(f"Error: No valid trials for {emb_type} embeddings!")
            results[emb_type] = None
            continue
        
        scores_data[emb_type] = (scores, labels)  # Lưu để vẽ biểu đồ
        
        print(f"Computing metrics on {len(scores)} valid trials")
        metrics = compute_far_frr_eer(scores, labels)
        print(f"EER: {metrics['EER']*100:.2f}% | FAR@EER: {metrics['FAR_at_EER']*100:.2f}% | "
              f"FRR@EER: {metrics['FRR_at_EER']*100:.2f}% | Thr(EER): {metrics['threshold_at_EER']:.4f}")
        print(f"Precision@EER: {metrics['precision_at_EER']*100:.2f}% | "
              f"Recall@EER: {metrics['recall_at_EER']*100:.2f}% | "
              f"F1@EER: {metrics['F1_at_EER']*100:.2f}%")
        print(f"Best F1: {metrics['best_F1']*100:.2f}% | "
              f"Precision@F1: {metrics['precision_at_best_F1']*100:.2f}% | "
              f"Recall@F1: {metrics['recall_at_best_F1']*100:.2f}% | "
              f"Thr(F1): {metrics['threshold_at_best_F1']:.4f}")
        print(f"AUC: {metrics['AUC']:.4f}")
        results[emb_type] = metrics
    
    # Vẽ và lưu các biểu đồ
    print("\n=== Plotting curves ===")
    plot_roc_curves(results, output_dir)
    plot_det_curves(results, output_dir)
    plot_precision_recall_curves(results, scores_data, output_dir)
    
    return results

if __name__ == "__main__":
    # Nếu muốn xóa cache cũ và extract lại từ đầu, uncomment dòng này:
    # clear_cache()
    
    # Sử dụng cache để tăng tốc (set use_cache=False để force re-extract)
    results = evaluate_dataset("dataset/jvs_ver1", output_dir="eval_results", use_cache=True)
    print("\n=== Final Results ===")
    print("Pyannote:", results["pyannote"])
    print("SpeechBrain:", results["speechbrain"])
    print("NeMo Titanet:", results["nemo_tianet"])
    print("NeMo Ecapa TDNN:", results["nemo_ecapa_tdnn"])

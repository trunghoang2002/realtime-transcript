import os
import random
import csv

ROOT = "dataset/jvs_ver1"  # thư mục gốc
CATEGORIES = ["parallel100", "nonpara30", "whisper10", "falset10"]

OUTPUT_CSV = "dataset_400_testcases.csv"


def load_transcripts(txt_path):
    """Đọc transcripts_utf8.txt → trả về dict: file_name → transcript"""
    mapping = {}
    if not os.path.exists(txt_path):
        return mapping

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            fname, text = line.split(":", 1)
            mapping[fname] = text
    return mapping


def pick_one_random_file(speaker_dir, category):
    """Chọn 1 file WAV + transcript bất kỳ trong thư mục category."""
    cat_dir = os.path.join(speaker_dir, category)

    transcript_path = os.path.join(cat_dir, "transcripts_utf8.txt")
    wav_dir = os.path.join(cat_dir, "wav24kHz16bit")

    if not os.path.exists(transcript_path) or not os.path.exists(wav_dir):
        return None

    transcripts = load_transcripts(transcript_path)
    if not transcripts:
        return None

    # Lọc các file .wav tồn tại
    wav_files = [
        f for f in os.listdir(wav_dir)
        if f.endswith(".wav") and f.replace(".wav", "") in transcripts
    ]

    if not wav_files:
        return None

    chosen = random.choice(wav_files)
    file_id = chosen.replace(".wav", "")

    return {
        "speaker": os.path.basename(speaker_dir),
        "category": category,
        "wav_path": os.path.join(wav_dir, chosen),
        "transcript": transcripts[file_id],
        "file_name": file_id,
    }


def build_dataset():
    dataset = []

    speakers = sorted([
        os.path.join(ROOT, d)
        for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d))
    ])

    for speaker_dir in speakers:
        for category in CATEGORIES:
            item = pick_one_random_file(speaker_dir, category)
            if item:
                dataset.append(item)
            else:
                print(f"[WARN] Missing data in: {speaker_dir}/{category}")

    return dataset


def save_csv(dataset):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["speaker", "category", "file_name", "wav_path", "transcript"]
        )
        writer.writeheader()
        writer.writerows(dataset)

    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    dataset = build_dataset()
    print(f"Collected {len(dataset)} samples.")
    save_csv(dataset)
from jiwer import wer, cer
import re

def eval(ground_truth_file, prediction_file):

    with open(ground_truth_file, "r") as f:
        ground_truth = f.read()
    with open(prediction_file, "r") as f:
        prediction = f.read()

    # normalize text
    ground_truth = ground_truth.lower()
    prediction = prediction.lower()

    pattern = r"[,.~!?・、。「」『』（）【】〈〉《》！？〜～…‥―：；※]"
    ground_truth = re.sub(pattern, "", ground_truth)
    prediction = re.sub(pattern, "", prediction)

    ground_truth = re.sub(r"\s+", " ", ground_truth).strip()
    prediction = re.sub(r"\s+", " ", prediction).strip()

    print("WER:", wer(ground_truth, prediction))
    print("CER:", cer(ground_truth, prediction))

lang = "ja" # ja | en
type = "realtime" # realtime | upload
model = "sensevoice-small" # sensevoice-small | whisper-small | whisper-medium

ground_truth_file = f"{lang}/ground_truth_full.txt"
prediction_file = f"{lang}/{type}/{model}/predict_full.txt"

eval(ground_truth_file, prediction_file)
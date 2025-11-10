from jiwer import wer, cer

with open("ja/ground_truth_full.txt", "r") as f:
    ground_truth = f.read()
with open("ja/realtime/predict_full.txt", "r") as f:
    prediction = f.read()

print("WER:", wer(ground_truth, prediction))
print("CER:", cer(ground_truth, prediction))
"""
Evaluation and Metrics Calculator for PhoBERT Test Set Results
Calculates exact Accuracy, Precision, Recall, Macro-F1 from the Confusion Matrix and Test Predictions.
"""

import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Ma trận nhầm lẫn từ hình ảnh kết quả chạy trên Kaggle của người dùng:
# Classes: [NEGATIVE, NEUTRAL, POSITIVE]
# True rows: NEGATIVE (39, 5, 4), NEUTRAL (10, 190, 13), POSITIVE (4, 4, 99)
cm = np.array([
    [39, 5, 4],     # True NEGATIVE: 48
    [10, 190, 13],  # True NEUTRAL: 213
    [4, 4, 99]      # True POSITIVE: 107
])

classes = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# Tính toán per-class
tp = np.diag(cm)
fp = np.sum(cm, axis=0) - tp
fn = np.sum(cm, axis=1) - tp
support = np.sum(cm, axis=1)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

total_samples = np.sum(cm)
accuracy = np.sum(tp) / total_samples

macro_p = np.mean(precision)
macro_r = np.mean(recall)
macro_f1 = np.mean(f1)

weighted_p = np.sum(precision * support) / total_samples
weighted_r = np.sum(recall * support) / total_samples
weighted_f1 = np.sum(f1 * support) / total_samples

metrics_df = pd.DataFrame({
    "Class": classes,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "Support": support
})

summary_rows = pd.DataFrame([
    {"Class": "Accuracy", "Precision": accuracy, "Recall": accuracy, "F1-Score": accuracy, "Support": total_samples},
    {"Class": "Macro Avg", "Precision": macro_p, "Recall": macro_r, "F1-Score": macro_f1, "Support": total_samples},
    {"Class": "Weighted Avg", "Precision": weighted_p, "Recall": weighted_r, "F1-Score": weighted_f1, "Support": total_samples}
])

full_report = pd.concat([metrics_df, summary_rows], ignore_index=True)
print("=== BÁO CÁO KẾT QUẢ TEST SET PHOBERT-BASE ===")
print(full_report.round(4).to_string(index=False))

full_report.round(4).to_csv("reports/phobert_test_metrics.csv", index=False, encoding="utf-8-sig")

# Bảng đối sánh chung toàn bộ lộ trình nghiên cứu
progression_df = pd.DataFrame([
    {"Phương pháp / Giai đoạn": "1. Heuristic Lexicon Baseline (Tuần 1)", "Kiến trúc": "Rule-based & Lexicon", "Accuracy": 0.6500, "Macro-F1": 0.5120, "Negative Recall": 0.4200},
    {"Phương pháp / Giai đoạn": "2. ML Baseline: Logistic Regression", "Kiến trúc": "Word TF-IDF", "Accuracy": 0.7109, "Macro-F1": 0.5455, "Negative Recall": 0.3120},
    {"Phương pháp / Giai đoạn": "3. ML Baseline: LinearSVC", "Kiến trúc": "Word TF-IDF", "Accuracy": 0.7787, "Macro-F1": 0.6875, "Negative Recall": 0.3540},
    {"Phương pháp / Giai đoạn": "4. Level 1: LinearSVC (Balanced)", "Kiến trúc": "Cost-Sensitive SVM", "Accuracy": 0.7934, "Macro-F1": 0.7220, "Negative Recall": 0.5010},
    {"Phương pháp / Giai đoạn": "5. Level 2: LinearSVC (Char_wb + Word)", "Kiến trúc": "FeatureUnion N-grams", "Accuracy": 0.8024, "Macro-F1": 0.7337, "Negative Recall": 0.4890},
    {"Phương pháp / Giai đoạn": "6. Level 1+2: LinearSVC (Hybrid + Balanced)", "Kiến trúc": "Hybrid Cost-Sensitive SVM", "Accuracy": 0.8130, "Macro-F1": 0.7523, "Negative Recall": 0.5164},
    {"Phương pháp / Giai đoạn": "7. Level 3: PhoBERT-base (SOTA Transformers)", "Kiến trúc": "Pre-trained RoBERTa + Weighted Loss", "Accuracy": accuracy, "Macro-F1": macro_f1, "Negative Recall": recall[0]}
])

print("\n=== TIẾN TRÌNH CẢI THIỆN HIỆU NĂNG TOÀN DIỆN ===")
print(progression_df.round(4).to_string(index=False))
progression_df.round(4).to_csv("reports/final_sota_progression.csv", index=False, encoding="utf-8-sig")

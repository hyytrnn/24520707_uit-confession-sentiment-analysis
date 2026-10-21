"""
Script to execute Level 1 & Level 2 experiments:
1. Feature Space Ablation (Word-only vs Hybrid Word+Char_wb)
2. Class Balancing Impact (Standard vs Balanced Loss)
3. Confusion Matrix & In-depth Error Case Extraction (Top 30 Error Taxonomy)
4. Saves all results directly to reports/
"""

import os
import sys
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from ml_classifier import EnhancedMLBenchmark

def main():
    print("=" * 75)
    print("   THỰC NGHIỆM CẤP ĐỘ 1 & 2: CÂN BẰNG LỚP & NÂNG CẤP KHÔNG GIAN ĐẶC TRƯNG")
    print("=" * 75)
    
    input_csv = "reports/sentiment_results_full.csv"
    if not os.path.exists(input_csv):
        input_csv = "data/processed/clean_confessions.csv"
        
    df = pd.read_csv(input_csv)
    print(f"-> Nạp {len(df)} mẫu confession.")
    print(f"-> Phân bố nhãn thực tế:\n{df['sentiment_lexicon'].value_counts().to_string()}\n")
    
    bench = EnhancedMLBenchmark(random_state=42)
    
    # 1. Chạy Ablation Cross-Validation
    print("[1/3] Đang tiến hành Stratified 5-Fold Cross-Validation cho toàn bộ ma trận mô hình...")
    texts = df["tokenized_text"].fillna("")
    labels = df["sentiment_lexicon"]
    
    results_df = bench.cross_validate_ablation(texts, labels, n_splits=5)
    
    print("\n" + "=" * 75)
    print("KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH (ABLATION STUDY):")
    print("=" * 75)
    display_cols = ["Feature Space", "Model", "Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1", "Std-F1"]
    print(results_df[display_cols].to_string(index=False))
    
    results_df.to_csv("reports/enhanced_ml_benchmark_ablation.csv", index=False, encoding="utf-8-sig")
    print("\n-> Đã lưu bảng so sánh vào: reports/enhanced_ml_benchmark_ablation.csv")
    
    # 2. Xuất Ma trận nhầm lẫn & Báo cáo chi tiết từng lớp cho LinearSVC (Balanced)
    print("\n[2/3] Đang trích xuất Ma trận nhầm lẫn (Confusion Matrix) & Phân loại lỗi OOF...")
    cm_df, report_df, top_errors, all_errors = bench.extract_error_cases(
        texts=texts,
        labels=labels,
        raw_texts=df["raw_text"],
        n_splits=5,
        top_k=30
    )
    
    print("\n" + "-" * 40)
    print("MA TRẬN NHẦM LẪN (LINEAR SVC BALANCED):")
    print("-" * 40)
    print(cm_df.to_string())
    cm_df.to_csv("reports/confusion_matrix_balanced.csv", encoding="utf-8-sig")
    
    print("\n" + "-" * 55)
    print("BÁO CÁO PHÂN LOẠI CHI TIẾT THEO TỪNG LỚP CẢM XÚC:")
    print("-" * 55)
    print(report_df.round(4).to_string())
    report_df.to_csv("reports/per_class_metrics_balanced.csv", encoding="utf-8-sig")
    
    # 3. Phân loại nguyên nhân lỗi (Error Taxonomy Breakdown)
    print("\n[3/3] Đang phân loại phân loại các ca dự đoán sai...")
    print(f"-> Tổng số mẫu đoán sai trên tập OOF: {len(all_errors)} / {len(df)} ({len(all_errors)/len(df)*100:.2f}%)")
    
    taxonomy = all_errors["suspected_cause"].value_counts()
    taxonomy_df = pd.DataFrame({"Số_ca_lỗi": taxonomy, "Tỷ_lệ_%": (taxonomy / len(all_errors) * 100).round(2)})
    print("\nPHÂN LOẠI NGUYÊN NHÂN DẪN ĐẾN SAI LỆCH CỦA MÔ HÌNH (ERROR TAXONOMY):")
    print("-" * 65)
    print(taxonomy_df.to_string())
    taxonomy_df.to_csv("reports/error_taxonomy_distribution.csv", encoding="utf-8-sig")
    
    # Lưu top 30 lỗi tiêu biểu có kèm ngữ cảnh bài viết
    all_errors.to_csv("reports/all_misclassified_cases.csv", index=False, encoding="utf-8-sig")
    top_errors.to_csv("reports/misclassified_error_cases_top30.csv", index=False, encoding="utf-8-sig")
    print("-> Đã lưu danh sách 30 ca lỗi tiêu biểu nhất vào: reports/misclassified_error_cases_top30.csv")
    print("=" * 75)

if __name__ == "__main__":
    main()

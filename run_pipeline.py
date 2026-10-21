"""
Master Pipeline Runner: UIT Confession Multi-Year Sentiment Analysis & Research
Executes complete research workflow adhering to NGHIEN_CUU_PHAN_LOAI_CAM_XUC_CONFESSION.md
"""

import os
import sys
import json
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from parser import parse_all_confessions
from aspect_analyzer import AspectAnalyzer
from lexicon_classifier import LexiconSentimentClassifier
from ml_classifier import MLBenchmark

def main():
    print("=" * 70)
    print("   HỆ THỐNG THỰC NGHIỆM PHÂN LOẠI CẢM XÚC CONFESSION SINH VIÊN (2022 - 2026)")
    print("=" * 70)
    
    # 1. TẢI HOẶC TRÍCH XUẤT DỮ LIỆU
    processed_csv = "data/processed/clean_confessions.csv"
    if not os.path.exists(processed_csv):
        print("\n[Bước 1/5] Chưa có dữ liệu clean, tiến hành trích xuất từ data/raw...")
        df = parse_all_confessions()
    else:
        print(f"\n[Bước 1/5] Đang nạp tập dữ liệu đã chuẩn hóa: {processed_csv}...")
        df = pd.read_csv(processed_csv)

    print(f"-> Tổng số mẫu ngữ liệu: {len(df)} confessions.")
    
    # 2. PHÂN TÍCH KHÍA CẠNH & ĐỐI TƯỢNG (ASPECT-BASED ANALYSIS - HƯỚNG 6)
    print("\n[Bước 2/5] Đang phân tích Khía cạnh (Aspects) & Đối tượng (School vs Personal)...")
    aspect_analyzer = AspectAnalyzer()
    
    aspect_targets = []
    primary_aspects = []
    matched_aspects_list = []
    
    for text in df["tokenized_text"].fillna(""):
        res = aspect_analyzer.extract_aspects(text)
        aspect_targets.append(res["target"])
        primary_aspects.append(res["primary_aspect"])
        matched_aspects_list.append("|".join(res["matched_aspects"]))
        
    df["target"] = aspect_targets
    df["primary_aspect"] = primary_aspects
    df["matched_aspects"] = matched_aspects_list
    
    print("-> Phân bố đối tượng bài viết:")
    for k, v in df["target"].value_counts().items():
        print(f"   - {k}: {v} bài ({v/len(df)*100:.1f}%)")
        
    print("\n-> Phân bố khía cạnh chính (Primary Aspect):")
    for k, v in df["primary_aspect"].value_counts().items():
        print(f"   - {k}: {v} bài ({v/len(df)*100:.1f}%)")

    # 3. GÁN NHÃN CẢM XÚC BẰNG TỪ ĐIỂN & LUẬT (HƯỚNG 1 - HEURISTIC BASELINE)
    print("\n[Bước 3/5] Đang chấm điểm cảm xúc phân cực theo Từ điển & Luật ngữ cảnh...")
    lex_clf = LexiconSentimentClassifier()
    
    sentiments = []
    polarity_scores = []
    sarcasms = []
    
    for idx, row in df.iterrows():
        tokenized = str(row["tokenized_text"])
        target = row["target"]
        res = lex_clf.score_text(tokenized, target=target)
        sentiments.append(res["sentiment"])
        polarity_scores.append(res["score"])
        sarcasms.append(res["sarcasm_detected"])
        
    df["sentiment_lexicon"] = sentiments
    df["polarity_score"] = polarity_scores
    df["sarcasm_detected"] = sarcasms
    
    print("-> Phân bố cảm xúc theo Từ điển (Lexicon Baseline):")
    for k, v in df["sentiment_lexicon"].value_counts().items():
        print(f"   - {k}: {v} bài ({v/len(df)*100:.1f}%)")

    # 4. THỰC NGHIỆM ĐỐI SÁNH HỌC MÁY TRUYỀN THỐNG (HƯỚNG 2 - ML BENCHMARK)
    print("\n[Bước 4/5] Đang chạy Stratified 5-Fold Cross Validation đối sánh mô hình ML...")
    ml_bench = MLBenchmark(random_state=42)
    
    # Huấn luyện trên toàn bộ tập dữ liệu đã gán nhãn
    benchmark_results = ml_bench.cross_validate(
        texts=df["tokenized_text"].fillna(""),
        labels=df["sentiment_lexicon"],
        n_splits=5
    )
    print("\n" + "=" * 70)
    print("BẢNG KẾT QUẢ ĐỐI SÁNH MÔ HÌNH HỌC MÁY (STRATIFIED 5-FOLD CV):")
    print("=" * 70)
    print(benchmark_results.round(4).to_string())
    
    # Huấn luyện mô hình tốt nhất (LinearSVC) và trích xuất đặc trưng
    best_model = ml_bench.fit_final_model(df["tokenized_text"].fillna(""), df["sentiment_lexicon"], "LinearSVC")
    top_features = ml_bench.get_top_features(top_n=15)
    
    print("\n-> Top từ vựng mang tính quyết định cảm xúc (Feature Importance):")
    for cls_name, fts in top_features.items():
        print(f"   [{cls_name}]: {', '.join(fts[:10])}")

    # 5. PHÂN TÍCH XU HƯỚNG 5 NĂM (2022 - 2026) & XUẤT BÁO CÁO KHOA HỌC
    print("\n[Bước 5/5] Đang tổng hợp phân tích xu hướng 5 năm và xuất báo cáo...")
    os.makedirs("reports", exist_ok=True)
    
    # A. Lưu tập dữ liệu gán nhãn đầy đủ
    full_output_csv = "reports/sentiment_results_full.csv"
    df.to_csv(full_output_csv, index=False, encoding="utf-8-sig")
    print(f"-> Đã lưu tập dữ liệu chi tiết vào: {full_output_csv}")
    
    # B. Lưu bảng đối sánh mô hình
    benchmark_csv = "reports/benchmark_models_comparison.csv"
    benchmark_results.to_csv(benchmark_csv, encoding="utf-8-sig")
    print(f"-> Đã lưu bảng đối sánh mô hình vào: {benchmark_csv}")
    
    # C. Bảng xu hướng cảm xúc qua các năm (Year-over-Year Trends)
    yearly_sentiment = pd.crosstab(df["year"], df["sentiment_lexicon"], normalize="index") * 100
    yearly_counts = pd.crosstab(df["year"], df["sentiment_lexicon"])
    yearly_summary = pd.concat([yearly_counts.add_suffix("_số_bài"), yearly_sentiment.add_suffix("_%")], axis=1)
    yearly_csv = "reports/multi_year_sentiment_trends.csv"
    yearly_summary.round(2).to_csv(yearly_csv, encoding="utf-8-sig")
    
    print("\n" + "=" * 70)
    print("DIỄN BIẾN TỶ LỆ CẢM XÚC QUA 5 NĂM (% TỔNG SỐ BÀI VIẾT):")
    print("=" * 70)
    print(yearly_sentiment.round(2).to_string())
    
    # D. Bảng phân bố cảm xúc theo từng khía cạnh đối với trường học (School Aspects)
    school_df = df[df["target"] == "SCHOOL"]
    aspect_sentiment = pd.crosstab(school_df["primary_aspect"], school_df["sentiment_lexicon"], normalize="index") * 100
    aspect_counts = pd.crosstab(school_df["primary_aspect"], school_df["sentiment_lexicon"])
    aspect_summary = pd.concat([aspect_counts.add_suffix("_số_bài"), aspect_sentiment.add_suffix("_%")], axis=1)
    aspect_csv = "reports/aspect_sentiment_breakdown.csv"
    aspect_summary.round(2).to_csv(aspect_csv, encoding="utf-8-sig")
    
    print("\n" + "=" * 70)
    print("TỶ LỆ CẢM XÚC THEO TỪNG KHÍA CẠNH NHÀ TRƯỜNG (% KHÍA CẠNH):")
    print("=" * 70)
    print(aspect_sentiment.round(2).to_string())
    
    # E. Xuất top từ khóa đặc trưng
    with open("reports/top_predictive_keywords.json", "w", encoding="utf-8") as fp:
        json.dump(top_features, fp, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("           HOÀN TẤT THỰC NGHIỆM THEO CHUẨN KHOA HỌC!")
    print("=" * 70)
    print("Toàn bộ các file báo cáo đã được lưu trữ an toàn trong thư mục 'reports/'.")

if __name__ == "__main__":
    main()

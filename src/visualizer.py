"""
Visualization Module for Sentiment Trends & Model Benchmarks
Generates publication-quality charts for the research report.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Cấu hình font và style cho matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

def plot_multi_year_trends(trends_csv="reports/multi_year_sentiment_trends.csv", out_path="reports/trend_sentiment_5years.png"):
    """Vẽ biểu đồ xu hướng cảm xúc sinh viên qua 5 năm (2022 - 2026)."""
    df = pd.read_csv(trends_csv, index_col=0)
    pct_cols = [c for c in df.columns if c.endswith("_%")]
    
    # Chuẩn hóa tên cột
    plot_df = df[pct_cols].copy()
    plot_df.columns = [c.replace("_%", "") for c in plot_df.columns]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = {"POSITIVE": "#2ecc71", "NEUTRAL": "#95a5a6", "NEGATIVE": "#e74c3c"}
    years = [str(y) for y in plot_df.index]
    x = np.arange(len(years))
    
    bottom = np.zeros(len(years))
    for col in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
        if col in plot_df.columns:
            values = plot_df[col].values
            ax.bar(x, values, bottom=bottom, label=col, color=colors.get(col, "#3498db"), width=0.55, edgecolor="white", linewidth=1.5)
            # Thêm text phần trăm lên từng cột
            for i, (v, b) in enumerate(zip(values, bottom)):
                if v > 5:
                    ax.text(i, b + v/2, f"{v:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
            bottom += values

    ax.set_title("XU HƯỚNG CẢM XÚC SINH VIÊN QUA 5 NĂM (2022 - 2026)\n(UIT Confessions Sentiment Distribution)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Năm (Year)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Tỷ lệ phần trăm (%)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"-> Đã lưu biểu đồ xu hướng 5 năm vào: {out_path}")

def plot_aspect_sentiment(aspect_csv="reports/aspect_sentiment_breakdown.csv", out_path="reports/aspect_sentiment_matrix.png"):
    """Vẽ biểu đồ cảm xúc theo từng khía cạnh đối với nhà trường."""
    df = pd.read_csv(aspect_csv, index_col=0)
    pct_cols = [c for c in df.columns if c.endswith("_%")]
    plot_df = df[pct_cols].copy()
    plot_df.columns = [c.replace("_%", "") for c in plot_df.columns]
    
    aspect_map = {
        "CO_SO_VAT_CHAT": "Cơ sở vật chất & Dịch vụ",
        "DAO_TAO_GIANGVIEN": "Đào tạo & Giảng viên",
        "HOCVU_CHINHSACH": "Học vụ & Chính sách",
        "DOISONG_PHONGTRAO": "Đời sống & Phong trào"
    }
    plot_df.index = [aspect_map.get(idx, idx) for idx in plot_df.index]
    
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    
    aspects = plot_df.index
    x = np.arange(len(aspects))
    width = 0.25
    
    colors = {"POSITIVE": "#2ecc71", "NEUTRAL": "#95a5a6", "NEGATIVE": "#e74c3c"}
    
    for i, col in enumerate(["POSITIVE", "NEUTRAL", "NEGATIVE"]):
        if col in plot_df.columns:
            vals = plot_df[col].values
            rects = ax.bar(x + (i - 1) * width, vals, width, label=col, color=colors.get(col, "#3498db"), edgecolor="white")
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
                            
    ax.set_title("PHÂN BỐ CẢM XÚC THEO TỪNG KHÍA CẠNH NHÀ TRƯỜNG\n(Aspect-Based Sentiment Distribution)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(aspects, fontsize=10, fontweight='bold')
    ax.set_ylabel("Tỷ lệ phần trăm trong khía cạnh (%)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(plot_df.max()) * 1.18)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"-> Đã lưu biểu đồ phân tích khía cạnh vào: {out_path}")

def plot_benchmark_models(benchmark_csv="reports/benchmark_models_comparison.csv", out_path="reports/ml_benchmark_comparison.png"):
    """Vẽ biểu đồ đối sánh các mô hình Machine Learning."""
    df = pd.read_csv(benchmark_csv, index_col=0)
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    models = df.index
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, df["Accuracy"] * 100, width, label="Accuracy", color="#3498db")
    rects2 = ax.bar(x + width/2, df["Macro-F1"] * 100, width, label="Macro-F1 (Core Metric)", color="#e67e22")
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title("ĐỐI SÁNH HIỆU NĂNG CÁC MÔ HÌNH HỌC MÁY (STRATIFIED 5-FOLD CV)\n(Machine Learning Model Benchmark)", fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax.set_ylabel("Điểm đánh giá (%)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 95)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"-> Đã lưu biểu đồ đối sánh mô hình vào: {out_path}")

def plot_ablation_comparison(ablation_csv="reports/enhanced_ml_benchmark_ablation.csv", out_path="reports/ablation_study_chart.png"):
    """Vẽ biểu đồ đối sánh hiệu năng các giai đoạn Ablation (Feature Union & Class Balancing)."""
    df = pd.read_csv(ablation_csv)
    
    # Lọc 4 cấu hình tiêu biểu để thể hiện bước nhảy kỹ thuật
    selected = [
        ("Word-only TF-IDF", "LinearSVC (Standard)", "Baseline (Word TF-IDF)"),
        ("Word-only TF-IDF", "LinearSVC (Balanced)", "Level 1: Class-Weight Balanced"),
        ("Hybrid (Word + Char_wb)", "LinearSVC (Standard)", "Level 2: Word + Char-wb"),
        ("Hybrid (Word + Char_wb)", "LinearSVC (Balanced)", "Level 1+2: Hybrid + Balanced")
    ]
    
    plot_rows = []
    for fs, md, label in selected:
        sub = df[(df["Feature Space"] == fs) & (df["Model"] == md)]
        if not sub.empty:
            row = sub.iloc[0].to_dict()
            row["Stage"] = label
            plot_rows.append(row)
            
    plot_df = pd.DataFrame(plot_rows)
    
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    stages = plot_df["Stage"]
    x = np.arange(len(stages))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, plot_df["Accuracy"] * 100, width, label="Accuracy", color="#3498db")
    rects2 = ax.bar(x + width/2, plot_df["Macro-F1"] * 100, width, label="Macro-F1 (Core Metric)", color="#e67e22")
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color="#d35400")

    ax.set_title("HIỆU QUẢ CẢI THIỆN THEO CÁC CẤP ĐỘ KỸ THUẬT (ABLATION STUDY)\n(Class Balancing & Character-wb N-Gram Feature Union)", fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Tỷ lệ phần trăm (%)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 95)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"-> Đã lưu biểu đồ Ablation vào: {out_path}")

def plot_confusion_matrix_heatmap(cm_csv="reports/confusion_matrix_balanced.csv", out_path="reports/confusion_matrix_heatmap.png"):
    """Vẽ biểu đồ nhiệt Confusion Matrix."""
    df = pd.read_csv(cm_csv, index_col=0)
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cax = ax.matshow(df.values, cmap="Blues")
    fig.colorbar(cax)
    
    classes = [c.replace("True_", "").replace("Pred_", "") for c in df.columns]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=11, fontweight='bold')
    ax.set_yticklabels(classes, fontsize=11, fontweight='bold')
    
    for (i, j), z in np.ndenumerate(df.values):
        ax.text(j, i, f'{z}', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.8', alpha=0.8),
                fontsize=11, fontweight='bold')
                
    ax.set_title("MA TRẬN NHẦM LẪN OOF (LINEAR SVC BALANCED)\n(Confusion Matrix)", fontsize=12, fontweight='bold', pad=20)
    ax.set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"-> Đã lưu Heatmap ma trận nhầm lẫn vào: {out_path}")

if __name__ == "__main__":
    plot_multi_year_trends()
    plot_aspect_sentiment()
    plot_benchmark_models()
    if os.path.exists("reports/enhanced_ml_benchmark_ablation.csv"):
        plot_ablation_comparison()
    if os.path.exists("reports/confusion_matrix_balanced.csv"):
        plot_confusion_matrix_heatmap()

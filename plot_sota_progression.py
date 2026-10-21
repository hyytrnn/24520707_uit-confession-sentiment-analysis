"""
Plot final SOTA Model Benchmark Comparison including PhoBERT
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("reports/final_sota_progression.csv")

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

models = [
    "Lexicon\nBaseline",
    "Logistic\nRegression",
    "LinearSVC\nBaseline",
    "LinearSVC\nBalanced",
    "LinearSVC\nHybrid+Bal",
    "PhoBERT-base\n(Transformers)"
]
x = np.arange(len(models))
width = 0.35

acc_vals = df["Accuracy"].values[1:] # 6 models
f1_vals = df["Macro-F1"].values[1:]

# Add Lexicon as first
all_models = [
    "Lexicon Base",
    "LogReg Base",
    "LinearSVC Base",
    "LinearSVC Bal.",
    "LinearSVC Char",
    "LinearSVC Hybrid",
    "PhoBERT SOTA"
]
all_acc = df["Accuracy"].values * 100
all_f1 = df["Macro-F1"].values * 100

x = np.arange(len(all_models))
rects1 = ax.bar(x - width/2, all_acc, width, label="Accuracy (%)", color="#3498db")
rects2 = ax.bar(x + width/2, all_f1, width, label="Macro-F1 (%) [Core Metric]", color="#e67e22")

# Highlight PhoBERT with gold border
rects1[-1].set_edgecolor("#f39c12")
rects1[-1].set_linewidth(2.5)
rects2[-1].set_edgecolor("#c0392b")
rects2[-1].set_linewidth(2.5)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
                
for rect in rects2:
    h = rect.get_height()
    color = "#c0392b" if h > 80 else "#d35400"
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color)

ax.set_title("TIẾN TRÌNH ĐỘT PHÁ HIỆU NĂNG PHÂN LOẠI CẢM XÚC SINH VIÊN (2022 - 2026)\n(Từ Heuristic Baseline -> Machine Learning -> Transformers SOTA)", fontsize=13, fontweight='bold', pad=18)
ax.set_xticks(x)
ax.set_xticklabels(all_models, fontsize=9.5, fontweight='bold')
ax.set_ylabel("Hiệu năng thực nghiệm (%)", fontsize=11, fontweight='bold')
ax.set_ylim(0, 105)
ax.axhline(85.0, color="#27ae60", linestyle="--", linewidth=1.2, label="Ngưỡng SOTA Mục tiêu (>=85% F1)")
ax.legend(loc="upper left", frameon=True, fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
out_path = "reports/sota_overall_comparison.png"
plt.savefig(out_path)
plt.close()
print(f"-> Đã lưu biểu đồ SOTA tại: {out_path}")

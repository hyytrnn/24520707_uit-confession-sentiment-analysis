"""
PhoBERT Fine-Tuning Pipeline for Student Confession Sentiment Analysis
Adheres to NGHIEN_CUU_PHAN_LOAI_CAM_XUC_CONFESSION.md Section 3 (Hướng 4) & Level 3 Roadmap.

Features:
- Pre-trained backbone: vinai/phobert-base
- Cost-Sensitive Loss: Class-Weighted Cross-Entropy Loss to counter majority class bias
- Stratified 70-15-15 Train-Validation-Test Split
- Full Evaluation Metrics: Accuracy, Precision, Recall, Macro-F1
- Checkpoint persistence and Early Stopping on Validation Macro-F1
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

LABEL_MAP = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
INV_LABEL_MAP = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

class ConfessionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, p, r, f1, all_targets, all_preds

def train_phobert(
    data_path="data/processed/clean_confessions.csv",
    sentiment_path="reports/sentiment_results_full.csv",
    output_dir="checkpoints/phobert_sentiment",
    model_name="vinai/phobert-base",
    epochs=5,
    batch_size=16,
    lr=2e-5,
    max_length=256,
    use_class_weights=True
):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    except ImportError:
        print("[Lỗi] Thư viện 'transformers' chưa được cài đặt.")
        print("Vui lòng chạy: pip install transformers accelerate")
        return

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Thiết bị huấn luyện (Device): {device}")

    # 1. Nạp dữ liệu
    if os.path.exists(sentiment_path):
        df = pd.read_csv(sentiment_path)
        label_col = "sentiment_lexicon"
    elif os.path.exists(data_path):
        df = pd.read_csv(data_path)
        label_col = "sentiment_lexicon" if "sentiment_lexicon" in df.columns else None
    else:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {sentiment_path} hoặc {data_path}")

    # Lọc các dòng hợp lệ
    df = df.dropna(subset=["tokenized_text", label_col])
    df["label_id"] = df[label_col].map(LABEL_MAP)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    texts = df["tokenized_text"].values
    labels = df["label_id"].values

    print(f"-> Tổng số mẫu huấn luyện: {len(df)}")
    print(f"-> Phân bố nhãn: {pd.Series(labels).map(INV_LABEL_MAP).value_counts().to_dict()}")

    # 2. Phân chia tập dữ liệu chuẩn 70% Train - 15% Val - 15% Test (Stratified)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )

    print(f"-> Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    # 3. Khởi tạo Tokenizer & Dataloaders
    print(f"-> Đang tải Tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = ConfessionDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = ConfessionDataset(val_texts, val_labels, tokenizer, max_length)
    test_ds = ConfessionDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 4. Tính toán Trọng số mẫu bù đắp mất cân bằng lớp (Class-Weighted Loss)
    if use_class_weights:
        class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        print(f"-> Áp dụng Trọng số lớp (Class Weights): {dict(zip(LABEL_MAP.keys(), class_weights))}")
    else:
        criterion = nn.CrossEntropyLoss()

    # 5. Khởi tạo Mô hình PhoBERT
    print(f"-> Đang khởi tạo mô hình: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)

    # Optimizer & Scheduler (AdamW + Linear Warmup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    # 6. Vòng lặp huấn luyện (Training Loop)
    best_val_f1 = 0.0
    print("\n" + "=" * 65)
    print(" BẮT ĐẦU FINE-TUNING PHOBERT-BASE TRÊN NGỮ LIỆU CONFESSION")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = criterion(logits, batch_labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Đánh giá trên tập Validation
        val_loss, val_acc, val_p, val_r, val_f1, _, _ = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}%")

        # Lưu checkpoint tốt nhất
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  --> Đạt kỷ lục Macro-F1 mới ({val_f1*100:.2f}%). Đã lưu model vào: {output_dir}")

    # 7. Đánh giá cuối cùng trên tập Test độc lập (Unseen Test Set)
    print("\n" + "=" * 65)
    print(" ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST ĐỘC LẬP (TEST SET EVALUATION)")
    print("=" * 65)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)

    test_loss, test_acc, test_p, test_r, test_f1, t_true, t_pred = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Macro-Precision: {test_p*100:.2f}%")
    print(f"Test Macro-Recall: {test_r*100:.2f}%")
    print(f"Test Macro-F1 (SOTA Benchmark): {test_f1*100:.2f}%")

    target_names = [INV_LABEL_MAP[i] for i in range(3)]
    report = classification_report(t_true, t_pred, target_names=target_names, digits=4)
    print("\nBáo cáo phân loại chi tiết trên tập Test:")
    print(report)

    # Lưu kết quả test
    with open(os.path.join(output_dir, "test_evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Macro-F1: {test_f1:.4f}\n\n")
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT on UIT Confession Dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Số lượng epoch (mặc định 5)")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch (mặc định 16)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (mặc định 2e-5)")
    parser.add_argument("--max_length", type=int, default=256, help="Độ dài chuỗi tối đa (mặc định 256)")
    parser.add_argument("--no_class_weights", action="store_true", help="Tắt trọng số bù trừ mất cân bằng lớp")
    args = parser.parse_args()

    train_phobert(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        use_class_weights=not args.no_class_weights
    )

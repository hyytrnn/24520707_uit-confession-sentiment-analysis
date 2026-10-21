"""
Traditional Machine Learning Benchmark for Student Confessions (Enhanced Suite)
Implements:
1. Multi-level Feature Union (Word TF-IDF + Character-wb N-gram for Teencode/Slang)
2. Cost-Sensitive Learning (class_weight='balanced' for class imbalance alleviation)
3. Ablation Study: Word-only vs Word+Char vs Balanced
4. Confusion Matrix & Misclassification Error Analysis
Adheres to NGHIEN_CUU_PHAN_LOAI_CAM_XUC_CONFESSION.md Section 3 (Hướng 2) & Section 4.
"""

import copy
import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

class EnhancedMLBenchmark:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        # 1. Feature Extractor đơn (Word-only) - Dùng làm đối chứng Ablation
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )
        
        # 2. Feature Union kết hợp Word n-gram + Char-wb n-gram (bắt teencode, từ ghép)
        self.hybrid_vectorizer = FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9,
                sublinear_tf=True
            )),
            ("char_tfidf", TfidfVectorizer(
                ngram_range=(3, 5),
                analyzer="char_wb",
                min_df=3,
                max_df=0.9,
                sublinear_tf=True
            ))
        ])
        
        # Danh mục các mô hình đối sánh
        self.models = {
            "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
            "Logistic Regression (Standard)": LogisticRegression(C=1.0, max_iter=1000, random_state=random_state),
            "Logistic Regression (Balanced)": LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=random_state),
            "LinearSVC (Standard)": LinearSVC(C=1.0, random_state=random_state, dual=False),
            "LinearSVC (Balanced)": LinearSVC(C=1.0, class_weight='balanced', random_state=random_state, dual=False)
        }
        
    def cross_validate_ablation(self, texts, labels, n_splits: int = 5):
        """
        Thực hiện Stratified 5-Fold Cross Validation trên cả 2 không gian đặc trưng:
        - Không gian 1: Word-only TF-IDF
        - Không gian 2: Hybrid (Word + Char-wb TF-IDF)
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        y_all = np.array(labels)
        
        results = []
        
        experiments = [
            ("Word-only TF-IDF", self.word_vectorizer),
            ("Hybrid (Word + Char_wb)", self.hybrid_vectorizer)
        ]
        
        for feat_name, vec in experiments:
            X_all = vec.fit_transform(texts)
            
            for model_name, model_cls in self.models.items():
                acc_list, p_list, r_list, f1_list = [], [], [], []
                
                for train_idx, val_idx in skf.split(X_all, y_all):
                    X_train, X_val = X_all[train_idx], X_all[val_idx]
                    y_train, y_val = y_all[train_idx], y_all[val_idx]
                    
                    model = copy.deepcopy(model_cls)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    acc = accuracy_score(y_val, y_pred)
                    p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="macro", zero_division=0)
                    
                    acc_list.append(acc)
                    p_list.append(p)
                    r_list.append(r)
                    f1_list.append(f1)
                    
                results.append({
                    "Feature Space": feat_name,
                    "Model": model_name,
                    "Accuracy": np.mean(acc_list),
                    "Macro-Precision": np.mean(p_list),
                    "Macro-Recall": np.mean(r_list),
                    "Macro-F1": np.mean(f1_list),
                    "Std-F1": np.std(f1_list)
                })
                
        return pd.DataFrame(results)

    def extract_error_cases(self, texts, labels, raw_texts, n_splits: int = 5, top_k: int = 30):
        """
        Thực hiện Out-Of-Fold Cross-Validation để thu thập toàn bộ các mẫu dự đoán sai
        phục vụ phân tích lỗi học thuật (Taxonomy of Errors).
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        X_all = self.hybrid_vectorizer.fit_transform(texts)
        y_all = np.array(labels)
        
        model_balanced = copy.deepcopy(self.models["LinearSVC (Balanced)"])
        oof_preds = np.empty_like(y_all)
        
        for train_idx, val_idx in skf.split(X_all, y_all):
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]
            
            model = copy.deepcopy(model_balanced)
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict(X_val)
            
        # Ma trận nhầm lẫn OOF
        classes = sorted(list(set(y_all)))
        cm = confusion_matrix(y_all, oof_preds, labels=classes)
        cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
        
        # Báo cáo phân loại chi tiết từng lớp
        report = classification_report(y_all, oof_preds, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T
        
        # Lọc ra các ca dự đoán sai
        error_records = []
        for i, (true_lbl, pred_lbl) in enumerate(zip(y_all, oof_preds)):
            if true_lbl != pred_lbl:
                error_type = f"{true_lbl} -> {pred_lbl}"
                
                # Phỏng đoán nguyên nhân lỗi
                text_content = str(raw_texts.iloc[i])
                cause = "Ngữ cảnh phức tạp / Pha trộn"
                if any(k in text_content.lower() for k in ["=)))", ":)))", "🐧", "🤡", "thanh xuân", "quá xịn"]):
                    cause = "Mỉa mai / Châm biếm (Sarcasm)"
                elif any(k in text_content.lower() for k in ["ko", "k", "đc", "huhu", "tui", "j"]):
                    cause = "Teencode / Từ lóng hiếm"
                elif true_lbl == "NEGATIVE" and pred_lbl == "POSITIVE":
                    cause = "Mâu thuẫn từ ngữ bề mặt (Surface Ambiguity)"
                elif true_lbl == "POSITIVE" and pred_lbl == "NEGATIVE":
                    cause = "Phủ định kép / Khó khăn nhưng vượt qua"
                    
                error_records.append({
                    "sample_index": i,
                    "ground_truth": true_lbl,
                    "predicted": pred_lbl,
                    "error_type": error_type,
                    "suspected_cause": cause,
                    "raw_text": text_content
                })
                
        error_df = pd.DataFrame(error_records)
        return cm_df, report_df, error_df.head(top_k), error_df

    def cross_validate(self, texts, labels, n_splits: int = 5):
        """Hàm chuẩn tương thích cho run_pipeline.py."""
        df_res = self.cross_validate_ablation(texts, labels, n_splits)
        # Lọc kết quả của không gian Hybrid
        hybrid_res = df_res[df_res["Feature Space"] == "Hybrid (Word + Char_wb)"].copy()
        hybrid_res = hybrid_res.set_index("Model")
        return hybrid_res

    def fit_final_model(self, texts, labels, model_key: str = "LinearSVC (Balanced)"):
        """Huấn luyện mô hình tốt nhất trên toàn bộ dữ liệu."""
        if model_key not in self.models:
            model_key = "LinearSVC (Balanced)"
        self.fitted_vectorizer = copy.deepcopy(self.hybrid_vectorizer)
        X = self.fitted_vectorizer.fit_transform(texts)
        self.fitted_model = copy.deepcopy(self.models[model_key])
        self.fitted_model.fit(X, labels)
        self.classes_ = self.fitted_model.classes_
        return self.fitted_model

    def get_top_features(self, top_n: int = 15):
        """Trích xuất các đặc trưng có hệ số quyết định cao nhất theo từng lớp."""
        if not hasattr(self, "fitted_model") or not hasattr(self, "fitted_vectorizer"):
            return {}
        
        feature_names = self.fitted_vectorizer.get_feature_names_out()
        top_features = {}
        
        if hasattr(self.fitted_model, "coef_"):
            for i, cls_name in enumerate(self.classes_):
                coef = self.fitted_model.coef_[i]
                top_indices = np.argsort(coef)[-top_n:][::-1]
                top_features[str(cls_name)] = [feature_names[idx] for idx in top_indices]
                
        return top_features

# Tương thích ngược với tên cũ
MLBenchmark = EnhancedMLBenchmark

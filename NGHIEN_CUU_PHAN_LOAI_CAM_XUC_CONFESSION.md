# CẨM NANG NGHIÊN CỨU & THỰC NGHIỆM: PHÂN LOẠI CẢM XÚC DỮ LIỆU CONFESSION SINH VIÊN
## (Sentiment Analysis on Student Confession Data: A Multi-Method Research Framework)

> **Áp dụng chuẩn phương pháp:** [antigravity-ai-research-methods](file:///C:/Users/Windows/.gemini/config/skills/antigravity-ai-research-methods/SKILL.md)  
> **Tài liệu tham chiếu:** [workflow-map.md](file:///C:/Users/Windows/.gemini/config/skills/antigravity-ai-research-methods/references/workflow-map.md) | [Chương 6: Thiết Kế Định Lượng](file:///C:/Users/Windows/.gemini/config/skills/antigravity-ai-research-methods/references/03-thiet-ke/chuong-06-dinh-luong.md) | [Chương 10: Phân Tích Dữ Liệu Định Lượng](file:///C:/Users/Windows/.gemini/config/skills/antigravity-ai-research-methods/references/04-thuc-hien/chuong-10-phan-tich-dinh-luong.md)

---

## MỤC LỤC
1. [Bối Cảnh & Đặc Thù Ngữ Liệu Confession Tiếng Việt](#1-bối-cảnh--đặc-thù-ngữ-liệu-confession-tiếng-việt)
2. [Đạo Đức Dữ Liệu & Khử Định Danh (Data Ethics & Anonymization)](#2-đạo-đức-dữ-liệu--khử-định-danh-data-ethics--anonymization)
3. [Khung Phương Pháp Đối Sánh Đa Tầng (Multi-Method Benchmark)](#3-khung-phương-pháp-đối-sánh-đa-tầng-multi-method-benchmark)
   - [Hướng 1: Dựa trên Từ điển & Luật (Lexicon & Rule-based)](#hướng-1-dựa-trên-từ-điển--luật-lexicon--rule-based)
   - [Hướng 2: Học máy truyền thống (Traditional Machine Learning)](#hướng-2-học-máy-truyền-thống-traditional-machine-learning)
   - [Hướng 3: Học sâu với Vector nhúng dày (Deep Learning & Dense Embeddings)](#hướng-3-học-sâu-với-vector-nhúng-dày-deep-learning--dense-embeddings)
   - [Hướng 4: Mô hình Transformer tiền huấn luyện (Pre-trained Transformers)](#hướng-4-mô-hình-transformer-tiền-huấn-luyện-pre-trained-transformers)
   - [Hướng 5: Mô hình Ngôn ngữ Lớn (Large Language Models - LLMs)](#hướng-5-mô-hình-ngôn-ngữ-lớn-large-language-models---llms)
   - [Hướng 6: Phương pháp Lai & Phân tích Khía cạnh (Ensemble & ABSA)](#hướng-6-phương-pháp-lai--phân-tích-khía-cạnh-ensemble--absa)
4. [Thiết Kế Thực Nghiệm Chuẩn Khoa Học (Experimental Protocol)](#4-thiết-kế-thực-nghiệm-chuẩn-khoa-học-experimental-protocol)
5. [Pipeline Tiền Xử Lý Mẫu (Data Preprocessing Implementation)](#5-pipeline-tiền-xử-lý-mẫu-data-preprocessing-implementation)
6. [Bộ Prompt Cookbook Theo Chuẩn RCTFC](#6-bộ-prompt-cookbook-theo-chuẩn-rctfc)
7. [Lộ Trình Triển Khai Thực Chiến 4 Tuần](#7-lộ-trình-triển-khai-thực-chiến-4-tuần)
8. [Khai Báo Sử Dụng AI & Tính Toàn Vẹn Học Thuật](#8-khai-báo-sử-dụng-ai--tính-toàn-vẹn-học-thuật)

---

## 1. Bối Cảnh & Đặc Thù Ngữ Liệu Confession Tiếng Việt

Dữ liệu Confession trên mạng xã hội (đặc biệt là sinh viên các trường đại học, ví dụ UIT Confessions) là dạng văn bản phi cấu trúc phản ánh chân thực tâm lý, cảm xúc và đời sống học đường. Đối với bài toán Xử lý Ngôn ngữ Tự nhiên (NLP), dữ liệu này sở hữu các đặc trưng dị biệt:

| Đặc trưng | Biểu hiện thực tế trong Confession | Thách thức đối với mô hình |
|---|---|---|
| **Teencode & Viết tắt** | `ko`, `hok`, `j`, `mk`, `đc`, `hum nay`, `tui`, `zậy`, `rùi` | Gây bùng nổ từ vựng ngoài từ điển (Out-Of-Vocabulary - OOV) |
| **Pha trộn mã (Code-mixing)** | `pass môn`, `tạch môn`, `crush`, `deadline dí`, `fix bug`, `review` | Làm nhiễu bộ tách từ thuần Việt hoặc bộ nhúng đơn ngữ |
| **Mỉa mai & Châm biếm (Sarcasm)** | *"Wifi trường mạnh quá, load từ tiết 1 đến tiết 4 mới vào được Moodle :)))"* | Nghĩa bề mặt đối lập hoàn toàn với cảm xúc thực (Positive $\leftrightarrow$ Negative) |
| **Đa cảm xúc (Mixed Sentiments)** | Khen thầy cô nhiệt tình nhưng chê cơ sở vật chất cũ kỹ | Khó gán 1 nhãn duy nhất cho toàn bài (cần Aspect-based) |
| **Biểu tượng cảm xúc (Emoji/Emoticon)** | `=)))`, `:(`, `T_T`, `🐧`, `🤡`, `❤️` | Chứa trọng số cảm xúc cực lớn nhưng thường bị xóa nhầm trong tiền xử lý |

---

## 2. Đạo Đức Dữ Liệu & Khử Định Danh (Data Ethics & Anonymization)

Theo hướng dẫn tại [Chương 16: Kiểm Duyệt Đạo Đức & AI Ethics](file:///C:/Users/Windows/.gemini/config/skills/antigravity-ai-research-methods/references/05-hoan-thien/chuong-16-dao-duc.md) và mẫu [Kế hoạch Quản lý Dữ liệu (DMP)](file:///C:/Users/Windows/.gemini/config/skills/antigravity-ai-research-methods/resources/templates/data-management-plan.md):

1. **Nguyên tắc Khử định danh (De-identification):** Dù confession là công khai, mọi thông tin nhận dạng cá nhân (PII) như:
   - Tên riêng sinh viên, giảng viên
   - Mã số sinh viên (MSSV), số điện thoại, link Facebook
   - Tên lớp sinh hoạt đặc thù
   $\rightarrow$ **Bắt buộc** phải được thay thế bằng token ẩn danh: `[NAME]`, `[STUDENT_ID]`, `[PHONE]`, `[URL]` trước khi đưa vào tập dữ liệu huấn luyện hoặc công bố.
2. **Lưu trữ an toàn:** Tách biệt tuyệt đối giữa thư mục `data/raw/` (dữ liệu thô gốc, không public) và `data/processed/` (dữ liệu đã khử định danh phục vụ nghiên cứu).

---

## 3. Khung Phương Pháp Đối Sánh Đa Tầng (Multi-Method Benchmark)

Để công trình nghiên cứu có giá trị học thuật cao, ta thiết lập ma trận thực nghiệm đối sánh 6 họ phương pháp:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MA TRẬN PHƯƠNG PHÁP NGHIÊN CỨU                  │
├────────────────────────────────────────────────────────────────────────┤
│  [1. Heuristic]    ──→ Lexicon & Rules (VietSentiWordNet, Emoji map)   │
│  [2. Machine Learning] ──→ TF-IDF + Naive Bayes, Logistic Reg, SVM     │
│  [3. Deep Learning]──→ Word2Vec/FastText + TextCNN, BiLSTM-Attention   │
│  [4. Transformers] ──→ PhoBERT, ViBERT, BARTpho, XLM-RoBERTa           │
│  [5. LLMs]         ──→ Gemini / Llama-3 (Zero-shot, Few-shot CoT, PEFT)│
│  [6. Hybrid/ABSA]  ──→ Stacking Ensemble, Aspect-Based Extraction      │
└────────────────────────────────────────────────────────────────────────┘
```

### Hướng 1: Dựa trên Từ điển & Luật (Lexicon & Rule-based)
* **Nguyên lý:** Xây dựng danh sách từ vựng mang sắc thái tích cực (+1), tiêu cực (-1). Điểm của văn bản là hàm kết hợp của các từ, có tính đến từ phủ định (`không`, `chẳng`, `chưa`) và từ chỉ mức độ (`rất`, `quá`, `cực kỳ`).
* **Tài nguyên:** 
  - Vietnamese Sentiment Lexicon, VietSentiWordNet.
  - Bảng ánh xạ Emoji cảm xúc sinh viên (`:))` $\rightarrow$ +1, `T_T` $\rightarrow$ -1).
* **Đánh giá:** Không cần train, tốc độ tức thì, tính diễn giải cao (white-box). Tuy nhiên thất bại trước teencode mới và câu mỉa mai.
* **Vai trò:** **Heuristic Baseline** (mốc tối thiểu để so sánh).

### Hướng 2: Học máy truyền thống (Traditional Machine Learning)
* **Trích xuất đặc trưng (Feature Engineering):**
  - **TF-IDF:** N-gram cấp độ từ (Unigram + Bigram) và N-gram cấp độ ký tự (3-gram, 4-gram để bắt teencode).
  - **Handcrafted Features:** Số lượng emoji, tỷ lệ viết hoa, độ dài văn bản, số lượng dấu cảm thán (`!`, `?`).
* **Thuật toán so sánh:**
  - **Multinomial Naive Bayes (MNB):** Baseline xác suất nhanh gọn.
  - **Logistic Regression (L2 Regularization):** Đo lường Feature Importance (trọng số của từng từ).
  - **Linear Support Vector Machine (LinearSVC):** Thường đạt kết quả vượt trội nhất trong nhóm ML truyền thống đối với văn bản tiếng Việt.
  - **LightGBM / XGBoost:** Khai thác tốt các đặc trưng thủ công kết hợp cùng TF-IDF.
* **Đánh giá:** Thời gian huấn luyện tính bằng giây, chạy mượt mà trên CPU cơ bản, baseline đối chứng bắt buộc trong các bài báo khoa học.

### Hướng 3: Học sâu với Vector nhúng dày (Deep Learning & Dense Embeddings)
* **Vector đại diện (Dense Embeddings):**
  - Pre-trained **Word2Vec** hoặc **FastText** tiếng Việt (FastText có cơ chế n-gram ký tự con giúp giải quyết bài toán OOV của teencode rất tốt).
  - Pre-trained **PhoW2V** (VinAI).
* **Kiến trúc mạng nơ-ron:**
  - **TextCNN (1D-CNN):** Sử dụng các kernel size (2, 3, 4) để bắt các cụm từ thể hiện cảm xúc cục bộ.
  - **BiLSTM / BiGRU:** Bắt phụ thuộc ngữ cảnh hai chiều (tiến và lùi).
  - **BiLSTM + Self-Attention:** Tự động gán trọng số tập trung vào những từ mang sắc thái quyết định trong confession dài.
* **Đánh giá:** Nắm bắt ngữ nghĩa tốt hơn TF-IDF, nhưng cần lượng dữ liệu gán nhãn tương đối để tối ưu tham số.

### Hướng 4: Mô hình Transformer tiền huấn luyện (Pre-trained Transformers - SOTA)
* **Các kiến trúc cốt lõi:**
  - **PhoBERT (`vinai/phobert-base`, `vinai/phobert-large`):** Mô hình SOTA chuẩn mực cho tiếng Việt. *Yêu cầu bắt buộc:* Phải tách từ (Word Segmentation qua `pyvi` hoặc `RDRsegmenter`).
  - **ViBERT (`FPTAI/vibert-base-cased`):** Mô hình BERT của FPT.
  - **BARTpho (`vinai/bartpho-syllable`):** Kiến trúc Seq2Seq mạnh mẽ, bản syllable không bắt buộc tách từ ghép.
  - **XLM-RoBERTa (`xlm-roberta-base`):** Mô hình đa ngôn ngữ (Multilingual), xử lý vượt trội khi confession có nhiều từ lóng tiếng Anh (code-mixing).
* **Chiến lược huấn luyện:**
  - **Full Fine-Tuning:** Huấn luyện toàn bộ weights với Learning Rate nhỏ ($2 \times 10^{-5}$, Warmup linear).
  - **Domain-Adaptive Pre-training (DAPT):** Dùng dữ liệu confession chưa gán nhãn để tiếp tục train Masked Language Model (MLM) cho PhoBERT trước khi fine-tune nhãn cảm xúc.

### Hướng 5: Mô hình Ngôn ngữ Lớn (Large Language Models - LLMs)
* **Khai thác năng lực suy luận ngữ cảnh sâu:**
  - **Zero-Shot / Few-Shot Prompting (In-Context Learning):** Sử dụng Google Gemini 2.5 Flash hoặc GPT-4o-mini kèm kỹ thuật Chain-of-Thought (CoT) để giải mã ẩn ý mỉa mai (Sarcasm).
  - **LLM as a Weak Annotator (Silver Labeling):** Sử dụng LLM gán nhãn tự động cho tập confession lớn, kết hợp thẩm định Human-in-the-loop để tăng tốc độ dán nhãn gấp 10 lần.
  - **Parameter-Efficient Fine-Tuning (PEFT / QLoRA):** Fine-tune các mô hình mã nguồn mở như `Vistral-7B`, `Llama-3-8B` với adapter LoRA trên tập confession.

### Hướng 6: Phương pháp Lai & Phân tích Khía cạnh (Ensemble & ABSA)
* **Ensemble (Voting / Stacking):** Kết hợp phân phối xác suất đầu ra của PhoBERT + SVM + Điểm Lexicon để đưa ra quyết định chuẩn xác nhất.
* **Aspect-Based Sentiment Analysis (ABSA):** Tách nhãn cảm xúc theo các khía cạnh cụ thể của sinh viên:
  - Khía cạnh Đào tạo / Giảng viên (Pos / Neg / Neu / None)
  - Khía cạnh Cơ sở vật chất / Dịch vụ (Pos / Neg / Neu / None)
  - Khía cạnh Đời sống / Tình cảm (Pos / Neg / Neu / None)

---

## 4. Thiết Kế Thực Nghiệm Chuẩn Khoa Học (Experimental Protocol)

### 4.1 Phân chia tập dữ liệu (Leakage-Safe Data Split)
* Chia tập theo tỷ lệ **Stratified 70% Train - 15% Validation - 15% Test** (giữ nguyên tỷ lệ các lớp phân loại).
* Cố định ngẫu nhiên: `random_state = 42`.
* **Benchmark chéo (Cross-corpus Validation):** Đánh giá chéo mô hình trên tập dữ liệu chuẩn đã công bố của trường:
  - **UIT-VSFC:** [Vietnamese Students' Feedback Corpus](https://github.com/hieunguyen1053/UIT-VSFC) (Đánh giá phản hồi sinh viên).
  - **UIT-VSMEC:** [Vietnamese Social Media Emotion Corpus](https://github.com/hieunguyen1053/UIT-VSMEC) (Cảm xúc mạng xã hội).

### 4.2 Thước đo đánh giá (Evaluation Metrics)
Vì dữ liệu mạng xã hội thường mất cân bằng (Imbalanced Class):
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Macro-F1} = \frac{1}{C} \sum_{i=1}^C F1_i \quad \text{(Thước đo quan trọng nhất)}$$
* Báo cáo đầy đủ: **Precision, Recall, Macro-F1, Weighted-F1, và Ma trận nhầm lẫn (Confusion Matrix)**.

### 4.3 Thiết kế Nghiên cứu Bóc Tách (Ablation Study)
Kiểm chứng đóng góp của từng khâu tiền xử lý vào hiệu năng cuối:
1. **Ablation 1 (Emoji):** Giữ nguyên vs. Ánh xạ thành text vs. Xóa bỏ emoji.
2. **Ablation 2 (Teencode):** Chuẩn hóa bằng từ điển vs. Giữ nguyên dạng gốc.
3. **Ablation 3 (Word Segmentation):** Sử dụng `pyvi` tách từ ghép vs. Tách mức âm tiết (syllable).

---

## 5. Pipeline Tiền Xử Lý Mẫu (Data Preprocessing Implementation)

Dưới đây là mã nguồn Python chuẩn mực có thể tích hợp trực tiếp vào thư mục mã nguồn của bạn:

```python
"""
Data Preprocessing Pipeline for Vietnamese Confession Sentiment Analysis
Author: Research Methodology on Antigravity
"""

import re
import unicodedata
from pyvi import ViTokenizer

# 1. Từ điển chuẩn hóa Teencode & Slang sinh viên
TEENCODE_DICT = {
    "ko": "không", "hok": "không", "k": "không", "kh": "không", "hổng": "không",
    "đc": "được", "dc": "được",
    "j": "gì", "gi": "gì",
    "mk": "mình", "mik": "mình",
    "bt": "biết", "bit": "biết",
    "vs": "với", "w": "với",
    "ng": "người", "ngừi": "người", "mn": "mọi người",
    "cfs": "confession", "ad": "admin",
    "r": "rồi", "rùi": "rồi",
    "h": "giờ", "hum": "hôm",
    "tui": "tôi", "t": "tôi", "m": "mày",
    "tạch": "rớt môn", "pass": "qua môn"
}

# 2. Ánh xạ Emoji / Emoticon phổ biến sang từ ngữ cảm xúc
EMOJI_DICT = {
    ":)": " vui ", ":))": " rất_vui ", "=)))": " cười_lớn ",
    ":(": " buồn ", ":'(": " khóc ", "T_T": " khóc ",
    "❤️": " yêu_thích ", "👍": " đồng_ý ", "😡": " phẫn_nộ ",
    "😭": " đau_khổ ", "🥰": " yêu_thích ", "🤡": " hề_hước "
}

class ConfessionPreprocessor:
    def __init__(self, use_word_segmentation: bool = True):
        self.use_word_segmentation = use_word_segmentation

    def normalize_unicode(self, text: str) -> str:
        """Đưa về chuẩn Unicode dựng sẵn (NFC)."""
        return unicodedata.normalize("NFC", text)

    def clean_pii(self, text: str) -> str:
        """Khử thông tin cá nhân (PII) để bảo vệ đạo đức dữ liệu."""
        # Ẩn URL
        text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
        # Ẩn số điện thoại
        text = re.sub(r"(0\d{9,10})", "[PHONE]", text)
        # Ẩn MSSV (ví dụ định dạng 8 chữ số)
        text = re.sub(r"\b\d{8}\b", "[STUDENT_ID]", text)
        return text

    def map_emojis_and_teencode(self, text: str) -> str:
        """Ánh xạ icon và chuẩn hóa teencode."""
        for emo, val in EMOJI_DICT.items():
            text = text.replace(emo, f" {val} ")
        
        words = text.split()
        normalized_words = [TEENCODE_DICT.get(w.lower(), w) for w in words]
        return " ".join(normalized_words)

    def process(self, text: str) -> str:
        """Quy trình tiền xử lý hoàn chỉnh."""
        if not isinstance(text, str):
            return ""
        
        text = self.normalize_unicode(text)
        text = self.clean_pii(text)
        text = self.map_emojis_and_teencode(text)
        
        # Xóa các ký tự thừa nhưng giữ lại dấu câu cảm xúc quan trọng (?, !)
        text = re.sub(r"[^\w\s\.\?!_\[\]]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        if self.use_word_segmentation:
            text = ViTokenizer.tokenize(text)
            
        return text

# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    processor = ConfessionPreprocessor(use_word_segmentation=True)
    sample_cfs = "Hok bt ad có duyệt ko, cfs này gửi bạn MSSV 21520000 lớp IT001 dễ thương ghê :)) pass môn rồi nha!"
    cleaned = processor.process(sample_cfs)
    print("Gốc  :", sample_cfs)
    print("Xử lý:", cleaned)
```

---

## 6. Bộ Prompt Cookbook Theo Chuẩn RCTFC

Áp dụng cấu trúc **Role - Context - Task - Format - Constraint** từ [Phụ lục A: Prompt Cookbook](file:///C:/Users/Windows/.gemini/config/skills/antigravity-ai-research-methods/references/06-phu-luc/phu-luc-a-prompt-cookbook.md):

### Prompt 1: Đánh Giá & Gán Nhãn Cảm Xúc Có Giải Mã Sarcasm (Zero-shot CoT)
```text
[ROLE]: Chuyên gia phân tích ngữ nghĩa và tâm lý ngôn ngữ học tiếng Việt.
[CONTEXT]: Bạn đang phân loại cảm xúc các bài đăng confession của sinh viên trường đại học kỹ thuật. Dữ liệu này chứa nhiều tiếng lóng, lối nói mỉa mai (sarcasm) hoặc than phiền ngầm về việc học, điểm số, giảng viên, cơ sở vật chất.
[TASK]: Đọc confession sau và phân loại nhãn cảm xúc chính xác:
Confession: """{noi_dung_confession}"""
1. Phân tích ngữ cảnh ngầm và xác định xem có yếu tố mỉa mai / châm biếm hay không.
2. Gán 1 nhãn cảm xúc duy nhất: [POSITIVE, NEGATIVE, NEUTRAL].
[FORMAT]: Trả về kết quả dưới dạng JSON:
{
  "has_sarcasm": true/false,
  "implicit_context": "Tóm tắt ngắn gọn 1 câu về ngữ cảnh ngầm",
  "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "confidence_score": 0.0 - 1.0
}
[CONSTRAINT]: Tuyệt đối không để cảm xúc bề mặt đánh lừa (ví dụ khen thiết bị 'quá xịn' kèm biểu tượng cười khóc khi máy bị hỏng thì phải là NEGATIVE). Nếu là bài tìm đồ hay hỏi môn học thuần túy, gán NEUTRAL.
```

### Prompt 2: Phân Tích Lỗi Sai Học Thuật (Error Analysis)
```text
[ROLE]: Reviewer học thuật hội nghị quốc tế chuyên ngành NLP và Xử lý Văn bản Xã hội.
[CONTEXT]: Mô hình PhoBERT-base của tôi sau khi fine-tune đạt Macro-F1 = 0.87 trên tập Test Confession. Tuy nhiên vẫn còn một số lượng mẫu bị nhầm lẫn giữa nhãn NEGATIVE và POSITIVE.
[TASK]: Dưới đây là danh sách 10 mẫu tiêu biểu mà mô hình dự đoán sai (Ground Truth vs Prediction). Hãy:
1. Phân loại các nguyên nhân lỗi thành một bảng Taxonomy of Errors (ví dụ: mỉa mai, ngữ cảnh pha trộn, nhãn gốc bị nhiễu do người gán nhãn bất đồng).
2. Đề xuất 3 kỹ thuật cụ thể để cải thiện mô hình ở giai đoạn tiếp theo.
[FORMAT]: Bảng phân tích chi tiết kèm tỷ lệ lỗi ước lượng và khuyến nghị học thuật.
[CONSTRAINT]: Lập luận dựa trên nguyên lý xử lý ngôn ngữ, không đưa ra nhận xét chủ quan, không bịa số liệu.
```

---

## 7. Lộ Trình Triển Khai Thực Chiến 4 Tuần

| Tuần | Giai đoạn nghiên cứu | Nhiệm vụ kỹ thuật trọng tâm | Sản phẩm bàn giao (Deliverable) |
|---|---|---|---|
| **Tuần 1** | **Khám phá & Chuẩn hóa Dữ liệu** | - Thu thập confession từ script crawl (`uit-cfs-apify`)<br>- Lọc trùng, khử định danh PII ([Chương 16](file:///C:/Users/Windows/.gemini/config/skills/antigravity-ai-research-methods/references/05-hoan-thien/chuong-16-dao-duc.md))<br>- Xây dựng Guideline gán nhãn & đo độ đồng thuận Fleiss' Kappa | Tập dữ liệu sạch (`data/clean.parquet`);<br>Bản kế hoạch quản lý dữ liệu ([DMP](file:///C:/Users/Windows/.gemini/config/skills/antigravity-ai-research-methods/resources/templates/data-management-plan.md)) |
| **Tuần 2** | **Xây dựng Baselines (ML & Lexicon)** | - Xây dựng từ điển Teencode & Emoji mapping<br>- Huấn luyện TF-IDF + Naive Bayes, Logistic Regression, LinearSVM<br>- Huấn luyện FastText + TextCNN / BiLSTM | Bảng kết quả Baselines;<br>Ma trận nhầm lẫn (Confusion Matrix) của SVM |
| **Tuần 3** | **Fine-tune Transformers & Thử nghiệm LLM** | - Fine-tune `PhoBERT-base`, `BARTpho-syllable`<br>- Chạy Zero/Few-shot với Gemini 2.5 Flash API<br>- Thực hiện Ablation Study (Emoji vs No-Emoji) | Bảng Benchmark toàn diện đa phương pháp;<br>Mô hình SOTA tốt nhất lưu tại `checkpoints/` |
| **Tuần 4** | **Phân Tích Lỗi & Viết Báo Cáo** | - Phân tích định lượng nguyên nhân lỗi (Error Analysis)<br>- Trực quan hóa diễn biến cảm xúc theo thời gian (Visualization)<br>- Soạn bài báo theo cấu trúc chuẩn IMRaD ([Chương 13](file:///C:/Users/Windows/.gemini/config/skills/antigravity-ai-research-methods/references/05-hoan-thien/chuong-13-viet-bai-bao.md)) | Bản thảo bài báo nghiên cứu (Draft Paper);<br>Bản Khai báo Sử dụng AI ([ai-disclosure.md](file:///C:/Users/Windows/.gemini/config/skills/antigravity-ai-research-methods/resources/templates/ai-disclosure.md)) |

---

## 8. Khai Báo Sử Dụng AI & Tính Toàn Vẹn Học Thuật

Tuân thủ nguyên tắc [ai-disclosure.md](file:///C:/Users/Windows/.gemini/config/skills/antigravity-ai-research-methods/resources/templates/ai-disclosure.md):
- AI được sử dụng trong nghiên cứu này như một **cộng sự kỹ thuật**: hỗ trợ sinh mã nguồn tiền xử lý, gợi ý từ điển teencode, và hỗ trợ dán nhãn thử nghiệm (weak supervision).
- Mọi quyết định học thuật, thiết kế phương pháp, thẩm định nhãn kiểm thử và diễn giải kết quả nghiên cứu hoàn toàn do tác giả con người chịu trách nhiệm toàn diện.

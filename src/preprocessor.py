"""
Data Preprocessing Pipeline for Vietnamese Confession Sentiment Analysis
Adheres to NGHIEN_CUU_PHAN_LOAI_CAM_XUC_CONFESSION.md Section 5
"""

import re
import unicodedata
from pyvi import ViTokenizer

# 1. Từ điển chuẩn hóa Teencode & Slang sinh viên UIT
TEENCODE_DICT = {
    "ko": "không", "hok": "không", "k": "không", "kh": "không", "hổng": "không", "hong": "không", "hông": "không", "khum": "không", "kô": "không",
    "đc": "được", "dc": "được", "dk": "được",
    "j": "gì", "gi": "gì", "z": "vậy", "zậy": "vậy", "dị": "vậy", "zì": "vậy",
    "mk": "mình", "mik": "mình", "mềnh": "mình",
    "bt": "biết", "bit": "biết", "bít": "biết",
    "vs": "với", "w": "với", "cùng": "với",
    "ng": "người", "ngừi": "người", "mn": "mọi người", "mng": "mọi người", "mụi": "mọi",
    "cfs": "confession", "ad": "admin",
    "r": "rồi", "rùi": "rồi", "r r": "rồi",
    "h": "giờ", "hum": "hôm", "hnay": "hôm nay", "hqa": "hôm qua",
    "tui": "tôi", "t": "tôi", "m": "mày",
    "tạch": "rớt môn", "pass": "qua môn",
    "pđt": "phòng đào tạo", "ctsv": "phòng công tác sinh viên",
    "đkhp": "đăng ký học phần", "dkhp": "đăng ký học phần",
    "hbkkht": "học bổng khuyến khích học tập", "hb": "học bổng",
    "sv": "sinh viên", "tsv": "tân sinh viên",
    "gv": "giảng viên", "thực hành": "thực hành",
    "qsu": "quân sự", "kqs": "khu quân sự", "ttgdqpan": "trung tâm giáo dục quốc phòng",
    "av": "anh văn", "av1": "anh văn 1", "av2": "anh văn 2", "av3": "anh văn 3",
    "dsa": "cấu trúc dữ liệu giải thuật", "oop": "lập trình hướng đối tượng",
    "nmlt": "nhập môn lập trình", "kltn": "khóa luận tốt nghiệp",
    "cs": "khoa học máy tính", "ce": "kỹ thuật máy tính", "se": "kỹ thuật phần mềm",
    "is": "hệ thống thông tin", "nc": "mạng máy tính", "ec": "thương mại điện tử"
}

# 2. Ánh xạ Emoji / Emoticon phổ biến sang từ ngữ cảm xúc tiếng Việt
EMOJI_DICT = {
    ":)": " vui ", ":))": " rất_vui ", "=)))": " cười_lớn ", ":)))": " cười_lớn ", ":D": " rất_vui ",
    ":(": " buồn ", ":'(": " khóc ", "T_T": " khóc ", ";-;": " khóc ", ":<<": " buồn ", ":<": " buồn ",
    "❤️": " yêu_thích ", "💖": " yêu_thích ", "👍": " đồng_ý ", "👎": " phản_đối ",
    "😡": " phẫn_nộ ", "🤬": " tức_giận ", "😭": " đau_khổ ", "🥺": " tha_thiết ",
    "🥰": " yêu_thích ", "😍": " say_mê ", "🤡": " hề_hước ", "🐧": " mỉa_mai ",
    "🤦": " bất_lực ", "💀": " tuyệt_vọng ", "💔": " đau_lòng "
}

class ConfessionPreprocessor:
    def __init__(self, use_word_segmentation: bool = True):
        self.use_word_segmentation = use_word_segmentation

    def normalize_unicode(self, text: str) -> str:
        """Đưa về chuẩn Unicode dựng sẵn (NFC)."""
        return unicodedata.normalize("NFC", text)

    def clean_pii(self, text: str) -> str:
        """Khử thông tin cá nhân (PII) theo đạo đức dữ liệu chuẩn Chương 16."""
        # Ẩn URL / link mạng xã hội
        text = re.sub(r"https?://\S+|www\.\S+|bit\.ly/\S+", "[URL]", text)
        # Ẩn email
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text)
        # Ẩn số điện thoại Việt Nam
        text = re.sub(r"(?:\+?84|0)(?:\d{9,10})\b", "[PHONE]", text)
        # Ẩn MSSV định dạng 8 chữ số (ví dụ 2152xxxx, 2252xxxx, 2052xxxx, 1952xxxx)
        text = re.sub(r"\b(?:1[5-9]|2[0-6])52\d{4}\b", "[STUDENT_ID]", text)
        # Ẩn biển số xe (ví dụ: 62N1-44475, 47S1- 306.14)
        text = re.sub(r"\b\d{2}[A-Z]\d[\s-]?\d{3,5}(?:\.\d{2})?\b", "[LICENSE_PLATE]", text)
        return text

    def map_emojis_and_teencode(self, text: str) -> str:
        """Ánh xạ icon cảm xúc và chuẩn hóa teencode."""
        for emo, val in EMOJI_DICT.items():
            text = text.replace(emo, f" {val} ")
        
        words = text.split()
        normalized_words = [TEENCODE_DICT.get(w.lower(), w) for w in words]
        return " ".join(normalized_words)

    def process(self, text: str) -> str:
        """Quy trình tiền xử lý hoàn chỉnh cho bài toán sentiment analysis."""
        if not isinstance(text, str):
            return ""
        
        text = self.normalize_unicode(text)
        text = self.clean_pii(text)
        text = self.map_emojis_and_teencode(text)
        
        # Xóa các ký tự phân cách rác nhưng giữ lại các dấu cảm xúc, ngoặc và nhãn token
        text = re.sub(r"[^\w\s\.\?!_\[\]]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        if self.use_word_segmentation:
            try:
                text = ViTokenizer.tokenize(text)
            except Exception:
                pass
            
        return text

if __name__ == "__main__":
    processor = ConfessionPreprocessor(use_word_segmentation=True)
    sample = "Hok bt ad có duyệt ko, gửi bạn MSSV 21520000 lớp IT001 dễ thương ghê :)) tạch môn rùi nha! Liên hệ 0901234567 hoặc bit.ly/cfs"
    print("Gốc:", sample)
    print("Xử lý:", processor.process(sample))

"""
Lexicon and Rule-Based Sentiment Classifier for Student Confessions
Implements Heuristic Baseline adhering to NGHIEN_CUU_PHAN_LOAI_CAM_XUC_CONFESSION.md Section 3 (Hướng 1)
"""

import re

# Từ điển sắc thái tích cực
POSITIVE_LEXICON = {
    # Khen ngợi chung
    "tốt": 1.0, "tuyệt": 1.5, "tuyệt_vời": 2.0, "xịn": 1.2, "xịn_sò": 1.5, "mượt": 1.2, "mượt_mà": 1.5,
    "ổn": 0.8, "ổn_áp": 1.0, "hài_lòng": 1.5, "tự_hào": 1.8, "yêu_thích": 1.5, "yêu_trường": 2.0,
    "thích": 1.0, "mê": 1.2, "đẹp": 1.0, "dễ_thương": 1.0, "đáng_yêu": 1.2, "dễ_gần": 1.0,
    # Giảng viên & Học tập
    "nhiệt_tình": 1.5, "tận_tâm": 1.8, "tâm_huyết": 2.0, "dễ_hiểu": 1.5, "chu_đáo": 1.5,
    "dễ_tính": 1.2, "chấm_dễ": 1.5, "qua_môn": 1.2, "đậu": 1.0, "điểm_cao": 1.5, "học_bổng": 1.2,
    "xuất_sắc": 1.8, "giỏi": 1.2, "chuyên_nghiệp": 1.5, "uy_tín": 1.5, "tử_tế": 1.5,
    # Cảm ơn & Tán thành
    "cảm_ơn": 1.2, "biết_ơn": 1.8, "cảm_kích": 1.8, "đồng_ý": 1.0, "ủng_hộ": 1.2,
    "vui": 1.0, "rất_vui": 1.5, "hạnh_phúc": 1.8, "chúc_mừng": 1.2, "đỉnh": 1.5, "cháy": 1.2
}

# Từ điển sắc thái tiêu cực
NEGATIVE_LEXICON = {
    # Bức xúc & Phàn nàn
    "tệ": -1.5, "tồi": -1.5, "dở": -1.2, "kém": -1.2, "bất_cập": -1.5, "bức_xúc": -1.8,
    "chán": -1.2, "thất_vọng": -1.8, "bất_bình": -1.5, "vô_lý": -1.8, "khó_chịu": -1.5,
    "chậm": -1.0, "chậm_trễ": -1.5, "lâu": -0.8, "quá_tải": -1.5, "kẹt": -1.2, "tắc": -1.2,
    "lag": -1.2, "hỏng": -1.5, "hư": -1.2, "nóng": -1.0, "bẩn": -1.5, "rác": -1.2,
    "ồn": -1.2, "ồn_ào": -1.5, "mất_trật_tự": -1.5, "vô_ý_thức": -2.0, "thiếu_ý_thức": -1.8,
    # Học vụ & Thi cử
    "rớt": -1.5, "rớt_môn": -1.8, "tạch": -1.8, "nợ_môn": -1.5, "cấm_thi": -2.0, "phạt": -1.2,
    "khó": -0.8, "khó_khăn": -1.2, "áp_lực": -1.5, "stress": -1.5, "bế_tắc": -1.8, "đuối": -1.2,
    "bất_lực": -1.5, "đau_khổ": -1.5, "khóc": -1.2, "buồn": -1.0, "tuyệt_vọng": -2.0,
    # Chỉ trích gay gắt
    "lừa": -1.8, "lừa_đảo": -2.0, "gian_lận": -2.0, "chơi_dơ": -1.8, "thượng_đẳng": -1.8,
    "khinh": -1.8, "khinh_thường": -2.0, "chửi": -1.5, "hèn": -1.8, "hề_hước": -0.8,
    "mỉa_mai": -1.2, "tiểu_nhân": -2.0, "ích_kỷ": -1.5, "vô_trách_nhiệm": -2.0
}

# Từ phủ định (đảo ngược sắc thái từ đi liền sau)
NEGATIONS = {"không", "chẳng", "chưa", "không_thể", "chả", "hổng", "khum"}

# Từ chỉ mức độ (khuếch đại cảm xúc)
INTENSIFIERS = {"rất": 1.5, "quá": 1.6, "cực_kỳ": 1.8, "siêu": 1.8, "vô_cùng": 1.8, "lắm": 1.4, "dã_man": 1.5}

class LexiconSentimentClassifier:
    def __init__(self):
        pass

    def score_text(self, tokenized_text: str, target: str = "SCHOOL"):
        """
        Tính điểm cảm xúc phân cực (Polarity Score) từ -1.0 đến +1.0.
        Nếu bài viết hoàn toàn là chuyện cá nhân (crush/tìm đồ) -> tự động gán NEUTRAL cho trường.
        """
        if not isinstance(tokenized_text, str) or not tokenized_text.strip():
            return {"sentiment": "NEUTRAL", "score": 0.0, "sarcasm_detected": False}

        words = tokenized_text.lower().split()
        total_score = 0.0
        match_count = 0
        i = 0
        
        while i < len(words):
            word = words[i]
            
            # Kiểm tra từ phủ định phía trước (ví dụ: 'không tốt', 'chẳng dễ hiểu')
            is_negated = False
            if i > 0 and words[i - 1] in NEGATIONS:
                is_negated = True
            elif i > 1 and words[i - 2] in NEGATIONS:
                is_negated = True
                
            # Kiểm tra từ khuếch đại (ví dụ: 'rất tốt', 'quá tệ')
            multiplier = 1.0
            if i > 0 and words[i - 1] in INTENSIFIERS:
                multiplier = INTENSIFIERS[words[i - 1]]

            word_val = 0.0
            if word in POSITIVE_LEXICON:
                word_val = POSITIVE_LEXICON[word]
            elif word in NEGATIVE_LEXICON:
                word_val = NEGATIVE_LEXICON[word]
                
            if word_val != 0.0:
                if is_negated:
                    word_val = -word_val * 0.8
                word_val *= multiplier
                total_score += word_val
                match_count += 1
                
            i += 1
            
        # Phát hiện Sarcasm (Mỉa mai): từ ngữ tích cực kèm biểu tượng cười cợt hoặc contrast
        sarcasm_detected = False
        raw_lower = tokenized_text.lower()
        if ("cười_lớn" in raw_lower or "mỉa_mai" in raw_lower or "hề_hước" in raw_lower) and total_score > 0:
            if any(k in raw_lower for k in ["moodle", "wifi", "thang_máy", "bãi_xe", "học_phí", "lịch_thi"]):
                total_score = -abs(total_score) * 1.2
                sarcasm_detected = True

        # Chuẩn hóa điểm về khoảng [-1.0, 1.0]
        if match_count > 0:
            normalized_score = total_score / (match_count * 1.5)
            normalized_score = max(-1.0, min(1.0, normalized_score))
        else:
            normalized_score = 0.0

        # Nếu là chuyện cá nhân (crush, pass đồ, tìm trọ) và không đề cập cơ sở/giảng viên:
        if target == "PERSONAL" and abs(normalized_score) < 0.5:
            sentiment = "NEUTRAL"
        else:
            if normalized_score > 0.15:
                sentiment = "POSITIVE"
            elif normalized_score < -0.15:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"

        return {
            "sentiment": sentiment,
            "score": round(normalized_score, 3),
            "sarcasm_detected": sarcasm_detected
        }

if __name__ == "__main__":
    clf = LexiconSentimentClassifier()
    tests = [
        "Thầy Dũng dạy rất nhiệt_tình và chu_đáo, bài giảng siêu dễ_hiểu!",
        "Thang_máy quá_tải kẹt cứng cả buổi sáng, bức_xúc vô_cùng.",
        "Wifi trường mượt quá cười_lớn load từ tiết 1 đến tiết 4 chưa xong Moodle.",
        "Em cần tìm bạn nam áo trắng hôm_nay ở thư_viện, bạn dễ_thương ghê."
    ]
    for t in tests:
        print(t)
        print(clf.score_text(t))

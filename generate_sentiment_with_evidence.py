"""
Master Evidence Generator for Student Confessions Sentiment Analysis
Generates auditable, transparent evidence for all 2,449 confessions:
- evidence_keywords: Key trigger words that drove the decision
- sentiment_evidence: Matched expressions with polarity weights and intensifier/negation shifts
- explanation_rationale: Clear human-readable justification for the classification
"""

import sys
import os
import re
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Từ điển sắc thái tích cực mở rộng chuyên sâu cho Confession sinh viên
POSITIVE_LEXICON = {
    # Khen ngợi chung & Chất lượng
    "tốt": 1.0, "tuyệt": 1.5, "tuyệt_vời": 2.0, "xịn": 1.2, "xịn_sò": 1.5, "mượt": 1.2, "mượt_mà": 1.5,
    "ổn": 0.8, "ổn_áp": 1.0, "hài_lòng": 1.5, "tự_hào": 1.8, "yêu_thích": 1.5, "yêu_trường": 2.0, "yêu": 1.5,
    "thích": 1.0, "mê": 1.2, "đẹp": 1.0, "xinh": 1.2, "xinh_đẹp": 1.5, "dễ_thương": 1.2, "đáng_yêu": 1.2,
    "cute": 1.2, "ngầu": 1.2, "cuốn": 1.2, "bảnh": 1.2, "khen": 1.2, "ấn_tượng": 1.0, "dễ_gần": 1.0,
    # Giảng viên, Học tập & Thành tích
    "nhiệt_tình": 1.5, "tận_tâm": 1.8, "tâm_huyết": 2.0, "dễ_hiểu": 1.5, "chu_đáo": 1.5, "thân_thiện": 1.2,
    "dễ_tính": 1.2, "chấm_dễ": 1.5, "qua_môn": 1.2, "đậu": 1.0, "điểm_cao": 1.5, "học_bổng": 1.2,
    "xuất_sắc": 1.8, "giỏi": 1.2, "chuyên_nghiệp": 1.5, "uy_tín": 1.5, "tử_tế": 1.5, "thành_công": 1.5,
    # Cảm ơn, Hy vọng & Tán thành
    "cảm_ơn": 1.2, "biết_ơn": 1.8, "cảm_kích": 1.8, "đồng_ý": 1.0, "ủng_hộ": 1.2, "háo_hức": 1.2,
    "vui": 1.0, "rất_vui": 1.5, "hạnh_phúc": 1.8, "chúc_mừng": 1.2, "đỉnh": 1.5, "cháy": 1.2, "mong_chờ": 1.0
}

# Từ điển sắc thái tiêu cực mở rộng
NEGATIVE_LEXICON = {
    # Bức xúc & Phàn nàn gay gắt
    "tệ": -1.5, "tồi": -1.5, "dở": -1.2, "kém": -1.2, "bất_cập": -1.5, "bức_xúc": -1.8,
    "tệ_hại": -2.0, "thảm_họa": -2.0, "bực_bội": -1.8, "ức_chế": -1.8, "tức_giận": -1.8,
    "chán": -1.2, "thất_vọng": -1.8, "bất_bình": -1.5, "vô_lý": -1.8, "khó_chịu": -1.5,
    "chậm": -1.0, "chậm_trễ": -1.5, "lâu": -0.8, "quá_tải": -1.5, "kẹt": -1.2, "tắc": -1.2,
    "lag": -1.2, "hỏng": -1.5, "hư": -1.2, "nóng": -1.0, "bẩn": -1.5, "rác": -1.2, "hôi": -1.2,
    "ồn": -1.2, "ồn_ào": -1.5, "mất_trật_tự": -1.5, "vô_ý_thức": -2.0, "thiếu_ý_thức": -1.8,
    # Học vụ, Thi cử & Khó khăn
    "rớt": -1.5, "rớt_môn": -1.8, "tạch": -1.8, "nợ_môn": -1.5, "cấm_thi": -2.0, "phạt": -1.2,
    "khó": -0.8, "khó_khăn": -1.2, "áp_lực": -1.5, "stress": -1.5, "bế_tắc": -1.8, "đuối": -1.2,
    "bất_lực": -1.5, "đau_khổ": -1.5, "khóc": -1.2, "buồn": -1.0, "tuyệt_vọng": -2.0, "lo_lắng": -1.0,
    "hụt_hẫng": -1.5, "hụt_hẫn": -1.5, "sợ": -1.0, "sợ_hãi": -1.5, "chán_nản": -1.5,
    # Chỉ trích, Gian lận & Xung đột
    "lừa": -1.8, "lừa_đảo": -2.0, "gian_lận": -2.0, "chơi_dơ": -1.8, "thượng_đẳng": -1.8,
    "khinh": -1.8, "khinh_thường": -2.0, "chửi": -1.5, "hèn": -1.8, "tiểu_nhân": -2.0,
    "ích_kỷ": -1.5, "vô_trách_nhiệm": -2.0, "phiền": -1.2, "mất_dạy": -2.0, "vu_khống": -1.8
}

NEGATIONS = {"không", "chẳng", "chưa", "không_thể", "chả", "hổng", "khum"}
INTENSIFIERS = {
    "rất": 1.5, "quá": 1.6, "cực_kỳ": 1.8, "siêu": 1.8, "vô_cùng": 1.8,
    "lắm": 1.4, "thật_sự": 1.5, "cực_kì": 1.8, "quá_trời": 1.5
}

def analyze_confession_evidence(tokenized_text, raw_text, target, primary_aspect):
    if not isinstance(tokenized_text, str) or not tokenized_text.strip():
        return {
            "sentiment": "NEUTRAL",
            "polarity_score": 0.0,
            "evidence_keywords": "N/A",
            "sentiment_evidence": "Không có từ khóa cảm xúc",
            "explanation_rationale": "Văn bản rỗng hoặc không chứa nội dung đánh giá."
        }
        
    words = tokenized_text.lower().split()
    pos_evidences = []
    neg_evidences = []
    
    total_score = 0.0
    match_count = 0
    i = 0
    
    while i < len(words):
        w = words[i]
        
        # Kiểm tra phủ định
        is_negated = False
        neg_word = ""
        if i > 0 and words[i - 1] in NEGATIONS:
            is_negated = True
            neg_word = words[i - 1]
        elif i > 1 and words[i - 2] in NEGATIONS:
            is_negated = True
            neg_word = words[i - 2]
            
        # Kiểm tra từ tăng cường mức độ
        intensifier = ""
        multiplier = 1.0
        if i > 0 and words[i - 1] in INTENSIFIERS:
            intensifier = words[i - 1]
            multiplier = INTENSIFIERS[words[i - 1]]
            
        if w in POSITIVE_LEXICON:
            base_val = POSITIVE_LEXICON[w]
            if is_negated:
                phrase = f"{neg_word} {w}" if not intensifier else f"{neg_word} {intensifier} {w}"
                eff_val = round(-base_val * 0.8 * multiplier, 2)
                neg_evidences.append(f"'{phrase}' (đảo nghĩa: {eff_val})")
                total_score += eff_val
            else:
                phrase = f"{intensifier} {w}".strip()
                eff_val = round(base_val * multiplier, 2)
                pos_evidences.append(f"'{phrase}' (+{eff_val})")
                total_score += eff_val
            match_count += 1
            
        elif w in NEGATIVE_LEXICON:
            base_val = NEGATIVE_LEXICON[w]
            if is_negated:
                phrase = f"{neg_word} {w}" if not intensifier else f"{neg_word} {intensifier} {w}"
                eff_val = round(-base_val * 0.8 * multiplier, 2)
                pos_evidences.append(f"'{phrase}' (giảm tiêu cực: +{eff_val})")
                total_score += eff_val
            else:
                phrase = f"{intensifier} {w}".strip()
                eff_val = round(base_val * multiplier, 2)
                neg_evidences.append(f"'{phrase}' ({eff_val})")
                total_score += eff_val
            match_count += 1
            
        i += 1

    # Phát hiện sắc thái Sarcasm (Mỉa mai / Châm biếm)
    sarcasm_detected = False
    raw_lower = str(raw_text).lower()
    if any(k in raw_lower for k in [":)))", "=)))", "🐧", "🤡", "cười_lớn", "mỉa_mai"]) and pos_evidences:
        if any(k in raw_lower for k in ["moodle", "wifi", "thang máy", "bãi xe", "học phí", "quá tải", "kẹt", "hèn"]):
            total_score = -abs(total_score) * 1.3
            sarcasm_detected = True

    # Chuẩn hóa điểm phân cực [-1.0, 1.0]
    if match_count > 0:
        norm_score = total_score / (match_count * 1.5)
        norm_score = max(-1.0, min(1.0, norm_score))
    else:
        norm_score = 0.0
        
    norm_score = round(norm_score, 3)

    # Quyết định nhãn cảm xúc
    if sarcasm_detected:
        sentiment = "NEGATIVE"
    elif target == "PERSONAL" and abs(norm_score) < 0.35:
        sentiment = "NEUTRAL"
    elif norm_score > 0.15:
        sentiment = "POSITIVE"
    elif norm_score < -0.15:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    # 1. Trích xuất danh sách từ khóa chứng cứ (key_evidence_tokens)
    key_tokens = []
    if sentiment == "NEGATIVE":
        for e in neg_evidences:
            tk = e.split()[0].replace("'", "")
            key_tokens.append(tk)
    elif sentiment == "POSITIVE":
        for e in pos_evidences:
            tk = e.split()[0].replace("'", "")
            key_tokens.append(tk)
    else:
        for e in (pos_evidences + neg_evidences):
            tk = e.split()[0].replace("'", "")
            key_tokens.append(tk)
            
    key_tokens_str = ", ".join(list(dict.fromkeys(key_tokens))[:5]) if key_tokens else "N/A (Trung tính)"

    # 2. Chi tiết căn cứ (sentiment_evidence)
    parts = []
    if pos_evidences:
        parts.append(f"Tín hiệu Tích cực: {', '.join(pos_evidences[:4])}")
    if neg_evidences:
        parts.append(f"Tín hiệu Tiêu cực: {', '.join(neg_evidences[:4])}")
    if sarcasm_detected:
        parts.append("Dấu hiệu mỉa mai: khen ngợi bề mặt nhưng kèm emoticon cười khẩy đối nghịch trong vấn đề bức xúc")
        
    sentiment_evidence_str = " | ".join(parts) if parts else "Không chứa từ ngữ cảm xúc mạnh (văn bản trung tính)"

    # 3. Diễn giải lý do khoa học tự nhiên (explanation_rationale)
    aspect_vn = {
        "CO_SO_VAT_CHAT": "Cơ sở vật chất & Dịch vụ",
        "DAO_TAO_GIANGVIEN": "Đào tạo & Giảng viên",
        "HOCVU_CHINHSACH": "Học vụ & Chính sách",
        "DOISONG_PHONGTRAO": "Đời sống & Phong trào",
        "CA_NHAN_TINH_CAM": "Tâm sự tình cảm & Cá nhân"
    }.get(primary_aspect, str(primary_aspect))

    if sentiment == "NEGATIVE":
        if sarcasm_detected:
            explanation = f"Xác định TIÊU CỰC do phát hiện sắc thái châm biếm/mỉa mai đối với {aspect_vn} (từ ngữ tích cực bị vô hiệu hóa bởi thái độ chỉ trích và biểu tượng cười cợt)."
        elif neg_evidences:
            sample_clues = ", ".join(neg_evidences[:3])
            explanation = f"Xác định TIÊU CỰC do bài viết xuất hiện các từ ngữ phàn nàn/bức xúc ({sample_clues}), làm điểm phân cực giảm xuống mức {norm_score:.2f}."
        else:
            explanation = f"Xác định TIÊU CỰC dựa trên ngữ cảnh bày tỏ sự khó khăn, bất bình về {aspect_vn}."
            
    elif sentiment == "POSITIVE":
        if pos_evidences:
            sample_clues = ", ".join(pos_evidences[:3])
            explanation = f"Xác định TÍCH CỰC do bài viết xuất hiện các từ ngữ khen ngợi/biết ơn/hài lòng ({sample_clues}), đạt điểm phân cực cao +{norm_score:.2f}."
        else:
            explanation = f"Xác định TÍCH CỰC dựa trên thái độ ủng hộ, động viên hoặc tình cảm yêu mến trong {aspect_vn}."
            
    else: # NEUTRAL
        if pos_evidences and neg_evidences:
            explanation = f"Xác định TRUNG LẬP do cảm xúc pha trộn (Mixed Sentiment): có cả từ khen ({len(pos_evidences)}) và từ chê ({len(neg_evidences)}) triệt tiêu lẫn nhau, điểm số tổng thể đạt mức cân bằng ({norm_score:.2f})."
        elif target == "PERSONAL":
            explanation = "Xác định TRUNG LẬP đối với Nhà trường (Nội dung mang tính chất chia sẻ đời sống cá nhân, tìm bạn bè/crush, không có ý kiến đánh giá trường học)."
        else:
            explanation = "Xác định TRUNG LẬP do bài viết mang tính chất hỏi đáp thông tin, thắc mắc thủ tục học vụ khách quan, không biểu lộ thái độ khen chê vượt ngưỡng."

    return {
        "sentiment": sentiment,
        "polarity_score": norm_score,
        "evidence_keywords": key_tokens_str,
        "sentiment_evidence": sentiment_evidence_str,
        "explanation_rationale": explanation
    }

def main():
    print("=" * 75)
    print("   HỆ THỐNG TRÍCH XUẤT CHỨNG CỨ MINH CHỨNG CẢM XÚC (EVIDENCE EXTRACTOR)")
    print("=" * 75)
    
    input_csv = "reports/sentiment_results_full.csv"
    if not os.path.exists(input_csv):
        input_csv = "data/processed/clean_confessions.csv"
        
    df = pd.read_csv(input_csv)
    print(f"-> Đang nạp {len(df)} confessions để trích xuất minh chứng...")
    
    sents = []
    scores = []
    keywords = []
    evidences = []
    explanations = []
    
    for idx, row in df.iterrows():
        tokenized = str(row.get("tokenized_text", ""))
        raw = str(row.get("raw_text", ""))
        target = str(row.get("target", "SCHOOL"))
        aspect = str(row.get("primary_aspect", "CHUNG"))
        
        res = analyze_confession_evidence(tokenized, raw, target, aspect)
        sents.append(res["sentiment"])
        scores.append(res["polarity_score"])
        keywords.append(res["evidence_keywords"])
        evidences.append(res["sentiment_evidence"])
        explanations.append(res["explanation_rationale"])
        
    df["sentiment_predicted"] = sents
    df["polarity_score"] = scores
    df["evidence_keywords"] = keywords
    df["sentiment_evidence"] = evidences
    df["explanation_rationale"] = explanations
    
    # Lưu file kết quả có minh chứng đầy đủ
    evidence_csv = "reports/sentiment_results_with_evidence.csv"
    df.to_csv(evidence_csv, index=False, encoding="utf-8-sig")
    print(f"-> Đã lưu tệp kết quả kèm chứng cứ đầy đủ vào: {evidence_csv}")
    
    # Cập nhật trực tiếp vào sentiment_results_full.csv
    df.to_csv("reports/sentiment_results_full.csv", index=False, encoding="utf-8-sig")
    print(f"-> Đã đồng bộ vào tệp chuẩn: reports/sentiment_results_full.csv")
    
    print("\n" + "-" * 75)
    print("MẪU 3 CONFESSION KÈM CHỨNG CỨ MINH CHỨNG THỰC TẾ:")
    print("-" * 75)
    sample_indices = [0, 2, 4] if len(df) > 4 else range(len(df))
    for i in sample_indices:
        r = df.iloc[i]
        print(f"\n[CFS #{r['cfs_id']} - Năm {r['year']}] Nhãn: {r['sentiment_predicted']} (Điểm: {r['polarity_score']})")
        print(f"  Văn bản gốc:      {r['raw_text'][:110]}...")
        print(f"  Từ khóa chứng cứ: {r['evidence_keywords']}")
        print(f"  Chi tiết căn cứ:  {r['sentiment_evidence']}")
        print(f"  Diễn giải lý do:  {r['explanation_rationale']}")
    print("=" * 75)

if __name__ == "__main__":
    main()

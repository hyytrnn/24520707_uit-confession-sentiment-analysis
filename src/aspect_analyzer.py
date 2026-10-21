"""
Aspect-Based & Target Extraction Module for UIT Confessions
Identifies whether a confession is school-related and categorizes specific aspects.
"""

import re

ASPECT_KEYWORDS = {
    "CO_SO_VAT_CHAT": [
        "thang máy", "bãi xe", "giữ xe", "gửi xe", "nhà xe", "bãi đậu", "ô tô", "xe máy",
        "wifi", "mạng", "kết nối", "điều hòa", "máy lạnh", "máy chiếu", "bàn ghế", "ổ cắm",
        "thư viện", "nhà vệ sinh", "nvs", "wc", "phòng học", "căn tin", "canteen",
        "tòa a", "tòa b", "tòa c", "tòa e", "b2", "b3", "b4", "b5", "b6", "b7",
        "hội trường", "cơ sở vật chất", "hạ tầng", "loa", "quạt", "đèn"
    ],
    "DAO_TAO_GIANGVIEN": [
        "thầy", "cô", "giảng viên", "môn học", "học phần", "đồ án", "bài tập", "lab", "wecode",
        "thi cử", "thi lại", "đề thi", "giữa kỳ", "cuối kỳ", "điểm số", "bảng điểm", "gpa",
        "cấu trúc dữ liệu", "dsa", "hướng đối tượng", "oop", "nhập môn lập trình", "nmlt",
        "hệ điều hành", "cơ sở dữ liệu", "mạng máy tính", "toán", "giải tích", "đại số",
        "khóa luận", "kltn", "chuyên ngành", "đại cương", "dạy", "chấm điểm", "giảng", "phản biện"
    ],
    "HOCVU_CHINHSACH": [
        "học phí", "tiền học", "học bổng", "khuyến khích học tập", "hbkkht", "giải ngân",
        "đăng ký học phần", "dkhp", "mở lớp", "hủy môn", "phòng đào tạo", "công tác sinh viên",
        "ctsv", "daa", "forum", "phúc khảo", "giới hạn tín chỉ", "tín chỉ", "chuẩn đầu ra",
        "chứng chỉ", "toeic", "tiếng anh", "anh văn", "bảo lưu", "chuyển ngành", "xóa học phần",
        "cấm thi", "quy chế", "học vụ"
    ],
    "DOISONG_PHONGTRAO": [
        "đoàn", "hội", "bch", "đoàn trường", "clb", "câu lạc bộ", "gala", "sự kiện",
        "chiến dịch", "mùa hè xanh", "mhx", "xuân tình nguyện", "xtn", "hành quân xanh",
        "thủ lĩnh sinh viên", "quân sự", "trung tâm giáo dục quốc phòng", "kqs", "hội thao",
        "sinh viên 5 tốt", "sv5t", "uit media", "lossless", "csac", "webdev", "bht", "tuyên dương"
    ],
    "CA_NHAN_TINH_CAM": [
        "crush", "người yêu", "ny", "tỏ tình", "làm quen", "xin info", "in4", "dễ thương",
        "đẹp trai", "xinh gái", "pass đồ", "pass lại", "balo", "mua lại", "bán lại",
        "tìm đồ", "đánh rơi", "bỏ quên", "rơi ví", "mất thẻ", "ở ghép", "tìm trọ",
        "bạn cùng phòng", "hẹn hò", "stalk", "say nắng"
    ]
}

class AspectAnalyzer:
    def __init__(self):
        self.aspect_patterns = {}
        for aspect, kw_list in ASPECT_KEYWORDS.items():
            # Build regex pattern for word boundary / flexible matching
            escaped_kws = [re.escape(kw) for kw in kw_list]
            self.aspect_patterns[aspect] = re.compile(r'\b(' + '|'.join(escaped_kws) + r')\b', re.IGNORECASE)

    def extract_aspects(self, text: str):
        """
        Xác định các khía cạnh (aspects) được đề cập trong bài viết.
        Trả về danh sách các aspects và mức độ liên quan đến nhà trường.
        """
        matched_aspects = []
        aspect_scores = {}
        
        for aspect, pattern in self.aspect_patterns.items():
            matches = pattern.findall(text)
            if matches:
                matched_aspects.append(aspect)
                aspect_scores[aspect] = len(matches)
                
        # Xác định đối tượng chính: 'SCHOOL' (Nhà trường) hay 'PERSONAL_OTHER' (Chuyện cá nhân/tình cảm)
        school_aspects = ["CO_SO_VAT_CHAT", "DAO_TAO_GIANGVIEN", "HOCVU_CHINHSACH", "DOISONG_PHONGTRAO"]
        school_score = sum(aspect_scores.get(asp, 0) for asp in school_aspects)
        personal_score = aspect_scores.get("CA_NHAN_TINH_CAM", 0)
        
        if school_score > 0:
            target = "SCHOOL"
            # Tìm primary aspect có điểm cao nhất trong school aspects
            primary_aspect = max(school_aspects, key=lambda a: aspect_scores.get(a, 0))
        elif personal_score > 0:
            target = "PERSONAL"
            primary_aspect = "CA_NHAN_TINH_CAM"
        else:
            target = "OTHER"
            primary_aspect = "KHONG_XAC_DINH"
            
        return {
            "target": target,
            "primary_aspect": primary_aspect,
            "matched_aspects": matched_aspects,
            "school_score": school_score,
            "personal_score": personal_score
        }

if __name__ == "__main__":
    analyzer = AspectAnalyzer()
    samples = [
        "Thang máy tòa B sáng nay kẹt từ tầng 1 đến tầng G, bãi giữ xe thì quá tải không có chỗ đậu!",
        "Thầy Dũng dạy OOP siêu nhiệt tình và tận tâm, bài tập thực hành dễ hiểu cực kỳ.",
        "Em muốn tìm bạn nam mặc áo trắng ngồi thư viện hôm qua, bạn dễ thương quá cho mình xin in4 với ạ."
    ]
    for s in samples:
        print(s[:60] + "...")
        print(analyzer.extract_aspects(s))

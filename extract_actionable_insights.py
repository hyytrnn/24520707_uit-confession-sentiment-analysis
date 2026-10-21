"""
Extract Actionable Administrative Insights & Issues from Negative School Confessions
Analyzes the 95 negative school confessions to pinpoint specific locations, facilities, and procedures.
"""

import os
import sys
import re
import pandas as pd
from collections import Counter

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def extract_insights(csv_path="reports/sentiment_results_full.csv"):
    df = pd.read_csv(csv_path)
    school_neg = df[(df["target"] == "SCHOOL") & (df["sentiment_lexicon"] == "NEGATIVE")].copy()
    
    print(f"Tổng số confession tiêu cực về nhà trường: {len(school_neg)} bài.")
    
    # Định nghĩa các chủ đề cụ thể (Specific Problem Clusters)
    topic_rules = {
        "Hạ tầng Mạng, Wifi & Hệ thống Moodle / ĐKHP": [
            r"wifi", r"mạng.*lag", r"moodle", r"đkhp", r"đăng ký học phần", r"server", r"nghẽn mạng", r"mạng.*yếu", r"mạng.*chậm"
        ],
        "Thang máy tòa B & Di chuyển": [
            r"thang máy", r"chờ thang", r"kẹt thang", r"chen lấn.*thang", r"thang.*tòa b"
        ],
        "Bãi giữ xe, Tầng hầm & Trật tự an ninh": [
            r"bãi xe", r"giữ xe", r"gửi xe", r"tầng hầm.*xe", r"hầm xe", r"dắt xe", r"quẹt thẻ.*xe", r"mất nón", r"mất mũ"
        ],
        "Trang thiết bị phòng học (Máy chiếu, Máy lạnh, Mic, Bàn ghế)": [
            r"máy chiếu", r"máy lạnh", r"điều hòa", r"quạt trần", r"micro", r"âm thanh.*phòng", r"bàn ghế.*hư", r"phòng.*nóng"
        ],
        "Nhà vệ sinh & Vệ sinh môi trường học đường": [
            r"nhà vệ sinh", r"toilet", r"wc", r"mùi hôi", r"hết nước", r"rác"
        ],
        "Chính sách Học bổng, Học phí & Điểm rèn luyện": [
            r"học bổng", r"hbkkht", r"học phí", r"tiền học", r"điểm rèn luyện", r"drl", r"công nợ"
        ],
        "Quy chế thi, Học vụ & Chuẩn đầu ra Ngoại ngữ": [
            r"cấm thi", r"phúc khảo", r"đổi điểm", r"chuẩn đầu ra", r"toeic", r"ielts", r"chứng chỉ", r"rớt môn"
        ],
        "Dịch vụ Sinh viên, Thủ tục Một cửa & Căn tin": [
            r"phòng đào tạo", r"pđt", r"ctsv", r"thủ tục", r"giấy xác nhận", r"căn tin", r"canteen"
        ]
    }
    
    clustered_records = []
    
    for idx, row in school_neg.iterrows():
        text = str(row["clean_text"]).lower()
        raw = str(row["raw_text"])
        matched_topics = []
        
        for topic, patterns in topic_rules.items():
            if any(re.search(pat, text) for pat in patterns):
                matched_topics.append(topic)
                
        if not matched_topics:
            matched_topics.append("Vấn đề học vụ & Đời sống chung khác")
            
        for top in matched_topics:
            clustered_records.append({
                "cfs_id": row.get("cfs_id", f"CFS_{idx}"),
                "year": row["year"],
                "topic": top,
                "aspect": row["primary_aspect"],
                "raw_text": raw[:300] + "..." if len(raw) > 300 else raw
            })
            
    res_df = pd.DataFrame(clustered_records)
    
    # Tổng kết theo chủ đề
    summary = res_df["topic"].value_counts().reset_index()
    summary.columns = ["Chủ đề bức xúc / Phàn nàn", "Số lượng bài phản ánh"]
    summary["Tỷ lệ (%)"] = (summary["Số lượng bài phản ánh"] / len(school_neg) * 100).round(1)
    
    print("\n" + "=" * 70)
    print("DANH SÁCH CÁC ĐIỂM NÓNG CƠ SỞ VẬT CHẤT & HÀNH CHÍNH (ACTIONABLE INSIGHTS):")
    print("=" * 70)
    print(summary.to_string(index=False))
    
    summary.to_csv("reports/actionable_insights_summary.csv", index=False, encoding="utf-8-sig")
    res_df.to_csv("reports/actionable_insights_detailed_cases.csv", index=False, encoding="utf-8-sig")
    print("\n-> Đã lưu các phát hiện thực tế vào: reports/actionable_insights_summary.csv")
    return summary, res_df

if __name__ == "__main__":
    extract_insights()

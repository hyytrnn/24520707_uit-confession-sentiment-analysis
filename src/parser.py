"""
Parser and Dataset Builder for Multi-Year UIT Confessions
Extracts, decomposes, anonymizes and standardizes confession texts.
"""

import os
import sys
import re
import glob
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.append(os.path.dirname(__file__))
from preprocessor import ConfessionPreprocessor

def extract_sub_confessions(post_text):
    """
    Tách các confession con bên trong một bài post Facebook.
    Ví dụ: '#3089 Nội dung 1 ----------- #3090 Nội dung 2'
    -> [('3089', 'Nội dung 1'), ('3090', 'Nội dung 2')]
    """
    # 1. Tìm các hashtag #<số> ở đầu dòng hoặc sau các dấu phân cách
    pattern = r'(?:^|[\r\n]|[-]{3,})\s*#\s*(?:UIT|cfs|CFS)?\s*(\d+)'
    matches = list(re.finditer(pattern, post_text, flags=re.IGNORECASE))
    
    # Nếu không tìm thấy ở đầu dòng, thử tìm ở bất kỳ đâu
    if not matches:
        pattern_any = r'#\s*(?:UIT|cfs|CFS)?\s*(\d+)'
        matches = list(re.finditer(pattern_any, post_text, flags=re.IGNORECASE))
        
    if not matches:
        clean = post_text.strip()
        # Loại bỏ các từ footer thừa
        clean = re.sub(r'(?:Ẩn bớt|See less|Xem bớt|See more|Xem thêm)\s*$', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'#UITconfessions:.*$', '', clean, flags=re.IGNORECASE).strip()
        if len(clean) > 15:
            return [("", clean)]
        return []
        
    results = []
    for i, match in enumerate(matches):
        cfs_id = match.group(1)
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(post_text)
        
        cfs_text = post_text[start_pos:end_pos].strip()
        # Xóa các ký tự phân cách ở đầu/cuối (: , - , . , ----)
        cfs_text = re.sub(r'^[:\-\.\s_]+', '', cfs_text).strip()
        cfs_text = re.sub(r'[-]{3,}.*$', '', cfs_text).strip()
        cfs_text = re.sub(r'(?:Ẩn bớt|See less|Xem bớt|See more|Xem thêm)\s*$', '', cfs_text, flags=re.IGNORECASE).strip()
        cfs_text = re.sub(r'#UITconfessions:.*$', '', cfs_text, flags=re.IGNORECASE).strip()
        
        if len(cfs_text) > 10:
            results.append((cfs_id, cfs_text))
            
    return results

def parse_all_confessions(raw_dir="data/raw"):
    """Duyệt và phân tích toàn bộ 5 năm confession."""
    processor = ConfessionPreprocessor(use_word_segmentation=True)
    processor_no_seg = ConfessionPreprocessor(use_word_segmentation=False)
    
    records = []
    file_paths = sorted(glob.glob(os.path.join(raw_dir, "confessions-*.txt")))
    
    if not file_paths:
        # Nếu chưa chuyển vào data/raw thì lấy ở thư mục hiện tại
        file_paths = sorted(glob.glob("confessions-*.txt"))
        
    print(f"Tìm thấy {len(file_paths)} file confession...")
    
    for fpath in file_paths:
        # Lấy năm từ tên file (ví dụ confessions-2024.txt -> 2024)
        m_year = re.search(r'confessions-(\d{4})', os.path.basename(fpath))
        year = int(m_year.group(1)) if m_year else 0
        
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
            content = fp.read()
            
        posts = re.split(r'===== POST \d+ =====', content)
        post_idx = 0
        for p in posts:
            p = p.strip()
            if not p:
                continue
            
            sub_cfs = extract_sub_confessions(p)
            for cfs_id, cfs_text in sub_cfs:
                clean_text = processor_no_seg.process(cfs_text)
                tokenized_text = processor.process(cfs_text)
                
                records.append({
                    "year": year,
                    "post_index": post_idx,
                    "cfs_id": cfs_id,
                    "raw_text": cfs_text,
                    "clean_text": clean_text,
                    "tokenized_text": tokenized_text,
                    "length": len(cfs_text.split())
                })
            post_idx += 1
            
    df = pd.DataFrame(records)
    print(f"Tổng số confession đã trích xuất: {len(df)}")
    print(df["year"].value_counts().sort_index())
    
    # Lưu file xử lý chuẩn
    os.makedirs("data/processed", exist_ok=True)
    out_csv = "data/processed/clean_confessions.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Đã lưu tập dữ liệu chuẩn hóa vào: {out_csv}")
    
    return df

if __name__ == "__main__":
    df = parse_all_confessions()

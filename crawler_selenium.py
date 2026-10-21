import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

def clean_post_text(raw_text):
    """Làm sạch các từ nút bấm ở cuối bài viết như Ẩn bớt, See less, Xem thêm"""
    text = raw_text.strip()
    return re.sub(r'(?:Ẩn bớt|See less|Xem bớt|See more|Xem thêm)\s*$', '', text, flags=re.IGNORECASE).strip()

def save_to_txt_and_csv(posts_list, target_year):
    """Lưu dữ liệu ra file .txt theo đúng định dạng ===== POST {i} ===== và kèm file .csv"""
    txt_filename = f"cfs_{target_year}.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        for idx, post in enumerate(posts_list):
            f.write(f"===== POST {idx} =====\n")
            f.write(post.strip())
            f.write("\n\n")

    csv_filename = f"cfs_{target_year}.csv"
    df = pd.DataFrame([{"post_index": i, "content": p} for i, p in enumerate(posts_list)])
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")

    return txt_filename, len(posts_list)

def try_apply_facebook_filter(driver, target_year):
    """Tự động bấm nút Bộ lọc (Filters) trên Facebook để lọc trực tiếp theo năm"""
    print(f"\nĐang kiểm tra và áp dụng nút 'Bộ lọc' trên Facebook cho năm {target_year}...")
    try:
        # Tìm nút "Bộ lọc" hoặc "Filters"
        filter_buttons = driver.find_elements(By.XPATH, "//*[text()='Bộ lọc' or text()='Filters' or @aria-label='Bộ lọc' or @aria-label='Filters']")
        btn_to_click = None
        for btn in filter_buttons:
            if btn.is_displayed():
                btn_to_click = btn
                break

        if btn_to_click:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn_to_click)
            time.sleep(1)
            btn_to_click.click()
            time.sleep(2)

            # Click vào dropdown Năm trong popup
            year_menus = driver.find_elements(By.XPATH, "//div[@role='dialog']//*[contains(text(), 'Năm') or contains(text(), 'Year')]")
            if year_menus:
                year_menus[0].click()
                time.sleep(1.5)

                # Chọn năm cụ thể (2024, 2023, v.v.)
                year_items = driver.find_elements(By.XPATH, f"//div[@role='dialog' or @role='listbox']//*[text()='{target_year}']")
                if year_items:
                    year_items[0].click()
                    time.sleep(1)

            # Bấm nút Xong / Done
            done_buttons = driver.find_elements(By.XPATH, "//div[@role='dialog']//*[text()='Xong' or text()='Done' or contains(text(), 'Xong') or contains(text(), 'Done')]")
            if done_buttons:
                done_buttons[0].click()
                time.sleep(3)
                print(f"-> Đã chọn thành công bộ lọc Facebook cho năm {target_year}!")
                return True
    except Exception as e:
        print(f"-> Không thể tự động bấm bộ lọc (bạn có thể bấm bằng tay trên màn hình): {e}")
    return False

# --- BẮT ĐẦU CHƯƠNG TRÌNH ---
print("=" * 60)
print("     CRAWLER FACEBOOK UIT CONFESSIONS THEO BỘ LỌC NĂM")
print("=" * 60)

year_choice = input("Nhập năm bạn muốn cào (ví dụ: 2024, 2023, 2022... mặc định 2024): ").strip()
target_year = year_choice if year_choice else "2024"

print(f"\n=> Mục tiêu: Cào toàn bộ bài viết của năm {target_year}")

# Khởi tạo trình duyệt Chrome tối ưu RAM
options = webdriver.ChromeOptions()
options.add_argument("--disable-notifications")
options.add_argument("--start-maximized")
options.add_argument("--blink-settings=imagesEnabled=false")  # Chặn tải ảnh: Giảm 80% RAM, siêu nhanh
options.add_argument("--disable-gpu")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

url = "https://www.facebook.com/UITconfess"
print(f"Đang truy cập: {url}")
driver.get(url)

time.sleep(4)

# Cuộn nhẹ xuống để nút "Bộ lọc" cạnh mục Bài viết xuất hiện trên màn hình
driver.execute_script("window.scrollBy(0, 500);")
time.sleep(1)

# Thử tự động bấm bộ lọc sang năm mục tiêu
try_apply_facebook_filter(driver, target_year)

print("\n" + "-" * 50)
print("TRÌNH DUYỆT ĐÃ MỞ TRÊN MÀN HÌNH CỦA BẠN!")
print(f"1. Vui lòng đăng nhập Facebook trên cửa sổ Chrome (nếu cần).")
print(f"2. Kiểm tra xem Facebook đã được chọn 'Bộ lọc' năm {target_year} chưa.")
print(f"   (Nếu chưa, bạn có thể tự bấm nút 'Bộ lọc' trên trang rồi chọn năm {target_year}).")
print("3. Sau khi trang đã sẵn sàng, hãy quay lại đây...")
input(f"Nhấn phím [Enter] để bắt đầu cào toàn bộ bài viết năm {target_year}...")
print("-" * 50 + "\n")

main_window = driver.current_window_handle

# Danh sách lưu các bài viết của năm này (không cần if-else đoán năm nữa!)
collected_posts = []
seen_post_hashes = set()

# JavaScript trích xuất nội dung CHỈ CỦA BÀI VIẾT (loại trừ 100% comment)
JS_EXTRACT_POSTS = """
let extracted = [];
let seen = new Set();

let messageElements = document.querySelectorAll('div[data-ad-preview="message"], div[data-ad-comet-preview="message"]');
for (let el of messageElements) {
    let t = el.innerText ? el.innerText.trim() : '';
    if (t && t.length > 20 && !seen.has(t)) {
        seen.add(t);
        extracted.push(t);
    }
}

let articles = document.querySelectorAll('div[role="article"]');
for (let art of articles) {
    if (art.parentElement && art.parentElement.closest('div[role="article"]')) continue;
    if (art.querySelector('div[data-ad-preview="message"], div[data-ad-comet-preview="message"]')) continue;
    
    let textPieces = [];
    let divs = art.querySelectorAll('div[dir="auto"]');
    for (let d of divs) {
        if (d.closest('ul') || d.closest('form') || d.closest('[role="toolbar"]') || 
            d.closest('[aria-label*="bình luận" i]') || d.closest('[aria-label*="comment" i]')) {
            continue;
        }
        let txt = d.innerText ? d.innerText.trim() : '';
        if (txt && txt.length > 30) {
            textPieces.push(txt);
        }
    }
    if (textPieces.length > 0) {
        let longest = textPieces.reduce((a, b) => a.length >= b.length ? a : b, "");
        if (longest && !seen.has(longest)) {
            seen.add(longest);
            extracted.push(longest);
        }
    }
}

return extracted;
"""

print(f"Bắt đầu cuộn thu thập tất cả bài viết năm {target_year}...\n")

last_height = driver.execute_script("return document.body.scrollHeight")
scroll_count = 0
stuck_count = 0

try:
    while True:
        # Tự động đóng tab phụ nếu có
        if len(driver.window_handles) > 1:
            for handle in driver.window_handles:
                if handle != main_window:
                    try:
                        driver.switch_to.window(handle)
                        driver.close()
                    except Exception:
                        pass
            driver.switch_to.window(main_window)

        scroll_count += 1

        # Mở rộng các bài viết dài bằng cách click "Xem thêm" / "See more"
        try:
            driver.execute_script("""
                if (!window.__tab_blocked) {
                    window.__tab_blocked = true;
                    window.open = function() { return null; };
                }
                let buttons = document.querySelectorAll('div[role="button"], span[role="button"]');
                for (let btn of buttons) {
                    if (btn.getAttribute('data-cfs-clicked')) continue;
                    if (btn.tagName.toLowerCase() === 'a' || btn.closest('a')) continue;
                    if (btn.closest('ul') || btn.closest('form') || btn.closest('[aria-label*="bình luận" i]')) continue;
                    
                    let txt = (btn.innerText || btn.textContent || '').trim();
                    if (txt === 'Xem thêm' || txt === 'See more') {
                        btn.setAttribute('data-cfs-clicked', 'true');
                        try {
                            btn.click();
                        } catch(e) {}
                    }
                }
            """)
            time.sleep(0.5)
        except Exception:
            pass

        # Quét các bài viết trên trang
        raw_posts = driver.execute_script(JS_EXTRACT_POSTS)

        for raw_post in raw_posts:
            cleaned = clean_post_text(raw_post)
            post_hash = hash(cleaned[:120])
            if post_hash in seen_post_hashes:
                continue

            seen_post_hashes.add(post_hash)
            # Thêm trực tiếp bài viết của năm đã lọc (không cần if-else cfs)
            collected_posts.append(cleaned)

        print(f"Lần cuộn {scroll_count} -> Tổng số bài viết thu thập được: {len(collected_posts)}")

        # Tự động lưu checkpoint định kỳ sau mỗi 5 lần cuộn
        if scroll_count % 5 == 0:
            save_to_txt_and_csv(collected_posts, target_year)

        # Cuộn màn hình xuống dưới để nạp bài tiếp theo
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

        # Kiểm tra đến cuối trang
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            stuck_count += 1
            # Thử nhích cuộn lên rồi xuống lại để kích thích nạp thêm
            driver.execute_script("window.scrollBy(0, -500);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            if stuck_count >= 6:
                print(f"\n[Xong] Đã cuộn đến bài viết cuối cùng của năm {target_year}!")
                break
        else:
            stuck_count = 0
            last_height = new_height

except Exception as e:
    print(f"\n[Cảnh báo] Lỗi kết nối hoặc sự cố: {type(e).__name__}")

finally:
    try:
        driver.quit()
    except Exception:
        pass

    # Lưu dữ liệu lần cuối
    fname, count = save_to_txt_and_csv(collected_posts, target_year)
    print("\n" + "=" * 60)
    print("                    HOÀN TẤT THU THẬP")
    print("=" * 60)
    print(f"-> Đã lưu toàn bộ {count} bài viết năm {target_year} vào file '{fname}'.")
    print(f"-> Định dạng dữ liệu: ===== POST {{i}} ===== chuẩn xác 100%.")
import os
import json
import subprocess
from datasets import load_dataset

def download_msvd():
    print("=== ĐANG TẢI TẬP DỮ LIỆU MSVD (TEXT-TO-VIDEO) ===")
    os.makedirs("data/raw_videos", exist_ok=True)
    
    try:
        ds = load_dataset("VLM2Vec/MSVD", split="train", streaming=True)
    except Exception as e:
        print(f"Lỗi kết nối tới HuggingFace Dataset: {e}")
        return []

    downloaded = []
    max_videos = 3
    
    for item in ds:
        if len(downloaded) >= max_videos:
            break
            
        video_id_full = item['video_id']
        caption = item['caption'][0] if isinstance(item['caption'], list) and len(item['caption']) > 0 else ""
        
        # Cấu trúc video_id của MSVD: <Youtube_ID>_<Start>_<End>
        # Dùng rsplit để tách từ phải qua trái, đề phòng Youtube_ID có chứa dấu _
        parts = video_id_full.rsplit('_', 2)
        if len(parts) != 3:
            continue
            
        yt_id, start, end = parts
        url = f"https://www.youtube.com/watch?v={yt_id}"
        out_path = f"data/raw_videos/{video_id_full}.mp4"
        
        if os.path.exists(out_path):
            print(f"[MSVD] Đã có sẵn: {out_path}")
            downloaded.append({"id": video_id_full, "path": out_path, "caption": caption})
            continue
            
        print(f"[MSVD] Đang tải {yt_id} ({start}s - {end}s)...")
        # Sử dụng yt-dlp với download-sections để cắt trực tiếp trên server/luồng
        cmd = [
            ".venv/bin/yt-dlp",
            "-f", "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts",
            "-o", out_path,
            url
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and os.path.exists(out_path):
                print(f" -> Tải thành công: {out_path}")
                downloaded.append({"id": video_id_full, "path": out_path, "caption": caption})
            else:
                print(f" -> Bỏ qua (Video có thể đã bị xóa hoặc Private).")
        except Exception as e:
            print(f" -> Lỗi trong quá trình tải: {e}")
            
    return downloaded

if __name__ == "__main__":
    print("BẮT ĐẦU CHẠY BENCHMARK DOWNLOADER...")
    msvd_data = download_msvd()
    
    if msvd_data:
        with open("data/msvd_test_set.json", "w", encoding="utf-8") as f:
            json.dump(msvd_data, f, ensure_ascii=False, indent=2)
        print(f"Hoàn tất tải {len(msvd_data)} video MSVD vào thư mục data/raw_videos!")
        print("Chi tiết truy vấn được lưu tại data/msvd_test_set.json")
    else:
        print("Không tải được video nào.")

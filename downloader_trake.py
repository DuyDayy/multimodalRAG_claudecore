import os
import json
import subprocess
from datasets import load_dataset

def download_activitynet_trake():
    print("=== ĐANG TẢI TẬP DỮ LIỆU TRAKE (ActivityNet Captions) ===")
    os.makedirs("data/raw_videos", exist_ok=True)
    
    try:
        ds = load_dataset("friedrichor/ActivityNet_Captions", split="val1", streaming=True)
    except Exception as e:
        print(f"Lỗi kết nối tới HuggingFace Dataset: {e}")
        return []

    downloaded = []
    max_videos = 3
    
    for item in ds:
        if len(downloaded) >= max_videos:
            break
            
        yt_id = item['video_id']
        if yt_id.startswith("v_"):
            yt_id = yt_id[2:]
            
        sentences = item['sentences']
        timestamps = item['timestamps']
        
        # Chọn đoạn clip đầu tiên làm ground truth
        if not sentences or not timestamps:
            continue
            
        query = sentences[0]
        start_time, end_time = timestamps[0]
        
        url = f"https://www.youtube.com/watch?v={yt_id}"
        out_path = f"data/raw_videos/TRAKE_{yt_id}.mp4"
        
        if os.path.exists(out_path):
            print(f"[TRAKE] Đã có sẵn: {out_path}")
            downloaded.append({"id": yt_id, "path": out_path, "query": query, "start": start_time, "end": end_time})
            continue
            
        print(f"[TRAKE] Đang tải YouTube ID {yt_id}...")
        cmd = [
            ".venv/bin/yt-dlp",
            "-f", "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--force-keyframes-at-cuts",
            "-o", out_path,
            url
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.exists(out_path):
                print(f" -> Tải thành công: {out_path}")
                downloaded.append({"id": yt_id, "path": out_path, "query": query, "start": start_time, "end": end_time})
            else:
                print(f" -> Bỏ qua (Video có thể đã bị xóa hoặc giới hạn).")
        except Exception as e:
            print(f" -> Lỗi tải: {e}")
            
    return downloaded

if __name__ == "__main__":
    print("BẮT ĐẦU CHẠY ACTIVITYNET TRAKE DOWNLOADER...")
    trake_data = download_activitynet_trake()
    
    if trake_data:
        with open("data/trake_test_set.json", "w", encoding="utf-8") as f:
            json.dump(trake_data, f, ensure_ascii=False, indent=2)
        print(f"Hoàn tất tải {len(trake_data)} video TRAKE vào thư mục data/raw_videos!")
        print("Chi tiết truy vấn được lưu tại data/trake_test_set.json")
    else:
        print("Không tải được video nào.")

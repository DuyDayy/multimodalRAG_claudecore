import os
import json
import subprocess
from datasets import load_dataset

def download_activitynet_qa():
    print("=== ĐANG TẢI TẬP DỮ LIỆU VIDEO QA (ActivityNetQA) ===")
    os.makedirs("data/raw_videos", exist_ok=True)
    
    try:
        ds = load_dataset("lmms-lab/ActivityNetQA", split="test", streaming=True)
        # Bỏ qua 1500 câu đầu tiên (chứa toàn bộ câu hỏi Yes/No giống nhau) để lấy dữ liệu phức tạp
        ds = ds.skip(1500)
    except Exception as e:
        print(f"Lỗi kết nối tới HuggingFace Dataset: {e}")
        return []

    downloaded = []
    max_videos = 3
    
    for item in ds:
        if len(downloaded) >= max_videos:
            break
            
        yt_id = item['video_name'] # VD: 1QIUV7WYKXg
        question = item['question']
        answer = item['answer']
        
        # Link Youtube chuẩn của ActivityNet
        if yt_id.startswith("v_"):
            yt_id = yt_id[2:]
            
        url = f"https://www.youtube.com/watch?v={yt_id}"
        out_path = f"data/raw_videos/ActNet_{yt_id}.mp4"
        
        if os.path.exists(out_path):
            print(f"[QA] Đã có sẵn: {out_path}")
            downloaded.append({"id": yt_id, "path": out_path, "question": question, "answer": answer})
            continue
            
        print(f"[QA] Đang tải YouTube ID {yt_id} (thời lượng đầy đủ để test QA)...")
        # Sử dụng yt-dlp 
        cmd = [
            ".venv/bin/yt-dlp",
            "-f", "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--force-keyframes-at-cuts",
            "-o", out_path,
            url
        ]
        
        try:
            # Video QA có thể dài 1-3 phút, set timeout 300s
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.exists(out_path):
                print(f" -> Tải thành công: {out_path}")
                downloaded.append({"id": yt_id, "path": out_path, "question": question, "answer": answer})
            else:
                print(f" -> Bỏ qua (Video có thể đã bị xóa hoặc giới hạn quốc gia).")
        except Exception as e:
            print(f" -> Lỗi trong quá trình tải: {e}")
            
    return downloaded

if __name__ == "__main__":
    print("BẮT ĐẦU CHẠY ACTIVITYNET QA DOWNLOADER...")
    qa_data = download_activitynet_qa()
    
    if qa_data:
        with open("data/qa_test_set.json", "w", encoding="utf-8") as f:
            json.dump(qa_data, f, ensure_ascii=False, indent=2)
        print(f"Hoàn tất tải {len(qa_data)} video QA vào thư mục data/raw_videos!")
        print("Chi tiết truy vấn được lưu tại data/qa_test_set.json")
    else:
        print("Không tải được video nào.")

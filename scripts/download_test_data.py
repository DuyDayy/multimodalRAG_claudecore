import os
import subprocess
import sys

def download_video(url: str, output_path: str):
    """
    Tải video từ YouTube dùng yt-dlp.
    Độ phân giải 720p để tiết kiệm dung lượng.
    """
    try:
        import yt_dlp
    except ImportError:
        print("Đang cài đặt yt-dlp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
        import yt_dlp

    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        'outtmpl': output_path,
        'noplaylist': True,
    }

    print(f"Đang tải video từ: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Hoàn tất tải video. Lưu tại {output_path}")

if __name__ == "__main__":
    # URL mặc định: Một video phong cảnh 4K dài khoảng 1 giờ (nhưng ta chỉ tải 720p)
    # Nếu URL này chết, người dùng có thể tự thay URL khác.
    # Sử dụng video Big Buck Bunny (khoảng 10 phút) để test luồng mượt mà
    URL = "https://www.youtube.com/watch?v=aqz-KE-bpKQ" 
    
    os.makedirs("data/raw_videos", exist_ok=True)
    out_file = "data/raw_videos/1h_test_video.mp4"
    
    if not os.path.exists(out_file):
        download_video(URL, out_file)
    else:
        print(f"Video đã tồn tại tại {out_file}, bỏ qua bước tải.")

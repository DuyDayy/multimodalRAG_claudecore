import os
import json
import logging
import zipfile
import subprocess
from tqdm import tqdm
try:
    import whisper
except ImportError:
    whisper = None

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def main():
    if not whisper:
        logger.error("Thư viện openai-whisper chưa được cài đặt. Vui lòng chạy: pip install openai-whisper")
        return

    zip_path = '/content/drive/MyDrive/V3C.zip'
    whisper_dir = '/content/drive/MyDrive/multimodalrag_colab/data/whisper'
    os.makedirs(whisper_dir, exist_ok=True)
    output_jsonl = os.path.join(whisper_dir, 'whisper_transcripts.jsonl')
    temp_dir = '/tmp/whisper_extract'
    
    if not os.path.exists(zip_path):
        logger.error(f"Không tìm thấy file {zip_path}")
        return

    os.makedirs(temp_dir, exist_ok=True)

    # Đọc các video đã xử lý để Resume (nếu chạy lại)
    processed_videos = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_videos.add(data.get("video_id"))
                except json.JSONDecodeError:
                    pass
        logger.info(f"Đã tìm thấy {len(processed_videos)} video đã bóc băng. Bỏ qua các video này.")

    logger.info("Đang nạp mô hình Whisper (Base)...")
    model = whisper.load_model("base")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = zf.namelist()
        video_files = [f for f in file_list if f.endswith('.mp4')]
        
        for vid_file in tqdm(video_files, desc="Bóc băng Video"):
            vid_id = os.path.basename(vid_file).split('.')[0]
            
            if vid_id in processed_videos:
                continue
                
            extracted_path = zf.extract(vid_file, path=temp_dir)
            audio_path = os.path.join(temp_dir, f"{vid_id}.wav")
            
            try:
                # Trích xuất audio bằng ffmpeg
                cmd = ["ffmpeg", "-i", extracted_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                # Bóc băng
                result = model.transcribe(audio_path)
                segments = result.get("segments", [])
                
                # Format kết quả
                formatted_segments = []
                for seg in segments:
                    formatted_segments.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip()
                    })
                    
                output_data = {
                    "video_id": vid_id,
                    "segments": formatted_segments
                }
                
                # Ghi ngay ra file JSONL để chống mất dữ liệu
                with open(output_jsonl, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                    
            except Exception as e:
                logger.error(f"Lỗi xử lý audio cho video {vid_id}: {e}")
            finally:
                # Dọn dẹp Temp File
                if os.path.exists(extracted_path):
                    os.remove(extracted_path)
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    
    logger.info(f"=== HOÀN TẤT BÓC BĂNG TOÀN BỘ VIDEO ===")

if __name__ == "__main__":
    main()

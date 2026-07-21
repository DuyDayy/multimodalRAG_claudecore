import os, json, shutil, logging, zipfile, glob
from tqdm import tqdm
from src.ingestion.embedder import MultimodalEmbedder
from src.ingestion.video_processor import process_video

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def run_encoder_pipeline():
    logger.info("=== BẮT ĐẦU QUY TRÌNH SẢN XUẤT DỮ LIỆU (ENCODER PIPELINE) ===")
    zip_path = '/content/drive/MyDrive/V3C.zip'
    
    data_dir = '/content/drive/MyDrive/multimodalrag_colab/data'
    faiss_dir = os.path.join(data_dir, 'faiss_db')
    whisper_dir = os.path.join(data_dir, 'whisper')
    frames_dir = os.path.join(data_dir, 'frames')
    npy_dir = os.path.join(data_dir, 'npy')
    temp_dir = "/tmp/aic_temp_extract"
    
    for d in [faiss_dir, whisper_dir, frames_dir, npy_dir, temp_dir]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(zip_path):
        logger.error(f"Lỗi: Không tìm thấy file {zip_path}.")
        return

    # Khởi tạo Embedder (Chưa có DB thì nó sẽ tạo mới rỗng)
    embedder = MultimodalEmbedder(index_dir=faiss_dir)
    
    logger.info(f"Mở file Streaming từ {zip_path} để bóc tách...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = zf.namelist()
        
        # Load Whisper Transcripts (nếu có)
        whisper_jsonl = os.path.join(whisper_dir, 'whisper_transcripts.jsonl')
        whisper_cache = {}
        if os.path.exists(whisper_jsonl):
            logger.info("Đang nạp file Whisper Transcripts vào bộ nhớ...")
            with open(whisper_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        whisper_cache[data["video_id"]] = data["segments"]
                    except json.JSONDecodeError:
                        pass
            logger.info(f"Đã nạp thoại cho {len(whisper_cache)} videos.")
        
        video_files = [f for f in file_list if f.endswith('.mp4')]
        logger.info(f"Bắt đầu trích xuất On-the-fly {len(video_files)} videos...")
        
        for idx, vid_file in enumerate(tqdm(video_files, desc="Trích xuất Video")):
            vid_id = os.path.basename(vid_file).split('.')[0]
            extracted_path = zf.extract(vid_file, path=temp_dir)
            
            try:
                audio_segments = whisper_cache.get(vid_id, [])
                frames_info = process_video(extracted_path, output_dir=frames_dir, audio_segments=audio_segments)[:15]
            except Exception as e:
                logger.error(f"Lỗi extract {vid_id}: {e}")
                os.remove(extracted_path)
                continue
            
            if frames_info:
                batch_data = [{"path": f["path"], "video_id": vid_id, "timestamp_sec": f["timestamp_sec"], "caption": "", "narrative_context": "", "audio_transcript": f.get("audio_transcript", "")} for f in frames_info]
                embedder.upsert_batch(batch_data)
            
            # Cleanup
            os.remove(extracted_path)
            for f in glob.glob(f"{frames_dir}/*"):
                os.remove(f)
                
        # Lưu file NPY xuống ổ cứng
        embedder.save_index(faiss_dir)
        logger.info("✅ Đã trích xuất và lưu Database NPY thành công!")
        
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    logger.info("HOÀN TẤT ENCODER PIPELINE. GPU ĐÃ ĐƯỢC GIẢI PHÓNG.")

if __name__ == "__main__":
    run_encoder_pipeline()

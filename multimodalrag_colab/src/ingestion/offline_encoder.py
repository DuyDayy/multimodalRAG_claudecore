import os
import time
import logging
import asyncio
from dotenv import load_dotenv
from src.ingestion.video_processor import process_video
from src.ingestion.embedder import MultimodalEmbedder
from src.ingestion.auxiliary_builder import auxiliary_builder

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_narrative_context(frames_info: list, i: int) -> str:
    """
    Xây dựng Narrative Context từ Whisper transcript + OCR (không cần Claude).
    Sử dụng cửa sổ trượt 3 frame (trước - hiện tại - sau).
    """
    current = frames_info[i]
    current_text = current.get("audio_transcript", "")
    
    # Lỗi 1: Thêm kiểm tra ranh giới thời gian (Time Boundary Check) max 15 giây
    TIME_THRESHOLD = 15.0
    
    prev_text = ""
    if i > 0:
        prev_frame = frames_info[i-1]
        if abs(current["timestamp_sec"] - prev_frame["timestamp_sec"]) <= TIME_THRESHOLD:
            prev_text = prev_frame.get("audio_transcript", "")
            
    next_text = ""
    if i < len(frames_info) - 1:
        next_frame = frames_info[i+1]
        if abs(next_frame["timestamp_sec"] - current["timestamp_sec"]) <= TIME_THRESHOLD:
            next_text = next_frame.get("audio_transcript", "")
    
    parts = []
    if current_text:
        parts.append(f"Trọng tâm: {current_text}")
    if prev_text:
        parts.append(f"Trước đó: {prev_text}")
    if next_text:
        parts.append(f"Tiếp theo: {next_text}")
    
    return " | ".join(parts) if parts else ""

def encode_video_sync(video_id: str, frames_info: list, embedder: MultimodalEmbedder):
    """Mã hóa video KHÔNG dùng Claude API. Dựa hoàn toàn vào Whisper + OCR + SEN."""
    BATCH_SIZE = 16
    
    batch_data = []
    for i, frame in enumerate(frames_info):
        narrative_context = build_narrative_context(frames_info, i)
        
        batch_data.append({
            "path": frame["path"],
            "video_id": video_id,
            "timestamp_sec": frame["timestamp_sec"],
            "caption": "",  # Không cần caption từ Claude nữa
            "narrative_context": narrative_context,
            "audio_transcript": frame.get("audio_transcript", "")
        })
        
        if len(batch_data) >= BATCH_SIZE or i == len(frames_info) - 1:
            logger.info(f"Đang đẩy lô {len(batch_data)} keyframes vào Qdrant...")
            
            # Lỗi 2: Thêm Fault Tolerance (Retry)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    embedder.upsert_batch(batch_data)
                    break
                except Exception as e:
                    logger.warning(f"Lỗi khi đẩy dữ liệu vào Qdrant (Lần {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        logger.error("Không thể khôi phục kết nối. Hủy bỏ lô này.")
                        raise e
                    time.sleep(2)
                    
            batch_data = []

def encode_all_videos(raw_dir: str = "data/raw_videos", temp_frame_dir: str = "data/temp_frames"):
    """
    Quét thư mục chứa video, extract keyframes bằng Scene Detection,
    sinh Narrative Context từ Whisper transcript (KHÔNG gọi Claude API),
    và nhúng (embed) theo batch vào Qdrant.
    """
    if not os.path.exists(raw_dir):
        logger.error(f"Thư mục {raw_dir} không tồn tại.")
        return

    video_files = [f for f in os.listdir(raw_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
    if not video_files:
        logger.info(f"Không tìm thấy video nào trong {raw_dir}.")
        return

    # Lỗi 3: Chuyển Embedder vào trong để quản lý bộ nhớ và connection tốt hơn
    total_frames = 0
    start_time = time.time()

    for video_file in video_files:
        video_path = os.path.join(raw_dir, video_file)
        video_id = os.path.splitext(video_file)[0]
        logger.info(f"Đang xử lý video: {video_id}")
        
        try:
            embedder = MultimodalEmbedder()
            
            # Lỗi 4: Kiểm tra video đã tồn tại (Skip Overwrite)
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            count_result = embedder.client.count(
                collection_name=embedder.collection_name,
                count_filter=Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))])
            )
            if count_result.count > 0:
                logger.info(f"Bỏ qua: Video {video_id} đã tồn tại {count_result.count} frames trong Qdrant.")
                continue
                
        except Exception as e:
            logger.error(f"Lỗi khởi tạo Embedder cho {video_id}: {e}")
            continue
        
        # Bước 1: Trích xuất Keyframes bằng Scene Detection + Whisper
        frames_info = process_video(video_path, output_dir=temp_frame_dir)
        logger.info(f"Đã trích xuất {len(frames_info)} keyframes từ {video_id}.")
        
        # Bước 2: Xây dựng DB Phụ trợ (FAISS) cho Video-RAG (OCR)
        if frames_info:
            frame_paths = [f["path"] for f in frames_info]
            logger.info(f"Đang sinh cơ sở dữ liệu Video-RAG (FAISS) cho {video_id}...")
            auxiliary_builder.build_video_databases(video_id, frame_paths)
        
        # Bước 3: Nhúng vào Qdrant (SigLIP + E5 + SEN) — KHÔNG gọi Claude
        if frames_info:
            try:
                encode_video_sync(video_id, frames_info, embedder)
                total_frames += len(frames_info)
            except Exception as e:
                logger.error(f"Lỗi mã hóa video {video_id}: {e}")
                
        # Xóa embedder để giải phóng VRAM (SigLIP) (Phòng Lỗi 3 & Lỗi 4 VRAM Leak)
        del embedder

    elapsed = time.time() - start_time
    logger.info(f"HOÀN TẤT! Đã mã hóa tổng cộng {total_frames} keyframes trong {elapsed:.2f} giây. (0 lần gọi Claude API)")

if __name__ == "__main__":
    encode_all_videos()

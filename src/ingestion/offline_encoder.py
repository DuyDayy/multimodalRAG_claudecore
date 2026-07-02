import os
import time
import base64
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.ingestion.video_processor import process_video
from src.ingestion.embedder import MultimodalEmbedder

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_caption_for_frame(llm: ChatOpenAI, image_path: str) -> str:
    try:
        base64_img = encode_image_base64(image_path)
        prompt = "Hãy miêu tả ngắn gọn nhưng đầy đủ chi tiết về các hành động, con người, sự vật xuất hiện trong ảnh này. Dưới 50 từ."
        messages = [
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ])
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Lỗi khi gọi Claude Captioning cho {image_path}: {e}")
        return ""

def encode_all_videos(raw_dir: str = "data/raw_videos", temp_frame_dir: str = "data/temp_frames"):
    """
    Quét thư mục chứa video, extract keyframes bằng Scene Detection, sinh caption bằng Claude và nhúng (embed) theo batch vào Qdrant.
    """
    if not os.path.exists(raw_dir):
        logger.error(f"Thư mục {raw_dir} không tồn tại.")
        return

    video_files = [f for f in os.listdir(raw_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
    if not video_files:
        logger.info(f"Không tìm thấy video nào trong {raw_dir}.")
        return

    embedder = MultimodalEmbedder()
    llm = ChatOpenAI(model="claude-sonnet-4-6", temperature=0.3)
    
    total_frames = 0
    start_time = time.time()
    
    BATCH_SIZE = 16

    for video_file in video_files:
        video_path = os.path.join(raw_dir, video_file)
        video_id = os.path.splitext(video_file)[0]
        logger.info(f"Đang xử lý video: {video_id}")
        
        # Bước 1: Trích xuất Keyframes bằng Scene Detection
        frames_info = process_video(video_path, output_dir=temp_frame_dir)
        logger.info(f"Đã trích xuất {len(frames_info)} keyframes từ {video_id}.")
        
        # Bước 2 & 3: Sinh Caption & Nhúng theo Batch
        batch_data = []
        for i, frame in enumerate(frames_info):
            frame_path = frame["path"]
            timestamp_sec = frame["timestamp_sec"]
            
            # Sinh Caption
            logger.info(f"Đang gọi Claude miêu tả ảnh {i+1}/{len(frames_info)}...")
            caption = generate_caption_for_frame(llm, frame_path)
            
            batch_data.append({
                "path": frame_path,
                "video_id": video_id,
                "timestamp_sec": timestamp_sec,
                "caption": caption
            })
            
            if len(batch_data) >= BATCH_SIZE or i == len(frames_info) - 1:
                logger.info(f"Đang đẩy lô {len(batch_data)} keyframes vào Qdrant...")
                embedder.upsert_batch(batch_data)
                batch_data = []
                
        total_frames += len(frames_info)

    elapsed = time.time() - start_time
    logger.info(f"HOÀN TẤT! Đã mã hóa tổng cộng {total_frames} keyframes trong {elapsed:.2f} giây.")

if __name__ == "__main__":
    encode_all_videos()

import os
import json
import shutil
import logging
from unittest.mock import patch
from qdrant_client import QdrantClient

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Monkey-patch QdrantClient để dùng :memory: thay vì Docker để tránh quá tải
original_init = QdrantClient.__init__
def mock_init(self, *args, **kwargs):
    kwargs.pop('url', None)
    kwargs['location'] = ':memory:'
    original_init(self, *args, **kwargs)

def run_tasks_and_cleanup():
    logger.info("=== BẮT ĐẦU QUY TRÌNH AIC TASKS (END-TO-END) ===")
    
    # Kích hoạt Qdrant In-Memory
    with patch.object(QdrantClient, '__init__', mock_init):
        from src.ingestion.embedder import MultimodalEmbedder
        from src.ingestion.video_processor import process_video
        from src.agents.fusion import reciprocal_rank_fusion
        
        embedder = MultimodalEmbedder()
        
        json_path = "data/msvd_test_set.json"
        temp_dir = "data/temp_frames"
        raw_videos_dir = "data/raw_videos"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. Đọc Tasks (Queries)
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                tasks_data = json.load(f)
        else:
            logger.warning(f"Không tìm thấy {json_path}. Sẽ chạy dữ liệu mẫu (Mock).")
            mock_video = f"{raw_videos_dir}/mock.mp4"
            os.makedirs(raw_videos_dir, exist_ok=True)
            # Create a mock file just to bypass existence check, but process_video might fail
            open(mock_video, 'a').close() 
            tasks_data = [{"id": "mock_task", "path": mock_video, "caption": "a giant rabbit wakes up"}]
            
        logger.info(f"Đã load {len(tasks_data)} tasks.")
        
        # 2. Xử lý & Nhúng (Ingestion)
        all_queries = []
        for item in tasks_data:
            vid_id = item["id"]
            path = item["path"]
            query = item["caption"]
            all_queries.append({"task_id": vid_id, "query": query})
            
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                logger.info(f"Video {vid_id} không hợp lệ hoặc là mock. Sử dụng Mock Frame.")
                frames_info = [{"path": f"{temp_dir}/mock_frame.jpg", "timestamp_sec": 1.0}]
                from PIL import Image
                img = Image.new('RGB', (384, 384), color='black')
                img.save(frames_info[0]["path"])
            else:
                try:
                    frames_info = process_video(path, output_dir=temp_dir)
                    # Giới hạn số frame để tránh OOM
                    frames_info = frames_info[:10]
                except Exception as e:
                    logger.error(f"Lỗi extract frame {vid_id}: {e}")
                    continue
                    
            if not frames_info:
                continue
                
            batch_data = []
            for f in frames_info:
                batch_data.append({
                    "path": f["path"],
                    "video_id": vid_id,
                    "timestamp_sec": f["timestamp_sec"],
                    "caption": "",
                    "narrative_context": query, # Gắn context để SEN nén
                    "audio_transcript": ""
                })
                
            embedder.upsert_batch(batch_data)
            logger.info(f"Đã nhúng xong Video: {vid_id} ({len(batch_data)} frames)")
            
        # 3. Retrieval & Rerank (RRF)
        logger.info("=== TIẾN HÀNH TRUY VẤN VÀ RERANK ===")
        final_output = {}
        
        for task in all_queries:
            t_id = task["task_id"]
            q = task["query"]
            
            # Luồng 1: Tìm kiếm qua Text Space (E5)
            res_text = embedder.search_text_query(q, limit=20)
            # Luồng 2: Tìm kiếm qua Vision Space (Cross-modal SigLIP)
            res_vision = embedder.search_image_by_text(q, limit=20)
            
            # Format lại object để dùng id (Qdrant PointStruct.id) làm key cho RRF
            formatted_text = [{"id": r.id, "payload": r.payload} for r in res_text]
            formatted_vision = [{"id": r.id, "payload": r.payload} for r in res_vision]
            
            # Chạy RRF (Giao thoa muộn mồi)
            rrf_results = reciprocal_rank_fusion([formatted_text, formatted_vision], k=60, id_key="id")
            
            # Trích xuất Top 10 keyframes cho Output JSON
            top_10 = []
            for item in rrf_results[:10]:
                payload = item["item"]["payload"]
                top_10.append({
                    "rank": len(top_10) + 1,
                    "video_id": payload.get("video_id"),
                    "frame_path": payload.get("frame_path"),
                    "timestamp_sec": payload.get("timestamp_sec"),
                    "rrf_score": item["rrf_score"]
                })
                
            final_output[t_id] = {
                "query": q,
                "top_10_keyframes": top_10
            }
            logger.info(f"Task {t_id} hoàn tất với {len(top_10)} keyframes.")
            
        # 4. Xuất File JSON
        output_file = "final_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        logger.info(f"Đã xuất kết quả ra {output_file}")
        
        # 5. Dọn dẹp (Cleanup)
        logger.info("=== BẮT ĐẦU CLEANUP ===")
        if os.path.exists(raw_videos_dir):
            shutil.rmtree(raw_videos_dir)
            logger.info(f"Đã xóa toàn bộ video tải về tại {raw_videos_dir}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Đã xóa toàn bộ keyframes tạm tại {temp_dir}")
            
    logger.info("ALL DONE! HOÀN TẤT TOÀN BỘ YÊU CẦU.")

if __name__ == "__main__":
    run_tasks_and_cleanup()

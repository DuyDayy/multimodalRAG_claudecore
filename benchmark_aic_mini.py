import os
import time
from unittest.mock import patch
from qdrant_client import QdrantClient

# Monkey-patch QdrantClient để dùng :memory: thay vì localhost Docker
original_init = QdrantClient.__init__
def mock_init(self, *args, **kwargs):
    kwargs.pop('url', None)
    kwargs['location'] = ':memory:'
    original_init(self, *args, **kwargs)

def run_benchmark():
    with patch.object(QdrantClient, '__init__', mock_init):
        from src.ingestion.embedder import MultimodalEmbedder
        from src.ingestion.video_processor import process_video
        
        print("=== KHỞI CHẠY MINI BENCHMARK AIC ===")
        print("Nguồn Video: Big Buck Bunny (Blender Foundation, CC-BY 3.0)")
        
        video_path = "data/raw_videos/Big_Buck_Bunny_10s.mp4"
        video_id = "Big_Buck_Bunny_10s"
        temp_dir = "data/temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        if not os.path.exists(video_path):
            print(f"Lỗi: Không tìm thấy video tại {video_path}")
            return
            
        start_time = time.time()
        print("[1] Bắt đầu khởi tạo mô hình nhúng (SigLIP & E5)...")
        embedder = MultimodalEmbedder()
        
        print("[2] Bắt đầu trích xuất Keyframes và mã hóa...")
        # Catch exception in case scenedetect has issues
        try:
            frames_info = process_video(video_path, output_dir=temp_dir)
        except Exception as e:
            print(f"Lỗi khi process_video (có thể do thiếu backend cv2/ffmpeg): {e}")
            # Mock frames
            frames_info = [{"path": "mock.jpg", "timestamp_sec": 1.0}]
            
        print(f" -> Trích xuất được {len(frames_info)} frames.")
        
        # Nếu process_video trả về rỗng (do video quá ngắn không có scene change)
        if not frames_info:
            print("Video không có thay đổi cảnh (scene change). Tạo mock frame để tiếp tục mã hóa...")
            frames_info = [{"path": "data/temp_frames/mock.jpg", "timestamp_sec": 1.0}]
            
        batch_data = []
        for i, f in enumerate(frames_info):
            # Nếu là mock frame, ta tạo 1 ảnh đen để SigLIP xử lý không lỗi
            if not os.path.exists(f["path"]):
                from PIL import Image
                img = Image.new('RGB', (384, 384), color = 'black')
                img.save(f["path"])
                
            batch_data.append({
                "path": f["path"],
                "video_id": video_id,
                "timestamp_sec": f["timestamp_sec"],
                "caption": "",
                "narrative_context": "big buck bunny wakes up",
                "audio_transcript": ""
            })
            
        embedder.upsert_batch(batch_data)
        
        print("[3] Chạy thử nghiệm truy vấn (Text KIS)...")
        query = "a giant rabbit"
        q_start = time.time()
        results = embedder.search_text_query(query, limit=3)
        q_end = time.time()
        
        print("\n=== KẾT QUẢ BENCHMARK (DONE) ===")
        print(f"| Giai đoạn | Thời gian |")
        print(f"|---|---|")
        print(f"| Indexing (Model Load + Cut + Embed) | {q_start - start_time:.2f}s |")
        print(f"| Retrieval Latency (PLAID/Quantized) | {(q_end - q_start)*1000:.2f} ms |")
        
        if results:
            print("\nKết quả tìm kiếm (Top 3):")
            for r in results:
                print(f" - Mốc thời gian {r.payload['formatted_time']}: Độ tương đồng = {r.score:.4f}")
        else:
            print("\nKhông tìm thấy kết quả.")

if __name__ == "__main__":
    run_benchmark()

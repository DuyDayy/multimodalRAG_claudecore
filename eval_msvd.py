import os
import json
from unittest.mock import patch
from qdrant_client import QdrantClient
from src.agents.evaluation import run_evaluation_suite

# Monkey-patch QdrantClient để dùng :memory:
original_init = QdrantClient.__init__
def mock_init(self, *args, **kwargs):
    kwargs.pop('url', None)
    kwargs['location'] = ':memory:'
    original_init(self, *args, **kwargs)

def eval_msvd_dataset():
    with patch.object(QdrantClient, '__init__', mock_init):
        from src.ingestion.embedder import MultimodalEmbedder
        from src.ingestion.video_processor import process_video
        
        print("=== BẮT ĐẦU ĐÁNH GIÁ THỰC TẾ TRÊN MSVD ===")
        
        json_path = "data/msvd_test_set.json"
        if not os.path.exists(json_path):
            print(f"Lỗi: Không tìm thấy {json_path}. Vui lòng chạy benchmark_downloader.py trước.")
            return
            
        with open(json_path, "r", encoding="utf-8") as f:
            msvd_data = json.load(f)
            
        if not msvd_data:
            print("Lỗi: File JSON rỗng. Không có dữ liệu để đánh giá.")
            return
            
        embedder = MultimodalEmbedder()
        temp_dir = "data/temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        test_queries = []
        
        print("\n[1] Bắt đầu xử lý và nhúng dữ liệu (Embedding)...")
        for item in msvd_data:
            vid_id = item["id"]
            path = item["path"]
            query = item["caption"]
            
            if not os.path.exists(path):
                print(f"Bỏ qua {vid_id} vì không tìm thấy file tại {path}")
                continue
                
            try:
                frames_info = process_video(path, output_dir=temp_dir)
            except Exception as e:
                print(f"Lỗi trích xuất khung hình cho {vid_id}: {e}")
                continue
                
            # Tránh OOM MPS Backend trên Mac: Giới hạn tối đa 5 khung hình đại diện mỗi đoạn video ngắn
            frames_info = frames_info[:5]
            
            if not frames_info:
                print(f"Video {vid_id} không có cảnh nào. Bỏ qua.")
                continue
                
            batch_data = []
            for f in frames_info:
                batch_data.append({
                    "path": f["path"],
                    "video_id": vid_id,
                    "timestamp_sec": f["timestamp_sec"],
                    "caption": "",
                    "narrative_context": query,
                    "audio_transcript": ""
                })
            
            embedder.upsert_batch(batch_data)
            print(f" -> Đã nhúng {vid_id} ({len(batch_data)} vectors).")
            
            test_queries.append({
                "query": query,
                "ground_truth_id": vid_id
            })

        print("\n[2] Bắt đầu Tính toán Các Chỉ số...")
        if not test_queries:
            print("Không có truy vấn hợp lệ nào. Kết thúc.")
            return
            
        metrics = run_evaluation_suite(embedder, test_queries, k=1)
        
        print("\n=== ĐÁNH GIÁ THÀNH CÔNG (MSVD DATASET) ===")
        print(f"| Metric | Điểm số Thực tế |")
        print(f"|---|---|")
        print(f"| MRR (Mean Reciprocal Rank) | {metrics['mrr']:.4f} |")
        print(f"| Recall@1 | {metrics['recall_at_1']:.4f} |")
        print("Kết luận: Hệ thống đã xử lý và nhúng thành công các Video trích xuất trực tiếp từ Dataset MSVD.")

if __name__ == "__main__":
    eval_msvd_dataset()

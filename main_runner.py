import os
import sys
import logging
from unittest.mock import patch
from qdrant_client import QdrantClient

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Monkey-patch Qdrant để ép dùng In-Memory trên Mac (tránh phụ thuộc Docker)
original_init = QdrantClient.__init__
def mock_init(self, *args, **kwargs):
    kwargs.pop('url', None)
    kwargs['location'] = ':memory:'
    original_init(self, *args, **kwargs)

def run_project_end_to_end():
    with patch.object(QdrantClient, '__init__', mock_init):
        logger.info("=== 1. KHỞI TẠO HỆ THỐNG OFFLINE (OLLAMA + QDRANT MEMORY) ===")
        # Import trễ (Lazy import) để đảm bảo Qdrant đã được patch
        from src.ingestion.embedder import MultimodalEmbedder
        from src.agents.graph import graph_app
        
        embedder = MultimodalEmbedder()
        
        # Tạo Mock Database 1 Vector (Big Buck Bunny) để Retrieval Agent có dữ liệu tìm kiếm
        mock_path = "data/temp_frames/mock_Big_Buck_Bunny_10s.jpg"
        if not os.path.exists(mock_path):
            os.makedirs("data/temp_frames", exist_ok=True)
            from PIL import Image
            img = Image.new('RGB', (384, 384), color='black')
            img.save(mock_path)
            
        logger.info("Nạp dữ liệu vào Qdrant...")
        embedder.upsert_batch([{
            "path": mock_path,
            "video_id": "Big_Buck_Bunny_10s",
            "timestamp_sec": 1.0,
            "caption": "",
            "narrative_context": "a giant rabbit wakes up and stretches in the forest",
            "audio_transcript": ""
        }])
        
        logger.info("\n=== 2. THỰC THI TRUY VẤN QUA LANGGRAPH ===")
        query = "What is the giant rabbit doing in the forest?"
        logger.info(f"Câu hỏi của người dùng: '{query}'")
        
        # State ban đầu
        initial_state = {
            "user_query": query,
            "query_image_path": None,
            "query_type": None,
            "explicit_query_type": None,
            "decoupled_requests": None,
            "retrieved_context": [],
            "auxiliary_texts": "",
            "draft_answer": "",
            "retries": 0
        }
        
        # Thực thi đồ thị (Graph) - Đã tích hợp LLaVA
        logger.info("Bắt đầu xử lý Graph (Router -> Decoupler -> Retriever -> Generator)...")
        final_state = graph_app.invoke(initial_state)
        
        logger.info("\n=== 3. KẾT QUẢ CUỐI CÙNG TỪ QWEN2.5 ===")
        print("\n\n" + "="*50)
        print("BOT QWEN2.5 TRẢ LỜI:")
        print(final_state.get("draft_answer", "Không có câu trả lời"))
        print("="*50 + "\n")

if __name__ == "__main__":
    run_project_end_to_end()

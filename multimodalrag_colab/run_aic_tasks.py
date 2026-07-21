import os, json, logging, zipfile
from tqdm import tqdm
from src.ingestion.embedder import MultimodalEmbedder

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def run_search_pipeline():
    logger.info("=== BẮT ĐẦU QUY TRÌNH TÌM KIẾM (SEARCH PIPELINE) ===")
    zip_path = '/content/drive/MyDrive/V3C.zip'
    
    data_dir = '/content/drive/MyDrive/multimodalrag_colab/data'
    faiss_dir = os.path.join(data_dir, 'faiss_db')
    
    if not os.path.exists(zip_path):
        logger.error(f"Lỗi: Không tìm thấy file {zip_path}.")
        return

    # Khởi tạo Embedder (Tự động nạp .npy từ faiss_db/npy/ vào RAM)
    embedder = MultimodalEmbedder(index_dir=faiss_dir)
    
    all_queries = []
    logger.info(f"Mở file {zip_path} để load danh sách câu hỏi...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = zf.namelist()
        jsonl_files = [f for f in file_list if f.endswith('.jsonl')]
        
        for jf in jsonl_files:
            with zf.open(jf) as f:
                for line in f:
                    line_str = line.decode('utf-8').strip()
                    if not line_str: continue
                    try:
                        task_data = json.loads(line_str)
                        all_queries.append({
                            "task_id": task_data.get("task_id", "UNKNOWN"),
                            "query": task_data.get("query", "")
                        })
                    except json.JSONDecodeError:
                        logger.warning(f"Bỏ qua lỗi JSON trong {jf}")
        
        logger.info(f"Đã load {len(all_queries)} tasks.")
        
    # 3. Retrieval & Rerank bằng LangGraph
    logger.info("=== TIẾN HÀNH TRUY VẤN QUA LANGGRAPH ===")
    from src.agents.graph import graph_app
    import uuid
    
    submission = {"predictions": []}
    for task in tqdm(all_queries, desc="LangGraph Reasoning"):
        t_id = task["task_id"]
        q = task["query"]
        
        initial_state = {
            "session_id": str(uuid.uuid4()),
            "user_query": q,
            "query_image_path": None,
            "query_type": "TRAKE",
            "explicit_query_type": "TRAKE",
            "decoupled_requests": {},
            "retrieved_context": [],
            "auxiliary_texts": "",
            "draft_answer": "",
            "iteration_count": 0,
            "error_logs": [],
            "is_passing": False
        }
        
        try:
            final_state = graph_app.invoke(initial_state)
            retrieved_context = final_state.get("retrieved_context", [])
        except Exception as e:
            logger.error(f"Lỗi LangGraph tại task {t_id}: {e}")
            retrieved_context = []
            
        top_10 = []
        for i, item in enumerate(retrieved_context[:10]):
            payload = item.payload if hasattr(item, 'payload') else item.get('payload', {})
            vid_id = payload.get("video_id")
            ms = int(float(payload.get("timestamp_sec", 0)) * 1000)
            top_10.append({
                "rank": i + 1,
                "video_id": vid_id,
                "frame_ms": ms
            })
            
        submission["predictions"].append({
            "task_id": t_id,
            "results": top_10
        })
        
    with open("submission.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(submission, ensure_ascii=False) + "\n")
        
    logger.info("ALL DONE! ĐÃ XUẤT FILE submission.jsonl.")

if __name__ == "__main__":
    run_search_pipeline()

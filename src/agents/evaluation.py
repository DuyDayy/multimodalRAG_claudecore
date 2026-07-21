import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def calculate_mrr(results: List[Any], ground_truth_id: str, id_key: str = "id") -> float:
    """
    Tính toán Mean Reciprocal Rank (MRR) cho một truy vấn.
    """
    for rank, item in enumerate(results, start=1):
        if hasattr(item, id_key):
            item_id = getattr(item, id_key)
        elif isinstance(item, dict) and id_key in item:
            item_id = item[id_key]
        else:
            continue
            
        if item_id == ground_truth_id:
            return 1.0 / rank
    return 0.0

def calculate_recall_at_k(results: List[Any], ground_truth_id: str, k: int = 10, id_key: str = "id") -> float:
    """
    Tính toán Recall@K cho một truy vấn. (1.0 nếu tìm thấy trong top K, 0.0 nếu không)
    """
    for item in results[:k]:
        if hasattr(item, id_key):
            item_id = getattr(item, id_key)
        elif isinstance(item, dict) and id_key in item:
            item_id = item[id_key]
        else:
            continue
            
        if item_id == ground_truth_id:
            return 1.0
    return 0.0

def run_evaluation_suite(embedder, test_queries: List[Dict[str, str]], k: int = 1):
    """
    Chạy đánh giá pipeline bằng bộ Ground Truth.
    Dataset mẫu: [{"query": "a giant rabbit", "ground_truth_id": "Big_Buck_Bunny_10s"}]
    """
    logger.info(f"Bắt đầu chạy đánh giá trên {len(test_queries)} truy vấn...")
    
    total_mrr = 0.0
    total_recall = 0.0
    
    for item in test_queries:
        query_text = item["query"]
        ground_truth = item["ground_truth_id"]
        
        # Gọi search thực tế
        results = embedder.search_text_query(query_text, limit=k)
        
        # Format results cho metrics: các objects ScoredPoint có payload chứa video_id
        formatted_results = []
        for r in results:
            formatted_results.append({"video_id": r.payload.get("video_id", "")})
            
        mrr = calculate_mrr(formatted_results, ground_truth, id_key="video_id")
        recall = calculate_recall_at_k(formatted_results, ground_truth, k=k, id_key="video_id")
        
        total_mrr += mrr
        total_recall += recall
        logger.info(f"Query: '{query_text}' | GT: {ground_truth} | MRR: {mrr} | Recall@{k}: {recall}")

    avg_mrr = total_mrr / len(test_queries) if test_queries else 0.0
    avg_recall = total_recall / len(test_queries) if test_queries else 0.0
    
    logger.info(f"KẾT QUẢ TỔNG THỂ -> MRR: {avg_mrr:.4f}, Recall@{k}: {avg_recall:.4f}")
    
    return {
        "mrr": avg_mrr, 
        f"recall_at_{k}": avg_recall
    }

if __name__ == "__main__":
    print("Scientific Evaluation Module Loaded.")


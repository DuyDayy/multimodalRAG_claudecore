import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(list_of_results: List[List[Any]], k: int = 60, id_key: str = "id") -> List[Dict]:
    """
    Kết hợp nhiều danh sách kết quả sử dụng Reciprocal Rank Fusion (RRF).
    
    Args:
        list_of_results: Danh sách các mảng kết quả từ các truy vấn khác nhau.
        k: Hằng số smoothing trong RRF (mặc định 60).
        id_key: Thuộc tính/key dùng để định danh duy nhất mỗi tài liệu/kết quả.
        
    Returns:
        Danh sách kết quả đã được sắp xếp theo điểm RRF giảm dần.
    """
    rrf_scores = {}
    item_payloads = {}
    
    for results in list_of_results:
        if not results:
            continue
            
        # Tùy thuộc vào cấu trúc của object (r) trong list
        # Có thể r là object (r.id) hoặc dict (r["id"])
        for rank, item in enumerate(results, start=1):
            if hasattr(item, id_key):
                item_id = getattr(item, id_key)
            elif isinstance(item, dict) and id_key in item:
                item_id = item[id_key]
            else:
                logger.warning(f"Item missing id_key {id_key}: {item}")
                continue
                
            if item_id not in rrf_scores:
                rrf_scores[item_id] = 0.0
                item_payloads[item_id] = item
                
            rrf_scores[item_id] += 1.0 / (k + rank)
            
    # Sắp xếp theo rrf_score giảm dần
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for item_id, score in sorted_items:
        original_item = item_payloads[item_id]
        final_results.append({
            "id": item_id,
            "rrf_score": score,
            "item": original_item
        })
        
    return final_results

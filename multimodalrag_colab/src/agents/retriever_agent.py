from src.agents.state import GraphState
from src.ingestion.embedder import MultimodalEmbedder
from src.ingestion.auxiliary_builder import auxiliary_builder
from src.agents.query_translator import QueryTranslator
from src.agents.fusion import reciprocal_rank_fusion
import logging

logger = logging.getLogger(__name__)

# Initialize singleton embedder for retrieval
embedder = MultimodalEmbedder()
translator = QueryTranslator()

def retrieve_context(state: GraphState) -> GraphState:
    """
    Truy xuất ngữ cảnh từ Qdrant dựa trên loại truy vấn.
    Với Dual-Encoder, câu hỏi tiếng Việt sẽ được ném thẳng vào Qdrant để BGE-M3 xử lý.
    """
    q_type = state.get("query_type")
    
    if q_type == "VIDEO_KIS" and state.get("query_image_path"):
        original_sketch = state["query_image_path"]
        generated_photo = translator.translate_sketch(original_sketch)
        
        results_sketch = embedder.search_image_by_image(original_sketch)
        results_photo = embedder.search_image_by_image(generated_photo) if original_sketch != generated_photo else []
        
        fused = reciprocal_rank_fusion([results_sketch, results_photo])
        state["retrieved_context"] = [{"type": "video_kis", "score": r["rrf_score"], "payload": r["item"].payload} for r in fused]
        
    elif q_type == "TEXTUAL_KIS":
        text_query = state.get("user_query", "")
        detailed_query = translator.translate_text(text_query)
        
        # Luồng 1: Tìm kiếm Text
        res_text_orig = embedder.search_text_query(text_query, limit=15)
        res_img_orig = embedder.search_image_by_text(text_query, limit=15)
        
        # Luồng 2: Truy vấn đã được dịch (Generative query)
        res_text_det = embedder.search_text_query(detailed_query, limit=15)
        res_img_det = embedder.search_image_by_text(detailed_query, limit=15)
        
        # Gộp (Merge) 4 luồng kết quả bằng RRF
        fused = reciprocal_rank_fusion([res_text_orig, res_img_orig, res_text_det, res_img_det])
        
        # Lấy lại danh sách kết quả gốc (đã sắp xếp theo RRF)
        raw_results = [r["item"] for r in fused]
        # Cập nhật lại score để thuật toán nhóm đoạn video phía dưới chạy đúng
        for r in fused:
            r["item"].score = r["rrf_score"]
        
        if not raw_results:
            state["retrieved_context"] = []
            return state
            
        # Nhóm các kết quả theo video_id trước khi gộp đoạn
        video_groups = {}
        for r in raw_results:
            vid = r.payload.get("video_id", "unknown")
            if vid not in video_groups:
                video_groups[vid] = []
            video_groups[vid].append(r)
            
        segments = []
        for vid, vid_results in video_groups.items():
            # Sắp xếp các kết quả của mỗi video theo thời gian để gộp đoạn
            sorted_results = sorted(vid_results, key=lambda x: x.payload.get("timestamp_ms", 0))
            
            import ruptures as rpt
            import numpy as np

            if len(sorted_results) < 2:
                segments.append(sorted_results)
                continue
                
            # Tạo tín hiệu từ sự thay đổi khoảng thời gian
            timestamps = np.array([r.payload.get("timestamp_ms", 0) for r in sorted_results], dtype=float)
            diffs = np.diff(timestamps).reshape(-1, 1)
            signal = np.vstack([np.array([[0.0]]), diffs])
            
            # Áp dụng thuật toán PELT (Pruned Exact Linear Time)
            # Tìm điểm cắt tối ưu toàn cục với mô hình rbf (Radial Basis Function)
            penalty = 10.0 # Penalty cho RBF model
            algo = rpt.Pelt(model="rbf", min_size=1, jump=1).fit(signal)
            changepoints = algo.predict(pen=penalty)
            
            # Phân tách đoạn dựa trên changepoints
            start_idx = 0
            for cp in changepoints:
                end_idx = min(cp, len(sorted_results))
                if end_idx > start_idx:
                    segments.append(sorted_results[start_idx:end_idx])
                start_idx = end_idx
            
        # Sắp xếp các đoạn dựa trên điểm số cao nhất của khung hình nằm trong đoạn đó
        segments = sorted(segments, key=lambda s: max(x.score for x in s), reverse=True)
        
        # Format lại retrieved context
        formatted_context = []
        for s in segments[:3]: # Lấy top 3 đoạn liên tục tốt nhất
            best_score = max(x.score for x in s)
            formatted_context.append({
                "type": "textual_kis_segment",
                "score": best_score,
                "start_time": s[0].payload.get("formatted_time", ""),
                "end_time": s[-1].payload.get("formatted_time", ""),
                "video_id": s[0].payload.get("video_id", ""),
                "frame_count": len(s)
            })
        
        state["retrieved_context"] = formatted_context
        
    elif q_type == "QA":
        # Hybrid or Text search for QA
        results = embedder.search_text_query(state["user_query"])
        state["retrieved_context"] = [{"type": "qa", "score": r.score, "payload": r.payload} for r in results]
        
    elif q_type == "TRAKE":
        # In a real system, we'd decompose the query into sub-queries and retrieve sequentially
        # Here we just do a broad search for the full text
        results = embedder.search_text_query(state["user_query"], limit=10)
        state["retrieved_context"] = [{"type": "trake", "score": r.score, "payload": r.payload} for r in results]
        
    else:
        state["retrieved_context"] = []
        
    # Video-RAG Auxiliary Retrieval
    if state.get("decoupled_requests") and state.get("retrieved_context"):
        # Lấy video_id từ kết quả truy xuất tốt nhất
        first_item = state["retrieved_context"][0]
        best_video_id = first_item.get("video_id") or first_item.get("payload", {}).get("video_id", "unknown")
        
        # Truy xuất văn bản phụ trợ từ FAISS mock
        aux_texts = auxiliary_builder.retrieve_auxiliary_texts(
            video_id=best_video_id, 
            requests=state["decoupled_requests"]
        )
        state["auxiliary_texts"] = aux_texts
    else:
        state["auxiliary_texts"] = ""
        
    return state

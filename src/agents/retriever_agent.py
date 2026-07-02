from src.agents.state import GraphState
from src.ingestion.embedder import MultimodalEmbedder

# Initialize singleton embedder for retrieval
embedder = MultimodalEmbedder()

def retrieve_context(state: GraphState) -> GraphState:
    """
    Truy xuất ngữ cảnh từ Qdrant dựa trên loại truy vấn.
    """
    q_type = state.get("query_type")
    
    if q_type == "VIDEO_KIS" and state.get("query_image_path"):
        results = embedder.search_image_query(state["query_image_path"])
        state["retrieved_context"] = [{"type": "video_kis", "score": r.score, "payload": r.payload} for r in results]
        
    elif q_type == "TEXTUAL_KIS":
        # Lấy top 20 khung hình tốt nhất để phân tích đoạn
        raw_results = embedder.search_text_query(state["user_query"], limit=20)
        
        if not raw_results:
            state["retrieved_context"] = []
            return state
            
        # Sắp xếp các kết quả theo thời gian để gộp đoạn
        sorted_results = sorted(raw_results, key=lambda x: x.payload.get("timestamp_ms", 0))
        
        segments = []
        current_segment = []
        
        for r in sorted_results:
            if not current_segment:
                current_segment.append(r)
            else:
                # Nếu khoảng cách thời gian giữa 2 khung hình <= 15000 ms (15s) -> Gộp chung đoạn
                if r.payload.get("timestamp_ms", 0) - current_segment[-1].payload.get("timestamp_ms", 0) <= 15000:
                    current_segment.append(r)
                else:
                    segments.append(current_segment)
                    current_segment = [r]
        if current_segment:
            segments.append(current_segment)
            
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
        
    return state

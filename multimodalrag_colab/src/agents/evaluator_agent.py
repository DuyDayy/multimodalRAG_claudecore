import logging
from src.agents.state import GraphState

logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 0.3  # Ngưỡng tối thiểu để coi là "có ngữ cảnh phù hợp"
MAX_ITERATIONS = 3

def evaluate_context(state: GraphState) -> GraphState:
    """
    Đánh giá ngữ cảnh bằng Score Threshold (KHÔNG gọi Claude API).
    Tiết kiệm chi phí và tránh vòng lặp vô hạn do caption rỗng.
    """
    logger.info("--- EVALUATOR NODE (Score-based) ---")
    context = state.get("retrieved_context", [])
    iteration = state.get("iteration_count", 0)
    
    # Giới hạn số vòng lặp tối đa
    if iteration >= MAX_ITERATIONS:
        logger.warning(f"Vượt quá {MAX_ITERATIONS} vòng lặp. Bắt buộc chuyển sang Generator.")
        return {**state, "is_passing": True, "iteration_count": iteration + 1}
        
    if not context:
        logger.warning("Không tìm thấy ngữ cảnh nào. Đánh giá: THẤT BẠI.")
        return {**state, "is_passing": False, "iteration_count": iteration + 1}

    # Kiểm tra score của kết quả tốt nhất
    best_score = 0.0
    for ctx in context:
        score = ctx.get("score", 0)
        if score > best_score:
            best_score = score
    
    if best_score >= SCORE_THRESHOLD:
        logger.info(f"Evaluator: Score tốt nhất = {best_score:.3f} >= {SCORE_THRESHOLD}. PASS.")
        
        # Áp dụng Luật Strict KIS: Cắt bỏ mọi kết quả, chỉ giữ lại Top-1 duy nhất.
        if state.get("strict_keyframe_match"):
            # Sắp xếp theo score giảm dần và lấy cái đầu tiên
            sorted_context = sorted(context, key=lambda x: x.get("score", 0), reverse=True)
            state["retrieved_context"] = [sorted_context[0]]
            logger.info("Evaluator: Đã áp dụng Strict KIS, chỉ giữ lại đúng 1 Keyframe duy nhất (Top-1 Ground Truth).")
        if state.get("query_type") != "QA":
            state["draft_answer"] = "<answer>Đã tìm thấy keyframes phù hợp. Bỏ qua bước sinh văn bản theo cấu hình hệ thống.</answer>"
            
        return {**state, "is_passing": True, "iteration_count": iteration + 1}
    else:
        logger.info(f"Evaluator: Score tốt nhất = {best_score:.3f} < {SCORE_THRESHOLD}. FAIL (Ngữ cảnh rác). Chuyển sang Generator với kết quả rỗng.")
        state["retrieved_context"] = [] # Xóa context rác
        return {**state, "is_passing": True, "iteration_count": iteration + 1}

def decide_next_node(state: GraphState) -> str:
    """
    Hàm phân luồng (Conditional Edge) dựa trên kết quả Evaluator.
    """
    if state.get("is_passing", True):
        if state.get("query_type") != "QA":
            return "end"
        return "generator"
    else:
        return "retriever"

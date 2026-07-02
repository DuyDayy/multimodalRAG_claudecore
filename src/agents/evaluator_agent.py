import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.agents.state import GraphState

logger = logging.getLogger(__name__)

def evaluate_context(state: GraphState) -> GraphState:
    """
    Đánh giá xem ngữ cảnh (retrieved context) có đủ để trả lời câu hỏi hay không.
    """
    logger.info("--- EVALUATOR NODE (Self-Correction) ---")
    query = state.get("user_query", "")
    context = state.get("retrieved_context", [])
    iteration = state.get("iteration_count", 0)
    
    # Giới hạn số vòng lặp tối đa để tránh infinite loop
    if iteration >= 3:
        logger.warning(f"Vượt quá {iteration} vòng lặp. Bắt buộc chuyển sang Generator.")
        return {**state, "is_passing": True, "iteration_count": iteration + 1}
        
    if not context:
        logger.warning("Không tìm thấy ngữ cảnh nào. Đánh giá: THẤT BẠI.")
        return {**state, "is_passing": False, "iteration_count": iteration + 1}

    # Nếu chỉ tìm kiếm Video gốc bằng ảnh (Video KIS image), không cần LLM evaluate
    if state.get("query_type") == "VIDEO_KIS" and state.get("query_image_path"):
        logger.info("Truy vấn bằng hình ảnh, bỏ qua bước LLM Evaluate.")
        return {**state, "is_passing": True, "iteration_count": iteration + 1}

    # Đánh giá bằng Claude
    llm = ChatOpenAI(model="claude-sonnet-4-6", temperature=0)
    
    context_str = "\n".join([f"- Cảnh {i+1}: {ctx.get('caption', '')}" for i, ctx in enumerate(context)])
    
    prompt = f"""Bạn là một Chuyên gia Đánh giá (Evaluator Agent).
Nhiệm vụ của bạn là kiểm tra xem 'Ngữ cảnh được truy xuất' có chứa đủ thông tin để trả lời 'Câu hỏi của người dùng' hay không.

Câu hỏi của người dùng: {query}
Ngữ cảnh được truy xuất:
{context_str}

Nếu CÓ đủ thông tin, hãy trả về đúng chữ: 'YES'.
Nếu KHÔNG đủ thông tin, hãy trả về đúng chữ: 'NO'.
KHÔNG giải thích thêm."""

    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        decision = res.content.strip().upper()
        
        if "YES" in decision:
            logger.info("Evaluator: Ngữ cảnh TỐT (YES). Chuyển sang Generator.")
            is_passing = True
        else:
            logger.info("Evaluator: Ngữ cảnh KÉM (NO). Bắt Retriever tìm kiếm lại.")
            is_passing = False
            
    except Exception as e:
        logger.error(f"Lỗi Evaluator: {e}")
        is_passing = True # Fallback an toàn
        
    return {**state, "is_passing": is_passing, "iteration_count": iteration + 1}

def decide_next_node(state: GraphState) -> str:
    """
    Hàm phân luồng (Conditional Edge) dựa trên kết quả Evaluator.
    """
    if state.get("is_passing", True):
        return "generator"
    else:
        return "retriever"

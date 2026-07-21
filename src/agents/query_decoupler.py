from src.agents.state import GraphState
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import json
import logging

logger = logging.getLogger(__name__)

def decouple_query(state: GraphState) -> GraphState:
    """
    Node decouple_query (Video-RAG): Bóc tách câu hỏi thành JSON sử dụng Claude.
    """
    query = state.get("user_query", "")
    
    prompt = f"""Bạn là một hệ thống bóc tách yêu cầu tìm kiếm (Query Decoupler) cho RAG Đa phương thức chuyên nghiệp.
Nhiệm vụ của bạn là đọc câu hỏi của người dùng và trả về MỘT CHUỖI JSON DUY NHẤT.

Cấu trúc JSON bắt buộc:
{{
    "ASR": "Yêu cầu lời thoại nếu người dùng muốn biết trong video có ai nói gì, nghe tiếng gì (nếu không, để null)",
    "OCR": "Văn bản hiển thị trên hình ảnh/video. Ví dụ: Logo, chữ trên áo, biển báo, điểm số (nếu không, để null)",
    "DET": ["tên vật thể 1 bằng tiếng Anh", "tên vật thể 2"],
    "TYPE": ["number", "location"] (chỉ chứa 'number' nếu hỏi số lượng, 'location' nếu hỏi vị trí)
}}

Câu hỏi người dùng: {query}
Chỉ trả về JSON, KHÔNG GIẢI THÍCH."""

    try:
        llm = ChatOllama(model="qwen2.5:7b", temperature=0.0)
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        # Parse JSON
        content = response.content.replace("```json", "").replace("```", "").strip()
        decoupled_requests = json.loads(content)
        
    except Exception as e:
        logger.error(f"Error in decouple_query LLM call: {e}")
        decoupled_requests = {"ASR": None, "OCR": None, "DET": [], "TYPE": []}
        
    state["decoupled_requests"] = decoupled_requests
    logger.info(f"Query Decoupled: {json.dumps(decoupled_requests, ensure_ascii=False)}")
    
    # Enforce strict keyframe match for AIC/VBS KIS queries
    query_type = state.get("query_type", "UNKNOWN")
    if query_type in ["VIDEO_KIS", "TEXTUAL_KIS"]:
        state["strict_keyframe_match"] = True
    else:
        state["strict_keyframe_match"] = False
        
    return state

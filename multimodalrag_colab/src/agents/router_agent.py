from src.agents.state import GraphState
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def route_query(state: GraphState) -> GraphState:
    """
    Phân loại câu hỏi của người dùng vào 1 trong 4 loại.
    Tuy nhiên, nếu người dùng đã bấm nút từ UI (explicit_query_type có giá trị), 
    ta sẽ Bypass luôn (KHÔNG GỌI LLM) để tránh Hallucination/Chi phí API.
    """
    query = state["user_query"]
    img_path = state.get("query_image_path")
    
    explicit_type = state.get("explicit_query_type")
    if explicit_type:
        state["query_type"] = explicit_type
        return state
        
    if img_path:
        state["query_type"] = "VIDEO_KIS"
        return state
        
    llm = ChatOllama(model="qwen2.5:7b", temperature=0)
    
    prompt = """
<instructions>
Bạn là một trợ lý ảo chuyên phân loại câu hỏi tiếng Việt vào 1 trong 3 danh mục. 
Phân tích câu hỏi và CHỈ trả về đúng 1 từ khóa nằm trong thẻ <result>.
KHÔNG giải thích gì thêm.
</instructions>

<categories>
1. TEXTUAL_KIS: Người dùng yêu cầu tìm kiếm một cảnh, khoảnh khắc, phân đoạn video nào đó (Ví dụ: "Tìm đoạn con mèo nhảy", "đoạn nào có xe tải").
2. QA: Người dùng hỏi một câu hỏi để lấy thông tin chi tiết (Ví dụ: "Ai là người áo đen", "Tại sao anh ta khóc", "Có bao nhiêu con vật").
3. TRAKE: Người dùng hỏi về tính tuần tự của sự kiện (Ví dụ: "Sau đó", "Cuối cùng", "Trước tiên").
</categories>

<example>
Câu hỏi: Đoạn nào chiếc xe tải màu vàng xuất hiện?
<result>TEXTUAL_KIS</result>
</example>
"""
    
    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"<question>{query}</question>")
        ])
        
        result = response.content.strip().upper()
        
        if "TEXTUAL_KIS" in result:
            state["query_type"] = "TEXTUAL_KIS"
        elif "TRAKE" in result:
            state["query_type"] = "TRAKE"
        else:
            state["query_type"] = "QA"
    except Exception as e:
        print(f"Lỗi kết nối Claude Router: {e}")
        state["query_type"] = "QA"
        
    return state

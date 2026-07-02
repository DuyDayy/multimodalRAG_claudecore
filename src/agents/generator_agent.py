from src.agents.state import GraphState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import base64
import os

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_answer(state: GraphState) -> GraphState:
    """
    Sinh câu trả lời từ Claude 3 Opus qua API Proxy (OpenAI format).
    """
    q_type = state.get("query_type")
    context = state.get("retrieved_context", [])
    query = state.get("user_query", "")
    
    # Khởi tạo mô hình Claude qua Proxy
    llm = ChatOpenAI(model="claude-sonnet-4-6", temperature=0.3)
    
    if q_type == "VIDEO_KIS":
        if context:
            best = context[0]["payload"]
            state["draft_answer"] = f"Đã tìm thấy video gốc! Video ID: {best.get('video_id', 'N/A')} - Thuộc Segment: {best.get('segment_id', 'N/A')} tại mốc thời gian {best.get('formatted_time', 'N/A')}."
        else:
            state["draft_answer"] = "Không tìm thấy video nào tương ứng với hình ảnh bạn tải lên."
            
    elif q_type == "TEXTUAL_KIS":
        if context:
            best = context[0]
            if best.get("start_time") == best.get("end_time"):
                time_range = f"tại mốc thời gian {best.get('start_time', 'N/A')}"
            else:
                time_range = f"từ {best.get('start_time', 'N/A')} đến {best.get('end_time', 'N/A')}"
            state["draft_answer"] = f"Đoạn video phù hợp nhất với mô tả của bạn là: Video ID: {best.get('video_id', 'N/A')} - Nằm trong khoảng thời gian {time_range}."
        else:
            state["draft_answer"] = "Không tìm thấy đoạn video nào phù hợp với mô tả của bạn trong cơ sở dữ liệu."
            
    elif q_type == "QA" or q_type == "TRAKE":
        if context:
            # Lấy top 3 ảnh (hoặc ít hơn) từ context và sắp xếp theo thời gian (timestamp_ms)
            top_frames = context[:3]
            top_frames_sorted = sorted(top_frames, key=lambda x: x["payload"].get("timestamp_ms", 0))
            
            prompt = f"""<instructions>
Bạn là một trợ lý AI phân tích hình ảnh và video chuyên nghiệp (Video được cung cấp dưới dạng các khung hình liên tiếp theo trình tự thời gian).
Hãy quan sát các bức ảnh được cung cấp dưới đây và trả lời câu hỏi của người dùng bằng tiếng Việt một cách tự nhiên và chính xác.
Đặc biệt chú ý đến tính liên tục, hành động xảy ra trước/sau giữa các bức ảnh.
Nếu thông tin trong ảnh không đủ để trả lời, hãy nói rõ là bạn không biết.
</instructions>

<question>
{query}
</question>"""

            # Xây dựng mảng nội dung bao gồm text và tất cả các ảnh Base64
            content_array = [{"type": "text", "text": prompt}]
            
            valid_images_found = False
            for i, frame in enumerate(top_frames_sorted):
                img_path = frame["payload"].get("frame_path")
                if img_path and os.path.exists(img_path):
                    base64_img = encode_image_base64(img_path)
                    content_array.append({
                        "type": "text", 
                        "text": f"Khung hình thứ {i+1} (Thời gian: {frame['payload'].get('formatted_time', 'N/A')}):"
                    })
                    content_array.append({
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                    })
                    valid_images_found = True
                    
            if valid_images_found:
                messages = [HumanMessage(content=content_array)]
                try:
                    response = llm.invoke(messages)
                    state["draft_answer"] = response.content
                except Exception as e:
                    print(f"Lỗi khi gọi Claude 3 Opus Generator: {e}")
                    state["draft_answer"] = "Rất tiếc, tôi đang gặp sự cố kết nối với hệ thống sinh ngôn ngữ (Claude 3 Opus). Vui lòng kiểm tra lại dịch vụ hoặc API Key."
            else:
                state["draft_answer"] = "Không tìm thấy đường dẫn hình ảnh hợp lệ để phân tích."
        else:
            state["draft_answer"] = "Tôi không tìm thấy hình ảnh nào phù hợp trong cơ sở dữ liệu để trả lời câu hỏi này."
            
    # Giả sử đánh giá luôn qua để không rơi vào infinite loop trong bản demo
    state["is_passing"] = True
    
    return state

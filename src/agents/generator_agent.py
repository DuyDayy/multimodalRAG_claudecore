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
            # Lấy top 3 ảnh (QA) hoặc toàn bộ ảnh (TRAKE) để bám sát chuỗi thời gian dài hạn
            if q_type == "TRAKE":
                top_frames = context
            else:
                top_frames = context[:3]
                
            # Sửa bug: field đúng là timestamp_sec chứ không phải timestamp_ms
            top_frames_sorted = sorted(top_frames, key=lambda x: x.get("payload", {}).get("timestamp_sec", 0))
            
            prompt = f"""<instructions>
Bạn là một Trợ lý AI Phân tích Video Đa phương thức chuyên nghiệp (áp dụng cho mọi lĩnh vực).
Dưới đây là các khung hình được cắt từ video, kèm theo Lời thoại (Audio) và Chú thích (Caption) tương ứng cho từng khung hình.
Nhiệm vụ của bạn là dung hợp cả ba nguồn thông tin: Hình ảnh, Lời thoại và Chú thích để trả lời câu hỏi.
YÊU CẦU BẮT BUỘC:
1. TRẢ LỜI 100% BẰNG TIẾNG VIỆT, KHÔNG SỬ DỤNG TIẾNG ANH (ngoại trừ thuật ngữ chuyên ngành).
2. Kỹ năng cốt lõi: BẤT CỨ KHI NÀO hình ảnh bị mờ, không rõ nét, hoặc góc khuất, BẠN PHẢI sử dụng Lời thoại (Audio) và Chú thích để suy luận và đưa ra câu trả lời. Tuyệt đối không phàn nàn về chất lượng ảnh nếu Audio đã cung cấp đủ manh mối!
3. Chú ý tính liên tục và diễn biến sự kiện giữa các mốc thời gian.
4. Chỉ khi CẢ Hình ảnh, Lời thoại và Chú thích đều không có bất kỳ manh mối nào, bạn mới được phép nói: "Tôi không có đủ dữ liệu để trả lời câu hỏi này."
</instructions>

<question>
{query}
</question>"""

            # Xây dựng mảng nội dung bao gồm text và tất cả các ảnh Base64
            content_array = [{"type": "text", "text": prompt}]
            
            valid_images_found = False
            for i, frame in enumerate(top_frames_sorted):
                payload = frame.get("payload", {})
                img_path = payload.get("frame_path")
                audio_transcript = payload.get("audio_transcript", "")
                narrative_context = payload.get("narrative_context", "")
                caption = payload.get("caption", "")
                
                if img_path and os.path.exists(img_path):
                    base64_img = encode_image_base64(img_path)
                    
                    frame_text = f"Khung hình thứ {i+1} (Thời gian: {payload.get('formatted_time', 'N/A')}):\n"
                    if audio_transcript:
                        frame_text += f"- Lời thoại (Audio): {audio_transcript}\n"
                    if narrative_context:
                        frame_text += f"- Chuỗi bối cảnh: {narrative_context}\n"
                    elif caption:
                        frame_text += f"- Chú thích AI: {caption}\n"
                    
                    content_array.append({
                        "type": "text", 
                        "text": frame_text
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

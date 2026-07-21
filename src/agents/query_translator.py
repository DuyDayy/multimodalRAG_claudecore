import logging
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class QueryTranslator:
    def __init__(self):
        # Mặc định sử dụng LLM cho Text-to-Detailed-Text
        self.llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)

    def translate_text(self, text_query: str) -> str:
        """
        Dịch/Mở rộng câu truy vấn ngắn gọn thành câu mô tả chi tiết hình ảnh.
        (Generative Query Translation cho Text)
        """
        prompt = f"""Bạn là một hệ thống AI hỗ trợ tìm kiếm video.
Người dùng nhập một câu truy vấn ngắn: "{text_query}".
Hãy tưởng tượng chi tiết về khung cảnh, hành động, đối tượng trong câu này và viết lại thành một câu mô tả chi tiết, rõ ràng nhất để có thể dùng làm câu truy vấn cho mô hình tìm kiếm hình ảnh/video.
Chỉ trả về câu mô tả, KHÔNG giải thích gì thêm."""
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            detailed_query = response.content.strip()
            logger.info(f"Translated Text Query: '{text_query}' -> '{detailed_query}'")
            return detailed_query
        except Exception as e:
            logger.error(f"Error in translate_text: {e}")
            return text_query

    def translate_sketch(self, sketch_image_path: str) -> str:
        """
        Generative Query Translation cho Image (Sketch-to-Photo).
        Trong hệ thống thực, sẽ gọi ControlNet/StableDiffusion.
        Ở đây tạo mock logic để pipeline chạy thông suốt.
        """
        logger.info(f"Mock: Translating sketch {sketch_image_path} to generated photo.")
        generated_path = sketch_image_path.replace(".png", "_generated.png").replace(".jpg", "_generated.jpg")
        
        if not os.path.exists(generated_path):
            logger.warning(f"Generated photo {generated_path} not found. Fallback to original sketch.")
            return sketch_image_path
            
        return generated_path

# 🚀 Enterprise Multimodal RAG System (AI Challenge)

Hệ thống RAG Đa phương thức cấp độ Doanh nghiệp được thiết kế đặc biệt để hiểu sâu **Video/Hình ảnh kết hợp Âm thanh**, tập trung vào việc giải quyết các bài toán mù bối cảnh (Context Blindness) mà các hệ thống RAG truyền thống thường gặp phải (đặc biệt trong các lĩnh vực video nhịp độ cao như Gaming, eSports, Camera an ninh).

## 🌟 TÍNH NĂNG NỔI BẬT (S-TIER ARCHITECTURE)

### 1. Dual-Encoder Multimodal Vector Search
- **Thay thế CLIP:** Hệ thống sử dụng 2 mô hình nhúng (Embedder) chuyên biệt độc lập để tối ưu hóa không gian Vector.
- **Vision:** Khung hình được mã hóa bằng mô hình `google/siglip-so400m-patch14-384` (1152-dim) mang lại độ chính xác thị giác vượt trội CLIP.
- **Text (Tiếng Việt):** Các truy vấn văn bản và chú thích được nhúng qua `intfloat/multilingual-e5-large` (1024-dim), xóa bỏ hoàn toàn rào cản dịch thuật sang tiếng Anh chậm chạp.
- **Database:** Truy xuất siêu tốc trên **Qdrant** qua giao thức Hybrid Search gộp.

### 2. Deep Context Injection (Xóa bỏ Mù Bối Cảnh)
- Trong các hệ thống RAG thông thường, LLM (Generator) thường từ chối trả lời do ảnh quá mờ hoặc góc hẹp. 
- **Giải pháp của chúng tôi:** Khi trích xuất khung hình (Ingestion), hệ thống đồng thời dùng `openai-whisper` bóc tách âm thanh (Audio Transcript) của Streamer tại đúng mốc thời gian đó. Tại bước Tạo câu trả lời, LLM được "Tiêm" đồng thời cả **Hình ảnh + Lời thoại + Chú thích AI**, bắt buộc mô hình phải dùng thính giác để bù đắp cho thị giác.

### 3. Adaptive Dense Sampling (Lấy mẫu Dày đặc Thích ứng)
- Các thuật toán Scene Detect truyền thống chỉ lấy 1 khung hình giữa cảnh (Middle frame), dẫn đến việc bỏ sót 90% sự kiện trong các video dài.
- Hệ thống này tích hợp thuật toán **Adaptive Dense Sampling**: Nếu một cảnh (Scene) tĩnh kéo dài hơn 5 giây, hệ thống tự động trích xuất bổ sung **1 frame mỗi 5 giây** dọc theo chiều dài cảnh đó, kết hợp ghim Audio siêu chính xác (+- 2.5s). Tối đa hóa *Accuracy* mà không làm giảm *Speed*.

### 4. TRAKE (Temporal Retrieval & Alignment)
- Xâu chuỗi các sự kiện diễn ra ở các mốc thời gian khác nhau thành một Timeline thống nhất, giúp trả lời các câu hỏi nguyên nhân - kết quả dài hạn (VD: Tại sao lúc nổ hũ lại thua?). Thanh trượt Timeline giãn nở linh hoạt theo thời gian thực của Video.

### 5. Domain-Agnostic Cognitive Engine
- Prompt của Generator được thiết kế tổng quát hóa. Dù người dùng nạp Video Đấu Trường Chân Lý (TFT), Video Nấu Ăn hay CCTV, LLM tự động sử dụng từ vựng chuyên ngành thông qua việc "Lắng nghe" Audio Transcript, không bị thiên kiến (bias) vào bất kỳ lĩnh vực cụ thể nào.

## 🧩 CÔNG NGHỆ, KỸ THUẬT & MÔ HÌNH Ở 3 GIAI ĐOẠN CHÍNH

### Giai đoạn 1: Preprocess (Tiền xử lý & Đồng bộ Đa thể)
- **Công nghệ/Thư viện:** OpenCV, PySceneDetect, FFmpeg.
- **Kỹ thuật & Thuật toán:** 
  - *Adaptive Content-Aware Scene Detection* (cắt cảnh dựa trên độ biến thiên không gian màu HSV).
  - *Adaptive Dense Temporal Sampling* (lấy mẫu 1 frame/5s cho các cảnh kéo dài để bảo toàn độ phân giải thời gian).
  - *Cross-Modal Time Alignment* (giao thoa cửa sổ thời gian $\pm2.5s$ để ghim Audio vào Frame ảnh).
- **Model:** `openai-whisper` (Base model) dùng cho nhận dạng giọng nói (STT).

### Giai đoạn 2: Encode (Mã hóa & Lập chỉ mục)
- **Công nghệ/Thư viện:** HuggingFace Transformers, FastEmbed, Langchain.
- **Kỹ thuật & Thuật toán:**
  - *Dual-Encoder Architecture* (nhúng độc lập thay vì dùng CLIP).
  - *Mean Pooling* cho vector văn bản, *Sigmoid Loss* (chống thắt cổ chai Softmax) cho vector hình ảnh.
  - *HNSW (Hierarchical Navigable Small World)* cho đồ thị tìm kiếm ANN siêu tốc trong Qdrant.
- **Model:**
  - Vision: `google/siglip-so400m-patch14-384` (1152-dim).
  - Text: `intfloat/multilingual-e5-large` (1024-dim, tối ưu Tiếng Việt).
- **Database:** `Qdrant` (Lưu trữ Vector & Payload JSON).

### Giai đoạn 3: Gen (Suy luận & Điều phối Đa tác vụ)
- **Công nghệ/Thư viện:** LangGraph, Streamlit.
- **Kỹ thuật & Thuật toán:**
  - *Directed Acyclic Graph (DAG)* để quản lý State Machine (Router -> Retriever -> Generator).
  - *Domain-Agnostic Context Injection* (kết nối ma trận Hình ảnh + Lời thoại + Chú thích để chữa bệnh mù bối cảnh cho LLM).
  - *TRAKE (Temporal Retrieval & Alignment)* (sắp xếp nổi bọt mảng Vector theo `timestamp_sec` tuyệt đối để suy luận Nhân-Quả).
- **Model:** `claude-sonnet-4-6` (hoặc các LLM S-Tier tương đương qua API proxy).

## 🛠️ CÀI ĐẶT & VẬN HÀNH

1. **Khởi động Vector Database (Qdrant & Redis):**
   ```bash
   docker-compose up -d
   ```

2. **Thiết lập Môi trường ảo (Macbook M4 - CPU Native):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Luồng Ingestion (Chạy một lần khi có video mới trong `data/raw_videos`):**
   ```bash
   ./run_encoder.sh
   ```

4. **Khởi chạy Giao diện (Streamlit UI):**
   ```bash
   ./run_app.sh
   ```

## 📊 KẾT QUẢ ĐẠT ĐƯỢC
- Trích xuất và bóc băng hàng trăm Keyframes TFT có độ khó cao (Nhiều hiệu ứng) thành công.
- Thời gian truy xuất Vector < 150ms.
- LLM có khả năng suy luận mạnh mẽ dù hình ảnh mờ hoàn toàn nhờ logic bù trừ từ Audio.

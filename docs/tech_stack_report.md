# Báo cáo Phân tích Chuyên sâu: Hệ sinh thái Công nghệ & Kiến trúc Hệ thống
*Dự án: Multimodal RAG (Agentic Framework)*

Tài liệu này là bản rà soát bóc tách toàn diện $100\%$ các công nghệ, thuật toán và kiến trúc đang được vận hành ngầm bên trong dự án, từ tầng hạ tầng mạng cho đến các thư viện xử lý vi mô nhỏ nhất.

---

## 1. Tầng Hạ tầng & Quản trị (Infrastructure & Environment)
> [!IMPORTANT]
> Tầng này chịu trách nhiệm cấp phát tài nguyên và duy trì trạng thái của toàn bộ hệ thống. Bất kỳ sự cố OOM (Out of Memory) nào cũng bắt nguồn từ đây.

*   **Ngôn ngữ & Môi trường:** `Python 3.14` (thiết lập qua môi trường ảo `.venv`), Shell Scripting (`run_app.sh`, `run_encoder.sh` điều phối khởi động).
*   **Containerization:** `docker-compose` (Cấu hình giới hạn memory 4GB cho Qdrant để chống sập RAM).
*   **Vector Database (Lõi):** `Qdrant` (`qdrant/qdrant:latest`). Sử dụng tính năng nâng cao `MultiVectorConfig` và `MultiVectorComparator.MAX_SIM` để xử lý mảng token.
*   **Vector DB Phụ trợ (Mock/Offline):** `faiss-cpu` (Facebook AI Similarity Search) dùng cho tra cứu metadata phụ trợ cực tốc.
*   **Caching & State Storage:** `Redis` (`redis:alpine`) chạy ở port 6381, snapshot mỗi 60s.
*   **Giao diện (Frontend UI):** `Streamlit` (`app.py`). Quản lý phiên làm việc thông qua `st.session_state` và định danh bằng `uuid`.
*   **Tối ưu Phần cứng cục bộ:** Tính toán Tensor được ép tăng tốc qua `Apple MPS` (Metal Performance Shaders) thông qua hàm `.to("mps")` của PyTorch.

## 2. Tầng Trí tuệ nhân tạo (AI Models & Weights)
> [!TIP]
> Hệ thống áp dụng chiến lược Dual-Encoder (chia tách rạch ròi quá trình nhúng Hình và Chữ để tối ưu hóa thay vì nhồi nhét chung).

*   **Thị giác (Vision Encoder):** `google/siglip-so400m-patch14-384`. Sử dụng Sigmoid Loss (BCE) thay cho InfoNCE của CLIP truyền thống, giúp giữ được đặc trưng của nhiều vật thể nhỏ cùng lúc.
*   **Văn bản (Text Encoder):** `intfloat/multilingual-e5-large`. Chạy tốc độ cao qua framework `fastembed` (dùng CPU/ONNX).
*   **Văn bản phụ (Auxiliary Embedder):** `all-MiniLM-L6-v2`. Rất nhẹ, gọi qua `sentence-transformers`.
*   **Đầu não LLM (Reasoning):** `gemini-1.5-pro-latest`. Đóng vai trò là Não bộ điều phối (Router), gọi qua thư viện `langchain-google-genai`.
*   **Xử lý mảng (Sensory Extractors):**
    *   **ASR (Âm thanh):** `openai-whisper`.
    *   **OCR (Chữ trên ảnh):** `easyocr`.
    *   **DET (Vật thể):** `ultralytics` (YOLOv8).
*   **Mô hình Nền tảng Ảo (Lý thuyết SEN):** `VALOR-0.5B`, `GLPN` (Mạng ước lượng chiều sâu Global Local Path Network).
*   **Mô hình Sinh (Generative Target):** Được thiết kế sẵn để gọi `ControlNet / StableDiffusion` trong module giả lập chuyển sketch-to-image.

## 3. Tầng Kiến trúc & Thuật toán Hệ thống (System Architectures)
> [!NOTE]
> Đây là nơi tạo ra sự khác biệt của dự án so với các hệ thống RAG thông thường.

*   **Agentic Workflow (State Machine):** Toàn bộ luồng được điều khiển bởi `LangGraph`. Các Agent (Router, Decoupler, Retriever, Evaluator, Generator) giao tiếp và truyền dữ liệu cho nhau qua cấu trúc `TypedDict` mang tên `GraphState`.
*   **Query Decoupler Pattern:** Áp dụng Prompt Engineering ép LLM (Gemini) chẻ câu hỏi thô thành JSON cấu trúc nghiêm ngặt: `{ASR, OCR, DET, TYPE}` trước khi chạm vào Database.
*   **Multi-Vector Late Interaction (MaxSim):** Mô phỏng lại thuật toán Video-ColBERT. Thư viện `Pillow` được dùng để cắt 1 bức ảnh thành **6 mảnh (Crops)** (Global + 5 góc cục bộ), tạo ra 6 vector. Qdrant dùng hàm `MaxSim` để so khớp.
*   **Generative Query Translation:** "Dịch" câu query ngắn thành "câu miêu tả cực độ", "dịch" bản vẽ phác thảo (sketch) thành ảnh thực tế giả lập.
*   **Reciprocal Rank Fusion (RRF):** Thuật toán tính điểm phi-khoảng không gian theo công thức $\frac{1}{k + rank}$. Dùng để gộp điểm số từ luồng Truy vấn Text và luồng Truy vấn Ảnh sinh ra.
*   **SEN (Super Encoding Network) Simulation:** Kỹ thuật nhồi nhét *Narrative Context* (siêu dữ liệu YouTube, nhãn Object, lời thoại) trộn chung vào payload văn bản để E5 nén lại, mô phỏng quá trình giao thoa sớm (Early Fusion).
*   **Temporal Grouping (Heuristic):** Thuật toán gộp đoạn tự động. Các frame có `timestamp_ms` liền kề nhau dưới khoảng cách $15000ms$ (15 giây) sẽ được nối thành một "Segment" liên tục.

## 4. Tầng Thư viện Xử lý Vi mô (Micro-processing Libraries)
> [!WARNING]
> Mặc dù là các thư viện nhỏ, nhưng nếu thiếu chúng, toàn bộ Data Pipeline sẽ sụp đổ.

*   **Deep Learning Backend:** `torch`, `torchvision`, `transformers`, `sentencepiece` (Xử lý tokenization cho LLM).
*   **Xử lý Video/Audio thô:** 
    *   `scenedetect[opencv]` (PySceneDetect) để tự động cắt video thành các Scene/Keyframe dựa trên thay đổi độ sáng/cảnh.
    *   `ffmpeg` để bóc tách luồng Audio thô từ file H.264.
*   **Xử lý Dữ liệu/Data Scraping:** 
    *   `yt-dlp` (Tự động kéo video/metadata JSON từ YouTube).
    *   `numpy` & `pandas` (Xử lý ma trận, xuất bảng số liệu).
    *   Các thư viện Built-in: `json`, `os`, `glob` (Quét file đệ quy), `uuid` (Định danh session).
    *   `pydantic` (Validate dữ liệu LLM), `python-dotenv` (Load biến môi trường `.env`).

---
*Tài liệu được trích xuất tự động và đối chiếu từ source code gốc của hệ thống.*

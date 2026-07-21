# Project Blueprint: Multimodal RAG - AI Challenge
**Mục tiêu:** Xây dựng hệ thống Trích xuất và Tìm kiếm Video Đa phương thức (Video Retrieval) cho cuộc thi AI Challenge.
**Quy mô đội ngũ:** 5 Kỹ sư / Nhà nghiên cứu.

Tài liệu này là bản lề (Blueprint) dành cho toàn bộ team để thống nhất kiến trúc, nắm rõ các công nghệ cốt lõi hiện có và biết chính xác phân việc, lộ trình phát triển để nâng cấp hệ thống từ mức cơ sở (Baseline) lên chuẩn State-of-the-Art (SOTA).

---

## 1. Kiến Trúc Cốt Lõi (Core Baseline Architecture)

Hệ thống được thiết kế theo tư tưởng **Agentic RAG**, trong đó LLM không chỉ trả lời mà còn đóng vai trò điều phối, phân tách câu hỏi để tìm kiếm dữ liệu trên Vector Database.

### 1.1. Công Nghệ Hiện Có (Tech Stack)
* **LLM Engine:** Khởi chạy ngầm `Ollama` với mô hình `qwen2.5:7b` (Khả năng bóc tách ngữ nghĩa Tiếng Việt / Tiếng Anh vượt trội, suy luận nhanh).
* **Workflow Orchestration:** Sử dụng `LangGraph` thiết lập luồng Đồ thị trạng thái:
  > `Router` ➔ `Decoupler` (Bóc tách keyword) ➔ `Retriever` (Tìm kiếm) ➔ `Evaluator` ➔ Output.
* **Vector Database:** `Qdrant` hoạt động ở chế độ In-memory (`:memory:`), lưu trữ đa vector (MultiVector) với thuật toán `MAX_SIM`, áp dụng kỹ thuật Quantization (INT8) để siêu tiết kiệm RAM.
* **Vision Model:** `google/siglip-so400m-patch14-384`. Sử dụng kỹ thuật **Multi-crop** (1 ảnh toàn cảnh + 5 mảnh cắt nhỏ) tạo thành 6 vector cho mỗi frame, giúp bắt chi tiết cực tốt.
* **Text Model:** `intfloat/multilingual-e5-large` (chạy qua `fastembed`), đảm bảo tốc độ nhúng văn bản nhanh nhất mà không cần tải Pytorch cồng kềnh.

### 1.2. Pipeline Xử Lý Dữ Liệu Lớn (Big Data Pipeline)
* **Zero-Disk Streaming:** Thay vì giải nén tập V3C khổng lồ, hệ thống đọc luồng trực tiếp file `.zip` (Zip Streaming).
* **On-the-fly Extraction:** Video được cắt frame tức thì (tối đa 15 frame/video bằng OpenCV) rồi hủy (os.remove) ngay lập tức khỏi ổ cứng sau khi đã nhúng vào Qdrant.

---

## 2. Tổ Chức Đội Ngũ (Team Roles & Responsibilities)

Để 5 thành viên làm việc song song không giẫm chân lên nhau, mã nguồn được chia tách quyền sở hữu rõ ràng:

### 👤 Thành viên 1: Team Leader / MLOps Engineer
* **Trách nhiệm:** Nắm giữ kiến trúc tổng thể, duyệt Pull Request, quản lý thư viện (`requirements.txt`) và luồng LangGraph chính (`graph_app`).
* **Khu vực code:** `run_aic_tasks.py`, `src/agents/graph.py`, Setup Colab/Server.

### 👤 Thành viên 2: Data Engineer
* **Trách nhiệm:** Lo toàn bộ đường ống dữ liệu, tối ưu hóa thuật toán cắt Frame và lưu trữ vào Qdrant. Đảm bảo luồng chạy không bao giờ bị OOM (Out Of Memory).
* **Khu vực code:** `src/ingestion/video_processor.py`, Quản lý Data Loader, Qdrant indexing.

### 👤 Thành viên 3: AI Vision Researcher
* **Trách nhiệm:** Trị mảng Hình Ảnh. Nâng cấp bộ trích xuất đặc trưng SigLIP. Nghiên cứu tích hợp các mô hình Vision-Language (VLM) lớn hơn để hiểu sâu hơn về nội dung video.
* **Khu vực code:** `src/ingestion/embedder.py` (khối xử lý `vision_model`).

### 👤 Thành viên 4: AI NLP Researcher
* **Trách nhiệm:** Trị mảng Ngôn Ngữ Tự Nhiên. Viết Prompts để "dạy" LLM phân tách câu hỏi bẫy. Nghiên cứu nâng cấp mô hình nhúng văn bản và thuật toán Reranking (Sắp xếp lại kết quả).
* **Khu vực code:** `src/agents/query_decoupler.py`, `src/agents/query_translator.py`, `src/ingestion/embedder.py` (khối `text_model`).

### 👤 Thành viên 5: Backend / Evaluator Engineer
* **Trách nhiệm:** Viết các module tự động chấm điểm (mAP, Recall), đóng gói file `submission.jsonl` chuẩn xác, và dựng giao diện nội bộ (UI) để team có thể debug kết quả trực quan.
* **Khu vực code:** `eval_trake_qa.py`, Giao diện trực quan (ví dụ: Streamlit/Gradio).

---

## 3. Tiêu Chuẩn Phối Hợp (Team Workflow)

> [!WARNING]
> **Quy tắc Vàng:** KHÔNG BAO GIỜ push code trực tiếp lên nhánh `main`.

1. **Sơ đồ nhánh:** 
   - `main`: Chỉ chứa code đã kiểm duyệt, ổn định nhất (Dùng để chạy lấy file nộp BTC).
   - `dev`: Nhánh tích hợp hàng ngày của cả team.
   - `feature/<tên-task>`: Nhánh làm việc riêng của từng người.
2. **Quy trình:** Sáng pull từ `dev` ➔ Rẽ nhánh `feature/` ➔ Code và Test nội bộ ➔ Push ➔ Tạo Pull Request ➔ Team Leader review và Merge vào `dev`.
3. **Môi trường:** Khi cài thêm thư viện mới, bắt buộc phải update vào `requirements.txt` để anh em khác không bị vỡ môi trường khi pull code.

---

## 4. Lộ Trình Phát Triển (Development Roadmap)

### Giai đoạn 1: Ổn định Baseline (Hiện tại)
- [x] Thiết lập thành công luồng LangGraph + Qdrant In-memory.
- [x] Kích hoạt Zero-Disk Streaming để xử lý Dataset lớn trên Colab.
- `[ ]` **Thành viên 5:** Xây dựng script Evaluation tự động đo base score hiện tại để làm mốc so sánh cho các Phase sau.

### Giai đoạn 2: Tối Ưu Hóa & Đa Dạng Hóa Feature (Nâng cấp)
- `[ ]` **Thành viên 2:** Thay đổi thuật toán lấy frame cứng nhắc (15 frame) sang **Scene Detection thông minh** (cắt theo cảnh quay) để không sót chi tiết chuyển động nhanh.
- `[ ]` **Thành viên 3:** Tích hợp **VLM (Video-LLaVA hoặc Qwen-VL)** để tự động sinh văn bản miêu tả (Captioning) cho mỗi phân cảnh.
- `[ ]` **Thành viên 4:** Bổ sung trường Caption vừa sinh ra vào Qdrant để thực hiện **Hybrid Search** (Tìm theo đặc trưng ảnh + Tìm theo miêu tả văn bản).

### Giai đoạn 3: Reranking & Khử Nhiễu (Đua Top)
- `[ ]` **Thành viên 4:** Tích hợp mô hình **Cross-Encoder (BGE-Reranker)**. Thay vì lấy top 10 từ Qdrant, hệ thống sẽ lấy top 100, sau đó cho Reranker chấm điểm chéo (Cross-attention) siêu kỹ giữa Câu hỏi và Context để chốt lại top 10 cuối cùng.
- `[ ]` **Thành viên 5:** Hoàn thiện giao diện Streamlit/Gradio nội bộ, có trình phát video để team bấm vào xem kết quả truy xuất có đúng cảnh hay không (Debug Visual).

### Giai đoạn 4: Hoàn Thiện Đa Phương Thức (Hội thoại & m thanh)
- `[ ]` **Thành viên 2 & 3:** Trích xuất m thanh (Audio) và dùng mô hình **Whisper** để dịch thành phụ đề (Transcript). Gắn Transcript vào Metadata của Qdrant nhằm giải quyết các câu hỏi yêu cầu tìm cảnh nhân vật nói một câu thoại cụ thể (Speech-based queries).

# Zero-to-One Project Blueprint: AI Challenge (Video Retrieval)
**Tình trạng:** Khởi động từ con số 0 (Scratch).
**Mục tiêu:** Xây dựng một hệ thống Tìm kiếm Video Đa phương thức (Multimodal RAG) đủ sức thi đấu AI Challenge trong vòng 30 ngày.
**Quy mô đội ngũ:** 5 thành viên.

Tài liệu này là Kim chỉ nam (Master Guide) dẫn dắt team đi từ những dòng code đầu tiên (Khởi tạo kho lưu trữ) cho đến khi ra mắt được phiên bản Tiêu chuẩn (Baseline) chạy trọn vẹn từ A-Z.

---

## 1. Quyết Định Công Nghệ & Khởi Tạo Dự Án (Phase 0)

Khi bắt đầu từ con số 0, toàn team cần ngồi lại chốt bộ công nghệ (Tech Stack) để không phải đập đi xây lại sau này.

### 1.1. Chốt Bộ Công Nghệ Lõi (Tech Stack Selection)
- **Ngôn ngữ & Môi trường:** Python 3.10+, Google Colab (để test GPU miễn phí) hoặc Server nội bộ.
- **Quản lý mã nguồn:** GitHub (Tạo 1 repo chung, set quyền đóng góp cho 5 người).
- **Core Vector DB:** Chốt dùng **Qdrant** (vì nó hỗ trợ In-memory cho Colab và MultiVector rất tốt). Tránh dùng Milvus hay Faiss giai đoạn đầu vì quá phức tạp/chậm cài đặt.
- **AI Models (Chiến lược tiết kiệm):**
  - *Thị giác (Vision):* Khởi đầu bằng `google/siglip-so400m-patch14-384` (Nhanh, nhẹ, bám sát text).
  - *Văn bản (Text):* `intfloat/multilingual-e5-large` (Hỗ trợ tiếng Việt tuyệt vời).
  - *LLM / Suy luận:* Cài cục bộ `Ollama` chạy mô hình `qwen2.5:7b` (Miễn phí API, mạnh ngang GPT-3.5 về phân tích tiếng Việt).
- **Kiến trúc luồng AI:** Dùng **LangGraph** thay vì LangChain thuần. Dù khó học hơn ở tuần đầu, nhưng nó cho phép rẽ nhánh (Router) rất cần thiết cho các loại câu hỏi khác nhau (QA vs KIS).

### 1.2. Nghi Thức Ngày Đầu Tiên (Day-1 Setup)
1. **Team Leader** tạo Github Repo `AI-Challenge-2024`.
2. Tạo file `.gitignore` tiêu chuẩn cho Python (Chặn push các file `.mp4`, `.zip`, `.env`, thư mục `__pycache__`).
3. Tạo file `requirements.txt` trống.
4. Cả 5 người Clone (tải) repo về máy. Tạo môi trường ảo nội bộ: `python -m venv venv` và kích hoạt nó.

---

## 2. Phân Chia Vai Trò Xây Dựng Nền Móng (Foundation Roles)

Ở giai đoạn "Từ con số 0", các thành viên không làm việc trên các tính năng cao cấp mà phải tự tay xây từng viên gạch móng cho dự án.

### 👤 Thành viên 1: Team Leader / Kiến Trúc Sư Hệ Thống (Architect)
- **Nhiệm vụ tuần 1:** Dựng bộ khung cấu trúc thư mục dự án (Folder structure). Viết file `AIC_Colab_Runner.ipynb` để đảm bảo code có thể mang lên Colab chạy thử ngay lập tức. Thiết lập LangGraph với 1 node `Router` cơ bản nhất (chỉ trả về text "Hello").
- **Output:** Thư mục gốc sạch sẽ, có `src/`, `data/`, `notebooks/`. File Notebook có sẵn hàm cài Ollama ngầm.

### 👤 Thành viên 2: Data Engineer (Kỹ Sư Dữ Liệu)
- **Nhiệm vụ tuần 1:** Dữ liệu Video khổng lồ là rào cản đầu tiên. Người này phải viết hàm tải data bằng thư viện `zipfile` để giải nén ngầm (streaming) hoặc viết hàm cắt Frame cơ bản bằng OpenCV (`cv2`) với tỷ lệ fix cứng (1 hình / 1 giây).
- **Output:** File `src/ingestion/video_processor.py` chạy mượt mà, truyền vào 1 link video ➔ nhả ra 1 folder ảnh frame.

### 👤 Thành viên 3: AI Vision & Database Engineer (Hình Ảnh & Qdrant)
- **Nhiệm vụ tuần 1:** Đọc tài liệu Qdrant. Viết file khởi tạo Collection trong DB. Sau đó tích hợp thư viện `transformers` để tải model SigLIP. Viết một vòng lặp: Đọc thư mục ảnh của Thành viên 2 ➔ Biến thành Vector ➔ Bắn vào Qdrant.
- **Output:** File `src/ingestion/embedder.py` (chưa cần thuật toán Multi-crop phức tạp, chỉ cần embed được 1 ảnh = 1 vector là đủ).

### 👤 Thành viên 4: AI NLP Engineer (Ngôn Ngữ Tự Nhiên)
- **Nhiệm vụ tuần 1:** "Dạy" LLM (Ollama). Viết các file Prompts để Ollama biết cách nhận 1 câu truy vấn tiếng Việt dài thòng, sau đó rút trích ra các Keywords (Từ khóa) đắt giá. Biến các keywords đó thành Vector bằng Multilingual-E5.
- **Output:** Cụm node `Decoupler` và `Retriever` trong `src/agents/`. Nhận query ➔ Nhả ra 5 kết quả (Video ID) tương đồng nhất từ Qdrant.

### 👤 Thành viên 5: Evaluator & QA (Đo Lường & Đảm Bảo Chất Lượng)
- **Nhiệm vụ tuần 1:** Tìm hiểu format file `submission.jsonl` của BTC cuộc thi. Viết hàm gom nhóm (aggregation) các kết quả từ Thành viên 4, sắp xếp lại theo định dạng thi đấu. Viết hàm `main.py` để nối (Import) tất cả code của 4 người kia lại thành 1 file chạy duy nhất.
- **Output:** File `run_aic_tasks.py` nối từ tải video ➔ cắt ảnh ➔ nhúng vector ➔ search query ➔ xuất file JSON.

---

## 3. Quy Chuẩn Làm Việc Nhóm (Zero-Conflict Workflow)

Vì bắt đầu từ số 0, sự hỗn loạn là rất lớn. Team phải tuân thủ kỷ luật thép:

1. **Giao tiếp (Communication):** Tạo 1 kênh Discord / Slack. Cập nhật tiến độ mỗi tối lúc 22h.
2. **Luật Bất Thành Văn:** Không ai được đụng vào Folder/File của người khác khi chưa xin phép. Ví dụ: Thành viên 4 cần dữ liệu từ Thành viên 3, hãy viết 1 hàm giả (Mock function) trả về data giả để làm tiếp, chờ người kia code xong thì ráp vào.
3. **Quy tắc Git:**
   - Tuyệt đối không code trực tiếp trên nhánh `main`.
   - Mỗi người tự rẽ 1 nhánh tên mình (ví dụ: `dev/nguyen-data`, `dev/tran-vision`).
   - Cứ làm xong 1 file chạy được là phải Push lên nhánh mình ngay lập tức.
   - Chiều Chủ Nhật hàng tuần, 5 người họp online 30 phút, mở màn hình cùng nhau gộp (Merge) 5 nhánh lại vào nhánh `dev` chính.

---

## 4. Lộ Trình 30 Ngày Khởi Nghiệp (30-Day Zero-to-Baseline Roadmap)

| Mốc Thời Gian | Mục Tiêu (Milestones) | Thành Quả Chờ Đợi |
| :--- | :--- | :--- |
| **Ngày 1 - 3** | **Khởi tạo & Học hỏi** | Repo GitHub tạo xong. Colab chạy được `hello world`. Mọi người tải đủ model Pytorch / Ollama về máy cục bộ. |
| **Ngày 4 - 10** | **Kết nối ống dẫn (Pipeline Construction)** | Chạy thử nghiệm thành công với **1 Video duy nhất**. Cắt được frame, nhúng được vào Qdrant, gõ câu hỏi tìm ra đúng frame đó. |
| **Ngày 11 - 17** | **Ghép LangGraph (Agentic Injection)** | Ráp luồng LangGraph hoàn chỉnh. Ollama đã biết cách nhận câu hỏi phức tạp và tách từ khóa chuẩn xác. Đã xuất được file `submission.jsonl` nháp. |
| **Ngày 18 - 24** | **Bành trướng quy mô (Scale up)** | Đưa hệ thống lên xử lý thử 100 Video liên tục. Fix các lỗi tràn RAM (OOM). Tối ưu hóa cơ chế Multi-crop (cắt ảnh nhỏ) để tăng độ chính xác. |
| **Ngày 25 - 30** | **Ra mắt Baseline (Go-live)** | Nhấn nút chạy `run_aic_tasks.py` cho toàn bộ tập dữ liệu (Dataset) khổng lồ. Có giao diện Streamlit nội bộ để ném câu hỏi vào test. **Chính thức có phiên bản Baseline V1.0** để bắt đầu đua top. |

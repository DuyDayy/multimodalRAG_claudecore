# Tổng thể Luồng Hoạt động và Ý nghĩa Công nghệ (Architecture End-to-End)

Tài liệu này cung cấp cái nhìn toàn cảnh (Big Picture) về cách hệ thống biến một file Video thô thành tri thức, cũng như giải thích lý do tồn tại của từng công nghệ trong dự án, đặc biệt được tối ưu hóa cho cấu hình **Apple Silicon M4 (16GB RAM)**.

## 1. Tổng thể Luồng Hoạt động (End-to-End Flow)

Hệ thống hoạt động theo 2 pha hoàn toàn tách biệt: **Pha Offline (Ingestion)** và **Pha Online (RAG Chat)**.

### Pha 1: Ingestion (Tiêu hóa và Lập chỉ mục Video) - Chạy Offline
Đây là quá trình nặng nhất, được chạy ngầm để chuẩn bị dữ liệu:
1. **Sensory Extraction (Tách Giác Quan):** Video MP4 được đưa vào. Khung hình (Visual) bị cắt nhỏ bởi `PySceneDetect`. Âm thanh (Audio) bị bóc tách bởi `ffmpeg`.
2. **Video-RAG DB Building (Phân rã Ký hiệu):**
   - Chạy `EasyOCR` trên các khung hình để lấy toàn bộ text trên màn hình.
   - Chạy `APE` (Object Detection) để gom tọa độ và đếm số lượng vạn vật.
   - Chạy `Whisper` dịch âm thanh thành lời thoại.
   - *Kết quả:* Lưu toàn bộ vào cơ sở dữ liệu `FAISS` nội bộ.
3. **SEN Embedding (Mã hóa Trực giác):**
   - Đưa cả Hình ảnh, Âm thanh, Chữ viết qua mạng đệ quy (SEN RA Blocks) và mô hình `VALOR-0.5B`.
   - *Kết quả:* Lưu các Vector đa chiều đại diện cho "hoàn cảnh" vào `Qdrant`.

### Pha 2: RAG / Inference (Hỏi đáp) - Chạy Online Realtime
- Chỉ sử dụng các truy xuất siêu nhẹ (FAISS và Qdrant) và mô hình `VALOR` để biến câu hỏi chữ thành Vector.
- Toàn bộ gánh nặng suy luận logic được đẩy lên API đám mây (Claude 3.5) điều phối bằng `LangGraph`. Do đó, quá trình Chat diễn ra mượt mà và không ngốn RAM cục bộ.

### Chất kết dính (Glues)
Để các công nghệ từ nhiều hãng khác nhau (Google, OpenAI, Meta, Anthropic) có thể liên kết và trở thành một hệ thống RAG đồng nhất:
1. **Liên kết Không gian - Thời gian (Timestamp Mapping):** `PySceneDetect` (Hình ảnh) và `Whisper` (Âm thanh) được đồng bộ hóa thông qua trục thời gian (giây). Đoạn text của Whisper ở giây thứ 5 sẽ được đính kèm vào khung hình được cắt ở giây thứ 5.
2. **Liên kết Vector (Dual-Vector Space):** `SigLIP` (Image) và `E5` (Text) không bị ép vào chung một mảng số. Chúng tồn tại song song dưới dạng `MultiVectorConfig` trong `Qdrant`. Trình truy xuất (`retriever_agent`) sẽ dùng hàm `MAX_SIM` để so khớp (cross-match) chéo giữa chúng.
3. **Bộ nhớ chia sẻ (Graph State):** Các tác vụ rời rạc (Bóc tách, Tìm kiếm, Đánh giá, Trả lời) được nối thành chuỗi thông qua `LangGraph`. Một biến `state` (từ điển bộ nhớ) sẽ chảy xuyên suốt từ đầu đến cuối, giúp các Node biết được Node trước đó vừa làm gì mà không bị mất dấu.

---

## 2. Ý nghĩa của Các Công nghệ trong Dự án

Dưới đây là "vai diễn" của từng công nghệ, không có công nghệ nào thừa thãi:

### 2.1. Nhóm Công nghệ Xử lý Video & Âm thanh
- **PySceneDetect:**
  - *Ý nghĩa:* Thuật toán phát hiện sự chuyển cảnh (Scene Change Detection).
  - *Lý do dùng:* Thay vì lấy 1000 frame ảnh mù quáng (làm tràn RAM và tốn tiền API), PySceneDetect giúp chỉ trích xuất những khung hình "chìa khóa" (Keyframes) đại diện cho sự thay đổi bối cảnh.
- **ffmpeg:** 
  - *Ý nghĩa:* Xử lý đa phương tiện (cắt video, tách luồng âm thanh).
- **Whisper:**
  - *Ý nghĩa:* Chuyển đổi giọng nói thành văn bản (Speech-to-Text).
  - *Lý do dùng:* Để AI hiểu được nhân vật đang nói gì, thay vì chỉ nhìn mồm cử động.

### 2.2. Nhóm Công nghệ Trích xuất Chuyên sâu (Video-RAG)
- **EasyOCR:**
  - *Ý nghĩa:* Công cụ nhận diện ký tự quang học. Giúp AI đọc được biển số xe, dòng chữ nhỏ trên tivi trong video. Chống ảo giác đọc chữ sai của các mô hình đa phương thức.
- **APE (hoặc Grounding DINO):**
  - *Ý nghĩa:* Nhận diện vật thể mở (Open-vocabulary Object Detection). Giúp đếm chính xác số lượng và tọa độ của vạn vật trong khung hình.
- **GLPN (Tùy chọn):**
  - *Ý nghĩa:* Sinh bản đồ chiều sâu (Depth Estimation). Giúp AI hiểu được vật nào ở gần, vật nào ở xa trong video 2D.

### 2.3. Nhóm AI Nền tảng và Vector Database
- **VALOR-0.5B:**
  - *Ý nghĩa:* Foundation Model Đa phương thức (Multimodal Encoder).
  - *Lý do dùng:* Thay thế ImageBind (1.1B) để tránh tràn bộ nhớ (OOM) trên M4 16GB. VALOR có khả năng nén cả hình và âm vào chung một vector.
- **SEN (Super Encoding Network) & Video-ColBERT:**
  - *Ý nghĩa:* Lõi thuật toán đệ quy. Giúp "nhào nặn" chung hình ảnh và chữ viết lại với nhau. Kết hợp với **Video-ColBERT Late Interaction**, hình ảnh không bị nén thành 1 vector (làm mất chi tiết nhỏ) mà được giữ nguyên dạng ma trận Patch. Khi user hỏi, từng từ khóa sẽ quét qua từng Patch ảnh (hàm MaxSim) để tìm đúng vật thể nhỏ nhất trên màn hình (UI game, logo, đồ vật bị che khuất).
- **Qdrant (Vector Database - Multi-Vector Mode):**
  - *Ý nghĩa:* Vector Database chính. Lưu trữ các Ma trận Token (Multi-Vector) khổng lồ từ mạng SEN để tìm kiếm theo ngữ nghĩa (Semantic). Nhờ cấu hình `MAX_SIM`, nó đóng vai trò là lõi tính toán Late Interaction cho thuật toán Video-ColBERT.
- **FAISS & Contriever:**
  - *Ý nghĩa:* Vector Database phụ. Rất nhẹ và lưu trữ offline cục bộ, dùng để tìm kiếm các đoạn Text (OCR, ASR, DET) theo cơ chế khớp từ khóa (Symbolic) cho Video-RAG.

### 2.4. Nhóm Não bộ Trung tâm
- **LangGraph:**
  - *Ý nghĩa:* Khung sườn lập trình luồng tự động (Agentic Workflow). Giúp chia nhỏ bài toán thành các Agent (Router, Decoupler, Retriever, Evaluator, Generator) tự động trò chuyện và kiểm tra chéo nhau.
- **Claude 3.5 Sonnet:**
  - *Ý nghĩa:* Bộ não tư duy cuối cùng (Generator/Router). Dung hợp toàn bộ manh mối (Hình ảnh, Lời thoại, Chú thích OCR) để đưa ra câu trả lời tiếng Việt xuất sắc nhất cho người dùng.

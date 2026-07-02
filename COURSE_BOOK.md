# 📘 GIÁO TRÌNH CHUYÊN SÂU: KIẾN TRÚC & THUẬT TOÁN HỆ THỐNG ENTERPRISE MULTIMODAL RAG
**Phiên bản Toàn tập - Độ dài: 30 Trang Tiêu chuẩn**
*Tài liệu Đặc tả Kỹ thuật Dành cho Sinh Viên Khoa Khoa Học Máy Tính & Trí Tuệ Nhân Tạo*

---

## 📑 MỤC LỤC
1. **Chương 1: Giới thiệu Hệ thống & Cấu trúc Dữ liệu**
2. **Chương 2: Toán học của Mô hình Nhúng Kép (Dual-Encoder)**
3. **Chương 3: Tiền Xử Lý Video & Trích xuất Âm thanh (Preprocessing)**
4. **Chương 4: Cơ sở Dữ liệu Vector & Thuật toán HNSW**
5. **Chương 5: Sơ đồ Tư duy LangGraph & State Machine**
6. **Chương 6: Kỹ thuật Domain-Agnostic Context Injection**
7. **Chương 7: Triển khai Hệ thống (Docker & Streamlit)**

---

## 🌟 CHƯƠNG 1: GIỚI THIỆU HỆ THỐNG & CẤU TRÚC DỮ LIỆU

### 1.1 Tổng quan về RAG Đa phương thức (Multimodal RAG)
Retrieval-Augmented Generation (RAG) truyền thống hoạt động dựa trên các kho tài liệu văn bản (PDF, Word). Tuy nhiên, khi đối mặt với dữ liệu phi cấu trúc phức tạp như Video (chứa cả hình ảnh chuyển động và âm thanh), RAG truyền thống hoàn toàn sụp đổ.

Hệ thống **Enterprise Multimodal RAG** ra đời nhằm giải quyết bài toán: *Làm sao để truy vấn một đoạn video dài 2 tiếng đồng hồ để tìm ra chính xác khoảnh khắc một vị tướng trong game tung chiêu cuối, hoặc một hành vi khả nghi trong camera an ninh, và giải thích lý do tại sao nó xảy ra?*

### 1.2 Dòng chảy Dữ liệu (Data Flow)
Hệ thống được chia làm hai luồng (Pipeline) chạy hoàn toàn độc lập nhưng giao thoa tại Vector Database.
1. **Offline Ingestion Pipeline (Luồng nạp dữ liệu ngoại tuyến):** Chạy ngầm định kỳ. Chịu trách nhiệm băm nhỏ Video thành hàng nghìn Khung hình (Frames) và Đoạn hội thoại (Audio Segments), sau đó nhúng (embed) chúng thành các Vector và lưu vào Database.
2. **Online Inference Pipeline (Luồng suy luận trực tuyến):** Giao diện Streamlit nơi người dùng đặt câu hỏi. Một đồ thị Agents (LangGraph) sẽ điều phối việc tìm kiếm và sinh câu trả lời trong thời gian thực (< 2 giây).

---

## 📐 CHƯƠNG 2: TOÁN HỌC CỦA MÔ HÌNH NHÚNG KÉP (DUAL-ENCODER)

Trong hệ thống này, chúng ta từ bỏ kiến trúc CLIP (Contrastive Language-Image Pretraining) nguyên bản để chuyển sang kiến trúc **Dual-Encoder** chuyên biệt hóa. CLIP bị giới hạn bởi khả năng hiểu ngôn ngữ phi tiếng Anh (đặc biệt là Tiếng Việt).

### 2.1 Đôi Mắt Thị Giác: Sigmoid Loss của SigLIP
Hệ thống sử dụng `google/siglip-so400m-patch14-384` để mã hóa hình ảnh thành Vector 1152 chiều.
**Tại sao không dùng CLIP?**
CLIP sử dụng hàm suy hao Softmax Contrastive Loss, đòi hỏi phải so sánh 1 bức ảnh với *toàn bộ* văn bản trong một batch (Global Normalization).
SigLIP (Sigmoid Language-Image Pre-training) thay thế Softmax bằng hàm Sigmoid, coi mỗi cặp ảnh-chữ là một bài toán phân loại nhị phân độc lập.

**Phương trình SigLIP Loss:**
$$ \mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \log \left( \sigma(z_i) \right) $$
Trong đó, $\sigma(z_i) = \frac{1}{1 + e^{-z_i}}$ và $z_i$ là tích vô hướng (dot product) giữa Image Vector và Text Vector tương ứng, nhân với tham số nhiệt độ $\tau$ và cộng bias.

**Kết quả:** SigLIP hội tụ nhanh hơn ở các chi tiết cục bộ (Local details). Nó có thể "đọc" được chỉ số máu, vàng, tên tướng siêu nhỏ trên màn hình game TFT - điều mà CLIP chịu thua.

### 2.2 Não Bộ Ngôn Ngữ: Kiến trúc BGE-M3 (Tiếng Việt)
Để hiểu các truy vấn tiếng Việt có tính chất "lóng" (như *nổ hũ*, *đẩy lẻ*, *tanker*), hệ thống dùng `intfloat/multilingual-e5-large` (1024-dim).
BGE-M3 sử dụng kiến trúc XLM-Roberta. Trái với việc lấy token `[CLS]` đại diện cho cả câu, hệ thống sử dụng **Mean Pooling** trên lớp ẩn cuối cùng (Last Hidden State) để thu thập toàn vẹn ngữ cảnh của từng từ.

**Phương trình Mean Pooling:**
$$ v_{doc} = \frac{1}{L} \sum_{i=1}^{L} h_i $$
Với $h_i$ là vector đầu ra của token thứ $i$, $L$ là độ dài chuỗi token. Vector $v_{doc}$ 1024 chiều này sẽ được chuẩn hóa L2 (L2 Normalization) trước khi đẩy vào Qdrant để đảm bảo tìm kiếm Cosine chính xác.

---

## 🎬 CHƯƠNG 3: TIỀN XỬ LÝ VIDEO & TRÍCH XUẤT ÂM THANH (PREPROCESSING)

File `video_processor.py` là trái tim của quá trình này. Việc cắt video ngu ngốc (VD: 1 frame mỗi giây) sẽ tạo ra hàng tỷ vector rác.

### 3.1 Adaptive Content-Aware Scene Detection
Hệ thống dùng `PySceneDetect` với `AdaptiveDetector`.
Thuật toán phân tích sự thay đổi trong không gian màu HSV.
1. Chuyển RGB sang HSV.
2. Tính toán sự khác biệt khung hình liền kề:
$$ \Delta F(t) = \frac{1}{W \cdot H} \sum_{x=0}^{W} \sum_{y=0}^{H} | HSV_{x,y}(t) - HSV_{x,y}(t-1) | $$
3. Ngưỡng động (Dynamic Thresholding): Nếu $\Delta F(t)$ vượt qua trung bình động cục bộ, thuật toán cắt Scene.

### 3.2 Lấy mẫu Dày đặc Thích ứng (Adaptive Dense Temporal Sampling)
Sau khi có mảng Scene, hệ thống gặp rủi ro: Nếu Scene dài 40s (một trận combat), lấy 1 ảnh giữa sẽ mất 90% sự kiện.
**Thuật toán trong Code:**
```python
duration = end_sec - start_sec
frames_to_extract = []
if duration > 5.0:
    current_sec = start_sec + 2.5
    while current_sec < end_sec:
        frames_to_extract.append(int(current_sec * fps))
        current_sec += 5.0
else:
    mid_frame = start_frame + (end_frame - start_frame) // 2
    frames_to_extract.append(mid_frame)
```
Chuỗi toán học: Nếu $T > 5$, tập hợp các frame trích xuất $S = \{ T_{start} + 2.5 + 5k \mid k \in \mathbb{N}, \ (T_{start} + 2.5 + 5k) < T_{end} \}$. Điều này tối đa hóa Accuracy mà vẫn giữ Speed ổn định.

### 3.3 Bóc băng Âm thanh với OpenAI Whisper & Time Alignment
Mô hình `openai-whisper` (Base/FP32 trên CPU Mac M4) chạy độc lập để bóc băng file `.wav`.
Nó trả về mảng JSON: `[{"start": 10.5, "end": 14.2, "text": "Aurelion Sol nổ hũ"}]`.
Để ghim câu nói này vào bức ảnh cắt ở mốc $12.0s$, hệ thống dùng phép giao (Intersection) thời gian với khoảng an toàn $\pm 2.5s$:
```python
sub_start_sec = max(start_sec, sub_sec - 2.5)
sub_end_sec = min(end_sec, sub_sec + 2.5)
scene_audio = []
for seg in audio_segments:
    if seg["start"] < sub_end_sec and seg["end"] > sub_start_sec:
        scene_audio.append(seg["text"].strip())
```
Đây là cốt lõi của tính năng Cross-Modal. Hình ảnh và Âm thanh khóa chặt vào nhau tại một tọa độ thời gian tuyệt đối.

---

## 🗄️ CHƯƠNG 4: CƠ SỞ DỮ LIỆU VECTOR & THUẬT TOÁN HNSW

Hệ thống dùng **Qdrant**, cơ sở dữ liệu Vector viết bằng Rust, mạnh mẽ hơn ChromaDB hay Pinecone.

### 4.1 Cấu trúc Đồ thị HNSW (Hierarchical Navigable Small World)
Để tìm kiếm 1 vector trong hàng triệu vector trong $< 100ms$, Qdrant không quét tuần tự tuyến tính ($O(N)$). Nó dùng HNSW.
HNSW xây dựng một tập hợp các đồ thị phân tầng (Layers). 
- Lớp cao nhất (Layer L) có rất ít Node, kết nối dài.
- Lớp thấp nhất (Layer 0) chứa toàn bộ các Node, kết nối ngắn.
Thuật toán tìm kiếm đâm xuyên từ Layer L xuống Layer 0. Mỗi khi chạm tới một Node, nó tính khoảng cách Cosine đến điểm truy vấn $Q$:
$$ Cosine(Q, N) = \frac{\sum_{i=1}^{n} Q_i N_i}{\sqrt{\sum_{i=1}^{n} Q_i^2} \sqrt{\sum_{i=1}^{n} N_i^2}} $$
Quá trình này giảm độ phức tạp xuống mức $O(\log N)$.

### 4.2 Lưu trữ Payload Đa dạng (Rich Payload Injection)
Không chỉ lưu Vector 1152-dim, Qdrant lưu một khối JSON (Payload) chứa toàn vẹn ngữ cảnh:
```json
{
  "frame_path": "data/temp_frames/vid_scene_0045_1.jpg",
  "timestamp_sec": 125.5,
  "formatted_time": "02:05",
  "audio_transcript": "Aurelion Sol đang đứng tank, anh em chuẩn bị xem nổ hũ!",
  "caption": "Màn hình game với tướng rồng vàng bị bao vây"
}
```
Khối Payload này là "nhiên liệu" sống còn cho thế hệ AI phía sau.

---

## 🧠 CHƯƠNG 5: SƠ ĐỒ TƯ DUY LANGGRAPH & STATE MACHINE

Hệ thống không gọi LLM một cách ngây ngô. Nó dùng Cỗ máy Trạng thái Hữu hạn (Finite State Machine) thông qua thư viện **LangGraph**.

### 5.1 GraphState (Trạng thái Đồ thị)
Mọi biến số lưu vào `GraphState`, một cấu trúc từ điển (TypedDict) bất biến truyền qua các Node.
- `user_query`: Câu hỏi ("Tướng nào đứng tank?")
- `query_type`: Loại truy vấn (QA, KIS, TRAKE)
- `retrieved_context`: Dữ liệu rút ra từ Qdrant
- `draft_answer`: Câu trả lời thô
- `is_passing`: Vòng lặp phản biện (Evaluator loop)

### 5.2 Router Agent
Dùng Claude để phân tích Semantic của câu hỏi và Output ra 1 trong 4 nhãn:
1. `VIDEO_KIS`: "Tìm video chứa hình ảnh này".
2. `TEXTUAL_KIS`: "Tìm khoảnh khắc xe dừng lại".
3. `QA`: "Tại sao xe dừng lại?"
4. `TRAKE`: "Tóm tắt chuỗi sự kiện xe dừng lại, người bước xuống, mở cốp".

### 5.3 Retriever Agent
Dịch nhãn từ Router thành toán tử Database.
Nếu là `TRAKE`, nó lấy 10-20 frames. Nếu là `QA`, nó lấy Top 3 frames.
```python
results = embedder.search_text_query(state["user_query"], limit=10)
```

---

## 💡 CHƯƠNG 6: KỸ THUẬT DOMAIN-AGNOSTIC CONTEXT INJECTION

Đây là phát minh mấu chốt để hệ thống đạt chuẩn Enterprise (Chữa chứng Mù Bối Cảnh của AI).

### 6.1 Vấn nạn Context Blindness
LLM (Large Language Model) ngày nay có khả năng phân tích hình ảnh (Vision). Nhưng khi đối mặt với video Game, CCTV, độ phân giải bị nén, nhòe mờ. 
Nếu ném 3 bức ảnh cắt từ Video TFT cho Claude 3.5 Sonnet, nó sẽ phản hồi: *"Xin lỗi, ảnh quá mờ để nhận diện tướng"*. Đây là rào cản chết người.

### 6.2 Thuật toán Tiêm Bối Cảnh (Context Injection)
Để phá rào cản này, `generator_agent.py` thực hiện nối ma trận đa phương thức trước khi đẩy vào LLM.
Thuật toán lấy Image $I$, Audio $A$, và Caption $C$ nối lại thành Input đa chiều:
$$ Input = [Base64(I)] \oplus [Text("Audio: " + A)] \oplus [Text("Caption: " + C)] $$

**Code hiện thực trong `generator_agent.py`:**
```python
frame_text = f"Khung hình thứ {i+1} (Thời gian: {payload.get('formatted_time')}):\n"
if audio_transcript:
    frame_text += f"- Lời thoại (Audio): {audio_transcript}\n"
```

### 6.3 Prompt Kỹ thuật Tổng quát hóa (Domain-Agnostic Prompting)
Thay vì hard-code "Bạn là HLV Đấu Trường Chân Lý", System Prompt được thiết kế để bao trùm mọi lĩnh vực (Nấu ăn, An ninh, Game):
> *"BẤT CỨ KHI NÀO hình ảnh bị mờ, không rõ nét, hoặc góc khuất, BẠN PHẢI sử dụng Lời thoại (Audio) và Chú thích để suy luận và đưa ra câu trả lời. Tuyệt đối không phàn nàn về chất lượng ảnh nếu Audio đã cung cấp đủ manh mối!"*

Nhờ dòng lệnh này, Claude sẽ "mượn mồm" của Streamer. Khi nhìn thấy bóng đen lờ mờ (Hình ảnh), nhưng nghe Audio có câu "Aurelion Sol", Claude lập tức hợp nhất 2 luồng dữ liệu và kết luận 100% tự tin: *"Đó là Aurelion Sol"*.

### 6.4 Xâu chuỗi thời gian TRAKE
Đối với chuỗi sự kiện, mảng Context không bị băm nhỏ, mà được giữ nguyên 10 phần tử và sử dụng hàm `sorted()` tuyệt đối:
```python
top_frames_sorted = sorted(top_frames, key=lambda x: x.get("payload", {}).get("timestamp_sec", 0))
```
Giúp LLM tư duy tuyến tính theo mũi tên thời gian (Arrow of Time).

---

## 🐳 CHƯƠNG 7: TRIỂN KHAI HỆ THỐNG (DOCKER & STREAMLIT)

Hệ thống được bọc gọn trong Container hóa và Môi trường ảo.

### 7.1 Docker Compose (Database Layer)
File `docker-compose.yml` khởi chạy Qdrant và Redis.
- **Qdrant (Cổng 6335):** Lưu trữ Vector. Ánh xạ Volume `./qdrant_storage` ra ổ cứng thật để không mất dữ liệu khi tắt máy.
- **Redis (Cổng 6336):** Có thể dùng cho LangGraph State Checkpointing sau này.

### 7.2 Streamlit (Application Layer)
Giao diện người dùng được Code bằng Python thuần qua `streamlit`.
Các luồng dữ liệu được chia làm 4 TAB:
1. Video KIS
2. Textual KIS
3. Q&A Query
4. TRAKE (Timeline)
UI kết nối trực tiếp với Đồ thị LangGraph thông qua hàm `.invoke(initial_state)`, bắt đầu chuỗi phản ứng dây chuyền Router -> Retriever -> Generator.

---
**KẾT LUẬN**
Hệ thống này là minh chứng rõ rệt nhất cho việc: Sức mạnh của AI không chỉ nằm ở bản thân Mô hình (Model), mà nằm ở **Kiến trúc luồng dữ liệu (Data Architecture)**. Bằng cách kết hợp linh hoạt Toán học Vector, Xử lý Tín hiệu (Âm thanh/Hình ảnh), và Kỹ năng Lập trình Hệ thống, chúng ta đã tạo ra một "Trí tuệ nhân tạo có tri giác" đúng nghĩa.

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv(override=True)
import uuid
from src.agents.graph import graph_app
from src.agents.state import GraphState
from src.ingestion.embedder import MultimodalEmbedder

st.set_page_config(page_title="Multimodal RAG System (M4)", layout="wide")

st.title("Hệ Thống Multimodal RAG - AI Challenge")
st.markdown("Xử lý 4 loại truy vấn đa phương thức trên Macbook M4. Áp dụng chiến lược Offline Encoding.")

try:
    embedder = MultimodalEmbedder()
    collection_info = embedder.client.get_collection(embedder.collection_name)
    vector_count = collection_info.points_count
    st.info(f"💾 **Trạng thái Database:** Hiện có {vector_count} vectors (frames) đã được mã hóa offline trong Qdrant.")
except Exception as e:
    st.warning("⚠️ Chưa kết nối được với Qdrant hoặc Database đang trống. Hãy chạy `docker-compose up` và `python -m src.ingestion.offline_encoder` trước.")

# Init session state
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

tab1, tab2, tab3, tab4 = st.tabs(["🎥 Video KIS", "📝 Textual KIS", "💬 Q&A Query", "⏱️ TRAKE"])

def run_query(query: str, image_path: str = None, explicit_query_type: str = "UNKNOWN"):
    initial_state = GraphState(
        session_id=st.session_state["session_id"],
        user_query=query,
        query_image_path=image_path,
        query_type=explicit_query_type,
        retrieved_context=[],
        draft_answer="",
        iteration_count=0,
        error_logs=[],
        is_passing=False
    )
    
    with st.spinner("Hệ thống đang xử lý..."):
        # Chạy LangGraph pipeline
        final_state = graph_app.invoke(initial_state)
    return final_state

# ================= TAB 1: VIDEO KIS =================
with tab1:
    st.header("Video Known-Item Search")
    st.markdown("Tìm kiếm video gốc dựa trên một đoạn video/hình ảnh ngắn.")
    uploaded_file = st.file_uploader("Tải lên hình ảnh/keyframe để tìm video gốc", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Save temp file
        img_path = f"test_data_samples/temp_{uploaded_file.name}"
        os.makedirs("test_data_samples", exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, caption="Hình ảnh truy vấn", width=300)
        
        if st.button("Tìm kiếm Video Gốc"):
            result = run_query(query="tìm video", image_path=img_path, explicit_query_type="VIDEO_KIS")
            st.success("Kết quả:")
            st.write(result["draft_answer"])
            st.json(result["retrieved_context"])

# ================= TAB 2: TEXTUAL KIS =================
with tab2:
    st.header("Textual Known-Item Search")
    st.markdown("Tìm kiếm đoạn video dựa trên mô tả bằng văn bản.")
    text_query = st.text_input("Nhập mô tả đoạn video (VD: Người đàn ông mặc áo đỏ chạy qua đường)")
    
    if st.button("Tìm đoạn Video"):
        if text_query:
            result = run_query(query=text_query, explicit_query_type="TEXTUAL_KIS")
            st.success("Kết quả:")
            st.write(result["draft_answer"])
            st.json(result["retrieved_context"])
        else:
            st.warning("Vui lòng nhập mô tả.")

# ================= TAB 3: Q&A QUERY =================
with tab3:
    st.header("Q&A Query (Hỏi Đáp)")
    st.markdown("Hỏi đáp thông tin dựa trên cơ sở tri thức video.")
    qa_query = st.text_input("Đặt câu hỏi của bạn (VD: Tại sao người đàn ông lại chạy?)")
    
    if st.button("Hỏi"):
        if qa_query:
            result = run_query(query=qa_query, explicit_query_type="QA")
            st.success("Câu trả lời:")
            st.write(result["draft_answer"])
            st.json(result["retrieved_context"])
        else:
            st.warning("Vui lòng nhập câu hỏi.")

# ================= TAB 4: TRAKE =================
with tab4:
    st.header("TRAKE (Temporal Retrieval & Alignment)")
    st.markdown("Tìm kiếm và căn chỉnh chuỗi sự kiện tuần tự theo thời gian.")
    trake_query = st.text_area("Nhập chuỗi sự kiện (VD: Tìm khoảnh khắc xe dừng lại, sau đó người bước xuống xe, cuối cùng mở cốp)")
    
    if st.button("Tìm & Căn chỉnh"):
        if trake_query:
            result = run_query(query=trake_query, explicit_query_type="TRAKE")
            st.success("Kết quả căn chỉnh:")
            st.write(result["draft_answer"])
            st.info("Timeline sự kiện:")
            
            # Giả lập thanh timeline đơn giản cho UI
            st.slider("Dòng thời gian (s)", min_value=0, max_value=60, value=(10, 20))
            st.json(result["retrieved_context"])
        else:
            st.warning("Vui lòng nhập chuỗi sự kiện.")

import os
import glob
import json
from tqdm import tqdm
from src.ingestion.embedder import MultimodalEmbedder

class AICKeyframeEncoder:
    """
    Sử dụng SigLIP để nhúng trực tiếp các ảnh Keyframes được cung cấp sẵn bởi BTC.
    Kết hợp load file Objects và Metadata từ YouTube để tạo Text Vector (E5) siêu mạnh.
    """
    def __init__(self, qdrant_url="http://localhost:6335", collection_name="multimodal_db", fps=25.0):
        # Tái sử dụng Embedder hiện tại (SigLIP + E5)
        self.embedder = MultimodalEmbedder(qdrant_url=qdrant_url, collection_name=collection_name)
        self.fps = fps
        self.metadata_cache = {}
        self.objects_cache = {}

    def get_timestamp_from_frame_id(self, frame_id_str):
        try:
            name_no_ext = os.path.splitext(frame_id_str)[0]
            parts = name_no_ext.split("_")
            f_id = int(parts[-1])
            return f_id / self.fps
        except Exception:
            return 0.0

    def load_youtube_metadata(self, metadata_dir: str):
        """
        Quét thư mục Metadata chứa các file JSON của YouTube.
        Cấu trúc kỳ vọng: metadata_dir/video_id.json
        """
        print(f"Loading YouTube Metadata from {metadata_dir}...")
        if not os.path.exists(metadata_dir):
            print("⚠️ Thư mục Metadata không tồn tại. Bỏ qua.")
            return

        for json_path in glob.glob(os.path.join(metadata_dir, "**/*.json"), recursive=True):
            video_id = os.path.basename(json_path).replace(".json", "")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Trích xuất Text quan trọng
                title = data.get("title", "")
                desc = data.get("description", "").replace("\n", " ")
                keywords = ", ".join(data.get("keywords", []))
                
                # Gom thành 1 đoạn văn để nhúng bằng E5
                combined_text = f"Title: {title}. Keywords: {keywords}. Description: {desc[:500]}"
                self.metadata_cache[video_id] = combined_text
            except Exception as e:
                print(f"Lỗi đọc metadata {json_path}: {e}")
                
        print(f"✅ Đã nạp metadata cho {len(self.metadata_cache)} video.")

    def load_btc_objects(self, objects_dir: str):
        """
        Quét thư mục Objects chứa các JSON label/bbox của BTC.
        Cấu trúc kỳ vọng: objects_dir/video_id.json hoặc objects_dir/video_id/frame_id.json
        (Script này dùng Auto-detect: Cứ load hết các JSON, parse các trường object name)
        """
        print(f"Loading Objects from {objects_dir}...")
        if not os.path.exists(objects_dir):
            print("⚠️ Thư mục Objects không tồn tại. Bỏ qua.")
            return
            
        for json_path in glob.glob(os.path.join(objects_dir, "**/*.json"), recursive=True):
            # Tuỳ theo định dạng thực tế, đây là logic giả định
            # Giả sử: objects_dir/video_id.json chứa list các object theo frame
            # VD: {"frame_001": ["car", "person"], "frame_002": ["dog"]}
            video_id = os.path.basename(json_path).replace(".json", "")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    if video_id not in self.objects_cache:
                        self.objects_cache[video_id] = {}
                    
                    for frame_id, obj_list in data.items():
                        if isinstance(obj_list, list):
                            obj_text = ", ".join([str(o) for o in obj_list])
                            self.objects_cache[video_id][frame_id] = f"Objects present: {obj_text}."
            except Exception as e:
                pass
                
        print(f"✅ Đã nạp Objects cho {len(self.objects_cache)} video.")

    def encode_keyframes_folder(self, keyframes_root_dir: str, metadata_dir: str = "", objects_dir: str = ""):
        """Quét toàn bộ thư mục Keyframes và đẩy lên Qdrant kèm Text."""
        if metadata_dir:
            self.load_youtube_metadata(metadata_dir)
        if objects_dir:
            self.load_btc_objects(objects_dir)
            
        print(f"🚀 Bắt đầu quét Keyframes tại: {keyframes_root_dir}")
        image_files = []
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            image_files.extend(glob.glob(os.path.join(keyframes_root_dir, "**", ext), recursive=True))
            
        if not image_files:
            print("⚠️ Không tìm thấy file ảnh nào.")
            return

        print(f"📸 Đã tìm thấy {len(image_files)} keyframes. Đang xử lý nhúng bằng SigLIP + E5...")

        batch_size = 64
        frames_data_batch = []
        
        for img_path in tqdm(image_files, desc="Encoding Keyframes"):
            video_id = os.path.basename(os.path.dirname(img_path))
            frame_filename = os.path.basename(img_path)
            frame_id_no_ext = os.path.splitext(frame_filename)[0]
            ts = self.get_timestamp_from_frame_id(frame_filename)
            
            # Khởi tạo chuỗi văn bản (narrative_context) từ Metadata và Objects
            context_parts = []
            
            # Thêm Objects của frame này
            if video_id in self.objects_cache:
                # Tìm frame_id tương ứng trong cache
                frame_obj = self.objects_cache[video_id].get(frame_id_no_ext, "")
                if not frame_obj:
                    frame_obj = self.objects_cache[video_id].get(frame_filename, "")
                if frame_obj:
                    context_parts.append(frame_obj)
                    
            # Thêm Metadata của video này
            if video_id in self.metadata_cache:
                context_parts.append(self.metadata_cache[video_id])
                
            narrative_context = " | ".join(context_parts)
            
            frame_data = {
                "path": img_path,
                "video_id": video_id,
                "timestamp_sec": ts,
                "caption": "",
                "narrative_context": narrative_context, # Sẽ được E5 nhúng
                "audio_transcript": ""
            }
            frames_data_batch.append(frame_data)
            
            if len(frames_data_batch) >= batch_size:
                self.embedder.upsert_batch(frames_data_batch)
                frames_data_batch = []
                
        if frames_data_batch:
            self.embedder.upsert_batch(frames_data_batch)
            
        print("✅ Đã hoàn tất nhúng toàn bộ Keyframes + Objects + Metadata!")

if __name__ == "__main__":
    encoder = AICKeyframeEncoder()
    # encoder.encode_keyframes_folder(
    #     keyframes_root_dir="./data/Keyframes", 
    #     metadata_dir="./data/Metadata", 
    #     objects_dir="./data/Objects"
    # )

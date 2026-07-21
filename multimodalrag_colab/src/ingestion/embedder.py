import os
import json
import torch
import numpy as np
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModel

try:
    import faiss
except ImportError:
    print("WARNING: faiss chưa được cài đặt. Hệ thống không thể lưu trữ Vector.")
    faiss = None

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None

logger = logging.getLogger(__name__)

class DummyPoint:
    """Giả lập cấu trúc PointStruct của Qdrant để tương thích ngược với LangGraph Agents."""
    def __init__(self, id, payload, score=None):
        self.id = id
        self.payload = payload
        self.score = score

class MultimodalEmbedder:
    """
    Bộ nhúng đa phương thức SEN: SigLIP (Text-to-Image) + DINOv3 (Image-to-Image) + E5-large (Text).
    Sử dụng kiến trúc Lưu trữ Kép FAISS.
    """
    def __init__(self, index_dir=None):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        # 1. Vision model (SigLIP) - Phụ trách Ngữ nghĩa
        self.vision_model_name = "google/siglip-so400m-patch14-384"
        self.vision_processor = AutoProcessor.from_pretrained(self.vision_model_name)
        self.vision_model = AutoModel.from_pretrained(self.vision_model_name).to(self.device)
        self.vision_model.eval()
        
        # 2. Vision model (DINOv2) - Phụ trách Hình học/Thị giác
        self.dino_model_name = "facebook/dinov2-large"
        logger.info(f"Đang nạp mô hình DINO ({self.dino_model_name}) vào bộ nhớ...")
        self.dino_processor = AutoProcessor.from_pretrained(self.dino_model_name)
        self.dino_model = AutoModel.from_pretrained(self.dino_model_name).to(self.device)
        self.dino_model.eval()
        
        # 3. Text model (Multilingual E5) - Phụ trách Văn bản
        if TextEmbedding:
            self.text_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")
        else:
            self.text_model = None
            logger.warning("fastembed chưa được cài đặt. Text search thuần túy sẽ không khả dụng.")
            
        # FAISS Triple Indices
        self.siglip_index = faiss.IndexFlatIP(1152) if faiss else None
        self.dino_index = faiss.IndexFlatIP(1024) if faiss else None
        self.text_index = faiss.IndexFlatIP(1024) if faiss else None
        self.metadata = []  # Từ điển map chung: FAISS ID -> Payload Dict
        
        # Raw Vectors Storage for .npy exporting
        self.raw_siglip_vectors = []
        self.raw_dino_vectors = []
        self.raw_text_vectors = []
        
        if index_dir and os.path.exists(index_dir):
            self.load_index(index_dir)

    def _get_crops(self, image: Image.Image):
        return [image]

    def embed_image_siglip(self, image_paths: list):
        """Trích xuất Vector 1152 chiều bằng SigLIP."""
        all_crops = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            all_crops.extend(self._get_crops(img))
            
        img_vectors = []
        batch_size = 12
        for i in range(0, len(all_crops), batch_size):
            batch = all_crops[i:i+batch_size]
            inputs = self.vision_processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.vision_model.get_image_features(**inputs)
                img_embeds = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
                img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
            img_vectors.extend(img_embeds.cpu().tolist())
        return img_vectors

    def embed_image_dino(self, image_paths: list):
        """Trích xuất Vector 1024 chiều bằng DINOv2."""
        all_crops = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            all_crops.extend(self._get_crops(img))
            
        img_vectors = []
        batch_size = 12
        for i in range(0, len(all_crops), batch_size):
            batch = all_crops[i:i+batch_size]
            inputs = self.dino_processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
                # DINOv2 cls_token
                img_embeds = outputs.last_hidden_state[:, 0, :]
                img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
            img_vectors.extend(img_embeds.cpu().tolist())
        return img_vectors

    def upsert_batch(self, frames_data: list):
        """Nạp dữ liệu vào cả 2 FAISS Indices và lưu metadata vào Python Dict."""
        if not frames_data or not faiss: return
        
        image_paths = [f["path"] for f in frames_data]
        
        # Encode cả 2 mô hình ảnh
        siglip_vectors = self.embed_image_siglip(image_paths)
        dino_vectors = self.embed_image_dino(image_paths)
        
        # Thêm vector vào 2 FAISS Indices ảnh
        self.siglip_index.add(np.array(siglip_vectors, dtype=np.float32))
        self.dino_index.add(np.array(dino_vectors, dtype=np.float32))
        
        # Thêm vector thô vào mảng phụ dự phòng để xuất file .npy
        self.raw_siglip_vectors.extend(siglip_vectors)
        self.raw_dino_vectors.extend(dino_vectors)
        
        # Xử lý Text Embedding cho nhánh E5 (Nếu có text)
        if self.text_model and self.text_index:
            text_contents = []
            for f in frames_data:
                txt = (f.get("caption", "") + " " + f.get("audio_transcript", "") + " " + f.get("narrative_context", "")).strip()
                if not txt: txt = "empty_frame"
                text_contents.append(txt)
            
            # Embed text using fastembed generator
            text_vectors = list(self.text_model.embed(text_contents))
            self.text_index.add(np.array(text_vectors, dtype=np.float32))
            self.raw_text_vectors.extend(text_vectors)
        
        # Thêm metadata (Dùng chung cho cả 2 mảng vì thứ tự nạp là giống hệt nhau)
        for i, frame in enumerate(frames_data):
            ts = frame["timestamp_sec"]
            minutes = int(ts // 60)
            seconds = int(ts % 60)
            payload = {
                "video_id": frame["video_id"],
                "timestamp_sec": frame["timestamp_sec"],
                "timestamp_ms": int(frame["timestamp_sec"] * 1000),
                "formatted_time": f"{minutes:02d}:{seconds:02d}",
                "frame_path": frame["path"],
                "caption": frame.get("caption", ""),
                "narrative_context": frame.get("narrative_context", ""),
                "audio_transcript": frame.get("audio_transcript", ""),
                "type": "frame"
            }
            self.metadata.append(payload)
            
        logger.info(f"FAISS: Đã nạp {len(frames_data)} vectors kép. Tổng số: {self.siglip_index.ntotal}")

    def search_image_by_text(self, text_query: str, limit=5):
        """Dùng SigLIP để tìm bằng TEXT (Semantic Search)"""
        if not faiss or self.siglip_index.ntotal == 0: return []
        
        inputs = self.vision_processor(text=[text_query], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vision_model.get_text_features(**inputs)
            text_embed = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
            text_embed = text_embed / text_embed.norm(p=2, dim=-1, keepdim=True)
            
        query_vector = text_embed.cpu().numpy().astype(np.float32)
        scores, indices = self.siglip_index.search(query_vector, limit)
        
        results = []
        for i in range(len(indices[0])):
            idx, score = indices[0][i], scores[0][i]
            if idx != -1 and idx < len(self.metadata):
                results.append(DummyPoint(id=int(idx), payload=self.metadata[idx], score=float(score)))
        return results

    def search_image_by_image(self, image_path: str, limit=5):
        """Dùng DINOv2 để tìm bằng HÌNH ẢNH (Visual/Geometry Search)"""
        if not faiss or self.dino_index.ntotal == 0: return []
        
        query_vector_list = self.embed_image_dino([image_path])
        query_vector = np.array(query_vector_list, dtype=np.float32)
        
        scores, indices = self.dino_index.search(query_vector, limit)
        
        results = []
        for i in range(len(indices[0])):
            idx, score = indices[0][i], scores[0][i]
            if idx != -1 and idx < len(self.metadata):
                results.append(DummyPoint(id=int(idx), payload=self.metadata[idx], score=float(score)))
        return results

    def search_text_query(self, text_query: str, limit=5):
        """Dùng E5 để tìm kiếm văn bản thuần túy (Audio/Caption Search)"""
        if not faiss or not self.text_index or self.text_index.ntotal == 0 or not self.text_model:
            return []
            
        query_vector = list(self.text_model.embed([text_query]))[0]
        query_vector = np.array([query_vector], dtype=np.float32)
        
        scores, indices = self.text_index.search(query_vector, limit)
        
        results = []
        for i in range(len(indices[0])):
            idx, score = indices[0][i], scores[0][i]
            if idx != -1 and idx < len(self.metadata):
                results.append(DummyPoint(id=int(idx), payload=self.metadata[idx], score=float(score)))
        return results

    def save_index(self, save_dir: str):
        """Lưu toàn bộ mảng dữ liệu thô (.npy) và metadata ra ổ đĩa. Khước từ định dạng .bin."""
        if not faiss: return
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
            
        # Xuất file NPY
        npy_dir = os.path.join(os.path.dirname(save_dir), 'npy')
        os.makedirs(npy_dir, exist_ok=True)
        if self.raw_siglip_vectors:
            np.save(os.path.join(npy_dir, 'vectors_siglip.npy'), np.array(self.raw_siglip_vectors, dtype=np.float32))
        if self.raw_dino_vectors:
            np.save(os.path.join(npy_dir, 'vectors_dino.npy'), np.array(self.raw_dino_vectors, dtype=np.float32))
        if self.raw_text_vectors:
            np.save(os.path.join(npy_dir, 'vectors_text.npy'), np.array(self.raw_text_vectors, dtype=np.float32))
            
        logger.info(f"Đã lưu Database ({len(self.raw_siglip_vectors)} vectors) vào {npy_dir}")

    def load_index(self, load_dir: str):
        """Tải 3 mạng FAISS hoàn toàn in-memory từ các file NPY."""
        if not faiss: return
        npy_dir = os.path.join(os.path.dirname(load_dir), 'npy')
        siglip_npy = os.path.join(npy_dir, "vectors_siglip.npy")
        dino_npy = os.path.join(npy_dir, "vectors_dino.npy")
        text_npy = os.path.join(npy_dir, "vectors_text.npy")
        meta_path = os.path.join(load_dir, "metadata.json")
        
        if os.path.exists(siglip_npy) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                
            # Đúc mảng SigLIP on-the-fly
            siglip_arr = np.load(siglip_npy)
            self.siglip_index = faiss.IndexFlatIP(1152)
            self.siglip_index.add(siglip_arr)
            self.raw_siglip_vectors = siglip_arr.tolist()
            
            # Đúc mảng DINO on-the-fly
            if os.path.exists(dino_npy):
                dino_arr = np.load(dino_npy)
                self.dino_index = faiss.IndexFlatIP(1024)
                self.dino_index.add(dino_arr)
                self.raw_dino_vectors = dino_arr.tolist()
            else:
                self.dino_index = faiss.IndexFlatIP(1024)
                
            # Đúc mảng Text on-the-fly
            if os.path.exists(text_npy):
                text_arr = np.load(text_npy)
                self.text_index = faiss.IndexFlatIP(1024)
                self.text_index.add(text_arr)
                self.raw_text_vectors = text_arr.tolist()
            else:
                self.text_index = faiss.IndexFlatIP(1024)
                
            logger.info(f"Đã tải In-Memory FAISS (SigLIP: {self.siglip_index.ntotal}, DINO: {self.dino_index.ntotal}, Text: {self.text_index.ntotal}) từ NPY.")
        else:
            logger.warning(f"Không tìm thấy DB tại {npy_dir}")

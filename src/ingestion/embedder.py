import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None


class MultimodalEmbedder:
    """
    Bộ nhúng đa phương thức: SigLIP (Image) + E5-large (Text).
    Tối ưu cho Apple M4 16GB RAM.
    """
    def __init__(self, qdrant_url="http://localhost:6335", collection_name="multimodal_db"):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        
        # Vision model (SigLIP)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.vision_model_name = "google/siglip-so400m-patch14-384"
        self.vision_processor = AutoProcessor.from_pretrained(self.vision_model_name)
        self.vision_model = AutoModel.from_pretrained(self.vision_model_name).to(self.device)
        self.vision_model.eval()
        
        # Text model (Multilingual E5 for Text embedding)
        if TextEmbedding:
            self.text_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")
        else:
            self.text_model = None
            print("WARNING: fastembed chưa được cài đặt. Text search sẽ không khả dụng.")
            
        self._init_collection()
        
    def _init_collection(self):
        if not self.client.collection_exists(self.collection_name):
            from qdrant_client.models import MultiVectorConfig, MultiVectorComparator, ScalarQuantizationConfig, ScalarType
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text": VectorParams(size=1024, distance=Distance.COSINE),
                    "image": VectorParams(
                        size=1152, 
                        distance=Distance.COSINE,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM
                        )
                    ),
                },
                quantization_config=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=True
                )
            )
            
    def embed_image(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        crops = self._get_crops(image)
        inputs = self.vision_processor(images=crops, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vision_model.get_image_features(**inputs)
            pooler_output = outputs.pooler_output
            img_embeds = pooler_output / pooler_output.norm(p=2, dim=-1, keepdim=True)
        return img_embeds.cpu().tolist()
        
    def _get_crops(self, image: Image.Image):
        """Tạo 5 mảnh crop từ ảnh gốc để mô phỏng Patch-level Late Interaction (Video-ColBERT)."""
        w, h = image.size
        cw, ch = w // 2, h // 2
        crops = [
            image, # Toàn cảnh (Global)
            image.crop((0, 0, cw, ch)), # Góc trên trái
            image.crop((cw, 0, w, ch)), # Góc trên phải
            image.crop((0, ch, cw, h)), # Góc dưới trái
            image.crop((cw, ch, w, h)), # Góc dưới phải
            image.crop((cw//2, ch//2, w - cw//2, h - ch//2)) # Trung tâm
        ]
        return crops

    def embed_image_batch(self, image_paths: list):
        """
        Trích xuất Multi-Vector cho mỗi hình ảnh. Mỗi hình ảnh tạo ra 6 vector (Global + 5 Crops).
        Qdrant sẽ dùng hàm MAX_SIM để chọn vector phù hợp nhất với câu hỏi.
        """
        all_crops = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            all_crops.extend(self._get_crops(img))
            
        img_vectors = []
        batch_size = 12  # Xử lý 12 crops / batch để tránh lỗi OOM
        for i in range(0, len(all_crops), batch_size):
            batch = all_crops[i:i+batch_size]
            inputs = self.vision_processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.vision_model.get_image_features(**inputs)
                
                # SigLIP get_image_features trả về trực tiếp Tensor, không có pooler_output (nếu dùng phương thức get_image_features)
                # Hoặc tuỳ version Transformers, nếu nó trả về đối tượng có pooler_output:
                if hasattr(outputs, "pooler_output"):
                    img_embeds = outputs.pooler_output
                elif hasattr(outputs, "image_embeds"):
                    img_embeds = outputs.image_embeds
                else:
                    img_embeds = outputs # Nếu trả về tensor trực tiếp
                    
                img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
            img_vectors.extend(img_embeds.cpu().tolist())
        
        # Nhóm lại mỗi 6 vector thành 1 MultiVector tương ứng với 1 hình ảnh gốc
        multi_vectors = []
        for i in range(0, len(img_vectors), 6):
            multi_vectors.append(img_vectors[i:i+6])
            
        return multi_vectors
        
    def upsert_batch(self, frames_data: list):
        """
        frames_data = [{"path": str, "video_id": str, "timestamp_sec": float, ...}, ...]
        """
        if not frames_data: return
        
        image_paths = [f["path"] for f in frames_data]
        img_vectors = self.embed_image_batch(image_paths)
        
        text_vectors = None
        if self.text_model:
            contexts = [f.get("narrative_context", "") for f in frames_data]
            text_vectors = list(self.text_model.embed(contexts))
        
        points = []
        for i, frame in enumerate(frames_data):
            vectors_dict = {"image": img_vectors[i]}
            if text_vectors is not None:
                vectors_dict["text"] = text_vectors[i].tolist()
                
            # Tính timestamp dạng đọc được
            ts = frame["timestamp_sec"]
            minutes = int(ts // 60)
            seconds = int(ts % 60)
            formatted_time = f"{minutes:02d}:{seconds:02d}"
                
            point = PointStruct(
                id=hash(frame["path"]) % (10 ** 8),
                vector=vectors_dict,
                payload={
                    "video_id": frame["video_id"],
                    "timestamp_sec": frame["timestamp_sec"],
                    "timestamp_ms": int(frame["timestamp_sec"] * 1000),
                    "formatted_time": formatted_time,
                    "frame_path": frame["path"],
                    "caption": frame.get("caption", ""),
                    "narrative_context": frame.get("narrative_context", ""),
                    "audio_transcript": frame.get("audio_transcript", ""),
                    "type": "frame"
                }
            )
            points.append(point)
            
        self.client.upsert(self.collection_name, points)
        
    def search_text_query(self, query: str, limit=5):
        if not self.text_model:
            print("ERROR: Không tìm thấy mô hình fastembed TextEmbedding.")
            return []
            
        query_vector_gen = self.text_model.embed([query])
        query_vector = list(query_vector_gen)[0].tolist()
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="text",
            limit=limit
        )
        return results.points

    def search_image_query(self, image_path: str, limit=5):
        query_vector = self.embed_image(image_path)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="image",
            limit=limit
        )
        return results.points

    def search_image_by_text(self, text_query: str, limit=5):
        """Dùng Text Encoder của SigLIP để tìm kiếm trong không gian Image (Cross-modal Video-ColBERT)."""
        inputs = self.vision_processor(text=[text_query], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vision_model.get_text_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                text_embed = outputs.pooler_output
            elif hasattr(outputs, "text_embeds"):
                text_embed = outputs.text_embeds
            else:
                text_embed = outputs
            text_embed = text_embed / text_embed.norm(p=2, dim=-1, keepdim=True)
            
        query_vector = text_embed.squeeze(0).cpu().tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="image",
            limit=limit
        )
        return results.points


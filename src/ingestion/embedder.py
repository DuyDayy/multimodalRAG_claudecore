import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image

# For Image embeddings in a real system, you would use a local SigLIP model via HuggingFace transformers
# Here we use a placeholder representation or fastembed if it supports vision
# For simplicity on M4, we can use the Qdrant client and Fastembed for Text.

class MultimodalEmbedder:
    def __init__(self, qdrant_url="http://localhost:6335", collection_name="multimodal_db"):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        
        # Vision model (SigLIP)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.vision_model_name = "google/siglip-so400m-patch14-384"
        self.vision_processor = AutoProcessor.from_pretrained(self.vision_model_name)
        self.vision_model = AutoModel.from_pretrained(self.vision_model_name).to(self.device)
        self.vision_model.eval()
        
        self._init_collection()
        
    def _init_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text": VectorParams(size=1024, distance=Distance.COSINE), # Size matches bge-m3
                    "image": VectorParams(size=1152, distance=Distance.COSINE) # Size for siglip-so400m-patch14-384
                }
            )
            
    def embed_image(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        inputs = self.vision_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vision_model.get_image_features(**inputs)
            # Normalize vector
            pooler_output = outputs.pooler_output
            img_embed = pooler_output / pooler_output.norm(p=2, dim=-1, keepdim=True)
        return img_embed.squeeze(0).cpu().tolist()
        
    def embed_image_batch(self, image_paths: list):
        images = [Image.open(p).convert('RGB') for p in image_paths]
        inputs = self.vision_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vision_model.get_image_features(**inputs)
            pooler_output = outputs.pooler_output
            img_embeds = pooler_output / pooler_output.norm(p=2, dim=-1, keepdim=True)
        return img_embeds.cpu().tolist()
        
    def upsert_batch(self, frames_data: list):
        """
        frames_data = [{"path": str, "video_id": str, "timestamp_sec": float, "caption": str}, ...]
        """
        if not frames_data: return
        
        image_paths = [f["path"] for f in frames_data]
        img_vectors = self.embed_image_batch(image_paths)
        
        points = []
        for i, frame in enumerate(frames_data):
            # Lưu caption vào payload để sau này dùng hoặc filter
            point = PointStruct(
                id=hash(frame["path"]) % (10 ** 8),
                vector={"image": img_vectors[i]},
                payload={
                    "video_id": frame["video_id"],
                    "timestamp": frame["timestamp_sec"],
                    "frame_path": frame["path"],
                    "caption": frame.get("caption", ""),
                    "type": "frame"
                }
            )
            points.append(point)
            
        self.client.upsert(self.collection_name, points)
        
    def embed_text_siglip(self, text: str):
        # Sử dụng SigLIP để mã hóa văn bản vào không gian vector 768-chiều (cùng không gian với ảnh)
        inputs = self.vision_processor(text=[text], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vision_model.get_text_features(**inputs)
            pooler_output = outputs.pooler_output
            text_embed = pooler_output / pooler_output.norm(p=2, dim=-1, keepdim=True)
        return text_embed.squeeze(0).cpu().tolist()

    def search_text_query(self, query: str, limit=5):
        # Dùng cross-modal retrieval: embed text query bằng SigLIP và so sánh với vector "image"
        query_vector = self.embed_text_siglip(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="image",  # Phải tìm trong trường "image" vì ta đang map Text -> Image
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

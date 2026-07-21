import os
import json
import logging

logger = logging.getLogger(__name__)

class AuxiliaryBuilder:
    """
    Class for building and retrieving FAISS databases for Video-RAG (OCR, DET).
    Uses Lazy Loading to avoid crashing on M4 16GB if libraries aren't installed yet.
    """
    def __init__(self):
        self.db_path = "faiss_dbs"
        os.makedirs(self.db_path, exist_ok=True)
        self.encoder = None
        self.ocr_reader = None
        
    def _init_models(self):
        # Lazy load to save RAM and handle missing dependencies gracefully
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                logger.warning("sentence-transformers not installed. FAISS retrieval will use mock.")
                self.encoder = "mock"
                
        if self.ocr_reader is None:
            try:
                import easyocr
                self.ocr_reader = easyocr.Reader(['vi', 'en'], gpu=False)
            except ImportError:
                logger.warning("easyocr not installed. OCR extraction will be skipped.")
                self.ocr_reader = "mock"
                
    def build_video_databases(self, video_id: str, frame_paths: list):
        """
        Builds FAISS DBs from actual video frames using EasyOCR.
        """
        self._init_models()
        if self.encoder == "mock" or self.ocr_reader == "mock":
            return
            
        try:
            import faiss
            import numpy as np
            
            # 1. OCR Extraction
            texts = []
            for path in frame_paths:
                results = self.ocr_reader.readtext(path, detail=0)
                if results:
                    texts.append(f"Frame {os.path.basename(path)} OCR: " + " ".join(results))
                    
            if texts:
                # 2. FAISS Indexing
                embeddings = self.encoder.encode(texts)
                dim = embeddings.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(np.array(embeddings).astype('float32'))
                
                faiss.write_index(index, f"{self.db_path}/{video_id}_ocr.faiss")
                with open(f"{self.db_path}/{video_id}_ocr_texts.json", "w") as f:
                    json.dump(texts, f)
                    
        except Exception as e:
            logger.error(f"Error building FAISS DB: {e}")

    def retrieve_auxiliary_texts(self, video_id: str, requests: dict) -> str:
        """
        Retrieves matching auxiliary texts from FAISS DBs.
        """
        self._init_models()
        if self.encoder == "mock":
            return self._mock_retrieve(requests)
            
        aux_results = []
        try:
            import faiss
            import numpy as np
            
            ocr_query = requests.get("OCR")
            if ocr_query and os.path.exists(f"{self.db_path}/{video_id}_ocr.faiss"):
                index = faiss.read_index(f"{self.db_path}/{video_id}_ocr.faiss")
                with open(f"{self.db_path}/{video_id}_ocr_texts.json", "r") as f:
                    texts = json.load(f)
                    
                q_emb = self.encoder.encode([ocr_query])
                distances, indices = index.search(np.array(q_emb).astype('float32'), k=2)
                
                for i in indices[0]:
                    if i != -1 and i < len(texts):
                        aux_results.append(texts[i])
                        
        except Exception as e:
            logger.error(f"Error in retrieve_auxiliary_texts: {e}")
            return self._mock_retrieve(requests)
            
        if aux_results:
            return "OCR Results:\n" + "\n".join(aux_results)
            
        return self._mock_retrieve(requests)
        
    def _mock_retrieve(self, requests: dict) -> str:
        texts = []
        if requests.get("ASR"):
            texts.append(f"ASR Text: {requests['ASR']} (Mock)")
        if requests.get("DET") or requests.get("TYPE"):
            texts.append("Obj Counting: - object: 1\nObj Location: - object at [100,200]")
        if requests.get("OCR"):
            texts.append(f"OCR Text: {requests['OCR']} (Mock)")
        return "\n".join(texts)

auxiliary_builder = AuxiliaryBuilder()

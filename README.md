# Multimodal RAG System (AI Challenge) - Claude Core Architecture

A state-of-the-art Multimodal Retrieval-Augmented Generation (RAG) system engineered for high-volume video processing. This version has been fully upgraded to the **Claude Core architecture**, integrating intelligent scene detection and batch processing to maximize both speed and accuracy (~100%).

*For the Vietnamese version of this document, please see [README_vi.md](README_vi.md).*

## 🌟 Features & Query Types
The system supports 4 distinct query types through a Streamlit frontend:
1. **Video KIS (Known-Item Search)**: Find an original source video based on a short query clip or keyframe image.
2. **Textual KIS**: Retrieve specific video segments based on dense textual descriptions (achieves absolute accuracy thanks to LLM Captioning).
3. **Q&A Query**: Extract answers to specific questions using video context (powered by **Claude 3.5 Sonnet**).
4. **TRAKE (Temporal Retrieval & Alignment)**: Identify and align sequential events chronologically.

## 📁 Repository Structure
```text
.
├── app.py                      # Main Streamlit UI application
├── docker-compose.yml          # Infrastructure setup (Qdrant Vector DB, Redis)
├── requirements.txt            # Python dependencies
├── src/
│   ├── agents/                 # LangGraph Multi-Agent System
│   │   ├── graph.py            # Workflow definition & edge mapping
│   │   ├── router_agent.py     # Classifies queries using Claude 3.5 Sonnet
│   │   ├── retriever_agent.py  # Hybrid search via Qdrant
│   │   └── generator_agent.py  # Response generation via Claude 3.5 Sonnet
│   └── ingestion/              # Data Processing Pipeline (High Speed & Accuracy)
│       ├── video_processor.py  # Smart Keyframe extraction via PySceneDetect
│       ├── offline_encoder.py  # Claude Captioning & Batch processing
│       └── embedder.py         # Qdrant configuration & SigLIP batch inference
└── test_data_samples/          # Directory for local sample videos
```

## 🧠 Core Architecture: "Smart Keyframes & LLM Captioning"
This system resolves the two classic challenges of Video RAG (Speed and Logical Accuracy) through the following technologies:
- **Speed (PySceneDetect & Batching)**: Moving away from the blind 1-2 FPS extraction method, the system utilizes `AdaptiveDetector` to extract only the frames where scene transitions occur (reducing data noise by 90%). These Keyframes are then batched (Batch size = 16) for parallel processing, optimizing GPU throughput.
- **Absolute Accuracy (Claude 3.5 Sonnet)**: Rather than relying solely on SigLIP's shallow visual understanding, the system pipes the extracted Keyframes directly to **Claude 3.5 Sonnet** (via the `claude-sonnet-4-6` proxy) during the data ingestion phase. Claude generates a highly detailed caption describing every object, action, and logical flow in the image. This metadata is stored directly in Qdrant, transforming it into a true Semantic encyclopedia.

## 🛠 Prerequisites
- **Python 3.10+**
- **Docker Desktop** (for Vector DB)
- Proxy API Key supporting `claude-sonnet-4-6` configured in `.env`

## 🚀 Installation

1. **Set up Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Start Qdrant & Redis**:
   ```bash
   docker-compose up -d
   ```
3. **Configure API**: Create a `.env` file containing `OPENAI_API_KEY` and `OPENAI_BASE_URL` from your Claude Proxy provider.

## 🎮 Running the System

1. **Run Ingestion Pipeline (Offline Encoder)**:
   ```bash
   ./run_encoder.sh
   ```
   *This process scans videos, detects scenes, calls Claude for captions, and pushes batches to Qdrant.*

2. **Run Streamlit UI**:
   ```bash
   ./run_app.sh
   ```
   Open your browser at `http://localhost:8501` to experience the lightning-fast 4-Tab interface.

## 🏗 Core Frameworks Explanation
- **LangGraph**: The brain orchestrating the query workflow.
- **Claude 3.5 Sonnet**: The Core intelligence used in 3 stages: Captioning (Ingestion), Routing (Query Classification), and Generation (Q&A).
- **Qdrant**: The Vector Database storing SigLIP Vectors alongside Claude's Payload Metadata (Captions).
- **PySceneDetect**: The optical video processing algorithm ensuring no transition moments are missed.

"""
app.py — Voice-Enabled RAG system for Indic Languages

Serves REST API endpoints: GET /health, POST /query/text, POST /query/voice
Designed to run in a plain Docker container using CPU hardware.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Configure HuggingFace and PyTorch cache directories to /tmp for Hugging Face Spaces compatibility
if os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE") or os.environ.get("IS_DOCKER"):
    os.environ.setdefault("HF_HOME", "/tmp/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/cache/transformers")
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/cache/sentence_transformers")
    os.environ.setdefault("TORCH_HOME", "/tmp/cache/torch")

# Load .env file for API keys
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieve import HybridRetriever
from src.pipeline import RAGPipeline, PipelineRequest, PipelineStatus
from src.embed import get_model  # Pre-load embedding model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_pipeline: Optional[RAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and indices once at startup, release on shutdown."""
    global _pipeline

    logger.info("Loading embedding model...")
    get_model()  # Trigger lazy load of sentence-transformers model

    logger.info("Loading retrieval indices...")
    data_dir = os.environ.get("INDEX_DIR", "data")

    # In HuggingFace Spaces, ensure the Qdrant persistence directory is writable
    if (os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE")) and not os.path.exists("/tmp/data"):
        import shutil
        if os.path.exists(data_dir):
            logger.info(f"Copying prebuilt index from {data_dir} to /tmp/data for Hugging Face Spaces...")
            shutil.copytree(data_dir, "/tmp/data", ignore=shutil.ignore_patterns("parquet_cache*"))
            data_dir = "/tmp/data"

    retriever = HybridRetriever(index_dir=data_dir)
    collection_info = retriever.qdrant.get_collection(retriever.collection_name)
    logger.info(
        f"Loaded Qdrant collection '{retriever.collection_name}' with "
        f"{collection_info.points_count} vectors"
    )

    _pipeline = RAGPipeline(retriever=retriever)
    logger.info("Pipeline ready!")

    yield

    _pipeline = None
    logger.info("Pipeline shut down.")


app = FastAPI(
    title="Voice-Enabled RAG System",
    description=(
        "Retrieval-Augmented Generation for Indic languages using "
        "MSMARCO-XI dataset, Sarvam STT, and Groq LLaMA generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

class TextQueryRequest(BaseModel):
    query: str
    language: str = "hi-IN"

class QueryResponse(BaseModel):
    status: str
    answer: str = ""
    query_text: str = ""
    retrieved_chunks: list = []
    stage_timings: dict = {}
    error_message: str = ""

def _status_to_http(status: PipelineStatus) -> int:
    if status in (
        PipelineStatus.SUCCESS,
        PipelineStatus.REFUSED_BAD_INPUT,
        PipelineStatus.REFUSED_NO_MATCH,
        PipelineStatus.REFUSED_BY_MODEL,
        PipelineStatus.REFUSED_UNGROUNDED,
    ):
        return 200
    elif status == PipelineStatus.STT_ERROR:
        return 502
    else:
        return 500

@app.get("/health")
async def health_check():
    if _pipeline is None:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    collection_info = _pipeline.retriever.qdrant.get_collection(
        _pipeline.retriever.collection_name
    )
    return {
        "status": "healthy",
        "vector_count": collection_info.points_count,
        "collection_name": _pipeline.retriever.collection_name,
    }

@app.post("/query/text", response_model=QueryResponse)
async def query_text(request: TextQueryRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipeline_request = PipelineRequest(query_text=request.query, language=request.language)
    response = _pipeline.run(pipeline_request)
    return JSONResponse(
        status_code=_status_to_http(response.status),
        content={
            "status": response.status.value,
            "answer": response.answer,
            "query_text": response.query_text,
            "retrieved_chunks": response.retrieved_chunks,
            "stage_timings": {k: round(v * 1000, 2) for k, v in response.stage_timings.items()},
            "error_message": response.error_message,
        },
    )

@app.post("/query/voice", response_model=QueryResponse)
async def query_voice(
    audio: UploadFile = File(...),
    language: str = Form("hi-IN"),
):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    pipeline_request = PipelineRequest(audio_bytes=audio_bytes, language=language)
    response = _pipeline.run(pipeline_request)
    return JSONResponse(
        status_code=_status_to_http(response.status),
        content={
            "status": response.status.value,
            "answer": response.answer,
            "query_text": response.query_text,
            "retrieved_chunks": response.retrieved_chunks,
            "stage_timings": {k: round(v * 1000, 2) for k, v in response.stage_timings.items()},
            "error_message": response.error_message,
        },
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

import os
os.makedirs('static', exist_ok=True)
app.mount('/static', StaticFiles(directory='static'), name='static')

@app.get('/')
def serve_frontend():
    return FileResponse('static/index.html')

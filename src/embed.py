"""
embed.py — Multilingual embedding using intfloat/multilingual-e5-small

Uses ONNX Runtime with INT8 quantization and the pure Rust `tokenizers` library 
to aggressively minimize memory usage, fitting comfortably under 512MB RAM constraints.

E5 models require specific prefixes:
  - "query: " for search queries
  - "passage: " for document passages
"""

import os
import numpy as np
from typing import List, Optional
from tokenizers import Tokenizer
import onnxruntime as ort

_tokenizer: Optional[Tokenizer] = None
_ort_session: Optional[ort.InferenceSession] = None
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "onnx_quantized")

def get_model():
    """Load ONNX inference session and pure Rust tokenizer.
    Lazy-loaded to avoid paying the model load cost at import time.
    """
    global _tokenizer, _ort_session
    if _ort_session is None:
        if not os.path.exists(_MODEL_DIR):
            raise RuntimeError(f"Quantized ONNX model not found at {_MODEL_DIR}.")
            
        # 1. Load Tokenizer (pure rust, saves ~400MB RAM compared to transformers AutoTokenizer)
        _tokenizer = Tokenizer.from_file(os.path.join(_MODEL_DIR, "tokenizer.json"))
        # E5 (XLM-R based) uses pad_id=1, but tokenizer.json handles padding automatically if enabled
        # Actually XLM-Roberta pad_token_id is 1. Let's explicitly set it.
        _tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        _tokenizer.enable_truncation(max_length=512)
        
        # 2. Load ONNX model with heavily optimized memory settings
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        
        # These two lines drastically reduce ONNX memory allocations
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        # Optional: Disable optimizations that cache additional tensors
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        _ort_session = ort.InferenceSession(
            os.path.join(_MODEL_DIR, "model_quantized.onnx"), 
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
    return _tokenizer, _ort_session


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding vectors (384 for e5-small)."""
    return 384


def _mean_pooling_and_normalize(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Perform mean pooling and L2 normalization using pure NumPy."""
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
    sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    pooled_embeddings = sum_embeddings / sum_mask
    norms = np.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
    normalized = pooled_embeddings / np.clip(norms, a_min=1e-9, a_max=None)
    return normalized.astype(np.float32)


def _encode_texts(texts: List[str]) -> np.ndarray:
    """Core encoding routine using ONNX Runtime."""
    tokenizer, ort_session = get_model()
    
    encoded = tokenizer.encode_batch(texts)
    
    input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    input_names = [i.name for i in ort_session.get_inputs()]
    if "token_type_ids" in input_names:
        # e5-small might require token_type_ids. Usually tokenizers provide them.
        token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)
        ort_inputs["token_type_ids"] = token_type_ids

    outputs = ort_session.run(None, ort_inputs)
    last_hidden_state = outputs[0]
    
    return _mean_pooling_and_normalize(last_hidden_state, attention_mask)


def encode_passages(texts: List[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
        
    prefixed = [f"passage: {t}" for t in texts]
    
    all_embeddings = []
    for i in range(0, len(prefixed), batch_size):
        batch_texts = prefixed[i:i + batch_size]
        batch_embeddings = _encode_texts(batch_texts)
        all_embeddings.append(batch_embeddings)
        
    return np.vstack(all_embeddings)


def encode_query(text: str) -> np.ndarray:
    prefixed = f"query: {text}"
    return _encode_texts([prefixed])

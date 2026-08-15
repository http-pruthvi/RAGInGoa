"""
embed.py — Multilingual embedding using intfloat/multilingual-e5-small

Uses ONNX Runtime with INT8 quantization to fit under 512MB RAM constraints.

E5 models require specific prefixes:
  - "query: " for search queries
  - "passage: " for document passages
This asymmetric encoding lets the model distinguish between short questions
and longer passages during contrastive training, improving retrieval quality.

Vectors are L2-normalized so that dot product = cosine similarity.
"""

import os
import numpy as np
from typing import List, Optional
from transformers import AutoTokenizer
import onnxruntime as ort

_tokenizer: Optional[AutoTokenizer] = None
_ort_session: Optional[ort.InferenceSession] = None
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "onnx_quantized")

def get_model():
    """Load ONNX inference session and tokenizer.
    Lazy-loaded to avoid paying the model load cost at import time.
    """
    global _tokenizer, _ort_session
    if _ort_session is None:
        if not os.path.exists(_MODEL_DIR):
            raise RuntimeError(f"Quantized ONNX model not found at {_MODEL_DIR}. Run conversion script first.")
            
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)
        
        # CPUExecutionProvider is sufficient and doesn't require extra large binaries
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        
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
    # Expand attention mask to match hidden state shape
    # last_hidden_state: (batch_size, seq_len, hidden_size)
    # attention_mask: (batch_size, seq_len)
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
    
    # Compute sums and counts
    sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    
    # Mean pool
    pooled_embeddings = sum_embeddings / sum_mask
    
    # L2 Normalize
    norms = np.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
    normalized = pooled_embeddings / np.clip(norms, a_min=1e-9, a_max=None)
    
    return normalized.astype(np.float32)


def _encode_texts(texts: List[str]) -> np.ndarray:
    """Core encoding routine using ONNX Runtime."""
    tokenizer, ort_session = get_model()
    
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='np')
    
    ort_inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }
    
    # Ensure token_type_ids is provided if the model requires it (some ONNX exports strictly require it)
    if "token_type_ids" in encoded:
        ort_inputs["token_type_ids"] = encoded["token_type_ids"]
    else:
        # Fallback to zeros if required by ONNX model input graph
        input_names = [i.name for i in ort_session.get_inputs()]
        if "token_type_ids" in input_names:
            ort_inputs["token_type_ids"] = np.zeros_like(encoded["input_ids"])

    outputs = ort_session.run(None, ort_inputs)
    
    # Output 0 is the last_hidden_state in standard ORT exports
    last_hidden_state = outputs[0]
    
    return _mean_pooling_and_normalize(last_hidden_state, encoded["attention_mask"])


def encode_passages(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Encode document passages into normalized embedding vectors.

    Adds the "passage: " prefix required by E5 models.

    Args:
        texts: List of passage texts to embed.
        batch_size: Batch size for encoding (default 64).

    Returns:
        numpy array of shape (len(texts), embedding_dim), L2-normalized.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
        
    prefixed = [f"passage: {t}" for t in texts]
    
    # Process in batches to avoid OOM or timeout
    all_embeddings = []
    for i in range(0, len(prefixed), batch_size):
        batch_texts = prefixed[i:i + batch_size]
        batch_embeddings = _encode_texts(batch_texts)
        all_embeddings.append(batch_embeddings)
        
    return np.vstack(all_embeddings)


def encode_query(text: str) -> np.ndarray:
    """Encode a single search query into a normalized embedding vector.

    Adds the "query: " prefix required by E5 models.

    Args:
        text: Query text to embed.

    Returns:
        numpy array of shape (1, embedding_dim), L2-normalized.
    """
    prefixed = f"query: {text}"
    return _encode_texts([prefixed])

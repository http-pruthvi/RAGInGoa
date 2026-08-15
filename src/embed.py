"""
embed.py — Multilingual embedding using intfloat/multilingual-e5-small

Uses ONNX Runtime with INT8 quantization and `sentencepiece` to achieve an ultra-low
memory footprint. By natively applying the Fairseq offset (+1) to the standard
sentencepiece model instead of relying on the Hugging Face `tokenizers` rust parser,
we avoid a massive 200MB memory spike from the XLM-R 250,000-token BPE dictionary load.

E5 models require specific prefixes:
  - "query: " for search queries
  - "passage: " for document passages
"""

import os
import numpy as np
from typing import List, Optional
import sentencepiece as spm
import onnxruntime as ort

_sp: Optional[spm.SentencePieceProcessor] = None
_ort_session: Optional[ort.InferenceSession] = None
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "onnx_quantized")

def get_model():
    global _sp, _ort_session
    if _ort_session is None:
        spm_path = os.path.join(_MODEL_DIR, "sentencepiece.bpe.model")
        onnx_path = os.path.join(_MODEL_DIR, "model_quantized.onnx")
        
        if not os.path.exists(spm_path) or not os.path.exists(onnx_path):
            raise RuntimeError(f"Models missing in {_MODEL_DIR}.")
            
        # 1. Load SentencePiece model (native C++, consumes only ~20MB compared to 270MB tokenizers)
        _sp = spm.SentencePieceProcessor(model_file=spm_path)
        
        # 2. Load ONNX model with aggressively optimized memory settings
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        
        _ort_session = ort.InferenceSession(
            onnx_path, 
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        print(f"ONNX Model loaded: {onnx_path} (Size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB)", flush=True)
        
    return _sp, _ort_session


def get_embedding_dimension() -> int:
    return 384


def _mean_pooling_and_normalize(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
    sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    pooled_embeddings = sum_embeddings / sum_mask
    norms = np.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
    normalized = pooled_embeddings / np.clip(norms, a_min=1e-9, a_max=None)
    return normalized.astype(np.float32)


def _encode_texts(texts: List[str]) -> np.ndarray:
    sp, ort_session = get_model()
    
    encoded_list = sp.encode(texts)
    
    # Calculate batch dimensions with truncation at 512
    max_len = min(512, max(len(x) for x in encoded_list) + 2)
    
    # 1 is the pad_token_id for XLM-R
    input_ids = np.ones((len(texts), max_len), dtype=np.int64)
    attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
    
    for i, sp_ids in enumerate(encoded_list):
        # Truncate raw sentencepiece IDs to 510 to leave room for <s> and </s>
        sp_ids = sp_ids[:510]
        seq_len = len(sp_ids) + 2
        
        # Fairseq offset logic: sp_id + 1. Then prepend <s> (0) and append </s> (2)
        mapped = [0] + [x + 1 for x in sp_ids] + [2]
        
        input_ids[i, :seq_len] = mapped
        attention_mask[i, :seq_len] = 1
    
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    input_names = [i.name for i in ort_session.get_inputs()]
    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = np.zeros_like(input_ids)

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

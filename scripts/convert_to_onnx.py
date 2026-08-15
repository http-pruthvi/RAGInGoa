import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

model_id = "intfloat/multilingual-e5-small"
onnx_dir = "models/onnx"
quantized_dir = "models/onnx_quantized"

os.makedirs(onnx_dir, exist_ok=True)
os.makedirs(quantized_dir, exist_ok=True)

print(f"Exporting {model_id} to ONNX...")
# Load model and export to ONNX
# ORTModelForFeatureExtraction will trace it
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.save_pretrained(onnx_dir)
tokenizer.save_pretrained(onnx_dir)

print("Applying dynamic INT8 quantization...")
# Quantize to INT8
quantizer = ORTQuantizer.from_pretrained(model)
dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
quantizer.quantize(save_dir=quantized_dir, quantization_config=dqconfig)
tokenizer.save_pretrained(quantized_dir)
print(f"Quantized model saved to {quantized_dir}")

# Verification
print("\n--- Verifying Quality ---")
test_sentences = [
    "query: What is the capital of France?",
    "passage: The capital of France is Paris.",
    "query: मैनहट्टन परियोजना क्या थी?",
    "passage: मैनहट्टन परियोजना द्वितीय विश्व युद्ध के दौरान एक शोध और विकास उपक्रम था।"
]

# Original PyTorch
print("Loading original PyTorch model...")
st_model = SentenceTransformer(model_id)
pt_embeddings = st_model.encode(test_sentences, convert_to_tensor=True)

# ONNX Quantized
print("Loading quantized ONNX model...")
import onnxruntime as ort
ort_sess = ort.InferenceSession(os.path.join(quantized_dir, "model_quantized.onnx"), providers=["CPUExecutionProvider"])
# get inputs
input_names = [i.name for i in ort_sess.get_inputs()]

def get_ort_embeddings(texts):
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='np')
    ort_inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }
    
    if "token_type_ids" in input_names:
        if "token_type_ids" in encoded:
            ort_inputs["token_type_ids"] = encoded["token_type_ids"]
        else:
            ort_inputs["token_type_ids"] = np.zeros_like(encoded["input_ids"], dtype=np.int64)
            
    outputs = ort_sess.run(None, ort_inputs)
    # E5 uses mean pooling. Output[0] is last_hidden_state
    last_hidden_state = torch.tensor(outputs[0])
    attention_mask = torch.tensor(encoded["attention_mask"])
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    return F.normalize(embeddings, p=2, dim=1)

ort_embeddings = get_ort_embeddings(test_sentences)

# Compare
cos_sim = F.cosine_similarity(pt_embeddings, ort_embeddings)
for i, sentence in enumerate(test_sentences):
    print(f"Similarity for '{sentence[:30]}...': {cos_sim[i].item():.4f}")

avg_sim = cos_sim.mean().item()
print(f"Average Cosine Similarity: {avg_sim:.4f}")

# Check sizes
onnx_size = os.path.getsize(os.path.join(quantized_dir, 'model_quantized.onnx'))
print(f"\nOriginal Size (approx): 471 MB")
print(f"Quantized Size: {onnx_size / 1024 / 1024:.2f} MB")

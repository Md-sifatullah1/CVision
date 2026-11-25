
import os
import resource

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

model_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="onnx/model_qint8_arm64.onnx",
)

tokenizer_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="tokenizer.json",
)

tokenizer = Tokenizer.from_file(tokenizer_path)
tokenizer.enable_truncation(max_length=256)
tokenizer.enable_padding(
    pad_id=0,
    pad_token="[PAD]",
)

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"],
    sess_options=ort.SessionOptions(),
)

texts = [
    "Developed Python applications using FastAPI.",
    "Experience building backend applications with Python.",
]

encoded = tokenizer.encode_batch(texts)

input_ids = np.array(
    [item.ids for item in encoded],
    dtype=np.int64,
)

attention_mask = np.array(
    [item.attention_mask for item in encoded],
    dtype=np.int64,
)

token_type_ids = np.zeros_like(input_ids)

available_inputs = {
    item.name for item in session.get_inputs()
}

inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
}

if "token_type_ids" in available_inputs:
    inputs["token_type_ids"] = token_type_ids

outputs = session.run(None, inputs)

token_embeddings = outputs[0]

# Reproduce Sentence Transformers mean pooling.
mask = attention_mask[:, :, None]

sentence_embeddings = (
        (token_embeddings * mask).sum(axis=1)
        / np.maximum(mask.sum(axis=1), 1)
)

# Normalize embeddings.
sentence_embeddings /= np.maximum(
    np.linalg.norm(
        sentence_embeddings,
        axis=1,
        keepdims=True,
    ),
    1e-12,
)

similarity = float(
    sentence_embeddings[0] @ sentence_embeddings[1]
)

print("ONNX-only model loaded successfully")
print("Embedding shape:", sentence_embeddings.shape)
print("Similarity:", round(similarity, 4))

usage = resource.getrusage(resource.RUSAGE_SELF)

print(
    "Maximum resident memory:",
    round(usage.ru_maxrss / (1024 * 1024), 1),
    "MiB",
)

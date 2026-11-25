
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    backend="onnx",
    model_kwargs={
        "file_name": "onnx/model_qint8_arm64.onnx",
        "provider": "CPUExecutionProvider",
    },
)

sentences = [
    "Developed Python applications using FastAPI.",
    "Experience building backend applications with Python.",
]

embeddings = model.encode(
    sentences,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

similarity = float(embeddings[0] @ embeddings[1])

print("ONNX model loaded successfully")
print("Embedding shape:", embeddings.shape)
print("Similarity:", round(similarity, 4))

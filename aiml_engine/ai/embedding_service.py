from sentence_transformers import SentenceTransformer
import torch

MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

    return _model


def generate_embedding(text):
    model = get_model()

    with torch.inference_mode():
        embedding = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    return embedding.tolist()
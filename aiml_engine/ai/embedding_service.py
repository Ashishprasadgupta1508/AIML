from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"

model = SentenceTransformer(MODEL_NAME)


def generate_embedding(text):
    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()

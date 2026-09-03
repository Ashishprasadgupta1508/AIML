from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def get_model():
    global _model

    if _model is None:
        _model = TextEmbedding(
            model_name=MODEL_NAME,
            threads=1,
        )

    return _model


def generate_embedding(text):
    model = get_model()

    embeddings = model.embed([text])

    embedding = next(embeddings)

    return embedding.tolist()
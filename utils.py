import numpy as np
import openai

client = openai.OpenAI()

# Helper functions


def create_embeddings(examples) -> np.ndarray:
    embeddings = client.embeddings.create(
        input=examples, model="text-embedding-3-small"
    )
    return np.array([embedding.embedding for embedding in embeddings.data])


def calculate_cosine_similarity(embedding_1, embedding_2):
    return np.dot(embedding_1, embedding_2) / (
        np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2)
    )

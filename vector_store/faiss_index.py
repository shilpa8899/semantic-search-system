import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):
        """
        Initialize FAISS index.

        dimension = size of embedding vectors (384)
        """
        self.index = faiss.IndexFlatL2(dimension)

    def add_vectors(self, embeddings):
        """
        Add embeddings to FAISS index.
        """
        self.index.add(embeddings)

    def search(self, query_vector, k=5):
        """
        Search for k most similar vectors.
        """

        distances, indices = self.index.search(query_vector, k)

        return distances, indices
from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:

    def __init__(self):
        """
        Load the embedding model.
        all-MiniLM-L6-v2 is a lightweight model that produces
        384-dimensional embeddings suitable for semantic search.
        """

        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode_documents(self, documents):
        """
        Convert a list of documents into embeddings.
        """

        embeddings = self.model.encode(
            documents,
            show_progress_bar=True
        )

        return np.array(embeddings)
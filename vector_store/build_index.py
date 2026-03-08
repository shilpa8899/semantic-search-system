import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import faiss
from vector_store.faiss_index import VectorStore


def main():

    print("Loading embeddings...")

    embeddings = np.load("data/document_embeddings.npy")

    dimension = embeddings.shape[1]

    print("Embedding dimension:", dimension)

    store = VectorStore(dimension)

    print("Building FAISS index...")

    store.add_vectors(embeddings)

    print("Saving FAISS index...")

    faiss.write_index(store.index, "data/faiss_index.bin")

    print("Index saved to data/faiss_index.bin")


if __name__ == "__main__":
    main()
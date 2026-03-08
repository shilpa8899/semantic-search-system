import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sentence_transformers import SentenceTransformer
from vector_store.faiss_index import VectorStore
from data.load_data import load_dataset


def main():

    print("Loading documents...")
    documents, _, _ = load_dataset()

    print("Loading embeddings...")
    embeddings = np.load("data/document_embeddings.npy")

    dimension = embeddings.shape[1]

    store = VectorStore(dimension)
    store.add_vectors(embeddings)

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    while True:

        query = input("\nEnter query (or type exit): ")

        if query.lower() == "exit":
            break

        query_embedding = model.encode([query])

        distances, indices = store.search(query_embedding, k=5)

        print("\nTop results:\n")

        for i in indices[0]:
            print("-----")
            print(documents[i][:300])

if __name__ == "__main__":
    main()
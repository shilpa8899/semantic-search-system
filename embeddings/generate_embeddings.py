import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

from data.load_data import load_dataset
from embeddings.embedder import Embedder


def main():

    print("Loading documents...")

    documents, _, _ = load_dataset()

    print("Total documents:", len(documents))

    embedder = Embedder()

    print("Generating embeddings...")

    embeddings = embedder.encode_documents(documents)

    print("Embeddings shape:", embeddings.shape)

    # Save embeddings to disk
    np.save("data/document_embeddings.npy", embeddings)

    print("Embeddings saved to data/document_embeddings.npy")


if __name__ == "__main__":
    main()
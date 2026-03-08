import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pickle
from clustering.fuzzy_cluster import perform_clustering


def main():

    print("Loading embeddings...")

    embeddings = np.load("data/document_embeddings.npy")

    gmm, probs = perform_clustering(embeddings, n_clusters=20)

    print("Saving clustering model...")

    with open("data/gmm_model.pkl", "wb") as f:
        pickle.dump(gmm, f)

    print("Saving cluster probabilities...")

    np.save("data/cluster_probs.npy", probs)

    print("Clustering model saved.")


if __name__ == "__main__":
    main()
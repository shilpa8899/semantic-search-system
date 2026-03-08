import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.mixture import GaussianMixture


def perform_clustering(embeddings, n_clusters=20):
    """
    Perform fuzzy clustering using Gaussian Mixture Model.
    Each document gets a probability distribution over clusters.
    """

    print("Running Gaussian Mixture clustering...")

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="tied",
        random_state=42
    )

    gmm.fit(embeddings)

    cluster_probabilities = gmm.predict_proba(embeddings)

    print("Clustering complete")

    return gmm, cluster_probabilities
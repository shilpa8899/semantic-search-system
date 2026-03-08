import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, similarity_threshold=0.85):
        """
        similarity_threshold determines how close two queries
        must be to count as a cache hit.
        """
        self.similarity_threshold = similarity_threshold
        self.cache = []

        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding):
        """
        Check if a similar query exists in cache.
        """

        for entry in self.cache:

            sim = cosine_similarity(
                query_embedding,
                entry["embedding"]
            )[0][0]

            if sim >= self.similarity_threshold:
                self.hit_count += 1

                return {
                    "hit": True,
                    "matched_query": entry["query"],
                    "similarity": float(sim),
                    "result": entry["result"],
                    "cluster": entry["cluster"]
                }

        self.miss_count += 1

        return {"hit": False}

    def store(self, query, embedding, result, cluster):

        entry = {
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        }

        self.cache.append(entry)

    def stats(self):

        total = self.hit_count + self.miss_count

        hit_rate = self.hit_count / total if total > 0 else 0

        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0
import pickle
import faiss
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from data.load_data import load_dataset
from vector_store.faiss_index import VectorStore
from cache.semantic_cache import SemanticCache


# ---------- Request Model ----------
class QueryRequest(BaseModel):
    query: str


# ---------- Initialize System ----------
print("Loading documents...")
documents, _, _ = load_dataset()

print("Loading embeddings...")
embeddings = np.load("data/document_embeddings.npy")

dimension = embeddings.shape[1]

print("Building FAISS index...")
print("Loading FAISS index...")
index = faiss.read_index("data/faiss_index.bin")

vector_store = VectorStore(dimension)
vector_store.index = index

print("Loading clustering model...")
with open("data/gmm_model.pkl", "rb") as f:
    gmm_model = pickle.load(f)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Initializing semantic cache...")
cache = SemanticCache()

app = FastAPI(title="Semantic Search API")


# ---------- Query Endpoint ----------
@app.post("/query")
def query_search(request: QueryRequest):

    query = request.query

    query_embedding = model.encode([query])

    # Check cache
    cache_result = cache.lookup(query_embedding)

    if cache_result["hit"]:
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cache_result["matched_query"],
            "similarity_score": cache_result["similarity"],
            "result": cache_result["result"],
            "dominant_cluster": cache_result["cluster"]
        }

    # Cache miss → search FAISS
    distances, indices = vector_store.search(query_embedding, k=1)

    doc_index = int(indices[0][0])

    result = documents[doc_index][:500]

    cluster_probs = gmm_model.predict_proba(query_embedding)
    dominant_cluster = int(np.argmax(cluster_probs))

    # Store result in cache
    cache.store(query, query_embedding, result, dominant_cluster)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": dominant_cluster
    }


# ---------- Cache Stats ----------
@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


# ---------- Clear Cache ----------
@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}
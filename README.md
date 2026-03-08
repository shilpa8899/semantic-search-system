Semantic Search System with Fuzzy Clustering and Semantic Cache
Overview

This project implements a semantic search system using the 20 Newsgroups dataset.
The system retrieves relevant documents based on the semantic meaning of queries rather than simple keyword matching.

To improve efficiency, the system also includes a semantic cache that avoids recomputing results for queries that are semantically similar to previously asked queries.

The entire system is exposed through a REST API built with FastAPI.

Key Features

Semantic document search using vector embeddings

Efficient similarity search using a vector database

Fuzzy clustering of documents using probabilistic clustering

Custom semantic cache built from scratch (no Redis or external caching tools)

FastAPI service for querying the system

System Architecture
User Query
   │
   ▼
Query Embedding (SentenceTransformer)
   │
   ▼
Semantic Cache Lookup
   │
   ├── Cache Hit → Return Cached Result
   │
   └── Cache Miss
           │
           ▼
      FAISS Vector Search
           │
           ▼
   Retrieve Most Similar Document
           │
           ▼
      Store Result in Cache
           │
           ▼
        Return Response
Dataset

This project uses the 20 Newsgroups dataset, a popular dataset for text classification and NLP research.

Dataset source:
https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

Dataset statistics:

~20,000 documents

20 topic categories

Example categories include:

comp.graphics

sci.space

rec.sport.baseball

talk.politics.guns

sci.med

During preprocessing, the following noisy components were removed:

email headers

message footers

quoted replies

This ensures embeddings capture the actual semantic content of messages.

Technologies Used

The project is implemented using the following technologies:

Python

FastAPI – REST API framework

SentenceTransformers – text embedding generation

FAISS – vector similarity search

Scikit-learn – clustering algorithms

NumPy – numerical computation

Uvicorn – ASGI server for running the API

Project Structure
trademarkia-semantic-search
│
├── api
│   └── main.py
│
├── cache
│   └── semantic_cache.py
│
├── clustering
│   ├── fuzzy_cluster.py
│   └── run_clustering.py
│
├── embeddings
│   ├── embedder.py
│   └── generate_embeddings.py
│
├── vector_store
│   ├── faiss_index.py
│   ├── build_index.py
│   └── search_engine.py
│
├── data
│
├── requirements.txt
├── README.md
└── .gitignore
Semantic Cache Design

Traditional caches only match identical queries.
This project implements a semantic cache that detects queries with similar meaning.

Each cache entry stores:

query text

query embedding

retrieved result

dominant cluster

When a new query arrives:

The query is embedded.

Its embedding is compared with cached embeddings.

If similarity exceeds a threshold, the cached result is returned.

Similarity threshold used:

0.85
Threshold Behavior
Threshold	Behavior
0.95	Very strict matching
0.85	Balanced matching
0.70	More aggressive caching
API Endpoints
POST /query

Accepts a natural language query and returns the most relevant document.

Example request:

{
  "query": "space shuttle launch"
}

Example response:

{
  "query": "space shuttle launch",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "...document text...",
  "dominant_cluster": 4
}
GET /cache/stats

Returns statistics about the semantic cache.

Example response:

{
  "total_entries": 2,
  "hit_count": 1,
  "miss_count": 1,
  "hit_rate": 0.5
}
DELETE /cache

Clears all cache entries.

Example response:

{
  "message": "Cache cleared"
}
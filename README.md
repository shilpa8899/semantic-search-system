# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a lightweight semantic search system built on the **20 Newsgroups dataset**. Instead of traditional keyword search, the system retrieves documents based on the **semantic meaning of a query** using vector embeddings.

To improve efficiency, the system includes a **semantic cache** that detects semantically similar queries and avoids recomputing results. The system is exposed through a **REST API built with FastAPI**.

---

## Key Features

- Semantic document retrieval using vector embeddings  
- Efficient similarity search using a vector database (FAISS)  
- Fuzzy clustering of documents using probabilistic clustering (Gaussian Mixture Models)  
- Custom semantic cache built from scratch (no Redis or external caching tools)  
- FastAPI service exposing the search system as an API  

---

## System Architecture

<img width="340" height="472" src="https://github.com/user-attachments/assets/d58b1100-8bda-4551-8fd9-baccbac4dca0" />

---

## Dataset

This project uses the **20 Newsgroups dataset**, a widely used dataset for text classification and NLP research.

Dataset source:  
https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

Dataset characteristics:

- Approximately **20,000 documents**
- **20 topic categories**

Example categories include:

- comp.graphics  
- sci.space  
- rec.sport.baseball  
- talk.politics.guns  
- sci.med  

During preprocessing, the following noisy components were removed:

- email headers  
- message footers  
- quoted replies  

This ensures embeddings capture the **actual semantic content of the documents**.

---

## Technologies Used

| Layer | Technology |
|------|-------------|
| Language | Python |
| API Framework | FastAPI |
| Embeddings | SentenceTransformers |
| Vector Search | FAISS |
| Clustering | Scikit-learn |
| Numerical Computing | NumPy |
| Server | Uvicorn |

---

## Project Structure

<img width="329" height="649" src="https://github.com/user-attachments/assets/a92513db-7311-41d6-9d6d-c442b692de02" />

---

## Semantic Cache Design

Traditional caching systems only match **identical queries**.  
This project implements a **semantic cache** that recognizes queries with **similar meaning** using cosine similarity between embeddings.

Each cache entry stores:

- query text  
- query embedding  
- retrieved result  
- dominant cluster  

When a new query arrives:

1. The query is converted to an embedding.
2. It is compared with cached embeddings using cosine similarity.
3. If similarity exceeds a threshold, the cached result is returned.

Similarity threshold used: **0.85**

---

### Threshold Behavior

| Threshold | Behavior |
|----------|-----------|
| 0.95 | Strict matching |
| 0.85 | Balanced matching |
| 0.70 | Aggressive caching |

---

# API Endpoints

## POST `/query`

Accepts a natural language query and returns the most relevant document.

### Example request

```json
{
  "query": "space shuttle launch"
}
```

### Example response

```json
{
  "query": "space shuttle launch",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "...document text...",
  "dominant_cluster": 4
}
```

---

## GET `/cache/stats`

Returns statistics about the semantic cache.

### Example response

```json
{
  "total_entries": 2,
  "hit_count": 1,
  "miss_count": 1,
  "hit_rate": 0.5
}
```

---

## DELETE `/cache`

Clears all cache entries.

### Example response

```json
{
  "message": "Cache cleared"
}
```

---

# API Documentation

The API is built using FastAPI, which automatically generates interactive documentation.

Open the documentation in your browser:

```
http://127.0.0.1:8000/docs
```

---

# API Demonstration

### FastAPI Documentation Interface

![API Docs](https://github.com/user-attachments/assets/dd63df72-aa29-4f98-a205-7360b6b5ec87)

### Query Endpoint Example

![Query Example](https://github.com/user-attachments/assets/231c9198-8b4c-4822-8746-dbf3befa9ec8)

### Cache Statistics Example

![Cache Stats](https://github.com/user-attachments/assets/bf1e2f4a-de18-4499-8a98-5f288c34db2f)

---

# Conclusion

This project demonstrates how modern NLP techniques such as embeddings, vector search, clustering, and semantic caching can be combined to build an efficient semantic search system.

The system improves performance by recognizing semantically similar queries and avoiding redundant computations while still returning relevant results.

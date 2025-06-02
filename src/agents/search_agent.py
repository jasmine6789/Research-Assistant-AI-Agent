import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import numpy as np

# Placeholder for NoteTaker logging
class NoteTaker:
    @staticmethod
    def log(event_type: str, data: dict):
        # TODO: Implement actual logging to MongoDB
        print(f"[NoteTaker] {event_type}: {data}")

class SearchAgent:
    def __init__(self, mongo_uri: str, db_name: str = "arxiv", collection_name: str = "papers"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query])[0]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embed_query(query)
        # Assume each paper doc has a 'embedding' field (list of floats)
        papers = list(self.collection.find({}, {"_id": 0, "title": 1, "abstract": 1, "embedding": 1, "authors": 1, "year": 1, "arxiv_id": 1}))
        scored = []
        for paper in papers:
            if "embedding" in paper:
                score = np.dot(query_embedding, paper["embedding"]) / (np.linalg.norm(query_embedding) * np.linalg.norm(paper["embedding"]))
                scored.append((score, paper))
        scored.sort(reverse=True, key=lambda x: x[0])
        top_papers = [p for _, p in scored[:top_k]]
        # Log the search event
        NoteTaker.log("search", {"query": query, "top_papers": [p["arxiv_id"] for p in top_papers]})
        return top_papers

# Example usage (to be removed in production)
if __name__ == "__main__":
    MONGO_URI = os.getenv("MONGO_URI")
    agent = SearchAgent(MONGO_URI)
    results = agent.search("transformer models for time series forecasting")
    for paper in results:
        print(f"{paper['title']} ({paper['year']}) - {paper['arxiv_id']}") 
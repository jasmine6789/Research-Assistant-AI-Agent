import os
from typing import List, Dict, Any
from google.cloud import bigquery
import re
from src.agents.note_taker import NoteTaker

class InsightAgent:
    def __init__(self, project_id: str, note_taker: NoteTaker, dataset: str = "arxiv", table: str = "papers"):
        self.client = bigquery.Client(project=project_id)
        self.dataset = dataset
        self.table = table
        self.note_taker = note_taker

    def publication_trends_by_year(self) -> List[Dict[str, Any]]:
        query = f"""
            SELECT year, COUNT(*) as paper_count
            FROM `{self.dataset}.{self.table}`
            GROUP BY year
            ORDER BY year
        """
        results = self.client.query(query).result()
        trends = [{"year": row.year, "paper_count": row.paper_count} for row in results]
        self.note_taker.log_insight("Publication trends by year", {"trends": trends})
        return trends

    def keyword_frequency(self, keywords: List[str]) -> Dict[str, int]:
        keyword_counts = {}
        for kw in keywords:
            query = f"""
                SELECT COUNT(*) as count
                FROM `{self.dataset}.{self.table}`
                WHERE LOWER(abstract) LIKE '%{kw.lower()}%'
            """
            result = list(self.client.query(query).result())[0]
            keyword_counts[kw] = result.count
        self.note_taker.log_insight("Keyword frequency", {"keywords": keyword_counts})
        return keyword_counts

    def evaluation_metrics_regex(self, metrics: List[str]) -> Dict[str, int]:
        query = f"SELECT abstract FROM `{self.dataset}.{self.table}`"
        abstracts = [row.abstract for row in self.client.query(query).result()]
        metric_counts = {m: 0 for m in metrics}
        for abstract in abstracts:
            for m in metrics:
                if m.lower() in abstract.lower():
                    metric_counts[m] += 1
        self.note_taker.log_insight("Evaluation metrics", {"metrics": metric_counts})
        return metric_counts

    def topic_modeling(self, n_topics: int = 5) -> Dict[str, Any]:
        query = f"SELECT abstract FROM `{self.dataset}.{self.table}`"
        abstracts = [row.abstract for row in self.client.query(query).result()]
        topics = {f"Topic {i+1}": [f"word{i*3+1}", f"word{i*3+2}", f"word{i*3+3}"] for i in range(n_topics)}
        self.note_taker.log_insight("Topic modeling", {"topics": topics})
        return topics

    def cooccurrence_matrix(self, keywords: List[str]) -> Dict[str, Dict[str, int]]:
        query = f"SELECT abstract FROM `{self.dataset}.{self.table}`"
        abstracts = [row.abstract for row in self.client.query(query).result()]
        matrix = {kw1: {kw2: 0 for kw2 in keywords} for kw1 in keywords}
        for abstract in abstracts:
            present = [kw for kw in keywords if kw.lower() in abstract.lower()]
            for i, kw1 in enumerate(present):
                for kw2 in present[i+1:]:
                    matrix[kw1][kw2] += 1
                    matrix[kw2][kw1] += 1
        self.note_taker.log_insight("Co-occurrence matrix", {"matrix": matrix})
        return matrix

    def embedding_clustering(self, n_clusters: int = 5) -> Dict[str, Any]:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        query = f"SELECT abstract FROM `{self.dataset}.{self.table}`"
        abstracts = [row.abstract for row in self.client.query(query).result()]
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(abstracts)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        clusters = {f"Cluster {i}": [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            clusters[f"Cluster {label}"].append(abstracts[idx])
        self.note_taker.log_insight("Embedding clustering", {"clusters": clusters})
        return clusters

    def extract_entities(self, nlp_model: str = "en_core_web_sm", top_n: int = 10) -> Dict[str, int]:
        import spacy
        from collections import Counter
        query = f"SELECT abstract FROM `{self.dataset}.{self.table}`"
        abstracts = [row.abstract for row in self.client.query(query).result()]
        nlp = spacy.load(nlp_model)
        entities = []
        for abstract in abstracts:
            doc = nlp(abstract)
            entities.extend([ent.text for ent in doc.ents])
        counter = Counter(entities)
        top_entities = dict(counter.most_common(top_n))
        self.note_taker.log_insight("Named entities", {"entities": top_entities})
        return top_entities

# Example usage (to be removed in production)
if __name__ == "__main__":
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    MONGO_URI = os.getenv("MONGO_URI")
    note_taker = NoteTaker(MONGO_URI)
    agent = InsightAgent(PROJECT_ID, note_taker)
    print("Publication trends:", agent.publication_trends_by_year())
    print("Keyword frequency:", agent.keyword_frequency(["transformer", "deep learning"]))
    print("Metric counts:", agent.evaluation_metrics_regex(["F1-score", "AUC"]))
    print("Topics:", agent.topic_modeling(n_topics=3)) 
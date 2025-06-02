import os
from typing import List, Dict, Any
from google.cloud import bigquery
import re
from src.agents.search_agent import NoteTaker

class InsightAgent:
    def __init__(self, project_id: str, dataset: str = "arxiv", table: str = "papers"):
        self.client = bigquery.Client(project=project_id)
        self.dataset = dataset
        self.table = table

    def publication_trends_by_year(self) -> List[Dict[str, Any]]:
        query = f"""
            SELECT year, COUNT(*) as paper_count
            FROM `{self.dataset}.{self.table}`
            GROUP BY year
            ORDER BY year
        """
        results = self.client.query(query).result()
        trends = [{"year": row.year, "paper_count": row.paper_count} for row in results]
        NoteTaker.log("insight_publication_trends", {"trends": trends})
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
        NoteTaker.log("insight_keyword_frequency", {"keywords": keyword_counts})
        return keyword_counts

    def evaluation_metrics_regex(self, metrics: List[str]) -> Dict[str, int]:
        # This is a placeholder for regex-based metric extraction
        # In practice, you would run this on the abstract text in Python after fetching from BigQuery
        query = f"SELECT abstract FROM `{self.dataset}.{self.table}`"
        abstracts = [row.abstract for row in self.client.query(query).result()]
        metric_counts = {m: 0 for m in metrics}
        for abstract in abstracts:
            for m in metrics:
                if re.search(rf"\\b{re.escape(m)}\\b", abstract, re.IGNORECASE):
                    metric_counts[m] += 1
        NoteTaker.log("insight_evaluation_metrics", {"metrics": metric_counts})
        return metric_counts

    def topic_modeling(self, n_topics: int = 5) -> Dict[str, Any]:
        # Placeholder for topic modeling (e.g., using sklearn or spaCy)
        # In practice, fetch abstracts and run LDA or similar
        query = f"SELECT abstract FROM `{self.dataset}.{self.table}`"
        abstracts = [row.abstract for row in self.client.query(query).result()]
        # TODO: Implement topic modeling (e.g., sklearn.decomposition.LatentDirichletAllocation)
        topics = {f"Topic {i+1}": [f"word{i*3+1}", f"word{i*3+2}", f"word{i*3+3}"] for i in range(n_topics)}
        NoteTaker.log("insight_topic_modeling", {"topics": topics})
        return topics

# Example usage (to be removed in production)
if __name__ == "__main__":
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    agent = InsightAgent(PROJECT_ID)
    print("Publication trends:", agent.publication_trends_by_year())
    print("Keyword frequency:", agent.keyword_frequency(["transformer", "deep learning"]))
    print("Metric counts:", agent.evaluation_metrics_regex(["F1-score", "AUC"]))
    print("Topics:", agent.topic_modeling(n_topics=3)) 
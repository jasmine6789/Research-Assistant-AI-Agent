from typing import List, Dict, Any
import re
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from src.agents.note_taker import NoteTaker
from src.agents.web_search_agent import WebSearchAgent

class WebInsightAgent:
    def __init__(self, note_taker: NoteTaker, search_agent: WebSearchAgent = None):
        self.note_taker = note_taker
        self.search_agent = search_agent
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def publication_trends_by_year(self, query: str = "machine learning", max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Analyze publication trends by year for a given topic by searching arXiv.
        """
        if not self.search_agent:
            return []
        
        # Search for papers on the topic
        papers = self.search_agent.search_arxiv(query, max_results=max_results)
        
        # Count papers by year
        year_counts = Counter()
        for paper in papers:
            year_counts[paper['year']] += 1
        
        # Convert to sorted list
        trends = [{"year": year, "paper_count": count} for year, count in sorted(year_counts.items())]
        
        self.note_taker.log_insight("Publication trends by year", {"query": query, "trends": trends})
        return trends

    def keyword_frequency(self, papers: List[Dict[str, Any]], keywords: List[str]) -> Dict[str, int]:
        """
        Analyze keyword frequency in the abstracts of the provided papers.
        """
        keyword_counts = {}
        
        for kw in keywords:
            count = 0
            for paper in papers:
                abstract = paper.get('abstract', '').lower()
                if kw.lower() in abstract:
                    count += 1
            keyword_counts[kw] = count
        
        self.note_taker.log_insight("Keyword frequency", {"keywords": keyword_counts})
        return keyword_counts

    def extract_common_keywords(self, papers: List[Dict[str, Any]], top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Extract the most common keywords from paper titles and abstracts.
        """
        # Combine all text
        all_text = []
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            all_text.append(text.lower())
        
        # Simple keyword extraction (could be improved with NLP)
        all_words = []
        for text in all_text:
            # Remove common words and extract meaningful terms
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            # Filter out common stop words
            stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 'but', 'words', 'use', 'each', 'which', 'she', 'his', 'how', 'now', 'did', 'get', 'has', 'him', 'may', 'new', 'say', 'its', 'two', 'way', 'who', 'oil', 'sit', 'set'}
            filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
            all_words.extend(filtered_words)
        
        # Count frequency
        word_counts = Counter(all_words)
        top_keywords = [{"keyword": word, "frequency": count} for word, count in word_counts.most_common(top_n)]
        
        self.note_taker.log_insight("Common keywords", {"keywords": top_keywords})
        return top_keywords

    def evaluation_metrics_regex(self, papers: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, int]:
        """
        Count mentions of evaluation metrics in paper abstracts.
        """
        metric_counts = {m: 0 for m in metrics}
        
        for paper in papers:
            abstract = paper.get('abstract', '').lower()
            for metric in metrics:
                # Use regex for more flexible matching
                pattern = re.compile(r'\b' + re.escape(metric.lower()) + r'\b')
                if pattern.search(abstract):
                    metric_counts[metric] += 1
        
        self.note_taker.log_insight("Evaluation metrics", {"metrics": metric_counts})
        return metric_counts

    def topic_modeling(self, papers: List[Dict[str, Any]], n_topics: int = 5) -> Dict[str, Any]:
        """
        Perform simple clustering-based topic modeling on paper abstracts.
        """
        if len(papers) < n_topics:
            n_topics = max(1, len(papers))
        
        # Create embeddings for abstracts
        abstracts = [paper.get('abstract', '') for paper in papers if paper.get('abstract')]
        if not abstracts:
            return {}
        
        embeddings = self.model.encode(abstracts)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Group papers by cluster and extract representative terms
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[f"Topic {label + 1}"].append(abstracts[i])
        
        # Extract representative words for each cluster
        topics = {}
        for topic_name, cluster_abstracts in clusters.items():
            # Simple word frequency approach for topic words
            all_words = []
            for abstract in cluster_abstracts:
                words = re.findall(r'\b[a-zA-Z]{4,}\b', abstract.lower())
                all_words.extend(words)
            
            word_counts = Counter(all_words)
            # Filter out very common words
            common_words = {'this', 'that', 'with', 'from', 'they', 'been', 'have', 'their', 'said', 'each', 'which', 'them', 'many', 'some', 'time', 'very', 'when', 'much', 'call', 'first', 'water', 'long', 'down', 'day', 'made', 'part'}
            filtered_counts = {w: c for w, c in word_counts.items() if w not in common_words}
            
            top_words = [word for word, _ in Counter(filtered_counts).most_common(5)]
            topics[topic_name] = top_words
        
        self.note_taker.log_insight("Topic modeling", {"topics": topics})
        return topics

    def cooccurrence_matrix(self, papers: List[Dict[str, Any]], keywords: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Calculate keyword co-occurrence matrix from paper abstracts.
        """
        matrix = {kw1: {kw2: 0 for kw2 in keywords} for kw1 in keywords}
        
        for paper in papers:
            abstract = paper.get('abstract', '').lower()
            present_keywords = [kw for kw in keywords if kw.lower() in abstract]
            
            # Count co-occurrences
            for i, kw1 in enumerate(present_keywords):
                for kw2 in present_keywords[i+1:]:
                    matrix[kw1][kw2] += 1
                    matrix[kw2][kw1] += 1  # Make matrix symmetric
        
        self.note_taker.log_insight("Co-occurrence matrix", {"matrix": matrix})
        return matrix

    def embedding_clustering(self, papers: List[Dict[str, Any]], n_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster papers based on abstract embeddings and return cluster assignments.
        """
        abstracts = [paper.get('abstract', '') for paper in papers if paper.get('abstract')]
        if len(abstracts) < n_clusters:
            n_clusters = max(1, len(abstracts))
        
        # Generate embeddings
        embeddings = self.model.encode(abstracts)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Group papers by cluster
        clusters = {f"Cluster {i}": [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            paper_info = {
                "title": papers[idx].get('title', ''),
                "arxiv_id": papers[idx].get('arxiv_id', ''),
                "year": papers[idx].get('year', ''),
                "abstract_snippet": abstracts[idx][:100] + "..."
            }
            clusters[f"Cluster {label}"].append(paper_info)
        
        self.note_taker.log_insight("Embedding clustering", {"clusters": clusters})
        return clusters

    def author_collaboration_analysis(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze author collaboration patterns.
        """
        author_counts = Counter()
        collaborations = Counter()
        
        for paper in papers:
            authors = paper.get('authors', [])
            for author in authors:
                author_counts[author] += 1
            
            # Count collaborations (pairs of authors)
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    collab_key = tuple(sorted([author1, author2]))
                    collaborations[collab_key] += 1
        
        # Get top authors and collaborations
        top_authors = [{"author": author, "paper_count": count} for author, count in author_counts.most_common(10)]
        top_collabs = [{"authors": list(authors), "collaboration_count": count} for authors, count in collaborations.most_common(10)]
        
        result = {
            "top_authors": top_authors,
            "top_collaborations": top_collabs,
            "total_unique_authors": len(author_counts),
            "total_collaborations": len(collaborations)
        }
        
        self.note_taker.log_insight("Author collaboration analysis", result)
        return result

    def analyze_papers(self, papers: List[Dict[str, Any]], query: str = "machine learning") -> Dict[str, Any]:
        """
        Comprehensive paper analysis using all available methods.
        Returns a unified insights dictionary.
        """
        try:
            print("   📊 Running publication trends analysis...")
            trends = self.publication_trends_by_year(query, max_results=len(papers))
            
            print("   🔍 Extracting common keywords...")
            common_keywords = self.extract_common_keywords(papers, top_n=15)
            
            print("   🏷️ Performing topic modeling...")
            topics = self.topic_modeling(papers, n_topics=min(5, len(papers)))
            
            print("   📈 Analyzing evaluation metrics...")
            standard_metrics = ["accuracy", "precision", "recall", "f1-score", "auc", "mae", "mse", "rmse", "bleu", "rouge"]
            metrics = self.evaluation_metrics_regex(papers, standard_metrics)
            
            print("   🤝 Analyzing author collaborations...")
            author_analysis = self.author_collaboration_analysis(papers)
            
            print("   🧩 Performing embedding clustering...")
            clusters = self.embedding_clustering(papers, n_clusters=min(3, len(papers)))
            
            # Create comprehensive insights
            insights = {
                "publication_trends": trends,
                "common_keywords": common_keywords,
                "topic_modeling": topics,
                "evaluation_metrics": metrics,
                "author_collaboration": author_analysis,
                "paper_clusters": clusters,
                "total_papers_analyzed": len(papers),
                "analysis_summary": {
                    "most_common_keyword": common_keywords[0]["keyword"] if common_keywords else "N/A",
                    "most_cited_metric": max(metrics.items(), key=lambda x: x[1])[0] if metrics else "N/A",
                    "top_author": author_analysis["top_authors"][0]["author"] if author_analysis.get("top_authors") else "N/A",
                    "dominant_topic": list(topics.keys())[0] if topics else "N/A"
                }
            }
            
            self.note_taker.log_insight("Comprehensive paper analysis", insights)
            return insights
            
        except Exception as e:
            print(f"   ⚠️ Error in paper analysis: {e}")
            # Return minimal insights on error
            return {
                "publication_trends": [],
                "common_keywords": [],
                "topic_modeling": {},
                "evaluation_metrics": {},
                "author_collaboration": {"top_authors": [], "top_collaborations": []},
                "paper_clusters": {},
                "total_papers_analyzed": len(papers),
                "analysis_summary": {"error": str(e)}
            }

# Example usage
if __name__ == "__main__":
    from src.agents.note_taker import NoteTaker
    import urllib.parse
    
    password = "Jasmine@0802"
    encoded_password = urllib.parse.quote_plus(password)
    MONGO_URI = f"mongodb+srv://jaschri:{encoded_password}@agent.wb3vq0q.mongodb.net/?retryWrites=true&w=majority&appName=Agent"
    
    note_taker = NoteTaker(MONGO_URI)
    search_agent = WebSearchAgent(note_taker)
    insight_agent = WebInsightAgent(note_taker, search_agent)
    
    # Search for papers
    papers = search_agent.search("transformer time series", top_k=10, max_results=20)
    
    # Analyze the papers
    print("📊 Publication trends:")
    trends = insight_agent.publication_trends_by_year("transformer time series", max_results=30)
    for trend in trends:
        print(f"  {trend['year']}: {trend['paper_count']} papers")
    
    print("\n🔍 Keyword frequency:")
    keywords = ["transformer", "attention", "time series", "forecasting", "LSTM"]
    freq = insight_agent.keyword_frequency(papers, keywords)
    for kw, count in freq.items():
        print(f"  {kw}: {count}")
    
    print("\n🏷️ Topic modeling:")
    topics = insight_agent.topic_modeling(papers, n_topics=3)
    for topic, words in topics.items():
        print(f"  {topic}: {', '.join(words)}") 
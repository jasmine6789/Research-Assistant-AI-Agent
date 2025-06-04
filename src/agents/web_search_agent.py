import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from src.agents.note_taker import NoteTaker
import time

class WebSearchAgent:
    def __init__(self, note_taker: NoteTaker):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.note_taker = note_taker
        self.arxiv_base_url = "http://export.arxiv.org/api/query"

    def search_arxiv(self, query: str, max_results: int = 20, category: str = "cs.LG") -> List[Dict[str, Any]]:
        """
        Search arXiv directly using their API for papers in the cs.LG category.
        """
        # Construct search query for arXiv API
        search_query = f"cat:{category} AND all:{query}"
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.arxiv_base_url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = self._parse_arxiv_entry(entry)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []

    def _parse_arxiv_entry(self, entry) -> Dict[str, Any]:
        """Parse a single arXiv entry from XML to our paper format."""
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        
        try:
            # Extract basic information
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            
            # Extract arXiv ID from the URL
            arxiv_url = entry.find('atom:id', ns).text
            arxiv_id = arxiv_url.split('/')[-1]
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns).text
                authors.append(name)
            
            # Extract publication date
            published = entry.find('atom:published', ns).text
            year = int(published.split('-')[0])
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', ns):
                categories.append(category.get('term'))
            
            return {
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'year': year,
                'arxiv_id': arxiv_id,
                'arxiv_url': arxiv_url,
                'categories': categories,
                'published': published
            }
            
        except Exception as e:
            print(f"Error parsing arXiv entry: {e}")
            return None

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for the search query."""
        return self.model.encode([query])[0]

    def rank_papers_by_similarity(self, papers: List[Dict[str, Any]], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rank papers by semantic similarity to the query using embeddings.
        """
        if not papers:
            return []
        
        query_embedding = self.embed_query(query)
        
        scored_papers = []
        for paper in papers:
            # Create text for embedding (title + abstract)
            paper_text = f"{paper['title']} {paper['abstract']}"
            paper_embedding = self.model.encode([paper_text])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, paper_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(paper_embedding)
            )
            
            paper['similarity_score'] = similarity
            scored_papers.append((similarity, paper))
        
        # Sort by similarity and return top-k
        scored_papers.sort(reverse=True, key=lambda x: x[0])
        return [paper for _, paper in scored_papers[:top_k]]

    def search(self, query: str, top_k: int = 5, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Main search function that searches arXiv and returns top-k most relevant papers.
        """
        print(f"🔍 Searching arXiv for: '{query}'")
        
        # Search arXiv
        papers = self.search_arxiv(query, max_results=max_results)
        print(f"📄 Found {len(papers)} papers from arXiv")
        
        if not papers:
            print("❌ No papers found")
            return []
        
        # Rank by semantic similarity
        ranked_papers = self.rank_papers_by_similarity(papers, query, top_k=top_k)
        print(f"✅ Returning top {len(ranked_papers)} most relevant papers")
        
        # Log the search and results
        self.note_taker.log_query(query)
        self.note_taker.log_selected_papers([p["arxiv_id"] for p in ranked_papers])
        
        return ranked_papers

    def search_multiple_queries(self, queries: List[str], top_k_per_query: int = 3) -> List[Dict[str, Any]]:
        """
        Search for multiple related queries and combine results.
        Useful for broader topic exploration.
        """
        all_papers = []
        seen_arxiv_ids = set()
        
        for query in queries:
            papers = self.search(query, top_k=top_k_per_query, max_results=15)
            
            # Add papers we haven't seen before
            for paper in papers:
                if paper['arxiv_id'] not in seen_arxiv_ids:
                    all_papers.append(paper)
                    seen_arxiv_ids.add(paper['arxiv_id'])
            
            # Rate limiting - be nice to arXiv
            time.sleep(1)
        
        return all_papers

# Example usage and testing
if __name__ == "__main__":
    from src.agents.note_taker import NoteTaker
    import os
    import urllib.parse
    
    password = "Jasmine@0802"
    encoded_password = urllib.parse.quote_plus(password)
    MONGO_URI = f"mongodb+srv://jaschri:{encoded_password}@agent.wb3vq0q.mongodb.net/?retryWrites=true&w=majority&appName=Agent"
    
    note_taker = NoteTaker(MONGO_URI)
    agent = WebSearchAgent(note_taker)
    
    # Test single query
    results = agent.search("transformer models for time series forecasting", top_k=3)
    
    print("\n🔍 Search Results:")
    for i, paper in enumerate(results, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
        print(f"   Year: {paper['year']}")
        print(f"   arXiv ID: {paper['arxiv_id']}")
        print(f"   Similarity: {paper.get('similarity_score', 0):.3f}")
        print(f"   Abstract: {paper['abstract'][:100]}...") 
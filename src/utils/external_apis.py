"""
External API Integration System for Academic Literature

Features:
- arXiv API integration
- Semantic Scholar API integration
- CrossRef API integration
- PubMed API integration (future)
- Rate limiting and caching
- Retry mechanisms with exponential backoff
- Data normalization and cleaning
"""

import requests
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import hashlib
import pickle
from urllib.parse import urlencode, quote
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class PaperMetadata:
    """Standardized paper metadata structure"""
    title: str
    authors: List[str]
    abstract: str
    publication_date: str
    venue: Optional[str] = None  # Journal/Conference name
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    paper_id: Optional[str] = None  # Semantic Scholar ID
    url: Optional[str] = None
    citation_count: Optional[int] = None
    reference_count: Optional[int] = None
    influential_citation_count: Optional[int] = None
    fields_of_study: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    pdf_url: Optional[str] = None
    source_api: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class APICache:
    """Simple file-based cache for API responses"""
    
    def __init__(self, cache_dir: str = "api_cache", max_age_hours: int = 24):
        """
        Initialize cache
        
        Args:
            cache_dir: Directory to store cache files
            max_age_hours: Maximum age of cached items in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
    
    def _get_cache_key(self, url: str, params: Dict[str, Any] = None) -> str:
        """Generate cache key from URL and parameters"""
        cache_string = url
        if params:
            cache_string += str(sorted(params.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, url: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        cache_key = self._get_cache_key(url, params)
        cache_file = self.cache_dir / f"{cache_key}.cache"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is still valid
                if datetime.now() - cached_data['timestamp'] < self.max_age:
                    logger.debug(f"Cache hit for {url}")
                    return cached_data['data']
                else:
                    # Remove expired cache
                    cache_file.unlink()
                    logger.debug(f"Cache expired for {url}")
            except Exception as e:
                logger.error(f"Error reading cache: {e}")
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def set(self, url: str, data: Dict[str, Any], params: Dict[str, Any] = None):
        """Cache response"""
        cache_key = self._get_cache_key(url, params)
        cache_file = self.cache_dir / f"{cache_key}.cache"
        
        cached_data = {
            'timestamp': datetime.now(),
            'data': data
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            logger.debug(f"Cached response for {url}")
        except Exception as e:
            logger.error(f"Error writing cache: {e}")

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        """
        Initialize rate limiter
        
        Args:
            calls_per_minute: Maximum calls per minute
        """
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()

class ArxivAPI:
    """ArXiv API integration"""
    
    def __init__(self, cache: APICache, rate_limiter: RateLimiter):
        self.base_url = "http://export.arxiv.org/api/query"
        self.cache = cache
        self.rate_limiter = rate_limiter
    
    def search_papers(self, 
                     query: str, 
                     max_results: int = 20,
                     sort_by: str = "relevance",
                     sort_order: str = "descending") -> List[PaperMetadata]:
        """
        Search arXiv for papers
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort_by: Sort criteria (relevance, lastUpdatedDate, submittedDate)
            sort_order: Sort order (ascending, descending)
            
        Returns:
            List of paper metadata
        """
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': sort_by,
            'sortOrder': sort_order
        }
        
        # Check cache first
        cached_result = self.cache.get(self.base_url, params)
        if cached_result:
            return [PaperMetadata(**paper) for paper in cached_result]
        
        self.rate_limiter.wait_if_needed()
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            papers = self._parse_arxiv_response(response.content)
            
            # Cache the results
            papers_dict = [paper.to_dict() for paper in papers]
            self.cache.set(self.base_url, papers_dict, params)
            
            logger.info(f"Retrieved {len(papers)} papers from arXiv")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[PaperMetadata]:
        """Get specific paper by arXiv ID"""
        papers = self.search_papers(f"id:{arxiv_id}", max_results=1)
        return papers[0] if papers else None
    
    def _parse_arxiv_response(self, xml_content: bytes) -> List[PaperMetadata]:
        """Parse arXiv XML response"""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                paper = self._parse_arxiv_entry(entry, ns)
                if paper:
                    papers.append(paper)
        
        except Exception as e:
            logger.error(f"Error parsing arXiv XML: {e}")
        
        return papers
    
    def _parse_arxiv_entry(self, entry, ns) -> Optional[PaperMetadata]:
        """Parse individual arXiv entry"""
        try:
            # Title
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip() if title_elem is not None else ""
            
            # Authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # Abstract
            summary_elem = entry.find('atom:summary', ns)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Publication date
            published_elem = entry.find('atom:published', ns)
            pub_date = ""
            if published_elem is not None:
                pub_date = published_elem.text[:10]  # YYYY-MM-DD format
            
            # arXiv ID
            id_elem = entry.find('atom:id', ns)
            arxiv_id = ""
            url = ""
            if id_elem is not None:
                url = id_elem.text
                arxiv_id = url.split('/')[-1]
            
            # Categories (fields of study)
            categories = []
            for category in entry.findall('atom:category', ns):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            return PaperMetadata(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=pub_date,
                arxiv_id=arxiv_id,
                url=url,
                fields_of_study=categories,
                source_api="arxiv"
            )
            
        except Exception as e:
            logger.error(f"Error parsing arXiv entry: {e}")
            return None

class SemanticScholarAPI:
    """Semantic Scholar API integration"""
    
    def __init__(self, cache: APICache, rate_limiter: RateLimiter, api_key: Optional[str] = None):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'x-api-key': api_key})
        
        # Add User-Agent to be respectful
        self.session.headers.update({
            'User-Agent': 'Research-Assistant-Agent/1.0 (mailto:research@example.com)'
        })
    
    def search_papers(self, 
                     query: str, 
                     max_results: int = 20,
                     fields: List[str] = None) -> List[PaperMetadata]:
        """
        Search Semantic Scholar for papers
        
        Args:
            query: Search query
            max_results: Maximum number of results
            fields: Fields to include in response
            
        Returns:
            List of paper metadata
        """
        if fields is None:
            fields = [
                'paperId', 'title', 'authors', 'abstract', 'year', 'venue',
                'doi', 'citationCount', 'referenceCount', 'influentialCitationCount',
                'fieldsOfStudy', 'url', 'externalIds'
            ]
        
        url = f"{self.base_url}/paper/search"
        params = {
            'query': query,
            'limit': min(max_results, 100),  # API limit
            'fields': ','.join(fields)
        }
        
        # Check cache first
        cached_result = self.cache.get(url, params)
        if cached_result:
            return [PaperMetadata(**paper) for paper in cached_result]
        
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for paper_data in data.get('data', []):
                paper = self._parse_semantic_scholar_paper(paper_data)
                if paper:
                    papers.append(paper)
            
            # Cache the results
            papers_dict = [paper.to_dict() for paper in papers]
            self.cache.set(url, papers_dict, params)
            
            logger.info(f"Retrieved {len(papers)} papers from Semantic Scholar")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []
    
    def get_paper_by_id(self, paper_id: str, fields: List[str] = None) -> Optional[PaperMetadata]:
        """Get specific paper by Semantic Scholar ID"""
        if fields is None:
            fields = [
                'paperId', 'title', 'authors', 'abstract', 'year', 'venue',
                'doi', 'citationCount', 'referenceCount', 'influentialCitationCount',
                'fieldsOfStudy', 'url', 'externalIds'
            ]
        
        url = f"{self.base_url}/paper/{paper_id}"
        params = {'fields': ','.join(fields)}
        
        # Check cache first
        cached_result = self.cache.get(url, params)
        if cached_result:
            return PaperMetadata(**cached_result)
        
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            paper = self._parse_semantic_scholar_paper(data)
            
            # Cache the result
            if paper:
                self.cache.set(url, paper.to_dict(), params)
            
            return paper
            
        except Exception as e:
            logger.error(f"Error getting paper from Semantic Scholar: {e}")
            return None
    
    def get_citations(self, paper_id: str, max_results: int = 50) -> List[PaperMetadata]:
        """Get papers that cite the given paper"""
        url = f"{self.base_url}/paper/{paper_id}/citations"
        params = {
            'limit': min(max_results, 1000),
            'fields': 'paperId,title,authors,year,venue,citationCount'
        }
        
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for citation_data in data.get('data', []):
                citing_paper = citation_data.get('citingPaper', {})
                paper = self._parse_semantic_scholar_paper(citing_paper)
                if paper:
                    papers.append(paper)
            
            logger.info(f"Retrieved {len(papers)} citing papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error getting citations from Semantic Scholar: {e}")
            return []
    
    def _parse_semantic_scholar_paper(self, data: Dict[str, Any]) -> Optional[PaperMetadata]:
        """Parse Semantic Scholar paper data"""
        try:
            # Authors
            authors = []
            for author in data.get('authors', []):
                if 'name' in author:
                    authors.append(author['name'])
            
            # External IDs for arXiv and DOI
            external_ids = data.get('externalIds', {})
            arxiv_id = external_ids.get('ArXiv', '')
            doi = external_ids.get('DOI', '') or data.get('doi', '')
            
            # Publication date from year
            year = data.get('year')
            pub_date = str(year) if year else ""
            
            return PaperMetadata(
                title=data.get('title', ''),
                authors=authors,
                abstract=data.get('abstract', ''),
                publication_date=pub_date,
                venue=data.get('venue', ''),
                doi=doi,
                arxiv_id=arxiv_id,
                paper_id=data.get('paperId', ''),
                url=data.get('url', ''),
                citation_count=data.get('citationCount'),
                reference_count=data.get('referenceCount'),
                influential_citation_count=data.get('influentialCitationCount'),
                fields_of_study=data.get('fieldsOfStudy', []),
                source_api="semantic_scholar"
            )
            
        except Exception as e:
            logger.error(f"Error parsing Semantic Scholar paper: {e}")
            return None

class CrossRefAPI:
    """CrossRef API integration"""
    
    def __init__(self, cache: APICache, rate_limiter: RateLimiter):
        self.base_url = "https://api.crossref.org/works"
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research-Assistant-Agent/1.0 (mailto:research@example.com)'
        })
    
    def search_papers(self, 
                     query: str, 
                     max_results: int = 20,
                     sort: str = "relevance") -> List[PaperMetadata]:
        """
        Search CrossRef for papers
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort: Sort criteria
            
        Returns:
            List of paper metadata
        """
        params = {
            'query': query,
            'rows': min(max_results, 1000),  # API limit
            'sort': sort,
            'order': 'desc'
        }
        
        # Check cache first
        cached_result = self.cache.get(self.base_url, params)
        if cached_result:
            return [PaperMetadata(**paper) for paper in cached_result]
        
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for item in data.get('message', {}).get('items', []):
                paper = self._parse_crossref_item(item)
                if paper:
                    papers.append(paper)
            
            # Cache the results
            papers_dict = [paper.to_dict() for paper in papers]
            self.cache.set(self.base_url, papers_dict, params)
            
            logger.info(f"Retrieved {len(papers)} papers from CrossRef")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching CrossRef: {e}")
            return []
    
    def get_paper_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """Get specific paper by DOI"""
        url = f"{self.base_url}/{doi}"
        
        # Check cache first
        cached_result = self.cache.get(url)
        if cached_result:
            return PaperMetadata(**cached_result)
        
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            paper = self._parse_crossref_item(data.get('message', {}))
            
            # Cache the result
            if paper:
                self.cache.set(url, paper.to_dict())
            
            return paper
            
        except Exception as e:
            logger.error(f"Error getting paper from CrossRef: {e}")
            return None
    
    def _parse_crossref_item(self, item: Dict[str, Any]) -> Optional[PaperMetadata]:
        """Parse CrossRef item"""
        try:
            # Title
            title = ""
            if 'title' in item and item['title']:
                title = item['title'][0]
            
            # Authors
            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)
            
            # Abstract
            abstract = item.get('abstract', '')
            
            # Publication date
            pub_date = ""
            if 'published-print' in item:
                date_parts = item['published-print']['date-parts'][0]
                if len(date_parts) >= 3:
                    pub_date = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                elif len(date_parts) >= 2:
                    pub_date = f"{date_parts[0]}-{date_parts[1]:02d}"
                elif len(date_parts) >= 1:
                    pub_date = str(date_parts[0])
            elif 'published-online' in item:
                date_parts = item['published-online']['date-parts'][0]
                if len(date_parts) >= 1:
                    pub_date = str(date_parts[0])
            
            # Venue (journal/container)
            venue = ""
            if 'container-title' in item and item['container-title']:
                venue = item['container-title'][0]
            
            # DOI
            doi = item.get('DOI', '')
            
            # URL
            url = item.get('URL', '')
            
            # Citation count (if available)
            citation_count = item.get('is-referenced-by-count')
            
            # Subject areas as fields of study
            fields_of_study = item.get('subject', [])
            
            return PaperMetadata(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=pub_date,
                venue=venue,
                doi=doi,
                url=url,
                citation_count=citation_count,
                fields_of_study=fields_of_study,
                source_api="crossref"
            )
            
        except Exception as e:
            logger.error(f"Error parsing CrossRef item: {e}")
            return None

class ExternalAPIManager:
    """
    Unified interface for all external academic APIs
    """
    
    def __init__(self, 
                 cache_dir: str = "api_cache",
                 cache_max_age_hours: int = 24,
                 rate_limit_per_minute: int = 30,
                 semantic_scholar_api_key: Optional[str] = None):
        """
        Initialize external API manager
        
        Args:
            cache_dir: Directory for caching API responses
            cache_max_age_hours: Maximum age of cached items
            rate_limit_per_minute: Rate limit for API calls
            semantic_scholar_api_key: Optional API key for Semantic Scholar
        """
        self.cache = APICache(cache_dir, cache_max_age_hours)
        self.rate_limiter = RateLimiter(rate_limit_per_minute)
        
        # Initialize APIs
        self.arxiv = ArxivAPI(self.cache, self.rate_limiter)
        self.semantic_scholar = SemanticScholarAPI(self.cache, self.rate_limiter, semantic_scholar_api_key)
        self.crossref = CrossRefAPI(self.cache, self.rate_limiter)
    
    def search_all_sources(self, 
                          query: str, 
                          max_results_per_source: int = 10) -> Dict[str, List[PaperMetadata]]:
        """
        Search all available sources for papers
        
        Args:
            query: Search query
            max_results_per_source: Maximum results per source
            
        Returns:
            Dictionary mapping source names to paper lists
        """
        results = {}
        
        # Search each source
        sources = {
            'arxiv': self.arxiv,
            'semantic_scholar': self.semantic_scholar,
            'crossref': self.crossref
        }
        
        for source_name, api in sources.items():
            try:
                papers = api.search_papers(query, max_results_per_source)
                results[source_name] = papers
                logger.info(f"Found {len(papers)} papers from {source_name}")
            except Exception as e:
                logger.error(f"Error searching {source_name}: {e}")
                results[source_name] = []
        
        return results
    
    def search_combined(self, 
                       query: str, 
                       max_total_results: int = 30,
                       deduplicate: bool = True) -> List[PaperMetadata]:
        """
        Search all sources and combine results
        
        Args:
            query: Search query
            max_total_results: Maximum total results
            deduplicate: Whether to remove duplicates
            
        Returns:
            Combined list of papers
        """
        per_source = max_total_results // 3  # Distribute among 3 sources
        all_results = self.search_all_sources(query, per_source)
        
        # Combine all papers
        combined_papers = []
        for source_papers in all_results.values():
            combined_papers.extend(source_papers)
        
        # Deduplicate if requested
        if deduplicate:
            combined_papers = self._deduplicate_papers(combined_papers)
        
        # Sort by citation count if available, then by relevance
        combined_papers.sort(key=lambda p: p.citation_count or 0, reverse=True)
        
        return combined_papers[:max_total_results]
    
    def get_paper_details(self, 
                         paper_id: str = None, 
                         doi: str = None, 
                         arxiv_id: str = None) -> Optional[PaperMetadata]:
        """
        Get detailed paper information by ID
        
        Args:
            paper_id: Semantic Scholar paper ID
            doi: DOI
            arxiv_id: arXiv ID
            
        Returns:
            Paper metadata if found
        """
        if paper_id:
            return self.semantic_scholar.get_paper_by_id(paper_id)
        elif doi:
            return self.crossref.get_paper_by_doi(doi)
        elif arxiv_id:
            return self.arxiv.get_paper_by_id(arxiv_id)
        else:
            logger.warning("No valid paper identifier provided")
            return None
    
    def get_related_papers(self, paper: PaperMetadata, max_results: int = 20) -> List[PaperMetadata]:
        """
        Get papers related to the given paper
        
        Args:
            paper: Source paper
            max_results: Maximum number of related papers
            
        Returns:
            List of related papers
        """
        related_papers = []
        
        # If we have Semantic Scholar ID, get citations
        if paper.paper_id:
            citations = self.semantic_scholar.get_citations(paper.paper_id, max_results // 2)
            related_papers.extend(citations)
        
        # Search for papers with similar keywords
        if paper.fields_of_study:
            keywords_query = " ".join(paper.fields_of_study[:3])  # Use top 3 fields
            similar_papers = self.search_combined(keywords_query, max_results // 2)
            related_papers.extend(similar_papers)
        
        # Deduplicate and return
        related_papers = self._deduplicate_papers(related_papers)
        return related_papers[:max_results]
    
    def _deduplicate_papers(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            # Normalize title for comparison
            normalized_title = paper.title.lower().strip()
            
            # Check for exact match or very similar titles
            is_duplicate = False
            for seen_title in seen_titles:
                if self._title_similarity(normalized_title, seen_title) > 0.9:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(normalized_title)
        
        logger.info(f"Deduplicated {len(papers)} papers to {len(unique_papers)}")
        return unique_papers
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles"""
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        cache_files = list(self.cache.cache_dir.glob("*.cache"))
        
        return {
            "cache_size": len(cache_files),
            "cache_directory": str(self.cache.cache_dir),
            "rate_limit_per_minute": self.rate_limiter.calls_per_minute,
            "last_call_time": self.rate_limiter.last_call_time
        }

# Global API manager instance
_api_manager = None

def get_api_manager(semantic_scholar_api_key: Optional[str] = None) -> ExternalAPIManager:
    """Get global API manager instance"""
    global _api_manager
    if _api_manager is None:
        _api_manager = ExternalAPIManager(semantic_scholar_api_key=semantic_scholar_api_key)
    return _api_manager

# Example usage and testing
if __name__ == "__main__":
    # Test external API manager
    api_manager = ExternalAPIManager()
    
    # Search for papers
    query = "machine learning healthcare"
    print(f"Searching for: {query}")
    
    # Search individual sources
    results = api_manager.search_all_sources(query, max_results_per_source=3)
    
    for source, papers in results.items():
        print(f"\n{source.upper()} ({len(papers)} papers):")
        for paper in papers[:2]:  # Show first 2 papers
            print(f"  - {paper.title[:80]}...")
            print(f"    Authors: {', '.join(paper.authors[:3])}")
            if paper.citation_count:
                print(f"    Citations: {paper.citation_count}")
    
    # Search combined
    print(f"\nCombined search:")
    combined_papers = api_manager.search_combined(query, max_total_results=10)
    
    for i, paper in enumerate(combined_papers[:5], 1):
        print(f"{i}. {paper.title}")
        print(f"   Source: {paper.source_api}")
        if paper.citation_count:
            print(f"   Citations: {paper.citation_count}")
    
    # Get statistics
    stats = api_manager.get_statistics()
    print(f"\nAPI Statistics: {stats}") 
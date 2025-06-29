"""
Advanced Citation and Reference Management System

Features:
- Multiple citation styles (APA, MLA, Chicago, IEEE, Nature, etc.)
- Integration with arXiv, CrossRef, and Semantic Scholar APIs
- BibTeX import/export
- Automatic citation formatting
- Duplicate detection and merging
- Citation validation and verification
"""

import re
import json
import requests
import bibtexparser
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import hashlib
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """Citation data structure"""
    title: str
    authors: List[str]
    year: str
    publication_type: str = "article"  # article, book, conference, etc.
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    citation_key: Optional[str] = None
    
    def __post_init__(self):
        """Generate citation key if not provided"""
        if not self.citation_key:
            self.citation_key = self._generate_citation_key()
    
    def _generate_citation_key(self) -> str:
        """Generate unique citation key"""
        # Use first author's last name + year + first word of title
        first_author = self.authors[0] if self.authors else "Unknown"
        last_name = first_author.split()[-1].lower()
        title_word = self.title.split()[0].lower() if self.title else "untitled"
        
        # Remove non-alphanumeric characters
        last_name = re.sub(r'[^a-z0-9]', '', last_name)
        title_word = re.sub(r'[^a-z0-9]', '', title_word)
        
        return f"{last_name}{self.year}{title_word}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class CitationFormatter:
    """Format citations in various academic styles"""
    
    @staticmethod
    def format_apa(citation: Citation) -> str:
        """Format citation in APA style"""
        authors = CitationFormatter._format_authors_apa(citation.authors)
        title = citation.title
        year = citation.year
        
        # Basic format: Authors (Year). Title.
        formatted = f"{authors} ({year}). {title}."
        
        # Add journal information if available
        if citation.journal:
            formatted = f"{authors} ({year}). {title}. *{citation.journal}*"
            
            if citation.volume:
                formatted += f", *{citation.volume}*"
                
            if citation.issue:
                formatted += f"({citation.issue})"
                
            if citation.pages:
                formatted += f", {citation.pages}"
            
            formatted += "."
        
        # Add DOI or URL if available
        if citation.doi:
            formatted += f" https://doi.org/{citation.doi}"
        elif citation.arxiv_id:
            formatted += f" arXiv:{citation.arxiv_id}"
        elif citation.url:
            formatted += f" {citation.url}"
        
        return formatted
    
    @staticmethod
    def format_mla(citation: Citation) -> str:
        """Format citation in MLA style"""
        authors = CitationFormatter._format_authors_mla(citation.authors)
        title = f'"{citation.title}"'
        
        formatted = f"{authors}. {title}"
        
        if citation.journal:
            formatted += f" *{citation.journal}*"
            
            if citation.volume:
                formatted += f", vol. {citation.volume}"
                
            if citation.issue:
                formatted += f", no. {citation.issue}"
            
            formatted += f", {citation.year}"
            
            if citation.pages:
                formatted += f", pp. {citation.pages}"
        else:
            formatted += f", {citation.year}"
        
        formatted += "."
        
        return formatted
    
    @staticmethod
    def format_chicago(citation: Citation) -> str:
        """Format citation in Chicago style"""
        authors = CitationFormatter._format_authors_chicago(citation.authors)
        title = f'"{citation.title}"'
        
        formatted = f"{authors}. {title}"
        
        if citation.journal:
            formatted += f" *{citation.journal}*"
            
            if citation.volume:
                formatted += f" {citation.volume}"
                
            if citation.issue:
                formatted += f", no. {citation.issue}"
            
            formatted += f" ({citation.year})"
            
            if citation.pages:
                formatted += f": {citation.pages}"
        else:
            formatted += f" ({citation.year})"
        
        formatted += "."
        
        return formatted
    
    @staticmethod
    def format_ieee(citation: Citation) -> str:
        """Format citation in IEEE style"""
        authors = CitationFormatter._format_authors_ieee(citation.authors)
        title = f'"{citation.title}"'
        
        formatted = f"{authors}, {title}"
        
        if citation.journal:
            formatted += f", *{citation.journal}*"
            
            if citation.volume:
                formatted += f", vol. {citation.volume}"
                
            if citation.issue:
                formatted += f", no. {citation.issue}"
            
            if citation.pages:
                formatted += f", pp. {citation.pages}"
            
            formatted += f", {citation.year}"
        else:
            formatted += f", {citation.year}"
        
        formatted += "."
        
        return formatted
    
    @staticmethod
    def format_nature(citation: Citation) -> str:
        """Format citation in Nature style"""
        authors = CitationFormatter._format_authors_nature(citation.authors)
        title = citation.title
        
        formatted = f"{authors}. {title}"
        
        if citation.journal:
            formatted += f" *{citation.journal}*"
            
            if citation.volume:
                formatted += f" **{citation.volume}**"
            
            if citation.pages:
                formatted += f", {citation.pages}"
            
            formatted += f" ({citation.year})"
        else:
            formatted += f" ({citation.year})"
        
        formatted += "."
        
        return formatted
    
    @staticmethod
    def _format_authors_apa(authors: List[str]) -> str:
        """Format authors for APA style"""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} & {authors[1]}"
        elif len(authors) <= 7:
            return ", ".join(authors[:-1]) + f", & {authors[-1]}"
        else:
            return ", ".join(authors[:6]) + ", ... " + authors[-1]
    
    @staticmethod
    def _format_authors_mla(authors: List[str]) -> str:
        """Format authors for MLA style"""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} and {authors[1]}"
        else:
            return f"{authors[0]} et al."
    
    @staticmethod
    def _format_authors_chicago(authors: List[str]) -> str:
        """Format authors for Chicago style"""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} and {authors[1]}"
        elif len(authors) <= 10:
            return ", ".join(authors[:-1]) + f", and {authors[-1]}"
        else:
            return ", ".join(authors[:7]) + ", et al."
    
    @staticmethod
    def _format_authors_ieee(authors: List[str]) -> str:
        """Format authors for IEEE style"""
        if not authors:
            return "Unknown Author"
        
        if len(authors) <= 6:
            return ", ".join(authors)
        else:
            return ", ".join(authors[:6]) + ", et al."
    
    @staticmethod
    def _format_authors_nature(authors: List[str]) -> str:
        """Format authors for Nature style"""
        if not authors:
            return "Unknown Author"
        
        if len(authors) <= 5:
            return ", ".join(authors)
        else:
            return f"{authors[0]} et al."

class ExternalAPIManager:
    """Manage external API integrations for citation data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research-Assistant-Agent/1.0 (mailto:research@example.com)'
        })
    
    def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arXiv for papers"""
        try:
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': query,
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                paper = self._parse_arxiv_entry(entry)
                if paper:
                    papers.append(paper)
            
            logger.info(f"Retrieved {len(papers)} papers from arXiv")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    def search_crossref(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search CrossRef for papers"""
        try:
            url = "https://api.crossref.org/works"
            params = {
                'query': query,
                'rows': max_results,
                'sort': 'relevance',
                'order': 'desc'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for item in data.get('message', {}).get('items', []):
                paper = self._parse_crossref_item(item)
                if paper:
                    papers.append(paper)
            
            logger.info(f"Retrieved {len(papers)} papers from CrossRef")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching CrossRef: {e}")
            return []
    
    def get_paper_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Get paper details by DOI"""
        try:
            url = f"https://api.crossref.org/works/{doi}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            paper = self._parse_crossref_item(data.get('message', {}))
            
            if paper:
                logger.info(f"Retrieved paper details for DOI: {doi}")
            
            return paper
            
        except Exception as e:
            logger.error(f"Error retrieving paper by DOI {doi}: {e}")
            return None
    
    def _parse_arxiv_entry(self, entry) -> Optional[Dict[str, Any]]:
        """Parse arXiv XML entry"""
        try:
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            title = entry.find('atom:title', ns)
            title = title.text.strip() if title is not None else ""
            
            # Get authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text.strip())
            
            # Get published date
            published = entry.find('atom:published', ns)
            year = ""
            if published is not None:
                year = published.text[:4]
            
            # Get arXiv ID
            arxiv_id = ""
            id_elem = entry.find('atom:id', ns)
            if id_elem is not None:
                arxiv_id = id_elem.text.split('/')[-1]
            
            # Get abstract
            summary = entry.find('atom:summary', ns)
            abstract = summary.text.strip() if summary is not None else ""
            
            return {
                'title': title,
                'authors': authors,
                'year': year,
                'arxiv_id': arxiv_id,
                'abstract': abstract,
                'source': 'arxiv'
            }
            
        except Exception as e:
            logger.error(f"Error parsing arXiv entry: {e}")
            return None
    
    def _parse_crossref_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse CrossRef item"""
        try:
            title = ""
            if 'title' in item and item['title']:
                title = item['title'][0]
            
            # Get authors
            authors = []
            if 'author' in item:
                for author in item['author']:
                    given = author.get('given', '')
                    family = author.get('family', '')
                    name = f"{given} {family}".strip()
                    if name:
                        authors.append(name)
            
            # Get publication year
            year = ""
            if 'published-print' in item:
                year = str(item['published-print']['date-parts'][0][0])
            elif 'published-online' in item:
                year = str(item['published-online']['date-parts'][0][0])
            
            # Get journal
            journal = ""
            if 'container-title' in item and item['container-title']:
                journal = item['container-title'][0]
            
            # Get DOI
            doi = item.get('DOI', '')
            
            # Get volume, issue, pages
            volume = item.get('volume', '')
            issue = item.get('issue', '')
            page = item.get('page', '')
            
            return {
                'title': title,
                'authors': authors,
                'year': year,
                'journal': journal,
                'volume': volume,
                'issue': issue,
                'pages': page,
                'doi': doi,
                'source': 'crossref'
            }
            
        except Exception as e:
            logger.error(f"Error parsing CrossRef item: {e}")
            return None

class CitationManager:
    """
    Comprehensive citation and reference management system
    """
    
    def __init__(self, bibliography_file: str = "bibliography.bib"):
        """
        Initialize citation manager
        
        Args:
            bibliography_file: Path to BibTeX bibliography file
        """
        self.bibliography_file = Path(bibliography_file)
        self.citations: Dict[str, Citation] = {}
        self.formatter = CitationFormatter()
        self.api_manager = ExternalAPIManager()
        
        # Load existing bibliography if it exists
        self.load_bibliography()
    
    def add_citation(self, citation: Citation) -> str:
        """
        Add citation to the collection
        
        Args:
            citation: Citation object to add
            
        Returns:
            Citation key
        """
        # Check for duplicates
        existing_key = self._find_duplicate(citation)
        if existing_key:
            logger.info(f"Duplicate citation found, merging with {existing_key}")
            self._merge_citations(existing_key, citation)
            return existing_key
        
        # Add new citation
        self.citations[citation.citation_key] = citation
        logger.info(f"Added citation: {citation.citation_key}")
        
        return citation.citation_key
    
    def create_citation_from_dict(self, data: Dict[str, Any]) -> Citation:
        """Create Citation object from dictionary"""
        return Citation(
            title=data.get('title', ''),
            authors=data.get('authors', []),
            year=str(data.get('year', data.get('published', ''))),
            publication_type=data.get('publication_type', 'article'),
            journal=data.get('journal', ''),
            volume=data.get('volume', ''),
            issue=data.get('issue', ''),
            pages=data.get('pages', ''),
            publisher=data.get('publisher', ''),
            doi=data.get('doi', ''),
            arxiv_id=data.get('arxiv_id', ''),
            url=data.get('url', ''),
            abstract=data.get('abstract', ''),
            keywords=data.get('keywords', [])
        )
    
    def search_and_add_papers(self, query: str, max_results: int = 10) -> List[str]:
        """
        Search external databases and add papers
        
        Args:
            query: Search query
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of citation keys for added papers
        """
        added_keys = []
        
        # Search arXiv
        arxiv_papers = self.api_manager.search_arxiv(query, max_results // 2)
        for paper_data in arxiv_papers:
            citation = self.create_citation_from_dict(paper_data)
            key = self.add_citation(citation)
            added_keys.append(key)
        
        # Search CrossRef
        crossref_papers = self.api_manager.search_crossref(query, max_results // 2)
        for paper_data in crossref_papers:
            citation = self.create_citation_from_dict(paper_data)
            key = self.add_citation(citation)
            added_keys.append(key)
        
        logger.info(f"Added {len(added_keys)} citations from search: {query}")
        return added_keys
    
    def format_citation(self, citation_key: str, style: str = "apa") -> str:
        """
        Format a citation in specified style
        
        Args:
            citation_key: Key of citation to format
            style: Citation style (apa, mla, chicago, ieee, nature)
            
        Returns:
            Formatted citation string
        """
        if citation_key not in self.citations:
            return f"[Citation not found: {citation_key}]"
        
        citation = self.citations[citation_key]
        
        if style.lower() == "apa":
            return self.formatter.format_apa(citation)
        elif style.lower() == "mla":
            return self.formatter.format_mla(citation)
        elif style.lower() == "chicago":
            return self.formatter.format_chicago(citation)
        elif style.lower() == "ieee":
            return self.formatter.format_ieee(citation)
        elif style.lower() == "nature":
            return self.formatter.format_nature(citation)
        else:
            return self.formatter.format_apa(citation)  # Default to APA
    
    def format_bibliography(self, style: str = "apa", citation_keys: List[str] = None) -> str:
        """
        Format complete bibliography
        
        Args:
            style: Citation style
            citation_keys: Specific citations to include (all if None)
            
        Returns:
            Formatted bibliography string
        """
        if citation_keys is None:
            citation_keys = list(self.citations.keys())
        
        # Sort citations alphabetically by first author's last name
        sorted_keys = sorted(citation_keys, key=lambda k: self.citations[k].authors[0].split()[-1].lower())
        
        bibliography = []
        for key in sorted_keys:
            formatted = self.format_citation(key, style)
            bibliography.append(formatted)
        
        return "\n\n".join(bibliography)
    
    def export_bibtex(self, output_file: str = None) -> str:
        """
        Export citations as BibTeX
        
        Args:
            output_file: Output file path (optional)
            
        Returns:
            BibTeX string
        """
        bibtex_entries = []
        
        for citation in self.citations.values():
            entry = self._citation_to_bibtex(citation)
            bibtex_entries.append(entry)
        
        bibtex_content = "\n\n".join(bibtex_entries)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(bibtex_content)
            logger.info(f"Exported BibTeX to: {output_file}")
        
        return bibtex_content
    
    def import_bibtex(self, bibtex_file: str) -> int:
        """
        Import citations from BibTeX file
        
        Args:
            bibtex_file: Path to BibTeX file
            
        Returns:
            Number of citations imported
        """
        try:
            with open(bibtex_file, 'r', encoding='utf-8') as f:
                bib_database = bibtexparser.load(f)
            
            imported_count = 0
            for entry in bib_database.entries:
                citation = self._bibtex_to_citation(entry)
                if citation:
                    self.add_citation(citation)
                    imported_count += 1
            
            logger.info(f"Imported {imported_count} citations from BibTeX file")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing BibTeX file {bibtex_file}: {e}")
            return 0
    
    def load_bibliography(self):
        """Load existing bibliography file"""
        if self.bibliography_file.exists():
            try:
                imported = self.import_bibtex(str(self.bibliography_file))
                logger.info(f"Loaded {imported} citations from bibliography file")
            except Exception as e:
                logger.error(f"Error loading bibliography: {e}")
    
    def save_bibliography(self):
        """Save current citations to bibliography file"""
        try:
            self.export_bibtex(str(self.bibliography_file))
            logger.info(f"Saved bibliography to {self.bibliography_file}")
        except Exception as e:
            logger.error(f"Error saving bibliography: {e}")
    
    def _find_duplicate(self, citation: Citation) -> Optional[str]:
        """Find duplicate citation by title similarity"""
        for key, existing in self.citations.items():
            # Simple duplicate detection based on title similarity
            if self._similarity_score(citation.title, existing.title) > 0.8:
                return key
        return None
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts"""
        # Simple similarity based on common words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_citations(self, existing_key: str, new_citation: Citation):
        """Merge new citation data with existing citation"""
        existing = self.citations[existing_key]
        
        # Update fields that are empty in existing citation
        if not existing.doi and new_citation.doi:
            existing.doi = new_citation.doi
        
        if not existing.abstract and new_citation.abstract:
            existing.abstract = new_citation.abstract
        
        if not existing.journal and new_citation.journal:
            existing.journal = new_citation.journal
        
        # Merge keywords
        if new_citation.keywords:
            existing_keywords = set(existing.keywords or [])
            new_keywords = set(new_citation.keywords)
            existing.keywords = list(existing_keywords.union(new_keywords))
    
    def _citation_to_bibtex(self, citation: Citation) -> str:
        """Convert Citation to BibTeX entry"""
        entry_type = "article" if citation.journal else "misc"
        
        fields = []
        fields.append(f'title = {{{citation.title}}}')
        
        if citation.authors:
            author_str = " and ".join(citation.authors)
            fields.append(f'author = {{{author_str}}}')
        
        fields.append(f'year = {{{citation.year}}}')
        
        if citation.journal:
            fields.append(f'journal = {{{citation.journal}}}')
        
        if citation.volume:
            fields.append(f'volume = {{{citation.volume}}}')
        
        if citation.issue:
            fields.append(f'number = {{{citation.issue}}}')
        
        if citation.pages:
            fields.append(f'pages = {{{citation.pages}}}')
        
        if citation.doi:
            fields.append(f'doi = {{{citation.doi}}}')
        
        if citation.arxiv_id:
            fields.append(f'note = {{arXiv:{citation.arxiv_id}}}')
        
        fields_str = ",\n  ".join(fields)
        return f"@{entry_type}{{{citation.citation_key},\n  {fields_str}\n}}"
    
    def _bibtex_to_citation(self, entry: Dict[str, Any]) -> Optional[Citation]:
        """Convert BibTeX entry to Citation"""
        try:
            title = entry.get('title', '').strip('{}')
            
            # Parse authors
            authors = []
            if 'author' in entry:
                author_str = entry['author']
                authors = [name.strip() for name in author_str.split(' and ')]
            
            year = entry.get('year', '')
            journal = entry.get('journal', '').strip('{}')
            volume = entry.get('volume', '').strip('{}')
            issue = entry.get('number', '').strip('{}')
            pages = entry.get('pages', '').strip('{}')
            doi = entry.get('doi', '').strip('{}')
            
            # Extract arXiv ID from note field
            arxiv_id = ""
            note = entry.get('note', '')
            if 'arXiv:' in note:
                arxiv_id = note.split('arXiv:')[1].strip('{}')
            
            citation = Citation(
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                volume=volume,
                issue=issue,
                pages=pages,
                doi=doi,
                arxiv_id=arxiv_id,
                citation_key=entry.get('ID', '')
            )
            
            return citation
            
        except Exception as e:
            logger.error(f"Error converting BibTeX entry: {e}")
            return None
    
    def get_citation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the citation collection"""
        total_citations = len(self.citations)
        
        # Count by publication type
        type_counts = {}
        year_counts = {}
        
        for citation in self.citations.values():
            pub_type = citation.publication_type
            type_counts[pub_type] = type_counts.get(pub_type, 0) + 1
            
            year = citation.year
            year_counts[year] = year_counts.get(year, 0) + 1
        
        return {
            "total_citations": total_citations,
            "publication_types": type_counts,
            "years": year_counts,
            "has_doi": sum(1 for c in self.citations.values() if c.doi),
            "has_arxiv": sum(1 for c in self.citations.values() if c.arxiv_id)
        }

# Global citation manager instance
_citation_manager = None

def get_citation_manager(bibliography_file: str = "bibliography.bib") -> CitationManager:
    """Get global citation manager instance"""
    global _citation_manager
    if _citation_manager is None:
        _citation_manager = CitationManager(bibliography_file)
    return _citation_manager

# Example usage and testing
if __name__ == "__main__":
    # Test citation manager
    cm = CitationManager("test_bibliography.bib")
    
    # Create sample citation
    citation = Citation(
        title="Machine Learning for Healthcare Applications",
        authors=["Dr. Jane Smith", "Dr. John Doe"],
        year="2024",
        journal="Journal of Medical AI",
        volume="10",
        issue="2",
        pages="123-145",
        doi="10.1234/jmai.2024.001"
    )
    
    # Add citation
    key = cm.add_citation(citation)
    print(f"Added citation with key: {key}")
    
    # Format in different styles
    styles = ["apa", "mla", "chicago", "ieee", "nature"]
    for style in styles:
        formatted = cm.format_citation(key, style)
        print(f"\n{style.upper()}: {formatted}")
    
    # Generate bibliography
    bibliography = cm.format_bibliography("apa")
    print(f"\nBibliography:\n{bibliography}")
    
    # Export BibTeX
    bibtex = cm.export_bibtex()
    print(f"\nBibTeX:\n{bibtex}")
    
    # Get statistics
    stats = cm.get_citation_statistics()
    print(f"\nStatistics: {stats}") 
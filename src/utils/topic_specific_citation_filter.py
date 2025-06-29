"""
Topic-Specific Citation Filtering System

This module provides intelligent filtering and ranking of citations based on:
- Research domain and topic analysis
- Medical/healthcare specialty matching
- Citation quality and relevance scoring
- Source prioritization (peer-reviewed > arXiv)
"""

import re
import json
import requests
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TopicAnalysis:
    """Analysis of research topic/hypothesis"""
    primary_domain: str
    secondary_domains: List[str]
    key_terms: List[str]
    medical_specialty: Optional[str]
    research_type: str  # empirical, review, theoretical, clinical
    target_conditions: List[str]
    methodologies: List[str]

class TopicSpecificCitationFilter:
    """
    Intelligent citation filtering based on research topic analysis
    """
    
    def __init__(self):
        """Initialize the citation filter"""
        
        # Define domain-specific keywords and sources
        self.domain_keywords = {
            'alzheimers': {
                'primary': ['alzheimer', 'dementia', 'cognitive decline', 'memory loss', 'MMSE', 'APOE', 'beta-amyloid', 'tau protein'],
                'secondary': ['neurodegeneration', 'aging', 'brain imaging', 'biomarkers', 'mild cognitive impairment'],
                'methodologies': ['clinical trial', 'longitudinal study', 'neuropsychological assessment', 'brain imaging'],
                'preferred_sources': ['nature', 'lancet', 'jama', 'nejm', 'alzheimers & dementia', 'journal of alzheimers disease']
            },
            'diabetes': {
                'primary': ['diabetes', 'glucose', 'insulin', 'glycemic control', 'HbA1c', 'blood sugar'],
                'secondary': ['metabolic syndrome', 'cardiovascular disease', 'diabetic complications'],
                'methodologies': ['clinical trial', 'meta-analysis', 'cohort study'],
                'preferred_sources': ['diabetes care', 'diabetologia', 'jama', 'nejm', 'diabetes']
            },
            'cardiovascular': {
                'primary': ['cardiovascular', 'heart disease', 'hypertension', 'blood pressure', 'cardiac'],
                'secondary': ['stroke', 'myocardial infarction', 'atherosclerosis', 'coronary artery'],
                'methodologies': ['randomized controlled trial', 'cohort study', 'meta-analysis'],
                'preferred_sources': ['circulation', 'jama cardiology', 'european heart journal', 'nejm']
            },
            'oncology': {
                'primary': ['cancer', 'tumor', 'oncology', 'malignancy', 'chemotherapy', 'radiation therapy'],
                'secondary': ['metastasis', 'biomarkers', 'immunotherapy', 'targeted therapy'],
                'methodologies': ['clinical trial', 'survival analysis', 'biomarker study'],
                'preferred_sources': ['jama oncology', 'nature cancer', 'lancet oncology', 'cancer cell']
            },
            'psychiatry': {
                'primary': ['depression', 'anxiety', 'mental health', 'psychiatric', 'mood disorder', 'PHQ-9'],
                'secondary': ['psychotherapy', 'antidepressant', 'cognitive behavioral therapy'],
                'methodologies': ['randomized controlled trial', 'systematic review', 'clinical assessment'],
                'preferred_sources': ['jama psychiatry', 'lancet psychiatry', 'american journal of psychiatry']
            },
            'infectious_disease': {
                'primary': ['infection', 'bacteria', 'virus', 'pathogen', 'antimicrobial', 'antibiotic'],
                'secondary': ['resistance', 'epidemiology', 'outbreak', 'vaccination'],
                'methodologies': ['epidemiological study', 'clinical trial', 'surveillance'],
                'preferred_sources': ['lancet infectious diseases', 'clinical infectious diseases', 'jama']
            }
        }
        
        # Research methodology keywords
        self.methodology_keywords = {
            'machine_learning': ['machine learning', 'neural network', 'deep learning', 'AI', 'artificial intelligence'],
            'clinical_trial': ['randomized controlled trial', 'RCT', 'clinical trial', 'intervention study'],
            'observational': ['cohort study', 'case-control', 'cross-sectional', 'longitudinal'],
            'meta_analysis': ['meta-analysis', 'systematic review', 'literature review'],
            'biomarker': ['biomarker', 'diagnostic marker', 'prognostic marker', 'molecular marker']
        }
        
        # Source quality rankings (higher = better)
        self.source_quality_scores = {
            'nature': 10, 'science': 10, 'cell': 10, 'nejm': 10, 'lancet': 10,
            'jama': 9, 'bmj': 9, 'plos medicine': 8, 'nature medicine': 9,
            'scientific reports': 7, 'plos one': 6, 'biorxiv': 4, 'arxiv': 3,
            'medrxiv': 4, 'preprint': 2
        }
    
    def analyze_topic(self, hypothesis: str, research_context: str = "") -> TopicAnalysis:
        """
        Analyze research topic to extract domain and key terms
        
        Args:
            hypothesis: Research hypothesis text
            research_context: Additional context about the research
            
        Returns:
            TopicAnalysis object with extracted information
        """
        text = (hypothesis + " " + research_context).lower()
        
        # Identify primary domain
        primary_domain = "general"
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords['primary']:
                score += text.count(keyword.lower()) * 3  # Primary keywords weighted higher
            for keyword in keywords['secondary']:
                score += text.count(keyword.lower()) * 1
            domain_scores[domain] = score
        
        if domain_scores:
            primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            if domain_scores[primary_domain] == 0:
                primary_domain = "general"
        
        # Extract secondary domains
        secondary_domains = [
            domain for domain, score in sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[1:3]
            if score > 0
        ]
        
        # Extract key terms
        key_terms = []
        if primary_domain in self.domain_keywords:
            for keyword in self.domain_keywords[primary_domain]['primary']:
                if keyword.lower() in text:
                    key_terms.append(keyword)
        
        # Identify medical specialty
        specialty_mapping = {
            'alzheimers': 'neurology',
            'diabetes': 'endocrinology',
            'cardiovascular': 'cardiology',
            'oncology': 'oncology',
            'psychiatry': 'psychiatry',
            'infectious_disease': 'infectious_disease'
        }
        medical_specialty = specialty_mapping.get(primary_domain)
        
        # Identify research type
        research_type = "empirical"
        if any(term in text for term in ['review', 'meta-analysis', 'systematic review']):
            research_type = "review"
        elif any(term in text for term in ['theory', 'theoretical', 'framework']):
            research_type = "theoretical"
        elif any(term in text for term in ['clinical', 'patient', 'treatment']):
            research_type = "clinical"
        
        # Extract target conditions
        target_conditions = []
        condition_patterns = [
            r'(\w+\s+disease)', r'(\w+\s+disorder)', r'(\w+\s+syndrome)',
            r'(alzheimer)', r'(diabetes)', r'(hypertension)', r'(cancer)', r'(depression)'
        ]
        for pattern in condition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            target_conditions.extend([match.lower() for match in matches])
        
        # Extract methodologies
        methodologies = []
        for method_type, keywords in self.methodology_keywords.items():
            if any(keyword.lower() in text for keyword in keywords):
                methodologies.append(method_type)
        
        return TopicAnalysis(
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            key_terms=key_terms,
            medical_specialty=medical_specialty,
            research_type=research_type,
            target_conditions=list(set(target_conditions)),
            methodologies=methodologies
        )
    
    def search_topic_specific_citations(self, topic_analysis: TopicAnalysis, max_citations: int = 15) -> List[str]:
        """
        Search for topic-specific citations and return formatted citation strings
        
        Args:
            topic_analysis: Analysis of the research topic
            max_citations: Maximum number of citations to return
            
        Returns:
            List of formatted citation strings
        """
        papers = []
        
        # Build search queries based on topic analysis
        search_queries = []
        
        # Primary domain query
        if topic_analysis.primary_domain in self.domain_keywords:
            primary_terms = self.domain_keywords[topic_analysis.primary_domain]['primary'][:3]
            search_queries.append(" AND ".join(primary_terms))
        
        # Key terms query
        if topic_analysis.key_terms:
            search_queries.append(" AND ".join(topic_analysis.key_terms[:3]))
        
        # Target conditions query
        if topic_analysis.target_conditions:
            search_queries.append(" OR ".join(topic_analysis.target_conditions[:2]))
        
        # Medical specialty query
        if topic_analysis.medical_specialty:
            specialty_terms = {
                'neurology': ['neurology', 'neurological', 'brain'],
                'cardiology': ['cardiology', 'cardiovascular', 'heart'],
                'endocrinology': ['endocrinology', 'metabolism', 'hormone'],
                'oncology': ['oncology', 'cancer', 'tumor'],
                'psychiatry': ['psychiatry', 'mental health', 'psychology'],
                'infectious_disease': ['infectious disease', 'infection', 'pathogen']
            }
            if topic_analysis.medical_specialty in specialty_terms:
                terms = specialty_terms[topic_analysis.medical_specialty]
                search_queries.append(" OR ".join(terms))
        
        # Search each query and collect results
        all_papers = []
        for query in search_queries[:2]:  # Limit to top 2 queries to avoid API limits
            try:
                query_papers = self._search_external_apis(query, max_results=8)
                all_papers.extend(query_papers)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
        
        # Remove duplicates and score papers
        unique_papers = self._remove_duplicate_papers(all_papers)
        scored_papers = []
        
        for paper in unique_papers:
            relevance_score = self._calculate_relevance_score(paper, topic_analysis)
            if relevance_score > 0.2:  # Only include papers with decent relevance
                scored_papers.append((relevance_score, paper))
        
        # Sort by relevance score
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        
        # Format citations
        citations = []
        for score, paper in scored_papers[:max_citations]:
            citation = self.format_citation(paper)
            citations.append(citation)
        
        # If we don't have enough citations, add domain-specific fallback citations
        if len(citations) < 5:
            fallback_citations = self._generate_fallback_citations(topic_analysis)
            citations.extend(fallback_citations[:5-len(citations)])
        
        return citations
    
    def _search_external_apis(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search external APIs for papers"""
        papers = []
        
        # Search CrossRef (peer-reviewed sources)
        try:
            crossref_papers = self._search_crossref(query, max_results // 2)
            papers.extend(crossref_papers)
        except Exception as e:
            logger.warning(f"CrossRef search failed: {e}")
        
        # Search arXiv (preprints, but lower priority)
        try:
            arxiv_papers = self._search_arxiv(query, max_results // 2)
            papers.extend(arxiv_papers)
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
        
        return papers
    
    def _search_crossref(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search CrossRef for peer-reviewed papers"""
        try:
            url = "https://api.crossref.org/works"
            params = {
                'query': query,
                'rows': max_results,
                'sort': 'relevance',
                'order': 'desc',
                'filter': 'type:journal-article'  # Focus on journal articles
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for item in data.get('message', {}).get('items', []):
                paper = self._parse_crossref_item(item)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching CrossRef: {e}")
            return []
    
    def _search_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arXiv for preprints"""
        try:
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f"all:{query}",
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                paper = self._parse_arxiv_entry(entry)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    def _parse_crossref_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse CrossRef item to standard format"""
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
            
            # Abstract (if available)
            abstract = item.get('abstract', '')
            
            # Publication year
            year = ""
            if 'published-print' in item:
                year = str(item['published-print']['date-parts'][0][0])
            elif 'published-online' in item:
                year = str(item['published-online']['date-parts'][0][0])
            
            # Journal
            journal = ""
            if 'container-title' in item and item['container-title']:
                journal = item['container-title'][0]
            
            # DOI
            doi = item.get('DOI', '')
            
            # Volume, issue, pages
            volume = item.get('volume', '')
            issue = item.get('issue', '')
            pages = item.get('page', '')
            
            return {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'year': year,
                'journal': journal,
                'volume': volume,
                'issue': issue,
                'pages': pages,
                'doi': doi,
                'source': 'crossref',
                'source_quality': self.source_quality_scores.get(journal.lower(), 3)
            }
            
        except Exception as e:
            logger.error(f"Error parsing CrossRef item: {e}")
            return None
    
    def _parse_arxiv_entry(self, entry) -> Optional[Dict[str, Any]]:
        """Parse arXiv entry to standard format"""
        try:
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
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
            year = ""
            if published_elem is not None:
                year = published_elem.text[:4]
            
            # arXiv ID
            id_elem = entry.find('atom:id', ns)
            arxiv_id = ""
            if id_elem is not None:
                arxiv_id = id_elem.text.split('/')[-1]
            
            return {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'year': year,
                'arxiv_id': arxiv_id,
                'source': 'arxiv',
                'source_quality': 3  # Lower quality for preprints
            }
            
        except Exception as e:
            logger.error(f"Error parsing arXiv entry: {e}")
            return None
    
    def _calculate_relevance_score(self, paper: Dict[str, Any], topic_analysis: TopicAnalysis) -> float:
        """Calculate relevance score for a paper based on topic analysis"""
        score = 0.0
        
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        full_text = f"{title} {abstract}"
        
        # Domain-specific keyword matching
        if topic_analysis.primary_domain in self.domain_keywords:
            domain_keywords = self.domain_keywords[topic_analysis.primary_domain]
            
            # Primary keywords (high weight)
            for keyword in domain_keywords['primary']:
                if keyword.lower() in full_text:
                    score += 0.3
            
            # Secondary keywords (medium weight)
            for keyword in domain_keywords['secondary']:
                if keyword.lower() in full_text:
                    score += 0.1
        
        # Key terms matching
        for term in topic_analysis.key_terms:
            if term.lower() in full_text:
                score += 0.2
        
        # Target conditions matching
        for condition in topic_analysis.target_conditions:
            if condition.lower() in full_text:
                score += 0.25
        
        # Source quality bonus
        source_quality = paper.get('source_quality', 3)
        score += source_quality / 50  # Normalize to 0-0.2 range
        
        return min(score, 1.0)
    
    def _remove_duplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title similarity"""
        if not papers:
            return []
        
        unique_papers = []
        seen_titles = []
        
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            if not title:
                continue
            
            # Check for similarity with existing titles
            is_duplicate = False
            for seen_title in seen_titles:
                similarity = self._calculate_title_similarity(title, seen_title)
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.append(title)
        
        return unique_papers
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles"""
        # Simple word-based similarity
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_fallback_citations(self, topic_analysis: TopicAnalysis) -> List[str]:
        """Generate high-quality fallback citations for specific domains"""
        fallback_citations = {
            'alzheimers': [
                "Jack, C. R., et al. (2018). NIA-AA Research Framework: Toward a biological definition of Alzheimer's disease. Alzheimer's & Dementia, 14(4), 535-562. https://doi.org/10.1016/j.jalz.2018.02.018",
                "Scheltens, P., et al. (2021). Alzheimer's disease. The Lancet, 397(10284), 1577-1590. https://doi.org/10.1016/S0140-6736(20)32205-4",
                "Livingston, G., et al. (2020). Dementia prevention, intervention, and care: 2020 report of the Lancet Commission. The Lancet, 396(10248), 413-446. https://doi.org/10.1016/S0140-6736(20)30367-6",
                "McKhann, G. M., et al. (2011). The diagnosis of dementia due to Alzheimer's disease. Alzheimer's & Dementia, 7(3), 263-269. https://doi.org/10.1016/j.jalz.2011.03.005"
            ],
            'diabetes': [
                "American Diabetes Association (2023). Standards of Medical Care in Diabetes—2023. Diabetes Care, 46(Supplement 1), S1-S291. https://doi.org/10.2337/dc23-Sint",
                "Zheng, Y., et al. (2018). Global aetiology and epidemiology of type 2 diabetes mellitus and its complications. Nature Reviews Endocrinology, 14(2), 88-98. https://doi.org/10.1038/nrendo.2017.151",
                "Davies, M. J., et al. (2022). Management of hyperglycemia in type 2 diabetes, 2022. Diabetes Care, 45(11), 2753-2786. https://doi.org/10.2337/dci22-0034"
            ],
            'cardiovascular': [
                "Virani, S. S., et al. (2021). Heart disease and stroke statistics—2021 update. Circulation, 143(8), e254-e743. https://doi.org/10.1161/CIR.0000000000000950",
                "Arnett, D. K., et al. (2019). 2019 ACC/AHA Guideline on the Primary Prevention of Cardiovascular Disease. Circulation, 140(11), e596-e646. https://doi.org/10.1161/CIR.0000000000000678",
                "Roth, G. A., et al. (2020). Global burden of cardiovascular diseases and risk factors, 1990–2019. Journal of the American College of Cardiology, 76(25), 2982-3021. https://doi.org/10.1016/j.jacc.2020.11.010"
            ],
            'oncology': [
                "Siegel, R. L., et al. (2023). Cancer statistics, 2023. CA: A Cancer Journal for Clinicians, 73(1), 233-254. https://doi.org/10.3322/caac.21763",
                "Hanahan, D. (2022). Hallmarks of cancer: new dimensions. Cancer Discovery, 12(1), 31-46. https://doi.org/10.1158/2159-8290.CD-21-1059",
                "Sung, H., et al. (2021). Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA: A Cancer Journal for Clinicians, 71(3), 209-249. https://doi.org/10.3322/caac.21660"
            ]
        }
        
        domain = topic_analysis.primary_domain
        if domain in fallback_citations:
            return fallback_citations[domain]
        else:
            # Generic high-quality healthcare citations
            return [
                "Collins, F. S., & Varmus, H. (2015). A new initiative on precision medicine. New England Journal of Medicine, 372(9), 793-795. https://doi.org/10.1056/NEJMp1500523",
                "Topol, E. J. (2019). High-performance medicine: the convergence of human and artificial intelligence. Nature Medicine, 25(1), 44-56. https://doi.org/10.1038/s41591-018-0300-7",
                "Chen, J. H., & Asch, S. M. (2017). Machine learning and prediction in medicine—beyond the peak of inflated expectations. New England Journal of Medicine, 376(26), 2507-2509. https://doi.org/10.1056/NEJMp1702071"
            ]
    
    def format_citation(self, paper: Dict[str, Any], style: str = "apa") -> str:
        """Format paper as academic citation"""
        authors = paper.get('authors', [])
        title = paper.get('title', 'Unknown title')
        year = paper.get('year', '2024')
        journal = paper.get('journal', '')
        volume = paper.get('volume', '')
        issue = paper.get('issue', '')
        pages = paper.get('pages', '')
        doi = paper.get('doi', '')
        arxiv_id = paper.get('arxiv_id', '')
        
        # Format authors
        if len(authors) > 6:
            author_str = f"{authors[0]} et al."
        elif len(authors) > 3:
            author_str = f"{', '.join(authors[:3])} et al."
        else:
            author_str = ", ".join(authors)
        
        # APA style citation
        citation = f"{author_str} ({year}). {title}."
        
        if journal:
            citation += f" *{journal}*"
            if volume:
                citation += f", *{volume}*"
            if issue:
                citation += f"({issue})"
            if pages:
                citation += f", {pages}"
            citation += "."
        
        if doi:
            citation += f" https://doi.org/{doi}"
        elif arxiv_id:
            citation += f" arXiv:{arxiv_id}"
        
        return citation

# Global instance
_citation_filter = None

def get_citation_filter() -> TopicSpecificCitationFilter:
    """Get global citation filter instance"""
    global _citation_filter
    if _citation_filter is None:
        _citation_filter = TopicSpecificCitationFilter()
    return _citation_filter 
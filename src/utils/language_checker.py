"""
Advanced Language and Grammar Checking System

Features:
- Grammar and spell checking using LanguageTool
- Writing style analysis and suggestions
- Academic writing enhancement
- Plagiarism detection (basic)
- Readability analysis
- Technical writing validation
"""

import re
import string
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import logging
from dataclasses import dataclass
from collections import Counter
import language_tool_python
import requests
import json

logger = logging.getLogger(__name__)

@dataclass
class LanguageIssue:
    """Represents a language/grammar issue"""
    category: str  # grammar, spelling, style, etc.
    severity: str  # low, medium, high
    message: str
    suggestion: str
    start_pos: int
    end_pos: int
    rule_id: str
    context: str

@dataclass
class ReadabilityMetrics:
    """Readability analysis metrics"""
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog_index: float
    avg_sentence_length: float
    avg_word_length: float
    complex_words_ratio: float
    passive_voice_ratio: float

@dataclass
class StyleAnalysis:
    """Writing style analysis"""
    avg_sentence_length: float
    sentence_length_variance: float
    word_variety_ratio: float  # unique words / total words
    academic_tone_score: float
    technical_term_density: float
    citation_density: float

class LanguageChecker:
    """
    Comprehensive language and grammar checking system
    """
    
    def __init__(self, language: str = "en-US"):
        """
        Initialize language checker
        
        Args:
            language: Language code (e.g., 'en-US', 'en-GB')
        """
        self.language = language
        self.tool = None
        self._initialize_language_tool()
        
        # Academic writing patterns
        self.academic_patterns = self._load_academic_patterns()
        self.technical_terms = self._load_technical_terms()
        
    def _initialize_language_tool(self):
        """Initialize LanguageTool"""
        try:
            self.tool = language_tool_python.LanguageTool(self.language)
            logger.info(f"Initialized LanguageTool for {self.language}")
        except Exception as e:
            logger.error(f"Error initializing LanguageTool: {e}")
            self.tool = None
    
    def _load_academic_patterns(self) -> Dict[str, List[str]]:
        """Load academic writing patterns and conventions"""
        return {
            "weak_verbs": [
                "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "get", "got"
            ],
            "transition_words": [
                "however", "furthermore", "moreover", "consequently", 
                "therefore", "nevertheless", "additionally", "similarly",
                "conversely", "meanwhile", "subsequently", "accordingly"
            ],
            "hedging_words": [
                "might", "could", "may", "possibly", "likely", "probably",
                "appears", "seems", "suggests", "indicates", "potentially"
            ],
            "filler_words": [
                "very", "really", "quite", "rather", "somewhat", "fairly",
                "pretty", "basically", "literally", "actually", "obviously"
            ],
            "academic_connectors": [
                "in contrast", "on the other hand", "as a result", "in conclusion",
                "to summarize", "in addition", "for example", "specifically",
                "in particular", "that is to say", "in other words"
            ]
        }
    
    def _load_technical_terms(self) -> Set[str]:
        """Load technical terms for various domains"""
        return {
            # Computer Science / AI
            "algorithm", "machine learning", "neural network", "deep learning",
            "artificial intelligence", "optimization", "classification", "regression",
            "clustering", "supervised learning", "unsupervised learning",
            
            # Medical / Healthcare
            "diagnosis", "prognosis", "pathology", "epidemiology", "biomarker",
            "clinical trial", "randomized controlled trial", "meta-analysis",
            
            # Statistics / Data Science
            "statistical significance", "p-value", "confidence interval",
            "correlation", "regression analysis", "hypothesis testing",
            "cross-validation", "feature selection", "dimensionality reduction",
            
            # Research Methodology
            "methodology", "systematic review", "literature review",
            "qualitative", "quantitative", "mixed methods", "validation"
        }
    
    def check_grammar_and_spelling(self, text: str) -> List[LanguageIssue]:
        """
        Check grammar and spelling using LanguageTool
        
        Args:
            text: Text to check
            
        Returns:
            List of language issues found
        """
        issues = []
        
        if not self.tool:
            logger.warning("LanguageTool not available, skipping grammar check")
            return issues
        
        try:
            matches = self.tool.check(text)
            
            for match in matches:
                issue = LanguageIssue(
                    category=self._categorize_issue(match.category),
                    severity=self._assess_severity(match),
                    message=match.message,
                    suggestion=", ".join(match.replacements[:3]) if match.replacements else "",
                    start_pos=match.offset,
                    end_pos=match.offset + match.errorLength,
                    rule_id=match.ruleId,
                    context=match.context
                )
                issues.append(issue)
            
            logger.info(f"Found {len(issues)} language issues")
            return issues
            
        except Exception as e:
            logger.error(f"Error during grammar check: {e}")
            return issues
    
    def analyze_readability(self, text: str) -> ReadabilityMetrics:
        """
        Analyze text readability using various metrics
        
        Args:
            text: Text to analyze
            
        Returns:
            ReadabilityMetrics object
        """
        sentences = self._split_sentences(text)
        words = self._split_words(text)
        syllables = sum(self._count_syllables(word) for word in words)
        
        # Basic metrics
        num_sentences = len(sentences)
        num_words = len(words)
        num_syllables = syllables
        
        if num_sentences == 0 or num_words == 0:
            return ReadabilityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        avg_sentence_length = num_words / num_sentences
        avg_syllables_per_word = num_syllables / num_words
        avg_word_length = sum(len(word) for word in words) / num_words
        
        # Complex words (3+ syllables)
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        complex_words_ratio = complex_words / num_words
        
        # Flesch Reading Ease
        flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        
        # Gunning Fog Index
        gunning_fog_index = 0.4 * (avg_sentence_length + (100 * complex_words_ratio))
        
        # Passive voice detection
        passive_voice_ratio = self._detect_passive_voice(text)
        
        return ReadabilityMetrics(
            flesch_reading_ease=max(0, min(100, flesch_reading_ease)),
            flesch_kincaid_grade=max(0, flesch_kincaid_grade),
            gunning_fog_index=max(0, gunning_fog_index),
            avg_sentence_length=avg_sentence_length,
            avg_word_length=avg_word_length,
            complex_words_ratio=complex_words_ratio,
            passive_voice_ratio=passive_voice_ratio
        )
    
    def analyze_writing_style(self, text: str) -> StyleAnalysis:
        """
        Analyze writing style for academic appropriateness
        
        Args:
            text: Text to analyze
            
        Returns:
            StyleAnalysis object
        """
        sentences = self._split_sentences(text)
        words = self._split_words(text)
        
        if not sentences or not words:
            return StyleAnalysis(0, 0, 0, 0, 0, 0)
        
        # Sentence length analysis
        sentence_lengths = [len(self._split_words(sentence)) for sentence in sentences]
        avg_sentence_length = statistics.mean(sentence_lengths)
        sentence_length_variance = statistics.variance(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Word variety
        unique_words = len(set(word.lower() for word in words))
        word_variety_ratio = unique_words / len(words)
        
        # Academic tone analysis
        academic_tone_score = self._calculate_academic_tone(text, words)
        
        # Technical term density
        technical_term_count = sum(1 for word in words if word.lower() in self.technical_terms)
        technical_term_density = technical_term_count / len(words)
        
        # Citation density (rough estimate)
        citation_density = self._estimate_citation_density(text)
        
        return StyleAnalysis(
            avg_sentence_length=avg_sentence_length,
            sentence_length_variance=sentence_length_variance,
            word_variety_ratio=word_variety_ratio,
            academic_tone_score=academic_tone_score,
            technical_term_density=technical_term_density,
            citation_density=citation_density
        )
    
    def suggest_improvements(self, text: str) -> Dict[str, List[str]]:
        """
        Suggest specific improvements for academic writing
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of improvement suggestions by category
        """
        suggestions = {
            "grammar": [],
            "style": [],
            "clarity": [],
            "academic_tone": [],
            "word_choice": []
        }
        
        # Grammar suggestions from LanguageTool
        grammar_issues = self.check_grammar_and_spelling(text)
        for issue in grammar_issues[:5]:  # Top 5 issues
            suggestions["grammar"].append(f"{issue.message} - Suggested: {issue.suggestion}")
        
        # Style suggestions
        style_analysis = self.analyze_writing_style(text)
        readability = self.analyze_readability(text)
        
        # Sentence length suggestions
        if style_analysis.avg_sentence_length > 25:
            suggestions["style"].append("Consider breaking down long sentences for better readability")
        elif style_analysis.avg_sentence_length < 15:
            suggestions["style"].append("Consider combining short sentences for better flow")
        
        # Readability suggestions
        if readability.flesch_reading_ease < 30:
            suggestions["clarity"].append("Text is quite difficult to read. Consider simplifying language")
        elif readability.flesch_reading_ease > 70:
            suggestions["academic_tone"].append("Text may be too simple for academic writing")
        
        # Passive voice suggestions
        if readability.passive_voice_ratio > 0.3:
            suggestions["style"].append("Consider reducing passive voice usage for more engaging writing")
        
        # Word choice suggestions
        words = self._split_words(text.lower())
        filler_word_count = sum(1 for word in words if word in self.academic_patterns["filler_words"])
        if filler_word_count > len(words) * 0.05:  # More than 5% filler words
            suggestions["word_choice"].append("Consider removing filler words (very, really, quite, etc.)")
        
        # Academic tone suggestions
        if style_analysis.academic_tone_score < 0.5:
            suggestions["academic_tone"].append("Consider using more formal academic language")
            suggestions["academic_tone"].append("Add more transition words and academic connectors")
        
        return suggestions
    
    def enhance_academic_writing(self, text: str) -> str:
        """
        Automatically enhance text for academic writing
        
        Args:
            text: Text to enhance
            
        Returns:
            Enhanced text
        """
        enhanced_text = text
        
        # Remove excessive filler words
        for filler in self.academic_patterns["filler_words"]:
            # Remove when used excessively
            pattern = rf'\b{filler}\b'
            count = len(re.findall(pattern, enhanced_text, re.IGNORECASE))
            if count > 3:  # Keep max 3 instances
                enhanced_text = re.sub(pattern, '', enhanced_text, count=count-3, flags=re.IGNORECASE)
        
        # Improve weak verb phrases
        weak_replacements = {
            "is able to": "can",
            "in order to": "to",
            "due to the fact that": "because",
            "at this point in time": "now",
            "it is important to note that": "",
            "it should be noted that": "",
        }
        
        for weak, strong in weak_replacements.items():
            enhanced_text = re.sub(
                re.escape(weak), strong, enhanced_text, flags=re.IGNORECASE
            )
        
        # Clean up spacing
        enhanced_text = re.sub(r'\s+', ' ', enhanced_text)
        enhanced_text = enhanced_text.strip()
        
        return enhanced_text
    
    def check_plagiarism_indicators(self, text: str) -> Dict[str, Any]:
        """
        Basic plagiarism detection indicators
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with plagiarism indicators
        """
        sentences = self._split_sentences(text)
        
        # Check for inconsistent writing style
        sentence_complexities = []
        for sentence in sentences:
            words = self._split_words(sentence)
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            sentence_complexities.append(avg_word_length)
        
        style_variance = statistics.variance(sentence_complexities) if len(sentence_complexities) > 1 else 0
        
        # Check for unusual formatting patterns
        formatting_issues = 0
        if re.search(r'[^\x00-\x7F]', text):  # Non-ASCII characters
            formatting_issues += 1
        
        if len(re.findall(r'  +', text)) > 5:  # Excessive double spaces
            formatting_issues += 1
        
        # Check for citation density inconsistency
        citation_pattern = r'\([^)]*\d{4}[^)]*\)|[\[\]][^]]*\d{4}[^]]*[\]\]]'
        citations_per_paragraph = []
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(paragraph.strip()) > 100:  # Only count substantial paragraphs
                citations = len(re.findall(citation_pattern, paragraph))
                citations_per_paragraph.append(citations)
        
        citation_variance = statistics.variance(citations_per_paragraph) if len(citations_per_paragraph) > 1 else 0
        
        return {
            "style_variance": style_variance,
            "formatting_issues": formatting_issues,
            "citation_variance": citation_variance,
            "risk_level": "high" if style_variance > 2 or formatting_issues > 2 else "low"
        }
    
    def _categorize_issue(self, category: str) -> str:
        """Categorize LanguageTool issue"""
        category_map = {
            "TYPOS": "spelling",
            "GRAMMAR": "grammar",
            "STYLE": "style",
            "PUNCTUATION": "punctuation",
            "CASING": "capitalization"
        }
        return category_map.get(category, "other")
    
    def _assess_severity(self, match) -> str:
        """Assess severity of language issue"""
        if hasattr(match, 'category'):
            if match.category in ["TYPOS", "GRAMMAR"]:
                return "high"
            elif match.category in ["STYLE", "PUNCTUATION"]:
                return "medium"
        return "low"
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_words(self, text: str) -> List[str]:
        """Split text into words"""
        # Remove punctuation and split
        text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
        return [word for word in text_no_punct.split() if word]
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _detect_passive_voice(self, text: str) -> float:
        """Detect passive voice usage ratio"""
        # Simple passive voice detection
        passive_patterns = [
            r'\b(is|are|was|were|being|been)\s+\w+ed\b',
            r'\b(is|are|was|were|being|been)\s+\w+en\b',
        ]
        
        sentences = self._split_sentences(text)
        passive_count = 0
        
        for sentence in sentences:
            for pattern in passive_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    passive_count += 1
                    break
        
        return passive_count / len(sentences) if sentences else 0
    
    def _calculate_academic_tone(self, text: str, words: List[str]) -> float:
        """Calculate academic tone score (0-1)"""
        score = 0.0
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Count academic indicators
        transition_count = sum(1 for word in words if word.lower() in self.academic_patterns["transition_words"])
        hedging_count = sum(1 for word in words if word.lower() in self.academic_patterns["hedging_words"])
        
        # Boost score for academic features
        score += (transition_count / total_words) * 0.3
        score += (hedging_count / total_words) * 0.2
        
        # Check for formal language patterns
        formal_patterns = [
            r'\bthis study\b', r'\bour research\b', r'\bthe findings\b',
            r'\bin conclusion\b', r'\bfurthermore\b', r'\bmoreover\b'
        ]
        
        formal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formal_patterns)
        score += min(0.3, formal_count / 10)  # Cap at 0.3
        
        # Penalize informal language
        informal_patterns = [
            r'\bcan\'t\b', r'\bdon\'t\b', r'\bwon\'t\b', r'\bisn\'t\b',
            r'\bi think\b', r'\bi believe\b', r'\bstuff\b', r'\bthings\b'
        ]
        
        informal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in informal_patterns)
        score -= min(0.2, informal_count / 5)  # Penalty up to 0.2
        
        return max(0.0, min(1.0, score))
    
    def _estimate_citation_density(self, text: str) -> float:
        """Estimate citation density in text"""
        # Common citation patterns
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2024)
            r'\[[^\]]*\d{4}[^\]]*\]',  # [Author, 2024]
            r'\([^)]*et al[^)]*\)',  # (Smith et al.)
            r'\[[^\]]*et al[^\]]*\]',  # [Smith et al.]
        ]
        
        total_citations = 0
        for pattern in citation_patterns:
            total_citations += len(re.findall(pattern, text))
        
        words = self._split_words(text)
        return total_citations / len(words) if words else 0

    def generate_report(self, text: str) -> Dict[str, Any]:
        """
        Generate comprehensive language analysis report
        
        Args:
            text: Text to analyze
            
        Returns:
            Comprehensive analysis report
        """
        # Perform all analyses
        grammar_issues = self.check_grammar_and_spelling(text)
        readability = self.analyze_readability(text)
        style_analysis = self.analyze_writing_style(text)
        suggestions = self.suggest_improvements(text)
        plagiarism_check = self.check_plagiarism_indicators(text)
        
        # Calculate overall scores
        grammar_score = max(0, 100 - len(grammar_issues) * 5)  # Deduct 5 points per issue
        readability_score = min(100, max(0, readability.flesch_reading_ease))
        style_score = style_analysis.academic_tone_score * 100
        
        overall_score = (grammar_score + readability_score + style_score) / 3
        
        return {
            "overall_score": round(overall_score, 1),
            "grammar_score": round(grammar_score, 1),
            "readability_score": round(readability_score, 1),
            "style_score": round(style_score, 1),
            "text_stats": {
                "word_count": len(self._split_words(text)),
                "sentence_count": len(self._split_sentences(text)),
                "avg_sentence_length": round(readability.avg_sentence_length, 1),
                "avg_word_length": round(readability.avg_word_length, 1)
            },
            "issues": {
                "grammar": [
                    {
                        "message": issue.message,
                        "suggestion": issue.suggestion,
                        "severity": issue.severity,
                        "category": issue.category
                    }
                    for issue in grammar_issues[:10]  # Top 10 issues
                ],
                "total_issues": len(grammar_issues)
            },
            "readability": {
                "flesch_reading_ease": round(readability.flesch_reading_ease, 1),
                "grade_level": round(readability.flesch_kincaid_grade, 1),
                "complexity": "High" if readability.gunning_fog_index > 16 else "Medium" if readability.gunning_fog_index > 12 else "Low"
            },
            "style": {
                "academic_tone": round(style_analysis.academic_tone_score, 2),
                "word_variety": round(style_analysis.word_variety_ratio, 2),
                "technical_density": round(style_analysis.technical_term_density, 3),
                "passive_voice_ratio": round(readability.passive_voice_ratio, 2)
            },
            "suggestions": suggestions,
            "plagiarism_check": plagiarism_check
        }

# Global language checker instance
_language_checker = None

def get_language_checker(language: str = "en-US") -> LanguageChecker:
    """Get global language checker instance"""
    global _language_checker
    if _language_checker is None:
        _language_checker = LanguageChecker(language)
    return _language_checker

# Example usage and testing
if __name__ == "__main__":
    # Test language checker
    checker = LanguageChecker()
    
    sample_text = """
    This paper presents a novel approach to machine learning that have significant implications 
    for healthcare applications. The methodology we developed combines deep learning techniques 
    with traditional statistical methods to achieve superior performance in disease detection tasks.
    Our results demonstrates that the proposed approach can identify early-stage diseases with 
    high accuracy, which is very important for patient outcomes. Furthermore, the model shows 
    robustness across different datasets and patient populations.
    """
    
    print("Language Analysis Report")
    print("=" * 50)
    
    # Generate comprehensive report
    report = checker.generate_report(sample_text)
    
    print(f"Overall Score: {report['overall_score']}/100")
    print(f"Grammar Score: {report['grammar_score']}/100")
    print(f"Readability Score: {report['readability_score']}/100")
    print(f"Style Score: {report['style_score']}/100")
    
    print(f"\nText Statistics:")
    for key, value in report['text_stats'].items():
        print(f"  {key}: {value}")
    
    print(f"\nTop Grammar Issues:")
    for issue in report['issues']['grammar'][:3]:
        print(f"  - {issue['message']}")
        if issue['suggestion']:
            print(f"    Suggestion: {issue['suggestion']}")
    
    print(f"\nStyle Suggestions:")
    for category, suggestions in report['suggestions'].items():
        if suggestions:
            print(f"  {category}:")
            for suggestion in suggestions[:2]:
                print(f"    - {suggestion}") 
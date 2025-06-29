"""
Semantic Model Selection System
Uses embedding similarity and dynamic topic inference for autonomous model discovery
Optimized for medical topics and research
"""
import os
import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from huggingface_hub import list_models
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class SemanticModelSelector:
    """
    Autonomous semantic model selection using embedding similarity and topic inference.
    Optimized for medical and healthcare research topics.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )
        self.inappropriate_indicators = {
            'nsfw', 'adult', 'sexual', 'explicit', 'porn', 'xxx', 'erotic', 'nude', 'naked',
            'violence', 'violent', 'gore', 'blood', 'weapon', 'gun', 'knife', 'bomb',
            'hate', 'racism', 'discrimination', 'toxic', 'offensive', 'harmful',
            'gaming', 'game', 'entertainment', 'fun', 'joke', 'meme', 'cartoon', 'anime',
            'celebrity', 'famous', 'person', 'individual', 'private'
        }
        
        # Medical-specific terms and patterns
        self.medical_terms = {
            'diseases': ['alzheimer', 'dementia', 'cancer', 'diabetes', 'heart', 'brain', 'tumor', 'disease', 'syndrome', 'disorder'],
            'medical_fields': ['medical', 'clinical', 'healthcare', 'biomedical', 'pathology', 'radiology', 'cardiology', 'neurology', 'oncology'],
            'data_types': ['mri', 'ct', 'xray', 'scan', 'image', 'ecg', 'eeg', 'ultrasound', 'microscopy'],
            'tasks': ['diagnosis', 'detection', 'classification', 'segmentation', 'prediction', 'screening', 'analysis']
        }
        
        self.inferred_task_types = []  # Initialize inferred task types
        
    def discover_relevant_models(self, hypothesis: str, max_models: int = 5) -> List[Dict[str, Any]]:
        """
        Discover semantically relevant models with focus on medical applications.
        """
        try:
            print(f"   🔍 Performing medical-focused semantic analysis...")
            
            # Step 1: Extract semantic features from hypothesis
            semantic_features = self._extract_semantic_features(hypothesis)
            
            # Step 2: Generate medical-focused search queries
            search_queries = self._generate_medical_queries(semantic_features)
            
            # Step 3: Search and collect candidate models
            candidate_models = self._search_medical_models(search_queries, max_models * 4)
            
            # Step 4: Semantic filtering and ranking
            relevant_models = self._semantic_filter_and_rank(candidate_models, semantic_features, hypothesis)
            
            # Step 5: Final selection with diversity
            final_models = self._select_diverse_relevant_models(relevant_models, max_models)
            
            print(f"   ✅ Selected {len(final_models)} medically relevant models")
            for i, model in enumerate(final_models, 1):
                relevance = model.get('semantic_score', 0)
                reason = model.get('selection_reason', 'Medical semantic match')
                print(f"      {i}. {model['id']} (score: {relevance:.3f}) - {reason}")
            
            return final_models
            
        except Exception as e:
            print(f"   ⚠️ Error in medical model discovery: {e}")
            return self._fallback_medical_models(hypothesis)

    def _extract_semantic_features(self, hypothesis: str) -> Dict[str, Any]:
        """
        Extract semantic features with emphasis on medical terminology.
        """
        # Clean and normalize text
        clean_text = self._clean_text(hypothesis)
        
        # Extract key terms with medical focus
        key_terms = self._extract_medical_key_terms(clean_text)
        
        # Infer task type dynamically
        task_type = self._infer_medical_task_type(clean_text, key_terms)
        
        # Infer data modality with medical focus
        data_modality = self._infer_medical_data_modality(clean_text, key_terms)
        
        # Extract medical domain concepts
        domain_concepts = self._extract_medical_domain_concepts(clean_text, key_terms)
        
        # Generate semantic embeddings
        semantic_vector = self._generate_semantic_vector(clean_text, key_terms)
        
        return {
            'original_text': hypothesis,
            'clean_text': clean_text,
            'key_terms': key_terms,
            'task_type': task_type,
            'data_modality': data_modality,
            'domain_concepts': domain_concepts,
            'semantic_vector': semantic_vector,
            'medical_relevance': self._assess_medical_relevance(clean_text, key_terms)
        }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for semantic analysis."""
        text = text.lower()
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_medical_key_terms(self, text: str) -> List[str]:
        """Extract key terms with emphasis on medical terminology."""
        
        key_terms = set()
        
        # Medical-specific patterns
        medical_patterns = [
            r'\b\w*(?:medical|clinical|healthcare|biomedical|pathology|radiology|diagnosis|disease|syndrome|disorder)\w*\b',
            r'\b\w*(?:alzheimer|dementia|cancer|diabetes|heart|brain|tumor|mri|ct|xray|scan)\w*\b',
            r'\b\w*(?:detection|classification|segmentation|prediction|screening|analysis)\w*\b',
            r'\b\w*(?:neural|deep|machine|artificial|learning|model|algorithm|network)\w*\b',
            r'\b[A-Z]{2,}\b',  # Acronyms (common in medical field)
            r'\b\w+(?:-\w+)+\b',  # Hyphenated terms
        ]
        
        # Extract using medical patterns
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_terms.update([match.lower() for match in matches])
        
        # Check for medical terms from our predefined lists
        words = text.split()
        for word in words:
            for category, terms in self.medical_terms.items():
                if any(term in word.lower() for term in terms):
                    key_terms.add(word.lower())
        
        # Extract using TF-IDF on individual words
        if len(words) > 1:
            try:
                corpus = [text, ' '.join(words[:len(words)//2]), ' '.join(words[len(words)//2:])]
                tfidf_matrix = self.vectorizer.fit_transform(corpus)
                feature_names = self.vectorizer.get_feature_names_out()
                
                tfidf_scores = tfidf_matrix[0].toarray()[0]
                top_indices = np.argsort(tfidf_scores)[-10:]
                
                for idx in top_indices:
                    if tfidf_scores[idx] > 0:
                        key_terms.add(feature_names[idx])
            except:
                key_terms.update([word for word in words if len(word) > 4])
        
        # Filter out common stop words
        stop_words = {'using', 'with', 'from', 'that', 'this', 'they', 'them', 'their', 'there', 'where', 'when', 'what', 'which', 'will', 'would', 'could', 'should'}
        key_terms = {term for term in key_terms if term not in stop_words and len(term) > 2}
        
        return list(key_terms)[:15]

    def _infer_medical_task_type(self, text: str, key_terms: List[str]) -> str:
        """Infer task type with medical context."""
        
        medical_task_patterns = {
            'medical_image_classification': ['classify', 'categorize', 'image', 'mri', 'ct', 'xray', 'scan', 'medical', 'diagnosis'],
            'disease_detection': ['detect', 'identify', 'screen', 'disease', 'disorder', 'syndrome', 'pathology'],
            'medical_segmentation': ['segment', 'region', 'mask', 'boundary', 'anatomical', 'organ', 'tissue'],
            'clinical_prediction': ['predict', 'forecast', 'prognosis', 'outcome', 'risk', 'survival'],
            'biomedical_text_classification': ['text', 'document', 'clinical', 'notes', 'report', 'literature'],
            'drug_discovery': ['drug', 'compound', 'molecule', 'pharmaceutical', 'therapy', 'treatment'],
            'genomic_analysis': ['gene', 'genetic', 'dna', 'rna', 'genomic', 'sequence', 'mutation'],
            'medical_nlp': ['nlp', 'natural language', 'text', 'clinical notes', 'medical records']
        }
        
        task_scores = {}
        all_terms = text.split() + key_terms
        
        for task, indicators in medical_task_patterns.items():
            score = 0
            for indicator in indicators:
                for term in all_terms:
                    if indicator in term.lower():
                        score += 1
            task_scores[task] = score
        
        # Store the highest scoring task types
        self.inferred_task_types = [task for task, score in task_scores.items() if score > 0]
        
        if task_scores and max(task_scores.values()) > 0:
            return max(task_scores, key=task_scores.get)
        
        return 'medical_classification'  # Default for medical domain

    def _infer_medical_data_modality(self, text: str, key_terms: List[str]) -> List[str]:
        """Infer data modality with medical context."""
        
        medical_modality_patterns = {
            'medical_image': ['image', 'mri', 'ct', 'xray', 'scan', 'ultrasound', 'microscopy', 'pathology', 'radiology'],
            'clinical_text': ['text', 'notes', 'report', 'record', 'document', 'clinical', 'nlp'],
            'time_series': ['time', 'temporal', 'ecg', 'eeg', 'monitoring', 'vital', 'signal'],
            'tabular': ['tabular', 'structured', 'demographic', 'lab', 'test', 'measurement'],
            'genomic': ['gene', 'genetic', 'dna', 'rna', 'sequence', 'genomic'],
            'multimodal': ['multimodal', 'multi-modal', 'fusion', 'combined']
        }
        
        detected_modalities = []
        all_terms = text.split() + key_terms
        
        for modality, indicators in medical_modality_patterns.items():
            for indicator in indicators:
                if any(indicator in term.lower() for term in all_terms):
                    detected_modalities.append(modality)
                    break
        
        return detected_modalities if detected_modalities else ['medical_image']  # Default

    def _extract_medical_domain_concepts(self, text: str, key_terms: List[str]) -> List[str]:
        """Extract medical domain concepts."""
        
        domain_concepts = set()
        all_terms = text.split() + key_terms
        
        # Check against medical term categories
        for term in all_terms:
            for category, medical_terms in self.medical_terms.items():
                if any(med_term in term.lower() for med_term in medical_terms):
                    domain_concepts.add(category)
                    domain_concepts.add(term.lower())
        
        # Add specific medical concepts based on patterns
        medical_concept_patterns = {
            'neurodegenerative': ['alzheimer', 'dementia', 'parkinson', 'huntington', 'als', 'neurodegenerative'],
            'cardiovascular': ['heart', 'cardiac', 'cardiovascular', 'ecg', 'ekg', 'blood pressure'],
            'oncology': ['cancer', 'tumor', 'oncology', 'malignant', 'benign', 'metastasis'],
            'radiology': ['mri', 'ct', 'xray', 'ultrasound', 'scan', 'imaging', 'radiology'],
            'pathology': ['pathology', 'biopsy', 'histology', 'microscopy', 'tissue'],
            'genetics': ['genetic', 'gene', 'dna', 'rna', 'genomic', 'mutation', 'hereditary']
        }
        
        for concept, indicators in medical_concept_patterns.items():
            if any(indicator in text.lower() for indicator in indicators):
                domain_concepts.add(concept)
        
        return list(domain_concepts)

    def _assess_medical_relevance(self, text: str, key_terms: List[str]) -> float:
        """Assess how medically relevant the text is."""
        
        medical_score = 0.0
        all_terms = text.split() + key_terms
        
        # Count medical terms
        for term in all_terms:
            for category, medical_terms in self.medical_terms.items():
                if any(med_term in term.lower() for med_term in medical_terms):
                    medical_score += 1.0
        
        # Bonus for medical-specific patterns
        medical_patterns = [
            r'\b(?:medical|clinical|healthcare|biomedical)\b',
            r'\b(?:diagnosis|treatment|therapy|patient)\b',
            r'\b(?:disease|disorder|syndrome|condition)\b',
            r'\b(?:mri|ct|xray|ultrasound|scan)\b'
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                medical_score += 2.0
        
        # Normalize by text length
        return min(medical_score / max(len(all_terms), 1), 1.0)

    def _generate_semantic_vector(self, text: str, key_terms: List[str]) -> np.ndarray:
        """Generate semantic vector representation."""
        try:
            combined_text = text + ' ' + ' '.join(key_terms)
            vector = self.vectorizer.fit_transform([combined_text])
            return vector.toarray()[0]
        except:
            return np.zeros(100)  # Fallback vector

    def _generate_medical_queries(self, semantic_features: Dict[str, Any]) -> List[str]:
        """Generate search queries optimized for medical models."""
        
        queries = []
        
        # Start with simple, effective search terms
        key_terms = semantic_features['key_terms'][:3]  # Top 3 terms only
        task_type = semantic_features['task_type']
        
        # Simple primary query - just key medical terms
        if semantic_features['medical_relevance'] > 0.2:
            # Use simple medical terms that are likely to match
            simple_medical_terms = []
            for term in key_terms:
                if any(med_word in term.lower() for med_word in ['medical', 'clinical', 'disease', 'brain', 'mri', 'alzheimer']):
                    simple_medical_terms.append(term)
            
            if simple_medical_terms:
                queries.append(' '.join(simple_medical_terms[:2]))  # Max 2 terms
            else:
                queries.append('medical')  # Fallback to simple 'medical'
        
        # Add broader search terms
        queries.extend([
            'medical classification',
            'biomedical',
            'clinical',
            'healthcare'
        ])
        
        # Add specific disease terms if present
        disease_terms = ['alzheimer', 'cancer', 'diabetes', 'heart', 'brain']
        for term in key_terms:
            for disease in disease_terms:
                if disease in term.lower():
                    queries.append(disease)
                    break
        
        # Add modality-specific terms
        for modality in semantic_features['data_modality']:
            if modality == 'medical_image':
                queries.append('medical image')
            elif modality == 'clinical_text':
                queries.append('clinical text')
        
        # Remove duplicates and limit
        unique_queries = []
        for query in queries:
            if query and query not in unique_queries:
                unique_queries.append(query)
        
        return unique_queries[:3]  # Limit to 3 queries for efficiency

    def _search_medical_models(self, queries: List[str], max_candidates: int) -> List[Dict[str, Any]]:
        """Search for medical models using optimized parameters."""
        
        all_candidates = []
        
        for query in queries:
            try:
                print(f"      Searching with query: '{query}'")
                
                # Use direct list_models parameters (no ModelFilter)
                models = list(list_models(
                    search=query,
                    limit=max_candidates // len(queries) + 5,
                    sort="downloads",
                    direction=-1
                ))
                
                for model in models:
                    metadata = self._get_model_metadata(model)
                    
                    # Apply medical relevance filtering
                    if self._is_medically_relevant_model(metadata, query):
                        all_candidates.append(metadata)
                        
            except Exception as e:
                print(f"      ⚠️ Error searching with query '{query}': {e}")
                continue
        
        # If no medically relevant models found, try with more general search
        if not all_candidates:
            print("      🔄 No medical models found, trying general search...")
            try:
                # Fallback to simple search without strict medical filtering
                general_models = list(list_models(
                    search="classification",
                    limit=max_candidates,
                    sort="downloads",
                    direction=-1
                ))
                
                for model in general_models[:5]:  # Take top 5 general models
                    metadata = self._get_model_metadata(model)
                    if not self._is_inappropriate_model(metadata):
                        all_candidates.append(metadata)
                
            except Exception as e:
                print(f"      ⚠️ Error in general search: {e}")
        
        # Remove duplicates based on model ID
        unique_candidates = []
        seen_ids = set()
        for candidate in all_candidates:
            if candidate['id'] not in seen_ids:
                unique_candidates.append(candidate)
                seen_ids.add(candidate['id'])
        
        print(f"      Found {len(unique_candidates)} unique candidates")
        return unique_candidates[:max_candidates]

    def _is_medically_relevant_model(self, metadata: Dict[str, Any], query: str) -> bool:
        """Check if a model is medically relevant."""
        
        model_text = self._get_model_text(metadata).lower()
        query_terms = query.lower().split()
        
        # Check for inappropriate content first
        if self._is_inappropriate_model(metadata):
            return False
        
        # Medical relevance scoring
        medical_score = 0
        
        # Check for medical terms in model text
        for category, terms in self.medical_terms.items():
            for term in terms:
                if term in model_text:
                    medical_score += 1
        
        # Check for query terms (more lenient)
        for term in query_terms:
            if term in model_text:
                medical_score += 0.5
        
        # Bonus for medical-specific tags
        tags = metadata.get('tags', [])
        medical_tags = ['medical', 'clinical', 'healthcare', 'biomedical', 'radiology', 'pathology']
        for tag in tags:
            if any(med_tag in tag.lower() for med_tag in medical_tags):
                medical_score += 2
        
        # Very lenient threshold - accept if any relevance found
        return medical_score >= 0.25  # Much lower threshold

    def _is_inappropriate_model(self, metadata: Dict[str, Any]) -> bool:
        """Check if a model is inappropriate."""
        
        model_id = metadata.get('id', '').lower()
        description = metadata.get('description', '').lower()
        tags = [tag.lower() for tag in metadata.get('tags', [])]
        
        # Check for inappropriate keywords
        for keyword in self.inappropriate_indicators:
            if (keyword in model_id or 
                keyword in description or 
                any(keyword in tag for tag in tags)):
                return True
        
        return False

    def _get_model_metadata(self, model) -> Dict[str, Any]:
        """Extract comprehensive metadata from a model."""
        
        model_info = {
            'id': getattr(model, 'id', 'unknown'),
            'downloads': getattr(model, 'downloads', 0),
            'tags': getattr(model, 'tags', []),
            'pipeline_tag': getattr(model, 'pipeline_tag', 'unknown'),
            'library': getattr(model, 'library_name', 'transformers'),
            'created_at': getattr(model, 'created_at', None),
            'last_modified': getattr(model, 'last_modified', None)
        }
        
        # Generate description from model attributes
        description_parts = []
        
        # Model name analysis
        model_name = model_info['id'].split('/')[-1] if '/' in model_info['id'] else model_info['id']
        description_parts.append(model_name.replace('-', ' ').replace('_', ' '))
        
        # Tags analysis
        if model_info['tags']:
            description_parts.extend([tag.replace('-', ' ').replace('_', ' ') for tag in model_info['tags'][:5]])
        
        # Pipeline tag
        if model_info['pipeline_tag'] != 'unknown':
            description_parts.append(model_info['pipeline_tag'].replace('-', ' ').replace('_', ' '))
        
        model_info['description'] = ' '.join(description_parts)
        
        return model_info

    def _semantic_filter_and_rank(self, candidates: List[Dict[str, Any]], semantic_features: Dict[str, Any], hypothesis: str) -> List[Dict[str, Any]]:
        """Filter and rank models using semantic similarity with medical focus."""
        
        relevant_models = []
        
        for model in candidates:
            try:
                # Calculate semantic similarity
                semantic_score = self._calculate_medical_semantic_similarity(model, semantic_features, hypothesis)
                
                # Filter out low-relevance models
                if semantic_score < 0.1:
                    continue
                
                # Add quality bonus
                quality_bonus = self._calculate_quality_score(model)
                
                # Medical relevance bonus
                medical_bonus = self._calculate_medical_relevance_bonus(model, semantic_features)
                
                # Calculate final score
                final_score = semantic_score + quality_bonus + medical_bonus
                
                # Generate selection reason
                selection_reason = self._generate_medical_selection_reason(model, semantic_features, semantic_score)
                
                model['semantic_score'] = final_score
                model['base_semantic_score'] = semantic_score
                model['quality_bonus'] = quality_bonus
                model['medical_bonus'] = medical_bonus
                model['selection_reason'] = selection_reason
                
                relevant_models.append(model)
                
            except Exception as e:
                print(f"         ⚠️ Error evaluating {model.get('id', 'unknown')}: {e}")
                continue
        
        # Sort by semantic score
        relevant_models.sort(key=lambda x: x['semantic_score'], reverse=True)
        
        print(f"      ✅ Filtered to {len(relevant_models)} medically relevant models")
        return relevant_models

    def _calculate_medical_semantic_similarity(self, model: Dict[str, Any], semantic_features: Dict[str, Any], hypothesis: str) -> float:
        """Calculate semantic similarity with medical emphasis."""
        
        score = 0.0
        model_text = self._get_model_text(model).lower()
        model_terms = model_text.split()
        
        # Medical term matching (higher weight)
        for category, terms in self.medical_terms.items():
            for term in terms:
                if term in model_text:
                    score += 0.5  # Higher weight for medical terms
        
        # Key term matching
        key_terms = [term.lower() for term in semantic_features['key_terms']]
        for term in key_terms:
            if term in model_text:
                score += 0.3
            # Partial matching
            for model_term in model_terms:
                if term in model_term or model_term in term:
                    score += 0.1
        
        # Task type matching
        task_type = semantic_features['task_type'].lower()
        if task_type in model_text:
            score += 0.4
        
        # Data modality matching
        for modality in semantic_features['data_modality']:
            if modality.lower() in model_text:
                score += 0.2
        
        # Domain concept matching
        for concept in semantic_features['domain_concepts']:
            if concept.lower() in model_text:
                score += 0.15
        
        # Medical relevance bonus
        score += semantic_features['medical_relevance'] * 0.3
        
        return min(score, 1.0)

    def _calculate_quality_score(self, model: Dict[str, Any]) -> float:
        """Calculate model quality score."""
        
        quality_score = 0.0
        
        # Download count (normalized)
        downloads = model.get('downloads', 0)
        if downloads > 1000:
            quality_score += min(np.log10(downloads) / 6, 0.3)  # Max 0.3 for downloads
        
        # Library preference (transformers is well-supported)
        library = model.get('library', '').lower()
        if library in ['transformers', 'pytorch', 'tensorflow']:
            quality_score += 0.1
        
        # Recent activity bonus
        if model.get('last_modified'):
            quality_score += 0.05
        
        return quality_score

    def _calculate_medical_relevance_bonus(self, model: Dict[str, Any], semantic_features: Dict[str, Any]) -> float:
        """Calculate bonus for medical relevance."""
        
        bonus = 0.0
        model_text = self._get_model_text(model).lower()
        tags = [tag.lower() for tag in model.get('tags', [])]
        
        # Medical tags bonus
        medical_tags = ['medical', 'clinical', 'healthcare', 'biomedical', 'radiology', 'pathology', 'diagnosis']
        for tag in tags:
            if any(med_tag in tag for med_tag in medical_tags):
                bonus += 0.2
        
        # Medical terms in model name/description
        for category, terms in self.medical_terms.items():
            for term in terms:
                if term in model_text:
                    bonus += 0.1
        
        return min(bonus, 0.5)  # Cap at 0.5

    def _generate_medical_selection_reason(self, model: Dict[str, Any], semantic_features: Dict[str, Any], semantic_score: float) -> str:
        """Generate selection reason with medical context."""
        
        reasons = []
        model_text = self._get_model_text(model).lower()
        
        # Check for medical relevance
        medical_matches = []
        for category, terms in self.medical_terms.items():
            for term in terms:
                if term in model_text:
                    medical_matches.append(term)
        
        if medical_matches:
            reasons.append(f"Medical relevance: {', '.join(medical_matches[:3])}")
        
        # Task type match
        if semantic_features['task_type'].lower() in model_text:
            reasons.append(f"Task match: {semantic_features['task_type']}")
        
        # Quality indicators
        downloads = model.get('downloads', 0)
        if downloads > 1000:
            reasons.append(f"Popular model ({downloads:,} downloads)")
        
        if not reasons:
            reasons.append("Semantic similarity match")
        
        return "; ".join(reasons)

    def _get_model_text(self, model: Dict[str, Any]) -> str:
        """Get searchable text from model metadata."""
        
        text_parts = []
        
        # Model ID
        text_parts.append(model['id'])
        
        # Description
        if 'description' in model:
            text_parts.append(model['description'])
        
        # Tags
        if model.get('tags'):
            text_parts.extend(model['tags'])
        
        # Pipeline tag
        if model.get('pipeline_tag', 'unknown') != 'unknown':
            text_parts.append(model['pipeline_tag'])
        
        return ' '.join(text_parts)

    def _select_diverse_relevant_models(self, models: List[Dict[str, Any]], max_models: int) -> List[Dict[str, Any]]:
        """Select diverse relevant models."""
        
        if len(models) <= max_models:
            return models
        
        selected = []
        remaining = models.copy()
        
        # Always include the top model
        if remaining:
            selected.append(remaining.pop(0))
        
        # Select diverse models
        while len(selected) < max_models and remaining:
            best_candidate = None
            best_diversity_score = -1
            
            for candidate in remaining:
                # Calculate diversity score
                diversity_score = self._calculate_diversity_score(candidate, selected)
                combined_score = candidate['semantic_score'] * 0.7 + diversity_score * 0.3
                
                if combined_score > best_diversity_score:
                    best_diversity_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected

    def _calculate_diversity_score(self, candidate: Dict[str, Any], selected: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for model selection."""
        
        if not selected:
            return 1.0
        
        candidate_text = self._get_model_text(candidate).lower()
        
        min_similarity = 1.0
        for selected_model in selected:
            selected_text = self._get_model_text(selected_model).lower()
            
            # Simple word overlap similarity
            candidate_words = set(candidate_text.split())
            selected_words = set(selected_text.split())
            
            if candidate_words and selected_words:
                overlap = len(candidate_words.intersection(selected_words))
                similarity = overlap / len(candidate_words.union(selected_words))
                min_similarity = min(min_similarity, similarity)
        
        return 1.0 - min_similarity  # Higher score for less similar models

    def _fallback_medical_models(self, hypothesis: str) -> List[Dict[str, Any]]:
        """Fallback medical models when search fails."""
        
        print("   🔄 Using fallback medical models...")
        
        fallback_models = [
                {
                'id': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                'pipeline_tag': 'fill-mask',
                'downloads': 50000,
                'tags': ['biomedical', 'pubmed', 'medical', 'nlp'],
                'description': 'Biomedical BERT model trained on PubMed abstracts and full texts',
                'semantic_score': 0.8,
                'selection_reason': 'Biomedical language model fallback'
            },
            {
                'id': 'emilyalsentzer/Bio_ClinicalBERT',
                'pipeline_tag': 'fill-mask',
                'downloads': 30000,
                'tags': ['clinical', 'biomedical', 'medical', 'bert'],
                'description': 'Clinical BERT model for biomedical text processing',
                'semantic_score': 0.75,
                'selection_reason': 'Clinical text processing fallback'
            },
            {
                'id': 'medicalai/ClinicalBERT',
                'pipeline_tag': 'text-classification',
                'downloads': 20000,
                'tags': ['medical', 'clinical', 'classification'],
                'description': 'Medical text classification model',
                    'semantic_score': 0.7,
                'selection_reason': 'Medical classification fallback'
                }
        ]
        
        return fallback_models[:3] 
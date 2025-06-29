#!/usr/bin/env python3
"""
Enhanced Hypothesis Generator for Any Research Domain

Generates sophisticated, research-gap-focused hypotheses that:
- Target underexplored problems
- Specify prediction/analysis targets
- Are feasible with structured datasets
- Address genuine research limitations
- Explain significance and impact
"""

import openai
from typing import Dict, Any, List, Optional, Tuple
import json
import re

class EnhancedHypothesisGenerator:
    """Generate sophisticated, domain-specific research hypotheses that address real research gaps."""
    
    def __init__(self, openai_client):
        self.client = openai_client
        
    def generate_research_hypothesis(self, user_topic: str, dataset_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a sophisticated research hypothesis for any domain.
        
        Args:
            user_topic: The research topic provided by the user
            dataset_info: Optional information about available dataset
            
        Returns:
            Dict containing hypothesis, gap analysis, impact, and methodology
        """
        
        # Step 1: Analyze the domain and identify research gaps
        domain_analysis = self._analyze_research_domain(user_topic)
        
        # Step 2: Generate hypothesis that addresses identified gaps
        hypothesis_data = self._generate_gap_focused_hypothesis(user_topic, domain_analysis, dataset_info)
        
        # Step 3: Validate and refine the hypothesis
        refined_hypothesis = self._validate_and_refine_hypothesis(hypothesis_data, dataset_info)
        
        return refined_hypothesis
    
    def _analyze_research_domain(self, topic: str) -> Dict[str, Any]:
        """Analyze the research domain to identify gaps and opportunities."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are an expert research analyst. Analyze the given topic and identify:
                    1. Current state of research in this field
                    2. Known limitations or gaps in existing studies
                    3. Underexplored problems or questions
                    4. Methodological challenges that need addressing
                    5. Practical applications that lack sufficient research
                    6. Interdisciplinary opportunities
                    
                    Focus on REAL, SPECIFIC gaps rather than generic statements."""},
                    {"role": "user", "content": f"Analyze the research landscape for: {topic}. Identify 3-5 specific research gaps or limitations that represent genuine opportunities for novel research. Be specific about what's missing or inadequately studied."}
                ],
                max_tokens=600,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Parse the analysis to extract key gaps
            gaps = self._extract_research_gaps(analysis)
            
            return {
                "full_analysis": analysis,
                "identified_gaps": gaps,
                "domain": self._extract_domain_from_topic(topic)
            }
            
        except Exception as e:
            print(f"Error analyzing domain: {e}")
            return {
                "full_analysis": f"Research in {topic} presents several methodological and practical challenges.",
                "identified_gaps": ["Limited methodological approaches", "Insufficient real-world validation"],
                "domain": "interdisciplinary"
            }
    
    def _generate_gap_focused_hypothesis(self, topic: str, domain_analysis: Dict, dataset_info: Optional[Dict]) -> Dict[str, Any]:
        """Generate a hypothesis that specifically addresses identified research gaps."""
        
        gaps = domain_analysis.get("identified_gaps", [])
        analysis = domain_analysis.get("full_analysis", "")
        
        # Prepare dataset context
        dataset_context = ""
        if dataset_info:
            dataset_context = f"""
            Available dataset characteristics:
            - Size: {dataset_info.get('shape', 'Unknown')}
            - Features: {dataset_info.get('columns', [])}
            - Target variable possibilities: {dataset_info.get('target_variable', 'Various')}
            """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are a world-class research scientist. Generate a sophisticated research hypothesis that:

                    REQUIREMENTS:
                    1. Addresses a SPECIFIC research gap (not generic)
                    2. Clearly specifies the prediction target (classification, regression, time-series, clustering, etc.)
                    3. Is testable with structured/tabular data
                    4. Reveals non-obvious insights or combines rarely studied factors
                    5. Has clear practical or theoretical significance
                    
                    AVOID:
                    - Generic claims like "X affects Y"
                    - Well-established relationships
                    - Overly broad hypotheses
                    - Untestable claims
                    
                    FORMAT YOUR RESPONSE AS:
                    HYPOTHESIS: [Specific, testable hypothesis with clear prediction target]
                    RESEARCH GAP: [What specific gap this addresses]
                    SIGNIFICANCE: [Why this matters for the field]
                    METHODOLOGY: [What type of analysis - classification/regression/etc.]
                    INNOVATION: [What's novel or non-obvious about this approach]"""},
                    {"role": "user", "content": f"""Topic: {topic}
                    
                    Research landscape analysis:
                    {analysis}
                    
                    Key gaps identified:
                    {', '.join(gaps)}
                    
                    {dataset_context}
                    
                    Generate a sophisticated hypothesis that specifically addresses one of these gaps and is feasible to test with structured data. Focus on revealing non-obvious insights or addressing known limitations."""}
                ],
                max_tokens=800,
                temperature=0.4
            )
            
            response_text = response.choices[0].message.content.strip()
            return self._parse_hypothesis_response(response_text)
            
        except Exception as e:
            print(f"Error generating hypothesis: {e}")
            return self._generate_fallback_hypothesis(topic, gaps)
    
    def _validate_and_refine_hypothesis(self, hypothesis_data: Dict, dataset_info: Optional[Dict]) -> Dict[str, Any]:
        """Validate and refine the generated hypothesis for feasibility and significance."""
        
        hypothesis = hypothesis_data.get("hypothesis", "")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are a research methodology expert. Evaluate and refine the given hypothesis for:
                    1. Testability with standard datasets
                    2. Specificity of prediction target
                    3. Novelty and significance
                    4. Feasibility with available data
                    
                    Suggest specific improvements while maintaining the core insight."""},
                    {"role": "user", "content": f"""Evaluate this hypothesis: {hypothesis}
                    
                    Research gap: {hypothesis_data.get('research_gap', '')}
                    Significance: {hypothesis_data.get('significance', '')}
                    
                    Dataset context: {dataset_info}
                    
                    Is this hypothesis:
                    1. Specific enough in its prediction target?
                    2. Testable with structured data?
                    3. Addressing a genuine research gap?
                    4. Likely to yield non-obvious insights?
                    
                    Provide a refined version if needed."""}
                ],
                max_tokens=500,
                temperature=0.2
            )
            
            refinement = response.choices[0].message.content.strip()
            
            # If refinement suggests improvements, update the hypothesis
            if "refined hypothesis:" in refinement.lower() or "improved hypothesis:" in refinement.lower():
                refined_text = self._extract_refined_hypothesis(refinement)
                if refined_text:
                    hypothesis_data["hypothesis"] = refined_text
                    hypothesis_data["refinement_notes"] = refinement
            
            return hypothesis_data
            
        except Exception as e:
            print(f"Error validating hypothesis: {e}")
            return hypothesis_data
    
    def _extract_research_gaps(self, analysis: str) -> List[str]:
        """Extract specific research gaps from the domain analysis."""
        gaps = []
        
        # Look for numbered lists or bullet points
        lines = analysis.split('\n')
        for line in lines:
            line = line.strip()
            if (line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')) and 
                len(line) > 10 and
                any(keyword in line.lower() for keyword in ['gap', 'limitation', 'lack', 'insufficient', 'underexplored', 'missing'])):
                clean_gap = re.sub(r'^[\d\.\-•\s]+', '', line).strip()
                if clean_gap:
                    gaps.append(clean_gap)
        
        # If no structured gaps found, extract sentences containing gap-related keywords
        if not gaps:
            sentences = analysis.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in ['gap', 'limitation', 'lack', 'insufficient', 'underexplored']):
                    gaps.append(sentence.strip())
        
        return gaps[:5]  # Limit to top 5 gaps
    
    def _extract_domain_from_topic(self, topic: str) -> str:
        """Extract the primary research domain from the topic."""
        domain_keywords = {
            'computer_science': ['algorithm', 'machine learning', 'AI', 'software', 'computing', 'neural network'],
            'medicine': ['disease', 'patient', 'clinical', 'medical', 'health', 'diagnosis', 'treatment'],
            'finance': ['investment', 'portfolio', 'trading', 'market', 'financial', 'risk', 'return'],
            'psychology': ['behavior', 'cognitive', 'mental', 'psychological', 'emotion', 'perception'],
            'biology': ['gene', 'protein', 'cell', 'organism', 'evolution', 'molecular', 'biological'],
            'physics': ['quantum', 'particle', 'energy', 'matter', 'physics', 'mechanics'],
            'social_science': ['social', 'society', 'cultural', 'demographic', 'community', 'population']
        }
        
        topic_lower = topic.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return domain
        
        return 'interdisciplinary'
    
    def _parse_hypothesis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the structured hypothesis response."""
        result = {
            "hypothesis": "",
            "research_gap": "",
            "significance": "",
            "methodology": "",
            "innovation": ""
        }
        
        # Extract sections using regex
        patterns = {
            "hypothesis": r"HYPOTHESIS:\s*(.+?)(?=RESEARCH GAP:|$)",
            "research_gap": r"RESEARCH GAP:\s*(.+?)(?=SIGNIFICANCE:|$)",
            "significance": r"SIGNIFICANCE:\s*(.+?)(?=METHODOLOGY:|$)",
            "methodology": r"METHODOLOGY:\s*(.+?)(?=INNOVATION:|$)",
            "innovation": r"INNOVATION:\s*(.+?)$"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()
        
        # If parsing failed, try simpler extraction
        if not result["hypothesis"]:
            lines = response_text.split('\n')
            for i, line in enumerate(lines):
                if 'hypothesis' in line.lower() and ':' in line:
                    result["hypothesis"] = line.split(':', 1)[1].strip()
                    break
        
        return result
    
    def _extract_refined_hypothesis(self, refinement_text: str) -> Optional[str]:
        """Extract a refined hypothesis from the validation response."""
        patterns = [
            r"refined hypothesis:\s*(.+?)(?=\n|$)",
            r"improved hypothesis:\s*(.+?)(?=\n|$)",
            r"better hypothesis:\s*(.+?)(?=\n|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, refinement_text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _generate_fallback_hypothesis(self, topic: str, gaps: List[str]) -> Dict[str, Any]:
        """Generate a fallback hypothesis if AI generation fails."""
        return {
            "hypothesis": f"Machine learning models incorporating multiple data modalities can predict outcomes in {topic} with higher accuracy than single-modality approaches, revealing previously undetected patterns in the data.",
            "research_gap": gaps[0] if gaps else "Limited integration of multimodal approaches",
            "significance": f"This addresses the need for more comprehensive predictive models in {topic}",
            "methodology": "Multi-class classification with ensemble methods",
            "innovation": "Novel combination of data sources and advanced feature engineering"
        }
    
    def format_hypothesis_for_display(self, hypothesis_data: Dict[str, Any]) -> str:
        """Format the hypothesis data for display."""
        hypothesis = hypothesis_data.get("hypothesis", "")
        gap = hypothesis_data.get("research_gap", "")
        significance = hypothesis_data.get("significance", "")
        methodology = hypothesis_data.get("methodology", "")
        innovation = hypothesis_data.get("innovation", "")
        
        formatted = f"""
🎯 **RESEARCH HYPOTHESIS:**
{hypothesis}

🔬 **RESEARCH GAP ADDRESSED:**
{gap}

💡 **SIGNIFICANCE & IMPACT:**
{significance}

📊 **METHODOLOGY:**
{methodology}

🚀 **INNOVATION:**
{innovation}
"""
        return formatted.strip()

# Example usage function
def generate_sophisticated_hypothesis(topic: str, openai_client, dataset_info: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function to generate a sophisticated research hypothesis.
    
    Args:
        topic: User-provided research topic
        openai_client: OpenAI client instance
        dataset_info: Optional dataset information
        
    Returns:
        Complete hypothesis data including gap analysis and significance
    """
    generator = EnhancedHypothesisGenerator(openai_client)
    return generator.generate_research_hypothesis(topic, dataset_info) 
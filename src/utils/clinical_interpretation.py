"""
Clinical Interpretation System

This module provides clinical context and interpretations for medical assessment scores,
biomarkers, and statistical findings. It maps numerical values to clinical significance
and real-world consequences for healthcare research.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re

@dataclass
class ClinicalRange:
    """Represents a clinical range with interpretation"""
    min_value: float
    max_value: float
    severity: str
    interpretation: str
    clinical_significance: str

@dataclass
class ClinicalContext:
    """Clinical context for a measurement or finding"""
    measurement: str
    value: float
    interpretation: str
    clinical_significance: str
    severity_level: str
    reference_ranges: List[ClinicalRange]

class ClinicalInterpreter:
    """
    Provides clinical interpretations for medical assessments and biomarkers
    """
    
    def __init__(self):
        self.assessment_ranges = self._initialize_assessment_ranges()
        self.biomarker_ranges = self._initialize_biomarker_ranges()
        self.effect_size_interpretations = self._initialize_effect_sizes()
    
    def _initialize_assessment_ranges(self) -> Dict[str, List[ClinicalRange]]:
        """Initialize clinical ranges for cognitive assessments"""
        return {
            'MMSE': [
                ClinicalRange(24, 30, 'Normal', 'Normal cognitive function', 
                            'No significant cognitive impairment'),
                ClinicalRange(18, 23, 'Mild', 'Mild cognitive impairment', 
                            'Early dementia or MCI; requires monitoring'),
                ClinicalRange(10, 17, 'Moderate', 'Moderate cognitive impairment', 
                            'Moderate dementia; significant functional decline'),
                ClinicalRange(0, 9, 'Severe', 'Severe cognitive impairment', 
                            'Severe dementia; requires constant care')
            ],
            'CDR': [
                ClinicalRange(0, 0, 'Normal', 'No dementia', 
                            'Normal cognitive and functional status'),
                ClinicalRange(0.5, 0.5, 'Very Mild', 'Very mild dementia/MCI', 
                            'Questionable dementia; mild functional impairment'),
                ClinicalRange(1, 1, 'Mild', 'Mild dementia', 
                            'Mild dementia; noticeable functional decline'),
                ClinicalRange(2, 2, 'Moderate', 'Moderate dementia', 
                            'Moderate dementia; requires assistance with daily activities'),
                ClinicalRange(3, 3, 'Severe', 'Severe dementia', 
                            'Severe dementia; requires constant supervision')
            ],
            'ADAS': [
                ClinicalRange(0, 9, 'Normal', 'Normal cognitive function', 
                            'No significant cognitive impairment'),
                ClinicalRange(10, 20, 'Mild', 'Mild cognitive impairment', 
                            'Early cognitive decline; monitor progression'),
                ClinicalRange(21, 35, 'Moderate', 'Moderate cognitive impairment', 
                            'Moderate dementia; functional decline evident'),
                ClinicalRange(36, 70, 'Severe', 'Severe cognitive impairment', 
                            'Severe dementia; significant functional loss')
            ]
        }
    
    def _initialize_biomarker_ranges(self) -> Dict[str, List[ClinicalRange]]:
        """Initialize clinical ranges for biomarkers"""
        return {
            'CSF_Abeta42': [
                ClinicalRange(600, 1600, 'Normal', 'Normal amyloid levels', 
                            'Low risk for Alzheimer\'s disease'),
                ClinicalRange(400, 599, 'Borderline', 'Borderline amyloid levels', 
                            'Increased risk; monitor progression'),
                ClinicalRange(0, 399, 'Abnormal', 'Low amyloid levels', 
                            'High risk for Alzheimer\'s disease')
            ]
        }
    
    def _initialize_effect_sizes(self) -> Dict[str, Dict[str, str]]:
        """Initialize effect size interpretations"""
        return {
            'cohens_d': {
                '0.2': 'small effect',
                '0.5': 'medium effect', 
                '0.8': 'large effect'
            }
        }
    
    def interpret_assessment_score(self, assessment: str, score: float) -> ClinicalContext:
        """
        Interpret a cognitive assessment score
        
        Args:
            assessment: Name of assessment (e.g., 'MMSE', 'CDR', 'ADAS-cog')
            score: The score value
            
        Returns:
            ClinicalContext with interpretation
        """
        assessment_key = assessment.upper().replace('-COG', '').replace('_', '')
        
        if assessment_key not in self.assessment_ranges:
            return self._get_generic_interpretation(assessment, score)
        
        ranges = self.assessment_ranges[assessment_key]
        
        for range_info in ranges:
            if range_info.min_value <= score <= range_info.max_value:
                return ClinicalContext(
                    measurement=assessment,
                    value=score,
                    interpretation=range_info.interpretation,
                    clinical_significance=range_info.clinical_significance,
                    severity_level=range_info.severity,
                    reference_ranges=ranges
                )
        
        # Handle out-of-range values
        return self._get_generic_interpretation(assessment, score)
    
    def interpret_score_change(self, assessment: str, baseline: float, followup: float, 
                             timeframe: str = "over time") -> Dict[str, Any]:
        """
        Interpret the clinical significance of a score change
        """
        change = followup - baseline
        
        baseline_context = self.interpret_assessment_score(assessment, baseline)
        followup_context = self.interpret_assessment_score(assessment, followup)
        
        # MMSE-specific thresholds
        if assessment.upper() == 'MMSE':
            if abs(change) < 2:
                magnitude = 'minimal'
                clinical_meaning = f"A {abs(change):.1f}-point change in MMSE represents minimal clinical change, likely within measurement error."
            elif abs(change) < 4:
                magnitude = 'mild'
                clinical_meaning = f"A {abs(change):.1f}-point {'decline' if change < 0 else 'improvement'} in MMSE indicates mild cognitive change that may signal early disease progression or treatment response."
            elif abs(change) < 6:
                magnitude = 'moderate' 
                clinical_meaning = f"A {abs(change):.1f}-point {'decline' if change < 0 else 'improvement'} in MMSE represents moderate cognitive change with likely functional implications."
            else:
                magnitude = 'severe'
                clinical_meaning = f"A {abs(change):.1f}-point {'decline' if change < 0 else 'improvement'} in MMSE indicates severe cognitive change requiring immediate clinical attention."
        else:
            magnitude = 'measurable'
            clinical_meaning = f"A {abs(change):.1f}-point change in {assessment} {timeframe}"
        
        return {
            'baseline_score': baseline,
            'followup_score': followup,
            'change': change,
            'magnitude': magnitude,
            'direction': 'decline' if change < 0 else 'improvement',
            'timeframe': timeframe,
            'baseline_interpretation': baseline_context.interpretation,
            'followup_interpretation': followup_context.interpretation,
            'clinical_meaning': clinical_meaning,
            'severity_change': f"{baseline_context.severity_level} → {followup_context.severity_level}"
        }
    
    def generate_clinical_discussion(self, results: Dict[str, Any], domain: str = "alzheimers") -> str:
        """
        Generate clinical discussion section with real-world context
        """
        discussion_parts = []
        
        # Add clinical interpretation header
        discussion_parts.append("## Clinical Interpretation and Real-World Significance\n")
        
        # Extract key findings and provide clinical context
        if 'feature_analysis' in results:
            feature_analysis = results['feature_analysis']
            top_features = feature_analysis.get('top_features', [])
            
            if 'MMSE' in top_features:
                discussion_parts.append(self._generate_mmse_discussion(results))
            
            if any(feature.upper().startswith('APOE') for feature in top_features):
                discussion_parts.append(self._generate_apoe_discussion())
                
        # Add statistical significance with clinical context
        discussion_parts.append(self._generate_statistical_clinical_context())
        
        # Add clinical implications
        discussion_parts.append(self._generate_clinical_implications(domain))
        
        return "\n".join(discussion_parts)
    
    def _generate_mmse_discussion(self, results: Dict[str, Any]) -> str:
        """Generate MMSE-specific clinical discussion"""
        return """
### Mini-Mental State Examination (MMSE) Clinical Context

The MMSE emerged as a key predictive variable in our analysis. From a clinical perspective:

**Clinical Thresholds and Interpretation:**
- **MMSE 24-30**: Normal cognitive function, no significant impairment
- **MMSE 18-23**: Mild cognitive impairment (MCI) or early dementia stage
- **MMSE 10-17**: Moderate dementia with significant functional decline
- **MMSE 0-9**: Severe dementia requiring constant care and supervision

**Clinical Significance of MMSE Changes:**
A 2-point decline in MMSE score may signal the transition from normal cognition to mild cognitive impairment, representing early disease progression that warrants closer monitoring and potential intervention. A 4-point decline typically indicates progression from MCI to mild dementia, with noticeable impacts on daily functioning and independence.

**Real-World Implications:**
- Patients with MMSE scores below 24 should undergo comprehensive neuropsychological evaluation
- A decline of ≥3 points over 6-12 months suggests active disease progression
- MMSE changes correlate with functional decline, affecting activities of daily living
- Caregivers should be educated about safety concerns when MMSE drops below 20

"""
    
    def _generate_apoe_discussion(self) -> str:
        """Generate APOE4-specific clinical discussion"""
        return """
### APOE4 Genetic Risk Factor Clinical Context

The APOE4 allele represents the strongest genetic risk factor for late-onset Alzheimer's disease:

**Clinical Risk Stratification:**
- **APOE4 negative (ε3/ε3)**: Baseline population risk (~10-15% lifetime risk)
- **APOE4 heterozygous (ε3/ε4)**: 2-3x increased risk (~25-30% lifetime risk)
- **APOE4 homozygous (ε4/ε4)**: 8-12x increased risk (~50-90% lifetime risk)

**Clinical Implications:**
- APOE4 carriers show earlier symptom onset (typically 5-10 years earlier)
- Enhanced amyloid deposition and faster cognitive decline in APOE4 carriers
- May influence treatment response to anti-amyloid therapies
- Important for genetic counseling and family planning discussions

**Patient Care Considerations:**
- APOE4 status should inform monitoring frequency and biomarker screening
- Lifestyle interventions may be particularly important for APOE4 carriers
- Consideration for earlier enrollment in preventive trials

"""
    
    def _generate_statistical_clinical_context(self) -> str:
        """Generate discussion linking statistical findings to clinical meaning"""
        return """
### Statistical Findings and Clinical Translation

**Effect Size Interpretation:**
- **Small effect (d=0.2)**: Detectable difference but limited clinical impact
- **Medium effect (d=0.5)**: Moderate clinical relevance with practical implications
- **Large effect (d=0.8)**: Substantial clinical impact affecting patient outcomes

**Clinical Significance vs. Statistical Significance:**
While p-values indicate the reliability of our findings, effect sizes determine clinical relevance. A statistically significant result with small effect size may have limited practical application, whereas large effect sizes suggest meaningful clinical differences that could influence treatment decisions.

**Practical Clinical Application:**
The predictive accuracy achieved by our model (if >80%) would be clinically actionable for:
- Risk stratification in clinical practice
- Informing patient and family discussions about prognosis
- Guiding timing of interventions and care planning
- Supporting clinical trial enrollment decisions

"""
    
    def _generate_clinical_implications(self, domain: str) -> str:
        """Generate clinical implications and recommendations"""
        if domain.lower() == 'alzheimers':
            return """
### Clinical Implications and Practice Recommendations

**Diagnostic Applications:**
1. **Risk Assessment**: The identified predictive factors can enhance early detection strategies
2. **Monitoring**: Regular assessment using these variables can track disease progression
3. **Treatment Planning**: Risk profiles can inform personalized intervention strategies

**Patient Care Integration:**
- Incorporate predictive models into routine clinical workflows
- Use risk scores to guide monitoring frequency (high-risk patients every 3-6 months)
- Inform patients and families about modifiable vs. non-modifiable risk factors
- Consider early intervention for high-risk individuals

**Healthcare System Impact:**
- Resource allocation based on risk stratification
- Cost-effective screening programs targeting high-risk populations
- Improved patient outcomes through earlier detection and intervention
- Enhanced clinical trial recruitment using predictive models

**Limitations and Considerations:**
- Model predictions should supplement, not replace, clinical judgment
- Regular model validation and updating required as new evidence emerges
- Ethical considerations regarding genetic information disclosure
- Need for diverse population validation before widespread implementation

"""
        return ""
    
    def _get_generic_interpretation(self, assessment: str, score: float) -> ClinicalContext:
        """Generate generic interpretation for unknown assessments"""
        return ClinicalContext(
            measurement=assessment,
            value=score,
            interpretation=f"Score of {score} on {assessment}",
            clinical_significance="Clinical significance requires domain-specific interpretation",
            severity_level="Unknown",
            reference_ranges=[]
        )

# Global instance
clinical_interpreter = ClinicalInterpreter()

def get_clinical_interpreter() -> ClinicalInterpreter:
    """Get the global clinical interpreter instance"""
    return clinical_interpreter 
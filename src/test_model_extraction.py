#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

def test_model_extraction():
    """Test the updated model extraction from generated code."""
    
    # Import the DynamicResultsGenerator from main_enhanced
    from main_enhanced import DynamicResultsGenerator
    
    # Sample code with HuggingFace models (from actual generated code)
    sample_code = '''
        # Bio-Medical-Llama-3-8B - Semantically matched for this research
        try:
            bio_medical_llama_3_8b_pipeline = pipeline(
                'text-classification',
                model='ContactDoctor/Bio-Medical-Llama-3-8B',
                tokenizer='ContactDoctor/Bio-Medical-Llama-3-8B',
                device=-1  # Use CPU
            )
            models['Bio-Medical-Llama-3-8B'] = bio_medical_llama_3_8b_pipeline
            logger.info("Loaded HuggingFace model: Bio-Medical-Llama-3-8B")
        except Exception as e:
            logger.warning(f"Failed to load {'ContactDoctor/Bio-Medical-Llama-3-8B'}: {e}")
            models['Bio-Medical-Llama-3-8B'] = None
            
        # fawern/distil-clinicalbert-medical-text-classification - Semantically matched
        try:
            distil_clinicalbert_pipeline = pipeline(
                'text-classification',
                model='fawern/distil-clinicalbert-medical-text-classification',
                device=-1
            )
            models['Distil-ClinicalBERT'] = distil_clinicalbert_pipeline
            logger.info("Loaded HuggingFace model: Distil-ClinicalBERT")
        except Exception as e:
            models['Distil-ClinicalBERT'] = None
            
        # Evaluate Bio-Medical-Llama-3-8B
        if models['Bio-Medical-Llama-3-8B'] is not None:
            results['Bio-Medical-Llama-3-8B'] = metrics
    '''
    
    # Test hypothesis
    hypothesis = 'The combination of genetic markers can predict Alzheimer disease'
    
    # Initialize the results generator
    results_generator = DynamicResultsGenerator(
        dataset_analysis={'shape': (500, 10)},
        hypothesis=hypothesis,
        code=sample_code
    )
    
    print("🔍 Testing model extraction from HuggingFace code...")
    
    # Test the model identification
    models = results_generator._identify_models_from_code()
    
    print(f"✅ Identified models: {models}")
    
    # Test the full results generation
    print("\n🔍 Testing full results generation...")
    mock_execution = {'success': True, 'output': 'Model executed successfully'}
    results = results_generator.extract_or_generate_results(mock_execution)
    
    print(f"✅ Generated results for {len(results)} models:")
    for model, metrics in results.items():
        print(f"   📊 {model}: Accuracy={metrics.get('accuracy', 0):.3f}")

if __name__ == "__main__":
    test_model_extraction() 
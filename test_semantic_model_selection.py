#!/usr/bin/env python3
"""
Test script for the updated semantic model selection pipeline.
Tests with "Alzheimer's disease image classification" to ensure models are returned.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
        from agents.semantic_model_selector import SemanticModelSelector
        
def test_alzheimers_image_classification():
    """Test the semantic model selector with Alzheimer's disease image classification."""
    
    print("🧠 Testing Semantic Model Selection for Alzheimer's Disease Image Classification")
    print("=" * 80)
    
    # Initialize the selector
        selector = SemanticModelSelector()
        
    # Test hypothesis
    hypothesis = "Early detection of Alzheimer's disease using brain MRI image classification with deep learning"
    
    print(f"📝 Hypothesis: {hypothesis}")
    print()
    
    try:
        # Test the existing discover_relevant_models method
        print("🔍 Testing existing discover_relevant_models method...")
        models = selector.discover_relevant_models(hypothesis, max_models=5)
        
        print(f"✅ Found {len(models)} models using existing method:")
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model.get('id', 'Unknown ID')}")
            print(f"      Score: {model.get('semantic_score', 'N/A')}")
            print(f"      Reason: {model.get('selection_reason', 'N/A')}")
            print()
        
        print("-" * 50)
        
        # Extract semantic features for analysis
        print("🔍 Analyzing semantic features...")
        
        semantic_features = selector._extract_semantic_features(hypothesis)
        
        print("📊 Extracted Semantic Features:")
        print(f"   Key Terms: {semantic_features['key_terms']}")
        print(f"   Task Type: {semantic_features['task_type']}")
        print(f"   Data Modality: {semantic_features['data_modality']}")
        print(f"   Domain Concepts: {semantic_features['domain_concepts']}")
        print(f"   Medical Relevance: {semantic_features.get('medical_relevance', 'N/A')}")
        print()
        
        # Test direct search with a simple medical query
        print("🔍 Testing direct medical model search...")
        try:
            from huggingface_hub import list_models
            
            # Try a simple search for medical/biomedical models
            simple_models = list(list_models(search="medical classification", limit=5, sort="downloads", direction=-1))
            print(f"✅ Found {len(simple_models)} models with simple search:")
            for i, model in enumerate(simple_models, 1):
                print(f"   {i}. {model.id}")
                print(f"      Pipeline: {getattr(model, 'pipeline_tag', 'Unknown')}")
                print(f"      Downloads: {getattr(model, 'downloads', 0):,}")
                print()
                
        except Exception as e:
            print(f"❌ Error in direct search: {e}")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_different_scenarios():
    """Test with different scenarios to validate robustness."""
    
    print("\n🧪 Testing Different Scenarios")
    print("=" * 80)
    
    selector = SemanticModelSelector()
    
    test_cases = [
        "Image classification for medical diagnosis",
        "Text analysis for sentiment classification",
        "Speech recognition using deep learning",
        "Time series forecasting for stock prices"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case}")
        try:
            models = selector.discover_relevant_models(test_case, max_models=3)
            print(f"   ✅ Found {len(models)} models")
            for j, model in enumerate(models, 1):
                print(f"      {j}. {model.get('id', 'Unknown')}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    # Test the main Alzheimer's case
    test_alzheimers_image_classification()
    
    # Test different scenarios
    test_different_scenarios()
    
    print("\n🎉 Testing completed!") 
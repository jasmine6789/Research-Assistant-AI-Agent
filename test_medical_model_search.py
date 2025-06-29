#!/usr/bin/env python3
"""
Test script for the medical-focused semantic model selection pipeline.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents.semantic_model_selector import SemanticModelSelector

def test_direct_huggingface_search():
    """Test direct HuggingFace search to validate API connection."""
    
    print("🔍 Testing Direct HuggingFace API Connection")
    print("=" * 60)
    
    try:
        from huggingface_hub import list_models
        
        # Test different medical search terms
        search_terms = [
            "medical",
            "biomedical",
            "clinical",
            "alzheimer",
            "disease classification"
        ]
        
        for term in search_terms:
            print(f"\n📝 Searching for: '{term}'")
            try:
                models = list(list_models(search=term, limit=3, sort="downloads", direction=-1))
                print(f"   ✅ Found {len(models)} models")
                for i, model in enumerate(models, 1):
                    print(f"      {i}. {model.id}")
                    print(f"         Downloads: {getattr(model, 'downloads', 0):,}")
                    print(f"         Pipeline: {getattr(model, 'pipeline_tag', 'Unknown')}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to import or use HuggingFace Hub: {e}")

def test_medical_model_selector():
    """Test the medical-focused semantic model selector."""
    
    print("\n🧠 Testing Medical Model Selector")
    print("=" * 60)
    
    selector = SemanticModelSelector()
    
    # Test with medical hypothesis
    hypothesis = "Classification of brain MRI images for Alzheimer's disease detection"
    
    print(f"📝 Hypothesis: {hypothesis}")
    print()
    
    try:
        # Extract semantic features
        print("🔍 Extracting semantic features...")
        semantic_features = selector._extract_semantic_features(hypothesis)
        
        print("📊 Semantic Features:")
        print(f"   Medical Relevance: {semantic_features.get('medical_relevance', 0):.2f}")
        print(f"   Key Terms: {semantic_features['key_terms'][:5]}")  # Show first 5
        print(f"   Task Type: {semantic_features['task_type']}")
        print(f"   Data Modality: {semantic_features['data_modality']}")
        print()
        
        # Test model discovery
        print("🔍 Discovering relevant models...")
        models = selector.discover_relevant_models(hypothesis, max_models=5)
        
        print(f"✅ Found {len(models)} relevant models:")
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model.get('id', 'Unknown')}")
            print(f"      Score: {model.get('semantic_score', 0):.3f}")
            print(f"      Reason: {model.get('selection_reason', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_search_criteria_adjustment():
    """Test with more relaxed search criteria."""
    
    print("\n🔧 Testing Search Criteria Adjustment")
    print("=" * 60)
    
    selector = SemanticModelSelector()
    
    # Lower the medical relevance threshold for testing
    print("🔍 Testing with broader search criteria...")
    
    # Test with simpler queries
    test_queries = [
        "image classification",
        "medical image",
        "brain MRI",
        "alzheimer detection"
    ]
    
    for query in test_queries:
        print(f"\n📝 Testing query: '{query}'")
        try:
            candidates = selector._search_medical_models([query], 10)
            print(f"   ✅ Found {len(candidates)} candidates")
            
            for i, candidate in enumerate(candidates[:3], 1):  # Show first 3
                print(f"      {i}. {candidate.get('id', 'Unknown')}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    # Test direct HuggingFace API
    test_direct_huggingface_search()
    
    # Test medical model selector
    test_medical_model_selector()
    
    # Test search criteria
    test_search_criteria_adjustment()
    
    print("\n🎉 Testing completed!") 
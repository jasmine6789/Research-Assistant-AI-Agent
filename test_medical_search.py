#!/usr/bin/env python3
"""
Test script for the medical-focused semantic model selection pipeline.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from agents.semantic_model_selector import SemanticModelSelector

def test_direct_huggingface_search():
    """Test direct HuggingFace search to validate API connection."""
    
    print(" Testing Direct HuggingFace API Connection")
    print("=" * 60)
    
    try:
        from huggingface_hub import list_models
        
        # Test simple search
        print(" Searching for medical models...")
        models = list(list_models(search="medical", limit=3, sort="downloads", direction=-1))
        print(f" Found {len(models)} models")
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model.id}")
            print(f"      Downloads: {getattr(model, \"downloads\", 0):,}")
                    
    except Exception as e:
        print(f" Error: {e}")

def test_medical_model_selector():
    """Test the medical model selector."""
    
    print("\\n Testing Medical Model Selector")
    print("=" * 60)
    
    selector = SemanticModelSelector()
    hypothesis = "Classification of brain MRI images for Alzheimer disease detection"
    
    print(f" Hypothesis: {hypothesis}")
    
    try:
        models = selector.discover_relevant_models(hypothesis, max_models=3)
        print(f" Found {len(models)} models")
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model.get(\"id\", \"Unknown\")}")
            
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    test_direct_huggingface_search()
    test_medical_model_selector()
    print("\\n Testing completed!")

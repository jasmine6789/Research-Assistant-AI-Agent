#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_huggingface_api():
    print("🔍 Testing HuggingFace API Connection")
    print("=" * 50)
    
    try:
        from huggingface_hub import list_models
        
        # Test basic search
        models = list(list_models(search="medical", limit=3))
        print(f"✅ Found {len(models)} models with 'medical' search")
        
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model.id}")
            downloads = getattr(model, 'downloads', 0)
            print(f"      Downloads: {downloads:,}")
            
    except Exception as e:
        print(f"❌ API Error: {e}")
        import traceback
        traceback.print_exc()

def test_medical_selector():
    print("\n🧠 Testing Medical Model Selector")
    print("=" * 50)
    
    try:
        from agents.semantic_model_selector import SemanticModelSelector
        
        selector = SemanticModelSelector()
        hypothesis = "Medical image classification for disease detection"
        
        print(f"📝 Hypothesis: {hypothesis}")
        
        # Test with relaxed criteria
        models = selector.discover_relevant_models(hypothesis, max_models=3)
        
        if models:
            print(f"✅ Found {len(models)} models")
            for i, model in enumerate(models, 1):
                print(f"   {i}. {model.get('id', 'Unknown')}")
        else:
            print("⚠️ No models found - checking fallback...")
            fallback_models = selector._fallback_medical_models(hypothesis)
            print(f"✅ Fallback returned {len(fallback_models)} models")
            for i, model in enumerate(fallback_models, 1):
                print(f"   {i}. {model.get('id', 'Unknown')}")
                
    except Exception as e:
        print(f"❌ Selector Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_huggingface_api()
    test_medical_selector()
    print("\n🎉 Testing completed!") 
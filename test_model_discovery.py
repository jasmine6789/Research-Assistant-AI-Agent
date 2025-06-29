#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from agents.enhanced_code_agent import EnhancedCodeAgent
from agents.note_taker import NoteTaker

def test_model_discovery():
    """Test the model discovery mechanism to see what models are being used."""
    
    # Initialize the code agent
    note_taker = NoteTaker()
    code_agent = EnhancedCodeAgent(os.getenv('CHATGPT_API_KEY'), note_taker)
    
    # Test hypothesis
    hypothesis = 'The combination of genetic markers (apoe4) and demographic factors can predict early Alzheimer disease detection'
    
    print("🔍 Testing semantic model discovery...")
    try:
        models = code_agent.discover_relevant_models(hypothesis, max_models=3)
        print(f"✅ Found {len(models)} semantic models:")
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model['id']}")
    except Exception as e:
        print(f"❌ Error in semantic discovery: {e}")
        models = []
    
    print("\n🔍 Testing domain appropriate models fallback...")
    try:
        domain = code_agent._analyze_research_domain(hypothesis)
        fallback_models = code_agent._get_domain_appropriate_models(domain, hypothesis)
        print(f"✅ Domain: {domain}")
        print(f"✅ Fallback models:")
        for i, model in enumerate(fallback_models, 1):
            print(f"   {i}. {model}")
    except Exception as e:
        print(f"❌ Error in fallback models: {e}")
        fallback_models = []
    
    print("\n📋 Summary:")
    print(f"- Semantic models found: {len(models)}")
    print(f"- Fallback models available: {len(fallback_models)}")
    
    if models:
        print("✅ System should use HuggingFace models")
        print(f"   Primary models: {[m['id'] for m in models]}")
    else:
        print("⚠️ System will use fallback models")
        print(f"   Fallback models: {fallback_models}")

if __name__ == "__main__":
    test_model_discovery() 
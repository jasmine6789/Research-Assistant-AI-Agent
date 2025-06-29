#!/usr/bin/env python3
"""
Test script for enhanced model selection functionality
"""
import os
import sys
sys.path.append('src')

from agents.enhanced_code_agent import EnhancedCodeAgent

class MockNoteTaker:
    def log(self, *args, **kwargs): 
        print(f"LOG: {args}")

def test_enhanced_model_selection():
    """Test the enhanced model selection with various hypotheses"""
    
    # Set up the agent
    OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found in environment")
        return
    
    note_taker = MockNoteTaker()
    agent = EnhancedCodeAgent(OPENAI_API_KEY, note_taker)
    
    # Test cases covering different domains
    test_cases = [
        {
            'name': 'Medical AI',
            'hypothesis': 'APOE4 genotype significantly influences Alzheimer disease progression and can be predicted using cognitive assessment scores'
        },
        {
            'name': 'Computer Vision', 
            'hypothesis': 'Deep convolutional neural networks can accurately detect and classify skin lesions in dermatological images'
        },
        {
            'name': 'Natural Language Processing',
            'hypothesis': 'Transformer-based models outperform LSTM networks in sentiment analysis of social media text'
        },
        {
            'name': 'Time Series Forecasting',
            'hypothesis': 'LSTM models combined with attention mechanisms improve stock price prediction accuracy over traditional ARIMA models'
        },
        {
            'name': 'Bioinformatics',
            'hypothesis': 'Protein sequence analysis using deep learning can predict protein folding structures better than traditional methods'
        }
    ]
    
    print("🧪 Testing Enhanced Model Selection System")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print(f"Hypothesis: {test_case['hypothesis']}")
        print("-" * 40)
        
        try:
            # Test semantic analysis
            semantic_analysis = agent._perform_semantic_analysis(test_case['hypothesis'])
            print(f"🔍 Domain detected: {semantic_analysis['primary_domain']}")
            print(f"🎯 Task type: {semantic_analysis['task_type']}")
            print(f"📊 Data modality: {semantic_analysis['data_modality']}")
            print(f"🔑 Key concepts: {semantic_analysis['core_concepts'][:3]}")
            
            # Test model discovery
            models = agent.discover_relevant_models(test_case['hypothesis'], max_models=3)
            
            if models:
                print(f"✅ Found {len(models)} relevant models:")
                for j, model in enumerate(models, 1):
                    relevance = model.get('relevance_score', 0)
                    print(f"   {j}. {model['id']} (relevance: {relevance:.2f})")
            else:
                print("❌ No relevant models found")
                
        except Exception as e:
            print(f"❌ Error in test case {i}: {e}")
        
        print()
    
    print("🎉 Enhanced Model Selection Test Complete!")

if __name__ == "__main__":
    test_enhanced_model_selection() 
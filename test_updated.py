import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents.semantic_model_selector import SemanticModelSelector

def test_updated_selector():
    print('Testing Updated Medical Model Selector')
    print('=' * 50)
    
    selector = SemanticModelSelector()
    hypothesis = 'Early detection of Alzheimers disease using brain MRI image classification'
    
    print(f'Hypothesis: {hypothesis}')
    print()
    
    try:
        models = selector.discover_relevant_models(hypothesis, max_models=3)
        
        if models:
            print(f'Found {len(models)} models:')
            for i, model in enumerate(models, 1):
                print(f'   {i}. {model.get(\"id\", \"Unknown\")}')
                print(f'      Score: {model.get(\"semantic_score\", 0):.3f}')
                print(f'      Reason: {model.get(\"selection_reason\", \"N/A\")}')
                print()
        else:
            print('No models found - trying fallback...')
            fallback = selector._fallback_medical_models(hypothesis)
            print(f'Fallback returned {len(fallback)} models')
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_updated_selector()

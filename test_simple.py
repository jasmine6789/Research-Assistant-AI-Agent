import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_huggingface_api():
    print('Testing HuggingFace API Connection')
    
    try:
        from huggingface_hub import list_models
        models = list(list_models(search='medical', limit=3))
        print(f'Found {len(models)} models')
        for i, model in enumerate(models, 1):
            print(f'   {i}. {model.id}')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    test_huggingface_api()

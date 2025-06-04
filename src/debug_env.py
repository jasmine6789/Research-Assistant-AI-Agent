import os
from dotenv import load_dotenv

def debug_env():
    print("Before loading .env:")
    print(f"CHATGPT_API_KEY: {os.getenv('CHATGPT_API_KEY', 'NOT SET')}")
    print(f"GOOGLE_CLOUD_PROJECT: {os.getenv('GOOGLE_CLOUD_PROJECT', 'NOT SET')}")
    
    # Load .env
    load_dotenv()
    
    print("\nAfter loading .env:")
    api_key = os.getenv('CHATGPT_API_KEY', 'NOT SET')
    project = os.getenv('GOOGLE_CLOUD_PROJECT', 'NOT SET')
    
    print(f"CHATGPT_API_KEY: {api_key}")
    print(f"Length of API key: {len(api_key) if api_key != 'NOT SET' else 0}")
    print(f"First 20 chars: {api_key[:20] if api_key != 'NOT SET' else 'N/A'}")
    print(f"Last 20 chars: {api_key[-20:] if api_key != 'NOT SET' else 'N/A'}")
    print(f"GOOGLE_CLOUD_PROJECT: {project}")
    
    # Test OpenAI connection
    if api_key != 'NOT SET' and len(api_key) > 20:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            # Try to list models to test authentication
            models = client.models.list()
            print(f"\n✅ OpenAI API key is valid! Found {len(models.data)} models.")
            print("Available models:", [m.id for m in models.data[:5]])  # Show first 5
        except Exception as e:
            print(f"\n❌ OpenAI API error: {str(e)}")
    else:
        print("\n❌ API key not properly loaded")

if __name__ == "__main__":
    debug_env() 
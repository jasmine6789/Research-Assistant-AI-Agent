import os
from dotenv import load_dotenv
from openai import OpenAI

def test_openai_simple():
    load_dotenv()
    api_key = os.getenv("CHATGPT_API_KEY")
    
    if not api_key:
        print("❌ No API key found in environment")
        return False
    
    print(f"🔑 Testing API key: {api_key[:20]}...{api_key[-10:]}")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Test 1: List models
        print("\n📋 Testing model access...")
        models = client.models.list()
        available_models = [m.id for m in models.data if 'gpt' in m.id]
        print(f"✅ Available GPT models: {available_models[:5]}")
        
        # Test 2: Try minimal API call
        print("\n🤖 Testing minimal chat completion...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, respond with just 'Hi!'"}],
            max_tokens=5,
            temperature=0
        )
        
        print(f"✅ API call successful! Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        if "429" in str(e):
            print("   💡 This is a quota/rate limit issue")
        elif "401" in str(e):
            print("   💡 This is an authentication issue")
        elif "404" in str(e):
            print("   💡 Model not found or no access")
        return False

if __name__ == "__main__":
    test_openai_simple() 
import os
from dotenv import load_dotenv
from openai import OpenAI
import time

def test_current_key():
    load_dotenv()
    api_key = os.getenv("CHATGPT_API_KEY")
    
    if not api_key:
        print("❌ No API key found in environment")
        return False
    
    print(f"🔑 Current API key: {api_key[:15]}...{api_key[-15:]}")
    print(f"📏 Key length: {len(api_key)} characters")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Test 1: List models
        print("\n📋 Testing model access...")
        models = client.models.list()
        available_models = [m.id for m in models.data if 'gpt' in m.id]
        print(f"✅ Available GPT models: {available_models[:5]}")
        
        # Test 2: Try minimal API call with rate limiting
        print("\n🤖 Testing minimal chat completion...")
        
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello'"}],
            max_tokens=10,
            temperature=0
        )
        
        print(f"✅ API call successful! Response: {response.choices[0].message.content}")
        print(f"📊 Usage: {response.usage}")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        error_str = str(e)
        
        if "429" in error_str:
            print("   💡 This is a rate limit issue")
            if "quota" in error_str.lower():
                print("   🔍 Quota exceeded - check billing/organization")
            else:
                print("   🔍 Rate limited - too many requests")
        elif "401" in error_str:
            print("   💡 Authentication issue - invalid API key")
        elif "404" in error_str:
            print("   💡 Model not found or no access")
        elif "insufficient_quota" in error_str:
            print("   💡 Billing/quota issue despite having credits")
            print("   🔧 Try regenerating your API key in OpenAI dashboard")
        
        return False

def test_with_organization():
    """Try with explicit organization if available"""
    load_dotenv()
    api_key = os.getenv("CHATGPT_API_KEY")
    
    try:
        # Try without organization first
        client = OpenAI(api_key=api_key)
        print("\n🏢 Testing without organization...")
        
        # Try a very minimal request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=3
        )
        
        print(f"✅ Success without organization: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Failed without organization: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing current API key configuration...")
    
    success = test_current_key()
    
    if not success:
        print("\n🔄 Trying alternative approaches...")
        test_with_organization()
        
        print("\n💡 Possible solutions:")
        print("   1. Regenerate your API key in OpenAI dashboard")
        print("   2. Check if key is from correct organization/project")
        print("   3. Wait a few minutes and try again")
        print("   4. Verify payment method is active") 
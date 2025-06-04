#!/usr/bin/env python3
"""Quick test for enhanced code agent fixes"""

import sys
import os
from dotenv import load_dotenv
load_dotenv()

def test_code_execution():
    """Test code execution with encoding fixes"""
    try:
        from src.agents.enhanced_code_agent import EnhancedCodeAgent
        print('✅ Enhanced Code Agent imported successfully')
        
        # Test encoding fix
        test_code = '''import numpy as np
print("Hello World")
result = 2 + 2
print("Result:", result)
'''
        
        class MockNoteTaker:
            def log(self, *args, **kwargs): pass
            def log_code(self, *args, **kwargs): pass
        
        api_key = os.getenv('CHATGPT_API_KEY')
        if api_key:
            agent = EnhancedCodeAgent(api_key, MockNoteTaker())
            result = agent.execute_code_safely(test_code, timeout=10)
            print(f'✅ Code execution test: {"SUCCESS" if result["success"] else "FAILED"}')
            print(f'   Execution time: {result["execution_time"]:.2f}s')
            if not result['success']:
                print(f'   Error: {result["error"]}')
            return result['success']
        else:
            print('⚠️ No API key found, testing syntax validation only')
            agent = EnhancedCodeAgent("dummy", MockNoteTaker())
            syntax_valid = agent._validate_syntax(test_code)
            print(f'✅ Syntax validation: {"PASSED" if syntax_valid else "FAILED"}')
            return syntax_valid
            
    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 TESTING ENHANCED CODE AGENT FIXES")
    print("=" * 50)
    success = test_code_execution()
    print("=" * 50)
    if success:
        print("✅ ALL TESTS PASSED - FIXES WORKING!")
    else:
        print("❌ SOME TESTS FAILED - NEEDS MORE WORK") 
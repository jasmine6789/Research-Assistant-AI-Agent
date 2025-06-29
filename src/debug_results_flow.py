#!/usr/bin/env python3
"""
Debug script to trace the results flow from code execution to paper generation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_results_flow():
    """Test the complete results flow to identify the root cause"""
    
    print("🔍 DEBUGGING RESULTS FLOW")
    print("="*80)
    
    # Test 1: Mock execution results
    print("\n1. Testing Mock Execution Results:")
    mock_execution_result = {
        'success': True,
        'output': '''
Random Forest: Accuracy=0.847, F1=0.849, Precision=0.851, Recall=0.847
Gradient Boosting: Accuracy=0.823, F1=0.826, Precision=0.829, Recall=0.823
SVM: Accuracy=0.798, F1=0.801, Precision=0.805, Recall=0.798
Cross-validation score: 0.834
''',
        'execution_time': 1.5
    }
    
    print(f"   ✅ Mock execution successful: {mock_execution_result['success']}")
    print(f"   📝 Mock output length: {len(mock_execution_result['output'])} chars")
    
    # Test 2: Simulate results extraction (from main_enhanced.py)
    print("\n2. Testing Results Extraction Logic:")
    import re
    
    output = mock_execution_result['output']
    models_found = {}
    
    # Test the exact regex from main_enhanced.py
    for line in output.split('\n'):
        if 'accuracy' in line.lower() and any(metric in line.lower() for metric in ['f1', 'precision', 'recall']):
            print(f"   🔍 Processing line: {line.strip()}")
            
            # Extract accuracy
            acc_match = re.search(r'[Aa]ccuracy[=:\s]*([0-9.]+)', line)
            prec_match = re.search(r'[Pp]recision[=:\s]*([0-9.]+)', line)
            rec_match = re.search(r'[Rr]ecall[=:\s]*([0-9.]+)', line)
            f1_match = re.search(r'[Ff]1[=:\s]*([0-9.]+)', line)
            
            # Try to identify model name
            model_name = 'Model'
            for model_keyword in ['Random Forest', 'Gradient Boosting', 'SVM', 'Logistic Regression']:
                if model_keyword.lower() in line.lower():
                    model_name = model_keyword
                    break
            
            # If we found metrics, add them
            if acc_match:
                performance_data = {'accuracy': float(acc_match.group(1))}
                
                if prec_match:
                    performance_data['precision'] = float(prec_match.group(1))
                if rec_match:
                    performance_data['recall'] = float(rec_match.group(1))
                if f1_match:
                    performance_data['f1_score'] = float(f1_match.group(1))
                
                models_found[model_name] = performance_data
                print(f"   ✅ Extracted {model_name}: {performance_data}")
    
    print(f"\n   📊 Total models extracted: {len(models_found)}")
    
    # Test 3: Create report_data structure
    print("\n3. Testing Report Data Structure:")
    report_data = {
        'model_results': {
            'performance_comparison': models_found,
            'execution_results': mock_execution_result
        }
    }
    
    print(f"   ✅ Report data created")
    print(f"   📊 Performance comparison: {len(report_data['model_results']['performance_comparison'])} models")
    
    # Test 4: Test table generation
    print("\n4. Testing Table Generation:")
    
    from agents.enhanced_report_agent import EnhancedReportAgent
    from agents.note_taker import NoteTaker
    
    # Mock note taker
    class MockNoteTaker:
        def log(self, *args, **kwargs):
            pass
    
    note_taker = MockNoteTaker()
    report_agent = EnhancedReportAgent(note_taker)
    
    # Test model comparison table
    model_table = report_agent._generate_model_comparison_table(report_data['model_results'])
    print(f"   📊 Model comparison table generated: {len(model_table)} chars")
    print(f"   🔍 Table preview: {model_table[:200]}...")
    
    # Test results showcase table  
    results_table = report_agent._generate_results_showcase_table(report_data['model_results'], [])
    print(f"   📊 Results showcase table generated: {len(results_table)} chars")
    print(f"   🔍 Table preview: {results_table[:200]}...")
    
    # Test 5: Check if tables contain actual data
    print("\n5. Testing Table Content:")
    
    if "0.847" in model_table and "Random Forest" in model_table:
        print("   ✅ Model comparison table contains REAL data")
    else:
        print("   ❌ Model comparison table missing REAL data")
        print(f"   🔍 Full table: {model_table}")
    
    if "0.847" in results_table and "Random Forest" in results_table:
        print("   ✅ Results showcase table contains REAL data")
    else:
        print("   ❌ Results showcase table missing REAL data")
        print(f"   🔍 Full table: {results_table}")
    
    return {
        'models_extracted': len(models_found),
        'table_generated': len(model_table) > 0,
        'models_found': models_found
    }

if __name__ == "__main__":
    results = test_results_flow()
    print(f"\n🎯 SUMMARY: {results}") 
#!/usr/bin/env python3
"""
Test optimized target detection with GEO genomics data
"""

import sys
import os
import time
sys.path.append('src')

from src.utils.enhanced_folder_dataset_manager import EnhancedFolderDatasetManager

def test_optimized_detection():
    """Test optimized target detection on large genomics dataset"""
    
    print("🧬 Testing optimized target detection on GEO data...")
    
    # Initialize manager
    manager = EnhancedFolderDatasetManager()
    
    # Start timing
    start_time = time.time()
    
    print(f"⏱️  Starting at: {time.strftime('%H:%M:%S')}")
    
    # Process the folder with GEO data
    result = manager.process_folder_input('src/user_datasets/AD')
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n⏱️  PERFORMANCE RESULTS:")
    print(f"✅ Total processing time: {elapsed:.2f} seconds")
    print(f"📊 Dataset shape: {result['combined_dataset']['dataframe'].shape if 'combined_dataset' in result else 'N/A'}")
    
    if 'combined_dataset' in result:
        target = result['combined_dataset']['target_column']
        print(f"🎯 Detected target: {target}")
        
        # Show optimization details
        if 'target_detection' in result:
            detection_info = result['target_detection']
            print(f"🔍 Target confidence: {detection_info.get('confidence', 'N/A')}")
            print(f"📋 Detection reasons: {detection_info.get('reasons', [])}")
    
    return result, elapsed

if __name__ == "__main__":
    result, time_taken = test_optimized_detection()
    
    if time_taken < 60:
        print(f"\n🚀 SUCCESS! Optimized detection completed in {time_taken:.1f} seconds")
        print("✅ Target detection is now optimized for large genomics datasets!")
    else:
        print(f"\n⚠️  Still slow ({time_taken:.1f} seconds). May need further optimization.") 
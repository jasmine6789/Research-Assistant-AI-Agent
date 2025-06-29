#!/usr/bin/env python3
"""
Test script for BiomarkerAD folder processing
"""

import sys
import os
sys.path.append('src/utils')

from enhanced_folder_dataset_manager import EnhancedFolderDatasetManager

def test_biomarker_folder():
    """Test processing the BiomarkerAD folder"""
    
    print("🧪 Testing BiomarkerAD Folder Processing")
    print("=" * 50)
    
    # Initialize folder manager
    folder_manager = EnhancedFolderDatasetManager()
    
    # Test folder path
    folder_path = "src/user_datasets/BiomarkerAD"
    
    print(f"📁 Processing folder: {folder_path}")
    
    try:
        # Process the folder
        result = folder_manager.process_folder(folder_path)
        
        if result['success']:
            print("✅ Folder processing successful!")
            print(f"📊 Dataset shape: {result['dataset'].shape}")
            print(f"📋 Columns: {list(result['dataset'].columns)[:10]}...")
            print(f"📁 Files processed: {result['combined_info']['files_processed']}")
            
            # Try auto-detecting target
            target = folder_manager.auto_detect_target(result['dataset'])
            print(f"🎯 Auto-detected target: {target}")
            
            return True
        else:
            print(f"❌ Folder processing failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_biomarker_folder() 
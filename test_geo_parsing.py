#!/usr/bin/env python3
"""
Test script for GEO Series Matrix parsing
"""

import sys
import os
sys.path.append('src')

from src.utils.enhanced_folder_dataset_manager import EnhancedFolderDatasetManager

def test_geo_parsing():
    """Test GEO Series Matrix parsing functionality"""
    
    print("🧬 Testing GEO Series Matrix parsing...")
    
    manager = EnhancedFolderDatasetManager()
    result = manager.process_folder_input('src/user_datasets/AD')
    
    print(f"\n📊 RESULTS:")
    print(f"✅ Success: {result['success']}")
    print(f"📁 Files processed: {result['files_processed']}")
    print(f"📁 Total files found: {result['total_files_found']}")
    
    if 'combined_dataset' in result:
        df = result['combined_dataset']['dataframe']
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns (first 10): {df.columns.tolist()[:10]}")
        print(f"🎯 Target column: {result['combined_dataset']['target_column']}")
        
        # Show some metadata from GEO file
        if 'processed_files' in result:
            for file_result in result['processed_files']:
                if 'metadata' in file_result and 'geo_metadata' in file_result['metadata']:
                    geo_meta = file_result['metadata']['geo_metadata']
                    print(f"\n🧬 GEO Metadata:")
                    print(f"   Title: {geo_meta.get('Series_title', 'N/A')}")
                    print(f"   Accession: {geo_meta.get('Series_geo_accession', 'N/A')}")
                    print(f"   Format: {file_result['metadata']['format']}")
                    break
    
    return result

if __name__ == "__main__":
    result = test_geo_parsing() 
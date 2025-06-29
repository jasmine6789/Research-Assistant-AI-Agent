#!/usr/bin/env python3
"""
Debug script to test individual file parsing for BiomarkerAD files
"""

import pandas as pd
import os

def debug_file_parsing():
    """Debug the parsing of BiomarkerAD files"""
    
    files = [
        "src/user_datasets/BiomarkerAD/a_MTBLS315_UPLC_NEG_nmfi_and_bsi_diagnosis.txt",
        "src/user_datasets/BiomarkerAD/a_MTBLS315_UPLC_POS_nmfi_and_bsi_diagnosis.txt"
    ]
    
    for file_path in files:
        if os.path.exists(file_path):
            print(f"\n🔍 Debugging: {os.path.basename(file_path)}")
            print("=" * 60)
            
            # Read first few lines
            with open(file_path, 'r', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(3)]
            
            print("📄 First 3 lines:")
            for i, line in enumerate(first_lines, 1):
                print(f"   {i}: {line[:100]}...")
                if '\t' in line:
                    print(f"      🔗 Tab count: {line.count(chr(9))}")
                
            print()
            
            # Try different parsing methods
            try:
                # Method 1: Tab-separated
                df_tab = pd.read_csv(file_path, delimiter='\t', low_memory=False)
                print(f"✅ Tab-separated parsing: {df_tab.shape}")
                print(f"   Columns: {list(df_tab.columns)[:5]}...")
            except Exception as e:
                print(f"❌ Tab-separated parsing failed: {e}")
            
            try:
                # Method 2: Comma-separated
                df_comma = pd.read_csv(file_path, low_memory=False)
                print(f"✅ Comma-separated parsing: {df_comma.shape}")
                print(f"   Columns: {list(df_comma.columns)[:5]}...")
            except Exception as e:
                print(f"❌ Comma-separated parsing failed: {e}")
            
            try:
                # Method 3: Auto-detect
                df_auto = pd.read_csv(file_path, sep=None, engine='python', low_memory=False)
                print(f"✅ Auto-detect parsing: {df_auto.shape}")
                print(f"   Columns: {list(df_auto.columns)[:5]}...")
            except Exception as e:
                print(f"❌ Auto-detect parsing failed: {e}")
        else:
            print(f"❌ File not found: {file_path}")

if __name__ == "__main__":
    debug_file_parsing() 
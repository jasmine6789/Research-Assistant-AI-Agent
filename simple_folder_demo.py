#!/usr/bin/env python3
"""
Simple Folder Input Demonstration

Shows how folder input with text files works conceptually.
"""

import os
import pandas as pd
import tempfile
import shutil
from pathlib import Path

def create_demo_files():
    """Create sample text files to demonstrate folder input"""
    
    # Create demo folder
    demo_folder = Path("demo_folder")
    demo_folder.mkdir(exist_ok=True)
    
    print("📁 Creating demo files...")
    
    # File 1: CSV-like text file
    csv_content = """patient_id,age,gender,diagnosis,outcome
P001,45,M,diabetes,improved
P002,32,F,hypertension,stable
P003,67,M,heart_disease,declined
P004,28,F,asthma,recovered
P005,55,M,arthritis,stable"""
    
    with open(demo_folder / "patients.txt", 'w') as f:
        f.write(csv_content)
    
    # File 2: Key-value text file
    kv_content = """patient_id: P006
age: 42
gender: F
condition: migraine
severity: moderate
outcome: improved

patient_id: P007
age: 38
gender: M
condition: flu
severity: mild
outcome: recovered

patient_id: P008
age: 59
gender: F
condition: pneumonia
severity: severe
outcome: stable"""
    
    with open(demo_folder / "records.txt", 'w') as f:
        f.write(kv_content)
    
    print(f"✅ Created demo folder: {demo_folder}")
    print(f"   📄 patients.txt (CSV-like format)")
    print(f"   📄 records.txt (key-value format)")
    
    return demo_folder

def parse_csv_like_text(content):
    """Parse CSV-like text content"""
    lines = content.strip().split('\n')
    if len(lines) < 2:
        return None
    
    headers = lines[0].split(',')
    data = []
    
    for line in lines[1:]:
        values = line.split(',')
        if len(values) == len(headers):
            data.append(dict(zip(headers, values)))
    
    return pd.DataFrame(data)

def parse_key_value_text(content):
    """Parse key-value text content"""
    lines = content.strip().split('\n')
    data = []
    current_record = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_record:
                data.append(current_record)
                current_record = {}
        elif ':' in line:
            key, value = line.split(':', 1)
            current_record[key.strip()] = value.strip()
    
    # Add last record
    if current_record:
        data.append(current_record)
    
    return pd.DataFrame(data) if data else None

def process_text_file(file_path):
    """Process a single text file with intelligent parsing"""
    
    print(f"📄 Processing: {file_path.name}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Try different parsing strategies
    strategies = [
        ("CSV-like", parse_csv_like_text),
        ("Key-value", parse_key_value_text)
    ]
    
    best_result = None
    best_strategy = None
    best_score = 0
    
    for strategy_name, parser in strategies:
        try:
            df = parser(content)
            if df is not None and len(df) > 0:
                # Simple scoring: more columns and rows = better
                score = len(df.columns) * len(df)
                if score > best_score:
                    best_score = score
                    best_result = df
                    best_strategy = strategy_name
        except:
            continue
    
    if best_result is not None:
        print(f"   ✅ Parsed with {best_strategy} strategy")
        print(f"   📊 Shape: {best_result.shape}")
        print(f"   📋 Columns: {list(best_result.columns)}")
        return best_result
    else:
        print(f"   ❌ Failed to parse")
        return None

def demonstrate_folder_processing():
    """Demonstrate folder processing with text files"""
    
    print("🚀 FOLDER INPUT WITH TEXT FILES DEMONSTRATION")
    print("="*60)
    
    # Create demo files
    demo_folder = create_demo_files()
    
    try:
        print(f"\n🔍 STEP 1: Discovering files in folder...")
        
        # Find supported files
        txt_files = list(demo_folder.glob("*.txt"))
        print(f"📁 Found {len(txt_files)} text files:")
        for file in txt_files:
            print(f"   - {file.name}")
        
        print(f"\n🔄 STEP 2: Processing each file...")
        
        # Process each file
        dataframes = []
        for txt_file in txt_files:
            df = process_text_file(txt_file)
            if df is not None:
                dataframes.append(df)
        
        if dataframes:
            print(f"\n🔗 STEP 3: Combining {len(dataframes)} dataframes...")
            
            # Get all unique columns
            all_columns = set()
            for df in dataframes:
                all_columns.update(df.columns)
            all_columns = list(all_columns)
            
            # Reindex all dataframes to have same columns
            aligned_dfs = []
            for df in dataframes:
                aligned_df = df.reindex(columns=all_columns)
                aligned_dfs.append(aligned_df)
            
            # Concatenate
            combined_df = pd.concat(aligned_dfs, ignore_index=True)
            
            print(f"✅ Successfully combined files!")
            print(f"📊 Combined shape: {combined_df.shape}")
            print(f"📋 Combined columns: {list(combined_df.columns)}")
            
            # Show sample data
            print(f"\n📋 Sample Combined Data:")
            print(combined_df.to_string())
            
            # Auto-detect target variable
            print(f"\n🎯 Auto-detecting target variable...")
            
            # Simple target detection: look for common target column names
            target_keywords = ['outcome', 'diagnosis', 'result', 'target', 'label', 'class']
            target_column = None
            
            for col in combined_df.columns:
                if any(keyword in col.lower() for keyword in target_keywords):
                    target_column = col
                    break
            
            if not target_column:
                # Fallback to last column
                target_column = combined_df.columns[-1]
            
            print(f"   🎯 Detected target: {target_column}")
            print(f"   📈 Unique values: {combined_df[target_column].nunique()}")
            print(f"   📊 Value distribution:")
            value_counts = combined_df[target_column].value_counts()
            for value, count in value_counts.items():
                print(f"      - {value}: {count}")
            
            print(f"\n🤖 WHAT HAPPENS NEXT:")
            print(f"✅ All agents will use this combined dataset:")
            print(f"   - 🧠 Hypothesis Agent: Generate hypothesis for {combined_df.shape[0]} samples")
            print(f"   - 💻 Code Agent: Create ML code with {combined_df.shape[1]} features")
            print(f"   - 📊 Visualization Agent: Visualize {len(dataframes)} files worth of data")
            print(f"   - 📄 Report Agent: Document multi-file text processing methodology")
            
            return combined_df, target_column
        
        else:
            print("❌ No files could be processed")
            return None, None
    
    finally:
        # Cleanup
        if demo_folder.exists():
            shutil.rmtree(demo_folder)
            print(f"\n🧹 Cleaned up demo folder")

def show_integration_workflow():
    """Show how this integrates with the research system"""
    
    print(f"\n🔧 INTEGRATION WITH RESEARCH SYSTEM:")
    print("="*50)
    
    print("""
📁 YOUR WORKFLOW:

1. 📂 Create folder with your text files:
   research_data/
   ├── experiment1.txt    (CSV format: "col1,col2,col3")
   ├── experiment2.txt    (Key-value: "key: value")
   ├── measurements.csv   (Standard CSV)
   └── results.xlsx       (Excel file)

2. 🔄 System automatically:
   ✅ Discovers all .txt, .csv, .xlsx, .json files
   ✅ Intelligently parses text files using multiple strategies
   ✅ Combines all data into unified dataset
   ✅ Detects target variable automatically
   ✅ Passes combined data to all agents

3. 🎯 Result:
   - Single dataset from multiple files
   - Smart text parsing (CSV-like, key-value, structured)
   - Auto target detection
   - Ready for ML pipeline

✨ ALL AGENTS USE THE COMBINED DATASET FROM YOUR FOLDER!
""")

if __name__ == "__main__":
    # Run demonstration
    combined_df, target_column = demonstrate_folder_processing()
    show_integration_workflow()
    
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print("="*60)
    if combined_df is not None:
        print(f"✅ Successfully processed folder with text files!")
        print(f"📊 Final dataset: {combined_df.shape}")
        print(f"🎯 Target variable: {target_column}")
        print(f"🚀 Ready for research pipeline!")
    else:
        print(f"❌ Processing failed")
    
    print(f"\n💡 KEY TAKEAWAY:")
    print(f"   You CAN input a folder containing .txt files,")
    print(f"   and ALL agents will read and use the combined data!") 
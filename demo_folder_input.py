#!/usr/bin/env python3
"""
Demonstration: Folder Input with Text Files

This script shows how to use a folder containing .txt files as input
for the research assistant system.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.append('src')

def create_sample_text_folder():
    """Create a sample folder with two .txt files for demonstration"""
    
    # Create temporary folder
    sample_folder = Path("demo_text_folder")
    sample_folder.mkdir(exist_ok=True)
    
    print("📁 Creating sample folder with text files...")
    
    # Text file 1: CSV-like format
    text1_content = """patient_id,age,gender,symptoms,diagnosis
P001,45,M,fever_cough_fatigue,COVID-19
P002,32,F,headache_nausea,Migraine
P003,67,M,chest_pain_shortness_breath,Heart_Disease
P004,28,F,sore_throat_runny_nose,Common_Cold
P005,55,M,joint_pain_stiffness,Arthritis
P006,41,F,abdominal_pain_bloating,IBS
P007,39,M,back_pain_muscle_weakness,Herniated_Disc
P008,52,F,dizziness_balance_issues,Vertigo
P009,35,M,skin_rash_itching,Eczema
P010,48,F,weight_loss_fatigue,Thyroid_Disorder"""
    
    with open(sample_folder / "patient_data.txt", 'w') as f:
        f.write(text1_content)
    
    # Text file 2: Key-value pairs format
    text2_content = """patient_id: P011
age: 29
gender: F
blood_pressure: 120/80
cholesterol: 180
glucose: 95
diagnosis: Normal

patient_id: P012
age: 63
gender: M
blood_pressure: 160/100
cholesterol: 280
glucose: 140
diagnosis: Hypertension

patient_id: P013
age: 47
gender: F
blood_pressure: 140/90
cholesterol: 220
glucose: 110
diagnosis: Pre-diabetes

patient_id: P014
age: 72
gender: M
blood_pressure: 180/110
cholesterol: 320
glucose: 180
diagnosis: Diabetes

patient_id: P015
age: 38
gender: F
blood_pressure: 110/70
cholesterol: 160
glucose: 88
diagnosis: Normal"""
    
    with open(sample_folder / "lab_results.txt", 'w') as f:
        f.write(text2_content)
    
    print(f"✅ Created sample folder: {sample_folder.absolute()}")
    print(f"   📄 patient_data.txt (CSV-like format)")
    print(f"   📄 lab_results.txt (key-value format)")
    
    return sample_folder

def demonstrate_folder_processing():
    """Demonstrate how folder input works with the research system"""
    
    print("🚀 DEMONSTRATION: FOLDER INPUT WITH TEXT FILES")
    print("="*60)
    
    # Create sample folder
    sample_folder = create_sample_text_folder()
    
    try:
        # Import the folder manager
        from utils.enhanced_folder_dataset_manager import EnhancedFolderDatasetManager
        
        # Initialize folder manager
        folder_manager = EnhancedFolderDatasetManager()
        
        print(f"\n🔍 STEP 1: Analyzing folder contents...")
        
        # Get folder summary
        summary = folder_manager.get_folder_summary(str(sample_folder))
        print(f"📊 Folder Summary:")
        print(f"   📁 Folder: {summary['folder_path']}")
        print(f"   📄 Total files: {summary['total_files']}")
        print(f"   ✅ Supported files: {summary['supported_files']}")
        print(f"   💾 Total size: {summary['total_size_mb']} MB")
        print(f"   📋 Files by format: {summary.get('files_by_format', {})}")
        
        print(f"\n🔄 STEP 2: Processing folder with concatenation strategy...")
        
        # Process the folder
        result = folder_manager.process_folder_input(
            folder_path=str(sample_folder),
            target_column=None,  # Auto-detect
            combine_strategy='concatenate',
            user_hint="medical patient data"
        )
        
        if result['success']:
            print(f"\n✅ FOLDER PROCESSING SUCCESS!")
            
            # Show processing results
            print(f"📊 Processing Results:")
            print(f"   📄 Files processed: {result['files_processed']}/{result['total_files_found']}")
            
            if 'combined_dataset' in result:
                combined_data = result['combined_dataset']
                print(f"   📈 Combined dataset shape: {combined_data['shape']}")
                print(f"   🎯 Target variable: {combined_data['target_column']}")
                print(f"   💾 Memory usage: {combined_data['memory_usage_mb']:.2f} MB")
                print(f"   📋 Columns: {', '.join(combined_data['columns'])}")
                
                # Show data sample
                df = combined_data['dataframe']
                print(f"\n📋 Sample Data (first 3 rows):")
                print(df.head(3).to_string())
                
                # Show target detection
                if 'target_detection' in result:
                    target_info = result['target_detection']
                    print(f"\n🎯 Target Detection:")
                    print(f"   🔍 Detected target: {target_info['detected_target']}")
                    print(f"   📈 Confidence: {target_info['confidence']:.2f}")
                    print(f"   💡 Reasons: {', '.join(target_info['reasons'])}")
                
                # Show individual file processing
                print(f"\n📄 Individual File Processing:")
                for i, file_result in enumerate(result['processed_files'], 1):
                    if 'metadata' in file_result:
                        metadata = file_result['metadata']
                        file_name = Path(metadata['file_path']).name
                        shape = metadata['shape']
                        format_type = metadata['format']
                        parsing = metadata.get('best_strategy', 'N/A')
                        print(f"   {i}. {file_name}")
                        print(f"      📊 Shape: {shape}")
                        print(f"      📝 Format: {format_type}")
                        print(f"      🔧 Parsing: {parsing}")
            
            print(f"\n🎯 HOW TO USE WITH RESEARCH SYSTEM:")
            print(f"✅ The agents will now use this combined dataset containing:")
            print(f"   - Data from {result['files_processed']} text files")
            print(f"   - Automatically detected target variable")
            print(f"   - {combined_data['shape'][0]} total samples")
            print(f"   - {combined_data['shape'][1]} features/columns")
            
            print(f"\n💻 CODE GENERATION will use:")
            print(f"   - Combined dataset with shape {combined_data['shape']}")
            print(f"   - Target variable: {combined_data['target_column']}")
            print(f"   - All {result['files_processed']} files' data merged")
            
            print(f"\n📊 VISUALIZATION will show:")
            print(f"   - Analysis across all files")
            print(f"   - Target variable distribution")
            print(f"   - Feature relationships from combined data")
            
            print(f"\n📄 REPORT will mention:")
            print(f"   - Multi-file dataset processing")
            print(f"   - Text file parsing methodology")
            print(f"   - Combined dataset characteristics")
        
        else:
            print(f"❌ FOLDER PROCESSING FAILED: {result['error']}")
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the enhanced folder dataset manager is available")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if sample_folder.exists():
            shutil.rmtree(sample_folder)
            print(f"\n🧹 Cleaned up demo folder")

def show_integration_example():
    """Show how to integrate folder input with the main research system"""
    
    print(f"\n📚 INTEGRATION WITH MAIN RESEARCH SYSTEM:")
    print("="*50)
    
    print("""
🔧 To use folder input in your research workflow:

1. 📁 Create a folder with your data files:
   my_research_data/
   ├── experiment1.txt    (CSV-like format)
   ├── experiment2.txt    (key-value format)
   ├── measurements.csv   (standard CSV)
   └── results.xlsx       (Excel file)

2. 🔄 Process the folder:
   ```python
   from utils.enhanced_folder_dataset_manager import process_folder_dataset
   
   result = process_folder_dataset(
       folder_path="my_research_data",
       combine_strategy="concatenate",
       user_hint="medical research data"
   )
   ```

3. 🎯 The system will:
   ✅ Auto-detect all supported files (.txt, .csv, .xlsx, .json)
   ✅ Parse text files with intelligent strategies
   ✅ Combine all data into unified dataset
   ✅ Detect target variable automatically
   ✅ Pass combined data to all agents

4. 🤖 All agents will use the combined dataset:
   - 🧠 Hypothesis Agent: Generate hypothesis for combined data
   - 💻 Code Agent: Create ML code using all files' data
   - 📊 Visualization Agent: Show analysis across all files
   - 📄 Report Agent: Document multi-file methodology

✨ RESULT: Complete research pipeline using multiple text files!
""")

if __name__ == "__main__":
    demonstrate_folder_processing()
    show_integration_example()
    
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print("="*60)
    print("✅ Folder input with text files is fully supported!")
    print("✅ All agents can use the combined dataset!")
    print("✅ Ready for production research workflows!") 
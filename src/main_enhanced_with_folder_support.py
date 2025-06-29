#!/usr/bin/env python3
"""
Enhanced Main System with Folder Input Support

This version supports:
- Folder input containing multiple files (CSV, Excel, JSON, TXT)
- Text file intelligent parsing
- Multi-file concatenation and processing
- All existing agent functionality with folder data
"""

import os
import sys
sys.path.append('src')

# Import all existing agents and utilities
from agents.enhanced_code_agent import EnhancedCodeAgent
from agents.enhanced_visualization_agent import EnhancedVisualizationAgent  
from agents.enhanced_report_agent import EnhancedReportAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.web_search_agent import WebSearchAgent
from agents.note_taker import NoteTaker
from user_dataset_manager import UserDatasetManager

# Import the new folder dataset manager
from utils.enhanced_folder_dataset_manager import EnhancedFolderDatasetManager, process_folder_dataset

import datetime
import uuid

def print_step(step_num, description):
    """Print formatted step header"""
    print(f"\n{'='*70}")
    print(f"🔄 STEP {step_num}: {description}")
    print(f"{'='*70}")

def main_with_folder_support():
    """Main function with folder input support"""
    
    print("🚀 ENHANCED RESEARCH ASSISTANT WITH FOLDER SUPPORT")
    print("="*70)
    print("Features:")
    print("✅ Individual file input (CSV, Excel, JSON)")
    print("✅ Folder input with multiple files")
    print("✅ Text file support (.txt) with intelligent parsing")
    print("✅ Multi-file concatenation and analysis")
    print("✅ All existing agent functionality")
    print("="*70)
    
    # Initialize agents
    OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
    if not OPENAI_API_KEY:
        print("❌ Error: CHATGPT_API_KEY environment variable not set")
        return
    
    # Mock note taker for demo
    class MockNoteTaker:
        def log(self, *args, **kwargs):
            pass
        def log_session_start(self, *args, **kwargs):
            pass
        def log_query(self, *args, **kwargs):
            pass
        def log_hypothesis(self, *args, **kwargs):
            pass
        def log_code(self, *args, **kwargs):
            pass
        def log_insights(self, *args, **kwargs):
            pass
        def log_visualization(self, *args, **kwargs):
            pass
        def log_report(self, *args, **kwargs):
            pass
    
    note_taker = MockNoteTaker()
    
    # Initialize agents
    web_search_agent = WebSearchAgent(note_taker)
    hypothesis_agent = HypothesisAgent(OPENAI_API_KEY, note_taker)
    enhanced_code_agent = EnhancedCodeAgent(OPENAI_API_KEY, note_taker)
    enhanced_viz_agent = EnhancedVisualizationAgent(note_taker)
    enhanced_report_agent = EnhancedReportAgent(note_taker)
    dataset_manager = UserDatasetManager(note_taker)
    folder_manager = EnhancedFolderDatasetManager()  # New folder manager
    
    print("✅ All Enhanced Agents Initialized (with Folder Support)")
    
    # Get research query
    print("\n🎯 RESEARCH QUERY INPUT")
    query = input("Enter your research topic or hypothesis: ").strip()
    if not query:
        query = "Machine learning approaches for predictive modeling and data analysis"
        print(f"Using default query: {query}")
    
    # Dataset input configuration with folder support
    print("\n💾 DATASET INPUT CONFIGURATION")
    print("Choose your dataset input option:")
    print("1. 📄 Single file (CSV, Excel, JSON)")
    print("2. 📁 Folder containing multiple files")
    print("3. 🚫 No dataset (use synthetic data)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    dataset_analysis = None
    combined_dataframe = None
    
    if choice == "1":
        # Single file input (existing functionality)
        print("📄 Single file input selected")
        use_custom_dataset = input("Do you want to use your own dataset file? (yes/no): ").strip().lower()
        
        if use_custom_dataset == 'yes':
            file_name = input("Enter dataset filename (e.g., 'data.csv'): ").strip()
            
            # Try multiple possible locations
            possible_paths = [
                os.path.join("src", "user_datasets", file_name),
                os.path.join("user_datasets", file_name),
                file_name,
                os.path.join("src", file_name)
            ]
            
            dataset_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    dataset_path = path
                    print(f"✅ Found dataset at: {dataset_path}")
                    break
            
            if dataset_path:
                df = dataset_manager.load_dataset(dataset_path)
                if df is not None:
                    df, new_to_original_mapping, original_to_new_mapping = dataset_manager.sanitize_columns(df)
                    
                    target_variable = input("Enter target variable name: ").strip()
                    if target_variable in original_to_new_mapping:
                        target_variable = original_to_new_mapping[target_variable]
                    
                    dataset_analysis = dataset_manager.analyze_dataset(df, target_variable, new_to_original_mapping)
                    combined_dataframe = df
                    print(f"✅ Single file processed: {df.shape}")
            else:
                print(f"❌ File '{file_name}' not found")
    
    elif choice == "2":
        # Folder input (new functionality)
        print("📁 Folder input selected")
        folder_path = input("Enter folder path (e.g., 'my_data_folder'): ").strip()
        
        if not folder_path:
            folder_path = "src/user_datasets"
            print(f"Using default folder: {folder_path}")
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"❌ Folder '{folder_path}' not found")
            print("Continuing without dataset...")
        else:
            print(f"🔍 Processing folder: {folder_path}")
            
            # Get folder summary first
            folder_summary = folder_manager.get_folder_summary(folder_path)
            print(f"📊 Folder Summary:")
            print(f"   📁 Total files: {folder_summary['total_files']}")
            print(f"   ✅ Supported files: {folder_summary['supported_files']}")
            print(f"   📋 Files by format: {folder_summary.get('files_by_format', {})}")
            
            if folder_summary['supported_files'] > 0:
                # Process the folder
                combine_strategy = input("Combination strategy (concatenate/separate) [concatenate]: ").strip()
                if not combine_strategy:
                    combine_strategy = "concatenate"
                
                user_hint = input("Data description hint (optional): ").strip()
                if not user_hint:
                    user_hint = "research data for analysis"
                
                print(f"🔄 Processing folder with '{combine_strategy}' strategy...")
                folder_result = folder_manager.process_folder_input(
                    folder_path=folder_path,
                    target_column=None,  # Auto-detect
                    combine_strategy=combine_strategy,
                    user_hint=user_hint
                )
                
                if folder_result['success']:
                    print(f"✅ Folder processing successful!")
                    print(f"   📄 Files processed: {folder_result['files_processed']}")
                    
                    if combine_strategy == "concatenate" and 'combined_dataset' in folder_result:
                        combined_data = folder_result['combined_dataset']
                        combined_dataframe = combined_data['dataframe']
                        target_column = combined_data['target_column']
                        
                        print(f"   📊 Combined dataset shape: {combined_data['shape']}")
                        print(f"   🎯 Target variable: {target_column}")
                        
                        # Convert to the format expected by existing agents
                        dataset_analysis = {
                            'shape': combined_data['shape'],
                            'total_rows': combined_data['shape'][0],
                            'total_columns': combined_data['shape'][1],
                            'columns': combined_data['columns'],
                            'target_info': {
                                'target_variable': target_column,
                                'task_type': 'classification'  # Default
                            },
                            'missing_percentage': combined_data['analysis']['missing_data']['missing_percentage'],
                            'class_balance': combined_data['analysis'].get('target_analysis', {}).get('value_counts', {}),
                            'files_processed': folder_result['files_processed'],
                            'source_type': 'folder_input',
                            'combine_strategy': combine_strategy
                        }
                        
                        print(f"   📈 Data quality: {combined_data['analysis']['data_quality']['overall_score']:.1f}/100")
                        
                        # Show files processed
                        print(f"   📂 Files included:")
                        for i, file_result in enumerate(folder_result['processed_files'], 1):
                            if 'metadata' in file_result:
                                metadata = file_result['metadata']
                                file_name = os.path.basename(metadata['file_path'])
                                shape = metadata['shape']
                                format_type = metadata['format']
                                print(f"      {i}. {file_name}: {shape} ({format_type})")
                    
                    elif combine_strategy == "separate":
                        print(f"   📊 Individual datasets: {len(folder_result['separate_datasets'])}")
                        # For simplicity, use the first dataset
                        if folder_result['separate_datasets']:
                            first_dataset = folder_result['separate_datasets'][0]
                            combined_dataframe = first_dataset['dataframe']
                            target_column = first_dataset.get('target_column')
                            
                            dataset_analysis = {
                                'shape': first_dataset['metadata']['shape'],
                                'total_rows': first_dataset['metadata']['shape'][0],
                                'total_columns': first_dataset['metadata']['shape'][1],
                                'columns': first_dataset['metadata']['columns'],
                                'target_info': {'target_variable': target_column, 'task_type': 'classification'},
                                'source_type': 'folder_input_first',
                                'combine_strategy': combine_strategy
                            }
                            print(f"   📄 Using first dataset: {first_dataset['metadata']['shape']}")
                
                else:
                    print(f"❌ Folder processing failed: {folder_result['error']}")
            else:
                print("❌ No supported files found in folder")
    
    else:
        print("🚫 No dataset selected - will use synthetic data")
    
    # Continue with existing pipeline...
    print_step(1, "PAPER SEARCH & RETRIEVAL")
    print("🔍 Searching for relevant papers...")
    papers = web_search_agent.search_papers(query, max_papers=5)
    
    if papers:
        print(f"✅ Found {len(papers)} relevant papers")
        for i, paper in enumerate(papers, 1):
            print(f"   {i}. {paper.get('title', 'Unknown title')[:80]}...")
    else:
        print("❌ No papers found, proceeding with local knowledge")
        papers = []
    
    print_step(2, "HYPOTHESIS GENERATION")
    print("🧠 Generating research hypothesis...")
    hypothesis = hypothesis_agent.generate_hypothesis(papers, dataset_analysis)
    print(f"✅ Generated hypothesis: {hypothesis}")
    
    # Create project folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    project_folder = f"output/project_{timestamp}"
    os.makedirs(project_folder, exist_ok=True)
    print(f"📁 Created project folder: {project_folder}")
    
    # Set project folder for all agents
    enhanced_code_agent.set_project_folder(project_folder)
    enhanced_viz_agent.set_project_folder(project_folder)
    enhanced_report_agent.set_project_folder(project_folder)
    
    print_step(3, "CODE GENERATION")
    print("💻 Generating research code...")
    
    # Generate code with dataset info
    code = enhanced_code_agent.generate_code(hypothesis, dataset_analysis)
    
    if code:
        print(f"✅ Code generated: {len(code)} characters")
        
        # Save code to project folder
        code_file_path = os.path.join(project_folder, "generated_research_code.py")
        with open(code_file_path, 'w') as f:
            f.write(code)
        print(f"💾 Code saved to: {code_file_path}")
        
        # Show dataset info used in code generation
        if dataset_analysis:
            if dataset_analysis.get('source_type') == 'folder_input':
                print(f"   📂 Used folder data: {dataset_analysis['files_processed']} files")
                print(f"   📊 Combined shape: {dataset_analysis['shape']}")
                print(f"   🎯 Target: {dataset_analysis['target_info']['target_variable']}")
            else:
                print(f"   📄 Used single file data: {dataset_analysis['shape']}")
    else:
        print("❌ Code generation failed")
    
    print_step(4, "VISUALIZATION GENERATION")
    print("📊 Generating visualizations...")
    visualizations = enhanced_viz_agent.generate_visualizations(hypothesis, dataset_analysis)
    
    if visualizations:
        print(f"✅ Generated {len(visualizations)} visualizations")
        for viz in visualizations:
            print(f"   📈 {viz.get('title', 'Unnamed')}: {viz.get('type', 'Unknown type')}")
    else:
        print("❌ Visualization generation failed")
    
    print_step(5, "REPORT GENERATION")
    print("📄 Generating research report...")
    
    report_data = {
        'hypothesis': hypothesis,
        'visualizations': visualizations,
        'code': code,
        'references': papers,
        'dataset_summary': dataset_analysis,
        'model_results': {}
    }
    
    report = enhanced_report_agent.generate_enhanced_academic_report(report_data)
    
    if report:
        print(f"✅ Research report generated: {len(report)} characters")
        
        # Save report
        report_file = os.path.join(project_folder, "research_report.md")
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"💾 Report saved to: {report_file}")
    else:
        print("❌ Report generation failed")
    
    # Final summary
    print(f"\n🎯 RESEARCH SESSION COMPLETE!")
    print(f"📁 Project folder: {project_folder}")
    print(f"📄 Files generated:")
    for file in os.listdir(project_folder):
        print(f"   - {file}")
    
    if dataset_analysis and dataset_analysis.get('source_type', '').startswith('folder_input'):
        print(f"\n✨ FOLDER INPUT SUCCESS:")
        print(f"   📂 Processed {dataset_analysis.get('files_processed', 0)} files")
        print(f"   📊 Final dataset: {dataset_analysis.get('shape', 'Unknown shape')}")
        print(f"   🎯 Target variable: {dataset_analysis.get('target_info', {}).get('target_variable', 'Unknown')}")
        print(f"   🔄 Strategy: {dataset_analysis.get('combine_strategy', 'Unknown')}")

if __name__ == "__main__":
    main_with_folder_support() 
#!/usr/bin/env python3
"""
Enhanced Folder Dataset Manager

This module provides comprehensive folder and multi-file dataset management with:
- Folder input processing (batch processing)
- Text file support (CSV, TSV, JSON, TXT, Excel)
- Multi-file concatenation and analysis
- Advanced data validation and quality checks
"""

import os
import pandas as pd
import numpy as np
import json
import io
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
import logging
from datetime import datetime
import re

# Import existing enhanced modules
from src.utils.enhanced_target_detection import (
    EnhancedTargetDetector, detect_target_variable
)
from src.utils.enhanced_data_validation import (
    EnhancedDataLoader, DataFileValidator, DataQualityAnalyzer,
    validate_data_file, load_and_validate_data
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedFolderDatasetManager:
    """
    Advanced folder and multi-file dataset manager with comprehensive features:
    - Multi-format support (CSV, Excel, JSON, TSV, TXT)
    - Folder batch processing
    - Text file parsing and structuring
    - Multi-file concatenation strategies
    - Intelligent target variable detection across files
    """
    
    def __init__(self, user_datasets_folder: str = "user_datasets"):
        """
        Initialize Enhanced Folder Dataset Manager
        
        Args:
            user_datasets_folder: Folder containing user datasets
        """
        self.user_datasets_folder = user_datasets_folder
        
        # Initialize components
        self.target_detector = EnhancedTargetDetector()
        self.data_loader = EnhancedDataLoader()
        self.quality_analyzer = DataQualityAnalyzer()
        
        # Supported file formats with text file support
        self.supported_formats = {
            '.csv': 'Comma-separated values',
            '.tsv': 'Tab-separated values', 
            '.xlsx': 'Excel spreadsheet',
            '.xls': 'Excel legacy format',
            '.xlsm': 'Excel macro-enabled',
            '.json': 'JSON format',
            '.jsonl': 'JSON Lines format',
            '.txt': 'Text file'
        }
        
        # Text file parsing strategies
        self.text_parsing_strategies = [
            'comma_separated',
            'tab_separated',
            'space_separated',
            'pipe_separated',
            'semicolon_separated',
            'structured_text',
            'key_value_pairs'
        ]
        
        # Create datasets folder if it doesn't exist
        self._setup_datasets_folder()
        
        logger.info(f"Enhanced Folder Dataset Manager initialized. Supported formats: {list(self.supported_formats.keys())}")
    
    def _setup_datasets_folder(self):
        """Setup the datasets folder structure"""
        if not os.path.exists(self.user_datasets_folder):
            os.makedirs(self.user_datasets_folder)
            logger.info(f"Created datasets folder: {self.user_datasets_folder}")
    
    def process_folder_input(self, folder_path: str, 
                           target_column: Optional[str] = None,
                           combine_strategy: str = 'concatenate',
                           user_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all supported files in a folder
        
        Args:
            folder_path: Path to folder containing data files
            target_column: Target column name (auto-detect if None)
            combine_strategy: How to combine multiple files ('concatenate', 'separate', 'merge')
            user_hint: User hint for data interpretation
            
        Returns:
            Comprehensive folder processing results
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            return {
                'success': False,
                'error': f"Folder does not exist: {folder_path}",
                'files_processed': 0
            }
        
        # Discover all supported files in folder
        supported_files = self._discover_supported_files(folder_path)
        
        if not supported_files:
            return {
                'success': False,
                'error': f"No supported files found in folder: {folder_path}",
                'files_processed': 0,
                'supported_formats': list(self.supported_formats.keys())
            }
        
        print(f"📁 Processing folder: {folder_path}")
        print(f"🔍 Found {len(supported_files)} supported files:")
        for file_info in supported_files:
            print(f"   - {file_info['filename']} ({file_info['format']})")
        
        # Process each file
        processed_files = []
        combined_dataframe = None
        
        for file_info in supported_files:
            print(f"\n📄 Processing: {file_info['filename']}")
            
            file_result = self._process_single_file(
                file_info['path'], 
                file_info['format'], 
                target_column, 
                user_hint
            )
            
            if file_result['success']:
                processed_files.append(file_result)
                print(f"   ✅ Successfully processed: {file_result.get('dataframe', 'No dataframe').shape if hasattr(file_result.get('dataframe'), 'shape') else 'Unknown shape'}")
                
                # Combine files according to strategy
                if combine_strategy == 'concatenate':
                    if combined_dataframe is None:
                        combined_dataframe = file_result['dataframe'].copy()
                        print(f"   📊 Initial dataset: {combined_dataframe.shape}")
                    else:
                        print(f"   🔗 Combining with existing data: {combined_dataframe.shape} + {file_result['dataframe'].shape}")
                        combined_dataframe = self._safe_concatenate(
                            combined_dataframe, 
                            file_result['dataframe']
                        )
                        print(f"   📊 Combined dataset: {combined_dataframe.shape}")
            else:
                print(f"   ❌ Failed to process {file_info['filename']}: {file_result.get('error', 'Unknown error')}")
        
        print(f"\n✅ Successfully combined {len(processed_files)} files")
        if combined_dataframe is not None:
            print(f"📊 Combined dataset shape: {combined_dataframe.shape}")
        else:
            print("❌ No data combined")
        
        # Generate comprehensive results
        result = {
            'success': True,
            'files_processed': len(processed_files),
            'total_files_found': len(supported_files),
            'combine_strategy': combine_strategy,
            'processed_files': processed_files,
            'folder_path': str(folder_path)
        }
        
        if combine_strategy == 'concatenate' and combined_dataframe is not None:
            # Enhanced target detection on combined data
            if target_column is None:
                target_candidates = self.target_detector.detect_target_variables(
                    combined_dataframe, user_hint
                )
                if target_candidates:
                    target_column = target_candidates[0].column_name
                    result['target_detection'] = {
                        'detected_target': target_column,
                        'confidence': target_candidates[0].confidence_score,
                        'reasons': target_candidates[0].reasons
                    }
            
            # Comprehensive analysis of combined dataset
            result['combined_dataset'] = {
                'dataframe': combined_dataframe,
                'shape': combined_dataframe.shape,
                'columns': combined_dataframe.columns.tolist(),
                'target_column': target_column,
                'memory_usage_mb': combined_dataframe.memory_usage(deep=True).sum() / (1024 * 1024),
                'analysis': self._analyze_dataset_comprehensive(
                    combined_dataframe, target_column, f"combined_{len(processed_files)}_files"
                )
            }
        
        elif combine_strategy == 'separate':
            result['separate_datasets'] = processed_files
        
        return result
    
    def _discover_supported_files(self, folder_path: Path) -> List[Dict[str, Any]]:
        """Discover all supported files in a folder"""
        supported_files = []
        
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                
                if file_ext in self.supported_formats:
                    file_stats = file_path.stat()
                    
                    supported_files.append({
                        'filename': file_path.name,
                        'path': file_path,
                        'format': self.supported_formats[file_ext],
                        'extension': file_ext,
                        'size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                        'modified': datetime.fromtimestamp(file_stats.st_mtime)
                    })
        
        # Sort by modification time (newest first)
        supported_files.sort(key=lambda x: x['modified'], reverse=True)
        return supported_files
    
    def _process_single_file(self, file_path: Path, file_format: str, 
                           target_column: Optional[str] = None,
                           user_hint: Optional[str] = None) -> Dict[str, Any]:
        """Process a single file with format-specific handling"""
        
        try:
            file_ext = file_path.suffix.lower()
            
            # For .txt files, try tab-separated first if it looks like tabular data
            if file_ext == '.txt':
                # Peek at the first line to see if it's tab-separated
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_lines = f.readlines()[:20]  # Read first 20 lines to detect GEO format
                    
                    # Check if it's a GEO Series Matrix file
                    is_geo_file = any(line.startswith('!') for line in first_lines[:10])
                    
                    if is_geo_file:
                        print(f"   🧬 Detected GEO Series Matrix format")
                        return self._process_geo_series_matrix(file_path, target_column, user_hint)
                    
                    first_line = first_lines[0].strip() if first_lines else ""
                    
                    print(f"   🔍 First line length: {len(first_line)}")
                    print(f"   🔍 Tab count: {first_line.count(chr(9))}")
                    print(f"   🔍 Split by tab count: {len(first_line.split(chr(9)))}")
                    
                    if '\t' in first_line and len(first_line.split('\t')) > 3:
                        print(f"   ✅ Detected tab-separated format")
                        # Looks like tab-separated data, try pandas directly
                        try:
                            df = pd.read_csv(file_path, delimiter='\t', low_memory=False)
                            print(f"   📊 Pandas read result: {df.shape}")
                            if len(df) > 0 and len(df.columns) > 1:
                                # Successfully parsed as tab-separated
                                print(f"   ✅ Parsed as tab-separated: {df.shape}")
                                
                                # Try quality analysis with error handling
                                try:
                                    quality_report = self.quality_analyzer.analyze_quality(df)
                                except Exception as quality_error:
                                    print(f"   ⚠️ Quality analysis error: {quality_error}")
                                    quality_report = {'error': str(quality_error)}
                                
                                # Try target detection
                                target_detection_info = {}
                                if target_column is None:
                                    try:
                                        target_candidates = self.target_detector.detect_target_variables(df, user_hint)
                                        if target_candidates:
                                            target_column = target_candidates[0].column_name
                                            target_detection_info = {
                                                'detected_target': target_column,
                                                'confidence': target_candidates[0].confidence_score,
                                                'reasons': target_candidates[0].reasons
                                            }
                                    except Exception as target_error:
                                        print(f"   ⚠️ Target detection error: {target_error}")
                                
                                return {
                                    'success': True,
                                    'dataframe': df,
                                    'parsing_strategy': 'tab_separated_direct',
                                    'target_column': target_column,
                                    'target_detection': target_detection_info,
                                    'quality_report': quality_report,
                                    'metadata': {
                                        'file_path': str(file_path),
                                        'format': 'Tab-separated text file',
                                        'shape': df.shape,
                                        'columns': df.columns.tolist(),
                                        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
                                    }
                                }
                            else:
                                print(f"   ⚠️ Tab-separated parsing gave empty result")
                        except Exception as e:
                            print(f"   ⚠️ Tab-separated parsing error: {e}")
                            pass  # Fall back to regular text processing
                    else:
                        print(f"   ℹ️ Not tab-separated format (tabs: {first_line.count(chr(9))}, splits: {len(first_line.split(chr(9)))})")
                except Exception as e:
                    print(f"   ⚠️ Error reading file for format detection: {e}")
                
                # Fall back to regular text file processing
                print(f"   🔄 Falling back to text processing...")
                return self._process_text_file(file_path, target_column, user_hint)
            
            # Use existing enhanced data loader for supported formats
            elif file_ext in ['.csv', '.tsv', '.xlsx', '.xls', '.xlsm', '.json', '.jsonl']:
                return self.data_loader.load_data(file_path, target_column, validate=True)
            
            else:
                return {
                    'success': False,
                    'error': f"Unsupported file format: {file_ext}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Error processing file {file_path.name}: {str(e)}"
            }
    
    def _process_text_file(self, file_path: Path, 
                          target_column: Optional[str] = None,
                          user_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Process text files with intelligent parsing
        
        Attempts multiple parsing strategies to structure the text data
        """
        
        try:
            # Read the text file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try different parsing strategies
            parsing_results = []
            
            for strategy in self.text_parsing_strategies:
                try:
                    parsed_df = self._parse_text_with_strategy(content, strategy)
                    if parsed_df is not None and len(parsed_df) > 0:
                        parsing_results.append({
                            'strategy': strategy,
                            'dataframe': parsed_df,
                            'columns': len(parsed_df.columns),
                            'rows': len(parsed_df),
                            'success_score': self._evaluate_parsing_success(parsed_df)
                        })
                except Exception as e:
                    continue  # Try next strategy
            
            if not parsing_results:
                # Fallback: treat as single-column text data
                lines = content.strip().split('\\n')
                parsed_df = pd.DataFrame({'text_content': lines})
                parsing_results.append({
                    'strategy': 'single_column_text',
                    'dataframe': parsed_df,
                    'columns': 1,
                    'rows': len(parsed_df),
                    'success_score': 0.3  # Low score for single column
                })
            
            # Select best parsing result
            best_result = max(parsing_results, key=lambda x: x['success_score'])
            df = best_result['dataframe']
            
            # Enhanced target detection
            target_detection_info = {}
            if target_column is None:
                target_candidates = self.target_detector.detect_target_variables(df, user_hint)
                if target_candidates:
                    target_column = target_candidates[0].column_name
                    target_detection_info = {
                        'detected_target': target_column,
                        'confidence': target_candidates[0].confidence_score,
                        'reasons': target_candidates[0].reasons
                    }
            
            # Data quality analysis
            quality_report = self.quality_analyzer.analyze_quality(df)
            
            return {
                'success': True,
                'dataframe': df,
                'parsing_strategy': best_result['strategy'],
                'target_column': target_column,
                'target_detection': target_detection_info,
                'quality_report': quality_report,
                'metadata': {
                    'file_path': str(file_path),
                    'format': f"Text file ({best_result['strategy']})",
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error processing text file {file_path.name}: {str(e)}"
            }
    
    def _parse_text_with_strategy(self, content: str, strategy: str) -> Optional[pd.DataFrame]:
        """Parse text content using a specific strategy"""
        
        lines = content.strip().split('\\n')
        if len(lines) < 2:  # Need at least header and one data row
            return None
        
        try:
            if strategy == 'comma_separated':
                return pd.read_csv(io.StringIO(content))
            
            elif strategy == 'tab_separated':
                return pd.read_csv(io.StringIO(content), delimiter='\\t')
            
            elif strategy == 'space_separated':
                return pd.read_csv(io.StringIO(content), delimiter=' ', skipinitialspace=True)
            
            elif strategy == 'pipe_separated':
                return pd.read_csv(io.StringIO(content), delimiter='|')
            
            elif strategy == 'semicolon_separated':
                return pd.read_csv(io.StringIO(content), delimiter=';')
            
            elif strategy == 'structured_text':
                # Try to parse structured text (e.g., key-value pairs)
                return self._parse_structured_text(lines)
            
            elif strategy == 'key_value_pairs':
                # Parse key: value format
                return self._parse_key_value_pairs(lines)
            
        except Exception:
            return None
        
        return None
    
    def _parse_structured_text(self, lines: List[str]) -> Optional[pd.DataFrame]:
        """Parse structured text format"""
        data = []
        headers = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for patterns like "field1: value1, field2: value2"
            if ':' in line and ',' in line:
                record = {}
                parts = line.split(',')
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        record[key] = value
                
                if record:
                    if headers is None:
                        headers = list(record.keys())
                    data.append(record)
        
        if data and len(data) >= 2:  # Need at least 2 records
            return pd.DataFrame(data)
        
        return None
    
    def _parse_key_value_pairs(self, lines: List[str]) -> Optional[pd.DataFrame]:
        """Parse key-value pair format"""
        data = []
        current_record = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_record:
                    data.append(current_record)
                    current_record = {}
                continue
            
            if ':' in line:
                key, value = line.split(':', 1)
                current_record[key.strip()] = value.strip()
        
        # Add last record if exists
        if current_record:
            data.append(current_record)
        
        if data and len(data) >= 2:
            return pd.DataFrame(data)
        
        return None
    
    def _evaluate_parsing_success(self, df: pd.DataFrame) -> float:
        """Evaluate how successful a parsing strategy was"""
        score = 0.0
        
        # More columns generally better (but not too many)
        if 2 <= len(df.columns) <= 20:
            score += 0.3
        elif len(df.columns) == 1:
            score += 0.1
        
        # More rows generally better
        if len(df) >= 10:
            score += 0.3
        elif len(df) >= 2:
            score += 0.2
        
        # Check for reasonable data types
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        if numeric_cols > 0:
            score += 0.2
        
        # Check for non-null data
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        score += completeness * 0.2
        
        return min(score, 1.0)
    
    def _safe_concatenate(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Safely concatenate two dataframes with different schemas"""
        
        # Get all unique columns
        all_columns = list(set(df1.columns.tolist() + df2.columns.tolist()))
        
        # Reindex both dataframes to have the same columns
        df1_reindexed = df1.reindex(columns=all_columns)
        df2_reindexed = df2.reindex(columns=all_columns)
        
        # Concatenate
        return pd.concat([df1_reindexed, df2_reindexed], ignore_index=True)
    
    def _analyze_dataset_comprehensive(self, df: pd.DataFrame, target_column: str, filename: str) -> Dict[str, Any]:
        """Comprehensive dataset analysis"""
        
        analysis = {
            'basic_info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            'missing_data': {
                'total_missing': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'missing_by_column': df.isnull().sum().to_dict()
            }
        }
        
        # Try data quality analysis with error handling
        try:
            quality_report = self.quality_analyzer.analyze_quality(df)
            analysis['data_quality'] = quality_report.__dict__
        except Exception as quality_error:
            print(f"   ⚠️ Quality analysis error: {quality_error}")
            analysis['data_quality'] = {
                'error': str(quality_error),
                'overall_score': 75.0,  # Default score
                'completeness_score': 75.0,
                'consistency_score': 75.0,
                'validity_score': 75.0,
                'uniqueness_score': 75.0,
                'issues': [],
                'recommendations': ['Quality analysis skipped due to error']
            }
        
        # Target variable analysis if provided
        if target_column and target_column in df.columns:
            try:
                target_series = df[target_column]
                analysis['target_analysis'] = {
                    'unique_values': target_series.nunique(),
                    'value_counts': target_series.value_counts().to_dict(),
                    'data_type': str(target_series.dtype),
                    'missing_count': target_series.isnull().sum()
                }
            except Exception as target_error:
                print(f"   ⚠️ Target analysis error: {target_error}")
                analysis['target_analysis'] = {'error': str(target_error)}
        
        return analysis
    
    def get_folder_summary(self, folder_path: str) -> Dict[str, Any]:
        """Get a summary of all files in a folder without processing them"""
        
        folder_path = Path(folder_path)
        if not folder_path.exists():
            return {'error': f'Folder does not exist: {folder_path}'}
        
        supported_files = self._discover_supported_files(folder_path)
        
        summary = {
            'folder_path': str(folder_path),
            'total_files': len(list(folder_path.iterdir())),
            'supported_files': len(supported_files),
            'supported_formats': list(self.supported_formats.keys()),
            'files_by_format': {},
            'total_size_mb': 0,
            'file_details': supported_files
        }
        
        # Aggregate by format
        for file_info in supported_files:
            format_name = file_info['format']
            if format_name not in summary['files_by_format']:
                summary['files_by_format'][format_name] = 0
            summary['files_by_format'][format_name] += 1
            summary['total_size_mb'] += file_info['size_mb']
        
        summary['total_size_mb'] = round(summary['total_size_mb'], 2)
        
        return summary
    
    def process_folder(self, folder_path: str) -> Dict[str, Any]:
        """
        Process folder - wrapper method for compatibility with main system
        
        Args:
            folder_path: Path to folder containing data files
            
        Returns:
            Processing results with success flag, dataset, and combined_info
        """
        result = self.process_folder_input(folder_path)
        
        if result['success']:
            # Extract the combined dataset if available
            combined_data = result.get('combined_dataset', {})
            dataset = combined_data.get('dataframe')
            
            # Extract filenames from processed files with error handling
            files_processed = []
            for f in result.get('processed_files', []):
                if isinstance(f, dict):
                    # Try different possible keys for filename
                    filename = None
                    if 'filename' in f:
                        filename = f['filename']
                    elif 'metadata' in f and isinstance(f['metadata'], dict):
                        if 'file_path' in f['metadata']:
                            filename = os.path.basename(f['metadata']['file_path'])
                        elif 'filename' in f['metadata']:
                            filename = f['metadata']['filename']
                    elif 'file_path' in f:
                        filename = os.path.basename(f['file_path'])
                    
                    if filename:
                        files_processed.append(filename)
                    else:
                        files_processed.append(f"unknown_file_{len(files_processed)}")
            
            # Create combined_info structure expected by main system
            combined_info = {
                'files_processed': files_processed,
                'total_files': result.get('total_files_found', 0),
                'combine_strategy': result.get('combine_strategy', 'concatenate'),
                'folder_path': result.get('folder_path', folder_path)
            }
            
            return {
                'success': True,
                'dataset': dataset,
                'combined_info': combined_info,
                'result': result
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'dataset': None,
                'combined_info': None
            }
    
    def auto_detect_target(self, df: pd.DataFrame) -> Optional[str]:
        """
        Auto-detect target variable in dataframe
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Detected target column name or None
        """
        if df is None or df.empty:
            return None
            
        try:
            # Use enhanced target detector
            candidates = self.target_detector.detect_target_variables(df)
            if candidates and len(candidates) > 0:
                return candidates[0].column_name
            
            # Fallback: look for common target column names
            common_targets = [
                'target', 'label', 'class', 'outcome', 'diagnosis', 'result',
                'y', 'dependent', 'response', 'prediction', 'classification'
            ]
            
            df_columns_lower = [col.lower() for col in df.columns]
            
            for target_name in common_targets:
                if target_name in df_columns_lower:
                    # Return original column name (with correct case)
                    idx = df_columns_lower.index(target_name)
                    return df.columns[idx]
                    
            # Look for columns with target-like patterns
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in ['target', 'label', 'class', 'outcome']):
                    return col
                    
            return None
            
        except Exception as e:
            logger.error(f"Error in auto_detect_target: {e}")
            return None

    def _process_geo_series_matrix(self, file_path: Path, 
                                  target_column: Optional[str] = None,
                                  user_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Process GEO Series Matrix files
        
        These files have metadata lines starting with '!' followed by a tab-separated data matrix
        """
        
        try:
            print(f"   🧬 Processing GEO Series Matrix file...")
            
            # Read the file and separate metadata from data
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find where metadata ends and data begins
            data_start_line = 0
            metadata = {}
            
            for i, line in enumerate(lines):
                if line.startswith('!'):
                    # Parse metadata
                    parts = line[1:].strip().split('\t', 1)
                    if len(parts) == 2:
                        key, value = parts
                        metadata[key] = value
                else:
                    # First non-metadata line
                    data_start_line = i
                    break
            
            print(f"   📊 Found {len(metadata)} metadata entries")
            print(f"   📊 Data starts at line {data_start_line + 1}")
            
            # Extract data lines
            data_lines = lines[data_start_line:]
            data_content = ''.join(data_lines)
            
            # Parse as tab-separated data
            df = pd.read_csv(io.StringIO(data_content), delimiter='\t', low_memory=False)
            
            print(f"   ✅ Successfully parsed GEO data: {df.shape}")
            print(f"   📊 Samples: {df.shape[1] - 1}, Genes/Probes: {df.shape[0]}")
            
            # Extract relevant metadata for description
            series_title = metadata.get('Series_title', 'Unknown GEO Series')
            series_accession = metadata.get('Series_geo_accession', 'Unknown')
            
            # Try quality analysis with error handling
            try:
                quality_report = self.quality_analyzer.analyze_quality(df)
            except Exception as quality_error:
                print(f"   ⚠️ Quality analysis error: {quality_error}")
                quality_report = {'error': str(quality_error)}
            
            # Try target detection
            target_detection_info = {}
            if target_column is None:
                try:
                    target_candidates = self.target_detector.detect_target_variables(df, user_hint)
                    if target_candidates:
                        target_column = target_candidates[0].column_name
                        target_detection_info = {
                            'detected_target': target_column,
                            'confidence': target_candidates[0].confidence_score,
                            'reasons': target_candidates[0].reasons
                        }
                except Exception as target_error:
                    print(f"   ⚠️ Target detection error: {target_error}")
            
            return {
                'success': True,
                'dataframe': df,
                'parsing_strategy': 'geo_series_matrix',
                'target_column': target_column,
                'target_detection': target_detection_info,
                'quality_report': quality_report,
                'metadata': {
                    'file_path': str(file_path),
                    'format': f'GEO Series Matrix ({series_accession})',
                    'title': series_title,
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                    'geo_metadata': metadata
                }
            }
            
        except Exception as e:
            print(f"   ❌ Error processing GEO file: {e}")
            return {
                'success': False,
                'error': f"Error processing GEO Series Matrix file {file_path.name}: {str(e)}"
            }

# Convenience functions
def process_folder_dataset(folder_path: str, 
                         target_column: Optional[str] = None,
                         combine_strategy: str = 'concatenate',
                         user_hint: Optional[str] = None) -> Dict[str, Any]:
    """Simple function to process a folder of datasets"""
    manager = EnhancedFolderDatasetManager()
    return manager.process_folder_input(folder_path, target_column, combine_strategy, user_hint)

def get_folder_file_summary(folder_path: str) -> Dict[str, Any]:
    """Simple function to get folder file summary"""
    manager = EnhancedFolderDatasetManager()
    return manager.get_folder_summary(folder_path) 
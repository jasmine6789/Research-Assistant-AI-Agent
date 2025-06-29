#!/usr/bin/env python3
"""
Enhanced Data Validation and Error Handling System

This module provides comprehensive data validation, schema checking, and
advanced error handling for various file formats including CSV, Excel, and JSON.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
import warnings
from pydantic import BaseModel, validator, ValidationError
from jsonschema import validate as json_validate, ValidationError as JsonValidationError
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import logging
from src.utils.error_handling import (
    BaseApplicationError, ValidationError as CustomValidationError,
    ErrorContext, ErrorCategory, ErrorSeverity
)

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str):
        """Add error message"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add warning message"""
        self.warnings.append(message)
    
    def add_suggestion(self, message: str):
        """Add suggestion message"""
        self.suggestions.append(message)

@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment"""
    overall_score: float  # 0-100 score
    completeness_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class FileFormatDetector:
    """Automatically detect and validate file formats"""
    
    SUPPORTED_FORMATS = {
        '.csv': 'CSV',
        '.xlsx': 'Excel',
        '.xls': 'Excel (Legacy)',
        '.xlsm': 'Excel (Macro-enabled)',
        '.json': 'JSON',
        '.jsonl': 'JSON Lines',
        '.txt': 'Text',
        '.tsv': 'Tab-separated values'
    }
    
    @classmethod
    def detect_format(cls, file_path: Union[str, Path]) -> Tuple[str, str]:
        """
        Detect file format from extension and content
        
        Returns:
            Tuple of (format_type, format_description)
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension in cls.SUPPORTED_FORMATS:
            return extension, cls.SUPPORTED_FORMATS[extension]
        
        # Try to detect from content if extension is unknown
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
                # Check for JSON
                if first_line.startswith('{') or first_line.startswith('['):
                    return '.json', 'JSON (detected)'
                
                # Check for CSV (comma-separated)
                if ',' in first_line:
                    return '.csv', 'CSV (detected)'
                
                # Check for TSV (tab-separated)
                if '\t' in first_line:
                    return '.tsv', 'TSV (detected)'
                
        except Exception:
            pass
        
        return extension, 'Unknown format'

class DataFileValidator:
    """Advanced data file validation with format-specific handling"""
    
    def __init__(self):
        self.detector = FileFormatDetector()
        self.encoding_candidates = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    def validate_file(self, file_path: Union[str, Path], 
                     expected_format: Optional[str] = None) -> ValidationResult:
        """
        Comprehensive file validation
        
        Args:
            file_path: Path to file
            expected_format: Expected format (auto-detect if None)
            
        Returns:
            ValidationResult with detailed findings
        """
        result = ValidationResult(is_valid=True)
        file_path = Path(file_path)
        
        # Basic file checks
        if not file_path.exists():
            result.add_error(f"File does not exist: {file_path}")
            return result
        
        if file_path.stat().st_size == 0:
            result.add_error("File is empty")
            return result
        
        if file_path.stat().st_size > 500 * 1024 * 1024:  # 500MB limit
            result.add_warning("File is very large (>500MB). Processing may be slow.")
        
        # Detect format
        format_ext, format_desc = self.detector.detect_format(file_path)
        result.metadata['detected_format'] = format_desc
        result.metadata['file_extension'] = format_ext
        
        # Format-specific validation
        if format_ext in ['.csv', '.tsv']:
            self._validate_csv_file(file_path, result, delimiter=',' if format_ext == '.csv' else '\t')
        elif format_ext in ['.xlsx', '.xls', '.xlsm']:
            self._validate_excel_file(file_path, result)
        elif format_ext in ['.json', '.jsonl']:
            self._validate_json_file(file_path, result)
        else:
            result.add_warning(f"Unsupported format: {format_desc}")
        
        return result
    
    def _validate_csv_file(self, file_path: Path, result: ValidationResult, delimiter: str = ','):
        """Validate CSV file"""
        try:
            # Try to detect encoding
            encoding = self._detect_encoding(file_path)
            result.metadata['encoding'] = encoding
            
            # Read with pandas for structural validation
            df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, nrows=1000)
            
            # Basic structure checks
            if len(df.columns) == 0:
                result.add_error("No columns detected in CSV file")
                return
            
            if len(df) == 0:
                result.add_error("No data rows found in CSV file")
                return
            
            # Column name validation
            self._validate_column_names(df.columns, result)
            
            # Data type consistency
            self._check_data_consistency(df, result)
            
            result.metadata['columns'] = len(df.columns)
            result.metadata['sample_rows'] = len(df)
            result.add_suggestion("CSV file structure looks good")
            
        except UnicodeDecodeError:
            result.add_error("File encoding issues detected. Try saving with UTF-8 encoding.")
        except pd.errors.EmptyDataError:
            result.add_error("CSV file appears to be empty or corrupted")
        except pd.errors.ParserError as e:
            result.add_error(f"CSV parsing error: {str(e)}")
        except Exception as e:
            result.add_error(f"Unexpected error reading CSV: {str(e)}")
    
    def _validate_excel_file(self, file_path: Path, result: ValidationResult):
        """Validate Excel file"""
        try:
            # Check if file can be opened
            with pd.ExcelFile(file_path) as xl:
                sheet_names = xl.sheet_names
                result.metadata['sheet_names'] = sheet_names
                result.metadata['num_sheets'] = len(sheet_names)
                
                if len(sheet_names) == 0:
                    result.add_error("No sheets found in Excel file")
                    return
                
                # Validate first sheet
                df = pd.read_excel(file_path, sheet_name=sheet_names[0], nrows=1000)
                
                if len(df.columns) == 0:
                    result.add_error("No columns detected in Excel sheet")
                    return
                
                if len(df) == 0:
                    result.add_error("No data rows found in Excel sheet")
                    return
                
                # Column name validation
                self._validate_column_names(df.columns, result)
                
                # Data type consistency
                self._check_data_consistency(df, result)
                
                result.metadata['columns'] = len(df.columns)
                result.metadata['sample_rows'] = len(df)
                
                if len(sheet_names) > 1:
                    result.add_suggestion(f"Excel file has {len(sheet_names)} sheets. Will use first sheet: '{sheet_names[0]}'")
                
        except FileNotFoundError:
            result.add_error("Excel file not found or cannot be accessed")
        except Exception as e:
            result.add_error(f"Error reading Excel file: {str(e)}")
    
    def _validate_json_file(self, file_path: Path, result: ValidationResult):
        """Validate JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    # JSON Lines format
                    lines = []
                    for i, line in enumerate(f):
                        if i >= 1000:  # Limit sample size
                            break
                        lines.append(json.loads(line.strip()))
                    data = lines
                else:
                    # Regular JSON
                    data = json.load(f)
            
            result.metadata['json_type'] = type(data).__name__
            
            if isinstance(data, list):
                result.metadata['num_records'] = len(data)
                if len(data) > 0 and isinstance(data[0], dict):
                    result.metadata['sample_keys'] = list(data[0].keys())
                    result.add_suggestion("JSON array of objects detected - suitable for tabular data")
                else:
                    result.add_warning("JSON array contains non-object elements")
            elif isinstance(data, dict):
                result.metadata['top_level_keys'] = list(data.keys())
                result.add_suggestion("JSON object detected - may need restructuring for tabular analysis")
            else:
                result.add_warning("JSON contains simple data type - may not be suitable for analysis")
            
        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON format: {str(e)}")
        except UnicodeDecodeError:
            result.add_error("JSON file encoding issues")
        except Exception as e:
            result.add_error(f"Error reading JSON file: {str(e)}")
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding"""
        for encoding in self.encoding_candidates:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Try to read first 1KB
                return encoding
            except UnicodeDecodeError:
                continue
        
        return 'utf-8'  # Default fallback
    
    def _validate_column_names(self, columns: pd.Index, result: ValidationResult):
        """Validate column names"""
        issues = []
        
        # Check for duplicate column names
        duplicates = columns[columns.duplicated()].tolist()
        if duplicates:
            result.add_error(f"Duplicate column names found: {duplicates}")
        
        # Check for empty or problematic column names
        for i, col in enumerate(columns):
            if pd.isna(col) or str(col).strip() == '':
                issues.append(f"Empty column name at position {i}")
            elif str(col).startswith('Unnamed:'):
                issues.append(f"Unnamed column detected: {col}")
        
        if issues:
            result.add_warning("Column name issues: " + "; ".join(issues))
            result.add_suggestion("Consider adding proper column headers")
    
    def _check_data_consistency(self, df: pd.DataFrame, result: ValidationResult):
        """Check data consistency and quality"""
        issues = []
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            issues.append(f"Completely empty columns: {empty_cols}")
        
        # Check for high missing value percentage
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        if high_missing:
            issues.append(f"Columns with >50% missing values: {high_missing}")
        
        # Check for suspicious data patterns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed data types in string columns
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    # Simple heuristic: if column has numbers and text mixed
                    has_numbers = any(str(val).replace('.', '').replace('-', '').isdigit() 
                                    for val in sample_values)
                    has_text = any(not str(val).replace('.', '').replace('-', '').isdigit() 
                                 for val in sample_values)
                    if has_numbers and has_text:
                        issues.append(f"Mixed data types in column '{col}'")
        
        if issues:
            result.add_warning("Data consistency issues: " + "; ".join(issues))

class DataQualityAnalyzer:
    """Comprehensive data quality analysis"""
    
    def analyze_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Perform comprehensive data quality analysis
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataQualityReport with detailed quality metrics
        """
        report = DataQualityReport(
            overall_score=0.0,
            completeness_score=0.0,
            consistency_score=0.0,
            validity_score=0.0,
            uniqueness_score=0.0
        )
        
        # Completeness Analysis
        report.completeness_score = self._calculate_completeness(df, report)
        
        # Consistency Analysis
        report.consistency_score = self._calculate_consistency(df, report)
        
        # Validity Analysis
        report.validity_score = self._calculate_validity(df, report)
        
        # Uniqueness Analysis
        report.uniqueness_score = self._calculate_uniqueness(df, report)
        
        # Overall score (weighted average)
        weights = {'completeness': 0.3, 'consistency': 0.25, 'validity': 0.25, 'uniqueness': 0.2}
        report.overall_score = (
            report.completeness_score * weights['completeness'] +
            report.consistency_score * weights['consistency'] +
            report.validity_score * weights['validity'] +
            report.uniqueness_score * weights['uniqueness']
        )
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        return report
    
    def _calculate_completeness(self, df: pd.DataFrame, report: DataQualityReport) -> float:
        """Calculate data completeness score"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        # Column-level analysis
        missing_by_col = (df.isnull().sum() / len(df) * 100)
        
        for col, missing_pct in missing_by_col.items():
            if missing_pct > 20:
                report.issues.append({
                    'type': 'completeness',
                    'severity': 'high' if missing_pct > 50 else 'medium',
                    'column': col,
                    'description': f"Column '{col}' has {missing_pct:.1f}% missing values"
                })
        
        return completeness
    
    def _calculate_consistency(self, df: pd.DataFrame, report: DataQualityReport) -> float:
        """Calculate data consistency score"""
        consistency_score = 100.0
        total_checks = 0
        failed_checks = 0
        
        for col in df.columns:
            total_checks += 1
            
            if df[col].dtype == 'object':
                # Check for inconsistent text formatting
                sample = df[col].dropna().head(1000)
                if len(sample) > 0:
                    # Check for mixed case patterns
                    has_lower = any(str(val).islower() for val in sample)
                    has_upper = any(str(val).isupper() for val in sample)
                    has_title = any(str(val).istitle() for val in sample)
                    
                    mixed_case_count = sum([has_lower, has_upper, has_title])
                    if mixed_case_count > 1:
                        failed_checks += 1
                        report.issues.append({
                            'type': 'consistency',
                            'severity': 'low',
                            'column': col,
                            'description': f"Column '{col}' has inconsistent text casing"
                        })
            
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Check for outliers
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    # Skip if no data after dropping NaN
                    continue
                    
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outlier_pct = len(outliers) / len(col_data) * 100
                
                if outlier_pct > 5:  # More than 5% outliers
                    failed_checks += 1
                    report.issues.append({
                        'type': 'consistency',
                        'severity': 'medium',
                        'column': col,
                        'description': f"Column '{col}' has {outlier_pct:.1f}% potential outliers"
                    })
        
        if total_checks > 0:
            consistency_score = ((total_checks - failed_checks) / total_checks) * 100
        
        return consistency_score
    
    def _calculate_validity(self, df: pd.DataFrame, report: DataQualityReport) -> float:
        """Calculate data validity score"""
        validity_score = 100.0
        total_checks = 0
        failed_checks = 0
        
        for col in df.columns:
            total_checks += 1
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Check for obviously invalid data patterns
            if df[col].dtype == 'object':
                # Check for suspicious patterns
                sample = col_data.head(1000)
                
                # Check for potential encoding issues
                suspicious_chars = sum(1 for val in sample 
                                     if any(ord(char) > 127 for char in str(val)))
                if suspicious_chars > len(sample) * 0.1:  # More than 10% have non-ASCII
                    failed_checks += 1
                    report.issues.append({
                        'type': 'validity',
                        'severity': 'medium',
                        'column': col,
                        'description': f"Column '{col}' may have encoding issues"
                    })
            
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Check for infinite values
                if np.isinf(col_data).any():
                    failed_checks += 1
                    report.issues.append({
                        'type': 'validity',
                        'severity': 'high',
                        'column': col,
                        'description': f"Column '{col}' contains infinite values"
                    })
        
        if total_checks > 0:
            validity_score = ((total_checks - failed_checks) / total_checks) * 100
        
        return validity_score
    
    def _calculate_uniqueness(self, df: pd.DataFrame, report: DataQualityReport) -> float:
        """Calculate data uniqueness score"""
        uniqueness_score = 100.0
        total_checks = len(df.columns)
        failed_checks = 0
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            duplicate_pct = (duplicate_rows / len(df)) * 100
            if duplicate_pct > 1:  # More than 1% duplicates
                failed_checks += 1
                report.issues.append({
                    'type': 'uniqueness',
                    'severity': 'medium',
                    'column': 'all',
                    'description': f"Dataset has {duplicate_pct:.1f}% duplicate rows"
                })
        
        # Check for columns with very low uniqueness (potential data quality issues)
        for col in df.columns:
            if len(df[col].dropna()) > 0:
                unique_pct = (df[col].nunique() / len(df[col].dropna())) * 100
                if unique_pct < 1 and df[col].nunique() > 1:  # Less than 1% unique, but not constant
                    report.issues.append({
                        'type': 'uniqueness',
                        'severity': 'low',
                        'column': col,
                        'description': f"Column '{col}' has very low uniqueness ({unique_pct:.1f}%)"
                    })
        
        if total_checks > 0:
            uniqueness_score = max(0, 100 - (failed_checks / total_checks) * 20)
        
        return uniqueness_score
    
    def _generate_recommendations(self, report: DataQualityReport):
        """Generate recommendations based on analysis"""
        if report.overall_score >= 90:
            report.recommendations.append("Excellent data quality! No major issues detected.")
        elif report.overall_score >= 70:
            report.recommendations.append("Good data quality with minor issues to address.")
        elif report.overall_score >= 50:
            report.recommendations.append("Moderate data quality. Several issues need attention.")
        else:
            report.recommendations.append("Poor data quality. Significant cleanup required.")
        
        # Specific recommendations based on issues
        if report.completeness_score < 80:
            report.recommendations.append("Consider handling missing values through imputation or removal.")
        
        if report.consistency_score < 80:
            report.recommendations.append("Standardize data formats and handle outliers.")
        
        if report.validity_score < 80:
            report.recommendations.append("Check for and fix data validity issues.")
        
        if report.uniqueness_score < 80:
            report.recommendations.append("Remove duplicate entries and investigate low-uniqueness columns.")

class EnhancedDataLoader:
    """Enhanced data loader with format support and validation"""
    
    def __init__(self):
        self.validator = DataFileValidator()
        self.quality_analyzer = DataQualityAnalyzer()
    
    def load_data(self, file_path: Union[str, Path], 
                  target_column: Optional[str] = None,
                  validate: bool = True) -> Dict[str, Any]:
        """
        Load data with comprehensive validation and quality analysis
        
        Args:
            file_path: Path to data file
            target_column: Target column name (optional)
            validate: Whether to perform validation
            
        Returns:
            Dictionary with loaded data and analysis results
        """
        file_path = Path(file_path)
        
        result = {
            'success': False,
            'dataframe': None,
            'validation': None,
            'quality_report': None,
            'metadata': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate file if requested
            if validate:
                validation_result = self.validator.validate_file(file_path)
                result['validation'] = validation_result
                
                if not validation_result.is_valid:
                    result['errors'].extend(validation_result.errors)
                    return result
                
                result['warnings'].extend(validation_result.warnings)
            
            # Load data based on format
            format_ext = file_path.suffix.lower()
            
            if format_ext == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
            elif format_ext == '.tsv':
                df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8')
            elif format_ext in ['.xlsx', '.xls', '.xlsm']:
                df = pd.read_excel(file_path)
            elif format_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        df = pd.json_normalize(data)
                    else:
                        df = pd.json_normalize([data])
            elif format_ext == '.jsonl':
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                df = pd.json_normalize(data)
            else:
                result['errors'].append(f"Unsupported file format: {format_ext}")
                return result
            
            result['dataframe'] = df
            result['metadata'] = {
                'file_path': str(file_path),
                'format': format_ext,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
            
            # Perform quality analysis if data loaded successfully
            if validate and len(df) > 0:
                quality_report = self.quality_analyzer.analyze_quality(df)
                result['quality_report'] = quality_report
            
            result['success'] = True
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            result['errors'].append(error_msg)
            logger.error(error_msg, exc_info=True)
        
        return result

# Convenience functions
def validate_data_file(file_path: Union[str, Path]) -> ValidationResult:
    """Simple function to validate a data file"""
    validator = DataFileValidator()
    return validator.validate_file(file_path)

def load_and_validate_data(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Simple function to load and validate data"""
    loader = EnhancedDataLoader()
    return loader.load_data(file_path, validate=True) 
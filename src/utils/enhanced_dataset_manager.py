#!/usr/bin/env python3
"""
Enhanced Dataset Manager with Advanced System Improvements

This module provides comprehensive dataset management with:
- Enhanced target detection algorithms
- Multi-format file support (CSV, Excel, JSON)
- Advanced data validation and error handling
- Real-time data quality monitoring and recommendations
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
import logging
from datetime import datetime
import json

# Import our enhanced modules
from src.utils.enhanced_target_detection import (
    EnhancedTargetDetector, TargetDetectionConfig, detect_target_variable
)
from src.utils.enhanced_data_validation import (
    EnhancedDataLoader, DataFileValidator, DataQualityAnalyzer,
    validate_data_file, load_and_validate_data
)
from src.utils.realtime_quality_monitor import (
    RealTimeQualityMonitor, MonitoringConfig, DataQualityRecommendationEngine
)
from src.utils.error_handling import (
    ErrorHandler, ErrorContext, ErrorCategory, ValidationError
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedDatasetManager:
    """
    Advanced dataset manager with comprehensive features:
    - Multi-format support (CSV, Excel, JSON, TSV)
    - Intelligent target variable detection
    - Real-time data quality monitoring
    - Advanced validation and error handling
    - Automated recommendations
    """
    
    def __init__(self, user_datasets_folder: str = "user_datasets"):
        """
        Initialize Enhanced Dataset Manager
        
        Args:
            user_datasets_folder: Folder containing user datasets
        """
        self.user_datasets_folder = user_datasets_folder
        self.error_handler = ErrorHandler()
        
        # Initialize components
        self.target_detector = EnhancedTargetDetector()
        self.data_loader = EnhancedDataLoader()
        self.quality_analyzer = DataQualityAnalyzer()
        self.recommendation_engine = DataQualityRecommendationEngine()
        
        # Real-time monitoring (optional)
        self.quality_monitor = None
        self.monitoring_enabled = False
        
        # Supported file formats
        self.supported_formats = {
            '.csv': 'Comma-separated values',
            '.tsv': 'Tab-separated values', 
            '.xlsx': 'Excel spreadsheet',
            '.xls': 'Excel legacy format',
            '.xlsm': 'Excel macro-enabled',
            '.json': 'JSON format',
            '.jsonl': 'JSON Lines format'
        }
        
        # Create datasets folder if it doesn't exist
        self._setup_datasets_folder()
        
        logger.info(f"Enhanced Dataset Manager initialized. Supported formats: {list(self.supported_formats.keys())}")
    
    def _setup_datasets_folder(self):
        """Setup the datasets folder structure"""
        if not os.path.exists(self.user_datasets_folder):
            os.makedirs(self.user_datasets_folder)
            
            # Create examples subfolder
            examples_folder = os.path.join(self.user_datasets_folder, "examples")
            os.makedirs(examples_folder, exist_ok=True)
            
            logger.info(f"Created datasets folder: {self.user_datasets_folder}")
    
    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about all available datasets
        
        Returns:
            Dictionary with dataset information including format, size, etc.
        """
        datasets = {}
        
        if not os.path.exists(self.user_datasets_folder):
            return datasets
        
        for file in os.listdir(self.user_datasets_folder):
            file_path = os.path.join(self.user_datasets_folder, file)
            
            if os.path.isfile(file_path):
                file_ext = Path(file).suffix.lower()
                
                if file_ext in self.supported_formats:
                    # Get file information
                    file_stats = os.stat(file_path)
                    file_size_mb = file_stats.st_size / (1024 * 1024)
                    
                    datasets[file] = {
                        'path': file_path,
                        'format': self.supported_formats[file_ext],
                        'extension': file_ext,
                        'size_mb': round(file_size_mb, 2),
                        'modified': datetime.fromtimestamp(file_stats.st_mtime),
                        'supported': True
                    }
                else:
                    # Unsupported format
                    datasets[file] = {
                        'path': file_path,
                        'format': 'Unsupported',
                        'extension': file_ext,
                        'size_mb': 0,
                        'modified': None,
                        'supported': False
                    }
        
        logger.info(f"Found {len(datasets)} files, {sum(1 for d in datasets.values() if d['supported'])} supported")
        return datasets
    
    def load_dataset_advanced(self, filename: str, 
                            target_column: Optional[str] = None,
                            enable_monitoring: bool = False,
                            user_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Advanced dataset loading with comprehensive analysis
        
        Args:
            filename: Name of dataset file
            target_column: Target column name (auto-detect if None)
            enable_monitoring: Enable real-time quality monitoring
            user_hint: User hint for target detection
            
        Returns:
            Comprehensive dataset information and analysis
        """
        file_path = os.path.join(self.user_datasets_folder, filename)
        
        try:
            # Step 1: Load and validate data
            logger.info(f"🔄 Loading dataset: {filename}")
            load_result = self.data_loader.load_data(file_path, target_column, validate=True)
            
            if not load_result['success']:
                return {
                    'success': False,
                    'errors': load_result['errors'],
                    'warnings': load_result['warnings']
                }
            
            df = load_result['dataframe']
            
            # Step 2: Enhanced target detection
            logger.info("🎯 Performing enhanced target detection...")
            if target_column is None:
                target_candidates = self.target_detector.detect_target_variables(df, user_hint)
                
                if target_candidates:
                    best_target = target_candidates[0]
                    target_column = best_target.column_name
                    target_detection_info = {
                        'detected_target': target_column,
                        'confidence': best_target.confidence_score,
                        'reasons': best_target.reasons,
                        'all_candidates': [
                            {
                                'name': c.column_name,
                                'confidence': c.confidence_score,
                                'data_type': c.data_type,
                                'reasons': c.reasons
                            } for c in target_candidates[:5]  # Top 5
                        ]
                    }
                else:
                    target_column = df.columns[-1]  # Fallback
                    target_detection_info = {
                        'detected_target': target_column,
                        'confidence': 0.5,
                        'reasons': ['Fallback to last column'],
                        'all_candidates': []
                    }
            else:
                # Validate user-provided target
                if target_column not in df.columns:
                    return {
                        'success': False,
                        'errors': [f"Target column '{target_column}' not found in dataset"],
                        'warnings': []
                    }
                
                target_detection_info = {
                    'detected_target': target_column,
                    'confidence': 1.0,
                    'reasons': ['User specified'],
                    'all_candidates': []
                }
            
            # Step 3: Comprehensive data analysis
            logger.info("📊 Performing comprehensive data analysis...")
            feature_columns = [col for col in df.columns if col != target_column]
            
            # Enhanced metadata
            metadata = self._analyze_dataset_comprehensive(df, target_column, filename)
            
            # Step 4: Quality analysis and recommendations
            quality_report = load_result.get('quality_report')
            if quality_report is None:
                quality_report = self.quality_analyzer.analyze_quality(df)
            
            recommendations = self.recommendation_engine.generate_recommendations(
                df, quality_report, []
            )
            
            # Step 5: Generate research hypothesis
            hypothesis = self.generate_hypothesis_enhanced(df, target_column, metadata)
            
            # Step 6: Setup monitoring if requested
            monitoring_info = None
            if enable_monitoring:
                monitoring_info = self._setup_monitoring(file_path)
            
            # Compile comprehensive result
            result = {
                'success': True,
                'dataframe': df,
                'filename': filename,
                'file_path': file_path,
                'target_column': target_column,
                'feature_columns': feature_columns,
                
                # Enhanced analysis
                'target_detection': target_detection_info,
                'metadata': metadata,
                'quality_report': quality_report,
                'recommendations': recommendations,
                'hypothesis': hypothesis,
                
                # Validation results
                'validation': load_result.get('validation'),
                'errors': load_result.get('errors', []),
                'warnings': load_result.get('warnings', []),
                
                # Monitoring
                'monitoring_enabled': enable_monitoring,
                'monitoring_info': monitoring_info,
                
                # Summary
                'summary': self._generate_enhanced_summary(metadata, quality_report, target_detection_info)
            }
            
            logger.info(f"✅ Dataset loaded successfully: {filename}")
            logger.info(f"   📈 Shape: {df.shape}")
            logger.info(f"   🎯 Target: {target_column} (confidence: {target_detection_info['confidence']:.3f})")
            logger.info(f"   📊 Quality Score: {quality_report.overall_score:.1f}/100")
            
            return result
            
        except Exception as e:
            error_context = ErrorContext(
                operation="load_dataset_advanced",
                component="EnhancedDatasetManager",
                additional_data={"filename": filename}
            )
            self.error_handler.handle_error(e, error_context, reraise=False)
            
            return {
                'success': False,
                'errors': [f"Failed to load dataset: {str(e)}"],
                'warnings': []
            }
    
    def _analyze_dataset_comprehensive(self, df: pd.DataFrame, target_column: str, filename: str) -> Dict[str, Any]:
        """Perform comprehensive dataset analysis"""
        
        features = [col for col in df.columns if col != target_column]
        feature_df = df[features]
        target_series = df[target_column]
        
        # Basic statistics
        n_samples = len(df)
        n_features = len(features)
        
        # Enhanced data type analysis
        data_types = {
            'numerical': [],
            'categorical': [],
            'binary': [],
            'datetime': [],
            'text': []
        }
        
        for col in features:
            col_data = df[col].dropna()
            unique_count = len(col_data.unique())
            
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                data_types['datetime'].append(col)
            elif df[col].dtype in ['int64', 'float64']:
                if unique_count == 2:
                    data_types['binary'].append(col)
                elif unique_count <= 10:
                    data_types['categorical'].append(col)
                else:
                    data_types['numerical'].append(col)
            elif df[col].dtype == 'object':
                if unique_count == 2:
                    data_types['binary'].append(col)
                elif unique_count <= 20:
                    data_types['categorical'].append(col)
                else:
                    data_types['text'].append(col)
        
        # Target analysis with enhanced detection
        target_analysis = self._analyze_target_comprehensive(target_series)
        
        # Missing values analysis
        missing_analysis = {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'missing_by_column': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        
        # Correlation analysis (for numerical features)
        correlation_info = {}
        numerical_cols = data_types['numerical'] + data_types['binary']
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols + [target_column]].corr()
            target_correlations = corr_matrix[target_column].drop(target_column).abs().sort_values(ascending=False)
            correlation_info = {
                'target_correlations': target_correlations.head(10).to_dict(),
                'highly_correlated_features': target_correlations[target_correlations > 0.5].index.tolist()
            }
        
        # Data quality indicators
        quality_indicators = {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'constant_columns': [col for col in df.columns if df[col].nunique() <= 1],
            'high_cardinality_columns': [col for col in df.columns 
                                       if df[col].dtype == 'object' and df[col].nunique() > 0.8 * len(df)]
        }
        
        # Memory usage analysis
        memory_usage = {
            'total_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'per_column_mb': (df.memory_usage(deep=True) / 1024 / 1024).to_dict()
        }
        
        return {
            'filename': filename,
            'basic_stats': {
                'n_samples': n_samples,
                'n_features': n_features,
                'n_total_columns': len(df.columns)
            },
            'data_types': data_types,
            'target_analysis': target_analysis,
            'missing_analysis': missing_analysis,
            'correlation_info': correlation_info,
            'quality_indicators': quality_indicators,
            'memory_usage': memory_usage,
            'sample_data': {
                'head': df.head(3).to_dict(),
                'column_samples': {col: df[col].dropna().unique()[:5].tolist() 
                                 for col in df.columns if df[col].dtype == 'object'}
            }
        }
    
    def _analyze_target_comprehensive(self, target_series: pd.Series) -> Dict[str, Any]:
        """Comprehensive target variable analysis"""
        
        target_data = target_series.dropna()
        unique_values = len(target_data.unique())
        
        # Determine task type with confidence
        if pd.api.types.is_numeric_dtype(target_data):
            if unique_values <= 2:
                task_type = "binary_classification"
                task_confidence = 0.95
            elif unique_values <= 20:
                task_type = "multiclass_classification"
                task_confidence = 0.8
            else:
                task_type = "regression"
                task_confidence = 0.9
        else:
            if unique_values <= 2:
                task_type = "binary_classification"
                task_confidence = 0.9
            else:
                task_type = "multiclass_classification"
                task_confidence = 0.85
        
        # Distribution analysis
        if task_type in ["binary_classification", "multiclass_classification"]:
            value_counts = target_data.value_counts()
            distribution_info = {
                'class_distribution': value_counts.to_dict(),
                'class_balance': value_counts.min() / value_counts.max(),
                'most_common_class': value_counts.index[0],
                'least_common_class': value_counts.index[-1]
            }
            
            # Check for class imbalance
            imbalance_ratio = value_counts.max() / value_counts.min()
            is_imbalanced = imbalance_ratio > 3
            
            distribution_info.update({
                'is_imbalanced': is_imbalanced,
                'imbalance_ratio': imbalance_ratio
            })
            
        else:  # Regression
            distribution_info = {
                'mean': float(target_data.mean()),
                'std': float(target_data.std()),
                'min': float(target_data.min()),
                'max': float(target_data.max()),
                'median': float(target_data.median()),
                'skewness': float(target_data.skew()),
                'kurtosis': float(target_data.kurtosis())
            }
        
        return {
            'task_type': task_type,
            'task_confidence': task_confidence,
            'unique_values': unique_values,
            'missing_values': target_series.isnull().sum(),
            'missing_percentage': (target_series.isnull().sum() / len(target_series)) * 100,
            'distribution_info': distribution_info
        }
    
    def generate_hypothesis_enhanced(self, df: pd.DataFrame, target_column: str, 
                                   metadata: Dict[str, Any]) -> str:
        """Generate enhanced research hypothesis based on comprehensive analysis"""
        
        target_analysis = metadata['target_analysis']
        data_types = metadata['data_types']
        correlation_info = metadata.get('correlation_info', {})
        
        # Base hypothesis components
        task_type = target_analysis['task_type']
        n_features = metadata['basic_stats']['n_features']
        n_samples = metadata['basic_stats']['n_samples']
        
        # Task-specific hypothesis generation
        if task_type == "binary_classification":
            base_hypothesis = f"Machine learning models can effectively predict the binary outcome '{target_column}'"
            
            if target_analysis['distribution_info'].get('is_imbalanced', False):
                base_hypothesis += " despite class imbalance challenges"
                
        elif task_type == "multiclass_classification":
            n_classes = target_analysis['unique_values']
            base_hypothesis = f"Machine learning models can accurately classify instances into {n_classes} distinct categories for '{target_column}'"
            
        else:  # Regression
            base_hypothesis = f"Machine learning models can predict the continuous target variable '{target_column}'"
            
            # Add specific insights for regression
            target_dist = target_analysis['distribution_info']
            if abs(target_dist.get('skewness', 0)) > 1:
                base_hypothesis += " accounting for the skewed distribution of the target variable"
        
        # Add feature-based insights
        feature_insights = []
        
        if len(data_types['numerical']) > 0:
            feature_insights.append(f"{len(data_types['numerical'])} numerical features")
        
        if len(data_types['categorical']) > 0:
            feature_insights.append(f"{len(data_types['categorical'])} categorical features")
        
        if len(data_types['binary']) > 0:
            feature_insights.append(f"{len(data_types['binary'])} binary features")
        
        # Add correlation insights
        if correlation_info.get('highly_correlated_features'):
            n_corr = len(correlation_info['highly_correlated_features'])
            feature_insights.append(f"{n_corr} features showing strong correlation with the target")
        
        # Construct enhanced hypothesis
        hypothesis = base_hypothesis
        
        if feature_insights:
            hypothesis += f" using a combination of {', '.join(feature_insights)}"
        
        # Add methodology suggestions
        if task_type in ["binary_classification", "multiclass_classification"]:
            if n_features > 50:
                hypothesis += ". Given the high dimensionality, feature selection techniques and ensemble methods are recommended"
            elif target_analysis['distribution_info'].get('is_imbalanced', False):
                hypothesis += ". Due to class imbalance, techniques like SMOTE, class weighting, or ensemble methods should be considered"
            else:
                hypothesis += ". Standard classification algorithms including Random Forest, SVM, and Gradient Boosting are expected to perform well"
        else:
            if target_analysis['distribution_info'].get('skewness', 0) > 1:
                hypothesis += ". Log transformation or robust regression techniques may improve model performance"
            else:
                hypothesis += ". Linear regression, Random Forest, and Gradient Boosting algorithms are suitable for this prediction task"
        
        # Add expected outcomes
        quality_score = metadata.get('quality_score', 75)  # Default if not available
        
        if quality_score > 85:
            hypothesis += ". High data quality suggests strong predictive performance is achievable"
        elif quality_score > 70:
            hypothesis += ". Moderate data quality indicates good predictive performance with proper preprocessing"
        else:
            hypothesis += ". Data quality issues should be addressed to achieve optimal model performance"
        
        return hypothesis
    
    def _setup_monitoring(self, file_path: str) -> Dict[str, Any]:
        """Setup real-time monitoring for the dataset"""
        try:
            if self.quality_monitor is None:
                config = MonitoringConfig(
                    check_interval=60,  # Check every minute
                    enable_file_watching=True,
                    enable_memory_monitoring=True
                )
                self.quality_monitor = RealTimeQualityMonitor(config)
                self.quality_monitor.start_monitoring([self.user_datasets_folder])
            
            self.quality_monitor.add_dataset(file_path)
            self.monitoring_enabled = True
            
            return {
                'status': 'enabled',
                'check_interval': 60,
                'monitoring_id': f"monitor_{Path(file_path).stem}"
            }
            
        except Exception as e:
            logger.warning(f"Failed to setup monitoring: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _generate_enhanced_summary(self, metadata: Dict[str, Any], 
                                 quality_report, target_detection_info: Dict[str, Any]) -> str:
        """Generate comprehensive dataset summary"""
        
        basic_stats = metadata['basic_stats']
        target_analysis = metadata['target_analysis']
        data_types = metadata['data_types']
        quality_indicators = metadata['quality_indicators']
        
        summary = f"""
📊 Dataset Analysis Summary
==========================

📈 Basic Information:
   • Samples: {basic_stats['n_samples']:,}
   • Features: {basic_stats['n_features']}
   • Target: {target_detection_info['detected_target']} (confidence: {target_detection_info['confidence']:.1%})
   • Task Type: {target_analysis['task_type'].replace('_', ' ').title()}

🎯 Target Variable Analysis:
   • Unique Values: {target_analysis['unique_values']}
   • Missing Values: {target_analysis['missing_values']} ({target_analysis['missing_percentage']:.1f}%)
   • Task Confidence: {target_analysis['task_confidence']:.1%}

📊 Feature Composition:
   • Numerical: {len(data_types['numerical'])}
   • Categorical: {len(data_types['categorical'])}
   • Binary: {len(data_types['binary'])}
   • DateTime: {len(data_types['datetime'])}
   • Text: {len(data_types['text'])}

📋 Data Quality Score: {quality_report.overall_score:.1f}/100
   • Completeness: {quality_report.completeness_score:.1f}/100
   • Consistency: {quality_report.consistency_score:.1f}/100
   • Validity: {quality_report.validity_score:.1f}/100
   • Uniqueness: {quality_report.uniqueness_score:.1f}/100

⚠️ Quality Issues:
   • Duplicate Rows: {quality_indicators['duplicate_rows']} ({quality_indicators['duplicate_percentage']:.1f}%)
   • Constant Columns: {len(quality_indicators['constant_columns'])}
   • High Cardinality Columns: {len(quality_indicators['high_cardinality_columns'])}

💾 Memory Usage: {metadata['memory_usage']['total_mb']:.1f} MB
        """
        
        return summary.strip()
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        if self.quality_monitor and self.monitoring_enabled:
            return self.quality_monitor.get_monitoring_status()
        else:
            return {
                'is_running': False,
                'message': 'Monitoring not enabled'
            }
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if self.quality_monitor:
            self.quality_monitor.stop_monitoring()
            self.monitoring_enabled = False
            logger.info("Real-time monitoring stopped")
    
    def validate_dataset_file(self, filename: str) -> Dict[str, Any]:
        """
        Validate a dataset file without loading it completely
        
        Args:
            filename: Name of dataset file
            
        Returns:
            Validation results
        """
        file_path = os.path.join(self.user_datasets_folder, filename)
        
        try:
            validation_result = validate_data_file(file_path)
            
            return {
                'success': validation_result.is_valid,
                'filename': filename,
                'format': validation_result.metadata.get('detected_format', 'Unknown'),
                'errors': validation_result.errors,
                'warnings': validation_result.warnings,
                'suggestions': validation_result.suggestions,
                'metadata': validation_result.metadata
            }
            
        except Exception as e:
            return {
                'success': False,
                'filename': filename,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'suggestions': []
            }
    
    def create_sample_datasets(self):
        """Create sample datasets for demonstration"""
        try:
            # Create different types of sample datasets
            self._create_classification_sample()
            self._create_regression_sample()
            self._create_mixed_sample()
            
            logger.info("Sample datasets created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create sample datasets: {e}")
    
    def _create_classification_sample(self):
        """Create a sample classification dataset"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        age = np.random.normal(35, 10, n_samples)
        income = np.random.normal(50000, 15000, n_samples)
        experience = np.random.normal(8, 4, n_samples)
        education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
        
        # Generate target (loan approval)
        approval_prob = (age * 0.01 + income * 0.00001 + experience * 0.05 + 
                        (education == 'PhD') * 0.3 + (education == 'Master') * 0.2)
        approval_prob = 1 / (1 + np.exp(-approval_prob + 3))  # Sigmoid
        loan_approved = (np.random.random(n_samples) < approval_prob).astype(int)
        
        df = pd.DataFrame({
            'age': age.astype(int),
            'annual_income': income.astype(int),
            'years_experience': experience.astype(int),
            'education_level': education,
            'loan_approved': loan_approved
        })
        
        sample_path = os.path.join(self.user_datasets_folder, "sample_loan_approval.csv")
        df.to_csv(sample_path, index=False)
    
    def _create_regression_sample(self):
        """Create a sample regression dataset"""
        np.random.seed(42)
        n_samples = 800
        
        # Generate features
        size = np.random.normal(2000, 500, n_samples)
        bedrooms = np.random.randint(1, 6, n_samples)
        age = np.random.uniform(0, 50, n_samples)
        location_score = np.random.uniform(1, 10, n_samples)
        
        # Generate target (house price)
        price = (size * 150 + bedrooms * 10000 - age * 1000 + 
                location_score * 15000 + np.random.normal(0, 20000, n_samples))
        price = np.maximum(price, 50000)  # Minimum price
        
        df = pd.DataFrame({
            'square_feet': size.astype(int),
            'bedrooms': bedrooms,
            'house_age': age.round(1),
            'location_score': location_score.round(1),
            'neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Rural'], n_samples),
            'house_price': price.astype(int)
        })
        
        sample_path = os.path.join(self.user_datasets_folder, "sample_house_prices.csv")
        df.to_csv(sample_path, index=False)
    
    def _create_mixed_sample(self):
        """Create a sample dataset with mixed data types"""
        np.random.seed(42)
        n_samples = 500
        
        # Create a customer churn dataset
        df = pd.DataFrame({
            'customer_id': [f"CUST_{i:04d}" for i in range(n_samples)],
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'monthly_charges': np.random.normal(65, 20, n_samples).round(2),
            'total_charges': np.random.normal(2000, 1500, n_samples).round(2),
            'contract_length': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No'], n_samples),
            'online_security': np.random.choice(['Yes', 'No'], n_samples),
            'satisfaction_score': np.random.randint(1, 6, n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
        })
        
        sample_path = os.path.join(self.user_datasets_folder, "sample_customer_churn.csv")
        df.to_csv(sample_path, index=False)

# Convenience functions for backward compatibility and easy usage
def load_enhanced_dataset(filename: str, 
                         user_datasets_folder: str = "user_datasets",
                         target_column: Optional[str] = None,
                         enable_monitoring: bool = False) -> Dict[str, Any]:
    """
    Convenience function to load dataset with enhanced features
    
    Args:
        filename: Dataset filename
        user_datasets_folder: Folder containing datasets
        target_column: Target column name (auto-detect if None)
        enable_monitoring: Enable real-time monitoring
        
    Returns:
        Enhanced dataset information
    """
    manager = EnhancedDatasetManager(user_datasets_folder)
    return manager.load_dataset_advanced(filename, target_column, enable_monitoring)

def get_dataset_recommendations(df: pd.DataFrame) -> List[str]:
    """Get data quality recommendations for a dataframe"""
    engine = DataQualityRecommendationEngine()
    analyzer = DataQualityAnalyzer()
    
    quality_report = analyzer.analyze_quality(df)
    return engine.generate_recommendations(df, quality_report, []) 
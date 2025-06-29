#!/usr/bin/env python3
"""
Enhanced Target Detection System

This module provides advanced algorithms for intelligent target variable detection
in datasets, using multiple strategies including semantic analysis, statistical
measures, and machine learning-based approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class TargetCandidate:
    """Represents a potential target variable candidate"""
    column_name: str
    confidence_score: float
    reasons: List[str]
    data_type: str
    unique_values: int
    missing_percentage: float
    distribution_info: Dict[str, Any]

@dataclass
class TargetDetectionConfig:
    """Configuration for target detection algorithms"""
    semantic_weight: float = 0.3
    statistical_weight: float = 0.4
    ml_weight: float = 0.3
    min_confidence_threshold: float = 0.5
    max_unique_values_classification: int = 50
    min_samples_for_ml: int = 100

class EnhancedTargetDetector:
    """Advanced target variable detection system"""
    
    def __init__(self, config: TargetDetectionConfig = None):
        """
        Initialize enhanced target detector
        
        Args:
            config: Detection configuration parameters
        """
        self.config = config or TargetDetectionConfig()
        
        # Semantic patterns for target detection (prioritized by relevance)
        self.semantic_patterns = {
            'explicit_targets': {
                'patterns': [
                    r'\btarget\b', r'\blabel\b', r'\bclass\b', r'\by\b',
                    r'\boutcome\b', r'\bresult\b', r'\boutput\b'
                ],
                'weight': 1.0
            },
            'medical_targets': {
                'patterns': [
                    r'\bdiagnosis\b', r'\bdisease\b', r'\bstatus\b',
                    r'\bsymptom\b', r'\bcondition\b', r'\bpatient\b'
                ],
                'weight': 0.9
            },
            'business_targets': {
                'patterns': [
                    r'\bprice\b', r'\bvalue\b', r'\bsales\b', r'\brevenue\b',
                    r'\bprofit\b', r'\bcost\b', r'\brating\b', r'\bscore\b'
                ],
                'weight': 0.8
            },
            'decision_targets': {
                'patterns': [
                    r'\bapproved\b', r'\baccepted\b', r'\brejected\b',
                    r'\bdecision\b', r'\bchurn\b', r'\bfailure\b',
                    r'\bsuccess\b', r'\bdefault\b'
                ],
                'weight': 0.85
            },
            'quality_targets': {
                'patterns': [
                    r'\bgrade\b', r'\bquality\b', r'\bperformance\b',
                    r'\befficiency\b', r'\bsatisfaction\b'
                ],
                'weight': 0.7
            }
        }
        
        # Common feature indicators (less likely to be targets)
        self.feature_indicators = [
            r'\bid\b', r'\bindex\b', r'\bkey\b', r'\bname\b',
            r'\btitle\b', r'\bdescription\b', r'\bcomment\b',
            r'\bdate\b', r'\btime\b', r'\btimestamp\b',
            r'\burl\b', r'\bemail\b', r'\baddress\b'
        ]
    
    def detect_target_variables(self, df: pd.DataFrame, 
                              user_hint: Optional[str] = None) -> List[TargetCandidate]:
        """
        Detect potential target variables using multiple strategies
        
        Args:
            df: Input dataframe
            user_hint: Optional user-provided hint about target variable
            
        Returns:
            List of target candidates sorted by confidence score
        """
        logger.info(f"Starting enhanced target detection for dataset with {len(df.columns)} columns")
        
        # OPTIMIZATION: Smart column filtering for large datasets
        columns_to_analyze = self._filter_columns_for_analysis(df)
        
        if len(columns_to_analyze) < len(df.columns):
            logger.info(f"Optimized analysis: analyzing {len(columns_to_analyze)} potential target columns (skipped {len(df.columns) - len(columns_to_analyze)} feature columns)")
        
        candidates = []
        
        # Process filtered columns as potential targets
        for column in columns_to_analyze:
            candidate = self._evaluate_target_candidate(df, column, user_hint)
            if candidate.confidence_score > self.config.min_confidence_threshold:
                candidates.append(candidate)
        
        # Sort by confidence score (descending)
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        logger.info(f"Found {len(candidates)} potential target variables")
        for i, candidate in enumerate(candidates[:3]):  # Log top 3
            logger.info(f"  {i+1}. {candidate.column_name} (confidence: {candidate.confidence_score:.3f})")
        
        return candidates
    
    def _filter_columns_for_analysis(self, df: pd.DataFrame) -> List[str]:
        """
        Smart column filtering for large datasets to speed up target detection
        
        For datasets with many columns (e.g., genomics data), this filters out
        obvious feature columns and focuses on potential metadata/target columns.
        """
        all_columns = df.columns.tolist()
        
        # For small datasets, analyze all columns
        if len(all_columns) <= 100:
            return all_columns
        
        logger.info(f"Large dataset detected ({len(all_columns)} columns). Applying smart filtering...")
        
        priority_columns = []
        
        # 1. PRIORITIZE: Columns with target-like names
        target_patterns = [
            r'\btarget\b', r'\blabel\b', r'\bclass\b', r'\by\b', r'\boutcome\b',
            r'\bresult\b', r'\bstatus\b', r'\bgroup\b', r'\btype\b', r'\bcategory\b',
            r'\bdiagnosis\b', r'\bdisease\b', r'\bcondition\b', r'\btreatment\b',
            r'\btime\b', r'\bage\b', r'\bgender\b', r'\bsex\b', r'\bresponse\b'
        ]
        
        for col in all_columns:
            col_lower = col.lower()
            for pattern in target_patterns:
                if re.search(pattern, col_lower):
                    priority_columns.append(col)
                    break
        
        # 2. PRIORITIZE: Non-numeric identifier-like columns
        metadata_columns = []
        for col in all_columns:
            if col not in priority_columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Check if it looks like metadata (low cardinality, descriptive)
                    unique_ratio = len(col_data.unique()) / len(col_data)
                    
                    # Likely metadata characteristics
                    if (unique_ratio <= 0.2 or  # Low cardinality
                        len(col_data.unique()) <= 50 or  # Few unique values
                        col_data.dtype == 'object'):  # Text-based
                        metadata_columns.append(col)
        
        # 3. SKIP: Obvious feature columns (genomics, high-cardinality numeric)
        skip_patterns = [
            r'^[A-Z0-9]+_at$',  # Affymetrix probe IDs
            r'^ENSG\d+',        # Ensembl gene IDs
            r'^\d+_s_at$',      # Another probe pattern
            r'^[A-Z]{2,10}\d+', # Gene symbols with numbers
            r'^chr\d+',         # Chromosome locations
            r'^\d+\.\d+$',      # Decimal numbers as column names
        ]
        
        genomics_columns_skipped = 0
        for col in all_columns:
            if col not in priority_columns and col not in metadata_columns:
                # Skip obvious genomics/feature columns
                skip_column = False
                for pattern in skip_patterns:
                    if re.search(pattern, col):
                        skip_column = True
                        genomics_columns_skipped += 1
                        break
                
                # Also skip if it's a highly unique numeric column (likely feature data)
                if not skip_column:
                    col_data = df[col].dropna()
                    if (len(col_data) > 0 and 
                        pd.api.types.is_numeric_dtype(col_data) and
                        len(col_data.unique()) / len(col_data) > 0.8):
                        genomics_columns_skipped += 1
                        skip_column = True
        
        # Combine prioritized columns
        columns_to_analyze = list(set(priority_columns + metadata_columns))
        
        # If we still have too many, take a representative sample
        if len(columns_to_analyze) > 200:
            logger.info(f"Still {len(columns_to_analyze)} columns after filtering. Taking top 200 by variance...")
            # Sort by variance to get most informative columns
            variances = []
            for col in columns_to_analyze:
                try:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        var = df[col].var()
                        variances.append((col, var if not pd.isna(var) else 0))
                    else:
                        # For categorical, use entropy-like measure
                        unique_ratio = len(df[col].unique()) / len(df[col])
                        variances.append((col, unique_ratio))
                except:
                    variances.append((col, 0))
            
            variances.sort(key=lambda x: x[1], reverse=True)
            columns_to_analyze = [col for col, _ in variances[:200]]
        
        logger.info(f"Smart filtering results:")
        logger.info(f"  - Priority columns (target-like names): {len(priority_columns)}")
        logger.info(f"  - Metadata columns (low cardinality): {len(metadata_columns)}")
        logger.info(f"  - Skipped feature/genomics columns: {genomics_columns_skipped}")
        logger.info(f"  - Final columns to analyze: {len(columns_to_analyze)}")
        
        return columns_to_analyze
    
    def _evaluate_target_candidate(self, df: pd.DataFrame, column: str, 
                                 user_hint: Optional[str] = None) -> TargetCandidate:
        """
        Evaluate a single column as target candidate
        
        Args:
            df: Input dataframe
            column: Column name to evaluate
            user_hint: User-provided hint
            
        Returns:
            TargetCandidate object with evaluation results
        """
        reasons = []
        scores = {}
        
        # Basic column analysis
        col_data = df[column].dropna()
        unique_values = len(col_data.unique())
        missing_percentage = (df[column].isnull().sum() / len(df)) * 100
        data_type = self._determine_data_type(col_data)
        
        # 1. Semantic Analysis
        semantic_score = self._calculate_semantic_score(column, reasons)
        scores['semantic'] = semantic_score
        
        # 2. Statistical Analysis
        statistical_score = self._calculate_statistical_score(df, column, reasons)
        scores['statistical'] = statistical_score
        
        # 3. Machine Learning-based Analysis (if enough data)
        ml_score = 0.0
        if len(df) >= self.config.min_samples_for_ml:
            ml_score = self._calculate_ml_score(df, column, reasons)
        scores['ml'] = ml_score
        
        # 4. User hint bonus
        if user_hint and user_hint.lower() in column.lower():
            scores['user_hint'] = 1.0
            reasons.append("Matches user hint")
        else:
            scores['user_hint'] = 0.0
        
        # Calculate weighted final score
        weights = {
            'semantic': self.config.semantic_weight,
            'statistical': self.config.statistical_weight,
            'ml': self.config.ml_weight,
            'user_hint': 0.2  # Bonus weight for user hints
        }
        
        confidence_score = sum(scores[key] * weights[key] for key in scores)
        confidence_score = min(confidence_score, 1.0)  # Cap at 1.0
        
        # Distribution analysis
        distribution_info = self._analyze_distribution(col_data, data_type)
        
        return TargetCandidate(
            column_name=column,
            confidence_score=confidence_score,
            reasons=reasons,
            data_type=data_type,
            unique_values=unique_values,
            missing_percentage=missing_percentage,
            distribution_info=distribution_info
        )
    
    def _calculate_semantic_score(self, column: str, reasons: List[str]) -> float:
        """Calculate semantic score based on column name patterns"""
        column_lower = column.lower().strip()
        
        # Check for feature indicators (negative score)
        for pattern in self.feature_indicators:
            if re.search(pattern, column_lower):
                pattern_clean = pattern.strip('\\b')
                reasons.append(f"Column name suggests feature variable ('{pattern_clean}')")
                return 0.1  # Very low score for likely features
        
        # Check semantic patterns
        max_score = 0.0
        best_match = None
        
        for category, info in self.semantic_patterns.items():
            for pattern in info['patterns']:
                if re.search(pattern, column_lower):
                    score = info['weight']
                    if score > max_score:
                        max_score = score
                        pattern_clean = pattern.strip('\\b')
                        best_match = (category, pattern_clean)
        
        if best_match:
            reasons.append(f"Semantic match: {best_match[0]} ('{best_match[1]}')")
        
        # Position-based scoring (last columns more likely to be targets)
        return max_score
    
    def _calculate_statistical_score(self, df: pd.DataFrame, column: str, 
                                   reasons: List[str]) -> float:
        """Calculate statistical score based on data characteristics"""
        col_data = df[column].dropna()
        
        if len(col_data) == 0:
            return 0.0
        
        score = 0.0
        unique_values = len(col_data.unique())
        total_samples = len(col_data)
        
        # 1. Unique value ratio analysis
        unique_ratio = unique_values / total_samples
        
        if unique_ratio < 0.1:  # Low cardinality (good for classification)
            score += 0.4
            reasons.append(f"Low cardinality ({unique_values} unique values)")
        elif unique_ratio > 0.8:  # High cardinality (might be identifier)
            score -= 0.3
            reasons.append("High cardinality (possible identifier)")
        
        # 2. Binary variable detection
        if unique_values == 2:
            score += 0.5
            reasons.append("Binary variable (common target type)")
        
        # 3. Categorical with reasonable number of classes
        if 3 <= unique_values <= self.config.max_unique_values_classification:
            score += 0.3
            reasons.append(f"Categorical with {unique_values} classes")
        
        # 4. Data type considerations
        if pd.api.types.is_numeric_dtype(col_data):
            if unique_values <= 20:  # Discrete numeric (good target)
                score += 0.2
                reasons.append("Discrete numeric variable")
            else:  # Continuous numeric
                score += 0.1
                reasons.append("Continuous numeric variable")
        
        # 5. Missing values penalty
        missing_ratio = (df[column].isnull().sum() / len(df))
        if missing_ratio > 0.1:
            penalty = min(0.3, missing_ratio)
            score -= penalty
            reasons.append(f"Missing values penalty ({missing_ratio:.1%})")
        
        return max(0.0, min(1.0, score))
    
    def _calculate_ml_score(self, df: pd.DataFrame, column: str, 
                          reasons: List[str]) -> float:
        """Calculate ML-based score using feature importance"""
        try:
            # Prepare features (all columns except the candidate target)
            feature_cols = [col for col in df.columns if col != column]
            
            if len(feature_cols) < 2:  # Need at least 2 features
                return 0.0
            
            # OPTIMIZATION: For large datasets, skip ML analysis or use sampling
            if len(feature_cols) > 1000:
                reasons.append("Skipped ML analysis (too many features)")
                return 0.0
            
            # OPTIMIZATION: Sample features for very wide datasets
            if len(feature_cols) > 100:
                # Take a representative sample of features
                sample_size = min(50, len(feature_cols))
                feature_cols = np.random.choice(feature_cols, size=sample_size, replace=False).tolist()
                reasons.append(f"ML analysis with {sample_size} sampled features")
            
            # OPTIMIZATION: Sample rows for very long datasets
            max_rows = 1000
            if len(df) > max_rows:
                df_sample = df.sample(n=max_rows, random_state=42)
                reasons.append(f"ML analysis with {max_rows} sampled rows")
            else:
                df_sample = df
            
            # Prepare data
            X = df_sample[feature_cols].copy()
            y = df_sample[column].copy()
            
            # Remove rows with missing target values
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]
            
            if len(y) < 20:  # Need minimum samples
                return 0.0
            
            # Handle categorical features (optimized)
            X_processed = self._preprocess_features_fast(X)
            
            # Determine if classification or regression task
            unique_values = len(y.unique())
            is_classification = (unique_values <= self.config.max_unique_values_classification 
                               or not pd.api.types.is_numeric_dtype(y))
            
            score = 0.0
            
            if is_classification:
                score = self._evaluate_classification_target_fast(X_processed, y, reasons)
            else:
                score = self._evaluate_regression_target_fast(X_processed, y, reasons)
            
            return score
            
        except Exception as e:
            logger.debug(f"ML scoring failed for column {column}: {e}")
            return 0.0
    
    def _preprocess_features_fast(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fast preprocessing for ML analysis with large datasets"""
        X_processed = X.copy()
        
        # Limit to numeric columns for speed, or convert categorical efficiently
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                # For large datasets, just convert to numeric if possible
                try:
                    X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                except:
                    # Simple hash-based encoding for categorical
                    X_processed[col] = X_processed[col].astype(str).apply(hash).abs() % 10000
            
            # Fill missing values quickly
            if X_processed[col].isnull().any():
                X_processed[col].fillna(X_processed[col].median() if pd.api.types.is_numeric_dtype(X_processed[col]) else 0, inplace=True)
        
        return X_processed
    
    def _evaluate_classification_target_fast(self, X: pd.DataFrame, y: pd.Series, 
                                           reasons: List[str]) -> float:
        """Fast classification evaluation for large datasets"""
        try:
            # Encode categorical target
            le = LabelEncoder()
            y_encoded = le.fit_transform(y.astype(str))
            
            # Use smaller, faster random forest
            rf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3, max_features='sqrt')
            rf.fit(X, y_encoded)
            
            # Quick mutual information estimate
            if len(X.columns) <= 20:
                mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
                avg_mi = np.mean(mi_scores)
            else:
                # Estimate from RF feature importance
                avg_mi = np.mean(rf.feature_importances_)
            
            # Class balance score
            class_counts = pd.Series(y_encoded).value_counts()
            balance_score = 1.0 - (class_counts.std() / class_counts.mean()) if class_counts.mean() > 0 else 0
            
            score = min(0.8, avg_mi * 2) + min(0.2, balance_score)
            
            reasons.append(f"Fast classification analysis (score: {score:.3f})")
            
            return score
            
        except Exception as e:
            logger.debug(f"Fast classification evaluation failed: {e}")
            return 0.0
    
    def _evaluate_regression_target_fast(self, X: pd.DataFrame, y: pd.Series, 
                                       reasons: List[str]) -> float:
        """Fast regression evaluation for large datasets"""
        try:
            # Use smaller, faster random forest
            rf = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=3, max_features='sqrt')
            rf.fit(X, y)
            
            # Quick mutual information estimate
            if len(X.columns) <= 20:
                mi_scores = mutual_info_regression(X, y, random_state=42)
                avg_mi = np.mean(mi_scores)
            else:
                # Estimate from RF feature importance
                avg_mi = np.mean(rf.feature_importances_)
            
            # Coefficient of variation
            y_std = y.std()
            y_mean = y.mean()
            cv = y_std / abs(y_mean) if y_mean != 0 else 0
            
            score = min(0.6, avg_mi * 2) + min(0.4, cv * 0.5)
            
            reasons.append(f"Fast regression analysis (score: {score:.3f})")
            
            return score
            
        except Exception as e:
            logger.debug(f"Fast regression evaluation failed: {e}")
            return 0.0
    
    def _determine_data_type(self, col_data: pd.Series) -> str:
        """Determine the data type category of a column"""
        if pd.api.types.is_numeric_dtype(col_data):
            unique_values = len(col_data.unique())
            if unique_values <= 20:
                return "discrete_numeric"
            else:
                return "continuous_numeric"
        elif len(col_data.unique()) == 2:
            return "binary"
        elif len(col_data.unique()) <= 20:
            return "categorical"
        else:
            return "text_or_id"
    
    def _analyze_distribution(self, col_data: pd.Series, data_type: str) -> Dict[str, Any]:
        """Analyze the distribution of column data"""
        info = {
            "type": data_type,
            "unique_count": len(col_data.unique()),
            "sample_size": len(col_data)
        }
        
        if data_type in ["discrete_numeric", "continuous_numeric"]:
            info.update({
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "median": float(col_data.median())
            })
        else:
            value_counts = col_data.value_counts().head(10)
            info["top_values"] = value_counts.to_dict()
            info["most_common"] = str(value_counts.index[0]) if len(value_counts) > 0 else None
        
        return info
    
    def get_best_target(self, df: pd.DataFrame, 
                       user_hint: Optional[str] = None) -> Optional[TargetCandidate]:
        """
        Get the best target variable candidate
        
        Args:
            df: Input dataframe
            user_hint: Optional user hint
            
        Returns:
            Best target candidate or None if no good candidates found
        """
        candidates = self.detect_target_variables(df, user_hint)
        
        if candidates and candidates[0].confidence_score > self.config.min_confidence_threshold:
            return candidates[0]
        
        return None
    
    def explain_detection(self, candidate: TargetCandidate) -> str:
        """Generate human-readable explanation for target detection"""
        explanation = f"Target Variable: {candidate.column_name}\n"
        explanation += f"Confidence Score: {candidate.confidence_score:.3f}\n"
        explanation += f"Data Type: {candidate.data_type}\n"
        explanation += f"Unique Values: {candidate.unique_values}\n"
        explanation += f"Missing Data: {candidate.missing_percentage:.1f}%\n\n"
        explanation += "Detection Reasons:\n"
        
        for i, reason in enumerate(candidate.reasons, 1):
            explanation += f"  {i}. {reason}\n"
        
        return explanation

# Convenience function for easy usage
def detect_target_variable(df: pd.DataFrame, 
                         user_hint: Optional[str] = None,
                         config: Optional[TargetDetectionConfig] = None) -> Optional[str]:
    """
    Simple function to detect the best target variable
    
    Args:
        df: Input dataframe
        user_hint: Optional user hint about target
        config: Optional detection configuration
        
    Returns:
        Name of best target variable or None
    """
    detector = EnhancedTargetDetector(config)
    best_candidate = detector.get_best_target(df, user_hint)
    return best_candidate.column_name if best_candidate else None 
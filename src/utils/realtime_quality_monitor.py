#!/usr/bin/env python3
"""
Real-time Data Quality Monitoring System

This module provides real-time monitoring of data quality with live recommendations,
alerts, and continuous assessment of datasets.
"""

import pandas as pd
import numpy as np
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging
from queue import Queue
import json
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import psutil
import memory_profiler
from src.utils.enhanced_data_validation import DataQualityAnalyzer, DataQualityReport
from src.utils.error_handling import ErrorHandler, ErrorContext, ErrorCategory

logger = logging.getLogger(__name__)

@dataclass
class QualityAlert:
    """Represents a data quality alert"""
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str  # 'completeness', 'consistency', 'validity', 'uniqueness'
    message: str
    affected_columns: List[str] = field(default_factory=list)
    recommendation: str = ""
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None

@dataclass
class QualityMetrics:
    """Real-time quality metrics snapshot"""
    timestamp: datetime
    overall_score: float
    completeness_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    data_size: int
    memory_usage: float
    processing_time: float

@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring"""
    check_interval: int = 30  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'overall_score': 70.0,
        'completeness_score': 80.0,
        'consistency_score': 75.0,
        'validity_score': 85.0,
        'uniqueness_score': 70.0
    })
    max_history_size: int = 1000
    enable_file_watching: bool = True
    enable_memory_monitoring: bool = True
    enable_performance_tracking: bool = True

class DataQualityRecommendationEngine:
    """Generates real-time recommendations for data quality improvement"""
    
    def __init__(self):
        self.recommendation_rules = {
            'completeness': self._completeness_recommendations,
            'consistency': self._consistency_recommendations,
            'validity': self._validity_recommendations,
            'uniqueness': self._uniqueness_recommendations
        }
    
    def generate_recommendations(self, df: pd.DataFrame, 
                               quality_report: DataQualityReport,
                               historical_metrics: List[QualityMetrics]) -> List[str]:
        """
        Generate real-time recommendations based on current data quality
        
        Args:
            df: Current dataframe
            quality_report: Latest quality assessment
            historical_metrics: Historical quality metrics
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Category-specific recommendations
        for category, score in [
            ('completeness', quality_report.completeness_score),
            ('consistency', quality_report.consistency_score),
            ('validity', quality_report.validity_score),
            ('uniqueness', quality_report.uniqueness_score)
        ]:
            if score < 80:  # Threshold for generating recommendations
                category_recs = self.recommendation_rules[category](df, quality_report, score)
                recommendations.extend(category_recs)
        
        # Trend-based recommendations
        if len(historical_metrics) >= 3:
            trend_recs = self._analyze_trends(historical_metrics)
            recommendations.extend(trend_recs)
        
        # Performance recommendations
        perf_recs = self._performance_recommendations(df, historical_metrics)
        recommendations.extend(perf_recs)
        
        return recommendations
    
    def _completeness_recommendations(self, df: pd.DataFrame, 
                                    report: DataQualityReport, score: float) -> List[str]:
        """Generate completeness-specific recommendations"""
        recommendations = []
        
        missing_analysis = df.isnull().sum() / len(df) * 100
        high_missing = missing_analysis[missing_analysis > 20]
        
        if len(high_missing) > 0:
            for col, missing_pct in high_missing.items():
                if missing_pct > 50:
                    recommendations.append(
                        f"🔴 CRITICAL: Column '{col}' has {missing_pct:.1f}% missing values. "
                        "Consider removing this column or investigating data collection issues."
                    )
                elif missing_pct > 30:
                    recommendations.append(
                        f"🟡 HIGH: Column '{col}' has {missing_pct:.1f}% missing values. "
                        "Consider imputation strategies (median/mode/forward-fill) or "
                        "investigate patterns in missing data."
                    )
                else:
                    recommendations.append(
                        f"🟢 MEDIUM: Column '{col}' has {missing_pct:.1f}% missing values. "
                        "Standard imputation methods should work well."
                    )
        
        # Pattern-based recommendations
        if score < 90:
            recommendations.append(
                "💡 TIP: Use `df.isnull().sum()` to identify missing value patterns. "
                "Consider using pandas `fillna()` with appropriate strategies."
            )
        
        return recommendations
    
    def _consistency_recommendations(self, df: pd.DataFrame, 
                                   report: DataQualityReport, score: float) -> List[str]:
        """Generate consistency-specific recommendations"""
        recommendations = []
        
        # Analyze categorical columns for consistency issues
        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().head(1000)
            if len(sample) > 0:
                # Check for case inconsistencies
                unique_values = sample.unique()
                case_variants = {}
                for val in unique_values:
                    lower_val = str(val).lower()
                    if lower_val in case_variants:
                        case_variants[lower_val].append(val)
                    else:
                        case_variants[lower_val] = [val]
                
                inconsistent_cases = {k: v for k, v in case_variants.items() if len(v) > 1}
                if inconsistent_cases:
                    recommendations.append(
                        f"🔧 CONSISTENCY: Column '{col}' has case inconsistencies. "
                        f"Examples: {list(inconsistent_cases.values())[0]}. "
                        "Use `df[col].str.lower()` or `df[col].str.title()` to standardize."
                    )
        
        # Check for outliers in numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            
            if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                recommendations.append(
                    f"📊 OUTLIERS: Column '{col}' has {len(outliers)} potential outliers "
                    f"({len(outliers)/len(df)*100:.1f}%). Consider using "
                    "`scipy.stats.zscore()` or IQR method for outlier detection."
                )
        
        return recommendations
    
    def _validity_recommendations(self, df: pd.DataFrame, 
                                report: DataQualityReport, score: float) -> List[str]:
        """Generate validity-specific recommendations"""
        recommendations = []
        
        # Check for data type issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric data is stored as string
                sample = df[col].dropna().head(1000)
                numeric_like = 0
                for val in sample:
                    try:
                        float(str(val).replace(',', '').replace('$', '').replace('%', ''))
                        numeric_like += 1
                    except ValueError:
                        pass
                
                if numeric_like > len(sample) * 0.8:  # More than 80% numeric-like
                    recommendations.append(
                        f"🔄 DATA TYPE: Column '{col}' appears to contain numeric data "
                        "stored as text. Consider using `pd.to_numeric(df[col], errors='coerce')` "
                        "to convert to proper numeric type."
                    )
        
        # Check for infinite values
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                inf_count = np.isinf(df[col]).sum()
                recommendations.append(
                    f"⚠️ INVALID: Column '{col}' contains {inf_count} infinite values. "
                    "Use `df.replace([np.inf, -np.inf], np.nan)` to handle infinite values."
                )
        
        return recommendations
    
    def _uniqueness_recommendations(self, df: pd.DataFrame, 
                                  report: DataQualityReport, score: float) -> List[str]:
        """Generate uniqueness-specific recommendations"""
        recommendations = []
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(df)) * 100
            if duplicate_pct > 5:
                recommendations.append(
                    f"🔁 DUPLICATES: Dataset has {duplicate_count} duplicate rows "
                    f"({duplicate_pct:.1f}%). Use `df.drop_duplicates()` to remove duplicates. "
                    "Consider investigating why duplicates exist."
                )
            else:
                recommendations.append(
                    f"🔁 DUPLICATES: Dataset has {duplicate_count} duplicate rows "
                    f"({duplicate_pct:.1f}%). Minor issue - use `df.drop_duplicates()` if needed."
                )
        
        # Check for low-uniqueness columns
        for col in df.columns:
            if len(df[col].dropna()) > 0:
                unique_pct = (df[col].nunique() / len(df[col].dropna())) * 100
                if unique_pct < 1 and df[col].nunique() > 1:
                    recommendations.append(
                        f"📉 LOW UNIQUENESS: Column '{col}' has very low uniqueness "
                        f"({unique_pct:.1f}%). Investigate if this column provides value "
                        "for analysis or consider combining with other features."
                    )
        
        return recommendations
    
    def _analyze_trends(self, historical_metrics: List[QualityMetrics]) -> List[str]:
        """Analyze quality trends over time"""
        recommendations = []
        
        if len(historical_metrics) < 3:
            return recommendations
        
        recent_metrics = historical_metrics[-3:]
        
        # Check for declining trends
        for metric_name in ['overall_score', 'completeness_score', 'consistency_score', 
                          'validity_score', 'uniqueness_score']:
            values = [getattr(m, metric_name) for m in recent_metrics]
            if len(values) >= 3:
                # Simple trend detection
                if values[-1] < values[-2] < values[-3]:
                    decline = values[-3] - values[-1]
                    recommendations.append(
                        f"📉 TREND ALERT: {metric_name.replace('_', ' ').title()} is declining "
                        f"(dropped {decline:.1f} points). Monitor data sources and "
                        "investigate potential issues."
                    )
        
        return recommendations
    
    def _performance_recommendations(self, df: pd.DataFrame, 
                                   historical_metrics: List[QualityMetrics]) -> List[str]:
        """Generate performance-related recommendations"""
        recommendations = []
        
        # Memory usage recommendations
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        if memory_usage_mb > 500:  # More than 500MB
            recommendations.append(
                f"💾 MEMORY: Dataset uses {memory_usage_mb:.1f}MB of memory. "
                "Consider using `df.dtypes` optimization, chunked processing, "
                "or `pd.read_csv(chunksize=1000)` for large files."
            )
        
        # Data size recommendations
        if len(df) > 100000:  # More than 100k rows
            recommendations.append(
                f"📏 SIZE: Dataset has {len(df):,} rows. Consider sampling for "
                "exploratory analysis: `df.sample(n=10000)` or use "
                "`sklearn.model_selection.train_test_split()` for analysis."
            )
        
        return recommendations

class FileWatcher(FileSystemEventHandler):
    """Watches for file system changes in dataset directories"""
    
    def __init__(self, monitor: 'RealTimeQualityMonitor'):
        self.monitor = monitor
        
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in ['.csv', '.xlsx', '.xls', '.json']:
                logger.info(f"File modified: {file_path}")
                self.monitor.schedule_quality_check(file_path)
    
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in ['.csv', '.xlsx', '.xls', '.json']:
                logger.info(f"New file detected: {file_path}")
                self.monitor.schedule_quality_check(file_path)

class RealTimeQualityMonitor:
    """Real-time data quality monitoring system"""
    
    def __init__(self, config: MonitoringConfig = None):
        """
        Initialize real-time quality monitor
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        self.quality_analyzer = DataQualityAnalyzer()
        self.recommendation_engine = DataQualityRecommendationEngine()
        self.error_handler = ErrorHandler()
        
        # State management
        self.is_running = False
        self.metrics_history = deque(maxlen=self.config.max_history_size)
        self.alerts_queue = Queue()
        self.active_datasets = {}  # file_path -> last_check_time
        
        # Threading components
        self.scheduler = BackgroundScheduler()
        self.file_observer = None
        self.monitoring_thread = None
        
        # Callbacks
        self.alert_callbacks: List[Callable[[QualityAlert], None]] = []
        self.metrics_callbacks: List[Callable[[QualityMetrics], None]] = []
    
    def add_alert_callback(self, callback: Callable[[QualityAlert], None]):
        """Add callback for quality alerts"""
        self.alert_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable[[QualityMetrics], None]):
        """Add callback for metrics updates"""
        self.metrics_callbacks.append(callback)
    
    def start_monitoring(self, watch_directories: Optional[List[str]] = None):
        """
        Start real-time monitoring
        
        Args:
            watch_directories: Directories to watch for file changes
        """
        if self.is_running:
            logger.warning("Monitor is already running")
            return
        
        logger.info("Starting real-time data quality monitoring")
        self.is_running = True
        
        # Start scheduler for periodic checks
        self.scheduler.start()
        
        # Setup file watching if enabled
        if self.config.enable_file_watching and watch_directories:
            self._setup_file_watching(watch_directories)
        
        # Start background monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Real-time monitoring started successfully")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if not self.is_running:
            return
        
        logger.info("Stopping real-time monitoring")
        self.is_running = False
        
        # Stop scheduler
        if self.scheduler.running:
            self.scheduler.shutdown()
        
        # Stop file observer
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Real-time monitoring stopped")
    
    def add_dataset(self, file_path: Union[str, Path], check_immediately: bool = True):
        """
        Add dataset for monitoring
        
        Args:
            file_path: Path to dataset file
            check_immediately: Whether to perform immediate quality check
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Dataset file not found: {file_path}")
            return
        
        self.active_datasets[str(file_path)] = datetime.now()
        logger.info(f"Added dataset for monitoring: {file_path}")
        
        if check_immediately:
            self.check_dataset_quality(file_path)
        
        # Schedule periodic checks
        self.scheduler.add_job(
            func=self.check_dataset_quality,
            trigger=IntervalTrigger(seconds=self.config.check_interval),
            args=[file_path],
            id=f"quality_check_{file_path.name}",
            replace_existing=True
        )
    
    def remove_dataset(self, file_path: Union[str, Path]):
        """Remove dataset from monitoring"""
        file_path = str(Path(file_path))
        
        if file_path in self.active_datasets:
            del self.active_datasets[file_path]
            
            # Remove scheduled job
            try:
                job_id = f"quality_check_{Path(file_path).name}"
                self.scheduler.remove_job(job_id)
            except Exception:
                pass
            
            logger.info(f"Removed dataset from monitoring: {file_path}")
    
    def check_dataset_quality(self, file_path: Union[str, Path]) -> Optional[QualityMetrics]:
        """
        Perform quality check on a specific dataset
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            Quality metrics or None if check failed
        """
        try:
            start_time = time.time()
            file_path = Path(file_path)
            
            # Load dataset
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return None
            
            # Perform quality analysis
            quality_report = self.quality_analyzer.analyze_quality(df)
            processing_time = time.time() - start_time
            
            # Calculate memory usage
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            # Create metrics snapshot
            metrics = QualityMetrics(
                timestamp=datetime.now(),
                overall_score=quality_report.overall_score,
                completeness_score=quality_report.completeness_score,
                consistency_score=quality_report.consistency_score,
                validity_score=quality_report.validity_score,
                uniqueness_score=quality_report.uniqueness_score,
                data_size=len(df),
                memory_usage=memory_usage,
                processing_time=processing_time
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Check for alerts
            self._check_alerts(metrics, quality_report, df)
            
            # Generate recommendations
            recommendations = self.recommendation_engine.generate_recommendations(
                df, quality_report, list(self.metrics_history)
            )
            
            # Trigger callbacks
            for callback in self.metrics_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"Error in metrics callback: {e}")
            
            # Log results
            logger.info(
                f"Quality check completed for {file_path.name}: "
                f"Overall Score: {quality_report.overall_score:.1f}, "
                f"Processing Time: {processing_time:.2f}s"
            )
            
            return metrics
            
        except Exception as e:
            error_context = ErrorContext(
                operation="quality_check",
                component="RealTimeQualityMonitor",
                additional_data={"file_path": str(file_path)}
            )
            self.error_handler.handle_error(e, error_context)
            return None
    
    def schedule_quality_check(self, file_path: Union[str, Path]):
        """Schedule a quality check for later execution"""
        if str(file_path) in self.active_datasets:
            # Update last check time and schedule immediate check
            self.active_datasets[str(file_path)] = datetime.now()
            
            # Run check in background
            threading.Thread(
                target=self.check_dataset_quality,
                args=[file_path],
                daemon=True
            ).start()
    
    def get_latest_metrics(self, count: int = 10) -> List[QualityMetrics]:
        """Get latest quality metrics"""
        return list(self.metrics_history)[-count:]
    
    def get_alerts(self, count: int = 50) -> List[QualityAlert]:
        """Get recent alerts"""
        alerts = []
        temp_queue = Queue()
        
        # Extract alerts without losing them
        while not self.alerts_queue.empty() and len(alerts) < count:
            alert = self.alerts_queue.get()
            alerts.append(alert)
            temp_queue.put(alert)
        
        # Put alerts back
        while not temp_queue.empty():
            self.alerts_queue.put(temp_queue.get())
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'is_running': self.is_running,
            'active_datasets': len(self.active_datasets),
            'total_checks': len(self.metrics_history),
            'pending_alerts': self.alerts_queue.qsize(),
            'last_check': self.metrics_history[-1].timestamp if self.metrics_history else None,
            'memory_usage': f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB"
        }
    
    def _setup_file_watching(self, directories: List[str]):
        """Setup file system watching"""
        try:
            self.file_observer = Observer()
            event_handler = FileWatcher(self)
            
            for directory in directories:
                if os.path.exists(directory):
                    self.file_observer.schedule(event_handler, directory, recursive=True)
                    logger.info(f"Watching directory: {directory}")
            
            self.file_observer.start()
            
        except Exception as e:
            logger.error(f"Failed to setup file watching: {e}")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_running:
            try:
                # Process any pending alerts
                self._process_alerts()
                
                # System health check
                if self.config.enable_memory_monitoring:
                    self._monitor_system_health()
                
                time.sleep(1)  # Small delay to prevent high CPU usage
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Longer delay on error
    
    def _check_alerts(self, metrics: QualityMetrics, 
                     quality_report: DataQualityReport, df: pd.DataFrame):
        """Check if any alerts should be triggered"""
        
        # Score-based alerts
        for score_name, threshold in self.config.alert_thresholds.items():
            score_value = getattr(metrics, score_name)
            
            if score_value < threshold:
                severity = self._determine_severity(score_value, threshold)
                alert = QualityAlert(
                    timestamp=datetime.now(),
                    severity=severity,
                    category=score_name.replace('_score', ''),
                    message=f"{score_name.replace('_', ' ').title()} dropped below threshold",
                    metric_value=score_value,
                    threshold_value=threshold,
                    recommendation=f"Current {score_name}: {score_value:.1f}, "
                                 f"Threshold: {threshold:.1f}. Investigate data quality issues."
                )
                
                self.alerts_queue.put(alert)
                
                # Trigger alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
    
    def _determine_severity(self, value: float, threshold: float) -> str:
        """Determine alert severity based on value and threshold"""
        difference = threshold - value
        
        if difference > 20:
            return 'critical'
        elif difference > 10:
            return 'high'
        elif difference > 5:
            return 'medium'
        else:
            return 'low'
    
    def _process_alerts(self):
        """Process and log alerts"""
        while not self.alerts_queue.empty():
            try:
                alert = self.alerts_queue.get_nowait()
                logger.warning(f"QUALITY ALERT [{alert.severity.upper()}]: {alert.message}")
            except:
                break
    
    def _monitor_system_health(self):
        """Monitor system health metrics"""
        try:
            # Memory usage
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                alert = QualityAlert(
                    timestamp=datetime.now(),
                    severity='high',
                    category='system',
                    message=f"High memory usage: {memory_percent:.1f}%",
                    recommendation="Consider reducing dataset size or optimizing memory usage"
                )
                self.alerts_queue.put(alert)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 95:
                alert = QualityAlert(
                    timestamp=datetime.now(),
                    severity='medium',
                    category='system',
                    message=f"High CPU usage: {cpu_percent:.1f}%",
                    recommendation="Monitor system performance and consider reducing monitoring frequency"
                )
                self.alerts_queue.put(alert)
                
        except Exception as e:
            logger.debug(f"System health monitoring error: {e}")

# Convenience functions for easy usage
def start_quality_monitoring(datasets: List[str], 
                           watch_directories: Optional[List[str]] = None,
                           config: Optional[MonitoringConfig] = None) -> RealTimeQualityMonitor:
    """
    Start monitoring multiple datasets
    
    Args:
        datasets: List of dataset file paths
        watch_directories: Directories to watch for changes
        config: Monitoring configuration
        
    Returns:
        Configured and started monitor instance
    """
    monitor = RealTimeQualityMonitor(config)
    
    # Add datasets
    for dataset in datasets:
        monitor.add_dataset(dataset)
    
    # Start monitoring
    monitor.start_monitoring(watch_directories)
    
    return monitor 
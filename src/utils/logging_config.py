"""
Comprehensive Logging Configuration for Research Assistant AI Agent

Features:
- Structured logging with JSON format
- Multiple handlers (file, console, rotating files)
- Performance monitoring and metrics
- Security-aware logging (no sensitive data)
- Context-aware logging with correlation IDs
- Integration with configuration management
"""

import logging
import logging.handlers
import json
import os
import sys
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
from pathlib import Path
import uuid
from contextlib import contextmanager

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured logging in JSON format
    """
    
    def __init__(self, include_fields: List[str] = None):
        """
        Initialize structured formatter
        
        Args:
            include_fields: List of fields to include in log records
        """
        super().__init__()
        self.include_fields = include_fields or [
            "timestamp", "level", "logger", "message", "module", "function"
        ]
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_data[key] = value
        
        # Filter fields if specified
        if self.include_fields:
            filtered_data = {}
            for field in self.include_fields:
                if field in log_data:
                    filtered_data[field] = log_data[field]
            log_data = filtered_data
        
        return json.dumps(log_data, default=str, ensure_ascii=False)

class SecurityFilter(logging.Filter):
    """
    Filter to remove sensitive information from logs
    """
    
    SENSITIVE_PATTERNS = [
        r'(?i)(api[_-]?key|token|password|secret|credential)[\s]*[:=][\s]*[\'"]?([^\s\'"]+)',
        r'(?i)(sk-[a-zA-Z0-9]{32,})',  # OpenAI API keys
        r'(?i)(mongodb\+srv://[^:]+:[^@]+@)',  # MongoDB connection strings with credentials
        r'(?i)(Bearer\s+[a-zA-Z0-9\-._~+/]+=*)',  # Bearer tokens
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter out sensitive information from log records
        
        Args:
            record: Log record to filter
            
        Returns:
            True if record should be logged, False otherwise
        """
        import re
        
        # Check message for sensitive data
        message = record.getMessage()
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, message):
                # Replace sensitive data with placeholder
                record.msg = re.sub(pattern, r'\1: [REDACTED]', str(record.msg))
                record.args = ()  # Clear args to prevent re-formatting issues
        
        # Check extra fields for sensitive data
        for key, value in record.__dict__.items():
            if isinstance(value, str):
                for pattern in self.SENSITIVE_PATTERNS:
                    if re.search(pattern, value):
                        setattr(record, key, '[REDACTED]')
        
        return True

class PerformanceLogger:
    """
    Context manager for performance logging
    """
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        """
        Initialize performance logger
        
        Args:
            logger: Logger instance
            operation: Operation name
            **context: Additional context for logging
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
        self.correlation_id = str(uuid.uuid4())[:8]
    
    def __enter__(self):
        """Start performance monitoring"""
        self.start_time = datetime.now()
        self.logger.info(
            f"Operation started: {self.operation}",
            extra={
                "operation": self.operation,
                "correlation_id": self.correlation_id,
                "status": "started",
                **self.context
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End performance monitoring and log results"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        log_data = {
            "operation": self.operation,
            "correlation_id": self.correlation_id,
            "duration": duration,
            "status": "completed" if exc_type is None else "failed",
            **self.context
        }
        
        if exc_type is not None:
            log_data.update({
                "error_type": exc_type.__name__,
                "error_message": str(exc_val)
            })
            self.logger.error(f"Operation failed: {self.operation}", extra=log_data)
        else:
            self.logger.info(f"Operation completed: {self.operation}", extra=log_data)

class LoggingConfig:
    """
    Centralized logging configuration manager
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize logging configuration
        
        Args:
            config: Logging configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.loggers = {}
        self._setup_directories()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration"""
        return {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "handlers": [
                {
                    "type": "console",
                    "level": "INFO"
                },
                {
                    "type": "file",
                    "filename": "logs/research_agent.log",
                    "level": "DEBUG",
                    "max_bytes": 10485760,  # 10MB
                    "backup_count": 5
                }
            ],
            "structured_logging": {
                "enabled": True,
                "format": "json",
                "fields": ["timestamp", "level", "logger", "message", "module"]
            },
            "security": {
                "filter_sensitive": True,
                "log_sensitive_data": False
            }
        }
    
    def _setup_directories(self):
        """Create necessary directories for log files"""
        for handler_config in self.config.get("handlers", []):
            if handler_config.get("type") == "file" and "filename" in handler_config:
                log_file = Path(handler_config["filename"])
                log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self, 
                     logger_name: str = None,
                     log_level: str = None,
                     log_file: str = None,
                     enable_structured_logging: bool = None,
                     enable_performance_logging: bool = True) -> logging.Logger:
        """
        Setup and configure logger
        
        Args:
            logger_name: Name of the logger
            log_level: Logging level
            log_file: Log file path
            enable_structured_logging: Enable structured JSON logging
            enable_performance_logging: Enable performance logging features
            
        Returns:
            Configured logger instance
        """
        logger_name = logger_name or "research_agent"
        log_level = log_level or self.config.get("level", "INFO")
        enable_structured_logging = enable_structured_logging if enable_structured_logging is not None else self.config.get("structured_logging", {}).get("enabled", True)
        
        # Get or create logger
        logger = logging.getLogger(logger_name)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Setup handlers based on configuration
        for handler_config in self.config.get("handlers", []):
            handler = self._create_handler(handler_config, enable_structured_logging)
            if handler:
                logger.addHandler(handler)
        
        # Add security filter if enabled
        if self.config.get("security", {}).get("filter_sensitive", True):
            security_filter = SecurityFilter()
            for handler in logger.handlers:
                handler.addFilter(security_filter)
        
        # Store logger reference
        self.loggers[logger_name] = logger
        
        # Add performance logging capabilities
        if enable_performance_logging:
            logger.performance = lambda operation, **context: PerformanceLogger(logger, operation, **context)
        
        return logger
    
    def _create_handler(self, handler_config: Dict[str, Any], structured_logging: bool) -> logging.Handler:
        """
        Create logging handler based on configuration
        
        Args:
            handler_config: Handler configuration
            structured_logging: Whether to use structured logging
            
        Returns:
            Configured logging handler
        """
        handler_type = handler_config.get("type", "console")
        level = handler_config.get("level", "INFO")
        
        # Create handler based on type
        if handler_type == "console":
            handler = logging.StreamHandler(sys.stdout)
        elif handler_type == "file":
            filename = handler_config.get("filename", "logs/app.log")
            max_bytes = handler_config.get("max_bytes", 10485760)
            backup_count = handler_config.get("backup_count", 5)
            
            # Ensure directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            handler = logging.handlers.RotatingFileHandler(
                filename=filename,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        elif handler_type == "syslog":
            handler = logging.handlers.SysLogHandler()
        else:
            return None
        
        # Set level
        handler.setLevel(getattr(logging, level.upper()))
        
        # Set formatter
        if structured_logging and self.config.get("structured_logging", {}).get("enabled", True):
            include_fields = self.config.get("structured_logging", {}).get("fields")
            formatter = StructuredFormatter(include_fields)
        else:
            format_string = handler_config.get("format") or self.config.get("format")
            formatter = logging.Formatter(format_string)
        
        handler.setFormatter(formatter)
        
        return handler
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get existing logger or create new one
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if name in self.loggers:
            return self.loggers[name]
        return self.setup_logging(name)
    
    def update_log_level(self, logger_name: str, level: str):
        """
        Update log level for specific logger
        
        Args:
            logger_name: Name of the logger
            level: New log level
        """
        if logger_name in self.loggers:
            logger = self.loggers[logger_name]
            logger.setLevel(getattr(logging, level.upper()))
            
            # Update handler levels
            for handler in logger.handlers:
                handler.setLevel(getattr(logging, level.upper()))

# Global logging configuration instance
_logging_config = None

def setup_logging(logger_name: str = None,
                 log_level: str = None, 
                 log_file: str = None,
                 enable_structured_logging: bool = True,
                 config: Dict[str, Any] = None) -> logging.Logger:
    """
    Global function to setup logging
    
    Args:
        logger_name: Name of the logger
        log_level: Logging level
        log_file: Log file path
        enable_structured_logging: Enable structured JSON logging
        config: Custom logging configuration
        
    Returns:
        Configured logger instance
    """
    global _logging_config
    
    if _logging_config is None:
        _logging_config = LoggingConfig(config)
    
    return _logging_config.setup_logging(
        logger_name=logger_name,
        log_level=log_level,
        log_file=log_file,
        enable_structured_logging=enable_structured_logging
    )

def get_logger(name: str = None) -> logging.Logger:
    """
    Get logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _logging_config
    
    if _logging_config is None:
        _logging_config = LoggingConfig()
    
    return _logging_config.get_logger(name or "research_agent")

@contextmanager
def log_performance(logger: logging.Logger, operation: str, **context):
    """
    Context manager for performance logging
    
    Args:
        logger: Logger instance
        operation: Operation name
        **context: Additional context
    """
    with PerformanceLogger(logger, operation, **context):
        yield

# Integration with configuration management
def setup_logging_from_config(config_manager):
    """
    Setup logging using configuration manager
    
    Args:
        config_manager: ConfigManager instance
    """
    logging_config = config_manager.get_section("logging")
    
    global _logging_config
    _logging_config = LoggingConfig(logging_config)
    
    return _logging_config.setup_logging()

# Example usage and testing
if __name__ == "__main__":
    # Test basic logging setup
    logger = setup_logging("test_logger", "DEBUG", enable_structured_logging=True)
    
    logger.info("Testing basic logging")
    logger.debug("Debug message", extra={"user_id": "test123", "action": "test"})
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test performance logging
    with log_performance(logger, "test_operation", user_id="test123"):
        import time
        time.sleep(0.1)  # Simulate work
    
    # Test sensitive data filtering
    logger.info("API Key: sk-1234567890abcdef")  # Should be filtered
    logger.info("Password: secret123")  # Should be filtered
    
    print("Logging test completed!") 
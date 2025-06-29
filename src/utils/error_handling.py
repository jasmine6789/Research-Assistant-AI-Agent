"""
Comprehensive Error Handling System for Research Assistant AI Agent

Features:
- Custom exception hierarchy
- Retry mechanisms with exponential backoff
- Circuit breaker pattern
- Error recovery strategies
- Detailed error reporting and logging
- Integration with monitoring systems
"""

import logging
import time
import functools
import traceback
from typing import Dict, Any, Optional, List, Callable, Type, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import threading

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    API_ERROR = "api_error"
    DATABASE_ERROR = "database_error"
    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_ERROR = "network_error"
    PROCESSING_ERROR = "processing_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ErrorContext:
    """Context information for errors"""
    operation: str
    component: str
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    additional_data: Dict[str, Any] = field(default_factory=dict)

class BaseApplicationError(Exception):
    """Base exception class for application-specific errors"""
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None,
                 original_exception: Optional[Exception] = None,
                 recoverable: bool = True):
        """
        Initialize base application error
        
        Args:
            message: Error message
            category: Error category
            severity: Error severity
            context: Error context information
            original_exception: Original exception that caused this error
            recoverable: Whether the error is recoverable
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext("unknown", "unknown")
        self.original_exception = original_exception
        self.recoverable = recoverable
        self.error_id = f"{self.category.value}_{int(time.time())}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation"""
        return {
            "error_id": self.error_id,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "context": {
                "operation": self.context.operation,
                "component": self.context.component,
                "user_id": self.context.user_id,
                "correlation_id": self.context.correlation_id,
                "timestamp": self.context.timestamp.isoformat(),
                "additional_data": self.context.additional_data
            },
            "original_exception": str(self.original_exception) if self.original_exception else None,
            "traceback": traceback.format_exc() if self.original_exception else None
        }

class RetryableError(BaseApplicationError):
    """Error that can be retried"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("recoverable", True)
        super().__init__(message, **kwargs)

class NonRetryableError(BaseApplicationError):
    """Error that should not be retried"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("recoverable", False)
        super().__init__(message, **kwargs)

class APIError(RetryableError):
    """API-related errors"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        kwargs.setdefault("category", ErrorCategory.API_ERROR)
        super().__init__(message, **kwargs)
        self.status_code = status_code

class DatabaseError(RetryableError):
    """Database-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.DATABASE_ERROR)
        super().__init__(message, **kwargs)

class ValidationError(NonRetryableError):
    """Validation errors"""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        kwargs.setdefault("category", ErrorCategory.VALIDATION_ERROR)
        kwargs.setdefault("severity", ErrorSeverity.LOW)
        super().__init__(message, **kwargs)
        self.field = field

class ConfigurationError(NonRetryableError):
    """Configuration-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION_ERROR)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)

class RateLimitError(RetryableError):
    """Rate limiting errors"""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        kwargs.setdefault("category", ErrorCategory.RATE_LIMIT_ERROR)
        super().__init__(message, **kwargs)
        self.retry_after = retry_after

class TimeoutError(RetryableError):
    """Timeout errors"""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None, **kwargs):
        kwargs.setdefault("category", ErrorCategory.TIMEOUT_ERROR)
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration

@dataclass
class RetryConfig:
    """Configuration for retry mechanism"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [RetryableError])

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception

class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to function"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs):
        """
        Call function with circuit breaker protection
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise NonRetryableError(
                        "Circuit breaker is OPEN - service unavailable",
                        category=ErrorCategory.PROCESSING_ERROR,
                        severity=ErrorSeverity.HIGH
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset"""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.config.recovery_timeout)
    
    def _on_success(self):
        """Handle successful call"""
        with self.lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN

def retry_with_backoff(config: RetryConfig = None):
    """
    Decorator for retry with exponential backoff
    
    Args:
        config: Retry configuration
        
    Returns:
        Decorator function
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if exception is retryable
                    if not any(isinstance(e, exc_type) for exc_type in config.retryable_exceptions):
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise
                    
                    # Don't retry on last attempt
                    if attempt == config.max_attempts - 1:
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter if enabled
                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    
                    time.sleep(delay)
            
            # All attempts failed
            raise RetryableError(
                f"All {config.max_attempts} retry attempts failed for {func.__name__}",
                original_exception=last_exception
            )
        
        return wrapper
    return decorator

def handle_api_error(max_retries: int = 3, backoff_factor: float = 2):
    """
    Decorator specifically for handling API errors
    
    Args:
        max_retries: Maximum number of retries
        backoff_factor: Backoff factor for delays
        
    Returns:
        Decorator function
    """
    config = RetryConfig(
        max_attempts=max_retries,
        backoff_factor=backoff_factor,
        retryable_exceptions=[APIError, RateLimitError, TimeoutError]
    )
    return retry_with_backoff(config)

class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize error handler
        
        Args:
            logger: Logger instance for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_callbacks = {}
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {}
        }
    
    def register_callback(self, error_type: Type[Exception], callback: Callable):
        """
        Register callback for specific error type
        
        Args:
            error_type: Exception type to handle
            callback: Callback function to execute
        """
        self.error_callbacks[error_type] = callback
    
    def handle_error(self, 
                    error: Exception,
                    context: Optional[ErrorContext] = None,
                    reraise: bool = True) -> Optional[Dict[str, Any]]:
        """
        Handle error with logging, callbacks, and reporting
        
        Args:
            error: Exception to handle
            context: Error context information
            reraise: Whether to re-raise the exception
            
        Returns:
            Error information dictionary
        """
        # Convert to application error if needed
        if not isinstance(error, BaseApplicationError):
            app_error = BaseApplicationError(
                str(error),
                context=context,
                original_exception=error
            )
        else:
            app_error = error
        
        # Update statistics
        self._update_error_stats(app_error)
        
        # Log error
        self._log_error(app_error)
        
        # Execute callbacks
        self._execute_callbacks(app_error)
        
        # Create error info
        error_info = app_error.to_dict()
        
        if reraise:
            raise app_error
        
        return error_info
    
    def _update_error_stats(self, error: BaseApplicationError):
        """Update error statistics"""
        self.error_stats["total_errors"] += 1
        
        category = error.category.value
        if category not in self.error_stats["errors_by_category"]:
            self.error_stats["errors_by_category"][category] = 0
        self.error_stats["errors_by_category"][category] += 1
        
        severity = error.severity.value
        if severity not in self.error_stats["errors_by_severity"]:
            self.error_stats["errors_by_severity"][severity] = 0
        self.error_stats["errors_by_severity"][severity] += 1
    
    def _log_error(self, error: BaseApplicationError):
        """Log error with appropriate level"""
        log_data = error.to_dict()
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error.message, extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(error.message, extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error.message, extra=log_data)
        else:
            self.logger.info(error.message, extra=log_data)
    
    def _execute_callbacks(self, error: BaseApplicationError):
        """Execute registered callbacks for error type"""
        for error_type, callback in self.error_callbacks.items():
            if isinstance(error, error_type):
                try:
                    callback(error)
                except Exception as e:
                    self.logger.error(f"Error callback failed: {e}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return self.error_stats.copy()
    
    def reset_stats(self):
        """Reset error statistics"""
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {}
        }

# Global error handler instance
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler

def handle_exceptions(context: ErrorContext = None, reraise: bool = True):
    """
    Decorator to handle exceptions automatically
    
    Args:
        context: Error context information
        reraise: Whether to re-raise exceptions
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or ErrorContext(
                    operation=func.__name__,
                    component=func.__module__
                )
                return get_error_handler().handle_error(e, error_context, reraise)
        return wrapper
    return decorator

# Example usage and testing
if __name__ == "__main__":
    # Test error handling
    handler = ErrorHandler()
    
    # Test basic error
    try:
        raise APIError("Test API error", status_code=500)
    except Exception as e:
        handler.handle_error(e, reraise=False)
    
    # Test retry decorator
    @retry_with_backoff(RetryConfig(max_attempts=3, base_delay=0.1))
    def failing_function():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise RetryableError("Random failure")
        return "Success!"
    
    try:
        result = failing_function()
        print(f"Function result: {result}")
    except Exception as e:
        print(f"Function failed: {e}")
    
    # Test circuit breaker
    circuit_breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))
    
    @circuit_breaker
    def unreliable_service():
        import random
        if random.random() < 0.8:  # 80% chance of failure
            raise Exception("Service unavailable")
        return "Service response"
    
    # Test circuit breaker behavior
    for i in range(5):
        try:
            result = unreliable_service()
            print(f"Service call {i+1}: {result}")
        except Exception as e:
            print(f"Service call {i+1} failed: {e}")
    
    # Print error statistics
    print("Error statistics:", handler.get_error_stats()) 
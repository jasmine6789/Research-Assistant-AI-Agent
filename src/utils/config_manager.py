"""
Configuration Management System for Research Assistant AI Agent

This module provides centralized configuration management with:
- YAML-based configuration files
- Environment-specific overrides
- Configuration validation
- Environment variable substitution
- Type conversion and validation
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors"""
    message: str
    config_path: str
    field: str = None

class ConfigManager:
    """
    Centralized configuration management system
    
    Features:
    - Load base configuration and environment-specific overrides
    - Validate configuration values and types
    - Support environment variable substitution
    - Provide easy access to nested configuration values
    - Hot reload configuration changes
    """
    
    def __init__(self, config_dir: str = "config", environment: str = None):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (development, production, etc.)
        """
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.config_cache = {}
        self.last_loaded = None
        
        # Load configuration
        self.config = self._load_configuration()
        self._validate_configuration()
    
    def _load_configuration(self) -> Dict[str, Any]:
        """
        Load configuration from YAML files with environment overrides
        
        Returns:
            Merged configuration dictionary
        """
        try:
            # Load base configuration
            base_config_path = self.config_dir / "base.yaml"
            if not base_config_path.exists():
                raise ConfigValidationError(
                    f"Base configuration file not found: {base_config_path}",
                    str(base_config_path)
                )
            
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
            
            logger.info(f"Loaded base configuration from {base_config_path}")
            
            # Load environment-specific configuration
            env_config_path = self.config_dir / f"{self.environment}.yaml"
            env_config = {}
            
            if env_config_path.exists():
                with open(env_config_path, 'r') as f:
                    env_config = yaml.safe_load(f)
                logger.info(f"Loaded environment configuration from {env_config_path}")
            else:
                logger.warning(f"Environment configuration not found: {env_config_path}")
            
            # Merge configurations
            merged_config = self._deep_merge(base_config, env_config)
            
            # Substitute environment variables
            merged_config = self._substitute_env_vars(merged_config)
            
            self.last_loaded = datetime.now()
            logger.info(f"Configuration loaded successfully for environment: {self.environment}")
            
            return merged_config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigValidationError(
                f"Configuration loading failed: {e}",
                str(self.config_dir)
            )
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override values taking precedence
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variables in configuration values
        
        Supports format: ${ENV_VAR_NAME} or ${ENV_VAR_NAME:default_value}
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with environment variables substituted
        """
        import re
        
        def substitute_value(value):
            if isinstance(value, str):
                # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
                pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
                
                def replace_env_var(match):
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ""
                    return os.getenv(var_name, default_value)
                
                return re.sub(pattern, replace_env_var, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value
        
        return substitute_value(config)
    
    def _validate_configuration(self):
        """
        Validate configuration values and required fields
        """
        required_fields = [
            "api.openai.model",
            "database.mongodb.database_name",
            "logging.level",
            "agents.search.max_papers_per_search"
        ]
        
        for field_path in required_fields:
            try:
                value = self.get(field_path)
                if value is None:
                    raise ConfigValidationError(
                        f"Required configuration field is missing or None",
                        str(self.config_dir),
                        field_path
                    )
            except KeyError:
                raise ConfigValidationError(
                    f"Required configuration field not found",
                    str(self.config_dir),
                    field_path
                )
        
        # Validate specific field types and ranges
        self._validate_field_types()
        self._validate_field_ranges()
        
        logger.info("Configuration validation completed successfully")
    
    def _validate_field_types(self):
        """Validate that configuration fields have correct types"""
        type_validations = {
            "api.openai.max_tokens": int,
            "api.openai.temperature": (int, float),
            "agents.search.max_papers_per_search": int,
            "agents.hypothesis.confidence_threshold": (int, float),
            "logging.level": str,
            "performance.max_concurrent_agents": int,
            "human_in_loop.enabled": bool
        }
        
        for field_path, expected_type in type_validations.items():
            try:
                value = self.get(field_path)
                if value is not None and not isinstance(value, expected_type):
                    raise ConfigValidationError(
                        f"Field {field_path} has invalid type. Expected {expected_type}, got {type(value)}",
                        str(self.config_dir),
                        field_path
                    )
            except KeyError:
                # Field might be optional
                pass
    
    def _validate_field_ranges(self):
        """Validate that numeric fields are within acceptable ranges"""
        range_validations = {
            "api.openai.max_tokens": (1, 32000),
            "api.openai.temperature": (0.0, 2.0),
            "agents.search.max_papers_per_search": (1, 20),
            "agents.hypothesis.confidence_threshold": (0.0, 1.0),
            "performance.max_concurrent_agents": (1, 50)
        }
        
        for field_path, (min_val, max_val) in range_validations.items():
            try:
                value = self.get(field_path)
                if value is not None and not (min_val <= value <= max_val):
                    raise ConfigValidationError(
                        f"Field {field_path} value {value} is out of range [{min_val}, {max_val}]",
                        str(self.config_dir),
                        field_path
                    )
            except KeyError:
                # Field might be optional
                pass
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., "api.openai.model")
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {key_path}")
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        logger.debug(f"Configuration updated: {key_path} = {value}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Section name (e.g., "api", "agents")
            
        Returns:
            Configuration section dictionary
        """
        return self.get(section, {})
    
    def reload(self):
        """
        Reload configuration from files
        """
        logger.info("Reloading configuration...")
        self.config = self._load_configuration()
        self._validate_configuration()
        logger.info("Configuration reloaded successfully")
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"
    
    def get_log_level(self) -> str:
        """Get logging level from configuration"""
        return self.get("logging.level", "INFO")
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration section"""
        return self.get_section("api")
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get configuration for specific agent
        
        Args:
            agent_name: Name of the agent (e.g., "search", "hypothesis")
            
        Returns:
            Agent configuration dictionary
        """
        return self.get(f"agents.{agent_name}", {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration section"""
        return self.get_section("database")
    
    def export_config(self, format: str = "yaml") -> str:
        """
        Export current configuration
        
        Args:
            format: Export format ("yaml" or "json")
            
        Returns:
            Configuration as string
        """
        if format.lower() == "json":
            return json.dumps(self.config, indent=2, default=str)
        else:
            return yaml.dump(self.config, default_flow_style=False, indent=2)
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Get configuration metadata
        
        Returns:
            Dictionary with configuration info
        """
        return {
            "environment": self.environment,
            "config_dir": str(self.config_dir),
            "last_loaded": self.last_loaded.isoformat() if self.last_loaded else None,
            "config_files": [
                str(self.config_dir / "base.yaml"),
                str(self.config_dir / f"{self.environment}.yaml")
            ]
        }

# Global configuration instance
_config_manager = None

def get_config(config_dir: str = "config", environment: str = None) -> ConfigManager:
    """
    Get global configuration manager instance
    
    Args:
        config_dir: Configuration directory
        environment: Environment name
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir, environment)
    
    return _config_manager

def reload_config():
    """Reload global configuration"""
    global _config_manager
    if _config_manager:
        _config_manager.reload()

# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    config = ConfigManager()
    
    print("Configuration loaded successfully!")
    print(f"Environment: {config.environment}")
    print(f"OpenAI Model: {config.get('api.openai.model')}")
    print(f"Max Papers: {config.get('agents.search.max_papers_per_search')}")
    print(f"Log Level: {config.get_log_level()}")
    
    # Test configuration sections
    api_config = config.get_api_config()
    print(f"API Config: {api_config}")
    
    # Test agent-specific config
    search_config = config.get_agent_config("search")
    print(f"Search Agent Config: {search_config}")
    
    # Export configuration
    print("\nConfiguration (YAML):")
    print(config.export_config("yaml")[:500] + "...") 
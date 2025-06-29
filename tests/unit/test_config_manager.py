"""
Unit Tests for Configuration Manager

Tests:
- Configuration loading and merging
- Environment variable substitution
- Validation and error handling
- Hot reloading functionality
- Environment-specific overrides
"""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.utils.config_manager import ConfigManager, ConfigValidationError, get_config, reload_config


class TestConfigManager:
    """Test cases for ConfigManager class"""
    
    def test_load_base_configuration(self, config_dir):
        """Test loading base configuration file"""
        config = ConfigManager(config_dir=config_dir, environment="test")
        
        assert config.config is not None
        assert config.get("api.openai.model") == "gpt-3.5-turbo"
        assert config.get("database.mongodb.database_name") == "test_research_assistant"
    
    def test_environment_specific_overrides(self, temp_dir):
        """Test environment-specific configuration overrides"""
        config_path = Path(temp_dir) / "config"
        config_path.mkdir(exist_ok=True)
        
        # Create base config
        base_config = {
            "api": {"openai": {"model": "gpt-4", "max_tokens": 4000}},
            "logging": {"level": "INFO"}
        }
        with open(config_path / "base.yaml", 'w') as f:
            yaml.dump(base_config, f)
        
        # Create development override
        dev_config = {
            "api": {"openai": {"model": "gpt-3.5-turbo", "max_tokens": 2000}},
            "logging": {"level": "DEBUG"}
        }
        with open(config_path / "development.yaml", 'w') as f:
            yaml.dump(dev_config, f)
        
        # Test development environment
        config = ConfigManager(config_dir=str(config_path), environment="development")
        
        assert config.get("api.openai.model") == "gpt-3.5-turbo"  # Overridden
        assert config.get("api.openai.max_tokens") == 2000  # Overridden
        assert config.get("logging.level") == "DEBUG"  # Overridden
    
    def test_environment_variable_substitution(self, temp_dir):
        """Test environment variable substitution in config values"""
        config_path = Path(temp_dir) / "config"
        config_path.mkdir(exist_ok=True)
        
        # Create config with environment variables
        base_config = {
            "api": {
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "model": "${OPENAI_MODEL:gpt-3.5-turbo}"  # With default
                }
            },
            "database": {
                "mongodb": {
                    "uri": "${MONGO_URI:mongodb://localhost:27017}"
                }
            }
        }
        with open(config_path / "base.yaml", 'w') as f:
            yaml.dump(base_config, f)
        
        # Set environment variables
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key_12345',
            'OPENAI_MODEL': 'gpt-4'
        }):
            config = ConfigManager(config_dir=str(config_path))
            
            assert config.get("api.openai.api_key") == "test_key_12345"
            assert config.get("api.openai.model") == "gpt-4"
            assert config.get("database.mongodb.uri") == "mongodb://localhost:27017"  # Default used
    
    def test_configuration_validation_success(self, config_dir):
        """Test successful configuration validation"""
        # This should not raise any exceptions
        config = ConfigManager(config_dir=config_dir, environment="test")
        assert config.config is not None
    
    def test_configuration_validation_missing_required_field(self, temp_dir):
        """Test validation failure for missing required fields"""
        config_path = Path(temp_dir) / "config"
        config_path.mkdir(exist_ok=True)
        
        # Create incomplete config missing required fields
        incomplete_config = {
            "api": {"openai": {}},  # Missing model
            "logging": {"level": "INFO"}
        }
        with open(config_path / "base.yaml", 'w') as f:
            yaml.dump(incomplete_config, f)
        
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager(config_dir=str(config_path))
        
        assert "Required configuration field" in str(exc_info.value)
    
    def test_configuration_validation_invalid_types(self, temp_dir):
        """Test validation failure for invalid field types"""
        config_path = Path(temp_dir) / "config"
        config_path.mkdir(exist_ok=True)
        
        # Create config with invalid types
        invalid_config = {
            "api": {
                "openai": {
                    "model": "gpt-3.5-turbo",
                    "max_tokens": "invalid_number",  # Should be int
                    "temperature": "not_a_float"  # Should be float
                }
            },
            "database": {"mongodb": {"database_name": "test"}},
            "logging": {"level": "INFO"},
            "agents": {"search": {"max_papers_per_search": 5}}
        }
        with open(config_path / "base.yaml", 'w') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager(config_dir=str(config_path))
        
        assert "invalid type" in str(exc_info.value)
    
    def test_configuration_validation_out_of_range(self, temp_dir):
        """Test validation failure for out-of-range values"""
        config_path = Path(temp_dir) / "config"
        config_path.mkdir(exist_ok=True)
        
        # Create config with out-of-range values
        invalid_config = {
            "api": {
                "openai": {
                    "model": "gpt-3.5-turbo",
                    "max_tokens": 50000,  # Too high
                    "temperature": 5.0  # Too high
                }
            },
            "database": {"mongodb": {"database_name": "test"}},
            "logging": {"level": "INFO"},
            "agents": {"search": {"max_papers_per_search": 100}}  # Too high
        }
        with open(config_path / "base.yaml", 'w') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager(config_dir=str(config_path))
        
        assert "out of range" in str(exc_info.value)
    
    def test_get_configuration_value_with_dot_notation(self, config_manager):
        """Test getting configuration values with dot notation"""
        # Test valid paths
        assert config_manager.get("api.openai.model") == "gpt-3.5-turbo"
        assert config_manager.get("agents.search.max_papers_per_search") == 3
        
        # Test with default value
        assert config_manager.get("nonexistent.path", "default") == "default"
        
        # Test missing path without default
        with pytest.raises(KeyError):
            config_manager.get("nonexistent.path")
    
    def test_set_configuration_value(self, config_manager):
        """Test setting configuration values with dot notation"""
        # Set new value
        config_manager.set("api.openai.temperature", 0.5)
        assert config_manager.get("api.openai.temperature") == 0.5
        
        # Set nested value in new section
        config_manager.set("new.section.value", "test")
        assert config_manager.get("new.section.value") == "test"
    
    def test_get_configuration_section(self, config_manager):
        """Test getting entire configuration sections"""
        api_config = config_manager.get_section("api")
        assert "openai" in api_config
        assert api_config["openai"]["model"] == "gpt-3.5-turbo"
        
        # Test non-existent section
        empty_section = config_manager.get_section("nonexistent")
        assert empty_section == {}
    
    def test_environment_detection(self, config_dir):
        """Test environment detection methods"""
        # Test development environment
        dev_config = ConfigManager(config_dir=config_dir, environment="development")
        assert dev_config.is_development() is True
        assert dev_config.is_production() is False
        
        # Test production environment
        prod_config = ConfigManager(config_dir=config_dir, environment="production")
        assert dev_config.is_development() is False
        assert dev_config.is_production() is True
    
    def test_agent_specific_configuration(self, config_manager):
        """Test getting agent-specific configuration"""
        search_config = config_manager.get_agent_config("search")
        assert search_config["max_papers_per_search"] == 3
        
        # Test non-existent agent
        empty_config = config_manager.get_agent_config("nonexistent")
        assert empty_config == {}
    
    def test_configuration_export(self, config_manager):
        """Test configuration export functionality"""
        # Test YAML export
        yaml_export = config_manager.export_config("yaml")
        assert "api:" in yaml_export
        assert "openai:" in yaml_export
        
        # Test JSON export
        json_export = config_manager.export_config("json")
        assert '"api"' in json_export
        assert '"openai"' in json_export
        
        # Verify it's valid JSON
        import json
        parsed_json = json.loads(json_export)
        assert "api" in parsed_json
    
    def test_configuration_info(self, config_manager):
        """Test configuration metadata retrieval"""
        info = config_manager.get_config_info()
        
        assert "environment" in info
        assert "config_dir" in info
        assert "last_loaded" in info
        assert "config_files" in info
        
        assert info["environment"] == "test"
        assert isinstance(info["config_files"], list)
    
    def test_reload_configuration(self, config_manager, temp_dir):
        """Test configuration hot reloading"""
        original_model = config_manager.get("api.openai.model")
        
        # Modify the base config file
        config_path = Path(config_manager.config_dir) / "base.yaml"
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data["api"]["openai"]["model"] = "gpt-4-updated"
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Reload configuration
        config_manager.reload()
        
        # Verify the change was loaded
        assert config_manager.get("api.openai.model") == "gpt-4-updated"
        assert config_manager.get("api.openai.model") != original_model
    
    def test_missing_base_config_file(self, temp_dir):
        """Test error handling when base config file is missing"""
        empty_config_dir = Path(temp_dir) / "empty_config"
        empty_config_dir.mkdir(exist_ok=True)
        
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager(config_dir=str(empty_config_dir))
        
        assert "Base configuration file not found" in str(exc_info.value)
    
    def test_invalid_yaml_syntax(self, temp_dir):
        """Test error handling for invalid YAML syntax"""
        config_path = Path(temp_dir) / "config"
        config_path.mkdir(exist_ok=True)
        
        # Create invalid YAML file
        with open(config_path / "base.yaml", 'w') as f:
            f.write("invalid: yaml: syntax: [unclosed")
        
        with pytest.raises(ConfigValidationError):
            ConfigManager(config_dir=str(config_path))
    
    def test_deep_merge_functionality(self, temp_dir):
        """Test deep merging of configuration dictionaries"""
        config_path = Path(temp_dir) / "config"
        config_path.mkdir(exist_ok=True)
        
        # Create base config
        base_config = {
            "api": {
                "openai": {"model": "gpt-4", "max_tokens": 4000},
                "other": {"setting": "base_value"}
            },
            "database": {"mongodb": {"uri": "base_uri"}}
        }
        with open(config_path / "base.yaml", 'w') as f:
            yaml.dump(base_config, f)
        
        # Create override config
        override_config = {
            "api": {
                "openai": {"max_tokens": 2000},  # Override specific value
                "new_service": {"endpoint": "new_endpoint"}  # Add new service
            }
        }
        with open(config_path / "test.yaml", 'w') as f:
            yaml.dump(override_config, f)
        
        config = ConfigManager(config_dir=str(config_path), environment="test")
        
        # Verify deep merge results
        assert config.get("api.openai.model") == "gpt-4"  # From base
        assert config.get("api.openai.max_tokens") == 2000  # Overridden
        assert config.get("api.other.setting") == "base_value"  # From base, not overridden
        assert config.get("api.new_service.endpoint") == "new_endpoint"  # Added
        assert config.get("database.mongodb.uri") == "base_uri"  # From base, not touched


class TestGlobalConfigManager:
    """Test global configuration manager functions"""
    
    def test_get_global_config_singleton(self, config_dir):
        """Test global config manager singleton behavior"""
        # Reset global state
        import src.utils.config_manager as config_module
        config_module._config_manager = None
        
        # Get config multiple times
        config1 = get_config(config_dir, "test")
        config2 = get_config(config_dir, "test")
        
        # Should be the same instance
        assert config1 is config2
    
    def test_reload_global_config(self, config_dir):
        """Test reloading global configuration"""
        # Initialize global config
        config = get_config(config_dir, "test")
        original_model = config.get("api.openai.model")
        
        # Modify config file
        config_path = Path(config_dir) / "base.yaml"
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data["api"]["openai"]["model"] = "gpt-4-global-test"
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Reload global config
        reload_config()
        
        # Get config again and verify change
        updated_config = get_config()
        assert updated_config.get("api.openai.model") == "gpt-4-global-test"


class TestConfigManagerEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_configuration_file(self, temp_dir):
        """Test handling of empty configuration file"""
        config_path = Path(temp_dir) / "config"
        config_path.mkdir(exist_ok=True)
        
        # Create empty YAML file
        with open(config_path / "base.yaml", 'w') as f:
            f.write("")
        
        with pytest.raises(ConfigValidationError):
            ConfigManager(config_dir=str(config_path))
    
    def test_none_configuration_values(self, temp_dir):
        """Test handling of None values in configuration"""
        config_path = Path(temp_dir) / "config"
        config_path.mkdir(exist_ok=True)
        
        # Create config with None values
        config_with_none = {
            "api": {
                "openai": {
                    "model": None,  # Required field with None value
                    "max_tokens": 1000
                }
            },
            "database": {"mongodb": {"database_name": "test"}},
            "logging": {"level": "INFO"},
            "agents": {"search": {"max_papers_per_search": 5}}
        }
        with open(config_path / "base.yaml", 'w') as f:
            yaml.dump(config_with_none, f)
        
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager(config_dir=str(config_path))
        
        assert "missing or None" in str(exc_info.value)
    
    def test_complex_environment_variable_substitution(self, temp_dir):
        """Test complex environment variable substitution scenarios"""
        config_path = Path(temp_dir) / "config"
        config_path.mkdir(exist_ok=True)
        
        # Create config with complex env var patterns
        complex_config = {
            "api": {
                "openai": {
                    "base_url": "${API_BASE_URL:https://api.openai.com}",
                    "timeout": "${API_TIMEOUT:30}",
                    "retries": "${API_RETRIES:3}"
                }
            },
            "database": {
                "mongodb": {
                    "uri": "mongodb://${DB_USER:user}:${DB_PASS:pass}@${DB_HOST:localhost}:${DB_PORT:27017}/${DB_NAME:test}"
                }
            }
        }
        with open(config_path / "base.yaml", 'w') as f:
            yaml.dump(complex_config, f)
        
        # Test with some env vars set, others using defaults
        with patch.dict(os.environ, {
            'API_TIMEOUT': '60',
            'DB_HOST': 'production.db.com',
            'DB_NAME': 'production_db'
        }):
            config = ConfigManager(config_dir=str(config_path))
            
            assert config.get("api.openai.base_url") == "https://api.openai.com"  # Default
            assert config.get("api.openai.timeout") == "60"  # From env
            assert config.get("api.openai.retries") == "3"  # Default
            
            expected_uri = "mongodb://user:pass@production.db.com:27017/production_db"
            assert config.get("database.mongodb.uri") == expected_uri
    
    def test_nested_environment_variable_substitution(self, temp_dir):
        """Test environment variable substitution in nested structures"""
        config_path = Path(temp_dir) / "config"
        config_path.mkdir(exist_ok=True)
        
        # Create config with nested env vars
        nested_config = {
            "services": {
                "list": [
                    "${SERVICE_1:service1}",
                    "${SERVICE_2:service2}"
                ],
                "mapping": {
                    "primary": "${PRIMARY_SERVICE:main}",
                    "secondary": "${SECONDARY_SERVICE:backup}"
                }
            }
        }
        with open(config_path / "base.yaml", 'w') as f:
            yaml.dump(nested_config, f)
        
        with patch.dict(os.environ, {
            'SERVICE_1': 'custom_service_1',
            'PRIMARY_SERVICE': 'custom_primary'
        }):
            # This config doesn't have required fields, so we need to mock validation
            with patch.object(ConfigManager, '_validate_configuration'):
                config = ConfigManager(config_dir=str(config_path))
                
                assert config.get("services.list.0") == "custom_service_1"
                assert config.get("services.list.1") == "service2"  # Default
                assert config.get("services.mapping.primary") == "custom_primary"
                assert config.get("services.mapping.secondary") == "backup"  # Default


@pytest.mark.performance
class TestConfigManagerPerformance:
    """Performance tests for configuration manager"""
    
    def test_config_loading_performance(self, config_dir, performance_timer):
        """Test configuration loading performance"""
        performance_timer.start()
        config = ConfigManager(config_dir=config_dir, environment="test")
        duration = performance_timer.stop()
        
        # Configuration loading should be fast (< 1 second)
        assert duration < 1.0
        assert config.config is not None
    
    def test_config_access_performance(self, config_manager, performance_timer):
        """Test configuration value access performance"""
        # Test many rapid accesses
        performance_timer.start()
        for _ in range(1000):
            config_manager.get("api.openai.model")
            config_manager.get("agents.search.max_papers_per_search")
        duration = performance_timer.stop()
        
        # Should handle 1000 accesses quickly (< 0.1 seconds)
        assert duration < 0.1 
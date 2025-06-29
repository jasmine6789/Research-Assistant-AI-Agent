"""
Pytest Configuration and Fixtures for Research Assistant AI Agent Tests

Provides:
- Common fixtures for testing
- Test data generation
- Mock configurations
- Database setup/teardown
- API mocking utilities
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch
import yaml
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config_manager import ConfigManager
from src.utils.logging_config import setup_logging
from src.utils.error_handling import ErrorHandler

# Test Configuration
TEST_CONFIG = {
    "api": {
        "openai": {
            "model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.7
        }
    },
    "database": {
        "mongodb": {
            "database_name": "test_research_assistant",
            "collection_name": "test_arxiv_papers"
        }
    },
    "agents": {
        "search": {
            "max_papers_per_search": 3
        },
        "hypothesis": {
            "max_iterations": 2
        },
        "code": {
            "enable_huggingface": False,
            "enable_execution": False
        }
    },
    "logging": {
        "level": "DEBUG",
        "handlers": [{"type": "console", "level": "DEBUG"}]
    }
}

# Sample test data
SAMPLE_PAPERS = [
    {
        "_id": "test_paper_1",
        "title": "Advances in Machine Learning for Healthcare",
        "abstract": "This paper presents novel machine learning approaches for healthcare applications, focusing on early disease detection and personalized treatment recommendations.",
        "authors": ["Dr. Jane Smith", "Dr. John Doe"],
        "published": "2024-01-15",
        "categories": ["cs.LG", "cs.AI"],
        "arxiv_id": "2401.12345",
        "embedding": [0.1, 0.2, 0.3] * 256  # Mock embedding vector
    },
    {
        "_id": "test_paper_2", 
        "title": "Quantum Machine Learning for Drug Discovery",
        "abstract": "We explore quantum computing applications in drug discovery, demonstrating significant improvements in molecular property prediction accuracy.",
        "authors": ["Dr. Alice Johnson", "Dr. Bob Wilson"],
        "published": "2024-02-20",
        "categories": ["cs.LG", "quant-ph"],
        "arxiv_id": "2402.67890",
        "embedding": [0.2, 0.3, 0.4] * 256
    },
    {
        "_id": "test_paper_3",
        "title": "Federated Learning for Privacy-Preserving AI",
        "abstract": "This work presents a comprehensive framework for federated learning that maintains privacy while enabling collaborative model training.",
        "authors": ["Dr. Carol Davis", "Dr. David Brown"],
        "published": "2024-03-10",
        "categories": ["cs.LG", "cs.CR"],
        "arxiv_id": "2403.11111",
        "embedding": [0.3, 0.4, 0.5] * 256
    }
]

SAMPLE_HYPOTHESIS = """
Hypothesis: Quantum machine learning algorithms can achieve superior performance in drug discovery 
tasks compared to classical machine learning approaches, particularly in molecular property prediction 
and drug-target interaction modeling, due to their ability to capture quantum mechanical effects 
and handle high-dimensional feature spaces more efficiently.
"""

SAMPLE_CODE = '''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def drug_discovery_ml_model():
    """
    Machine learning model for drug discovery property prediction
    """
    # Generate synthetic molecular data
    np.random.seed(42)
    n_molecules = 1000
    n_features = 50
    
    # Mock molecular descriptors
    X = np.random.randn(n_molecules, n_features)
    # Mock target property (e.g., binding affinity)
    y = np.random.randn(n_molecules)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "model": model,
        "mse": mse,
        "r2_score": r2,
        "predictions": y_pred
    }

if __name__ == "__main__":
    results = drug_discovery_ml_model()
    print(f"MSE: {results['mse']:.4f}")
    print(f"R² Score: {results['r2_score']:.4f}")
'''

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration"""
    return TEST_CONFIG.copy()

@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files"""
    temp_path = tempfile.mkdtemp(prefix="research_agent_test_")
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def config_dir(temp_dir):
    """Create test configuration directory with config files"""
    config_path = Path(temp_dir) / "config"
    config_path.mkdir(exist_ok=True)
    
    # Create base config file
    base_config_path = config_path / "base.yaml"
    with open(base_config_path, 'w') as f:
        yaml.dump(TEST_CONFIG, f)
    
    # Create test config file
    test_config_path = config_path / "test.yaml"
    test_overrides = {
        "development": {
            "debug_mode": True,
            "mock_apis": True
        }
    }
    with open(test_config_path, 'w') as f:
        yaml.dump(test_overrides, f)
    
    return str(config_path)

@pytest.fixture
def config_manager(config_dir):
    """Provide configured ConfigManager for tests"""
    return ConfigManager(config_dir=config_dir, environment="test")

@pytest.fixture
def test_logger():
    """Provide test logger"""
    return setup_logging("test_logger", "DEBUG", enable_structured_logging=False)

@pytest.fixture
def error_handler(test_logger):
    """Provide error handler for tests"""
    return ErrorHandler(test_logger)

@pytest.fixture
def sample_papers():
    """Provide sample paper data"""
    return SAMPLE_PAPERS.copy()

@pytest.fixture
def sample_hypothesis():
    """Provide sample hypothesis"""
    return SAMPLE_HYPOTHESIS

@pytest.fixture
def sample_code():
    """Provide sample generated code"""
    return SAMPLE_CODE

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = Mock()
    
    # Mock chat completion response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Mock AI response for testing"
    
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client

@pytest.fixture
def mock_mongodb_client():
    """Mock MongoDB client for testing"""
    mock_client = Mock()
    mock_db = Mock()
    mock_collection = Mock()
    
    # Setup mock collection with sample data
    mock_collection.find.return_value = SAMPLE_PAPERS
    mock_collection.aggregate.return_value = SAMPLE_PAPERS[:2]  # Vector search results
    mock_collection.insert_one.return_value = Mock(inserted_id="test_id")
    mock_collection.update_one.return_value = Mock(modified_count=1)
    mock_collection.delete_one.return_value = Mock(deleted_count=1)
    
    mock_db.__getitem__.return_value = mock_collection
    mock_client.__getitem__.return_value = mock_db
    
    return mock_client

@pytest.fixture
def mock_huggingface_model():
    """Mock HuggingFace model for testing"""
    mock_model = Mock()
    
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [101, 102, 103]  # Mock token IDs
    mock_tokenizer.decode.return_value = "Mock decoded text"
    
    # Mock model outputs
    mock_outputs = Mock()
    mock_outputs.last_hidden_state = Mock()
    mock_outputs.pooler_output = np.random.randn(1, 768)  # Mock embeddings
    
    mock_model.return_value = mock_outputs
    
    return {"model": mock_model, "tokenizer": mock_tokenizer}

@pytest.fixture
def mock_file_system(temp_dir):
    """Mock file system operations"""
    output_dir = Path(temp_dir) / "generated_papers"
    output_dir.mkdir(exist_ok=True)
    
    return {
        "output_dir": str(output_dir),
        "temp_dir": temp_dir
    }

@pytest.fixture
def sample_visualization_data():
    """Provide sample data for visualization testing"""
    return {
        "performance_data": {
            "classical_ml": [0.85, 0.87, 0.89, 0.91, 0.88],
            "quantum_ml": [0.89, 0.92, 0.94, 0.96, 0.95],
            "epochs": [1, 2, 3, 4, 5]
        },
        "accuracy_trends": {
            "training_accuracy": [0.7, 0.8, 0.85, 0.9, 0.92],
            "validation_accuracy": [0.68, 0.75, 0.82, 0.87, 0.89],
            "epochs": [1, 2, 3, 4, 5]
        },
        "feature_importance": {
            "features": ["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"],
            "importance": [0.25, 0.2, 0.18, 0.22, 0.15]
        }
    }

@pytest.fixture
def mock_environment_variables(monkeypatch):
    """Mock environment variables for testing"""
    test_env_vars = {
        "CHATGPT_API_KEY": "test_api_key_12345",
        "MONGO_URI": "mongodb://test:test@localhost:27017/test_db",
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "ENVIRONMENT": "test"
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)
    
    return test_env_vars

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    # Reset config manager
    import src.utils.config_manager as config_module
    config_module._config_manager = None
    
    # Reset logging config
    import src.utils.logging_config as logging_module
    logging_module._logging_config = None
    
    # Reset error handler
    import src.utils.error_handling as error_module
    error_module._error_handler = None

@pytest.fixture
def performance_timer():
    """Fixture for measuring test performance"""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = datetime.now()
        
        def stop(self):
            self.end_time = datetime.now()
            return (self.end_time - self.start_time).total_seconds()
        
        def get_duration(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return None
    
    return PerformanceTimer()

# Pytest markers for test categorization
pytest_markers = [
    "unit: Unit tests for individual components",
    "integration: Integration tests for component interactions",
    "e2e: End-to-end tests for complete workflows",
    "performance: Performance and load tests",
    "security: Security-related tests",
    "slow: Tests that take longer to run",
    "requires_api: Tests that require external API access",
    "requires_db: Tests that require database access"
]

def pytest_configure(config):
    """Configure pytest with custom markers"""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)

@pytest.fixture
def assert_performance(performance_timer):
    """Assertion helper for performance requirements"""
    def _assert_performance(max_duration_seconds: float):
        def decorator(test_func):
            def wrapper(*args, **kwargs):
                performance_timer.start()
                result = test_func(*args, **kwargs)
                duration = performance_timer.stop()
                assert duration <= max_duration_seconds, \
                    f"Test took {duration:.2f}s, expected <= {max_duration_seconds}s"
                return result
            return wrapper
        return decorator
    return _assert_performance

@pytest.fixture
def mock_research_pipeline():
    """Mock complete research pipeline for testing"""
    mock_pipeline = Mock()
    
    # Mock pipeline methods
    mock_pipeline.search_papers.return_value = SAMPLE_PAPERS[:2]
    mock_pipeline.generate_hypothesis.return_value = SAMPLE_HYPOTHESIS
    mock_pipeline.generate_code.return_value = SAMPLE_CODE
    mock_pipeline.create_visualizations.return_value = {"chart_path": "/mock/chart.png"}
    mock_pipeline.generate_paper.return_value = "Mock research paper content"
    
    # Mock pipeline state
    mock_pipeline.current_state = "initialized"
    mock_pipeline.results = {}
    
    return mock_pipeline

# Helper functions for tests
def create_test_paper(title: str = "Test Paper", 
                     abstract: str = "Test abstract",
                     authors: List[str] = None) -> Dict[str, Any]:
    """Create test paper data"""
    return {
        "_id": f"test_{title.lower().replace(' ', '_')}",
        "title": title,
        "abstract": abstract,
        "authors": authors or ["Test Author"],
        "published": datetime.now().isoformat()[:10],
        "categories": ["cs.LG"],
        "arxiv_id": "test.12345",
        "embedding": [0.1] * 768
    }

def create_mock_response(content: str = "Mock response") -> Mock:
    """Create mock API response"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = content
    return mock_response

# Test data generators
class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def generate_papers(count: int = 5) -> List[Dict[str, Any]]:
        """Generate multiple test papers"""
        papers = []
        for i in range(count):
            paper = create_test_paper(
                title=f"Test Paper {i+1}",
                abstract=f"Abstract for test paper {i+1} focusing on machine learning research.",
                authors=[f"Author {i+1}", f"Co-Author {i+1}"]
            )
            papers.append(paper)
        return papers
    
    @staticmethod
    def generate_hypothesis(topic: str = "machine learning") -> str:
        """Generate test hypothesis"""
        return f"""
        Hypothesis: Advanced {topic} techniques can significantly improve performance 
        in real-world applications by addressing key challenges in data quality, 
        model interpretability, and computational efficiency.
        """
    
    @staticmethod
    def generate_code_snippet(language: str = "python") -> str:
        """Generate test code snippet"""
        if language == "python":
            return '''
import numpy as np
import matplotlib.pyplot as plt

def test_function():
    """Test function for validation"""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title("Test Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    
    return {"status": "success", "data": y}

if __name__ == "__main__":
    result = test_function()
    print(f"Result: {result['status']}")
'''
        return f"// Test code for {language}"

@pytest.fixture
def test_data_generator():
    """Provide test data generator"""
    return TestDataGenerator()

# Performance benchmarks
class PerformanceBenchmarks:
    """Define performance benchmarks for testing"""
    
    SEARCH_PAPERS_MAX_TIME = 5.0  # seconds
    GENERATE_HYPOTHESIS_MAX_TIME = 30.0
    GENERATE_CODE_MAX_TIME = 45.0
    CREATE_VISUALIZATION_MAX_TIME = 10.0
    GENERATE_PAPER_MAX_TIME = 60.0
    FULL_PIPELINE_MAX_TIME = 300.0  # 5 minutes

@pytest.fixture
def performance_benchmarks():
    """Provide performance benchmarks"""
    return PerformanceBenchmarks() 
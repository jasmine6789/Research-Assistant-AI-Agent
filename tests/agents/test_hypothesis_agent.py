import pytest
from unittest.mock import Mock, patch
from src.agents.hypothesis_agent import HypothesisAgent

# Mock paper data
MOCK_PAPERS = [
    {
        "title": "Example Paper 1",
        "abstract": "This is an example abstract",
        "authors": ["Author 1", "Author 2"],
        "year": 2023,
        "arxiv_id": "2301.12345"
    }
]

# Mock OpenAI response
MOCK_HYPOTHESIS = """Main Hypothesis: Example hypothesis statement
Key Assumptions:
- Assumption 1
- Assumption 2
Proposed Testing Methodology:
- Method 1
- Method 2
Expected Outcomes:
- Outcome 1
- Outcome 2"""

@pytest.fixture
def mock_openai():
    """Fixture to create a mock OpenAI client using patch(new=...) on the agent's import path"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = MOCK_HYPOTHESIS
    mock_response.id = "mock_generation_id"
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    with patch('src.agents.hypothesis_agent.OpenAI', new=lambda *args, **kwargs: mock_client):
        yield mock_client

@pytest.fixture
def hypothesis_agent(mock_openai):
    """Fixture to create a HypothesisAgent instance with mocked dependencies"""
    return HypothesisAgent("mock_api_key")

def test_hypothesis_agent_initialization(hypothesis_agent):
    """Test HypothesisAgent initialization"""
    assert hypothesis_agent is not None
    assert hypothesis_agent.client is not None
    assert hypothesis_agent.system_prompt is not None

def test_generate_hypothesis(hypothesis_agent):
    """Test hypothesis generation functionality"""
    result = hypothesis_agent.generate_hypothesis(MOCK_PAPERS)
    
    assert isinstance(result, dict)
    assert "hypothesis" in result
    assert "papers" in result
    assert "generation_id" in result
    assert result["hypothesis"] == MOCK_HYPOTHESIS
    assert result["papers"] == MOCK_PAPERS
    assert result["generation_id"] == "mock_generation_id"

def test_refine_hypothesis(hypothesis_agent):
    """Test hypothesis refinement functionality"""
    current_hypothesis = {
        "hypothesis": MOCK_HYPOTHESIS,
        "papers": MOCK_PAPERS,
        "generation_id": "original_id"
    }
    
    feedback = "Make the hypothesis more specific"
    result = hypothesis_agent.refine_hypothesis(current_hypothesis, feedback)
    
    assert isinstance(result, dict)
    assert "hypothesis" in result
    assert "papers" in result
    assert "generation_id" in result
    assert "previous_generation_id" in result
    assert result["previous_generation_id"] == "original_id"

def test_regenerate_hypothesis(hypothesis_agent):
    """Test hypothesis regeneration functionality"""
    current_hypothesis = {
        "hypothesis": MOCK_HYPOTHESIS,
        "papers": MOCK_PAPERS,
        "generation_id": "original_id"
    }
    
    result = hypothesis_agent.refine_hypothesis(current_hypothesis, "", regenerate=True)
    
    assert isinstance(result, dict)
    assert "hypothesis" in result
    assert "papers" in result
    assert "generation_id" in result
    assert result["generation_id"] == "mock_generation_id" 
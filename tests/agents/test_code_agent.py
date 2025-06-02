import pytest
from unittest.mock import MagicMock, patch
from src.agents.code_agent import CodeAgent

@pytest.fixture
def code_agent():
    return CodeAgent(openai_api_key="dummy_key")

def test_generate_code(code_agent):
    with patch('openai.ChatCompletion.create') as mock_create:
        mock_create.return_value.choices = [MagicMock(message=MagicMock(content="print('test')"))]
        code = code_agent.generate_code("test hypothesis", language="python")
        assert code == "print('test')"

def test_generate_code_with_template(code_agent):
    with patch('openai.ChatCompletion.create') as mock_create:
        mock_create.return_value.choices = [MagicMock(message=MagicMock(content="print('test')"))]
        code = code_agent.generate_code("test hypothesis", language="python", template="def main(): pass")
        assert code == "print('test')"

def test_run_pylint(code_agent):
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "No issues found"
        result = code_agent.run_pylint("print('test')")
        assert result["passed"] is True
        assert result["output"] == "No issues found"

def test_execute_code(code_agent):
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "test output"
        mock_run.return_value.stderr = ""
        result = code_agent.execute_code("print('test')")
        assert result["success"] is True
        assert result["output"] == "test output"
        assert "execution_time" in result
        assert "memory_usage" in result

def test_wrap_in_pytest(code_agent):
    code = "print('test')"
    wrapped_code = code_agent.wrap_in_pytest(code)
    assert "import pytest" in wrapped_code
    assert "def test_hypothesis():" in wrapped_code

def test_add_feedback(code_agent):
    code_agent.add_feedback(5, "Great code!")
    feedback = code_agent.get_feedback()
    assert len(feedback) == 1
    assert feedback[0]["rating"] == 5
    assert feedback[0]["suggestion"] == "Great code!"

def test_add_code_template(code_agent):
    code_agent.add_code_template("test_template", "def main(): pass")
    template = code_agent.get_code_template("test_template")
    assert template == "def main(): pass" 
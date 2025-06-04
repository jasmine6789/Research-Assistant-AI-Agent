import os
import subprocess
import tempfile
from openai import OpenAI
import logging
import time
import psutil
from typing import Dict, Any, List, Optional
from src.agents.note_taker import NoteTaker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeAgent:
    def __init__(self, openai_api_key: str, note_taker: NoteTaker):
        self.client = OpenAI(api_key=openai_api_key)
        self.note_taker = note_taker
        self.feedback = []
        self.code_templates = {}

    def generate_code(self, hypothesis: str, language: str = "python", template: Optional[str] = None) -> str:
        try:
            system_content = f"You are a code generator that produces high-quality, research-useful {language} code to test a given hypothesis. Follow best practices, include docstrings, and use type hints."
            if template:
                system_content += f" Use the following template as a base: {template}"
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"Generate {language} code to test the following hypothesis: {hypothesis}"}
                ]
            )
            code = response.choices[0].message.content
            self.note_taker.log("code_generation", {"hypothesis": hypothesis, "language": language, "template": template})
            return code
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise

    def run_pylint(self, code: str) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
            temp.write(code.encode())
            temp_path = temp.name
        try:
            result = subprocess.run(['pylint', temp_path], capture_output=True, text=True)
            self.note_taker.log("pylint_check", {"passed": result.returncode == 0, "output": result.stdout})
            return {"passed": result.returncode == 0, "output": result.stdout}
        except Exception as e:
            logger.error(f"Error running pylint: {e}")
            raise
        finally:
            os.unlink(temp_path)

    def execute_code(self, code: str) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
            temp.write(code.encode())
            temp_path = temp.name
        try:
            start_time = time.time()
            process = subprocess.run(['python', temp_path], capture_output=True, text=True)
            end_time = time.time()
            execution_time = end_time - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            result = {
                "success": process.returncode == 0,
                "output": process.stdout,
                "error": process.stderr,
                "execution_time": execution_time,
                "memory_usage": memory_usage
            }
            self.note_taker.log("code_execution", result)
            return result
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            raise
        finally:
            os.unlink(temp_path)

    def wrap_in_pytest(self, code: str) -> str:
        return f"""
import pytest

{code}

def test_hypothesis():
    # Add assertions here based on the generated code
    pass
"""

    def add_feedback(self, rating: int, suggestion: Optional[str] = None) -> None:
        self.feedback.append({"rating": rating, "suggestion": suggestion})
        self.note_taker.log_feedback(f"Code feedback: rating={rating}, suggestion={suggestion}")
        logger.info(f"Feedback added: rating={rating}, suggestion={suggestion}")

    def get_feedback(self) -> List[Dict[str, Any]]:
        return self.feedback

    def add_code_template(self, name: str, template: str) -> None:
        self.code_templates[name] = template
        self.note_taker.log("code_template_added", {"name": name})
        logger.info(f"Code template added: {name}")

    def get_code_template(self, name: str) -> Optional[str]:
        return self.code_templates.get(name)

# Example usage (to be removed in production)
if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
    MONGO_URI = os.getenv("MONGO_URI")
    note_taker = NoteTaker(MONGO_URI)
    agent = CodeAgent(OPENAI_API_KEY, note_taker)
    code = agent.generate_code("Test hypothesis")
    print("Generated code:", code)
    print("Pylint result:", agent.run_pylint(code))
    print("Execution result:", agent.execute_code(code)) 
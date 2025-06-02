import os
import subprocess
import tempfile
import openai
import logging
import time
import psutil
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeAgent:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        self.feedback = []
        self.code_templates = {}

    def generate_code(self, hypothesis: str, language: str = "python", template: Optional[str] = None) -> str:
        try:
            system_content = f"You are a code generator that produces high-quality, research-useful {language} code to test a given hypothesis. Follow best practices, include docstrings, and use type hints."
            if template:
                system_content += f" Use the following template as a base: {template}"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"Generate {language} code to test the following hypothesis: {hypothesis}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise

    def run_pylint(self, code: str) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
            temp.write(code.encode())
            temp_path = temp.name
        try:
            result = subprocess.run(['pylint', temp_path], capture_output=True, text=True)
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
            return {
                "success": process.returncode == 0,
                "output": process.stdout,
                "error": process.stderr,
                "execution_time": execution_time,
                "memory_usage": memory_usage
            }
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
        logger.info(f"Feedback added: rating={rating}, suggestion={suggestion}")

    def get_feedback(self) -> List[Dict[str, Any]]:
        return self.feedback

    def add_code_template(self, name: str, template: str) -> None:
        self.code_templates[name] = template
        logger.info(f"Code template added: {name}")

    def get_code_template(self, name: str) -> Optional[str]:
        return self.code_templates.get(name) 
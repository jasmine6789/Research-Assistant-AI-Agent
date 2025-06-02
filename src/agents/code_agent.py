import os
import subprocess
import tempfile
import openai
from typing import Dict, Any, List

class CodeAgent:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key

    def generate_code(self, hypothesis: str) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a code generator that produces Python code to test a given hypothesis."},
                {"role": "user", "content": f"Generate Python code to test the following hypothesis: {hypothesis}"}
            ]
        )
        return response.choices[0].message.content

    def run_pylint(self, code: str) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
            temp.write(code.encode())
            temp_path = temp.name
        try:
            result = subprocess.run(['pylint', temp_path], capture_output=True, text=True)
            return {"passed": result.returncode == 0, "output": result.stdout}
        finally:
            os.unlink(temp_path)

    def execute_code(self, code: str) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
            temp.write(code.encode())
            temp_path = temp.name
        try:
            result = subprocess.run(['python', temp_path], capture_output=True, text=True)
            return {"success": result.returncode == 0, "output": result.stdout, "error": result.stderr}
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
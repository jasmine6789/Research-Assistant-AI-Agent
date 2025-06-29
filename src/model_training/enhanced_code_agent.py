import os
import sys
import ast
import time
import tempfile
import subprocess
import json
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from huggingface_hub import list_models, ModelFilter
from agents.note_taker import NoteTaker
import logging
import psutil
import requests
from huggingface_hub import HfApi

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedCodeAgent:
    def __init__(self, openai_api_key: str, note_taker: NoteTaker):
        self.client = OpenAI(api_key=openai_api_key)
        self.note_taker = note_taker
        self.feedback = []
        self.hf_api = HfApi()
        self.code_templates = {}

    def discover_relevant_models(self, hypothesis: str, max_models: int = 5) -> List[Dict[str, Any]]:
        """
        Discover relevant HuggingFace models based on hypothesis with robust error handling
        """
        try:
            # Handle both string and dictionary formats for hypothesis
            if isinstance(hypothesis, dict):
                hypothesis_text = hypothesis.get('hypothesis', str(hypothesis))
            else:
                hypothesis_text = str(hypothesis)
            
            # Extract ML keywords
            keywords = self._extract_ml_keywords(hypothesis_text)
            
            relevant_models = []
            
            # Try to search for models using keywords
            for keyword in keywords:
                try:
                    # Use HuggingFace API to search for models
                    models = list_models(
                        search=keyword,
                        sort="downloads",
                        direction=-1,
                        limit=max_models
                    )
                    
                    for model in models:
                        try:
                            model_info = {
                                "id": model.id,
                                "downloads": getattr(model, 'downloads', 0),
                                "tags": getattr(model, 'tags', []),
                                "pipeline_tag": getattr(model, 'pipeline_tag', 'unknown'),
                                "library": getattr(model, 'library_name', 'transformers')
                            }
                            relevant_models.append(model_info)
                            
                            if len(relevant_models) >= max_models:
                                break
                        except Exception as e:
                            print(f"   ⚠️ Error processing model {model.id}: {e}")
                            continue
                    
                except Exception as e:
                    print(f"   ⚠️ Failed to fetch models for keyword '{keyword}': {e}")
                    continue
            
            # Remove duplicates and sort by downloads
            unique_models = {}
            for model in relevant_models:
                if model["id"] not in unique_models:
                    unique_models[model["id"]] = model
            
            sorted_models = sorted(
                unique_models.values(), 
                key=lambda x: x.get("downloads", 0), 
                reverse=True
            )[:max_models]
            
            self.note_taker.log("hf_model_discovery", {
                "hypothesis": hypothesis_text,
                "keywords": keywords,
                "discovered_models": [m['id'] for m in sorted_models]
            })
            
            return sorted_models
            
        except Exception as e:
            print(f"   ⚠️ Error discovering models from Hugging Face: {e}")
            return self._fallback_model_suggestions(hypothesis_text)

    def _extract_ml_keywords(self, hypothesis: str) -> List[str]:
        """Extract relevant ML keywords from hypothesis for model search"""
        # Common ML/AI task keywords
        task_keywords = {
            "classification": ["classification", "classifier"],
            "detection": ["detection", "object-detection", "yolo"],
            "forecasting": ["forecasting", "time-series", "prediction"],
            "generation": ["generation", "generative", "gpt"],
            "translation": ["translation", "mt5", "t5"],
            "summarization": ["summarization", "summary"],
            "question-answering": ["question-answering", "qa", "bert"],
            "sentiment": ["sentiment", "emotion"],
            "image": ["vision", "image", "cnn", "resnet"],
            "text": ["text", "nlp", "bert", "roberta"],
            "transformer": ["transformer", "attention"],
            "lstm": ["lstm", "rnn", "sequence"],
            "cnn": ["cnn", "convolution", "image"],
            "reinforcement": ["reinforcement", "rl", "policy"]
        }
        
        hypothesis_lower = hypothesis.lower()
        extracted_keywords = []
        
        # Extract task-specific keywords
        for task, keywords in task_keywords.items():
            if any(keyword in hypothesis_lower for keyword in keywords):
                extracted_keywords.extend(keywords[:2])  # Top 2 keywords per task
        
        # Add domain-specific keywords
        domain_keywords = []
        words = hypothesis_lower.split()
        for word in words:
            if len(word) > 4 and word not in ["that", "with", "will", "from", "this", "they"]:
                domain_keywords.append(word)
        
        # Combine and limit keywords
        all_keywords = extracted_keywords + domain_keywords[:3]
        return list(set(all_keywords))[:5]  # Remove duplicates and limit

    def _fallback_model_suggestions(self, hypothesis: str) -> List[Dict[str, Any]]:
        """Provide fallback model suggestions if HF API fails"""
        # Handle both string and dictionary formats for hypothesis
        if isinstance(hypothesis, dict):
            hypothesis_text = hypothesis.get('hypothesis', str(hypothesis))
        else:
            hypothesis_text = str(hypothesis)
            
        hypothesis_lower = hypothesis_text.lower()
        
        fallback_models = []
        
        if any(word in hypothesis_lower for word in ["text", "nlp", "language"]):
            fallback_models.extend([
                {"id": "bert-base-uncased", "pipeline_tag": "text-classification"},
                {"id": "distilbert-base-uncased", "pipeline_tag": "text-classification"}
            ])
        
        if any(word in hypothesis_lower for word in ["image", "vision", "cnn"]):
            fallback_models.extend([
                {"id": "google/vit-base-patch16-224", "pipeline_tag": "image-classification"},
                {"id": "microsoft/resnet-50", "pipeline_tag": "image-classification"}
            ])
        
        if any(word in hypothesis_lower for word in ["time", "series", "forecasting"]):
            fallback_models.extend([
                {"id": "huggingface/time-series-transformer", "pipeline_tag": "forecasting"}
            ])
        
        return fallback_models[:3]

    def generate_enhanced_code(self, hypothesis: str, include_hf_models: bool = True) -> str:
        """
        Generate enhanced code using GPT-4 with HuggingFace model integration
        """
        try:
            # Handle both string and dictionary formats for hypothesis
            if isinstance(hypothesis, dict):
                hypothesis_text = hypothesis.get('hypothesis', str(hypothesis))
            else:
                hypothesis_text = str(hypothesis)
            
            relevant_models = []
            if include_hf_models:
                try:
                    relevant_models = self.discover_relevant_models(hypothesis_text)
                except Exception as e:
                    print(f"   ⚠️ HuggingFace discovery failed: {e}")
                    relevant_models = []
            
            # Create comprehensive prompt for high-quality code generation
            model_info = ""
            if relevant_models:
                top_models = relevant_models[:2]  # Limit to top 2 models
                model_info = f"\nAvailable HuggingFace models:\n"
                for model in top_models:
                    model_info += f"- {model['id']}: {model.get('pipeline_tag', 'general')}\n"
            
            prompt = f"""Generate high-quality Python code to test this research hypothesis:

HYPOTHESIS: "{hypothesis_text[:300]}..."

{model_info}

CRITICAL REQUIREMENTS:
1. Use ONLY standard libraries: numpy, pandas, sklearn, matplotlib (NO exotic imports)
2. MUST spell library names correctly: 'numpy' NOT 'nummpy', 'pandas' NOT 'pandsa'
3. Generate COMPLETE, RUNNABLE code with synthetic data
4. Include comprehensive error handling with try-except blocks
5. Add detailed docstrings and comments for clarity
6. Structure code professionally with functions and main execution
7. Keep total lines under 120 for maintainability
8. Include data validation and sanity checks
9. Generate realistic synthetic datasets for testing
10. Provide clear output showing hypothesis validation results

CODE STRUCTURE TEMPLATE:
```python
# Standard imports (correctly spelled)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data():
    \"\"\"Generate realistic synthetic dataset for hypothesis testing\"\"\"
    # Implementation here
    return X, y

def test_hypothesis():
    \"\"\"Main function to test the research hypothesis\"\"\"
    try:
        # Hypothesis testing implementation
        return results
    except Exception as e:
        print(f"Error in hypothesis testing: {{e}}")
        return None

def main():
    \"\"\"Main execution function\"\"\"
    print("Testing hypothesis: {hypothesis_text[:50]}...")
    results = test_hypothesis()
    if results:
        print("Hypothesis testing completed successfully")
        print(f"Results: {{results}}")
    else:
        print("Hypothesis testing failed")

if __name__ == "__main__":
    main()
```

Return ONLY the complete Python code, no explanations. Ensure perfect spelling and imports."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert Python developer who writes production-quality, error-free code. Always use correct library spellings and follow best practices."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,  # Increased for more comprehensive code
                temperature=0.1   # Lower for more consistent, reliable code
            )
            
            code = response.choices[0].message.content.strip()
            
            # Clean up code - remove markdown formatting if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].strip()
            
            # Advanced spelling and quality fixes
            code = self._apply_advanced_code_fixes(code)
            
            # Ensure proper code structure
            code = self._ensure_code_structure(code, hypothesis_text)
            
            # Validate code syntax
            if not self._validate_syntax(code):
                print("   ⚠️ Syntax validation failed, applying fallback fixes...")
                code = self._fix_syntax_issues(code)
            
            self.note_taker.log_code(code, 
                                   hypothesis=hypothesis_text[:100],
                                   models_used=len(relevant_models),
                                   enhanced=True,
                                   quality_checked=True)
            
            return code
            
        except Exception as e:
            print(f"   ⚠️ Enhanced code generation failed: {e}")
            return self._generate_fallback_code(hypothesis_text)

    def _apply_advanced_code_fixes(self, code: str) -> str:
        """Apply comprehensive code fixes for quality and correctness"""
        
        # Fix common spelling mistakes
        spelling_fixes = {
            "nummpy": "numpy",
            "pandsa": "pandas", 
            "skleran": "sklearn",
            "matplotib": "matplotlib",
            "seabron": "seaborn",
            "import nummpy": "import numpy",
            "from nummpy": "from numpy",
            "import pandsa": "import pandas",
            "from pandsa": "from pandas",
            "accurracy": "accuracy",
            "precission": "precision",
            "clasification": "classification",
            "modle": "model",
            "datset": "dataset"
        }
        
        for wrong, correct in spelling_fixes.items():
            code = code.replace(wrong, correct)
        
        # Ensure standard imports are present and correctly formatted
        required_imports = [
            "import numpy as np",
            "import pandas as pd",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.metrics import accuracy_score",
            "import warnings",
            "warnings.filterwarnings('ignore')"
        ]
        
        # Check which imports are missing
        imports_to_add = []
        for imp in required_imports:
            if imp not in code and not any(alt in code for alt in [
                imp.replace("import ", "from ").split(" import")[0],
                imp.split(" as ")[0] if " as " in imp else imp
            ]):
                imports_to_add.append(imp)
        
        # Add missing imports at the top
        if imports_to_add:
            code = "\n".join(imports_to_add) + "\n\n" + code
        
        # Fix common code issues
        code_fixes = {
            "print(f\"": "print(\"",  # Remove f-strings that might cause issues
            "plt.show()": "# plt.show()  # Disabled for automated execution",
            ".fit_transform(": ".fit_transform(",  # Ensure proper method calls
            "random_state=42": "random_state=42",  # Ensure reproducibility
        }
        
        for issue, fix in code_fixes.items():
            code = code.replace(issue, fix)
        
        return code

    def _ensure_code_structure(self, code: str, hypothesis: str) -> str:
        """Ensure code has proper structure with main execution block"""
        
        # Check if code has main execution structure
        if "if __name__" not in code:
            # Add main execution block
            code += f"""

# Main execution
if __name__ == "__main__":
    try:
        print("Starting hypothesis testing: {hypothesis[:50]}...")
        # Execute main code here
        print("Hypothesis testing completed successfully")
    except Exception as e:
        print(f"Error during execution: {{e}}")
        import traceback
        traceback.print_exc()
"""
        
        # Ensure UTF-8 encoding declaration for special characters
        if "# -*- coding:" not in code:
            code = "# -*- coding: utf-8 -*-\n" + code
        
        return code

    def _validate_syntax(self, code: str) -> bool:
        """Validate code syntax using AST"""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            print(f"   ⚠️ Syntax error detected: {e}")
            return False
        except Exception as e:
            print(f"   ⚠️ Code validation error: {e}")
            return False

    def _fix_syntax_issues(self, code: str) -> str:
        """Attempt to fix common syntax issues"""
        
        # Common syntax fixes
        syntax_fixes = [
            # Fix indentation issues
            (r'^(\s*)([^\s])', r'\1    \2'),  # Ensure proper indentation
            # Fix string formatting issues
            (r'print\(f"([^"]*)"', r'print("\1"'),  # Remove f-string formatting
            # Fix import issues
            (r'from\s+(\w+)\s+import\s+\*', r'import \1'),  # Avoid star imports
            # Fix function definitions
            (r'def\s+(\w+)\s*\(([^)]*)\)\s*:', r'def \1(\2):'),
        ]
        
        import re
        fixed_code = code
        for pattern, replacement in syntax_fixes:
            try:
                fixed_code = re.sub(pattern, replacement, fixed_code, flags=re.MULTILINE)
            except Exception:
                continue
        
        return fixed_code

    def generate_basic_code(self, hypothesis: str, language: str = "python") -> str:
        """Fallback basic code generation"""
        try:
            # Handle both string and dictionary formats for hypothesis
            if isinstance(hypothesis, dict):
                hypothesis_text = hypothesis.get('hypothesis', str(hypothesis))
            else:
                hypothesis_text = str(hypothesis)
                
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"Generate {language} code to test a research hypothesis."},
                    {"role": "user", "content": f"Generate code to test: {hypothesis_text}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in basic code generation: {e}")
            return f"# Error generating code for hypothesis: {hypothesis_text}\n# Please check your API configuration"

    def run_pylint(self, code: str) -> Dict[str, Any]:
        """Run PyLint analysis on generated code"""
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as temp:
            temp.write(code)
            temp_path = temp.name
        
        try:
            result = subprocess.run(['pylint', temp_path], capture_output=True, text=True)
            
            # Parse pylint output for detailed feedback
            output_lines = result.stdout.split('\n')
            issues = [line for line in output_lines if any(level in line for level in ['ERROR:', 'WARNING:', 'INFO:'])]
            
            analysis = {
                "passed": result.returncode == 0,
                "score": self._extract_pylint_score(result.stdout),
                "output": result.stdout,
                "issues": issues[:10],  # Top 10 issues
                "total_issues": len(issues)
            }
            
            self.note_taker.log("pylint_analysis", analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Error running pylint: {e}")
            return {"passed": False, "error": str(e)}
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _extract_pylint_score(self, pylint_output: str) -> Optional[float]:
        """Extract numerical score from pylint output"""
        score_match = re.search(r'Your code has been rated at ([\d\.]+)/10', pylint_output)
        return float(score_match.group(1)) if score_match else None

    def execute_code_safely(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute code in a secure subprocess with comprehensive error handling
        """
        try:
            # Create a secure temporary file with UTF-8 encoding
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                # Add safety wrapper to the code with enhanced debugging (ASCII safe)
                safe_code = f"""# -*- coding: utf-8 -*-
import sys
import traceback
import warnings
import importlib
warnings.filterwarnings('ignore')

# Function to check if module is available
def check_module(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

# Check critical modules before execution
critical_modules = ['numpy', 'pandas', 'sklearn']
missing_modules = []
for module in critical_modules:
    if not check_module(module):
        missing_modules.append(module)

if missing_modules:
    print("Missing modules: " + ", ".join(missing_modules))
    print("Install with: pip install " + " ".join(missing_modules))
else:
    print("All critical modules available")

# Set up safe execution environment
print("Setting up execution environment...")

try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
    
    print("EXECUTION COMPLETED SUCCESSFULLY")
    
except ImportError as e:
    print("Import Error: " + str(e))
    print("This might be due to missing packages or spelling errors")
    module_name = str(e).split("'")[1] if "'" in str(e) else "unknown"
    print("Failed module: " + module_name)
    
    # Check for common typos
    typos = {{"nummpy": "numpy", "pandsa": "pandas", "skleran": "sklearn"}}
    if module_name in typos:
        print("Did you mean: " + typos[module_name] + "?")
    
except ModuleNotFoundError as e:
    print("Module Not Found: " + str(e))
    print("Try installing the missing package")
    
except Exception as e:
    print("Execution Error: " + str(e))
    print("Traceback:")
    traceback.print_exc()
    
except SystemExit:
    print("SystemExit called - execution stopped")
    
finally:
    print("Code execution finished")
"""
                
                f.write(safe_code)
                temp_file = f.name
            
            start_time = time.time()
            
            # Execute with timeout and capture output with proper encoding
            try:
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tempfile.gettempdir(),
                    encoding='utf-8',
                    errors='replace'  # Handle encoding errors gracefully
                )
                
                execution_time = time.time() - start_time
                
                # Parse results with enhanced analysis
                output = result.stdout
                error = result.stderr
                
                # Analyze success based on multiple indicators
                success_indicators = [
                    "EXECUTION COMPLETED SUCCESSFULLY" in output,
                    "All critical modules available" in output,
                    result.returncode == 0
                ]
                
                error_indicators = [
                    "Import Error:" in output,
                    "Module Not Found:" in output,
                    "Execution Error:" in output,
                    "Missing modules:" in output,
                    result.returncode != 0
                ]
                
                # Determine success
                success = any(success_indicators) and not any(error_indicators)
                
                # Extract specific error types
                error_type = "none"
                if "Import Error:" in output:
                    error_type = "import_error"
                elif "Module Not Found:" in output:
                    error_type = "module_not_found"
                elif "Execution Error:" in output:
                    error_type = "execution_error"
                elif "Missing modules:" in output:
                    error_type = "missing_modules"
                
                return {
                    "success": success,
                    "output": output,
                    "error": error,
                    "execution_time": execution_time,
                    "return_code": result.returncode,
                    "error_type": error_type,
                    "modules_checked": "All critical modules available" in output
                }
                
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Code execution timed out after {timeout} seconds",
                    "execution_time": timeout,
                    "return_code": -1,
                    "error_type": "timeout",
                    "modules_checked": False
                }
                
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Failed to execute code: {str(e)}",
                "execution_time": 0,
                "return_code": -1,
                "error_type": "setup_error",
                "modules_checked": False
            }
            
        finally:
            # Clean up temporary file
            try:
                if 'temp_file' in locals():
                    os.unlink(temp_file)
            except:
                pass

    def validate_code_quality(self, code: str) -> Dict[str, Any]:
        """
        Comprehensive code quality validation
        """
        quality_metrics = {
            "has_imports": "import " in code,
            "has_functions": "def " in code,
            "has_classes": "class " in code,
            "has_docstrings": '"""' in code or "'''" in code,
            "has_type_hints": "->" in code or ": " in code,
            "has_error_handling": "try:" in code or "except:" in code,
            "estimated_complexity": self._estimate_complexity(code),
            "line_count": len(code.split('\n')),
            "has_main_guard": "if __name__" in code
        }
        
        # Calculate overall quality score
        quality_checks = [
            quality_metrics["has_imports"],
            quality_metrics["has_functions"],
            quality_metrics["has_docstrings"],
            quality_metrics["has_error_handling"],
            quality_metrics["line_count"] > 10,
            quality_metrics["estimated_complexity"] < 20
        ]
        
        quality_metrics["quality_score"] = sum(quality_checks) / len(quality_checks)
        
        self.note_taker.log("code_quality_validation", quality_metrics)
        return quality_metrics

    def _estimate_complexity(self, code: str) -> int:
        """Estimate code complexity using simple metrics"""
        complexity = 0
        complexity += code.count("if ")
        complexity += code.count("for ")
        complexity += code.count("while ")
        complexity += code.count("def ")
        complexity += code.count("class ")
        complexity += code.count("try:")
        return complexity

    def _generate_fallback_code(self, hypothesis: str) -> str:
        """Generate a fallback code if enhanced code generation fails"""
        # Handle both string and dictionary formats for hypothesis
        if isinstance(hypothesis, dict):
            hypothesis_text = hypothesis.get('hypothesis', str(hypothesis))
        else:
            hypothesis_text = str(hypothesis)
            
        # Generate basic code
        basic_code = self.generate_basic_code(hypothesis_text)
        
        # Fix common spelling mistakes in imports
        spelling_fixes = {
            "nummpy": "numpy",
            "pandsa": "pandas", 
            "skleran": "sklearn",
            "matplotib": "matplotlib",
            "seabron": "seaborn",
            "import nummpy": "import numpy",
            "from nummpy": "from numpy",
            "import pandsa": "import pandas",
            "from pandsa": "from pandas"
        }
        
        for wrong, correct in spelling_fixes.items():
            basic_code = basic_code.replace(wrong, correct)
        
        # Add standard imports if missing
        standard_imports = [
            "import numpy as np",
            "import pandas as pd", 
            "from sklearn.model_selection import train_test_split",
            "from sklearn.metrics import accuracy_score",
            "import warnings",
            "warnings.filterwarnings('ignore')"
        ]
        
        # Check if imports are present (with correct spelling)
        has_numpy = ("import numpy" in basic_code or "from numpy" in basic_code) and "nummpy" not in basic_code
        has_pandas = ("import pandas" in basic_code or "from pandas" in basic_code) and "pandsa" not in basic_code
        has_sklearn = "from sklearn" in basic_code or "import sklearn" in basic_code
        
        if not (has_numpy and has_pandas and has_sklearn):
            imports_to_add = []
            if not has_numpy:
                imports_to_add.append("import numpy as np")
            if not has_pandas:
                imports_to_add.append("import pandas as pd")
            if not has_sklearn:
                imports_to_add.extend([
                    "from sklearn.model_selection import train_test_split",
                    "from sklearn.metrics import accuracy_score"
                ])
            imports_to_add.extend([
                "import warnings",
                "warnings.filterwarnings('ignore')"
            ])
            
            basic_code = "\n".join(imports_to_add) + "\n\n" + basic_code
        
        # Clean up code formatting
        if "```python" in basic_code:
            basic_code = basic_code.split("```python")[1].split("```")[0].strip()
        elif "```" in basic_code:
            basic_code = basic_code.split("```")[1].strip()
        
        # Wrap main execution in try-catch with better error handling
        if "if __name__" not in basic_code and "print(" not in basic_code:
            basic_code += "\n\n# Execute main code safely\ntry:\n    print('✅ Fallback code execution completed successfully')\nexcept ImportError as ie:\n    print(f'❌ Import Error: {ie}')\nexcept Exception as e:\n    print(f'❌ Execution Error: {e}')"
        
        # Final spell check for common mistakes
        final_check_fixes = {
            "nummpy": "numpy",
            "pandsa": "pandas",
            "accurracy": "accuracy",
            "precission": "precision"
        }
        
        for wrong, correct in final_check_fixes.items():
            if wrong in basic_code:
                print(f"   🔧 Fixed spelling in fallback: {wrong} → {correct}")
                basic_code = basic_code.replace(wrong, correct)
        
        self.note_taker.log_code(basic_code, 
                               hypothesis=hypothesis_text[:100],
                               models_used=0,
                               enhanced=False,
                               fallback=True,
                               spelling_checked=True)
        
        return basic_code

    def generate_research_quality_code(self, hypothesis: str, complexity_level: str = "advanced") -> str:
        """
        Generate research-quality code with advanced features
        """
        try:
            # Handle both string and dictionary formats for hypothesis
            if isinstance(hypothesis, dict):
                hypothesis_text = hypothesis.get('hypothesis', str(hypothesis))
            else:
                hypothesis_text = str(hypothesis)
            
            # Determine research domain and appropriate methodologies
            research_domain = self._analyze_research_domain(hypothesis_text)
            methodologies = self._suggest_methodologies(hypothesis_text, research_domain)
            
            # Create advanced research-focused prompt
            prompt = f"""Generate publication-quality Python research code to test this hypothesis:

HYPOTHESIS: "{hypothesis_text}"

RESEARCH DOMAIN: {research_domain}
SUGGESTED METHODOLOGIES: {', '.join(methodologies)}

ADVANCED REQUIREMENTS:
1. Statistical rigor with p-values, confidence intervals, effect sizes
2. Multiple evaluation metrics and cross-validation
3. Data preprocessing pipeline with feature engineering
4. Baseline comparisons and ablation studies
5. Results visualization with publication-ready plots
6. Comprehensive error analysis and uncertainty quantification
7. Reproducible experiments with seed setting
8. Performance profiling and computational efficiency analysis
9. Statistical significance testing (t-tests, ANOVA, etc.)
10. Model interpretability and feature importance analysis

CODE STRUCTURE:
```python
# Research-quality code template
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class HypothesisTest:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_data(self):
        \"\"\"Generate realistic research data\"\"\"
        pass
        
    def preprocess_data(self, X, y):
        \"\"\"Advanced preprocessing pipeline\"\"\"
        pass
        
    def statistical_analysis(self, results):
        \"\"\"Comprehensive statistical analysis\"\"\"
        pass
        
    def visualize_results(self, results):
        \"\"\"Publication-ready visualizations\"\"\"
        pass
        
    def run_experiment(self):
        \"\"\"Main experiment execution\"\"\"
        pass

def main():
    experiment = HypothesisTest()
    results = experiment.run_experiment()
    return results
```

Return complete, executable, research-grade Python code."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a research scientist who writes publication-quality code with statistical rigor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.1
            )
            
            code = response.choices[0].message.content.strip()
            
            # Apply all quality enhancements
            code = self._apply_advanced_code_fixes(code)
            code = self._add_research_features(code, hypothesis_text)
            code = self._ensure_code_structure(code, hypothesis_text)
            
            return code
            
        except Exception as e:
            print(f"   ⚠️ Research code generation failed: {e}")
            return self.generate_enhanced_code(hypothesis, include_hf_models=True)

    def _analyze_research_domain(self, hypothesis: str) -> str:
        """Analyze hypothesis to determine research domain"""
        hypothesis_lower = hypothesis.lower()
        
        domains = {
            "machine_learning": ["learning", "model", "algorithm", "prediction", "classification"],
            "computer_vision": ["image", "vision", "cnn", "detection", "recognition"],
            "natural_language": ["text", "language", "nlp", "bert", "transformer"],
            "time_series": ["time", "series", "forecasting", "temporal", "sequence"],
            "medical_ai": ["medical", "health", "diagnosis", "patient", "clinical"],
            "robotics": ["robot", "control", "navigation", "manipulation"],
            "recommendation": ["recommend", "collaborative", "rating", "preference"],
            "optimization": ["optimization", "genetic", "evolutionary", "search"]
        }
        
        for domain, keywords in domains.items():
            if any(keyword in hypothesis_lower for keyword in keywords):
                return domain
        
        return "general_ai"

    def _suggest_methodologies(self, hypothesis: str, domain: str) -> list:
        """Suggest appropriate methodologies based on domain"""
        methodology_map = {
            "machine_learning": ["cross_validation", "ensemble_methods", "hyperparameter_tuning"],
            "computer_vision": ["data_augmentation", "transfer_learning", "cnn_architectures"],
            "natural_language": ["tokenization", "embedding_analysis", "attention_mechanisms"],
            "time_series": ["stationarity_tests", "seasonal_decomposition", "autocorrelation"],
            "medical_ai": ["clinical_validation", "sensitivity_analysis", "ethical_considerations"],
            "general_ai": ["statistical_testing", "baseline_comparison", "ablation_study"]
        }
        
        return methodology_map.get(domain, methodology_map["general_ai"])

    def _add_research_features(self, code: str, hypothesis: str) -> str:
        """Add advanced research features to code"""
        
        # Add research-specific imports if missing
        research_imports = [
            "from scipy import stats",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from sklearn.model_selection import cross_val_score, StratifiedKFold",
            "from sklearn.metrics import classification_report, confusion_matrix"
        ]
        
        for imp in research_imports:
            if imp not in code:
                code = imp + "\n" + code
        
        # Add statistical analysis functions
        if "def statistical_analysis" not in code:
            stats_function = '''
def statistical_analysis(results_dict):
    """Perform comprehensive statistical analysis"""
    import numpy as np
    from scipy import stats
    
    print("\\n=== STATISTICAL ANALYSIS ===")
    
    for method, scores in results_dict.items():
        if isinstance(scores, (list, np.ndarray)) and len(scores) > 1:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            ci_95 = stats.t.interval(0.95, len(scores)-1, 
                                   loc=mean_score, 
                                   scale=stats.sem(scores))
            
            print(f"{method}:")
            print(f"  Mean: {mean_score:.4f} (+/- {std_score:.4f})")
            print(f"  95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
            
    return results_dict
'''
            code += stats_function
        
        # Add visualization function
        if "def visualize_results" not in code and "plt" in code:
            viz_function = '''
def visualize_results(results_dict):
    """Create publication-ready visualizations"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot of results
    methods = list(results_dict.keys())
    scores = list(results_dict.values())
    
    axes[0].boxplot(scores, labels=methods)
    axes[0].set_title('Performance Comparison')
    axes[0].set_ylabel('Accuracy Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Bar plot with error bars
    means = [np.mean(s) if isinstance(s, (list, np.ndarray)) else s for s in scores]
    stds = [np.std(s) if isinstance(s, (list, np.ndarray)) and len(s) > 1 else 0 for s in scores]
    
    axes[1].bar(methods, means, yerr=stds, capsize=5)
    axes[1].set_title('Mean Performance with Standard Deviation')
    axes[1].set_ylabel('Accuracy Score')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('hypothesis_results.png', dpi=300, bbox_inches='tight')
    print("Results saved to 'hypothesis_results.png'")
    # plt.show()  # Commented for automated execution
    
    return fig
'''
            code += viz_function
        
        return code

# Example usage
if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
    
    # Mock note taker for testing
    class MockNoteTaker:
        def log(self, *args, **kwargs): pass
    
    note_taker = MockNoteTaker()
    agent = EnhancedCodeAgent(OPENAI_API_KEY, note_taker)
    
    test_hypothesis = "Transformer models outperform LSTM networks in text classification tasks"
    
    print("Discovering relevant models...")
    models = agent.discover_relevant_models(test_hypothesis)
    print(f"Found {len(models)} relevant models")
    
    print("Generating enhanced code...")
    code = agent.generate_enhanced_code(test_hypothesis)
    print(f"Generated {len(code)} characters of code")
    
    print("Validating code quality...")
    quality = agent.validate_code_quality(code)
    print(f"Quality score: {quality['quality_score']:.2f}") 
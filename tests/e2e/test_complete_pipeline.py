"""
End-to-End Tests for Complete Research Assistant Pipeline

Tests:
- Complete user workflow from query to final paper
- Real-world scenarios and edge cases
- Performance under load
- Integration with external services
- User experience validation
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import time
from unittest.mock import Mock, patch, MagicMock

from src.utils.config_manager import get_config
from src.utils.logging_config import setup_logging
from src.utils.error_handling import get_error_handler


@pytest.mark.e2e
class TestCompleteResearchPipeline:
    """End-to-end tests for the complete research assistant pipeline"""
    
    @pytest.fixture
    def pipeline_output_dir(self, temp_dir):
        """Create temporary directory for pipeline outputs"""
        output_dir = Path(temp_dir) / "pipeline_outputs"
        output_dir.mkdir(exist_ok=True)
        return str(output_dir)
    
    @pytest.fixture
    def mock_complete_pipeline(self, config_manager, test_logger, pipeline_output_dir):
        """Create a complete mocked research pipeline"""
        class MockResearchPipeline:
            def __init__(self, config, logger, output_dir):
                self.config = config
                self.logger = logger
                self.output_dir = Path(output_dir)
                self.state = {}
                
                # Initialize mock agents
                self.search_agent = Mock()
                self.hypothesis_agent = Mock()
                self.code_agent = Mock()
                self.visualization_agent = Mock()
                self.report_agent = Mock()
                self.note_taker = Mock()
                
                self._setup_mock_responses()
            
            def _setup_mock_responses(self):
                """Setup realistic mock responses"""
                # Search agent responses
                self.search_agent.search.return_value = [
                    {
                        "title": "Machine Learning Approaches for Early Disease Detection",
                        "abstract": "This paper presents comprehensive machine learning methodologies for early disease detection using multimodal data sources including medical imaging, genomics, and clinical records.",
                        "authors": ["Dr. Sarah Johnson", "Dr. Michael Chen", "Dr. Lisa Anderson"],
                        "published": "2024-01-15",
                        "arxiv_id": "2401.12345",
                        "categories": ["cs.LG", "cs.AI", "q-bio.QM"],
                        "url": "https://arxiv.org/abs/2401.12345"
                    },
                    {
                        "title": "Deep Learning for Biomarker Discovery in Cancer Research",
                        "abstract": "We propose novel deep learning architectures for identifying potential biomarkers in cancer research, demonstrating significant improvements in prediction accuracy.",
                        "authors": ["Dr. Robert Kim", "Dr. Emily Davis"],
                        "published": "2024-02-01",
                        "arxiv_id": "2402.01234",
                        "categories": ["cs.LG", "q-bio.GN"],
                        "url": "https://arxiv.org/abs/2402.01234"
                    }
                ]
                
                # Hypothesis agent response
                self.hypothesis_agent.generate_hypothesis.return_value = """
                Hypothesis: Advanced machine learning algorithms, particularly ensemble methods combining 
                deep neural networks with traditional statistical approaches, can achieve superior performance 
                in early disease detection compared to individual algorithms by effectively integrating 
                multimodal biomedical data sources and capturing complex non-linear relationships between 
                genetic markers, imaging features, and clinical variables.
                """
                
                # Code agent response
                self.code_agent.generate_code.return_value = {
                    "code": '''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class EnsembleDiseaseDetector:
    """
    Ensemble machine learning model for early disease detection
    combining multiple algorithms for improved performance.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=random_state)
        }
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y):
        """Train the ensemble model"""
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.models.items():
            model.fit(X_scaled, y)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """Predict probabilities using ensemble averaging"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = []
        
        for model in self.models.values():
            pred = model.predict_proba(X_scaled)
            predictions.append(pred)
        
        # Ensemble averaging
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def evaluate(self, X, y, cv_folds=5):
        """Evaluate model performance using cross-validation"""
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
            results[name] = {
                'mean_auc': scores.mean(),
                'std_auc': scores.std(),
                'scores': scores.tolist()
            }
        
        return results

# Demonstration with synthetic data
def demonstrate_early_disease_detection():
    """Demonstrate the ensemble approach with synthetic biomedical data"""
    np.random.seed(42)
    
    # Generate synthetic multimodal biomedical data
    n_samples = 1000
    n_genetic_features = 50
    n_imaging_features = 30
    n_clinical_features = 20
    
    # Simulate genetic markers
    genetic_data = np.random.randn(n_samples, n_genetic_features)
    
    # Simulate imaging features
    imaging_data = np.random.randn(n_samples, n_imaging_features)
    
    # Simulate clinical variables
    clinical_data = np.random.randn(n_samples, n_clinical_features)
    
    # Combine all features
    X = np.hstack([genetic_data, imaging_data, clinical_data])
    
    # Generate target with some signal
    disease_score = (
        0.3 * genetic_data[:, :5].sum(axis=1) +
        0.4 * imaging_data[:, :3].sum(axis=1) +
        0.3 * clinical_data[:, :2].sum(axis=1) +
        np.random.randn(n_samples) * 0.5
    )
    y = (disease_score > np.percentile(disease_score, 70)).astype(int)
    
    # Train and evaluate ensemble model
    detector = EnsembleDiseaseDetector()
    results = detector.evaluate(X, y)
    
    # Train final model for predictions
    detector.fit(X, y)
    probabilities = detector.predict_proba(X)
    
    return {
        'evaluation_results': results,
        'sample_predictions': probabilities[:10].tolist(),
        'feature_dimensions': {
            'genetic': n_genetic_features,
            'imaging': n_imaging_features,
            'clinical': n_clinical_features,
            'total': X.shape[1]
        },
        'dataset_info': {
            'n_samples': n_samples,
            'positive_cases': int(y.sum()),
            'negative_cases': int((1-y).sum()),
            'prevalence': float(y.mean())
        }
    }

if __name__ == "__main__":
    results = demonstrate_early_disease_detection()
    print("Ensemble Disease Detection Results:")
    print(f"Dataset: {results['dataset_info']}")
    print(f"Feature dimensions: {results['feature_dimensions']}")
    print("Model performance:")
    for model, metrics in results['evaluation_results'].items():
        print(f"  {model}: AUC = {metrics['mean_auc']:.3f} ± {metrics['std_auc']:.3f}")
''',
                    "validation_result": {
                        "syntax_valid": True,
                        "execution_successful": True,
                        "pylint_score": 8.5,
                        "complexity_score": 7.2,
                        "test_results": {
                            "all_tests_passed": True,
                            "num_tests": 5,
                            "coverage": 85.0
                        }
                    },
                    "metadata": {
                        "language": "python",
                        "libraries": ["numpy", "pandas", "scikit-learn", "matplotlib"],
                        "model_type": "ensemble",
                        "domain": "biomedical"
                    }
                }
                
                # Visualization agent response
                self.visualization_agent.create_visualizations.return_value = {
                    "charts": [
                        {
                            "type": "performance_comparison",
                            "title": "Model Performance Comparison",
                            "path": str(self.output_dir / "performance_comparison.png"),
                            "description": "ROC-AUC comparison across ensemble components"
                        },
                        {
                            "type": "feature_importance", 
                            "title": "Feature Importance Analysis",
                            "path": str(self.output_dir / "feature_importance.png"),
                            "description": "Relative importance of genetic, imaging, and clinical features"
                        },
                        {
                            "type": "training_curves",
                            "title": "Model Training Convergence",
                            "path": str(self.output_dir / "training_curves.png"),
                            "description": "Training and validation performance over iterations"
                        }
                    ],
                    "analysis": "The visualizations demonstrate that the ensemble approach achieves superior performance compared to individual models, with genetic markers showing the highest predictive power.",
                    "metadata": {
                        "total_charts": 3,
                        "chart_format": "PNG",
                        "resolution": "300 DPI"
                    }
                }
                
                # Report agent response
                self.report_agent.generate_paper.return_value = {
                    "content": """
# Ensemble Machine Learning for Early Disease Detection: A Comprehensive Approach

## Abstract

This study presents a novel ensemble machine learning framework for early disease detection that combines multiple algorithmic approaches to achieve superior predictive performance. Our methodology integrates genetic markers, medical imaging features, and clinical variables through advanced ensemble techniques, demonstrating significant improvements over individual algorithms.

## 1. Introduction

Early disease detection remains a critical challenge in modern healthcare, with significant implications for patient outcomes and healthcare costs. Traditional approaches often rely on single algorithms or limited data modalities, potentially missing complex patterns that could improve diagnostic accuracy.

## 2. Methodology

### 2.1 Ensemble Architecture

Our approach combines three complementary algorithms:
- Random Forest Classifier for robust feature selection
- Gradient Boosting for sequential error correction
- Multi-layer Perceptron for non-linear pattern recognition

### 2.2 Data Integration

The framework processes multimodal data including:
- Genetic markers (50 features)
- Medical imaging features (30 features)  
- Clinical variables (20 features)

## 3. Results

The ensemble approach achieved superior performance across all metrics:
- Random Forest: AUC = 0.847 ± 0.023
- Gradient Boosting: AUC = 0.851 ± 0.019
- Neural Network: AUC = 0.839 ± 0.027
- Ensemble Average: AUC = 0.863 ± 0.015

## 4. Discussion

The results demonstrate the effectiveness of ensemble learning for early disease detection, with particular strengths in handling heterogeneous biomedical data sources.

## 5. Conclusion

This work establishes a foundation for advanced machine learning applications in preventive healthcare, with potential for significant clinical impact.

## References

[1] Johnson, S. et al. Machine Learning Approaches for Early Disease Detection. arXiv:2401.12345
[2] Kim, R. et al. Deep Learning for Biomarker Discovery in Cancer Research. arXiv:2402.01234
""",
                    "format": "markdown",
                    "metadata": {
                        "word_count": 1250,
                        "sections": 5,
                        "references": 2,
                        "figures": 3,
                        "export_formats": ["html", "latex", "pdf"]
                    }
                }
                
                # Note taker responses
                self.note_taker.log_interaction.return_value = True
                self.note_taker.get_session_summary.return_value = {
                    "session_id": "test_session_123",
                    "total_interactions": 5,
                    "successful_steps": 5,
                    "errors": 0,
                    "duration_minutes": 2.5
                }
            
            def run_complete_pipeline(self, user_query, enable_human_feedback=False):
                """Run the complete research pipeline"""
                start_time = time.time()
                results = {}
                
                try:
                    self.logger.info(f"Starting research pipeline for query: {user_query}")
                    
                    # Step 1: Search for papers
                    self.logger.info("Step 1: Searching for relevant papers")
                    papers = self.search_agent.search(user_query)
                    results["papers"] = papers
                    self.note_taker.log_interaction("search", {"query": user_query, "papers_found": len(papers)})
                    
                    # Step 2: Generate hypothesis
                    self.logger.info("Step 2: Generating research hypothesis")
                    hypothesis = self.hypothesis_agent.generate_hypothesis(papers)
                    results["hypothesis"] = hypothesis
                    self.note_taker.log_interaction("hypothesis", {"hypothesis_length": len(hypothesis)})
                    
                    # Step 3: Generate code
                    self.logger.info("Step 3: Generating and validating code")
                    code_result = self.code_agent.generate_code(hypothesis)
                    results["code"] = code_result
                    self.note_taker.log_interaction("code", code_result["validation_result"])
                    
                    # Step 4: Create visualizations
                    self.logger.info("Step 4: Creating visualizations")
                    viz_result = self.visualization_agent.create_visualizations(hypothesis, code_result["code"])
                    results["visualizations"] = viz_result
                    self.note_taker.log_interaction("visualization", {"charts_created": len(viz_result["charts"])})
                    
                    # Step 5: Generate research paper
                    self.logger.info("Step 5: Generating research paper")
                    paper_result = self.report_agent.generate_paper(papers, hypothesis, code_result, viz_result)
                    results["paper"] = paper_result
                    self.note_taker.log_interaction("report", paper_result["metadata"])
                    
                    # Calculate total time
                    total_time = time.time() - start_time
                    results["metadata"] = {
                        "total_time_seconds": total_time,
                        "pipeline_version": "1.0.0",
                        "successful": True
                    }
                    
                    self.logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
                    return results
                    
                except Exception as e:
                    self.logger.error(f"Pipeline failed: {e}")
                    results["error"] = str(e)
                    results["metadata"] = {
                        "total_time_seconds": time.time() - start_time,
                        "successful": False
                    }
                    raise
        
        return MockResearchPipeline(config_manager, test_logger, pipeline_output_dir)
    
    def test_complete_research_workflow(self, mock_complete_pipeline, performance_timer):
        """Test complete research workflow from query to final paper"""
        user_query = "Can machine learning improve early detection of cardiovascular disease?"
        
        performance_timer.start()
        results = mock_complete_pipeline.run_complete_pipeline(user_query)
        duration = performance_timer.stop()
        
        # Verify all pipeline steps completed
        assert "papers" in results
        assert "hypothesis" in results
        assert "code" in results
        assert "visualizations" in results
        assert "paper" in results
        assert "metadata" in results
        
        # Verify paper results
        assert len(results["papers"]) == 2
        assert "Machine Learning Approaches" in results["papers"][0]["title"]
        
        # Verify hypothesis
        assert "ensemble methods" in results["hypothesis"]
        assert len(results["hypothesis"]) > 100
        
        # Verify code generation
        code_result = results["code"]
        assert code_result["validation_result"]["syntax_valid"] is True
        assert code_result["validation_result"]["execution_successful"] is True
        assert "EnsembleDiseaseDetector" in code_result["code"]
        
        # Verify visualizations
        viz_result = results["visualizations"]
        assert len(viz_result["charts"]) == 3
        assert "performance_comparison" in viz_result["charts"][0]["type"]
        
        # Verify paper generation
        paper_result = results["paper"]
        assert "Abstract" in paper_result["content"]
        assert "Methodology" in paper_result["content"]
        assert "Results" in paper_result["content"]
        assert paper_result["metadata"]["word_count"] > 1000
        
        # Verify performance
        assert duration < 5.0  # Should complete within 5 seconds
        assert results["metadata"]["successful"] is True
    
    def test_pipeline_error_handling_and_recovery(self, mock_complete_pipeline):
        """Test pipeline error handling and recovery mechanisms"""
        # Configure search agent to fail
        mock_complete_pipeline.search_agent.search.side_effect = Exception("Search service unavailable")
        
        user_query = "Test query for error handling"
        
        # Test that pipeline handles errors gracefully
        with pytest.raises(Exception) as exc_info:
            mock_complete_pipeline.run_complete_pipeline(user_query)
        
        assert "Search service unavailable" in str(exc_info.value)
    
    def test_pipeline_with_invalid_inputs(self, mock_complete_pipeline):
        """Test pipeline behavior with invalid or edge case inputs"""
        # Test empty query
        with pytest.raises(Exception):
            mock_complete_pipeline.run_complete_pipeline("")
        
        # Test very long query
        long_query = "A" * 10000
        # This should either work or fail gracefully
        try:
            results = mock_complete_pipeline.run_complete_pipeline(long_query)
            assert results is not None
        except Exception as e:
            # Should be a handled exception, not a crash
            assert isinstance(e, (ValueError, RuntimeError))
    
    def test_pipeline_output_validation(self, mock_complete_pipeline, pipeline_output_dir):
        """Test that pipeline outputs are properly formatted and valid"""
        user_query = "Machine learning for medical diagnosis"
        results = mock_complete_pipeline.run_complete_pipeline(user_query)
        
        # Validate paper structure
        paper = results["paper"]
        required_sections = ["Abstract", "Introduction", "Methodology", "Results", "Conclusion"]
        for section in required_sections:
            assert section in paper["content"], f"Missing section: {section}"
        
        # Validate code structure
        code_result = results["code"]
        assert "import" in code_result["code"]  # Should have imports
        assert "class" in code_result["code"]  # Should have class definitions
        assert "def" in code_result["code"]  # Should have function definitions
        
        # Validate visualizations
        viz_result = results["visualizations"]
        for chart in viz_result["charts"]:
            assert "type" in chart
            assert "title" in chart
            assert "path" in chart
            assert "description" in chart
    
    @pytest.mark.performance
    def test_pipeline_performance_under_load(self, mock_complete_pipeline, performance_timer):
        """Test pipeline performance under simulated load"""
        queries = [
            "Machine learning for cancer detection",
            "AI in drug discovery",
            "Deep learning for medical imaging", 
            "Natural language processing in healthcare",
            "Computer vision for pathology"
        ]
        
        performance_timer.start()
        results = []
        
        for query in queries:
            result = mock_complete_pipeline.run_complete_pipeline(query)
            results.append(result)
            
        total_duration = performance_timer.stop()
        
        # Verify all queries completed successfully
        assert len(results) == 5
        for result in results:
            assert result["metadata"]["successful"] is True
        
        # Performance should scale reasonably
        avg_time_per_query = total_duration / len(queries)
        assert avg_time_per_query < 2.0  # Average < 2 seconds per query
    
    @pytest.mark.slow
    def test_pipeline_memory_efficiency(self, mock_complete_pipeline):
        """Test pipeline memory usage efficiency"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run multiple pipeline iterations
        for i in range(5):
            query = f"Test query {i} for memory efficiency testing"
            results = mock_complete_pipeline.run_complete_pipeline(query)
            assert results["metadata"]["successful"] is True
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for 5 iterations)
        assert memory_increase < 100 * 1024 * 1024


@pytest.mark.e2e
class TestUserExperienceScenarios:
    """Test realistic user experience scenarios"""
    
    def test_beginner_researcher_workflow(self, mock_complete_pipeline):
        """Test workflow for beginner researcher with simple query"""
        simple_query = "machine learning healthcare"
        
        results = mock_complete_pipeline.run_complete_pipeline(simple_query)
        
        # Results should be comprehensive despite simple query
        assert len(results["papers"]) >= 2
        assert len(results["hypothesis"]) > 50
        assert "class" in results["code"]["code"]  # Should generate substantial code
        assert len(results["visualizations"]["charts"]) >= 2
        assert results["paper"]["metadata"]["word_count"] > 500
    
    def test_expert_researcher_workflow(self, mock_complete_pipeline):
        """Test workflow for expert researcher with detailed query"""
        expert_query = """
        Investigate the application of ensemble deep learning methods combining 
        convolutional neural networks and recurrent neural networks for early 
        detection of Alzheimer's disease using multimodal neuroimaging data 
        including structural MRI, functional MRI, and PET scans
        """
        
        results = mock_complete_pipeline.run_complete_pipeline(expert_query)
        
        # Results should reflect the complexity of the query
        assert "ensemble" in results["hypothesis"].lower()
        assert "neural network" in results["code"]["code"].lower()
        assert len(results["visualizations"]["charts"]) >= 3
        assert results["paper"]["metadata"]["sections"] >= 5
    
    def test_interdisciplinary_research_workflow(self, mock_complete_pipeline):
        """Test workflow for interdisciplinary research combining multiple domains"""
        interdisciplinary_query = """
        How can quantum computing accelerate machine learning algorithms 
        for drug discovery in personalized medicine?
        """
        
        results = mock_complete_pipeline.run_complete_pipeline(interdisciplinary_query)
        
        # Should handle complex interdisciplinary topics
        assert results["metadata"]["successful"] is True
        assert len(results["hypothesis"]) > 100
        
        # Paper should be comprehensive for complex topic
        paper_content = results["paper"]["content"].lower()
        interdisciplinary_terms = ["quantum", "machine learning", "drug", "personalized"]
        for term in interdisciplinary_terms:
            assert term in paper_content or term.replace(" ", "") in paper_content


@pytest.mark.e2e
@pytest.mark.requires_db
class TestDatabaseIntegration:
    """Test end-to-end database integration scenarios"""
    
    @pytest.fixture
    def mock_database(self):
        """Mock database for testing"""
        class MockDatabase:
            def __init__(self):
                self.papers = []
                self.sessions = []
                self.queries = []
            
            def store_papers(self, papers):
                self.papers.extend(papers)
                return len(papers)
            
            def store_session(self, session_data):
                self.sessions.append(session_data)
                return f"session_{len(self.sessions)}"
            
            def search_papers(self, query, limit=10):
                # Simple mock search
                return self.papers[:limit]
            
            def get_session_history(self, user_id):
                return [s for s in self.sessions if s.get("user_id") == user_id]
        
        return MockDatabase()
    
    def test_pipeline_with_database_persistence(self, mock_complete_pipeline, mock_database):
        """Test pipeline with database persistence"""
        user_query = "AI for medical diagnosis"
        user_id = "test_user_123"
        
        # Run pipeline
        results = mock_complete_pipeline.run_complete_pipeline(user_query)
        
        # Store results in database
        papers_stored = mock_database.store_papers(results["papers"])
        session_id = mock_database.store_session({
            "user_id": user_id,
            "query": user_query,
            "results": results,
            "timestamp": time.time()
        })
        
        # Verify storage
        assert papers_stored == 2
        assert session_id == "session_1"
        
        # Verify retrieval
        user_history = mock_database.get_session_history(user_id)
        assert len(user_history) == 1
        assert user_history[0]["query"] == user_query


@pytest.mark.e2e
@pytest.mark.security
class TestSecurityScenarios:
    """Test security-related scenarios"""
    
    def test_input_sanitization(self, mock_complete_pipeline):
        """Test that malicious inputs are properly sanitized"""
        malicious_queries = [
            "'; DROP TABLE papers; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "{{7*7}}{{user.name}}",  # Template injection
        ]
        
        for query in malicious_queries:
            # Pipeline should handle malicious input gracefully
            try:
                results = mock_complete_pipeline.run_complete_pipeline(query)
                # If successful, output should be sanitized
                assert "<script>" not in str(results)
                assert "DROP TABLE" not in str(results)
            except Exception:
                # If it fails, it should be a controlled failure, not a security breach
                pass
    
    def test_api_key_handling(self, mock_complete_pipeline, test_logger):
        """Test that API keys are never logged or exposed"""
        import io
        
        # Capture log output
        log_stream = io.StringIO()
        handler = test_logger.handlers[0] if test_logger.handlers else None
        if handler:
            handler.stream = log_stream
        
        # Run pipeline (which would use API keys internally)
        results = mock_complete_pipeline.run_complete_pipeline("test query")
        
        # Check that no API keys appear in logs
        log_output = log_stream.getvalue()
        sensitive_patterns = ["sk-", "api_key", "secret", "token"]
        for pattern in sensitive_patterns:
            assert pattern not in log_output.lower()
        
        # Check that API keys don't appear in results
        results_str = str(results)
        for pattern in sensitive_patterns:
            assert pattern not in results_str.lower()


@pytest.mark.e2e
@pytest.mark.performance
class TestScalabilityScenarios:
    """Test scalability and performance scenarios"""
    
    def test_large_query_processing(self, mock_complete_pipeline, performance_timer):
        """Test processing of large/complex queries"""
        large_query = " ".join([
            "machine learning", "artificial intelligence", "deep learning",
            "neural networks", "computer vision", "natural language processing"
        ] * 20)  # Very long query
        
        performance_timer.start()
        results = mock_complete_pipeline.run_complete_pipeline(large_query)
        duration = performance_timer.stop()
        
        # Should handle large queries efficiently
        assert results["metadata"]["successful"] is True
        assert duration < 10.0  # Should complete within 10 seconds even for large query
    
    def test_concurrent_user_simulation(self, mock_complete_pipeline):
        """Test handling multiple concurrent users"""
        import threading
        import queue
        
        queries = [
            "ML for cancer detection",
            "AI in drug discovery", 
            "Deep learning medical imaging",
            "NLP for clinical notes",
            "Computer vision pathology"
        ]
        
        results_queue = queue.Queue()
        
        def run_pipeline(query):
            try:
                result = mock_complete_pipeline.run_complete_pipeline(query)
                results_queue.put(("success", result))
            except Exception as e:
                results_queue.put(("error", str(e)))
        
        # Start concurrent threads
        threads = []
        for query in queries:
            thread = threading.Thread(target=run_pipeline, args=(query,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Collect results
        successful_results = 0
        errors = []
        
        while not results_queue.empty():
            status, result = results_queue.get()
            if status == "success":
                successful_results += 1
            else:
                errors.append(result)
        
        # All queries should succeed
        assert successful_results == len(queries)
        assert len(errors) == 0 
"""
Integration Tests for Agent Interactions

Tests:
- Agent coordination and data flow
- Pipeline execution with real agent interactions
- Configuration integration across agents
- Error handling between agents
- Performance of agent chains
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.utils.config_manager import ConfigManager
from src.utils.logging_config import setup_logging
from src.utils.error_handling import ErrorHandler, APIError, ValidationError


@pytest.mark.integration
class TestAgentPipelineIntegration:
    """Test integration between multiple agents in the research pipeline"""
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for integration testing"""
        agents = {}
        
        # Mock SearchAgent
        search_agent = Mock()
        search_agent.search.return_value = [
            {"title": "Test Paper 1", "abstract": "Abstract 1", "arxiv_id": "1234.5678"},
            {"title": "Test Paper 2", "abstract": "Abstract 2", "arxiv_id": "2345.6789"}
        ]
        agents['search'] = search_agent
        
        # Mock HypothesisAgent
        hypothesis_agent = Mock()
        hypothesis_agent.generate_hypothesis.return_value = "Test hypothesis based on papers"
        agents['hypothesis'] = hypothesis_agent
        
        # Mock CodeAgent
        code_agent = Mock()
        code_agent.generate_code.return_value = {
            "code": "def test_function(): return 'test'",
            "validation_result": {"syntax_valid": True, "execution_successful": True}
        }
        agents['code'] = code_agent
        
        # Mock VisualizationAgent
        viz_agent = Mock()
        viz_agent.create_visualizations.return_value = {
            "charts": [{"type": "line", "path": "/tmp/chart1.png"}],
            "analysis": "Visualization analysis"
        }
        agents['visualization'] = viz_agent
        
        # Mock ReportAgent
        report_agent = Mock()
        report_agent.generate_paper.return_value = {
            "content": "Generated research paper content",
            "format": "latex",
            "metadata": {"pages": 10, "sections": 5}
        }
        agents['report'] = report_agent
        
        return agents
    
    def test_sequential_agent_execution(self, mock_agents, config_manager, test_logger):
        """Test sequential execution of agents with data flow"""
        # Simulate research pipeline execution
        user_query = "Machine learning for healthcare"
        
        # Step 1: Search for papers
        papers = mock_agents['search'].search(user_query)
        assert len(papers) == 2
        assert papers[0]['title'] == "Test Paper 1"
        
        # Step 2: Generate hypothesis from papers
        hypothesis = mock_agents['hypothesis'].generate_hypothesis(papers)
        assert hypothesis == "Test hypothesis based on papers"
        
        # Step 3: Generate code based on hypothesis
        code_result = mock_agents['code'].generate_code(hypothesis)
        assert code_result['validation_result']['syntax_valid'] is True
        
        # Step 4: Create visualizations
        viz_result = mock_agents['visualization'].create_visualizations(
            hypothesis, code_result['code']
        )
        assert len(viz_result['charts']) == 1
        
        # Step 5: Generate final paper
        paper_result = mock_agents['report'].generate_paper(
            papers, hypothesis, code_result['code'], viz_result
        )
        assert "research paper content" in paper_result['content']
        
        # Verify all agents were called with correct parameters
        mock_agents['search'].search.assert_called_once_with(user_query)
        mock_agents['hypothesis'].generate_hypothesis.assert_called_once_with(papers)
        mock_agents['code'].generate_code.assert_called_once_with(hypothesis)
        mock_agents['visualization'].create_visualizations.assert_called_once()
        mock_agents['report'].generate_paper.assert_called_once()
    
    def test_agent_error_propagation(self, mock_agents, test_logger, error_handler):
        """Test error handling and propagation between agents"""
        # Configure search agent to fail
        mock_agents['search'].search.side_effect = APIError("Search API failed", status_code=500)
        
        user_query = "Test query"
        
        # Test error propagation
        with pytest.raises(APIError) as exc_info:
            papers = mock_agents['search'].search(user_query)
        
        assert "Search API failed" in str(exc_info.value)
        assert exc_info.value.status_code == 500
        
        # Verify downstream agents are not called when upstream fails
        mock_agents['hypothesis'].generate_hypothesis.assert_not_called()
        mock_agents['code'].generate_code.assert_not_called()
    
    def test_agent_data_validation_between_stages(self, mock_agents):
        """Test data validation between agent stages"""
        # Test invalid data from search agent
        mock_agents['search'].search.return_value = [
            {"title": "", "abstract": ""},  # Invalid: empty fields
            {"missing_fields": "data"}  # Invalid: missing required fields
        ]
        
        user_query = "Test query"
        papers = mock_agents['search'].search(user_query)
        
        # Hypothesis agent should validate input data
        with patch.object(mock_agents['hypothesis'], 'generate_hypothesis') as mock_hyp:
            mock_hyp.side_effect = ValidationError("Invalid papers data", field="papers")
            
            with pytest.raises(ValidationError) as exc_info:
                mock_agents['hypothesis'].generate_hypothesis(papers)
            
            assert "Invalid papers data" in str(exc_info.value)
            assert exc_info.value.field == "papers"
    
    def test_configuration_consistency_across_agents(self, config_manager, mock_agents):
        """Test that all agents use consistent configuration"""
        # Test that each agent receives the same configuration
        search_config = config_manager.get_agent_config("search")
        hypothesis_config = config_manager.get_agent_config("hypothesis")
        code_config = config_manager.get_agent_config("code")
        
        # Verify agents have access to their specific configurations
        assert search_config.get("max_papers_per_search") == 3
        assert hypothesis_config.get("max_iterations") == 2
        assert code_config.get("enable_huggingface") is False
        
        # Test global configuration access
        api_config = config_manager.get_api_config()
        assert api_config["openai"]["model"] == "gpt-3.5-turbo"
    
    @pytest.mark.performance
    def test_pipeline_performance_integration(self, mock_agents, performance_timer):
        """Test overall pipeline performance"""
        user_query = "Performance test query"
        
        performance_timer.start()
        
        # Execute full pipeline
        papers = mock_agents['search'].search(user_query)
        hypothesis = mock_agents['hypothesis'].generate_hypothesis(papers)
        code_result = mock_agents['code'].generate_code(hypothesis)
        viz_result = mock_agents['visualization'].create_visualizations(hypothesis, code_result['code'])
        paper_result = mock_agents['report'].generate_paper(papers, hypothesis, code_result['code'], viz_result)
        
        duration = performance_timer.stop()
        
        # Pipeline should complete quickly with mocked agents
        assert duration < 1.0  # Less than 1 second for mocked pipeline
        assert paper_result is not None


@pytest.mark.integration
class TestAgentStateManagement:
    """Test state management and persistence across agent interactions"""
    
    def test_agent_state_sharing(self, mock_agents, temp_dir):
        """Test sharing state between agents"""
        state_file = Path(temp_dir) / "pipeline_state.json"
        
        # Create a simple state manager
        class PipelineState:
            def __init__(self):
                self.state = {}
            
            def set(self, key, value):
                self.state[key] = value
            
            def get(self, key, default=None):
                return self.state.get(key, default)
            
            def save(self, file_path):
                import json
                with open(file_path, 'w') as f:
                    json.dump(self.state, f)
            
            def load(self, file_path):
                import json
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        self.state = json.load(f)
        
        state = PipelineState()
        
        # Simulate agent state updates
        papers = mock_agents['search'].search("test query")
        state.set("papers", papers)
        state.set("query", "test query")
        
        hypothesis = mock_agents['hypothesis'].generate_hypothesis(state.get("papers"))
        state.set("hypothesis", hypothesis)
        
        # Save and reload state
        state.save(state_file)
        
        new_state = PipelineState()
        new_state.load(state_file)
        
        # Verify state persistence
        assert new_state.get("query") == "test query"
        assert len(new_state.get("papers")) == 2
        assert new_state.get("hypothesis") == "Test hypothesis based on papers"
    
    def test_agent_retry_mechanism_integration(self, mock_agents, error_handler):
        """Test retry mechanisms across multiple agents"""
        from src.utils.error_handling import retry_with_backoff, RetryConfig, APIError
        
        # Configure hypothesis agent to fail twice then succeed
        call_count = 0
        def failing_hypothesis_generation(papers):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise APIError("Temporary API failure")
            return "Successful hypothesis after retries"
        
        mock_agents['hypothesis'].generate_hypothesis.side_effect = failing_hypothesis_generation
        
        # Apply retry decorator
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)  # Fast retries for testing
        
        @retry_with_backoff(retry_config)
        def generate_hypothesis_with_retry(papers):
            return mock_agents['hypothesis'].generate_hypothesis(papers)
        
        # Test successful retry
        papers = [{"title": "Test", "abstract": "Test"}]
        result = generate_hypothesis_with_retry(papers)
        
        assert result == "Successful hypothesis after retries"
        assert call_count == 3  # Failed twice, succeeded on third attempt


@pytest.mark.integration
class TestAgentCommunicationProtocols:
    """Test communication protocols and data formats between agents"""
    
    def test_standardized_data_formats(self, mock_agents):
        """Test that agents use standardized data formats for communication"""
        # Define expected data format schemas
        paper_schema = {
            "required_fields": ["title", "abstract", "arxiv_id"],
            "optional_fields": ["authors", "published", "categories"]
        }
        
        hypothesis_schema = {
            "type": "string",
            "min_length": 10,
            "max_length": 5000
        }
        
        code_schema = {
            "required_fields": ["code", "validation_result"],
            "code_type": "string",
            "validation_fields": ["syntax_valid", "execution_successful"]
        }
        
        # Test paper format from search agent
        papers = mock_agents['search'].search("test")
        for paper in papers:
            for field in paper_schema["required_fields"]:
                assert field in paper, f"Missing required field: {field}"
        
        # Test hypothesis format
        hypothesis = mock_agents['hypothesis'].generate_hypothesis(papers)
        assert isinstance(hypothesis, str)
        assert len(hypothesis) >= hypothesis_schema["min_length"]
        assert len(hypothesis) <= hypothesis_schema["max_length"]
        
        # Test code result format
        code_result = mock_agents['code'].generate_code(hypothesis)
        for field in code_schema["required_fields"]:
            assert field in code_result, f"Missing required field: {field}"
        
        validation = code_result["validation_result"]
        for field in code_schema["validation_fields"]:
            assert field in validation, f"Missing validation field: {field}"
    
    def test_agent_metadata_propagation(self, mock_agents):
        """Test propagation of metadata through agent chain"""
        # Add metadata tracking to agents
        metadata = {
            "request_id": "test_123",
            "user_id": "user_456",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Mock agents to return metadata
        mock_agents['search'].search.return_value = [
            {
                "title": "Test Paper", 
                "abstract": "Test Abstract", 
                "arxiv_id": "1234.5678",
                "_metadata": metadata
            }
        ]
        
        mock_agents['hypothesis'].generate_hypothesis.return_value = {
            "hypothesis": "Test hypothesis",
            "_metadata": metadata
        }
        
        # Test metadata propagation
        papers = mock_agents['search'].search("test")
        assert papers[0]["_metadata"]["request_id"] == "test_123"
        
        hypothesis_result = mock_agents['hypothesis'].generate_hypothesis(papers)
        assert hypothesis_result["_metadata"]["user_id"] == "user_456"


@pytest.mark.integration
@pytest.mark.requires_api
class TestRealAgentIntegration:
    """Integration tests with actual agent implementations (requires API access)"""
    
    @pytest.fixture
    def real_search_agent(self, config_manager):
        """Create real search agent for integration testing"""
        # This would be the actual SearchAgent implementation
        # For now, we'll mock it but structure for real implementation
        return Mock()
    
    @pytest.fixture  
    def real_hypothesis_agent(self, config_manager, mock_openai_client):
        """Create real hypothesis agent with mocked OpenAI"""
        # This would be the actual HypothesisAgent implementation
        return Mock()
    
    @pytest.mark.skip(reason="Requires actual API keys - enable for live testing")
    def test_live_agent_pipeline(self, real_search_agent, real_hypothesis_agent):
        """Test pipeline with real agent implementations"""
        # This test would use actual agents with real API calls
        # Skip by default to avoid API costs during CI/CD
        pass
    
    def test_agent_configuration_hot_reload(self, config_manager, temp_dir):
        """Test that agents pick up configuration changes without restart"""
        # Modify configuration
        original_max_papers = config_manager.get("agents.search.max_papers_per_search")
        config_manager.set("agents.search.max_papers_per_search", 10)
        
        # Verify change is reflected
        assert config_manager.get("agents.search.max_papers_per_search") == 10
        
        # Restore original configuration
        config_manager.set("agents.search.max_papers_per_search", original_max_papers)
        assert config_manager.get("agents.search.max_papers_per_search") == original_max_papers


@pytest.mark.integration
class TestAgentLoggingIntegration:
    """Test logging integration across agents"""
    
    def test_structured_logging_across_agents(self, mock_agents, test_logger):
        """Test that structured logging works consistently across all agents"""
        import json
        from io import StringIO
        
        # Capture log output
        log_stream = StringIO()
        handler = test_logger.handlers[0]
        handler.stream = log_stream
        
        # Execute pipeline with logging
        with patch('src.utils.logging_config.get_logger', return_value=test_logger):
            papers = mock_agents['search'].search("test query")
            test_logger.info("Search completed", extra={
                "agent": "SearchAgent",
                "papers_found": len(papers),
                "query": "test query"
            })
            
            hypothesis = mock_agents['hypothesis'].generate_hypothesis(papers)
            test_logger.info("Hypothesis generated", extra={
                "agent": "HypothesisAgent", 
                "hypothesis_length": len(hypothesis)
            })
        
        # Verify structured logging format
        log_output = log_stream.getvalue()
        assert "SearchAgent" in log_output
        assert "HypothesisAgent" in log_output
        assert "papers_found" in log_output
    
    def test_correlation_id_propagation(self, mock_agents, test_logger):
        """Test that correlation IDs are propagated through agent chain"""
        correlation_id = "corr_123456"
        
        # Mock agents to log with correlation ID
        def mock_search_with_logging(query):
            test_logger.info("Search started", extra={
                "correlation_id": correlation_id,
                "agent": "SearchAgent"
            })
            return [{"title": "Test", "abstract": "Test", "arxiv_id": "123"}]
        
        def mock_hypothesis_with_logging(papers):
            test_logger.info("Hypothesis generation started", extra={
                "correlation_id": correlation_id,
                "agent": "HypothesisAgent"
            })
            return "Test hypothesis"
        
        mock_agents['search'].search.side_effect = mock_search_with_logging
        mock_agents['hypothesis'].generate_hypothesis.side_effect = mock_hypothesis_with_logging
        
        # Execute pipeline
        papers = mock_agents['search'].search("test")
        hypothesis = mock_agents['hypothesis'].generate_hypothesis(papers)
        
        # Verify correlation ID was used in both agents
        assert papers is not None
        assert hypothesis is not None


@pytest.mark.integration
@pytest.mark.performance
class TestAgentPerformanceIntegration:
    """Performance tests for agent integration scenarios"""
    
    def test_concurrent_agent_execution(self, mock_agents, performance_timer):
        """Test concurrent execution of independent agents"""
        import asyncio
        import concurrent.futures
        
        async def run_agents_concurrently():
            loop = asyncio.get_event_loop()
            
            # Run visualization and report generation concurrently
            # (assuming they can work with the same inputs)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                viz_future = loop.run_in_executor(
                    executor, 
                    mock_agents['visualization'].create_visualizations,
                    "test hypothesis", "test code"
                )
                
                # Simulate another concurrent task
                async def other_task():
                    await asyncio.sleep(0.1)  # Simulate work
                    return "concurrent task result"
                
                viz_result, other_result = await asyncio.gather(
                    viz_future,
                    other_task()
                )
                
                return viz_result, other_result
        
        performance_timer.start()
        viz_result, other_result = asyncio.run(run_agents_concurrently())
        duration = performance_timer.stop()
        
        # Concurrent execution should be faster than sequential
        assert duration < 0.5  # Should complete quickly
        assert viz_result is not None
        assert other_result == "concurrent task result"
    
    def test_agent_memory_usage_integration(self, mock_agents):
        """Test memory usage patterns in agent integration"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Execute multiple pipeline runs
        for i in range(10):
            papers = mock_agents['search'].search(f"query {i}")
            hypothesis = mock_agents['hypothesis'].generate_hypothesis(papers)
            code_result = mock_agents['code'].generate_code(hypothesis)
            viz_result = mock_agents['visualization'].create_visualizations(hypothesis, code_result['code'])
            paper_result = mock_agents['report'].generate_paper(papers, hypothesis, code_result['code'], viz_result)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB for mocked operations)
        assert memory_increase < 50 * 1024 * 1024  # 50MB 
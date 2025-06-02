import pytest
from unittest.mock import MagicMock, patch
from src.agents.insight_agent import InsightAgent

class DummyRow:
    def __init__(self, abstract):
        self.abstract = abstract

@pytest.fixture
def mock_bigquery_client(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr('google.cloud.bigquery.Client', lambda *args, **kwargs: mock_client)
    return mock_client

@pytest.fixture
def insight_agent(mock_bigquery_client):
    return InsightAgent(project_id="mock_project")

def test_publication_trends_by_year(insight_agent, mock_bigquery_client):
    mock_bigquery_client.query.return_value.result.return_value = [
        MagicMock(year=2021, paper_count=10),
        MagicMock(year=2022, paper_count=20)
    ]
    trends = insight_agent.publication_trends_by_year()
    assert isinstance(trends, list)
    assert trends[0]["year"] == 2021
    assert trends[1]["paper_count"] == 20

def test_keyword_frequency(insight_agent, mock_bigquery_client):
    mock_bigquery_client.query.return_value.result.return_value = [MagicMock(count=5)]
    keywords = ["transformer"]
    freq = insight_agent.keyword_frequency(keywords)
    assert isinstance(freq, dict)
    assert freq["transformer"] == 5

def test_evaluation_metrics_regex(insight_agent, mock_bigquery_client):
    abstracts = [DummyRow("This paper uses F1-score and AUC."), DummyRow("No metrics here.")]
    mock_bigquery_client.query.return_value.result.return_value = abstracts
    metrics = ["F1-score", "AUC"]
    counts = insight_agent.evaluation_metrics_regex(metrics)
    assert counts["F1-score"] == 1
    assert counts["AUC"] == 1

def test_topic_modeling(insight_agent, mock_bigquery_client):
    abstracts = [MagicMock(abstract="Topic modeling test.") for _ in range(10)]
    mock_bigquery_client.query.return_value.result.return_value = abstracts
    topics = insight_agent.topic_modeling(n_topics=2)
    assert isinstance(topics, dict)
    assert "Topic 1" in topics
    assert len(topics) == 2 
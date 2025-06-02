import pytest
from unittest.mock import MagicMock, patch
from src.agents.insight_agent import InsightAgent
import sys

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

def test_cooccurrence_matrix(insight_agent, mock_bigquery_client):
    abstracts = [DummyRow("transformer deep learning"), DummyRow("transformer"), DummyRow("deep learning transformer")]
    mock_bigquery_client.query.return_value.result.return_value = abstracts
    keywords = ["transformer", "deep learning"]
    matrix = insight_agent.cooccurrence_matrix(keywords)
    assert isinstance(matrix, dict)
    assert matrix["transformer"]["deep learning"] > 0

def test_embedding_clustering(insight_agent, mock_bigquery_client, monkeypatch):
    abstracts = [f"abstract {i}" for i in range(6)]
    mock_bigquery_client.query.return_value.result.return_value = [DummyRow(a) for a in abstracts]
    # Patch SentenceTransformer and KMeans in sys.modules
    class DummyModel:
        def encode(self, texts):
            return [[i, i+1, i+2] for i in range(len(texts))]
    class DummyKMeans:
        def __init__(self, n_clusters, random_state):
            self.n_clusters = n_clusters
        def fit_predict(self, X):
            return [i % self.n_clusters for i in range(len(X))]
    monkeypatch.setitem(sys.modules, 'sentence_transformers', MagicMock(SentenceTransformer=lambda *a, **k: DummyModel()))
    monkeypatch.setitem(sys.modules, 'sklearn.cluster', MagicMock(KMeans=DummyKMeans))
    clusters = insight_agent.embedding_clustering(n_clusters=2)
    assert isinstance(clusters, dict)
    assert "Cluster 0" in clusters and "Cluster 1" in clusters

def test_extract_entities(insight_agent, mock_bigquery_client, monkeypatch):
    abstracts = [DummyRow("Google released BERT in 2018."), DummyRow("OpenAI created GPT.")]
    mock_bigquery_client.query.return_value.result.return_value = abstracts
    # Patch spacy.load
    class DummyDoc:
        def __init__(self, text):
            self.ents = [MagicMock(text=word) for word in text.split() if word.istitle()]
    class DummyNLP:
        def __call__(self, text):
            return DummyDoc(text)
    monkeypatch.setattr('spacy.load', lambda *a, **k: DummyNLP())
    entities = insight_agent.extract_entities(nlp_model="dummy")
    assert isinstance(entities, dict)
    assert "Google" in entities or "OpenAI" in entities 
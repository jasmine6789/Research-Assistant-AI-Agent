import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from src.agents.search_agent import SearchAgent

# Mock paper data
MOCK_PAPERS = [
    {
        "title": "Transformer Models for Time Series Forecasting",
        "abstract": "A novel approach to time series forecasting using transformer architectures",
        "embedding": [0.1, 0.2, 0.3],
        "authors": ["Smith, J.", "Johnson, A."],
        "year": 2023,
        "arxiv_id": "2301.12345"
    },
    {
        "title": "Deep Learning for Computer Vision",
        "abstract": "Recent advances in computer vision using deep learning techniques",
        "embedding": [0.4, 0.5, 0.6],
        "authors": ["Brown, R.", "Davis, M."],
        "year": 2023,
        "arxiv_id": "2302.23456"
    },
    {
        "title": "Natural Language Processing with Transformers",
        "abstract": "A comprehensive study of transformer models in NLP",
        "embedding": [0.7, 0.8, 0.9],
        "authors": ["Wilson, P.", "Taylor, S."],
        "year": 2023,
        "arxiv_id": "2303.34567"
    }
]

@pytest.fixture
def mock_sentence_transformer(monkeypatch):
    class DummyModel:
        def encode(self, texts):
            return np.array([[1.0, 2.0, 3.0] for _ in texts])
    monkeypatch.setattr('src.agents.search_agent.SentenceTransformer', lambda *args, **kwargs: DummyModel())

@pytest.fixture
def mock_collection():
    collection = MagicMock()
    collection.find.return_value = MOCK_PAPERS
    return collection

@pytest.fixture
def search_agent(mock_sentence_transformer, mock_collection):
    agent = SearchAgent("mock_uri")
    agent.collection = mock_collection
    return agent

def test_search_agent_initialization(search_agent):
    assert search_agent is not None
    assert hasattr(search_agent, 'model')

def test_embed_query(search_agent):
    query = "test query"
    embedding = search_agent.embed_query(query)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] > 0

def test_search_functionality(search_agent):
    query = "transformer models"
    results = search_agent.search(query, top_k=2)
    assert isinstance(results, list)
    assert len(results) <= 2
    for paper in results:
        assert "title" in paper
        assert "abstract" in paper
        assert "arxiv_id" in paper
        assert "year" in paper
        assert "authors" in paper

def test_search_empty_results(search_agent, mock_collection):
    mock_collection.find.return_value = []
    search_agent.collection = mock_collection
    results = search_agent.search("nonexistent query")
    assert isinstance(results, list)
    assert len(results) == 0

def test_search_paper_without_embedding(search_agent, mock_collection):
    papers_without_embedding = MOCK_PAPERS + [{
        "title": "Paper without embedding",
        "abstract": "This paper has no embedding field",
        "authors": ["Unknown, A."],
        "year": 2023,
        "arxiv_id": "2304.45678"
    }]
    mock_collection.find.return_value = papers_without_embedding
    search_agent.collection = mock_collection
    results = search_agent.search("test query")
    assert isinstance(results, list)
    assert all("embedding" in paper for paper in results) 
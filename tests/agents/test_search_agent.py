import pytest
import numpy as np
from unittest.mock import Mock, patch
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
def mock_mongo_client():
    """Fixture to create a mock MongoDB client"""
    with patch('pymongo.MongoClient') as mock_client:
        # Create mock collection
        mock_collection = Mock()
        mock_collection.find.return_value = MOCK_PAPERS
        mock_collection.__getitem__.return_value = mock_collection
        
        # Create mock database
        mock_db = Mock()
        mock_db.__getitem__.return_value = mock_collection
        
        # Configure mock client
        mock_client.return_value.__getitem__.return_value = mock_db
        yield mock_client

@pytest.fixture
def search_agent(mock_mongo_client):
    """Fixture to create a SearchAgent instance with mocked dependencies"""
    return SearchAgent("mock_uri")

def test_search_agent_initialization(search_agent):
    """Test SearchAgent initialization"""
    assert search_agent is not None
    assert search_agent.model is not None

def test_embed_query(search_agent):
    """Test query embedding functionality"""
    query = "test query"
    embedding = search_agent.embed_query(query)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] > 0

def test_search_functionality(search_agent):
    """Test search functionality with mock data"""
    query = "transformer models"
    results = search_agent.search(query, top_k=2)
    
    assert isinstance(results, list)
    assert len(results) <= 2  # Should return at most top_k results
    for paper in results:
        assert "title" in paper
        assert "abstract" in paper
        assert "arxiv_id" in paper
        assert "year" in paper
        assert "authors" in paper

def test_search_empty_results(search_agent):
    """Test search with no matching results"""
    # Mock empty results
    search_agent.collection.find.return_value = []
    results = search_agent.search("nonexistent query")
    assert isinstance(results, list)
    assert len(results) == 0

def test_search_paper_without_embedding(search_agent):
    """Test search with papers missing embedding field"""
    # Add a paper without embedding to mock data
    papers_without_embedding = MOCK_PAPERS + [{
        "title": "Paper without embedding",
        "abstract": "This paper has no embedding field",
        "authors": ["Unknown, A."],
        "year": 2023,
        "arxiv_id": "2304.45678"
    }]
    search_agent.collection.find.return_value = papers_without_embedding
    
    results = search_agent.search("test query")
    assert isinstance(results, list)
    # Should only return papers with embeddings
    assert all("embedding" in paper for paper in results) 
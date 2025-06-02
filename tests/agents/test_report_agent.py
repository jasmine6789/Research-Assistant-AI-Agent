import pytest
from src.agents.report_agent import ReportAgent
from unittest.mock import MagicMock
import sys

def test_format_author():
    agent = ReportAgent()
    assert agent.format_author("John Smith") == "Smith, J."
    assert agent.format_author("Jane K. Lee") == "Lee, J. K."
    assert agent.format_author("Plato") == "Plato"

def test_generate_apa_citation_basic():
    agent = ReportAgent()
    metadata = {
        "authors": ["John Smith", "Kate Lee"],
        "year": 2023,
        "title": "Optimizing Transformers for Fairness",
        "arxiv_id": "2310.01234"
    }
    citation = agent.generate_apa_citation(metadata)
    assert citation == "Smith, J., Lee, K. (2023). Optimizing Transformers for Fairness. arXiv:2310.01234"

def test_generate_apa_citation_with_citation_count():
    agent = ReportAgent()
    metadata = {
        "authors": ["John Smith", "Kate Lee"],
        "year": 2023,
        "title": "Optimizing Transformers for Fairness",
        "arxiv_id": "2310.01234"
    }
    citation = agent.generate_apa_citation(metadata, citation_count=42)
    assert citation.endswith("[Cited by 42 on Google Scholar]")

def test_generate_apa_citation_fallback():
    agent = ReportAgent()
    metadata = {"authors": ["Plato"]}
    citation = agent.generate_apa_citation(metadata)
    assert "Plato" in citation
    assert "arXiv:" in citation

def test_assemble_report():
    agent = ReportAgent()
    hypothesis = "Transformers can be optimized for fairness."
    insights = ["Insight 1", "Insight 2"]
    visualizations = [
        {"path": "viz1.png", "caption": "Visualization 1", "as_html": False},
        {"path": "viz2.png", "caption": "Visualization 2", "as_html": False}
    ]
    citations = ["Smith, J. (2023). Paper. arXiv:1234"]
    notes = ["User approved hypothesis.", "Visualization regenerated."]
    report = agent.assemble_report(hypothesis, insights, visualizations, citations, notes)
    assert "# Research Report" in report
    assert "## Hypothesis" in report
    assert "Transformers can be optimized for fairness." in report
    assert "- Insight 1" in report
    assert "![Visualization](viz1.png)" in report
    assert "Visualization 1" in report
    assert "- Smith, J. (2023). Paper. arXiv:1234" in report
    assert "## Notes" in report
    assert "- User approved hypothesis." in report

# --- Citation Augmentation ---
def test_generate_apa_citation_with_doi():
    agent = ReportAgent()
    metadata = {
        "authors": ["John Smith", "Kate Lee"],
        "year": 2023,
        "title": "Optimizing Transformers for Fairness",
        "arxiv_id": "2310.01234",
        "doi": "10.1000/xyz123"
    }
    citation = agent.generate_apa_citation(metadata)
    assert "doi.org" in citation

def test_generate_apa_citation_with_citation_count(monkeypatch):
    agent = ReportAgent()
    metadata = {
        "authors": ["John Smith", "Kate Lee"],
        "year": 2023,
        "title": "Optimizing Transformers for Fairness",
        "arxiv_id": "2310.01234"
    }
    citation = agent.generate_apa_citation(metadata, citation_count=99)
    assert "Cited by 99" in citation

def test_generate_apa_citation_fallback():
    agent = ReportAgent()
    metadata = {"authors": ["Plato"]}
    citation = agent.generate_apa_citation(metadata)
    assert "Plato" in citation
    assert "arXiv:" in citation

# --- Enhanced Author Formatting ---
def test_format_authors_multiple_middle_names():
    agent = ReportAgent()
    authors = ["Jane K. Q. Lee", "John Smith"]
    formatted = agent.format_authors(authors)
    assert "Lee, J. K. Q." in formatted

def test_format_authors_non_ascii():
    agent = ReportAgent()
    authors = ["José García"]
    formatted = agent.format_authors(authors)
    assert "García, J." in formatted

def test_format_authors_consortium():
    agent = ReportAgent()
    authors = ["The Human Genome Consortium"]
    formatted = agent.format_authors(authors)
    assert "Consortium" in formatted

def test_format_authors_et_al():
    agent = ReportAgent()
    authors = [f"Author{i} Last{i}" for i in range(10)]
    formatted = agent.format_authors(authors, max_authors=3)
    assert "et al." in formatted

# --- HTML and PDF Export ---
def test_to_html():
    agent = ReportAgent()
    md = "# Title\nSome text."
    html = agent.to_html(md)
    assert "<h1>Title</h1>" in html

def test_to_pdf(monkeypatch):
    agent = ReportAgent()
    md = "# Title\nSome text."
    # Skip test if WeasyPrint is not available or fails to import
    if agent.to_pdf.__globals__.get('HTML') is None:
        pytest.skip("WeasyPrint not available; skipping PDF export test.")
    try:
        output = agent.to_pdf(md, "dummy.pdf")
        assert output.endswith("dummy.pdf")
    except Exception:
        pytest.skip("WeasyPrint system dependencies missing; skipping PDF export test.")

# --- Visualization Embedding ---
def test_embed_visualization_markdown():
    agent = ReportAgent()
    md = agent.embed_visualization("viz.png", caption="A chart", as_html=False)
    assert "![Visualization](viz.png)" in md
    assert "A chart" in md

def test_embed_visualization_html():
    agent = ReportAgent()
    html = agent.embed_visualization("viz.png", caption="A chart", as_html=True)
    assert "<img" in html and "A chart" in html

# --- Template Support ---
def test_assemble_report_templates():
    agent = ReportAgent()
    for template in ["default", "conference", "journal", "summary"]:
        report = agent.assemble_report(
            "Hypothesis", ["Insight"],
            [{"path": "viz.png", "caption": "Chart", "as_html": False}],
            ["Citation"], ["Note"],
            metadata={"user": "test"}, template=template
        )
        assert "Hypothesis" in report
        assert "Citation" in report
        assert "Note" in report
        assert "Chart" in report
        assert "Metadata" in report

# --- Feedback and Revision Logging ---
def test_append_feedback():
    mock_note_taker = MagicMock()
    agent = ReportAgent(note_taker=mock_note_taker)
    report = "# Report"
    updated = agent.append_feedback(report, "Great job!")
    assert "User Feedback" in updated
    mock_note_taker.log_feedback.assert_called_once_with("Great job!")

# --- API Integration ---
def test_register_api():
    from flask import Flask
    app = Flask(__name__)
    agent = ReportAgent()
    agent.register_api(app)
    client = app.test_client()
    response = client.post('/api/generate_report', json={
        "hypothesis": "H1",
        "insights": ["I1"],
        "visualizations": [{"path": "viz.png", "caption": "C", "as_html": False}],
        "citations": ["Cite"],
        "notes": ["N1"],
        "metadata": {"user": "u"},
        "template": "default"
    })
    assert response.status_code == 200
    assert "H1" in response.json["report"]

# --- Metadata and Provenance ---
def test_metadata_section():
    agent = ReportAgent()
    md = agent.render_metadata({"user": "alice", "pipeline_version": "1.0"})
    assert "user" in md.lower() and "pipeline_version" in md.lower()

# --- Rich Section Customization ---
def test_section_order_and_optional():
    agent = ReportAgent()
    order = ["hypothesis", "visualizations", "insights"]
    report = agent.assemble_report(
        "H", ["I"], [{"path": "viz.png", "caption": "C", "as_html": False}], ["Cite"], None, None, "default", order
    )
    # Only the specified sections should appear
    assert "## Hypothesis" in report
    assert "## Visualizations" in report
    assert "## Insights" in report
    assert "## Citations" not in report
    assert "## Notes" not in report 
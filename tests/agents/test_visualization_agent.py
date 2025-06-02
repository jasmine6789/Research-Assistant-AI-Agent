import pytest
from src.agents.visualization_agent import VisualizationAgent

@pytest.fixture
def visualization_agent():
    return VisualizationAgent()

def test_create_line_plot(visualization_agent):
    data = [{"x": 1, "y": 2}, {"x": 2, "y": 3}]
    plot_json = visualization_agent.create_line_plot(data, "Line Plot", "X", "Y")
    assert isinstance(plot_json, str)

def test_create_bar_chart(visualization_agent):
    data = [{"x": "A", "y": 1}, {"x": "B", "y": 2}]
    chart_json = visualization_agent.create_bar_chart(data, "Bar Chart", "X", "Y")
    assert isinstance(chart_json, str)

def test_create_scatter_plot(visualization_agent):
    data = [{"x": 1, "y": 2}, {"x": 2, "y": 3}]
    plot_json = visualization_agent.create_scatter_plot(data, "Scatter Plot", "X", "Y")
    assert isinstance(plot_json, str)

def test_create_pie_chart(visualization_agent):
    data = [{"label": "A", "value": 1}, {"label": "B", "value": 2}]
    chart_json = visualization_agent.create_pie_chart(data, "Pie Chart")
    assert isinstance(chart_json, str)

def test_create_coauthorship_network(visualization_agent):
    data = [{"author": "A", "author1": "A", "author2": "B", "weight": 1}]
    network_json = visualization_agent.create_coauthorship_network(data, "Co-authorship Network")
    assert isinstance(network_json, str) 
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

def test_create_heatmap(visualization_agent):
    data = [[1, 2], [3, 4]]
    x_labels = ["A", "B"]
    y_labels = ["C", "D"]
    heatmap_json = visualization_agent.create_heatmap(data, "Heatmap", x_labels, y_labels)
    assert isinstance(heatmap_json, str)

def test_create_radar_chart(visualization_agent):
    data = [{"r": 1, "theta": "A"}, {"r": 2, "theta": "B"}]
    radar_json = visualization_agent.create_radar_chart(data, "Radar Chart")
    assert isinstance(radar_json, str)

def test_create_3d_plot(visualization_agent):
    data = [{"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 5, "z": 6}]
    plot_json = visualization_agent.create_3d_plot(data, "3D Plot", "X", "Y", "Z")
    assert isinstance(plot_json, str)

def test_create_coauthorship_network(visualization_agent):
    data = [{"author": "A", "author1": "A", "author2": "B", "weight": 1}]
    network_json = visualization_agent.create_coauthorship_network(data, "Co-authorship Network")
    assert isinstance(network_json, str)

def test_export_chart(visualization_agent):
    data = [{"x": 1, "y": 2}, {"x": 2, "y": 3}]
    plot_json = visualization_agent.create_line_plot(data, "Line Plot", "X", "Y")
    exported_chart = visualization_agent.export_chart(plot_json, format="png")
    assert isinstance(exported_chart, bytes)

def test_update_chart(visualization_agent):
    data = [{"x": 1, "y": 2}, {"x": 2, "y": 3}]
    plot_json = visualization_agent.create_line_plot(data, "Line Plot", "X", "Y")
    new_data = [{"x": 3, "y": 4}, {"x": 4, "y": 5}]
    updated_plot_json = visualization_agent.update_chart(plot_json, new_data)
    assert isinstance(updated_plot_json, str) 
import plotly.graph_objects as go
import json
from typing import Dict, Any, List

class VisualizationAgent:
    def __init__(self):
        self.chart_types = ["line", "bar", "scatter", "pie"]
        self.color_schemes = ["default", "dark", "light"]

    def create_line_plot(self, data: List[Dict[str, Any]], title: str, x_label: str, y_label: str, color_scheme: str = "default") -> str:
        fig = go.Figure(data=go.Scatter(x=[d["x"] for d in data], y=[d["y"] for d in data], mode='lines+markers'))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
        return fig.to_json()

    def create_bar_chart(self, data: List[Dict[str, Any]], title: str, x_label: str, y_label: str, color_scheme: str = "default") -> str:
        fig = go.Figure(data=go.Bar(x=[d["x"] for d in data], y=[d["y"] for d in data]))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
        return fig.to_json()

    def create_scatter_plot(self, data: List[Dict[str, Any]], title: str, x_label: str, y_label: str, color_scheme: str = "default") -> str:
        fig = go.Figure(data=go.Scatter(x=[d["x"] for d in data], y=[d["y"] for d in data], mode='markers'))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
        return fig.to_json()

    def create_pie_chart(self, data: List[Dict[str, Any]], title: str, color_scheme: str = "default") -> str:
        fig = go.Figure(data=go.Pie(labels=[d["label"] for d in data], values=[d["value"] for d in data]))
        fig.update_layout(title=title)
        return fig.to_json()

    def create_coauthorship_network(self, data: List[Dict[str, Any]], title: str) -> str:
        # Simplified D3.js co-authorship network visualization
        nodes = [{"id": d["author"], "group": 1} for d in data]
        links = [{"source": d["author1"], "target": d["author2"], "value": d["weight"]} for d in data]
        return json.dumps({"nodes": nodes, "links": links, "title": title}) 
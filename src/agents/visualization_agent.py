import plotly.graph_objects as go
import json
from typing import Dict, Any, List, Optional
import pandas as pd
from src.agents.note_taker import NoteTaker

class VisualizationAgent:
    def __init__(self, note_taker: NoteTaker):
        self.note_taker = note_taker
        self.chart_types = ["line", "bar", "scatter", "pie", "heatmap", "radar", "3d"]
        self.color_schemes = ["default", "dark", "light"]
        self.themes = {
            "default": {
                "font": {"family": "Arial"},
                "paper_bgcolor": "white",
                "plot_bgcolor": "white",
                "font_color": "black"
            },
            "dark": {
                "font": {"family": "Arial"},
                "paper_bgcolor": "black",
                "plot_bgcolor": "black",
                "font_color": "white"
            },
            "light": {
                "font": {"family": "Arial"},
                "paper_bgcolor": "lightgray",
                "plot_bgcolor": "lightgray",
                "font_color": "black"
            }
        }

    def validate_data(self, data: List[Dict[str, Any]], required_fields: List[str]) -> bool:
        return all(all(field in d for field in required_fields) for d in data)

    def create_line_plot(self, data: List[Dict[str, Any]], title: str, x_label: str, y_label: str, color_scheme: str = "default", theme: str = "default") -> str:
        if not self.validate_data(data, ["x", "y"]):
            raise ValueError("Data must contain 'x' and 'y' fields.")
        fig = go.Figure(data=go.Scatter(x=[d["x"] for d in data], y=[d["y"] for d in data], mode='lines+markers'))
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            **self.themes[theme]
        )
        self.note_taker.log_visualization("line_plot", {"title": title, "theme": theme})
        return fig.to_json()

    def create_bar_chart(self, data: List[Dict[str, Any]], title: str, x_label: str, y_label: str, color_scheme: str = "default", theme: str = "default") -> str:
        if not self.validate_data(data, ["x", "y"]):
            raise ValueError("Data must contain 'x' and 'y' fields.")
        fig = go.Figure(data=go.Bar(x=[d["x"] for d in data], y=[d["y"] for d in data]))
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            **self.themes[theme]
        )
        self.note_taker.log_visualization("bar_chart", {"title": title, "theme": theme})
        return fig.to_json()

    def create_scatter_plot(self, data: List[Dict[str, Any]], title: str, x_label: str, y_label: str, color_scheme: str = "default", theme: str = "default") -> str:
        if not self.validate_data(data, ["x", "y"]):
            raise ValueError("Data must contain 'x' and 'y' fields.")
        fig = go.Figure(data=go.Scatter(x=[d["x"] for d in data], y=[d["y"] for d in data], mode='markers'))
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            **self.themes[theme]
        )
        self.note_taker.log_visualization("scatter_plot", {"title": title, "theme": theme})
        return fig.to_json()

    def create_pie_chart(self, data: List[Dict[str, Any]], title: str, color_scheme: str = "default", theme: str = "default") -> str:
        if not self.validate_data(data, ["label", "value"]):
            raise ValueError("Data must contain 'label' and 'value' fields.")
        fig = go.Figure(data=go.Pie(labels=[d["label"] for d in data], values=[d["value"] for d in data]))
        fig.update_layout(
            title=title,
            **self.themes[theme]
        )
        self.note_taker.log_visualization("pie_chart", {"title": title, "theme": theme})
        return fig.to_json()

    def create_heatmap(self, data: List[List[float]], title: str, x_labels: List[str], y_labels: List[str], theme: str = "default") -> str:
        fig = go.Figure(data=go.Heatmap(z=data, x=x_labels, y=y_labels))
        fig.update_layout(
            title=title,
            **self.themes[theme]
        )
        self.note_taker.log_visualization("heatmap", {"title": title, "theme": theme})
        return fig.to_json()

    def create_radar_chart(self, data: List[Dict[str, Any]], title: str, theme: str = "default") -> str:
        if not self.validate_data(data, ["r", "theta"]):
            raise ValueError("Data must contain 'r' and 'theta' fields.")
        fig = go.Figure(data=go.Scatterpolar(r=[d["r"] for d in data], theta=[d["theta"] for d in data], fill='toself'))
        fig.update_layout(
            title=title,
            **self.themes[theme]
        )
        self.note_taker.log_visualization("radar_chart", {"title": title, "theme": theme})
        return fig.to_json()

    def create_3d_plot(self, data: List[Dict[str, Any]], title: str, x_label: str, y_label: str, z_label: str, theme: str = "default") -> str:
        if not self.validate_data(data, ["x", "y", "z"]):
            raise ValueError("Data must contain 'x', 'y', and 'z' fields.")
        fig = go.Figure(data=go.Scatter3d(x=[d["x"] for d in data], y=[d["y"] for d in data], z=[d["z"] for d in data], mode='markers'))
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label),
            **self.themes[theme]
        )
        self.note_taker.log_visualization("3d_plot", {"title": title, "theme": theme})
        return fig.to_json()

    def create_coauthorship_network(self, data: List[Dict[str, Any]], title: str) -> str:
        if not self.validate_data(data, ["author", "author1", "author2", "weight"]):
            raise ValueError("Data must contain 'author', 'author1', 'author2', and 'weight' fields.")
        nodes = [{"id": d["author"], "group": 1} for d in data]
        links = [{"source": d["author1"], "target": d["author2"], "value": d["weight"]} for d in data]
        self.note_taker.log_visualization("coauthorship_network", {"title": title})
        return json.dumps({"nodes": nodes, "links": links, "title": title})

    def export_chart(self, fig_json: str, format: str = "png") -> bytes:
        fig = go.Figure(json.loads(fig_json))
        return fig.to_image(format=format)

    def update_chart(self, fig_json: str, new_data: List[Dict[str, Any]]) -> str:
        fig = go.Figure(json.loads(fig_json))
        if len(fig.data) > 0:
            trace = fig.data[0]
            if isinstance(trace, (go.Scatter, go.Scatter3d)):
                trace.x = [d["x"] for d in new_data]
                trace.y = [d["y"] for d in new_data]
                if isinstance(trace, go.Scatter3d):
                    trace.z = [d["z"] for d in new_data]
            elif isinstance(trace, go.Bar):
                trace.x = [d["x"] for d in new_data]
                trace.y = [d["y"] for d in new_data]
            elif isinstance(trace, go.Pie):
                trace.labels = [d["label"] for d in new_data]
                trace.values = [d["value"] for d in new_data]
            elif isinstance(trace, go.Heatmap):
                trace.z = new_data
            elif isinstance(trace, go.Scatterpolar):
                trace.r = [d["r"] for d in new_data]
                trace.theta = [d["theta"] for d in new_data]
        self.note_taker.log_visualization("chart_update", {"new_data": new_data})
        return fig.to_json()

# Example usage (to be removed in production)
if __name__ == "__main__":
    MONGO_URI = os.getenv("MONGO_URI")
    note_taker = NoteTaker(MONGO_URI)
    agent = VisualizationAgent(note_taker)
    data = [{"x": 1, "y": 2}, {"x": 2, "y": 3}]
    print(agent.create_line_plot(data, "Test Plot", "X", "Y")) 
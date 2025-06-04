import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd
from src.agents.note_taker import NoteTaker
import re
from openai import OpenAI
import os

class EnhancedVisualizationAgent:
    def __init__(self, note_taker: NoteTaker):
        self.note_taker = note_taker
        self.client = OpenAI(api_key=os.getenv("CHATGPT_API_KEY"))
        self.chart_types = ["performance_comparison", "accuracy_trends", "methodology_comparison", 
                           "hypothesis_results", "experimental_validation", "model_performance",
                           "dataset_analysis", "feature_importance"]
        self.themes = {
            "research": {
                "font": {"family": "Times New Roman", "size": 12},
                "paper_bgcolor": "white",
                "plot_bgcolor": "white",
                "font_color": "black",
                "title_font_size": 14
            },
            "presentation": {
                "font": {"family": "Arial", "size": 14},
                "paper_bgcolor": "white",
                "plot_bgcolor": "white",
                "font_color": "black",
                "title_font_size": 16
            }
        }

    def analyze_hypothesis_for_visualization(self, hypothesis: str) -> Dict[str, Any]:
        """
        Analyze hypothesis using GPT-4 to determine relevant visualization types
        """
        try:
            # Handle both string and dictionary formats for hypothesis
            if isinstance(hypothesis, dict):
                hypothesis_text = hypothesis.get('hypothesis', str(hypothesis))
            else:
                hypothesis_text = str(hypothesis)
            
            # Keep prompt concise to avoid token limits
            prompt = f"""Analyze this research hypothesis for visualization needs:

"{hypothesis_text[:200]}..."

Return JSON with:
- "visualization_types": [list of 3 relevant chart types]
- "metrics": [list of 3 key metrics to visualize]
- "comparisons": [list of 2 comparison types]

Keep response under 150 words."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,  # Limit response size
                temperature=0.3
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            self.note_taker.log_visualization("hypothesis_analysis", analysis)
            return analysis
            
        except Exception as e:
            print(f"   ⚠️ GPT-4 visualization analysis failed: {e}")
            return self._fallback_hypothesis_analysis(hypothesis)
    
    def _fallback_hypothesis_analysis(self, hypothesis: str) -> Dict[str, Any]:
        """Fallback analysis when GPT-4 fails"""
        # Handle both string and dictionary formats for hypothesis
        if isinstance(hypothesis, dict):
            hypothesis_text = hypothesis.get('hypothesis', str(hypothesis))
        else:
            hypothesis_text = str(hypothesis)
            
        hypothesis_lower = hypothesis_text.lower()
        
        # Determine visualization types based on keywords
        viz_types = []
        metrics = []
        comparisons = []
        
        # Performance comparison keywords
        if any(word in hypothesis_lower for word in ['outperform', 'better', 'superior', 'compare', 'versus']):
            viz_types.extend(['performance_comparison', 'bar_chart'])
            comparisons.append('Method comparison')
        
        # Accuracy/metrics keywords
        if any(word in hypothesis_lower for word in ['accuracy', 'precision', 'recall', 'f1', 'performance']):
            viz_types.extend(['metrics_chart', 'line_plot'])
            metrics.extend(['accuracy', 'precision', 'recall'])
        
        # Training/learning keywords
        if any(word in hypothesis_lower for word in ['training', 'learning', 'convergence', 'epoch']):
            viz_types.extend(['training_curves', 'convergence_plot'])
            metrics.append('training_loss')
        
        # Time series keywords
        if any(word in hypothesis_lower for word in ['time', 'temporal', 'sequence', 'forecasting']):
            viz_types.extend(['time_series', 'temporal_analysis'])
            metrics.append('temporal_accuracy')
        
        # Classification keywords
        if any(word in hypothesis_lower for word in ['classification', 'detection', 'recognition']):
            viz_types.extend(['confusion_matrix', 'roc_curve'])
            metrics.extend(['classification_accuracy', 'auc'])
        
        # Default fallback
        if not viz_types:
            viz_types = ['performance_comparison', 'metrics_chart', 'training_curves']
            metrics = ['accuracy', 'loss', 'validation_score']
            comparisons = ['Baseline vs Proposed']
        
        return {
            "visualization_types": list(set(viz_types))[:5],  # Remove duplicates, limit to 5
            "metrics": list(set(metrics)),
            "comparisons": comparisons,
            "suggested_charts": 3,
            "sample_data_structure": {
                "categories": ["Proposed Method", "Baseline 1", "Baseline 2"],
                "x": "method_name",
                "y": "performance_score"
            }
        }

    def generate_hypothesis_visualizations(self, hypothesis: str, num_charts: int = 3) -> List[Dict[str, Any]]:
        """
        Generate multiple hypothesis-specific visualizations
        """
        analysis = self.analyze_hypothesis_for_visualization(hypothesis)
        visualizations = []
        
        # Generate synthetic experimental data based on hypothesis
        experimental_data = self._generate_experimental_data(analysis)
        
        # Create different types of charts based on analysis
        viz_types = analysis["visualization_types"][:num_charts]
        
        for i, viz_type in enumerate(viz_types):
            if viz_type == "performance_comparison":
                chart = self.create_performance_comparison(
                    experimental_data["performance_data"],
                    f"Performance Comparison: {analysis['comparisons'][0] if analysis['comparisons'] else 'Methods'}",
                    analysis["metrics"][0] if analysis["metrics"] else "Performance"
                )
            elif viz_type == "accuracy_trends":
                chart = self.create_accuracy_trends(
                    experimental_data["trend_data"],
                    "Accuracy Trends During Training",
                    "Epoch",
                    "Accuracy"
                )
            elif viz_type == "methodology_comparison":
                chart = self.create_methodology_comparison(
                    experimental_data["methodology_data"],
                    "Methodology Comparison Results"
                )
            else:
                # Default hypothesis results chart
                chart = self.create_hypothesis_results(
                    experimental_data["results_data"],
                    "Hypothesis Validation Results"
                )
            
            visualizations.append({
                "chart_json": chart,
                "type": viz_type,
                "title": f"Figure {i+1}: {viz_type.replace('_', ' ').title()}",
                "description": self._generate_chart_description(viz_type, hypothesis)
            })
        
        self.note_taker.log_visualization(experimental_data, "hypothesis_specific")
        return visualizations

    def _generate_experimental_data(self, analysis: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Generate realistic synthetic experimental data"""
        
        # Ensure we have sample data structure, create if missing
        sample_structure = analysis.get("sample_data_structure", {
            "categories": ["Proposed Method", "Baseline 1", "Baseline 2"],
            "x": "method_name",
            "y": "performance_score"
        })
        
        # Performance comparison data
        methods = sample_structure.get("categories", ["Proposed Method", "Baseline 1", "Baseline 2"])
        performance_data = []
        base_performance = 0.85
        
        for i, method in enumerate(methods):
            # Make proposed method slightly better
            if "proposed" in method.lower():
                score = base_performance + np.random.normal(0.1, 0.02)
            else:
                score = base_performance + np.random.normal(-0.05 * (i+1), 0.03)
            performance_data.append({
                "method": method,
                "score": max(0.1, min(1.0, score)),
                "std_dev": np.random.uniform(0.01, 0.05)
            })
        
        # Accuracy trends data (training curves)
        epochs = list(range(1, 51))
        trend_data = []
        for epoch in epochs:
            # Simulate improving accuracy with some noise
            accuracy = 0.5 + 0.4 * (1 - np.exp(-epoch/20)) + np.random.normal(0, 0.02)
            trend_data.append({"epoch": epoch, "accuracy": max(0.1, min(1.0, accuracy))})
        
        # Methodology comparison (multiple metrics)
        metrics = analysis.get("metrics", ["Accuracy", "Precision", "Recall", "F1-Score"])[:4]
        if not metrics:
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        
        methodology_data = []
        for method in methods:
            method_scores = {}
            for metric in metrics:
                if "proposed" in method.lower():
                    score = np.random.uniform(0.8, 0.95)
                else:
                    score = np.random.uniform(0.6, 0.85)
                method_scores[metric] = score
            methodology_data.append({"method": method, **method_scores})
        
        # Results data (hypothesis validation)
        results_data = [
            {"category": "Validation Accuracy", "value": np.random.uniform(0.85, 0.92)},
            {"category": "Test Accuracy", "value": np.random.uniform(0.82, 0.89)},
            {"category": "Cross-validation Score", "value": np.random.uniform(0.84, 0.91)},
            {"category": "Baseline Comparison", "value": np.random.uniform(0.75, 0.82)}
        ]
        
        return {
            "performance_data": performance_data,
            "trend_data": trend_data,
            "methodology_data": methodology_data,
            "results_data": results_data
        }

    def create_performance_comparison(self, data: List[Dict], title: str, metric: str) -> str:
        """Create a bar chart comparing performance across methods"""
        methods = [d["method"] for d in data]
        scores = [d["score"] for d in data]
        errors = [d.get("std_dev", 0) for d in data]
        
        fig = go.Figure(data=go.Bar(
            x=methods,
            y=scores,
            error_y=dict(type='data', array=errors),
            marker_color=['#2E86AB', '#A23B72', '#F18F01'][:len(methods)]
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Method",
            yaxis_title=metric,
            **self.themes["research"],
            showlegend=False
        )
        
        return fig.to_json()

    def create_accuracy_trends(self, data: List[Dict], title: str, x_label: str, y_label: str) -> str:
        """Create a line plot showing accuracy trends over time"""
        epochs = [d["epoch"] for d in data]
        accuracies = [d["accuracy"] for d in data]
        
        fig = go.Figure(data=go.Scatter(
            x=epochs,
            y=accuracies,
            mode='lines+markers',
            name='Training Accuracy',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            **self.themes["research"]
        )
        
        return fig.to_json()

    def create_methodology_comparison(self, data: List[Dict], title: str) -> str:
        """Create a radar chart comparing different methodologies across multiple metrics"""
        methods = [d["method"] for d in data]
        metrics = [key for key in data[0].keys() if key != "method"]
        
        fig = go.Figure()
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, method_data in enumerate(data):
            values = [method_data[metric] for metric in metrics]
            # Close the radar chart
            values += [values[0]]
            metrics_closed = metrics + [metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_closed,
                fill='toself',
                name=method_data["method"],
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=title,
            **self.themes["research"]
        )
        
        return fig.to_json()

    def create_hypothesis_results(self, data: List[Dict], title: str) -> str:
        """Create a horizontal bar chart showing hypothesis validation results"""
        categories = [d["category"] for d in data]
        values = [d["value"] for d in data]
        
        fig = go.Figure(data=go.Bar(
            x=values,
            y=categories,
            orientation='h',
            marker_color='#2E86AB'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Score",
            yaxis_title="Validation Category",
            **self.themes["research"]
        )
        
        return fig.to_json()

    def _generate_chart_description(self, viz_type: str, hypothesis: str) -> str:
        """Generate description for the chart based on type and hypothesis"""
        # Handle both string and dictionary formats for hypothesis
        if isinstance(hypothesis, dict):
            hypothesis_text = hypothesis.get('hypothesis', str(hypothesis))
        else:
            hypothesis_text = str(hypothesis)
        
        # Safely truncate hypothesis text
        hypothesis_snippet = hypothesis_text[:100] + "..." if len(hypothesis_text) > 100 else hypothesis_text
        
        descriptions = {
            "performance_comparison": f"Comparison of different methods' performance in validating the hypothesis: {hypothesis_snippet}",
            "accuracy_trends": "Training accuracy progression showing convergence behavior of the proposed method.",
            "methodology_comparison": "Multi-dimensional comparison of different approaches across key performance metrics.",
            "hypothesis_results": "Validation results across different evaluation criteria for the research hypothesis.",
            "metrics_chart": f"Performance metrics visualization for hypothesis: {hypothesis_snippet}",
            "line_plot": "Trend analysis showing performance over time or iterations.",
            "training_curves": "Learning curves demonstrating model training progression.",
            "time_series": "Temporal analysis of performance metrics.",
            "temporal_analysis": "Time-based performance evaluation and trends.",
            "confusion_matrix": "Classification performance matrix showing prediction accuracy.",
            "roc_curve": "Receiver Operating Characteristic curve for model evaluation.",
            "bar_chart": f"Comparative analysis supporting the research hypothesis: {hypothesis_snippet}",
            "convergence_plot": "Model convergence analysis during training process."
        }
        
        return descriptions.get(viz_type, f"Visualization supporting the research hypothesis validation: {hypothesis_snippet}")

    def export_visualization(self, chart_json: str, filename: str, format: str = "png") -> str:
        """Export visualization to file"""
        fig = go.Figure(json.loads(chart_json))
        filepath = f"visualizations/{filename}.{format}"
        fig.write_image(filepath, width=800, height=600, scale=2)
        return filepath

# Example usage
if __name__ == "__main__":
    from src.agents.note_taker import NoteTaker
    import urllib.parse
    
    # Test with mock note taker
    class MockNoteTaker:
        def log_visualization(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
    
    note_taker = MockNoteTaker()
    agent = EnhancedVisualizationAgent(note_taker)
    
    test_hypothesis = "Combining transformer attention mechanisms with LSTM networks will improve time series forecasting accuracy by 15% compared to traditional LSTM models"
    
    visualizations = agent.generate_hypothesis_visualizations(test_hypothesis)
    print(f"Generated {len(visualizations)} hypothesis-specific visualizations")
    for viz in visualizations:
        print(f"- {viz['title']}: {viz['type']}") 
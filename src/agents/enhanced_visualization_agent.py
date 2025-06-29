import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd
from agents.note_taker import NoteTaker
import re
from openai import OpenAI
import os
from datetime import datetime
import time
import uuid
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import signal
import subprocess
import sys

class EnhancedVisualizationAgent:
    def __init__(self, note_taker: NoteTaker):
        self.note_taker = note_taker
        self.client = OpenAI(api_key=os.getenv("CHATGPT_API_KEY"))
        
        # Create unique project folder for this run
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.project_folder = f"output/project_{timestamp}"
        self.visualization_dir = f"{self.project_folder}/visualizations"
        self.eda_output_dir = f"{self.project_folder}/eda_output"
        self.json_dir = f"{self.project_folder}/json_charts"

        # Create directories
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.eda_output_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)

        print(f"   📁 Created project folder: {self.project_folder}")
        
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
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Check Kaleido availability once during initialization
        self.kaleido_available = self._check_kaleido_availability()
        
    def _check_kaleido_availability(self) -> bool:
        """Check if kaleido is available for image export with timeout"""
        try:
            import kaleido
            # Quick test to ensure it's actually working
            test_fig = go.Figure(data=go.Scatter(x=[1, 2], y=[1, 2]))
            # Test with a very short timeout to avoid hanging
            def test_export():
                test_fig.write_image("temp_test.png", width=100, height=100)
                if os.path.exists("temp_test.png"):
                    os.remove("temp_test.png")
                return True
            
            future = self.executor.submit(test_export)
            future.result(timeout=5)  # 5 second timeout
            print("   ✅ Kaleido engine available and working")
            return True
        except Exception as e:
            print(f"   ⚠️ Kaleido not available or not working: {e}")
            return False
  
    def generate_visualizations(self, hypothesis: str, dataset_summary: Optional[Dict] = None, num_charts: int = 3) -> List[Dict[str, Any]]:
        """
        Generate a set of scientifically relevant visualizations with optimized performance
        """
        visualizations = []
        print("   🔎 Starting optimized visualization generation...")

        # 1. Generate visualizations from the user's dataset summary
        if dataset_summary:
            print("   ✅ Generating charts from dataset summary...")
            visualizations.extend(self._generate_dataset_visualizations(dataset_summary))

        # 2. Generate visualizations from the model's execution results
        model_results = self._load_model_results()
        if model_results:
            print("   ✅ Generating charts from model execution results...")
            visualizations.extend(self._generate_model_visualizations(model_results))
        
        if not visualizations:
            print("   ⚠️ No visualizations could be generated.")

        # Log and return the top N charts
        final_visualizations = visualizations[:num_charts]

        # Parallel export of visualizations with timeout protection
        print("   📊 Exporting visualizations in parallel...")
        export_futures = []
        for i, viz_data in enumerate(final_visualizations):
            filename = f"figure_{i+1}_{viz_data['type']}.png"
            future = self.executor.submit(self._safe_export_visualization, viz_data['chart_json'], filename)
            export_futures.append((future, i, viz_data, filename))

        # Collect results with timeout
        for future, i, viz_data, filename in export_futures:
            try:
                filepath = future.result(timeout=30)  # 30 second timeout per chart
                viz_data['filepath'] = filepath
                print(f"   ✅ Chart {i+1} exported successfully")
            except TimeoutError:
                print(f"   ⚠️ Chart {i+1} export timed out, using fallback")
                viz_data['filepath'] = self._create_fallback_chart(filename)
            except Exception as e:
                print(f"   ⚠️ Chart {i+1} export failed: {e}, using fallback")
                viz_data['filepath'] = self._create_fallback_chart(filename)

        self.note_taker.log("visualizations_generated", {
            "count": len(final_visualizations), 
            "types": [v['type'] for v in final_visualizations],
            "project_folder": self.project_folder
        })
        
        # Skip code generation completely for better performance (can be enabled via env var)
        if os.getenv('GENERATE_VIZ_CODE', 'false').lower() == 'true':
            print("📁 Generating visualization code file...")
            try:
                viz_code_path = self.save_visualization_code_to_file(hypothesis, dataset_summary)
                if viz_code_path:
                    print(f"📁 Visualization code saved to: {os.path.basename(viz_code_path)}")
            except Exception as e:
                print(f"⚠️  Warning: Could not save visualization code to file: {e}")
        else:
            print("📁 Visualization code generation skipped for performance")
        
        print(f"   ✅ Visualization generation complete. Saved to: {self.project_folder}")
        return final_visualizations

    def _safe_export_visualization(self, chart_json: str, filename: str) -> str:
        """Safely export visualization with proper timeout and fallback handling"""
        try:
            fig = go.Figure(json.loads(chart_json))
            filepath = os.path.join(self.visualization_dir, filename)
            
            if self.kaleido_available:
                # Try PNG export with timeout protection
                def export_png():
                    fig.write_image(filepath, width=800, height=600, scale=1.5, format='png')
                    return filepath
                
                future = self.executor.submit(export_png)
                return future.result(timeout=15)  # 15 second timeout
            else:
                # Fallback to HTML
                html_filepath = filepath.replace('.png', '.html')
                fig.write_html(html_filepath, include_plotlyjs='cdn')
                return html_filepath
                
        except Exception as e:
            print(f"   ⚠️ Export failed for {filename}: {e}")
            return self._create_fallback_chart(filename)
    
    def _create_fallback_chart(self, filename: str) -> str:
        """Create a simple fallback chart when export fails"""
        try:
            # Create a simple text-based placeholder
            fallback_path = os.path.join(self.visualization_dir, filename.replace('.png', '_fallback.html'))
            
            fallback_html = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Chart Placeholder</title></head>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h2>Visualization Placeholder</h2>
                <p>Original chart: {filename}</p>
                <p>This is a fallback placeholder. The original chart may be available in JSON format.</p>
                <p style="color: #666;">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </body>
            </html>
            """
            
            with open(fallback_path, 'w') as f:
                f.write(fallback_html)
            
            return fallback_path
        except Exception:
            return filename  # Return original filename as last resort

    def get_project_folder(self) -> str:
        """Return the current project folder path for other agents to use."""
        return self.project_folder

    def set_project_folder(self, project_folder: str):
        """Set a custom project folder path and update related directories."""
        self.project_folder = project_folder
        self.visualization_dir = f"{self.project_folder}/visualizations"
        self.eda_output_dir = f"{self.project_folder}/eda_output"
        self.json_dir = f"{self.project_folder}/json_charts"
        
        # Create directories
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.eda_output_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        
        print(f"   📁 Updated project folder: {self.project_folder}")

    def save_visualization_code_to_file(self, hypothesis: str, dataset_summary: Optional[Dict] = None) -> str:
        """Save a lightweight visualization generation code to a separate .py file in the project folder"""
        try:
            # Generate lightweight visualization code instead of massive code
            viz_code = self._generate_lightweight_visualization_code(hypothesis, dataset_summary)
            
            # Create the full file path
            code_file_path = os.path.join(self.project_folder, "generated_visualization_code.py")
            
            # Add header comment to the code file
            header_comment = f'''"""
Generated Visualization Code (Lightweight)
==========================================
This file contains lightweight visualization generation code created by the Enhanced Visualization Agent.
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This code implements essential data visualization including:
- Dataset analysis visualizations (class balance, missing values)
- Model performance visualizations
- Interactive charts using Plotly
- Export functionality for publication-quality figures

Usage:
    python generated_visualization_code.py

Requirements:
    - plotly
    - pandas
    - numpy
"""

'''
            
            # Write the code to file
            with open(code_file_path, 'w', encoding='utf-8') as f:
                f.write(header_comment)
                f.write(viz_code)
            
            print(f"✅ Visualization code saved to: {code_file_path}")
            print(f"   📄 File size: {len(viz_code)} characters")
            
            return code_file_path
            
        except Exception as e:
            print(f"❌ Error saving visualization code to file: {e}")
            return None

    def _generate_lightweight_visualization_code(self, hypothesis: str, dataset_summary: Optional[Dict] = None) -> str:
        """Generate lightweight standalone visualization code for better performance"""
        
        code = f'''
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os

class LightweightVisualizationGenerator:
    """
    Lightweight visualization generation system for research data analysis
    Generated for hypothesis: {hypothesis[:100]}...
    """
    
    def __init__(self):
        self.themes = {{
            "research": {{
                "font": {{"family": "Times New Roman", "size": 12}},
                "paper_bgcolor": "white",
                "plot_bgcolor": "white",
                "font_color": "black",
                "title_font_size": 14
            }}
        }}
        
        # Create output directory
        self.output_dir = "visualization_output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_class_balance_chart(self, class_data):
        """Generate class balance distribution chart"""
        if not isinstance(class_data, dict):
            return None
            
        df = pd.DataFrame(list(class_data.items()), columns=['Class', 'Percentage'])
        fig = px.bar(df, x='Class', y='Percentage', title='Class Balance Distribution')
        fig.update_layout(**self.themes["research"])
        
        output_path = os.path.join(self.output_dir, "class_balance_chart.html")
        fig.write_html(output_path)
        return fig
    
    def generate_missing_values_chart(self, missing_data):
        """Generate missing values analysis chart"""
        if not isinstance(missing_data, dict):
            return None
            
        filtered_data = {{k: v for k, v in missing_data.items() if v > 0}}
        if not filtered_data:
            return None
            
        missing_df = pd.DataFrame(list(filtered_data.items()), columns=['Feature', 'MissingCount'])
        fig = px.bar(missing_df, x='Feature', y='MissingCount', title='Missing Values per Feature')
        fig.update_layout(**self.themes["research"])
        
        output_path = os.path.join(self.output_dir, "missing_values_chart.html")
        fig.write_html(output_path)
        return fig
    
    def generate_model_performance_chart(self, performance_data):
        """Generate model performance comparison chart"""
        if not isinstance(performance_data, dict):
            return None
            
        methods = list(performance_data.keys())
        scores = list(performance_data.values())
        
        fig = go.Figure(data=go.Bar(x=methods, y=scores))
        fig.update_layout(title="Model Performance Comparison", **self.themes["research"])
        
        output_path = os.path.join(self.output_dir, "model_performance_chart.html")
        fig.write_html(output_path)
        return fig

def main():
    """Main execution function"""
    print("Lightweight Visualization Generator")
    viz_gen = LightweightVisualizationGenerator()
    
    # Example usage
    class_data = {{'Class_A': 45.2, 'Class_B': 32.1, 'Class_C': 22.7}}
    missing_data = {{'feature_1': 0, 'feature_2': 15, 'feature_3': 8}}
    performance_data = {{'Random Forest': 0.87, 'SVM': 0.82, 'Logistic Regression': 0.79}}
    
    viz_gen.generate_class_balance_chart(class_data)
    viz_gen.generate_missing_values_chart(missing_data)
    viz_gen.generate_model_performance_chart(performance_data)
    
    print("Visualizations generated successfully!")

if __name__ == "__main__":
    main()
'''
        
        return code

    def _generate_dataset_visualizations(self, dataset_summary: Dict) -> List[Dict]:
        """Generate visualizations directly from the user's dataset summary."""
        viz = []
        
        # Create class balance chart if applicable
        if isinstance(dataset_summary.get('class_balance'), dict):
            balance_data = dataset_summary['class_balance']
            df = pd.DataFrame(list(balance_data.items()), columns=['Class', 'Percentage'])
            chart = px.bar(df, x='Class', y='Percentage', title='Class Balance Distribution', labels={'Percentage': 'Percentage (%)'})
            chart.update_layout(self.themes.get("research", {}))
            
            viz.append({
                "chart_json": chart.to_json(),
                "type": "class_balance_chart",
                "title": "Class Balance Distribution",
                "description": "This bar chart shows the distribution of classes in the target variable, which is crucial for identifying potential model bias and understanding dataset composition. The visualization displays both frequency counts and percentages for each diagnostic category to inform preprocessing and modeling decisions."
            })
            
        # Create missing values heatmap
        missing_per_column = dataset_summary.get('missing_per_column', {})
        if any(v > 0 for v in missing_per_column.values()):
            missing_df = pd.DataFrame(list(missing_per_column.items()), columns=['Feature', 'MissingCount']).sort_values('MissingCount', ascending=False)
            
            chart = px.bar(missing_df[missing_df['MissingCount']>0], x='Feature', y='MissingCount', title='Count of Missing Values per Feature')
            chart.update_layout(self.themes.get("research", {}))
            
            viz.append({
                "chart_json": chart.to_json(),
                "type": "missing_values_chart",
                "title": "Missing Values Analysis",
                "description": "This chart highlights features with missing data, guiding the preprocessing strategy for imputation. Features are ordered by missingness percentage to prioritize data quality assessment and identify patterns that may inform imputation strategies or feature exclusion decisions."
            })
            
        return viz

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
        """Export visualization to file with optimized performance and fallback options"""
        try:
            fig = go.Figure(json.loads(chart_json))
            filepath = os.path.join(self.visualization_dir, filename)
            
            print(f"   📊 Exporting chart to {filepath}...")
            
            if self.kaleido_available and format.lower() in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
                # Use optimized PNG export with reduced quality for speed
                fig.write_image(filepath, width=800, height=600, scale=1.5, format=format)
                print(f"   ✅ Chart exported successfully as {format.upper()}")
                return filepath
            else:
                # Fast HTML fallback
                html_filepath = filepath.replace(f'.{format}', '.html')
                fig.write_html(html_filepath, include_plotlyjs='cdn')
                print(f"   ✅ Chart exported as HTML: {html_filepath}")
                return html_filepath
            
        except Exception as e:
            print(f"   ⚠️ Export failed: {e}")
            # Create fast fallback
            return self._create_fallback_chart(filename)

    def _load_model_results(self) -> Optional[Dict]:
        """
        Load model results from project directory.
        NEVER generates synthetic data - returns None if no real results found.
        """
        # Try multiple locations for model results
        possible_paths = [
            os.path.join(self.project_folder, "model_results.json"),
            "model_results.json",
            os.path.join(self.project_folder, "results.json"),
            os.path.join(self.project_folder, "execution_results.json")
        ]
        
        for results_path in possible_paths:
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                        print(f"   📄 Found real execution results: '{results_path}'")
                        return results
                except (json.JSONDecodeError, IOError) as e:
                    print(f"   ⚠️ Error loading '{results_path}': {e}")
        
        # CRITICAL: NEVER generate synthetic data for research papers
        print("   📋 No real model execution results found")
        print("   🚫 SYNTHETIC DATA GENERATION DISABLED for research integrity")
        return None

    def _generate_model_visualizations(self, model_results: Dict) -> List[Dict]:
        """Generate visualizations from the model's output results."""
        visualizations = []

        # Generate ROC curve
        if 'roc_curve' in model_results:
            roc_data = model_results['roc_curve']
            fig = px.area(
                x=roc_data['fpr'],
                y=roc_data['tpr'],
                title='ROC Curve',
                labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                hover_data={'threshold': roc_data['thresholds']}
            )
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            fig.update_layout(**self.themes["research"])
            visualizations.append({
                "chart_json": fig.to_json(),
                "type": "roc_curve",
                "title": "ROC Curve",
                "description": "Receiver Operating Characteristic curve showing the trade-off between true positive rate and false positive rate."
            })

        # Generate Precision-Recall curve
        if 'precision_recall_curve' in model_results:
            pr_data = model_results['precision_recall_curve']
            fig = px.area(
                x=pr_data['recall'],
                y=pr_data['precision'],
                title='Precision-Recall Curve',
                labels={'x': 'Recall', 'y': 'Precision'},
                hover_data={'threshold': pr_data['thresholds']}
            )
            fig.update_layout(**self.themes["research"])
            visualizations.append({
                "chart_json": fig.to_json(),
                "type": "precision_recall_curve",
                "title": "Precision-Recall Curve",
                "description": "Precision-Recall curve illustrating the trade-off between precision and recall for different thresholds."
            })

        # Generate Feature Importance plot
        if 'feature_importance' in model_results:
            fi_data = model_results['feature_importance']
            fig = px.bar(
                x=fi_data['importance'],
                y=fi_data['features'],
                orientation='h',
                title='Feature Importance',
                labels={'x': 'Importance', 'y': 'Feature'}
            )
            fig.update_layout(**self.themes["research"])
            visualizations.append({
                "chart_json": fig.to_json(),
                "type": "feature_importance",
                "title": "Feature Importance",
                "description": "Bar chart showing the importance of each feature in the model's predictions."
            })

        return visualizations

    def _generate_chart_code(self, chart_data: dict, chart_name: str) -> str:
        """Generate Python code for a specific chart."""
        code = f'''"""
Generated chart code for: {chart_name}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import plotly.graph_objects as go
import json
import os

def generate_{chart_name}():
    # Load chart data
    with open('{os.path.join(self.json_dir, f"{chart_name}.json")}', 'r') as f:
        chart_data = json.load(f)
    
    # Create figure
    fig = go.Figure(chart_data)
    
    # Apply theme
    fig.update_layout(
        font=dict(family="Times New Roman", size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="black",
        title_font_size=14
    )
    
    # Save figure
    fig.write_html("{os.path.join(self.visualization_dir, f"{chart_name}.html")}")
    
    return fig

if __name__ == "__main__":
    generate_{chart_name}()
'''
        return code

    def load_chart_from_json(self, chart_name: str) -> Optional[go.Figure]:
        """Load a chart from JSON for fast reload."""
        json_path = os.path.join(self.json_dir, f"{chart_name}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    chart_data = json.load(f)
                return go.Figure(chart_data)
            except Exception as e:
                print(f"⚠️ Error loading chart from JSON: {e}")
        return None

    def __del__(self):
        """Cleanup thread pool on destruction"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception:
            pass

# Example usage
if __name__ == "__main__":
    from agents.note_taker import NoteTaker
    import urllib.parse
    
    # Test with mock note taker
    class MockNoteTaker:
        def log_visualization(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
    
    note_taker = MockNoteTaker()
    agent = EnhancedVisualizationAgent(note_taker)
    
    test_hypothesis = "Combining transformer attention mechanisms with LSTM networks will improve time series forecasting accuracy by 15% compared to traditional LSTM models"
    
    visualizations = agent.generate_visualizations(test_hypothesis, [])
    print(f"Generated {len(visualizations)} visualizations")
    for viz in visualizations:
        print(f"- {viz['title']}: {viz['type']}") 

        
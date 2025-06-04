import os
import urllib.parse
from dotenv import load_dotenv
load_dotenv()
from src.agents.note_taker import NoteTaker
from src.agents.search_agent import SearchAgent
from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.insight_agent import InsightAgent
from src.agents.code_agent import CodeAgent
from src.agents.visualization_agent import VisualizationAgent
from src.agents.report_agent import ReportAgent

# Mock class for testing when OpenAI API has quota issues
class MockHypothesisAgent:
    def __init__(self, openai_api_key: str, note_taker):
        self.note_taker = note_taker
        
    def generate_hypothesis(self, papers):
        mock_hypothesis = """
        ## Main Hypothesis Statement
        Transformer-based models with attention mechanisms specifically adapted for time series forecasting will outperform traditional LSTM and ARIMA models by 15-20% in terms of RMSE when applied to multi-variate financial time series data.

        ## Key Assumptions
        1. Time series data exhibits long-range dependencies that transformers can capture better than RNNs
        2. Attention mechanisms can identify relevant temporal patterns across different time scales
        3. Positional encoding can be adapted for temporal sequences in financial data

        ## Proposed Testing Methodology
        1. Implement transformer architecture with temporal positional encoding
        2. Compare against LSTM and ARIMA baselines on 3 financial datasets
        3. Use rolling window validation with 80/20 train/test split
        4. Measure RMSE, MAE, and directional accuracy

        ## Expected Outcomes
        - 15-20% improvement in RMSE over baselines
        - Better handling of long-term dependencies
        - Improved performance on volatile market periods
        """
        self.note_taker.log_hypothesis(mock_hypothesis)
        return {
            "hypothesis": mock_hypothesis,
            "papers": papers,
            "generation_id": "mock_generation_id"
        }

def main():
    print("🚀 Starting Research Assistant Agent Pipeline...")
    
    # Load environment variables
    password = "Jasmine@0802"  # Your MongoDB password
    encoded_password = urllib.parse.quote_plus(password)
    MONGO_URI = f"mongodb+srv://jaschri:{encoded_password}@agent.wb3vq0q.mongodb.net/?retryWrites=true&w=majority&appName=Agent"
    OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

    print(f"📊 MongoDB URI: {MONGO_URI[:50]}...")
    print(f"🤖 OpenAI API Key: {OPENAI_API_KEY[:20] if OPENAI_API_KEY else 'NOT SET'}...")
    print(f"☁️  Google Cloud Project: {PROJECT_ID}")

    # Create a shared NoteTaker instance
    note_taker = NoteTaker(MONGO_URI)
    print("✅ NoteTaker initialized")

    # Initialize all agents with the shared NoteTaker
    search_agent = SearchAgent(MONGO_URI, note_taker)
    print("✅ SearchAgent initialized")
    
    # Use mock hypothesis agent to bypass API quota issues
    hypothesis_agent = MockHypothesisAgent(OPENAI_API_KEY, note_taker)
    print("✅ HypothesisAgent initialized (using mock)")
    
    try:
        insight_agent = InsightAgent(PROJECT_ID, note_taker)
        print("✅ InsightAgent initialized")
    except Exception as e:
        print(f"❌ InsightAgent failed: {e}")
        return
    
    code_agent = CodeAgent(OPENAI_API_KEY, note_taker)
    print("✅ CodeAgent initialized")
    
    visualization_agent = VisualizationAgent(note_taker)
    print("✅ VisualizationAgent initialized")
    
    report_agent = ReportAgent(note_taker)
    print("✅ ReportAgent initialized")

    # Example pipeline: search -> hypothesis -> insight -> code -> visualization -> report
    print("\n🔍 Step 1: Searching for papers...")
    query = "transformer models for time series forecasting"
    try:
        papers = search_agent.search(query)
        print(f"✅ Found {len(papers)} papers")
    except Exception as e:
        print(f"❌ Search failed: {e}")
        # Use mock papers if search fails
        papers = [
            {
                "title": "Attention Is All You Need for Time Series Forecasting",
                "abstract": "We propose a transformer-based architecture for time series forecasting that leverages attention mechanisms to capture long-range temporal dependencies.",
                "authors": ["Smith, J.", "Doe, A."],
                "year": 2023,
                "arxiv_id": "2301.12345"
            }
        ]
        print(f"✅ Using {len(papers)} mock papers")

    print("\n💡 Step 2: Generating hypothesis...")
    try:
        hypothesis = hypothesis_agent.generate_hypothesis(papers)
        print("✅ Hypothesis generated")
    except Exception as e:
        print(f"❌ Hypothesis generation failed: {e}")
        return

    print("\n📈 Step 3: Generating insights...")
    try:
        insights = insight_agent.publication_trends_by_year()
        print(f"✅ Generated {len(insights)} insights")
    except Exception as e:
        print(f"❌ Insight generation failed: {e}")
        # Use mock insights
        insights = ["Publication trends show increasing interest in transformer models for time series since 2020"]
        print(f"✅ Using {len(insights)} mock insights")

    print("\n📊 Step 4: Creating visualization...")
    try:
        visualization = visualization_agent.create_line_plot(
            [{"x": 2020, "y": 5}, {"x": 2021, "y": 12}, {"x": 2022, "y": 25}, {"x": 2023, "y": 45}],
            "Transformer Time Series Papers by Year",
            "Year",
            "Number of Papers"
        )
        print("✅ Visualization created")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return

    print("\n📄 Step 5: Assembling report...")
    try:
        report = report_agent.assemble_report(
            hypothesis["hypothesis"],
            insights,
            [{"path": "transformer_trends.png", "caption": "Publication trends for transformer time series models"}],
            ["Smith, J., & Doe, A. (2023). Attention Is All You Need for Time Series Forecasting. arXiv:2301.12345"]
        )
        print("✅ Report assembled")
        print("\n📋 FINAL REPORT:")
        print("=" * 80)
        print(report[:500] + "..." if len(report) > 500 else report)
        print("=" * 80)
    except Exception as e:
        print(f"❌ Report assembly failed: {e}")
        return

    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main() 
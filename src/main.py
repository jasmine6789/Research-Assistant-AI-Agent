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

def main():
    # Load environment variables
    password = "Jasmine@0802"  # Your MongoDB password
    encoded_password = urllib.parse.quote_plus(password)
    MONGO_URI = f"mongodb+srv://jaschri:{encoded_password}@agent.wb3vq0q.mongodb.net/?retryWrites=true&w=majority&appName=Agent"
    OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

    # Create a shared NoteTaker instance
    note_taker = NoteTaker(MONGO_URI)

    # Initialize all agents with the shared NoteTaker
    search_agent = SearchAgent(MONGO_URI, note_taker)
    hypothesis_agent = HypothesisAgent(OPENAI_API_KEY, note_taker)
    insight_agent = InsightAgent(PROJECT_ID, note_taker)
    code_agent = CodeAgent(OPENAI_API_KEY, note_taker)
    visualization_agent = VisualizationAgent(note_taker)
    report_agent = ReportAgent(note_taker)

    # Example pipeline: search -> hypothesis -> insight -> code -> visualization -> report
    query = "transformer models for time series forecasting"
    papers = search_agent.search(query)
    hypothesis = hypothesis_agent.generate_hypothesis(papers)
    insights = insight_agent.publication_trends_by_year()
    code = code_agent.generate_code(hypothesis["hypothesis"])
    visualization = visualization_agent.create_line_plot(
        [{"x": 1, "y": 2}, {"x": 2, "y": 3}],
        "Test Plot",
        "X",
        "Y"
    )
    report = report_agent.assemble_report(
        hypothesis["hypothesis"],
        ["Test insight"],
        [{"path": "test.png", "caption": "Test visualization"}],
        ["Test citation"]
    )

    print("Report generated:", report)

if __name__ == "__main__":
    main() 
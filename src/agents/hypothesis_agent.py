import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.agents.note_taker import NoteTaker

class HypothesisAgent:
    def __init__(self, openai_api_key: str, note_taker: NoteTaker):
        self.client = OpenAI(api_key=openai_api_key, base_url="https://api.openai.com/v1")
        self.note_taker = note_taker
        self.system_prompt = """You are a research hypothesis generator for machine learning papers.
        Given a set of research papers, generate a novel, testable hypothesis that builds upon their findings.
        The hypothesis should be:
        1. Specific and measurable
        2. Based on the provided papers
        3. Novel and interesting
        4. Testable with available data
        Format the hypothesis as:
        - Main hypothesis statement
        - Key assumptions
        - Proposed testing methodology
        - Expected outcomes"""

    def generate_hypothesis(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a research hypothesis from the provided papers."""
        paper_summaries = []
        for paper in papers:
            summary = f"Title: {paper['title']}\nAbstract: {paper['abstract']}\nAuthors: {', '.join(paper['authors'])}\nYear: {paper['year']}\n"
            paper_summaries.append(summary)

        prompt = f"""Based on the following research papers, generate a novel hypothesis:

{chr(10).join(paper_summaries)}

Please generate a hypothesis that builds upon these papers."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        hypothesis = response.choices[0].message.content
        self.note_taker.log_hypothesis(hypothesis)

        return {
            "hypothesis": hypothesis,
            "papers": papers,
            "generation_id": response.id
        }

    def refine_hypothesis(self, 
                         current_hypothesis: Dict[str, Any], 
                         feedback: str,
                         regenerate: bool = False) -> Dict[str, Any]:
        """Refine the hypothesis based on user feedback."""
        if regenerate:
            return self.generate_hypothesis(current_hypothesis["papers"])

        prompt = f"""Current hypothesis:
{current_hypothesis['hypothesis']}

User feedback:
{feedback}

Please refine the hypothesis based on this feedback while maintaining its core ideas."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        refined_hypothesis = response.choices[0].message.content
        self.note_taker.log_hypothesis(refined_hypothesis, refined=True)

        return {
            "hypothesis": refined_hypothesis,
            "papers": current_hypothesis["papers"],
            "generation_id": response.id,
            "previous_generation_id": current_hypothesis["generation_id"]
        }

# Example usage (to be removed in production)
if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
    MONGO_URI = os.getenv("MONGO_URI")
    note_taker = NoteTaker(MONGO_URI)
    agent = HypothesisAgent(OPENAI_API_KEY, note_taker)
    
    example_papers = [
        {
            "title": "Example Paper 1",
            "abstract": "This is an example abstract",
            "authors": ["Author 1", "Author 2"],
            "year": 2023,
            "arxiv_id": "2301.12345"
        }
    ]
    
    hypothesis = agent.generate_hypothesis(example_papers)
    print("Initial Hypothesis:", hypothesis["hypothesis"])
    
    refined = agent.refine_hypothesis(
        hypothesis,
        "Please make the hypothesis more specific about the implementation details.",
        regenerate=False
    )
    print("\nRefined Hypothesis:", refined["hypothesis"]) 
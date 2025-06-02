import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.agents.search_agent import NoteTaker
from unittest.mock import patch
import mongomock

class HypothesisAgent:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
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
        # Prepare paper summaries for the prompt
        paper_summaries = []
        for paper in papers:
            summary = f"Title: {paper['title']}\nAbstract: {paper['abstract']}\nAuthors: {', '.join(paper['authors'])}\nYear: {paper['year']}\n"
            paper_summaries.append(summary)

        # Create the prompt
        prompt = f"""Based on the following research papers, generate a novel hypothesis:

{chr(10).join(paper_summaries)}

Please generate a hypothesis that builds upon these papers."""

        # Generate hypothesis using OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        hypothesis = response.choices[0].message.content

        # Log the hypothesis generation
        with patch('src.agents.note_taker.MongoClient', new=mongomock.MongoClient):
            NoteTaker.log("hypothesis_generation", {
                "papers": [p["arxiv_id"] for p in papers],
                "hypothesis": hypothesis
            })

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
            # Generate a completely new hypothesis
            return self.generate_hypothesis(current_hypothesis["papers"])

        # Refine existing hypothesis
        prompt = f"""Current hypothesis:
{current_hypothesis['hypothesis']}

User feedback:
{feedback}

Please refine the hypothesis based on this feedback while maintaining its core ideas."""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        refined_hypothesis = response.choices[0].message.content

        # Log the refinement
        with patch('src.agents.note_taker.MongoClient', new=mongomock.MongoClient):
            NoteTaker.log("hypothesis_refinement", {
                "original_hypothesis": current_hypothesis["hypothesis"],
                "feedback": feedback,
                "refined_hypothesis": refined_hypothesis,
                "regenerated": regenerate
            })

        return {
            "hypothesis": refined_hypothesis,
            "papers": current_hypothesis["papers"],
            "generation_id": response.id,
            "previous_generation_id": current_hypothesis["generation_id"]
        }

# Example usage (to be removed in production)
if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
    agent = HypothesisAgent(OPENAI_API_KEY)
    
    # Example papers (would come from SearchAgent in practice)
    example_papers = [
        {
            "title": "Example Paper 1",
            "abstract": "This is an example abstract",
            "authors": ["Author 1", "Author 2"],
            "year": 2023,
            "arxiv_id": "2301.12345"
        }
    ]
    
    # Generate initial hypothesis
    hypothesis = agent.generate_hypothesis(example_papers)
    print("Initial Hypothesis:", hypothesis["hypothesis"])
    
    # Refine hypothesis
    refined = agent.refine_hypothesis(
        hypothesis,
        "Please make the hypothesis more specific about the implementation details.",
        regenerate=False
    )
    print("\nRefined Hypothesis:", refined["hypothesis"]) 
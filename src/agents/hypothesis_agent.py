import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from agents.note_taker import NoteTaker
from agents.enhanced_hypothesis_generator import EnhancedHypothesisGenerator
import openai
import time

class HypothesisAgent:
    def __init__(self, openai_api_key: str, note_taker: NoteTaker):
        self.client = OpenAI(api_key=openai_api_key, base_url="https://api.openai.com/v1")
        self.note_taker = note_taker
        self.enhanced_generator = EnhancedHypothesisGenerator(self.client)
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
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for HypothesisAgent")

    def generate_sophisticated_hypothesis(self, user_topic: str, papers: List[Dict[str, Any]] = None, dataset_analysis=None) -> Dict[str, Any]:
        """
        Generate a sophisticated, research-gap-focused hypothesis for any domain.
        
        Args:
            user_topic: The research topic provided by the user
            papers: Optional list of relevant papers for context
            dataset_analysis: Optional dataset information
            
        Returns:
            Dict containing comprehensive hypothesis data including gap analysis and significance
        """
        try:
            # Prepare dataset info for enhanced generator
            dataset_info = None
            if dataset_analysis:
                dataset_info = {
                    "shape": dataset_analysis.get("shape"),
                    "columns": dataset_analysis.get("columns", []),
                    "target_variable": dataset_analysis.get("target_variable")
                }
            
            # Generate enhanced hypothesis
            hypothesis_data = self.enhanced_generator.generate_research_hypothesis(user_topic, dataset_info)
            
            # Add papers context if provided
            if papers:
                hypothesis_data["related_papers"] = papers
                
            # Format for display
            formatted_display = self.enhanced_generator.format_hypothesis_for_display(hypothesis_data)
            hypothesis_data["formatted_display"] = formatted_display
            
            # Log the generation
            if self.note_taker:
                log_data = {
                    "topic": user_topic,
                    "hypothesis": hypothesis_data.get("hypothesis", ""),
                    "research_gap": hypothesis_data.get("research_gap", ""),
                    "methodology": hypothesis_data.get("methodology", ""),
                    "has_dataset": dataset_analysis is not None,
                    "has_papers": papers is not None and len(papers) > 0
                }
                self.note_taker.log("sophisticated_hypothesis_generation", log_data)
            
            return hypothesis_data
            
        except Exception as e:
            print(f"❌ Error generating sophisticated hypothesis: {e}")
            if self.note_taker:
                self.note_taker.log("hypothesis_generation_error", {"error": str(e)})
            
            # Fallback to simple hypothesis
            return {
                "hypothesis": f"Machine learning models can improve predictive accuracy in {user_topic} by incorporating multiple data modalities and advanced feature engineering techniques.",
                "research_gap": "Limited integration of multimodal approaches in current research",
                "significance": f"This addresses the need for more comprehensive predictive models in {user_topic}",
                "methodology": "Multi-class classification with ensemble methods",
                "innovation": "Novel combination of data sources and advanced techniques",
                "formatted_display": f"**Research Hypothesis**: Machine learning models can improve predictive accuracy in {user_topic}"
            }

    def generate_hypothesis(self, papers: List[Dict[str, Any]], dataset_analysis=None, feedback=None) -> str:
        """
        Generates a research hypothesis based on selected papers, an optional dataset summary, and user feedback.
        """
        try:
            prompt = self._build_hypothesis_prompt(papers, dataset_analysis, feedback)

            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert research scientist. Your task is to generate a concise, testable, and innovative research hypothesis based on the provided literature and data summary."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            hypothesis = response.choices[0].message.content.strip()
            
            # Log the successful generation
            if self.note_taker:
                log_data = {
                    "papers_count": len(papers), 
                    "has_dataset": dataset_analysis is not None,
                    "has_feedback": feedback is not None
                }
                self.note_taker.log("hypothesis_generation_success", log_data)
                
            return hypothesis

        except openai.OpenAIError as e:
            print(f"❌ OpenAI API error in HypothesisAgent: {e}")
            if self.note_taker:
                self.note_taker.log("hypothesis_generation_error", {"error": str(e)})
            return "Error: Could not generate hypothesis due to an API error."
        except Exception as e:
            print(f"❌ An unexpected error occurred in HypothesisAgent: {e}")
            if self.note_taker:
                self.note_taker.log("hypothesis_generation_error", {"error": str(e)})
            return "Error: An unexpected error occurred."

    def _build_hypothesis_prompt(self, papers, dataset_analysis=None, feedback=None):
        """Builds the prompt for the GPT-4 model."""
        
        # Handle cases with no papers
        if papers and len(papers) > 0:
            # Summarize papers
            paper_summaries = "\n".join([f"- {p.get('title', 'N/A')}: {p.get('summary', 'N/A')[:250]}..." for p in papers[:5]])
            
            prompt = (
                "Based on the following research papers, please generate a clear and specific research hypothesis.\n\n"
                "## Relevant Literature:\n"
                f"{paper_summaries}\n\n"
            )
        else:
            # No papers available - generate hypothesis from dataset and general knowledge
            prompt = (
                "Generate a clear and specific research hypothesis for the given research context. "
                "Since no specific literature papers are available, base your hypothesis on the dataset characteristics "
                "and general domain knowledge.\n\n"
                "## Research Context:\n"
                "- Limited external literature available\n"
                "- Focus on dataset-driven hypothesis generation\n\n"
            )

        if dataset_analysis:
            data_summary_text = self._format_dataset_summary(dataset_analysis)
            prompt += (
                "## User-Provided Dataset Summary:\n"
                f"{data_summary_text}\n\n"
                "Given this data, the hypothesis MUST be something that is directly testable with the provided columns. "
                "It should connect the research topic to the specific features available in the dataset.\n\n"
            )
        else:
            prompt += (
                "The hypothesis should be general enough to be tested with synthetic data, focusing on common methodologies in this research area.\n\n"
            )

        if feedback:
            prompt += (
                "## User Feedback for Improvement:\n"
                f"{feedback}\n\n"
                "Please refine the hypothesis based on this feedback.\n\n"
            )
            
        prompt += (
            "## Task:\n"
            "Generate one single, concise research hypothesis statement. The statement should be innovative, clear, and directly testable. "
            "Do not return a list of hypotheses. Frame it as a formal scientific hypothesis."
            "\n\n**Hypothesis:**"
        )
        
        return prompt

    def _format_dataset_summary(self, analysis):
        """Formats the dataset analysis into a string for the prompt."""
        if not analysis:
            return "No dataset provided."
        
        summary = []
        summary.append(f"- Shape: {analysis.get('shape')}")
        summary.append(f"- Columns: {', '.join(analysis.get('columns', []))}")
        if analysis.get('target_variable'):
            summary.append(f"- Target Variable: {analysis.get('target_variable')}")
        if analysis.get('size_warning'):
            summary.append(f"- Note: {analysis.get('size_warning')}")
        if isinstance(analysis.get('class_balance'), dict):
            balance_str = ", ".join([f"{k}: {v:.1f}%" for k, v in analysis['class_balance'].items()])
            summary.append(f"- Class Balance: {balance_str}")

        return "\n".join(summary)

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
    print("Initial Hypothesis:", hypothesis)
    
    refined = agent.refine_hypothesis(
        hypothesis,
        "Please make the hypothesis more specific about the implementation details.",
        regenerate=False
    )
    print("\nRefined Hypothesis:", refined["hypothesis"]) 
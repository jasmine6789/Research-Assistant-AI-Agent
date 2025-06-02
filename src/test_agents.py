import os
from dotenv import load_dotenv
from agents.search_agent import SearchAgent
from agents.hypothesis_agent import HypothesisAgent

def test_search_agent():
    print("\n=== Testing SearchAgent ===")
    try:
        # Initialize SearchAgent
        search_agent = SearchAgent(os.getenv("MONGO_URI"))
        print("✓ SearchAgent initialized successfully")

        # Test search functionality
        query = "transformer models for time series forecasting"
        print(f"\nSearching for: {query}")
        results = search_agent.search(query, top_k=2)
        
        print(f"\nFound {len(results)} papers:")
        for paper in results:
            print(f"\nTitle: {paper['title']}")
            print(f"Authors: {', '.join(paper['authors'])}")
            print(f"Year: {paper['year']}")
            print(f"arXiv ID: {paper['arxiv_id']}")
            print("-" * 50)
        
        return results
    except Exception as e:
        print(f"Error testing SearchAgent: {str(e)}")
        return None

def test_hypothesis_agent(papers):
    print("\n=== Testing HypothesisAgent ===")
    try:
        # Initialize HypothesisAgent
        hypothesis_agent = HypothesisAgent(os.getenv("CHATGPT_API_KEY"))
        print("✓ HypothesisAgent initialized successfully")

        # Generate initial hypothesis
        print("\nGenerating initial hypothesis...")
        hypothesis = hypothesis_agent.generate_hypothesis(papers)
        print("\nInitial Hypothesis:")
        print(hypothesis["hypothesis"])
        print("-" * 50)

        # Test refinement
        feedback = "Please make the hypothesis more specific about the implementation details."
        print(f"\nRefining hypothesis with feedback: {feedback}")
        refined = hypothesis_agent.refine_hypothesis(hypothesis, feedback)
        print("\nRefined Hypothesis:")
        print(refined["hypothesis"])
        print("-" * 50)

        # Test regeneration
        print("\nRegenerating hypothesis...")
        regenerated = hypothesis_agent.refine_hypothesis(hypothesis, "", regenerate=True)
        print("\nRegenerated Hypothesis:")
        print(regenerated["hypothesis"])
        print("-" * 50)

    except Exception as e:
        print(f"Error testing HypothesisAgent: {str(e)}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    required_vars = ["MONGO_URI", "CHATGPT_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please ensure your .env file contains all required variables.")
        return

    # Test SearchAgent
    papers = test_search_agent()
    
    # If SearchAgent test was successful, test HypothesisAgent
    if papers:
        test_hypothesis_agent(papers)

if __name__ == "__main__":
    main() 
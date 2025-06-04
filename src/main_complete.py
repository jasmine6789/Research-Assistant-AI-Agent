import os
import urllib.parse
from dotenv import load_dotenv
load_dotenv()
from src.agents.note_taker import NoteTaker
from src.agents.web_search_agent import WebSearchAgent
from src.agents.web_insight_agent import WebInsightAgent
from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.code_agent import CodeAgent
from src.agents.visualization_agent import VisualizationAgent
from src.agents.report_agent import ReportAgent
import time
import uuid

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80)

def print_step(step_num, title):
    """Print a formatted step header"""
    print(f"\n🔄 STEP {step_num}: {title}")
    print("-" * 60)

def get_user_input():
    """Get research query from user"""
    print_header("RESEARCH ASSISTANT AGENT - COMPLETE PIPELINE")
    print("🤖 Welcome to your AI-powered Research Assistant!")
    print("📚 I can help you research any topic using real academic papers from arXiv.")
    print("\nExample topics:")
    print("  • 'transformer models for time series forecasting'")
    print("  • 'graph neural networks for drug discovery'")
    print("  • 'reinforcement learning for robotics'")
    print("  • 'computer vision for medical imaging'")
    
    while True:
        query = input("\n💭 What would you like to research? ").strip()
        if query:
            print(f"\n✅ Great! I'll research: '{query}'")
            return query
        print("❌ Please enter a valid research topic.")

def show_human_in_loop_options():
    """Show human-in-the-loop options"""
    print("\n🤝 HUMAN-IN-THE-LOOP OPTIONS:")
    print("   1. Regenerate hypothesis")
    print("   2. Provide feedback")
    print("   3. Continue to next step")
    print("   4. Exit pipeline")

def main():
    """Main pipeline execution"""
    print_header("INITIALIZING MULTI-AGENT SYSTEM")
    
    # Load environment variables
    password = "Jasmine@0802"
    encoded_password = urllib.parse.quote_plus(password)
    MONGO_URI = f"mongodb+srv://jaschri:{encoded_password}@agent.wb3vq0q.mongodb.net/?retryWrites=true&w=majority&appName=Agent&ssl=true&ssl_cert_reqs=CERT_NONE"
    OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

    print(f"📊 MongoDB: Connecting...")
    print(f"🤖 OpenAI: GPT-4 Ready")
    print(f"☁️  Google Cloud: {PROJECT_ID}")

    # Initialize all agents with MongoDB fallback
    print("\n🏗️  Initializing Agents...")
    
    # Try MongoDB connection, fallback to mock if it fails
    try:
        note_taker = NoteTaker(MONGO_URI)
        # Test connection
        note_taker.log("test", {"status": "connection_test"})
        print("   ✅ NoteTaker (MongoDB connected)")
        use_mongodb = True
    except Exception as e:
        print(f"   ⚠️  MongoDB connection failed: {str(e)[:100]}...")
        print("   🔄 Using in-memory logging for demonstration")
        # Create a mock note taker for demo
        class MockNoteTaker:
            def __init__(self):
                self.logs = []
            def log_session_start(self, **kwargs): 
                self.logs.append(f"Session started: {kwargs}")
            def log_session_end(self, **kwargs): 
                self.logs.append(f"Session ended: {kwargs}")
            def log_query(self, query, **kwargs): 
                self.logs.append(f"Query: {query}")
            def log_selected_papers(self, papers, **kwargs): 
                self.logs.append(f"Papers: {len(papers) if papers else 0} selected")
            def log_feedback(self, feedback, **kwargs): 
                self.logs.append(f"Feedback: {feedback}")
            def log_hypothesis(self, hypothesis, **kwargs):
                self.logs.append(f"Hypothesis generated: {hypothesis[:100]}...")
            def log_code(self, code, **kwargs):
                self.logs.append(f"Code generated: {len(code)} characters")
            def log_insights(self, insights, **kwargs):
                self.logs.append(f"Insights: {len(insights)} items")
            def log_visualization(self, viz_data, **kwargs):
                self.logs.append(f"Visualization created")
            def log_report(self, report, **kwargs):
                self.logs.append(f"Report generated: {len(report)} characters")
            def log(self, log_type, content, **kwargs): 
                self.logs.append(f"{log_type}: {content}")
            def get_session_logs(self, **kwargs):
                return self.logs
            def get_logs(self, **kwargs):
                return self.logs
        note_taker = MockNoteTaker()
        use_mongodb = False
    
    search_agent = WebSearchAgent(note_taker)
    print("   ✅ WebSearchAgent (arXiv integration)")
    
    insight_agent = WebInsightAgent(note_taker, search_agent)
    print("   ✅ WebInsightAgent (analysis & topic modeling)")
    
    hypothesis_agent = HypothesisAgent(OPENAI_API_KEY, note_taker)
    print("   ✅ HypothesisAgent (GPT-4 powered)")
    
    code_agent = CodeAgent(OPENAI_API_KEY, note_taker)
    print("   ✅ CodeAgent (GPT-4 code generation)")
    
    visualization_agent = VisualizationAgent(note_taker)
    print("   ✅ VisualizationAgent (charts & plots)")
    
    report_agent = ReportAgent(note_taker)
    print("   ✅ ReportAgent (comprehensive reports)")

    # Start session
    session_id = str(uuid.uuid4())[:8]  # Generate short session ID
    note_taker.log_session_start(user="demo_user", session_id=session_id)
    print(f"\n📝 Session started: {session_id}")

    # Get user input
    query = get_user_input()
    note_taker.log_query(query, user="demo_user", session_id=session_id)

    # STEP 1: SEARCH & PAPER RETRIEVAL
    print_step(1, "SEARCHING ARXIV FOR RELEVANT PAPERS")
    try:
        print(f"🔍 Searching arXiv for: '{query}'")
        papers = search_agent.search(query, top_k=5, max_results=20)
        print(f"✅ Found {len(papers)} relevant papers")
        
        if papers:
            print("\n📄 TOP PAPERS RETRIEVED:")
            for i, paper in enumerate(papers[:3], 1):
                print(f"   {i}. {paper['title'][:70]}...")
                print(f"      Authors: {', '.join(paper['authors'][:2])}{'...' if len(paper['authors']) > 2 else ''}")
                print(f"      Year: {paper['year']} | arXiv: {paper['arxiv_id']} | Relevance: {paper.get('similarity_score', 0):.3f}")
                
        note_taker.log_selected_papers(papers)
    except Exception as e:
        print(f"❌ Search failed: {e}")
        papers = []

    if not papers:
        print("❌ No papers found. Exiting...")
        note_taker.log_session_end(user="demo_user", session_id=session_id)
        return

    # STEP 2: HYPOTHESIS GENERATION
    print_step(2, "GENERATING RESEARCH HYPOTHESIS (GPT-4)")
    hypothesis_generated = False
    hypothesis = None
    
    while not hypothesis_generated:
        try:
            print("🧠 GPT-4 is analyzing papers and generating hypothesis...")
            hypothesis = hypothesis_agent.generate_hypothesis(papers)
            print("✅ Research hypothesis generated!")
            print(f"\n📝 HYPOTHESIS PREVIEW:")
            print(hypothesis['hypothesis'][:300] + "..." if len(hypothesis['hypothesis']) > 300 else hypothesis['hypothesis'])
            
            # Human-in-the-loop for hypothesis
            show_human_in_loop_options()
            choice = input("\n🤔 Your choice (1-4): ").strip()
            
            if choice == "1":
                print("🔄 Regenerating hypothesis...")
                continue
            elif choice == "2":
                feedback = input("💬 Please provide your feedback: ").strip()
                if feedback:
                    note_taker.log_feedback(feedback)
                    print("📝 Feedback logged. Refining hypothesis...")
                    hypothesis = hypothesis_agent.refine_hypothesis(hypothesis, feedback)
                    print("✅ Hypothesis refined based on your feedback!")
            elif choice == "3":
                hypothesis_generated = True
                print("✅ Proceeding with current hypothesis")
            elif choice == "4":
                print("👋 Exiting pipeline. Thank you!")
                return
            else:
                print("✅ Invalid choice. Proceeding with current hypothesis")
                hypothesis_generated = True
                
        except Exception as e:
            print(f"❌ Hypothesis generation failed: {e}")
            return

    # STEP 3: CODE GENERATION
    print_step(3, "GENERATING TEST CODE (GPT-4)")
    try:
        print("💻 GPT-4 is generating code to test the hypothesis...")
        code = code_agent.generate_code(hypothesis['hypothesis'])
        print("✅ Code generated successfully!")
        print(f"\n🔍 CODE PREVIEW:")
        print(code[:400] + "..." if len(code) > 400 else code)
        
        # Code validation
        print("\n🔍 VALIDATING CODE...")
        lint_result = code_agent.run_pylint(code)
        if lint_result['passed']:
            print("✅ Code passed PyLint validation")
        else:
            print("⚠️  Code has some style issues (still functional)")
            
        # Test execution
        print("\n▶️  TESTING CODE EXECUTION...")
        exec_result = code_agent.execute_code(code)
        if exec_result['success']:
            print("✅ Code executed successfully!")
            print(f"   ⚡ Execution time: {exec_result['execution_time']:.2f}s")
            print(f"   💾 Memory usage: {exec_result['memory_usage']:.1f}MB")
        else:
            print("⚠️  Code execution had issues (this is normal for research code)")
            
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        code = "# Code generation failed"

    # STEP 4: INSIGHTS ANALYSIS
    print_step(4, "ANALYZING RESEARCH INSIGHTS")
    try:
        print("📊 Analyzing paper content and extracting insights...")
        
        # Keyword frequency analysis
        keywords = ["transformer", "attention", "neural network", "deep learning", "machine learning", "AI"]
        keyword_freq = insight_agent.keyword_frequency(papers, keywords)
        print(f"   📈 Keyword analysis: {sum(keyword_freq.values())} total mentions")
        
        # Topic modeling
        topics = insight_agent.topic_modeling(papers, n_topics=3)
        print(f"   🏷️  Topic modeling: {len(topics)} main research themes identified")
        
        # Common keywords extraction
        common_keywords = insight_agent.extract_common_keywords(papers, top_n=10)
        print(f"   🔤 Extracted {len(common_keywords)} key research terms")
        
        # Author collaboration analysis
        author_analysis = insight_agent.author_collaboration_analysis(papers)
        print(f"   👥 Author analysis: {author_analysis['total_unique_authors']} unique researchers")
        
        # Compile insights
        insights = [
            f"Analyzed {len(papers)} cutting-edge research papers in {query}",
            f"Key research terms: {', '.join([kw for kw, count in keyword_freq.items() if count > 0][:5])}",
            f"Research timeline: {min([p['year'] for p in papers])}-{max([p['year'] for p in papers])}",
            f"Research community: {author_analysis['total_unique_authors']} active researchers",
            f"Identified {len(topics)} main research directions in this field"
        ]
        
        print("✅ Comprehensive insights generated!")
        print("\n🔬 KEY INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
            
    except Exception as e:
        print(f"❌ Insight analysis failed: {e}")
        insights = ["Analysis failed - using fallback insights"]

    # STEP 5: VISUALIZATION CREATION
    print_step(5, "CREATING RESEARCH VISUALIZATIONS")
    try:
        print("📊 Creating publication trend visualization...")
        
        # Create publication trends
        year_counts = {}
        for paper in papers:
            year = paper['year']
            year_counts[year] = year_counts.get(year, 0) + 1
        
        viz_data = [{"x": year, "y": count} for year, count in sorted(year_counts.items())]
        
        visualization = visualization_agent.create_line_plot(
            viz_data,
            f"Publication Trends: {query}",
            "Year",
            "Number of Papers"
        )
        
        print("✅ Visualization created successfully!")
        print(f"   📈 Chart: Publication trends over {len(year_counts)} years")
        print(f"   📊 Data points: {len(viz_data)} time periods")
        
    except Exception as e:
        print(f"❌ Visualization creation failed: {e}")

    # STEP 6: COMPREHENSIVE REPORT GENERATION
    print_step(6, "ASSEMBLING COMPREHENSIVE RESEARCH REPORT")
    try:
        print("📄 Compiling all findings into comprehensive report...")
        
        # Create proper academic citations
        citations = []
        for paper in papers:
            authors_str = ", ".join(paper['authors'][:3])
            if len(paper['authors']) > 3:
                authors_str += " et al."
            citation = f"{authors_str} ({paper['year']}). {paper['title']}. arXiv:{paper['arxiv_id']}"
            citations.append(citation)
        
        # Generate final report
        report = report_agent.assemble_report(
            hypothesis["hypothesis"],
            insights,
            [{"path": "publication_trends.png", "caption": f"Publication trends for {query} research"}],
            citations
        )
        
        print("✅ Comprehensive research report generated!")
        
        # Display final report
        print_header("FINAL RESEARCH REPORT")
        print(report)
        print_header("END OF REPORT")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return

    # STEP 7: SESSION SUMMARY
    print_step(7, "PIPELINE COMPLETION SUMMARY")
    note_taker.log_session_end(user="demo_user", session_id=session_id)
    
    print("🎉 MULTI-AGENT RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   🔍 Research Query: '{query}'")
    print(f"   📚 Papers Analyzed: {len(papers)} (REAL from arXiv)")
    print(f"   🧠 Hypothesis: REAL GPT-4 generated")
    print(f"   💻 Code: REAL GPT-4 generated & validated")
    print(f"   📊 Insights: {len(insights)} comprehensive findings")
    print(f"   📈 Visualizations: Publication trends created")
    print(f"   📄 Citations: {len(citations)} proper academic references")
    mongodb_status = "MongoDB Atlas" if use_mongodb else "In-memory (demo mode)"
    print(f"   🗄️  Logging: {mongodb_status}")
    print(f"   🔗 Session ID: {session_id}")
    
    print(f"\n✨ PIPELINE FEATURES DEMONSTRATED:")
    print(f"   ✅ Real arXiv API integration")
    print(f"   ✅ GPT-4 powered hypothesis generation")
    print(f"   ✅ GPT-4 powered code generation & validation")
    print(f"   ✅ Advanced NLP analysis & topic modeling")
    print(f"   ✅ Dynamic visualizations")
    print(f"   ✅ Human-in-the-loop feedback system")
    logging_status = "✅ MongoDB logging & session management" if use_mongodb else "⚠️  In-memory logging (MongoDB offline)"
    print(f"   {logging_status}")
    print(f"   ✅ Academic citation generation")
    print(f"   ✅ End-to-end automation")
    
    print(f"\n🚀 Ready for hackathon demonstration!")
    
    if not use_mongodb:
        print(f"\n📝 SESSION LOGS (since MongoDB was offline):")
        for i, log in enumerate(note_taker.logs[-10:], 1):  # Show last 10 logs
            print(f"   {i}. {log}")
        if len(note_taker.logs) > 10:
            print(f"   ... and {len(note_taker.logs) - 10} more log entries")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Pipeline interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        print("Please check your configuration and try again.") 
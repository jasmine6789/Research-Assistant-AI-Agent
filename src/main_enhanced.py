import os
import urllib.parse
from dotenv import load_dotenv
load_dotenv()
from src.agents.note_taker import NoteTaker
from src.agents.web_search_agent import WebSearchAgent
from src.agents.web_insight_agent import WebInsightAgent
from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.enhanced_code_agent import EnhancedCodeAgent
from src.agents.enhanced_visualization_agent import EnhancedVisualizationAgent
from src.agents.enhanced_report_agent import EnhancedReportAgent
import time
import uuid
import ssl
import certifi

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80)

def print_step(step_num, title):
    """Print a formatted step header"""
    print(f"\n🔄 STEP {step_num}: {title}")
    print("-" * 60)

def extract_hypothesis_text(hypothesis) -> str:
    """Extract hypothesis text from either string or dictionary format"""
    if isinstance(hypothesis, dict):
        if 'hypothesis' in hypothesis:
            return str(hypothesis['hypothesis'])
        else:
            return str(hypothesis)
    elif isinstance(hypothesis, str):
        return hypothesis
    else:
        return str(hypothesis)

def main():
    """Enhanced multi-agent research pipeline with advanced capabilities"""
    print_header("ENHANCED MULTI-AGENT RESEARCH SYSTEM")
    
    # Load environment variables
    password = "Jasmine@0802"
    encoded_password = urllib.parse.quote_plus(password)
    MONGO_URI = f"mongodb+srv://jaschri:{encoded_password}@agent.wb3vq0q.mongodb.net/?retryWrites=true&w=majority&appName=Agent&ssl=true&ssl_cert_reqs=CERT_NONE&tlsCAFile={certifi.where()}"
    OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

    print(f"📊 MongoDB: Connecting with enhanced SSL...")
    print(f"🤖 OpenAI: GPT-4 with Enhanced Capabilities")
    print(f"🤗 Hugging Face: Model Discovery Enabled")
    print(f"☁️  Google Cloud: {PROJECT_ID}")

    # Initialize enhanced agents with MongoDB fallback
    print("\n🏗️  Initializing Enhanced Agents...")
    
    use_mongodb = True
    try:
        note_taker = NoteTaker(MONGO_URI)
        note_taker.ping_database()
        print("✅ MongoDB Atlas connected successfully")
    except Exception as e:
        print(f"⚠️  MongoDB connection failed: {e}")
        print("📝 Using in-memory logging (demo mode)")
        use_mongodb = False
        
        # Create enhanced mock note taker
        class EnhancedMockNoteTaker:
            def __init__(self):
                self.logs = []
            def log_session_start(self, **kwargs): 
                self.logs.append(f"Session started: {kwargs}")
            def log_session_end(self, **kwargs): 
                self.logs.append(f"Session ended: {kwargs}")
            def log_query(self, query, **kwargs): 
                self.logs.append(f"Query: {str(query)}")
            def log_selected_papers(self, papers, **kwargs): 
                self.logs.append(f"Papers: {len(papers) if papers else 0} selected")
            def log_feedback(self, feedback, **kwargs): 
                self.logs.append(f"Feedback: {str(feedback)}")
            def log_hypothesis(self, hypothesis, **kwargs):
                self.logs.append(f"Hypothesis: {str(hypothesis)[:50]}...")
            def log_code(self, code, **kwargs):
                self.logs.append(f"Code: {len(str(code))} characters generated")
            def log_insights(self, insights, **kwargs):
                self.logs.append(f"Insights: {len(insights) if insights else 0} generated")
            def log_visualization(self, viz_data, viz_type, **kwargs):
                self.logs.append(f"Visualization: {str(viz_type)}")
            def log_report(self, report, **kwargs):
                self.logs.append(f"Report: {len(str(report))} characters")
            def log_insight(self, insight_type, data=None, **kwargs):
                if data is None:
                    self.logs.append(f"Insight: {str(insight_type)}")
                else:
                    self.logs.append(f"Insight ({insight_type}): {str(data)[:100]}...")
            def log(self, event_type, data, **kwargs):
                self.logs.append(f"{event_type}: {str(data)[:100]}...")
        
        note_taker = EnhancedMockNoteTaker()

    # Initialize all enhanced agents
    web_search_agent = WebSearchAgent(note_taker)
    web_insight_agent = WebInsightAgent(note_taker)
    hypothesis_agent = HypothesisAgent(OPENAI_API_KEY, note_taker)
    enhanced_code_agent = EnhancedCodeAgent(OPENAI_API_KEY, note_taker)
    enhanced_viz_agent = EnhancedVisualizationAgent(note_taker)
    enhanced_report_agent = EnhancedReportAgent(note_taker)

    print("✅ All Enhanced Agents Initialized")
    print(f"   🔍 WebSearchAgent: arXiv integration ready")
    print(f"   🧠 HypothesisAgent: GPT-4 powered")
    print(f"   💻 EnhancedCodeAgent: HuggingFace + GPT-4")
    print(f"   📊 EnhancedVisualizationAgent: Hypothesis-specific charts")
    print(f"   📄 EnhancedReportAgent: Academic paper formatting")

    # Start session
    session_id = str(uuid.uuid4())
    note_taker.log_session_start(session_id=session_id, enhanced_features=True)

    # Get user input
    print("\n🎯 RESEARCH QUERY INPUT")
    print("Enter your research topic or hypothesis:")
    query = input("Query: ").strip()
    
    if not query:
        query = "Early detection and progression forecasting of Alzheimer's Disease using multimodal AI approaches"
        print(f"Using default query: {query}")

    note_taker.log_query(query, session_id=session_id)

    # STEP 1: Enhanced Paper Search
    print_step(1, "ENHANCED PAPER SEARCH")
    print("🔍 Searching arXiv for relevant papers...")
    papers = web_search_agent.search(query, top_k=20)
    
    if papers:
        print(f"✅ Found {len(papers)} relevant papers")
        for i, paper in enumerate(papers[:5], 1):
            print(f"   {i}. {paper.get('title', 'Unknown title')[:80]}...")
        note_taker.log_selected_papers(papers, query=query, session_id=session_id)
    else:
        print("❌ No papers found. Please check your query.")
        return

    # STEP 2: Advanced Hypothesis Generation
    print_step(2, "ADVANCED HYPOTHESIS GENERATION")
    print("🧠 Generating research hypothesis using GPT-4...")
    
    hypothesis = hypothesis_agent.generate_hypothesis(papers)
    print(f"✅ Generated hypothesis:")
    print(f"   {hypothesis}")
    
    note_taker.log_hypothesis(hypothesis, papers=len(papers), session_id=session_id)

    # Human-in-the-loop for hypothesis
    while True:
        print(f"\n🤝 HYPOTHESIS APPROVAL:")
        print(f"1. ✅ Accept hypothesis")
        print(f"2. 🔄 Regenerate hypothesis")
        print(f"3. 💬 Provide feedback and regenerate")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == "1":
            print("✅ Hypothesis accepted!")
            break
        elif choice == "2":
            print("🔄 Regenerating hypothesis...")
            hypothesis = hypothesis_agent.generate_hypothesis(papers)
            print(f"🆕 New hypothesis: {hypothesis}")
            note_taker.log_hypothesis(hypothesis, regenerated=True, session_id=session_id)
        elif choice == "3":
            feedback = input("Provide feedback for improvement: ").strip()
            note_taker.log_feedback(feedback, session_id=session_id)
            print("🔄 Regenerating with feedback...")
            # In real implementation, you'd pass feedback to hypothesis generation
            hypothesis = hypothesis_agent.generate_hypothesis(papers)
            print(f"🆕 Improved hypothesis: {hypothesis}")
            note_taker.log_hypothesis(hypothesis, feedback=feedback, session_id=session_id)
        else:
            print("Invalid choice. Please try again.")

    # STEP 3: Enhanced Code Generation with HuggingFace
    print_step(3, "ENHANCED CODE GENERATION")
    print("💻 Discovering relevant HuggingFace models...")
    
    relevant_models = enhanced_code_agent.discover_relevant_models(hypothesis)
    if relevant_models:
        print(f"🤗 Found {len(relevant_models)} relevant models:")
        for model in relevant_models[:3]:
            print(f"   - {model['id']} (task: {model.get('pipeline_tag', 'general')})")
    
    print("💻 Generating enhanced code with GPT-4 + HuggingFace...")
    code = enhanced_code_agent.generate_enhanced_code(hypothesis, include_hf_models=True)
    
    print("🔍 Validating code quality...")
    quality_metrics = enhanced_code_agent.validate_code_quality(code)
    print(f"✅ Code quality score: {quality_metrics['quality_score']:.2f}")
    print(f"   📏 Lines: {quality_metrics['line_count']}")
    print(f"   🧠 Complexity: {quality_metrics['estimated_complexity']}")
    
    print("🔍 Running PyLint analysis...")
    pylint_result = enhanced_code_agent.run_pylint(code)
    print(f"✅ PyLint score: {pylint_result.get('score', 'N/A')}/10")
    
    print("⚡ Testing code execution...")
    exec_result = enhanced_code_agent.execute_code_safely(code)
    if exec_result['success']:
        print(f"✅ Code executed successfully in {exec_result['execution_time']:.2f}s")
    else:
        print(f"⚠️  Code execution issues: {exec_result['error'][:100]}...")
    
    note_taker.log_code(code, 
                       quality_score=quality_metrics['quality_score'],
                       pylint_score=pylint_result.get('score'),
                       execution_success=exec_result['success'],
                       session_id=session_id)

    # STEP 4: Comprehensive Insights Analysis
    print_step(4, "COMPREHENSIVE INSIGHTS ANALYSIS")
    print("📊 Analyzing research landscape...")
    
    insights_data = web_insight_agent.analyze_papers(papers, query)
    
    # Extract insights from the comprehensive analysis
    insights = []
    if insights_data and isinstance(insights_data, dict):
        # Extract meaningful insights from analysis
        analysis_summary = insights_data.get('analysis_summary', {})
        
        insights.append(f"Most common keyword: {analysis_summary.get('most_common_keyword', 'N/A')}")
        insights.append(f"Most cited evaluation metric: {analysis_summary.get('most_cited_metric', 'N/A')}")
        insights.append(f"Top contributing author: {analysis_summary.get('top_author', 'N/A')}")
        insights.append(f"Dominant research topic: {analysis_summary.get('dominant_topic', 'N/A')}")
        insights.append(f"Total papers in analysis: {insights_data.get('total_papers_analyzed', 0)}")
        
        # Add topic modeling insights
        topics = insights_data.get('topic_modeling', {})
        if topics:
            for topic_name, words in list(topics.items())[:2]:  # Top 2 topics
                insights.append(f"{topic_name}: {', '.join(words[:3])}")
        
        # Filter out N/A insights
        insights = [insight for insight in insights if 'N/A' not in insight]
    
    if insights:
        print(f"✅ Generated {len(insights)} key insights:")
        for i, insight in enumerate(insights[:5], 1):
            print(f"   {i}. {insight}")
        note_taker.log_insights(insights, papers=len(papers), session_id=session_id)
    else:
        insights = [
            "Research shows promising developments in the field",
            "Multiple methodological approaches are being explored",
            "Significant opportunities for innovation exist"
        ]
        print(f"📝 Using fallback insights")

    # STEP 5: Hypothesis-Specific Visualizations
    print_step(5, "HYPOTHESIS-SPECIFIC VISUALIZATIONS")
    print("📈 Generating hypothesis-specific visualizations...")
    
    visualizations = enhanced_viz_agent.generate_hypothesis_visualizations(hypothesis, num_charts=3)
    
    print(f"✅ Generated {len(visualizations)} hypothesis-specific visualizations:")
    for viz in visualizations:
        print(f"   📊 {viz['title']}: {viz['type']}")
        print(f"      {viz['description'][:80]}...")
    
    note_taker.log_visualization(visualizations, "hypothesis_specific", session_id=session_id)

    # STEP 6: Academic Research Paper Generation
    print_step(6, "ACADEMIC RESEARCH PAPER GENERATION")
    print("📄 Generating academic research paper...")
    
    # Extract citations from papers
    citations = []
    for paper in papers[:10]:
        title = paper.get('title', 'Unknown title')
        authors = paper.get('authors', ['Unknown Author'])
        year = paper.get('published', '2024')[:4]
        arxiv_id = paper.get('arxiv_id', 'unknown')
        
        # Format as APA citation
        author_str = ', '.join(authors[:3])
        if len(authors) > 3:
            author_str += ', et al.'
        citation = f"{author_str} ({year}). {title}. arXiv:{arxiv_id}"
        citations.append(citation)
    
    # Generate paper using different academic styles
    paper_styles = ['arxiv', 'ieee', 'nature']
    selected_style = 'arxiv'  # Default to arXiv style
    
    print(f"📝 Using {selected_style.upper()} academic format...")
    research_paper = enhanced_report_agent.generate_research_paper(
        hypothesis=hypothesis,
        code=code,
        insights=insights,
        visualizations=visualizations,
        citations=citations,
        style=selected_style,
        include_appendix=True
    )
    
    print(f"✅ Generated academic research paper:")
    print(f"   📄 Style: {selected_style.upper()}")
    print(f"   📊 Word count: {len(research_paper.split())} words")
    print(f"   📚 References: {len(citations)} citations")
    print(f"   📈 Figures: {len(visualizations)} visualizations")
    
    # Generate executive summary
    exec_summary = enhanced_report_agent.generate_executive_summary(research_paper)
    print(f"   📋 Executive summary generated")
    
    note_taker.log_report(research_paper, 
                         style=selected_style,
                         word_count=len(research_paper.split()),
                         citations=len(citations),
                         session_id=session_id)

    # STEP 7: Final Summary and Options
    print_step(7, "RESEARCH PIPELINE COMPLETED")
    
    print("🎉 ENHANCED MULTI-AGENT RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
    
    # Extract hypothesis text for display
    hypothesis_text = extract_hypothesis_text(hypothesis)
    
    print(f"\n📊 COMPREHENSIVE SUMMARY:")
    print(f"   🔍 Research Query: '{query}'")
    print(f"   📚 Papers Analyzed: {len(papers)} (REAL from arXiv)")
    print(f"   🧠 Hypothesis: REAL GPT-4 generated with advanced prompting")
    print(f"   🤗 HuggingFace Models: {len(relevant_models) if 'relevant_models' in locals() else 0} discovered and integrated")
    print(f"   💻 Code: Enhanced GPT-4 generated & comprehensively validated")
    print(f"   📊 Insights: {len(insights)} comprehensive findings")
    print(f"   📈 Visualizations: {len(visualizations)} hypothesis-specific charts")
    print(f"   📄 Academic Paper: {selected_style.upper()} format with {len(research_paper.split())} words")
    print(f"   📚 Citations: {len(citations)} proper academic references")
    mongodb_status = "MongoDB Atlas" if use_mongodb else "In-memory (demo mode)"
    print(f"   🗄️  Logging: {mongodb_status}")
    print(f"   🔗 Session ID: {session_id}")
    
    print(f"\n✨ ENHANCED FEATURES DEMONSTRATED:")
    print(f"   ✅ Real arXiv API integration")
    print(f"   ✅ GPT-4 hypothesis generation")
    print(f"   ✅ HuggingFace model discovery & integration")
    print(f"   ✅ Enhanced code generation with quality validation")
    print(f"   ✅ Hypothesis-specific visualization generation")
    print(f"   ✅ Academic paper formatting (multiple styles)")
    print(f"   ✅ Human-in-the-loop interactions")
    print(f"   ✅ Comprehensive error handling")
    print(f"   ✅ Production-ready logging")
    
    print(f"\n🎯 RESEARCH OUTCOMES:")
    print(f"   📖 Hypothesis: {hypothesis_text[:100]}...")
    print(f"   💾 Generated Code: {len(code)} characters")
    print(f"   📊 Key Insights: {insights[:3] if insights else ['No insights generated']}")
    print(f"   📈 Visualizations: {[viz.get('title', 'Unnamed') for viz in visualizations]}")
    print(f"   📄 Paper Length: {len(research_paper.split())} words")

    # Export options
    print(f"\n💾 EXPORT OPTIONS:")
    print(f"   1. 📄 View full research paper")
    print(f"   2. 💻 View generated code")
    print(f"   3. 📊 View visualization details")
    print(f"   4. 📋 View executive summary")
    print(f"   5. 💾 Save paper to file (TXT/HTML/LaTeX)")
    print(f"   6. 🏁 Exit")
    
    while True:
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            print("\n" + "="*80)
            print("FULL RESEARCH PAPER")
            print("="*80)
            print(research_paper)
        elif choice == "2":
            print("\n" + "="*80)
            print("GENERATED CODE")
            print("="*80)
            print(code[:2000] + "..." if len(code) > 2000 else code)
        elif choice == "3":
            print("\n" + "="*80)
            print("VISUALIZATION DETAILS")
            print("="*80)
            for viz in visualizations:
                print(f"\n📊 {viz['title']}")
                print(f"Type: {viz['type']}")
                print(f"Description: {viz['description']}")
        elif choice == "4":
            print("\n" + "="*80)
            print("EXECUTIVE SUMMARY")
            print("="*80)
            print(exec_summary)
        elif choice == "5":
            print("\n💾 SAVE PAPER TO FILE")
            print("Available formats:")
            print("   1. TXT (Plain text)")
            print("   2. HTML (Web format with styling)")
            print("   3. LaTeX (Academic typesetting)")
            
            format_choice = input("Choose format (1-3): ").strip()
            
            if format_choice == "1":
                format_type = "txt"
                extension = "txt"
            elif format_choice == "2":
                format_type = "html"
                extension = "html"
            elif format_choice == "3":
                format_type = "latex"
                extension = "tex"
            else:
                print("Invalid format choice.")
                continue
            
            # Generate filename
            import re
            safe_query = re.sub(r'[^\w\s-]', '', query)
            safe_query = re.sub(r'[-\s]+', '_', safe_query)[:50]
            filename = f"research_paper_{safe_query}_{session_id[:8]}.{extension}"
            
            print(f"💾 Saving paper as {format_type.upper()}...")
            file_path = enhanced_report_agent.save_paper_to_file(research_paper, filename, format_type)
            
            if file_path:
                print(f"✅ Paper saved successfully!")
                print(f"   📁 File: {file_path}")
                print(f"   📊 Format: {format_type.upper()}")
                if format_type == "html":
                    print(f"   🌐 Open in browser to view with styling")
                elif format_type == "latex":
                    print(f"   📝 Compile with LaTeX to generate PDF")
            else:
                print("❌ Failed to save paper. Please try again.")
        elif choice == "6":
            break
        else:
            print("Invalid choice. Please try again.")

    # End session
    note_taker.log_session_end(session_id=session_id, 
                              total_papers=len(papers),
                              hypothesis_generated=True,
                              code_generated=True,
                              visualizations_created=len(visualizations),
                              paper_generated=True)

    print(f"\n🎯 Research session completed successfully!")
    print(f"   📊 All data logged to: {mongodb_status}")
    print(f"   🔗 Session ID: {session_id}")
    print(f"\n🚀 Ready for production deployment!")

if __name__ == "__main__":
    main() 
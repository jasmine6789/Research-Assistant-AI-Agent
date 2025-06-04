import datetime
from typing import List, Dict, Optional, Any
import markdown2
import os
from flask import Flask, request, jsonify
from src.agents.note_taker import NoteTaker

try:
    from scholarly import scholarly
except ImportError:
    scholarly = None

class ReportAgent:
    def __init__(self, note_taker: NoteTaker):
        self.note_taker = note_taker
        self.templates = {
            'default': self.default_template,
            'conference': self.conference_template,
            'journal': self.journal_template,
            'summary': self.summary_template
        }
        self.branding = {
            'logo_url': None,
            'color_scheme': 'default'
        }

    # --- Citation Augmentation ---
    def fetch_citation_count(self, title: str) -> Optional[int]:
        """Fetch citation count from Google Scholar using scholarly."""
        if scholarly is None:
            return None
        try:
            search = next(scholarly.search_pubs(title))
            return search.get('num_citations', None)
        except Exception:
            return None

    def generate_apa_citation(self, metadata: Dict, citation_count: Optional[int] = None) -> str:
        """
        Generate an APA-style citation from arXiv metadata, with DOI and citation count if available.
        """
        authors = metadata.get('authors', [])
        year = metadata.get('year', datetime.datetime.now().year)
        title = metadata.get('title', 'Untitled')
        arxiv_id = metadata.get('arxiv_id', '')
        doi = metadata.get('doi', None)
        formatted_authors = self.format_authors(authors)
        citation = f"{formatted_authors} ({year}). {title}. "
        if doi:
            citation += f"https://doi.org/{doi}"
        else:
            citation += f"arXiv:{arxiv_id}"
        if citation_count is not None:
            citation += f" [Cited by {citation_count} on Google Scholar]"
        self.note_taker.log("citation_generated", {"citation": citation})
        return citation

    # --- Enhanced Author Formatting ---
    def format_authors(self, authors: List[str], max_authors: int = 5) -> str:
        """Format author list for APA, using 'et al.' if too long, handle consortiums and non-ASCII."""
        if not authors:
            return "Anonymous"
        if any('consortium' in a.lower() for a in authors):
            return ', '.join(authors)
        formatted = [self.format_author(a) for a in authors[:max_authors]]
        if len(authors) > max_authors:
            formatted.append('et al.')
        return ', '.join(formatted)

    def format_author(self, author: str) -> str:
        """Format author name as Last, F. (handles multiple middle names, non-ASCII, etc.)"""
        if ',' in author:
            return author  # Already formatted
        parts = author.split()
        if len(parts) == 1:
            return parts[0]
        last = parts[-1]
        initials = [p[0] + '.' for p in parts[:-1] if p]
        return f"{last}, {' '.join(initials)}"

    # --- Visualization Embedding ---
    def embed_visualization(self, viz_path: str, caption: Optional[str] = None, as_html: bool = False) -> str:
        """Embed visualization as HTML <img> or Markdown image, with optional caption."""
        if as_html:
            img_tag = f'<img src="{viz_path}" alt="Visualization">'
            if caption:
                img_tag += f'<div><em>{caption}</em></div>'
            return img_tag
        else:
            md = f"![Visualization]({viz_path})"
            if caption:
                md += f"\n*{caption}*"
            return md

    # --- Report Assembly and Templates ---
    def assemble_report(self, hypothesis: str, insights: List[str], visualizations: List[Dict[str, Any]], citations: List[str], notes: Optional[List[str]] = None, metadata: Optional[Dict] = None, template: str = 'default', section_order: Optional[List[str]] = None) -> str:
        """
        Assemble the research report as a Markdown string, supporting templates, metadata, and section order.
        visualizations: list of dicts with keys 'path', 'caption', 'as_html'
        """
        if not section_order:
            section_order = ['metadata', 'hypothesis', 'insights', 'visualizations', 'citations', 'notes']
        sections = {
            'metadata': self.render_metadata(metadata),
            'hypothesis': f"## Hypothesis\n{hypothesis}\n",
            'insights': self.render_insights(insights),
            'visualizations': self.render_visualizations(visualizations),
            'citations': self.render_citations(citations),
            'notes': self.render_notes(notes)
        }
        # Use template
        template_func = self.templates.get(template, self.default_template)
        report = template_func(sections, section_order)
        self.note_taker.log("report_assembled", {"template": template, "sections": list(sections.keys())})
        return report

    def render_metadata(self, metadata: Optional[Dict]) -> str:
        if not metadata:
            return ""
        lines = ["## Metadata"]
        for k, v in metadata.items():
            lines.append(f"- **{k.capitalize()}**: {v}")
        lines.append(f"- **Generated**: {datetime.datetime.now().isoformat()}")
        return '\n'.join(lines) + '\n'

    def render_insights(self, insights: List[str]) -> str:
        if not insights:
            return ""
        return "## Insights\n" + '\n'.join(f"- {i}" for i in insights) + '\n'

    def render_visualizations(self, visualizations: List[Dict[str, Any]]) -> str:
        if not visualizations:
            return ""
        lines = ["## Visualizations"]
        for viz in visualizations:
            lines.append(self.embed_visualization(viz['path'], viz.get('caption'), viz.get('as_html', False)))
        return '\n'.join(lines) + '\n'

    def render_citations(self, citations: List[str]) -> str:
        if not citations:
            return ""
        return "## Citations\n" + '\n'.join(f"- {c}" for c in citations) + '\n'

    def render_notes(self, notes: Optional[List[str]]) -> str:
        if not notes:
            return ""
        return "## Notes\n" + '\n'.join(f"- {n}" for n in notes) + '\n'

    # --- Templates ---
    def default_template(self, sections, order):
        return "# Research Report\n\n" + '\n'.join(sections[s] for s in order if sections[s])

    def conference_template(self, sections, order):
        return "# Conference Paper\n\n" + '\n'.join(sections[s] for s in order if sections[s])

    def journal_template(self, sections, order):
        return "# Journal Article\n\n" + '\n'.join(sections[s] for s in order if sections[s])

    def summary_template(self, sections, order):
        return "# Executive Summary\n\n" + '\n'.join(sections[s] for s in order if sections[s])

    # --- HTML and PDF Export ---
    def to_html(self, markdown_report: str) -> str:
        """Convert Markdown report to HTML."""
        return markdown2.markdown(markdown_report)

    def to_pdf(self, markdown_report: str, output_path: str) -> Optional[str]:
        """Convert Markdown report to PDF using WeasyPrint. Returns output path or None."""
        try:
            from weasyprint import HTML
        except ImportError:
            return None
        try:
            html = self.to_html(markdown_report)
            HTML(string=html).write_pdf(output_path)
            self.note_taker.log("report_exported", {"format": "pdf", "output_path": output_path})
            return output_path
        except Exception:
            return None

    # --- Feedback and Revision Logging ---
    def append_feedback(self, report: str, feedback: str) -> str:
        """Append user feedback to the report and log with NoteTaker if available."""
        self.note_taker.log_feedback(feedback)
        return report + f"\n\n## User Feedback\n- {feedback}\n"

    # --- API Integration (Flask) ---
    def register_api(self, app: Flask):
        """Register report generation endpoint with a Flask app."""
        @app.route('/api/generate_report', methods=['POST'])
        def generate_report_api():
            data = request.json
            hypothesis = data.get('hypothesis', '')
            insights = data.get('insights', [])
            visualizations = data.get('visualizations', [])
            citations = data.get('citations', [])
            notes = data.get('notes', [])
            metadata = data.get('metadata', {})
            template = data.get('template', 'default')
            section_order = data.get('section_order', None)
            report = self.assemble_report(hypothesis, insights, visualizations, citations, notes, metadata, template, section_order)
            return jsonify({'report': report})

# Example usage (to be removed in production)
if __name__ == "__main__":
    MONGO_URI = os.getenv("MONGO_URI")
    note_taker = NoteTaker(MONGO_URI)
    agent = ReportAgent(note_taker)
    report = agent.assemble_report("Test hypothesis", ["Test insight"], [], [])
    print(report) 
"""
Template Management System for Research Papers

Features:
- Multiple academic formats (IEEE, ACM, arXiv, Nature, etc.)
- Jinja2-based templating with custom filters
- Dynamic section generation
- Bibliography formatting
- Custom template creation and validation
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TemplateManager:
    """
    Comprehensive template management for academic papers
    """
    
    def __init__(self, templates_dir: str = "templates"):
        """
        Initialize template manager
        
        Args:
            templates_dir: Directory containing template files
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self._add_custom_filters()
        
        # Create default templates if they don't exist
        self._create_default_templates()
        
        # Load template configurations
        self.template_configs = self._load_template_configs()
    
    def _add_custom_filters(self):
        """Add custom Jinja2 filters for academic formatting"""
        
        def format_authors(authors):
            """Format author list according to academic standards"""
            if not authors:
                return "Unknown Author"
            
            if isinstance(authors, str):
                return authors
            
            if len(authors) == 1:
                return authors[0]
            elif len(authors) == 2:
                return f"{authors[0]} and {authors[1]}"
            elif len(authors) <= 5:
                return ", ".join(authors[:-1]) + f", and {authors[-1]}"
            else:
                return ", ".join(authors[:3]) + " et al."
        
        def format_date(date_str, format_type="academic"):
            """Format dates for academic papers"""
            if not date_str:
                return datetime.now().strftime("%B %Y")
            
            try:
                if isinstance(date_str, str):
                    # Parse various date formats
                    for fmt in ["%Y-%m-%d", "%Y-%m", "%Y"]:
                        try:
                            date_obj = datetime.strptime(date_str[:len(fmt)], fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        return date_str
                else:
                    date_obj = date_str
                
                if format_type == "academic":
                    return date_obj.strftime("%B %Y")
                elif format_type == "iso":
                    return date_obj.strftime("%Y-%m-%d")
                else:
                    return date_obj.strftime("%B %d, %Y")
                    
            except Exception:
                return str(date_str)
        
        def format_citation(paper, style="apa"):
            """Format paper citation in specified style"""
            if not paper:
                return ""
            
            title = paper.get('title', 'Unknown Title')
            authors = paper.get('authors', ['Unknown Author'])
            year = paper.get('published', datetime.now().year)
            
            if isinstance(year, str):
                year = year[:4]
            
            formatted_authors = format_authors(authors)
            
            if style.lower() == "apa":
                arxiv_id = paper.get('arxiv_id', '')
                if arxiv_id:
                    return f"{formatted_authors} ({year}). {title}. arXiv:{arxiv_id}."
                else:
                    return f"{formatted_authors} ({year}). {title}."
            
            elif style.lower() == "ieee":
                return f'{formatted_authors}, "{title}," {year}.'
            
            elif style.lower() == "nature":
                return f"{formatted_authors}. {title}. Preprint at https://arxiv.org/abs/{paper.get('arxiv_id', '')} ({year})."
            
            else:  # Default APA
                return f"{formatted_authors} ({year}). {title}."
        
        def format_abstract(abstract, max_words=250):
            """Format and truncate abstract"""
            if not abstract:
                return ""
            
            words = abstract.split()
            if len(words) <= max_words:
                return abstract
            
            truncated = " ".join(words[:max_words])
            return truncated + "..."
        
        def format_keywords(keywords):
            """Format keywords list"""
            if not keywords:
                return ""
            
            if isinstance(keywords, str):
                return keywords
            
            return ", ".join(keywords)
        
        def format_section_number(sections, current_section):
            """Generate section numbering"""
            try:
                index = sections.index(current_section)
                return f"{index + 1}."
            except (ValueError, AttributeError):
                return ""
        
        # Register filters
        self.env.filters['format_authors'] = format_authors
        self.env.filters['format_date'] = format_date
        self.env.filters['format_citation'] = format_citation
        self.env.filters['format_abstract'] = format_abstract
        self.env.filters['format_keywords'] = format_keywords
        self.env.filters['format_section_number'] = format_section_number
    
    def _create_default_templates(self):
        """Create default academic paper templates"""
        
        # arXiv template
        arxiv_template = """
# {{ title }}

{% if authors %}
**Authors:** {{ authors | format_authors }}
{% endif %}

{% if affiliation %}
**Affiliation:** {{ affiliation }}
{% endif %}

{% if date %}
**Date:** {{ date | format_date }}
{% endif %}

## Abstract

{{ abstract | format_abstract }}

{% if keywords %}
**Keywords:** {{ keywords | format_keywords }}
{% endif %}

## 1. Introduction

{{ introduction }}

## 2. Related Work

{{ related_work }}

## 3. Methodology

{{ methodology }}

## 4. Results and Discussion

{{ results }}

## 5. Conclusion

{{ conclusion }}

{% if acknowledgments %}
## Acknowledgments

{{ acknowledgments }}
{% endif %}

## References

{% for citation in citations %}
[{{ loop.index }}] {{ citation | format_citation('apa') }}
{% endfor %}

{% if appendix %}
## Appendix

{{ appendix }}
{% endif %}
"""
        
        # IEEE template
        ieee_template = """
\\documentclass[conference]{IEEEtran}
\\usepackage{graphicx}
\\usepackage{amsmath}

\\begin{document}

\\title{ {{ title }} }

\\author{
\\IEEEauthorblockN{ {{ authors | format_authors }} }
{% if affiliation %}
\\IEEEauthorblockA{ {{ affiliation }} }
{% endif %}
}

\\maketitle

\\begin{abstract}
{{ abstract | format_abstract }}
\\end{abstract}

\\begin{IEEEkeywords}
{{ keywords | format_keywords }}
\\end{IEEEkeywords}

\\section{Introduction}
{{ introduction }}

\\section{Related Work}
{{ related_work }}

\\section{Methodology}
{{ methodology }}

\\section{Results and Discussion}
{{ results }}

\\section{Conclusion}
{{ conclusion }}

{% if acknowledgments %}
\\section{Acknowledgment}
{{ acknowledgments }}
{% endif %}

\\begin{thebibliography}{1}
{% for citation in citations %}
\\bibitem{ref{{ loop.index }}}
{{ citation | format_citation('ieee') }}
{% endfor %}
\\end{thebibliography}

\\end{document}
"""
        
        # Nature template
        nature_template = """
# {{ title }}

{% if authors %}
{{ authors | format_authors }}{% if affiliation %}<sup>1</sup>{% endif %}
{% endif %}

{% if affiliation %}
<sup>1</sup>{{ affiliation }}
{% endif %}

## Abstract

{{ abstract | format_abstract(200) }}

{% if keywords %}
**Subject terms:** {{ keywords | format_keywords }}
{% endif %}

## Introduction

{{ introduction }}

## Results

{{ results }}

## Discussion

{{ conclusion }}

## Methods

{{ methodology }}

{% if acknowledgments %}
## Acknowledgments

{{ acknowledgments }}
{% endif %}

## References

{% for citation in citations %}
{{ loop.index }}. {{ citation | format_citation('nature') }}
{% endfor %}

{% if appendix %}
## Supplementary Information

{{ appendix }}
{% endif %}
"""
        
        # ACM template
        acm_template = """
\\documentclass[sigconf]{acmart}

\\begin{document}

\\title{ {{ title }} }

{% if authors %}
\\author{ {{ authors | format_authors }} }
{% endif %}

{% if affiliation %}
\\affiliation{
  \\institution{ {{ affiliation }} }
}
{% endif %}

\\begin{abstract}
{{ abstract | format_abstract }}
\\end{abstract}

\\begin{CCSXML}
<ccs2012>
{% for keyword in keywords %}
<concept>
<concept_id>{{ keyword }}</concept_id>
<concept_desc>{{ keyword }}</concept_desc>
<concept_significance>300</concept_significance>
</concept>
{% endfor %}
</ccs2012>
\\end{CCSXML}

\\ccsdesc[300]{ {{ keywords | format_keywords }} }

\\keywords{ {{ keywords | format_keywords }} }

\\maketitle

\\section{Introduction}
{{ introduction }}

\\section{Related Work}
{{ related_work }}

\\section{Methodology}
{{ methodology }}

\\section{Results}
{{ results }}

\\section{Conclusion}
{{ conclusion }}

{% if acknowledgments %}
\\begin{acks}
{{ acknowledgments }}
\\end{acks}
{% endif %}

\\bibliographystyle{ACM-Reference-Format}
\\bibliography{references}

{% if appendix %}
\\appendix
\\section{Additional Details}
{{ appendix }}
{% endif %}

\\end{document}
"""
        
        # Save templates
        templates = {
            "arxiv.md": arxiv_template,
            "ieee.tex": ieee_template,
            "nature.md": nature_template,
            "acm.tex": acm_template
        }
        
        for filename, content in templates.items():
            template_path = self.templates_dir / filename
            if not template_path.exists():
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(content.strip())
                logger.info(f"Created default template: {filename}")
    
    def _load_template_configs(self) -> Dict[str, Any]:
        """Load template configuration metadata"""
        configs = {
            "arxiv": {
                "name": "arXiv Preprint",
                "format": "markdown",
                "extension": "md",
                "description": "Standard arXiv preprint format",
                "sections": ["introduction", "related_work", "methodology", "results", "conclusion"],
                "citation_style": "apa",
                "max_abstract_words": 250
            },
            "ieee": {
                "name": "IEEE Conference",
                "format": "latex",
                "extension": "tex",
                "description": "IEEE conference paper format",
                "sections": ["introduction", "related_work", "methodology", "results", "conclusion"],
                "citation_style": "ieee",
                "max_abstract_words": 200
            },
            "nature": {
                "name": "Nature Journal",
                "format": "markdown",
                "extension": "md",
                "description": "Nature journal article format",
                "sections": ["introduction", "results", "conclusion", "methodology"],
                "citation_style": "nature",
                "max_abstract_words": 200
            },
            "acm": {
                "name": "ACM Conference",
                "format": "latex",
                "extension": "tex",
                "description": "ACM conference paper format",
                "sections": ["introduction", "related_work", "methodology", "results", "conclusion"],
                "citation_style": "acm",
                "max_abstract_words": 150
            }
        }
        
        return configs
    
    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available templates with their metadata"""
        return self.template_configs.copy()
    
    def generate_paper(self,
                      template_name: str,
                      content: Dict[str, Any],
                      output_path: Optional[str] = None) -> str:
        """
        Generate paper using specified template
        
        Args:
            template_name: Name of template to use (e.g., 'arxiv', 'ieee')
            content: Dictionary containing paper content
            output_path: Optional path to save generated paper
            
        Returns:
            Generated paper content as string
        """
        try:
            # Get template configuration
            config = self.template_configs.get(template_name)
            if not config:
                raise ValueError(f"Template '{template_name}' not found")
            
            # Load template
            template_filename = f"{template_name}.{config['extension']}"
            template = self.env.get_template(template_filename)
            
            # Prepare context with default values
            context = {
                'title': content.get('title', 'Research Paper'),
                'authors': content.get('authors', ['Unknown Author']),
                'affiliation': content.get('affiliation', ''),
                'date': content.get('date', datetime.now()),
                'abstract': content.get('abstract', ''),
                'keywords': content.get('keywords', []),
                'introduction': content.get('introduction', ''),
                'related_work': content.get('related_work', ''),
                'methodology': content.get('methodology', ''),
                'results': content.get('results', ''),
                'conclusion': content.get('conclusion', ''),
                'acknowledgments': content.get('acknowledgments', ''),
                'citations': content.get('citations', []),
                'appendix': content.get('appendix', ''),
                'template_config': config
            }
            
            # Generate paper
            generated_paper = template.render(**context)
            
            # Save to file if path provided
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(generated_paper)
                
                logger.info(f"Paper generated and saved to: {output_path}")
            
            return generated_paper
            
        except Exception as e:
            logger.error(f"Error generating paper with template '{template_name}': {e}")
            raise
    
    def validate_template(self, template_name: str) -> Dict[str, Any]:
        """
        Validate template syntax and structure
        
        Args:
            template_name: Name of template to validate
            
        Returns:
            Validation results dictionary
        """
        try:
            config = self.template_configs.get(template_name)
            if not config:
                return {"valid": False, "error": f"Template '{template_name}' not found"}
            
            template_filename = f"{template_name}.{config['extension']}"
            template_path = self.templates_dir / template_filename
            
            if not template_path.exists():
                return {"valid": False, "error": f"Template file '{template_filename}' not found"}
            
            # Try to load and parse template
            template = self.env.get_template(template_filename)
            
            # Test render with minimal context
            test_context = {
                'title': 'Test Title',
                'authors': ['Test Author'],
                'abstract': 'Test abstract',
                'introduction': 'Test introduction',
                'methodology': 'Test methodology',
                'results': 'Test results',
                'conclusion': 'Test conclusion',
                'citations': []
            }
            
            rendered = template.render(**test_context)
            
            return {
                "valid": True,
                "template_path": str(template_path),
                "config": config,
                "rendered_length": len(rendered)
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "template_path": str(self.templates_dir / f"{template_name}.{config.get('extension', 'md')}")
            }
    
    def create_custom_template(self,
                              name: str,
                              template_content: str,
                              config: Dict[str, Any]) -> bool:
        """
        Create a custom template
        
        Args:
            name: Template name
            template_content: Jinja2 template content
            config: Template configuration
            
        Returns:
            True if created successfully
        """
        try:
            # Validate config
            required_config_keys = ["format", "extension", "description", "citation_style"]
            for key in required_config_keys:
                if key not in config:
                    raise ValueError(f"Missing required config key: {key}")
            
            # Save template file
            template_filename = f"{name}.{config['extension']}"
            template_path = self.templates_dir / template_filename
            
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
            
            # Update template configs
            self.template_configs[name] = config
            
            # Validate the new template
            validation = self.validate_template(name)
            if not validation["valid"]:
                # Remove invalid template
                template_path.unlink()
                del self.template_configs[name]
                raise ValueError(f"Invalid template: {validation['error']}")
            
            logger.info(f"Custom template '{name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating custom template '{name}': {e}")
            return False
    
    def get_template_preview(self, template_name: str) -> str:
        """
        Get a preview of template with sample data
        
        Args:
            template_name: Name of template
            
        Returns:
            Preview of rendered template
        """
        sample_content = {
            'title': 'Sample Research Paper Title',
            'authors': ['Dr. Jane Smith', 'Dr. John Doe'],
            'affiliation': 'University Research Institute',
            'abstract': 'This is a sample abstract demonstrating the template format. It provides an overview of the research methodology and key findings.',
            'keywords': ['machine learning', 'artificial intelligence', 'research'],
            'introduction': 'This section introduces the research problem and objectives...',
            'related_work': 'Previous work in this field has focused on...',
            'methodology': 'Our approach consists of the following steps...',
            'results': 'The experimental results demonstrate...',
            'conclusion': 'In conclusion, this work contributes...',
            'citations': [
                {'title': 'Sample Paper 1', 'authors': ['Author A'], 'published': '2024', 'arxiv_id': '2024.1234'},
                {'title': 'Sample Paper 2', 'authors': ['Author B', 'Author C'], 'published': '2023', 'arxiv_id': '2023.5678'}
            ]
        }
        
        return self.generate_paper(template_name, sample_content)

# Global template manager instance
_template_manager = None

def get_template_manager(templates_dir: str = "templates") -> TemplateManager:
    """Get global template manager instance"""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager(templates_dir)
    return _template_manager

# Example usage and testing
if __name__ == "__main__":
    # Test template manager
    tm = TemplateManager("test_templates")
    
    print("Available templates:")
    for name, config in tm.get_available_templates().items():
        print(f"  {name}: {config['description']}")
    
    # Test template generation
    sample_content = {
        'title': 'Advanced Machine Learning for Healthcare Applications',
        'authors': ['Dr. Sarah Johnson', 'Dr. Michael Chen'],
        'abstract': 'This paper presents novel machine learning approaches for healthcare...',
        'introduction': 'Healthcare applications present unique challenges...',
        'methodology': 'Our methodology combines deep learning with traditional approaches...',
        'results': 'Experimental results show significant improvements...',
        'conclusion': 'This work demonstrates the potential of ML in healthcare...'
    }
    
    # Generate arXiv paper
    arxiv_paper = tm.generate_paper('arxiv', sample_content)
    print(f"\nGenerated arXiv paper length: {len(arxiv_paper)} characters")
    
    # Validate templates
    for template_name in tm.get_available_templates().keys():
        validation = tm.validate_template(template_name)
        print(f"Template {template_name}: {'Valid' if validation['valid'] else 'Invalid'}") 
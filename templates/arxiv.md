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
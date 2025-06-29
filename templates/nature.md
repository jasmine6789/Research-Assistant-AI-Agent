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
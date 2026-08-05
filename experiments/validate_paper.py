# -*- coding: utf-8 -*-
"""Validate the LaTeX paper structure."""
import re
import os

PAPER_PATH = os.path.join(os.path.dirname(__file__), '..', 'paper', 'elsarticle-template-harv.tex')

with open(PAPER_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Check balanced braces
open_b = content.count('{')
close_b = content.count('}')
status = 'OK' if open_b == close_b else 'MISMATCH'
print('Braces: %d open, %d close %s' % (open_b, close_b, status))

# 2. Check sections
sections = [
    'Introduction', 'Related Work', 'Material and Methods',
    'Experimental Results', 'Discussion', 'Conclusion', 'Declarations'
]
for s in sections:
    found = '\\section{%s}' % s in content
    status = 'OK' if found else 'MISSING'
    print("Section '%s': %s" % (s, status))

# 3. Check subsections in Experimental Results
subsections = [
    'Silhouette Score Comparison', 'Ablation Study', 'Parameter Robustness',
    'Embedding Visualization', 'Performance Across Clustering Algorithms',
    'Clustering Stability', 'Additional Evaluation Criteria',
    'Statistical Significance', 'Runtime Analysis', 'Practical Improvements'
]
for s in subsections:
    found = '\\subsection{%s}' % s in content
    status = 'OK' if found else 'MISSING'
    print("Subsection '%s': %s" % (s, status))

# 4. Check tables
tables = [
    'tab:silhouette_details', 'tab:ablation_dimension', 'tab:sensitivity_walk',
    'tab:sensitivity_walks', 'tab:sensitivity_window', 'tab:stability',
    'tab:statistical', 'tab:runtime', 'tab:improvements'
]
for t in tables:
    defined = '\\label{%s}' % t in content
    referenced = '\\ref{%s}' % t in content
    d_str = 'OK' if defined else 'MISSING'
    r_str = 'OK' if referenced else 'NOT USED'
    print("Table '%s': defined=%s referenced=%s" % (t, d_str, r_str))

# 5. Check figures
figures = [
    'fig:random-walk', 'fig:silhouette_scores', 'fig:silhouette',
    'fig:comparision', 'fig:tsne', 'fig:silhouette_score_comparison',
    'fig:DBI_comparison', 'fig:CHI_comparison'
]
for f_name in figures:
    defined = '\\label{%s}' % f_name in content
    referenced = '\\ref{%s}' % f_name in content
    d_str = 'OK' if defined else 'MISSING'
    r_str = 'OK' if referenced else 'NOT USED'
    print("Figure '%s': defined=%s referenced=%s" % (f_name, d_str, r_str))

# 6. Check image files
import re
images = set(re.findall(r'includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', content))
print('\nImage references: %d' % len(images))
for img in sorted(images):
    exists = os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'paper', img))
    status = 'OK' if exists else 'MISSING'
    print('  %s: %s' % (img, status))

# 7. Check labels vs refs
label_defs = set(re.findall(r'\\label\{([^}]+)\}', content))
label_refs = set(re.findall(r'\\ref\{([^}]+)\}', content))
missing = label_refs - label_defs
if missing:
    print('\nMISSING labels: %s' % missing)
else:
    print('\nAll %d referenced labels are defined' % len(label_refs))

# 8. Check bibliography
bibitems = set(re.findall(r'\\bibitem\{([^}]+)\}', content))
citations_raw = re.findall(r'\\cite\{([^}]+)\}', content)
citation_keys = set()
for c in citations_raw:
    for key in c.split(','):
        citation_keys.add(key.strip())
missing_cites = citation_keys - bibitems
if missing_cites:
    print('MISSING citations: %s' % missing_cites)
else:
    print('All %d citations have bibliography entries (%d total)' % (len(citation_keys), len(bibitems)))

print('\nValidation complete.')

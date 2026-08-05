# -*- coding: utf-8 -*-
"""Add t-SNE figures to the LaTeX paper."""

import os

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
LATEX_PATH = os.path.join(PAPER_DIR, "elsarticle-template-harv.tex")

with open(LATEX_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Find the PCA figure and add t-SNE figures after it
pca_figure_end = content.find(r"\caption{Comparison of \textit{Student-Course Representation} (PCA-reduced) and \textit{DeepWalk embeddings} for course No.\ 5.}")
pca_figure_end = content.find(r"\end{figure}", pca_figure_end) + len(r"\end{figure}")

# Add t-SNE figures after the PCA figure
tsne_figures = r"""

\begin{figure}[t]  
\centering  
\begin{subfigure}{0.48\textwidth}  
\centering  
\includegraphics[width=\textwidth]{exp_F_tSNE_BoW_course5.png}  
\caption{t-SNE of the \textit{Student-Course Representation}.}  
\label{fig:tsnea}  
\end{subfigure}  
\hfill  
\begin{subfigure}{0.48\textwidth}  
\centering  
\includegraphics[width=\textwidth]{exp_F_tSNE_DeepWalk_course5.png}  
\caption{t-SNE of DeepWalk embeddings.}  
\label{fig:tsneb}  
\end{subfigure}  
\caption{t-SNE visualization of \textit{Student-Course Representation} and \textit{DeepWalk embeddings} for course No.\ 5, confirming clear cluster separation in the embedding space.}  
\label{fig:tsne}  
\end{figure}
"""

content = content[:pca_figure_end] + tsne_figures + content[pca_figure_end:]

with open(LATEX_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Added t-SNE figures. New file length: {len(content)}")

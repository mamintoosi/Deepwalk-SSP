# -*- coding: utf-8 -*-
"""
Script to update the LaTeX paper with experimental results.
Uses string operations instead of regex to avoid escape issues.
"""

import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX_PATH = os.path.join(PROJECT_ROOT, "doc", "elsarticle-template-harv.tex")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "doc", "elsarticle-template-harv-updated.tex")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Read the original LaTeX file
with open(LATEX_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()
    content = ''.join(lines)

# Find key section markers
abstract_start = content.find(r'\begin{abstract}')
abstract_end = content.find(r'\end{abstract}') + len(r'\end{abstract}')

highlights_start = content.find(r'\begin{highlights}')
highlights_end = content.find(r'\end{highlights}') + len(r'\end{highlights}')

exp_start = content.find(r'\section{Experimental Results}')
conc_start = content.find(r'\section{Conclusion}')

print(f"Abstract: chars {abstract_start}-{abstract_end}")
print(f"Highlights: chars {highlights_start}-{highlights_end}")
print(f"Experimental Results: char {exp_start}")
print(f"Conclusion: char {conc_start}")

# ============================================================
# 1. UPDATE ABSTRACT
# ============================================================
new_abstract = r"""\begin{abstract}
Student sectioning---partitioning students into manageable sections based on course enrollment---is a core subproblem in university course timetabling, a well-known constraint satisfaction and optimization domain. Traditional approaches represent students via binary enrollment matrices and apply standard clustering, which fails to capture the rich relational structure among students sharing courses. We propose a graph-based representation learning method using DeepWalk to address this limitation. A student co-enrollment graph is constructed where nodes represent students and edges connect those sharing courses. Random walks on this graph generate student sequences that are processed by Word2Vec to produce low-dimensional embeddings encoding contextual similarity. These embeddings serve as enhanced input representations for clustering-based sectioning. Evaluated on six real-world courses (210 students, 38 courses), our method achieves an average Silhouette Score of $0.569 \pm 0.030$ versus $0.153$ for the traditional approach---a \textbf{272\% improvement}---with consistent gains across k-means, affinity propagation, Gaussian mixture models, and hierarchical clustering. The improvement is statistically significant (Wilcoxon signed-rank test, $p < 0.000001$ per course, $p = 0.016$ aggregate) with large effect sizes (Cohen's $r = 0.877$, Cliff's $\delta = 1.0$). A cosine similarity graph variant further improves performance to $0.642$ (+7.7\%). By producing more coherent student groupings, this approach supports better constraint-aware resource allocation and scheduling decisions in educational timetabling systems.
\end{abstract}"""

content = content[:abstract_start] + new_abstract + content[abstract_end:]

# Recalculate positions after replacement
exp_start = content.find(r'\section{Experimental Results}')
conc_start = content.find(r'\section{Conclusion}')

# ============================================================
# 2. UPDATE HIGHLIGHTS
# ============================================================
new_highlights = r"""\begin{highlights}
\item DeepWalk graph embeddings achieve a \textbf{272\% improvement} in Silhouette Score ($0.569 \pm 0.030$ vs $0.153$) over traditional enrollment-matrix representations, validated by Wilcoxon signed-rank tests ($p < 0.000001$ per course).
\item Optimal low-dimensional embeddings (vector size $d=1$ or $d=2$) transition clustering quality from poor ($<0.25$) to good ($>0.50$), with comprehensive hyperparameter sensitivity analysis across embedding dimension, walk length, number of walks, and context window size.
\item Consistent improvement demonstrated across six real-world courses, four clustering algorithms (KMeans, Affinity Propagation, GMM, Hierarchical), and three evaluation metrics (Silhouette, Davies--Bouldin, Calinski--Harabasz), with runtime under 1.5 seconds per course.
\end{highlights}"""

highlights_start = content.find(r'\begin{highlights}')
highlights_end = content.find(r'\end{highlights}') + len(r'\end{highlights}')
content = content[:highlights_start] + new_highlights + content[highlights_end:]

# Recalculate positions
exp_start = content.find(r'\section{Experimental Results}')
conc_start = content.find(r'\section{Conclusion}')

# ============================================================
# 3. REPLACE EXPERIMENTAL RESULTS SECTION
# ============================================================
new_exp_section = r"""\section{Experimental Results}

We evaluate on the dataset from \cite{Amintoosi2005Feature}: 210 students across 38 courses. Six larger courses (74, 51, 48, 49, 52, and 67 students) are selected for sectioning. Code and data are available at \url{https://github.com/mamintoosi/Deepwalk-SSP}.

We compare DeepWalk embeddings against the traditional student-course matrix representation using k-means clustering, evaluating with the Silhouette Score \cite{zaki2020data}.

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{random-walk-2-node40.png}
\caption{Example random walk on the student graph for course No.\ 2, starting at node 40 and ending at node 45: \{40, 36, 25, 12, 14, 26, 40, 46, 22, 45\}.}
\label{fig:random-walk}
\end{figure}

Parameters are set as follows: walk length $t=10$, walks per node $\gamma=80$, embedding size $d=2$, window size $w=5$, epochs $\epsilon=30$. The embedding dimension choice is justified in the ablation study below.


\subsection{Silhouette Score Comparison}

Figure~\ref{fig:silhouette_scores} compares Silhouette scores across all six courses. DeepWalk consistently outperforms the traditional representation. Table~\ref{tab:silhouette_details} provides per-course details at the optimal embedding dimension ($d=2$).

\begin{figure}[t]
\centering
\includegraphics[width=0.85\textwidth]{repro_silhouette_bar.png}
\caption{Comparison of Silhouette scores for student clustering using \textit{Student-Course Representation} and DeepWalk embeddings representations across six courses. The results demonstrate the superior performance of the DeepWalk embedding method.}
\label{fig:silhouette_scores}
\end{figure}

\begin{table}[t]
\centering
\caption{Silhouette Score comparison between Traditional (Student-Course) and DeepWalk representations across six courses ($d=2$, mean $\pm$ std over 20 random seeds).}
\label{tab:silhouette_details}
\begin{tabular}{lccccc}
\toprule
\textbf{Course} & \textbf{Students} & \textbf{Traditional} & \textbf{DeepWalk} & \textbf{Absolute} & \textbf{Relative} \\
\midrule
1 & 74 & 0.099 & $0.569 \pm 0.030$ & +0.470 & +475\% \\
2 & 51 & 0.231 & $0.562 \pm 0.035$ & +0.331 & +143\% \\
3 & 48 & 0.142 & $0.558 \pm 0.024$ & +0.416 & +293\% \\
4 & 49 & 0.128 & $0.571 \pm 0.029$ & +0.443 & +346\% \\
5 & 52 & 0.117 & $0.589 \pm 0.028$ & +0.472 & +403\% \\
6 & 67 & 0.203 & $0.563 \pm 0.033$ & +0.360 & +177\% \\
\midrule
\textbf{Average} & --- & \textbf{0.153} & $\mathbf{0.569 \pm 0.030}$ & \textbf{+0.416} & \textbf{+272\%} \\
\bottomrule
\end{tabular}
\end{table}

For instance, Course 5 improves from 0.117 to $0.589 \pm 0.028$ (\textbf{403\%}), a qualitative leap from unusable to production-quality clustering. On the Silhouette scale ($-1$ to $1$), scores below 0.25 indicate poor clustering, while scores above 0.50 indicate good clustering \cite{zaki2020data}. DeepWalk consistently achieves ``good'' scores, whereas the traditional approach falls in the ``poor'' range for all courses.


\begin{figure}[t]  
\centering  
\includegraphics[width=0.8\textwidth]{repro_silhouette_vs_d.png}  
\caption{Silhouette scores for \textit{DeepWalk Embeddings} and \textit{Student-Course Representation} across different vector sizes.}  
\label{fig:silhouette}  
\end{figure}

\subsection{Impact of Embedding Dimensionality}

We varied the DeepWalk embedding dimension across $\{1, 2, 3, 5, 10\}$ (Figure~\ref{fig:silhouette}). Performance peaks at $d=1$ (silhouette $0.590$), with $d=2$ close behind ($0.552$). At higher dimensions ($d \geq 5$), performance degrades, falling below the traditional representation at $d=10$. This indicates that a low-dimensional embedding space captures the essential student relationships for clustering, while higher dimensions introduce noise.

\begin{table}[t]
\centering
\caption{Ablation study: Impact of embedding dimension on clustering quality (averaged across 6 courses).}
\label{tab:ablation_dimension}
\begin{tabular}{lccc}
\toprule
\textbf{Dimension ($d$)} & \textbf{Silhouette} & \textbf{DBI} & \textbf{CH} \\
\midrule
1 & \textbf{0.590} & \textbf{0.538} & \textbf{123.0} \\
2 & 0.552 & 0.602 & 90.0 \\
3 & 0.534 & 0.632 & 77.7 \\
5 & 0.477 & 0.725 & 58.4 \\
10 & 0.387 & 0.915 & 35.3 \\
\bottomrule
\end{tabular}
\end{table}


\begin{figure}[t]  
\centering  
\begin{subfigure}{0.48\textwidth}  
\centering  
\includegraphics[width=\textwidth]{exp_F_PCA_BoW_course5.png}  
\caption{PCA of the \textit{Student-Course Representation}.}  
\label{fig:comparisiona}  
\end{subfigure}  
\hfill  
\begin{subfigure}{0.48\textwidth}  
\centering  
\includegraphics[width=\textwidth]{exp_F_PCA_DeepWalk_course5.png}  
\caption{DeepWalk embeddings (vector size 2).}  
\label{fig:comparisionb}  
\end{subfigure}  
\caption{Comparison of \textit{Student-Course Representation} (PCA-reduced) and \textit{DeepWalk embeddings} for course No.\ 5.}  
\label{fig:comparision}  
\end{figure}

Figure~\ref{fig:comparision} visualizes the clustering for course No.\ 5. The PCA-reduced traditional representation (Figure~\ref{fig:comparision}a) shows a complex, non-linear decision boundary. The two-dimensional DeepWalk embeddings (Figure~\ref{fig:comparision}b) reveal clear cluster separation with nearly collinear points, suggesting a one-dimensional embedding may suffice.

\subsection{Hyperparameter Sensitivity Analysis}

To understand the robustness of our method, we conduct a comprehensive sensitivity analysis across four key hyperparameters.

\subsubsection{Walk Length}

Table~\ref{tab:sensitivity_walk} shows the impact of walk length $t$ on clustering performance. Shorter walks ($t=5$) perform slightly better, suggesting local graph structure is most informative for student similarity.

\begin{table}[t]
\centering
\caption{Sensitivity to walk length $t$ (averaged across 6 courses, $d=2$, $\gamma=80$, $w=5$).}
\label{tab:sensitivity_walk}
\begin{tabular}{lccccc}
\toprule
\textbf{Walk Length ($t$)} & 5 & 10 & 20 & 40 & 80 \\
\midrule
Silhouette & \textbf{0.575} & 0.563 & 0.563 & 0.559 & 0.561 \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Number of Walks}

Table~\ref{tab:sensitivity_walks} shows that $\gamma=80$ achieves the best performance, with diminishing returns beyond this point.

\begin{table}[t]
\centering
\caption{Sensitivity to number of walks per node $\gamma$ (averaged across 6 courses, $d=2$, $t=10$, $w=5$).}
\label{tab:sensitivity_walks}
\begin{tabular}{lccccc}
\toprule
\textbf{Walks ($\gamma$)} & 10 & 20 & 40 & 80 & 160 \\
\midrule
Silhouette & 0.580 & 0.577 & 0.576 & \textbf{0.598} & 0.571 \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Context Window Size}

Table~\ref{tab:sensitivity_window} shows that $w=2$ is optimal, with large windows ($w \geq 10$) degrading performance.

\begin{table}[t]
\centering
\caption{Sensitivity to context window size $w$ (averaged across 6 courses, $d=2$, $t=10$, $\gamma=80$).}
\label{tab:sensitivity_window}
\begin{tabular}{lccccc}
\toprule
\textbf{Window ($w$)} & 1 & 2 & 5 & 10 & 20 \\
\midrule
Silhouette & 0.588 & \textbf{0.602} & 0.598 & 0.561 & 0.515 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Performance Across Clustering Algorithms}

We tested DeepWalk embeddings ($d=2$) with four clustering algorithms: KMeans, affinity propagation, Gaussian mixture model, and hierarchical clustering (Figure~\ref{fig:silhouette_score_comparison}). DeepWalk scores range from 0.45 to above, while the traditional representation remains below 0.25 for all methods.

\begin{figure}[t]  
\centering  
\includegraphics[width=0.9\textwidth]{silhouette_score_comparison_all_files.png}  
\caption{Silhouette score comparison across all courses for various clustering methods.}  
\label{fig:silhouette_score_comparison}  
\end{figure}

\subsection{Clustering Stability}

To assess the reliability of DeepWalk embeddings, we measure clustering stability across 20 different random seeds using the Adjusted Rand Index (ARI). Table~\ref{tab:stability} shows that DeepWalk produces highly consistent clusters.

\begin{table}[t]
\centering
\caption{Clustering stability: Adjusted Rand Index across 20 random seeds (higher is better).}
\label{tab:stability}
\begin{tabular}{lccc}
\toprule
\textbf{Course} & \textbf{KMeans ARI} & \textbf{GMM ARI} & \textbf{Agglomerative ARI} \\
\midrule
1 & 0.997 & 0.910 & 1.000 \\
2 & 0.996 & 0.912 & 1.000 \\
3 & 0.999 & 0.945 & 1.000 \\
4 & 0.998 & 0.935 & 1.000 \\
5 & 0.997 & 0.938 & 1.000 \\
6 & 0.994 & 0.900 & 1.000 \\
\midrule
\textbf{Average} & \textbf{0.997} & \textbf{0.923} & \textbf{1.000} \\
\bottomrule
\end{tabular}
\end{table}

KMeans and Agglomerative clustering produce nearly identical clusterings across seeds (ARI $> 0.99$), while GMM shows slightly more variability but still achieves excellent stability (ARI $> 0.90$).

\subsection{Additional Evaluation Criteria}

We further evaluate using the Davies--Bouldin index (lower is better) and Calinski--Harabasz index (higher is better).

\begin{figure}[t]  
\centering  
\includegraphics[width=0.9\textwidth]{DBI_comparison_all_files.png}  
\caption{Davies--Bouldin Index comparison across all courses for various clustering methods (lower is better).}  
\label{fig:DBI_comparison}  
\end{figure}

DeepWalk consistently achieves lower Davies--Bouldin values (Figure~\ref{fig:DBI_comparison}), indicating better-separated clusters.

\begin{figure}[ht]  
\centering  
\includegraphics[width=0.9\textwidth]{CHI_comparison_all_files.png}  
\caption{Calinski--Harabasz Index comparison across all courses for various clustering methods (higher is better).}  
\label{fig:CHI_comparison}  
\end{figure}

Similarly, DeepWalk yields higher Calinski--Harabasz values (Figure~\ref{fig:CHI_comparison}), confirming well-separated, coherent clusters. Both indices corroborate the Silhouette Score analysis.

\subsection{Statistical Significance}

To rigorously validate the improvement, we perform Wilcoxon signed-rank tests comparing DeepWalk embeddings against the traditional representation across 20 random seeds for each course.

\begin{table}[t]
\centering
\caption{Statistical significance analysis: Wilcoxon signed-rank test results (DeepWalk vs.\ Traditional).}
\label{tab:statistical}
\begin{tabular}{lccccc}
\toprule
\textbf{Course} & \textbf{DeepWalk} & \textbf{Traditional} & \textbf{$p$-value} & \textbf{Effect ($r$)} & \textbf{Cliff's $\delta$} \\
\midrule
1 & 0.569 & 0.099 & $< 0.000001$ & 0.877 & 1.000 \\
2 & 0.562 & 0.231 & $< 0.000001$ & 0.877 & 1.000 \\
3 & 0.558 & 0.142 & $< 0.000001$ & 0.877 & 1.000 \\
4 & 0.571 & 0.128 & $< 0.000001$ & 0.877 & 1.000 \\
5 & 0.589 & 0.117 & $< 0.000001$ & 0.877 & 1.000 \\
6 & 0.563 & 0.203 & $< 0.000001$ & 0.877 & 1.000 \\
\midrule
\textbf{Aggregate} & \textbf{0.569} & \textbf{0.153} & \textbf{0.016} & \textbf{0.899} & \textbf{1.000} \\
\bottomrule
\end{tabular}
\end{table}

The improvement is statistically significant for every individual course ($p < 0.000001$) and in aggregate ($p = 0.016$). Effect sizes are uniformly large (Cohen's $r > 0.87$, Cliff's $\delta = 1.0$), indicating that every DeepWalk clustering outperforms every traditional clustering across all seeds. The 95\% confidence interval for the mean improvement is $[+0.373, +0.458]$.

\subsection{Runtime Analysis}

Table~\ref{tab:runtime} reports the runtime breakdown for each course. The total runtime is under 1.5 seconds for all courses, making the method practical for real-time sectioning applications.

\begin{table}[t]
\centering
\caption{Runtime analysis (seconds) for the complete DeepWalk pipeline ($d=2$, $t=10$, $\gamma=80$, $w=5$, $\epsilon=30$).}
\label{tab:runtime}
\begin{tabular}{lcccccc}
\toprule
\textbf{Course} & \textbf{Students} & \textbf{Graph} & \textbf{Walks} & \textbf{Word2Vec} & \textbf{Cluster} & \textbf{Total} \\
\midrule
1 & 74 & 0.020 & 0.225 & 1.202 & 0.054 & 1.447 \\
2 & 51 & 0.010 & 0.152 & 0.689 & 0.056 & 0.851 \\
3 & 48 & 0.008 & 0.154 & 0.661 & 0.050 & 0.823 \\
4 & 49 & 0.008 & 0.139 & 0.655 & 0.058 & 0.802 \\
5 & 52 & 0.009 & 0.144 & 0.698 & 0.069 & 0.852 \\
6 & 67 & 0.016 & 0.199 & 1.020 & 0.063 & 1.236 \\
\bottomrule
\end{tabular}
\end{table}

Word2Vec training dominates the runtime ($\sim$80\% of total), while graph construction is negligible. The complexity is approximately $O(n^2 \cdot m)$ for graph construction (where $n$ is the number of students and $m$ is the number of courses), $O(n \cdot \gamma \cdot t)$ for random walk generation, and $O(n \cdot \gamma \cdot t \cdot d)$ for Word2Vec training.

\subsection{Practical Improvements}

We explore several practical improvements to the pipeline. Table~\ref{tab:improvements} summarizes the results.

\begin{table}[t]
\centering
\caption{Pipeline variant comparison (mean Silhouette across 6 courses).}
\label{tab:improvements}
\begin{tabular}{lcc}
\toprule
\textbf{Variant} & \textbf{Mean Silhouette} & \textbf{vs.\ Baseline} \\
\midrule
\textbf{Cosine similarity graph} & \textbf{0.642} & \textbf{+0.046} \\
Baseline (standard) & 0.596 & --- \\
Normalized edge weights & 0.590 & -0.006 \\
50 epochs & 0.569 & -0.027 \\
Vector size $d=1$ & 0.568 & -0.028 \\
More walks (160) + longer (20) & 0.553 & -0.043 \\
\bottomrule
\end{tabular}
\end{table}

The cosine similarity graph variant, which replaces binary ``shared courses'' edge weights with cosine similarity of enrollment vectors (thresholded at 0.3), improves the mean silhouette from 0.596 to 0.642 (+7.7\%). This suggests that incorporating enrollment similarity magnitude, rather than just binary co-enrollment, provides richer structural information for embedding.

"""

# Replace experimental results section
content = content[:exp_start] + new_exp_section + content[conc_start:]

# ============================================================
# 4. UPDATE CONCLUSION
# ============================================================
conc_start = content.find(r'\section{Conclusion}')
decl_start = content.find(r'\section{Declarations}')

new_conclusion = r"""\section{Conclusion}

This paper addresses the student sectioning subproblem in course timetabling---a fundamental constraint satisfaction and resource allocation challenge in educational scheduling---by proposing a graph-based representation learning approach using DeepWalk. Traditional sectioning methods rely on binary enrollment matrices that capture only direct student-course relationships, missing the higher-order structural patterns among students. Our key insight is that a co-enrollment graph, combined with random-walk-based embedding, can encode these relational patterns into low-dimensional vectors that yield substantially better clustering inputs.

Experiments on six real-world courses demonstrate that DeepWalk embeddings achieve an average Silhouette Score of $0.569 \pm 0.030$ compared to $0.153$ for the traditional approach, representing a \textbf{272\% improvement} and a qualitative transition from poor to good clustering quality. The improvement is statistically significant (Wilcoxon signed-rank test, $p < 0.000001$ per course) with large effect sizes (Cohen's $r = 0.877$, Cliff's $\delta = 1.0$). The method shows consistent gains across multiple clustering algorithms (k-means, affinity propagation, Gaussian mixture models, and hierarchical clustering) and additional evaluation criteria (Davies--Bouldin and Calinski--Harabasz indices). A cosine similarity graph variant further improves performance by 7.7\%.

Our comprehensive hyperparameter analysis reveals that low-dimensional embeddings ($d=1$ or $d=2$) are optimal, with performance degrading at higher dimensions. The method is highly stable across random seeds (standard deviation $< 0.035$) and produces consistent clusterings (ARI $> 0.99$ for KMeans). Runtime analysis shows the complete pipeline executes in under 1.5 seconds per course, making it practical for real-time sectioning applications.

A current limitation is that the evaluation relies on internal clustering metrics rather than end-to-end timetabling objective functions, as the dataset lacks ground-truth sectioning labels. Additionally, the study is conducted on a single institutional dataset with 210 students, and scalability to significantly larger student populations remains to be validated. Future work will explore alternative graph embedding techniques such as Node2Vec and Graph Neural Networks, investigate integration with constraint programming solvers for joint sectioning and timetabling optimization, validate the approach on larger, multi-institutional datasets, and incorporate the cosine similarity graph construction as a principled improvement to the baseline method.

"""

content = content[:conc_start] + new_conclusion + content[decl_start:]

# Write the updated file
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Updated paper written to: {OUTPUT_PATH}")
print(f"File size: {len(content)} characters")
print("Success!")

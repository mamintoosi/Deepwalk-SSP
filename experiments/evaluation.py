# -*- coding: utf-8 -*-
"""
Evaluation Module for DeepWalk-SSP
====================================
Clustering algorithms and evaluation metrics.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ── Clustering Functions ──────────────────────────────────────────────────────

def run_kmeans(data, n_clusters=2, random_state=0, n_init=10):
    """K-Means clustering."""
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(data)
    return labels


def run_affinity_propagation(data):
    """Affinity Propagation clustering."""
    ap = AffinityPropagation(random_state=0)
    labels = ap.fit_predict(data)
    return labels


def run_gmm(data, n_clusters=2, random_state=0):
    """Gaussian Mixture Model clustering."""
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    labels = gmm.fit_predict(data)
    return labels


def run_agglomerative(data, n_clusters=2):
    """Agglomerative (hierarchical) clustering."""
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    labels = ac.fit_predict(data)
    return labels


CLUSTERING_FUNCTIONS = {
    "kmeans": run_kmeans,
    "affinity": run_affinity_propagation,
    "gmm": run_gmm,
    "agglomerative": run_agglomerative,
}

# Methods that do NOT accept random_state
METHODS_WITHOUT_SEED = {"affinity", "agglomerative"}


# ── Evaluation Metrics ────────────────────────────────────────────────────────

def compute_silhouette(data, labels):
    """Silhouette Score (-1 to 1, higher is better)."""
    if len(np.unique(labels)) < 2:
        return np.nan
    return silhouette_score(data, labels)


def compute_davies_bouldin(data, labels):
    """Davies-Bouldin Index (lower is better)."""
    if len(np.unique(labels)) < 2:
        return np.nan
    return davies_bouldin_score(data, labels)


def compute_calinski_harabasz(data, labels):
    """Calinski-Harabasz Index (higher is better)."""
    if len(np.unique(labels)) < 2:
        return np.nan
    return calinski_harabasz_score(data, labels)


def compute_wcss(data, labels):
    """Within-cluster Sum of Squares (lower is better)."""
    unique_labels = np.unique(labels)
    wcss = 0.0
    for k in unique_labels:
        cluster_data = data[labels == k]
        centroid = np.mean(cluster_data, axis=0)
        wcss += np.sum((cluster_data - centroid) ** 2)
    return wcss


def compute_balance_score(labels):
    """Balance score: ratio of average to maximum cluster size."""
    _, counts = np.unique(labels, return_counts=True)
    avg_count = np.mean(counts)
    max_count = np.max(counts)
    return avg_count / max_count if max_count > 0 else 0


def compute_all_metrics(data, labels):
    """Compute all clustering metrics at once."""
    return {
        "silhouette": compute_silhouette(data, labels),
        "davies_bouldin": compute_davies_bouldin(data, labels),
        "calinski_harabasz": compute_calinski_harabasz(data, labels),
        "wcss": compute_wcss(data, labels),
        "balance": compute_balance_score(labels),
    }


def evaluate_clustering(data, n_clusters=2, methods=None, random_state=0):
    """
    Evaluate multiple clustering algorithms on data.
    
    Args:
        data: Feature matrix (n_samples x n_features).
        n_clusters: Number of clusters.
        methods: List of method names to evaluate.
        random_state: Random seed for reproducibility.
        
    Returns:
        Dictionary mapping method name → {labels, metrics}.
    """
    if methods is None:
        methods = list(CLUSTERING_FUNCTIONS.keys())
    
    results = {}
    for method in methods:
        try:
            fn = CLUSTERING_FUNCTIONS[method]
            if method in METHODS_WITHOUT_SEED:
                labels = fn(data)
            else:
                labels = fn(data, n_clusters=n_clusters, random_state=random_state)
            
            metrics = compute_all_metrics(data, labels)
            metrics["n_clusters_found"] = len(np.unique(labels))
            results[method] = {"labels": labels, "metrics": metrics}
        except Exception as e:
            print(f"Error with {method}: {e}")
            results[method] = {"labels": None, "metrics": {}}
    
    return results


# ── Embedding Dimensionality Reduction ────────────────────────────────────────

def reduce_pca(data, n_components=2):
    """Reduce dimensions using PCA."""
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    explained_var = pca.explained_variance_ratio_
    return reduced, explained_var


def reduce_tsne(data, n_components=2, perplexity=30, random_state=0):
    """Reduce dimensions using t-SNE."""
    # Adjust perplexity for small datasets
    actual_perplexity = min(perplexity, data.shape[0] - 1)
    if actual_perplexity < 5:
        actual_perplexity = 5
    
    tsne = TSNE(
        n_components=n_components, 
        perplexity=actual_perplexity, 
        random_state=random_state,
        max_iter=1000,
        learning_rate='auto',
        init='pca',
    )
    reduced = tsne.fit_transform(data)
    return reduced


# ── Clustering Stability ──────────────────────────────────────────────────────

def compute_clustering_stability(data, n_clusters=2, method="kmeans", 
                                   n_runs=20, base_seed=0):
    """
    Measure clustering stability across multiple random seeds.
    
    Uses Adjusted Rand Index (ARI) to measure pairwise agreement
    between clusterings from different seeds.
    
    Args:
        data: Feature matrix.
        n_clusters: Number of clusters.
        method: Clustering method name.
        n_runs: Number of different seeds to test.
        base_seed: Starting seed.
        
    Returns:
        Dictionary with stability metrics.
    """
    all_labels = []
    for i in range(n_runs):
        seed = base_seed + i
        try:
            fn = CLUSTERING_FUNCTIONS[method]
            if method in METHODS_WITHOUT_SEED:
                labels = fn(data)
            else:
                labels = fn(data, n_clusters=n_clusters, random_state=seed)
            all_labels.append(labels)
        except Exception:
            continue
    
    if len(all_labels) < 2:
        return {"mean_ari": np.nan, "std_ari": np.nan, "min_ari": np.nan, 
                "max_ari": np.nan, "n_valid_runs": 0, "pairwise_ari": []}
    
    # Compute pairwise ARI between all runs
    ari_scores = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            ari = adjusted_rand_score(all_labels[i], all_labels[j])
            ari_scores.append(ari)
    
    ari_scores = np.array(ari_scores)
    
    return {
        "mean_ari": np.mean(ari_scores),
        "std_ari": np.std(ari_scores),
        "min_ari": np.min(ari_scores),
        "max_ari": np.max(ari_scores),
        "n_valid_runs": len(all_labels),
        "pairwise_ari": ari_scores,
    }

# -*- coding: utf-8 -*-
"""
Complete Experiment Suite for DeepWalk-SSP
==========================================
Runs all experiments (A through F), statistical analysis, improvements,
and generates all tables, figures, and reports.

Usage:
    'E:/conda/envs/pth-gpu/python.exe' -m experiments.run_all_experiments
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import pickle
import networkx as nx
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.config import *
from experiments.core import (
    read_class, create_graph_from_bow, generate_random_walks,
    train_word2vec, run_pipeline, load_existing_results
)
from experiments.evaluation import (
    evaluate_clustering, compute_all_metrics, compute_clustering_stability,
    reduce_pca, reduce_tsne, CLUSTERING_FUNCTIONS
)
from experiments.plotting import (
    plot_silhouette_comparison_bar, plot_silhouette_vs_vectorsize,
    plot_clustering_methods_comparison, plot_embedding_visualization,
    plot_hyperparameter_sensitivity, plot_runtime_analysis,
    plot_statistical_comparison
)
from experiments.stats import (
    wilcoxon_signed_rank_test, compute_effect_size_r,
    compute_cliffs_delta, compute_bootstrap_ci,
    benjamini_hochberg_correction, run_full_statistical_analysis
)


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title):
    """Print formatted subsection header."""
    print(f"\n--- {title} ---")


# ============================================================
# STEP 2: Reproduce Existing Experiments
# ============================================================
def reproduce_existing():
    """Reproduce the original experiments and verify results."""
    print_header("STEP 2: Reproducing Existing Experiments")

    # Load existing results
    df_existing, graphs_existing = load_existing_results(RESULTS_DIR)
    
    if df_existing is not None:
        print(f"Loaded existing df.xlsx with {len(df_existing)} rows")
        print(f"Loaded {len(graphs_existing)} graphs")
    else:
        print("No existing results found. Running from scratch...")

    # Run full pipeline for all files and vector sizes
    all_results = []
    all_graphs = {}
    timing_results = {}

    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        timing_results[i] = {}
        
        for vector_size in VECTOR_SIZES:
            print(f"  Processing Course {i}, d={vector_size}...")
            
            result = run_pipeline(
                filepath,
                vector_size=vector_size,
                walk_length=DEFAULT_PARAMS["walk_length"],
                num_walks=DEFAULT_PARAMS["num_walks"],
                window=DEFAULT_PARAMS["window"],
                epochs=DEFAULT_PARAMS["epochs"],
                seed=0,
            )
            
            if result["success"]:
                # Cluster on BoW
                scm, _ = read_class(filepath)
                bow_results = evaluate_clustering(scm, n_clusters=2, random_state=0)
                
                # Cluster on embeddings
                emb_results = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
                
                kmeans_bow = bow_results["kmeans"]["metrics"]["silhouette"]
                kmeans_emb = emb_results["kmeans"]["metrics"]["silhouette"]
                balance_bow = bow_results["kmeans"]["metrics"]["balance"]
                balance_emb = emb_results["kmeans"]["metrics"]["balance"]
                
                all_results.append({
                    "file_index": i,
                    "vector_size": vector_size,
                    "BoW.silhouette_score": kmeans_bow,
                    "BoW.balance_score": balance_bow,
                    "BoW.labels": bow_results["kmeans"]["labels"].tolist(),
                    "embeddings.silhouette_score": kmeans_emb,
                    "embeddings.balance_score": balance_emb,
                    "embeddings.labels": emb_results["kmeans"]["labels"].tolist(),
                    "num_students": result["num_students"],
                    "num_edges": result["num_edges"],
                    "avg_degree": result["avg_degree"],
                })
                
                all_graphs[(i, vector_size)] = result["graph"]
                
                if vector_size == DEFAULT_PARAMS["vector_size"]:
                    timing_results[i] = result["timing"]
                    timing_results[i]["num_walks_generated"] = result["num_walks"]
            else:
                print(f"    FAILED for Course {i}, d={vector_size}")

    # Create results DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    df.to_excel(os.path.join(RESULTS_DIR, "df_reproduced.xlsx"), index=False)
    with open(os.path.join(RESULTS_DIR, "graphs_reproduced.pkl"), "wb") as f:
        pickle.dump({"all_graphs": all_graphs}, f)
    
    # Compare with existing results
    if df_existing is not None:
        print_subheader("Comparison with Previously Saved Results")
        for i in FILE_INDICES:
            old_row = df_existing[(df_existing["file_index"] == i) & 
                                  (df_existing["vector_size"] == 2)]
            new_row = df[(df["file_index"] == i) & (df["vector_size"] == 2)]
            
            if len(old_row) > 0 and len(new_row) > 0:
                old_bow = old_row["BoW.silhouette_score"].values[0]
                new_bow = new_row["BoW.silhouette_score"].values[0]
                old_emb = old_row["embeddings.silhouette_score"].values[0]
                new_emb = new_row["embeddings.silhouette_score"].values[0]
                
                bow_match = "OK" if abs(old_bow - new_bow) < 0.01 else f"DIFF({old_bow:.4f} vs {new_bow:.4f})"
                emb_match = "OK" if abs(old_emb - new_emb) < 0.01 else f"DIFF({old_emb:.4f} vs {new_emb:.4f})"
                print(f"  Course {i}: BoW={bow_match}, Emb={emb_match}")

    # Print summary table
    print_subheader("Reproduced Results (d=2)")
    df_d2 = df[df["vector_size"] == 2].copy()
    print(df_d2[["file_index", "BoW.silhouette_score", "embeddings.silhouette_score"]].to_string(index=False))

    avg_bow = df_d2["BoW.silhouette_score"].mean()
    avg_emb = df_d2["embeddings.silhouette_score"].mean()
    print(f"\n  Average BoW Silhouette:     {avg_bow:.3f}")
    print(f"  Average DeepWalk Silhouette: {avg_emb:.3f}")
    print(f"  Improvement:                +{(avg_emb - avg_bow):.3f} ({((avg_emb/avg_bow)-1)*100:.0f}%)")

    # Generate original figures
    print_subheader("Generating Original Figures")
    fig1 = plot_silhouette_comparison_bar(df, vector_size=2, filename="repro_silhouette_bar.png")
    print(f"  Saved: {fig1}")
    
    fig2 = plot_silhouette_vs_vectorsize(df, filename="repro_silhouette_vs_d.png")
    print(f"  Saved: {fig2}")

    return df, all_graphs, timing_results


# ============================================================
# STEP 4A: Hyperparameter Sensitivity
# ============================================================
def experiment_hyperparameter_sensitivity(df, all_graphs):
    """Experiment A: Hyperparameter sensitivity analysis."""
    print_header("STEP 4A: Hyperparameter Sensitivity Analysis")
    
    results = {}
    
    # --- Embedding Dimension ---
    print_subheader("A.1: Embedding Dimension Sensitivity")
    dim_results = {}
    for d in VECTOR_SIZES:
        course_metrics = {"silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
        for i in FILE_INDICES:
            filepath = os.path.join(DATA_DIR, f"{i}.txt")
            result = run_pipeline(filepath, vector_size=d, seed=0)
            if result["success"]:
                metrics = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
                for k in course_metrics:
                    course_metrics[k].append(metrics["kmeans"]["metrics"][k])
        dim_results[d] = {k: np.nanmean(v) for k, v in course_metrics.items()}
        print(f"  d={d:2d}: silhouette={dim_results[d]['silhouette']:.3f}, "
              f"DBI={dim_results[d]['davies_bouldin']:.3f}, "
              f"CH={dim_results[d]['calinski_harabasz']:.1f}")
    
    fig = plot_hyperparameter_sensitivity(
        dim_results, "d", "Embedding Dimension (d)",
        "Sensitivity to Embedding Dimension",
        "exp_A_sensitivity_embedding_dim.png"
    )
    results["embedding_dim"] = dim_results
    
    # --- Walk Length ---
    print_subheader("A.2: Walk Length Sensitivity")
    walk_results = {}
    for wl in WALK_LENGTHS:
        course_metrics = {"silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
        for i in FILE_INDICES:
            filepath = os.path.join(DATA_DIR, f"{i}.txt")
            result = run_pipeline(filepath, walk_length=wl, seed=0)
            if result["success"]:
                metrics = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
                for k in course_metrics:
                    course_metrics[k].append(metrics["kmeans"]["metrics"][k])
        walk_results[wl] = {k: np.nanmean(v) for k, v in course_metrics.items()}
        print(f"  t={wl:2d}: silhouette={walk_results[wl]['silhouette']:.3f}")
    
    fig = plot_hyperparameter_sensitivity(
        walk_results, "t", "Walk Length (t)",
        "Sensitivity to Walk Length",
        "exp_A_sensitivity_walk_length.png"
    )
    results["walk_length"] = walk_results
    
    # --- Number of Walks ---
    print_subheader("A.3: Number of Walks Sensitivity")
    walks_results = {}
    for nw in NUM_WALKS_RANGE:
        course_metrics = {"silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
        for i in FILE_INDICES:
            filepath = os.path.join(DATA_DIR, f"{i}.txt")
            result = run_pipeline(filepath, num_walks=nw, seed=0)
            if result["success"]:
                metrics = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
                for k in course_metrics:
                    course_metrics[k].append(metrics["kmeans"]["metrics"][k])
        walks_results[nw] = {k: np.nanmean(v) for k, v in course_metrics.items()}
        print(f"  gamma={nw:3d}: silhouette={walks_results[nw]['silhouette']:.3f}")
    
    fig = plot_hyperparameter_sensitivity(
        walks_results, "gamma", "Number of Walks per Node (gamma)",
        "Sensitivity to Number of Walks",
        "exp_A_sensitivity_num_walks.png"
    )
    results["num_walks"] = walks_results
    
    # --- Context Window Size ---
    print_subheader("A.4: Context Window Size Sensitivity")
    window_results = {}
    for w in WINDOW_SIZES:
        course_metrics = {"silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
        for i in FILE_INDICES:
            filepath = os.path.join(DATA_DIR, f"{i}.txt")
            result = run_pipeline(filepath, window=w, seed=0)
            if result["success"]:
                metrics = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
                for k in course_metrics:
                    course_metrics[k].append(metrics["kmeans"]["metrics"][k])
        window_results[w] = {k: np.nanmean(v) for k, v in course_metrics.items()}
        print(f"  w={w:2d}: silhouette={window_results[w]['silhouette']:.3f}")
    
    fig = plot_hyperparameter_sensitivity(
        window_results, "w", "Context Window Size (w)",
        "Sensitivity to Context Window Size",
        "exp_A_sensitivity_window_size.png"
    )
    results["window_size"] = window_results
    
    # Save results
    with open(os.path.join(RESULTS_DIR, "exp_A_hyperparameter_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# ============================================================
# STEP 4B: Runtime Analysis
# ============================================================
def experiment_runtime():
    """Experiment B: Runtime analysis."""
    print_header("STEP 4B: Runtime Analysis")
    
    runtime_data = {}
    
    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        print(f"  Course {i}...", end=" ")
        
        result = run_pipeline(
            filepath,
            vector_size=DEFAULT_PARAMS["vector_size"],
            walk_length=DEFAULT_PARAMS["walk_length"],
            num_walks=DEFAULT_PARAMS["num_walks"],
            window=DEFAULT_PARAMS["window"],
            epochs=DEFAULT_PARAMS["epochs"],
            seed=0,
            verbose=False,
        )
        
        if result["success"]:
            runtime_data[i] = result["timing"]
            runtime_data[i]["num_students"] = result["num_students"]
            runtime_data[i]["num_edges"] = result["num_edges"]
            print(f"Total: {result['timing']['total']:.3f}s")
            
            # Also measure clustering time
            t0 = time.perf_counter()
            scm, _ = read_class(filepath)
            evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
            clustering_time = time.perf_counter() - t0
            runtime_data[i]["clustering"] = clustering_time
        else:
            print("FAILED")
            runtime_data[i] = {"total": 0}
    
    # Print summary
    print_subheader("Runtime Summary")
    for i, t in runtime_data.items():
        print(f"  Course {i} ({t.get('num_students', '?')} students): "
              f"graph={t.get('graph_construction', 0):.3f}s, "
              f"walks={t.get('random_walks', 0):.3f}s, "
              f"w2v={t.get('word2vec_training', 0):.3f}s, "
              f"cluster={t.get('clustering', 0):.3f}s, "
              f"total={t.get('total', 0):.3f}s")
    
    # Plot
    fig = plot_runtime_analysis(runtime_data, "exp_B_runtime_analysis.png")
    print(f"  Saved: {fig}")
    
    # Scale analysis
    print_subheader("Scaling Analysis")
    students = [runtime_data[i].get("num_students", 0) for i in FILE_INDICES]
    totals = [runtime_data[i].get("total", 0) for i in FILE_INDICES]
    
    if len(students) > 1:
        # Fit polynomial
        coeffs = np.polyfit(students, totals, 2)
        print(f"  Polynomial fit (quadratic): {coeffs[0]:.6f}*n^2 + {coeffs[1]:.6f}*n + {coeffs[2]:.6f}")
    
    # Save results
    with open(os.path.join(RESULTS_DIR, "exp_B_runtime.json"), "w") as f:
        json.dump(runtime_data, f, indent=2, default=str)
    
    return runtime_data


# ============================================================
# STEP 4C: Random Seed Stability
# ============================================================
def experiment_seed_stability():
    """Experiment C: Random seed stability analysis."""
    print_header("STEP 4C: Random Seed Stability Analysis")
    
    stability_results = {}
    
    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        scm, _ = read_class(filepath)
        
        print(f"\n  Course {i} ({scm.shape[0]} students):")
        
        course_stability = {
            "kmeans": {m: [] for m in ["silhouette", "davies_bouldin", "calinski_harabasz", "wcss"]},
            "bow_silhouette": [],
        }
        
        # BoW is deterministic (KMeans with fixed seed) - compute once
        bow_results = evaluate_clustering(scm, n_clusters=2, random_state=0)
        bow_sil = bow_results["kmeans"]["metrics"]["silhouette"]
        
        for seed in range(NUM_SEEDS):
            result = run_pipeline(
                filepath,
                vector_size=DEFAULT_PARAMS["vector_size"],
                walk_length=DEFAULT_PARAMS["walk_length"],
                num_walks=DEFAULT_PARAMS["num_walks"],
                window=DEFAULT_PARAMS["window"],
                epochs=DEFAULT_PARAMS["epochs"],
                seed=seed,
            )
            
            if result["success"]:
                emb_metrics = evaluate_clustering(
                    result["embeddings"], n_clusters=2, random_state=0
                )
                
                for m in ["silhouette", "davies_bouldin", "calinski_harabasz", "wcss"]:
                    val = emb_metrics["kmeans"]["metrics"][m]
                    course_stability["kmeans"][m].append(val)
                
                course_stability["bow_silhouette"].append(bow_sil)
        
        # Compute statistics
        stats_summary = {}
        for m in ["silhouette", "davies_bouldin", "calinski_harabasz", "wcss"]:
            vals = np.array(course_stability["kmeans"][m])
            stats_summary[m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "median": float(np.median(vals)),
                "values": vals.tolist(),
            }
        
        stability_results[i] = stats_summary
        
        print(f"    Silhouette: {stats_summary['silhouette']['mean']:.3f} "
              f"± {stats_summary['silhouette']['std']:.3f} "
              f"[{stats_summary['silhouette']['min']:.3f}, {stats_summary['silhouette']['max']:.3f}]")
        print(f"    DBI:        {stats_summary['davies_bouldin']['mean']:.3f} "
              f"± {stats_summary['davies_bouldin']['std']:.3f}")
        print(f"    CH:         {stats_summary['calinski_harabasz']['mean']:.1f} "
              f"± {stats_summary['calinski_harabasz']['std']:.1f}")
    
    # Print aggregate statistics
    print_subheader("Aggregate Statistics Across All Courses")
    all_sil_means = [stability_results[i]["silhouette"]["mean"] for i in FILE_INDICES]
    all_sil_stds = [stability_results[i]["silhouette"]["std"] for i in FILE_INDICES]
    print(f"  Silhouette: mean={np.mean(all_sil_means):.3f} ± {np.mean(all_sil_stds):.3f}")
    
    # Save
    with open(os.path.join(RESULTS_DIR, "exp_C_seed_stability.json"), "w") as f:
        json.dump(stability_results, f, indent=2)
    
    return stability_results


# ============================================================
# STEP 4D: Clustering Stability
# ============================================================
def experiment_clustering_stability():
    """Experiment D: Clustering stability analysis."""
    print_header("STEP 4D: Clustering Stability Analysis")
    
    clustering_stability = {}
    
    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        result = run_pipeline(filepath, seed=0)
        
        if not result["success"]:
            continue
        
        print(f"\n  Course {i} ({result['num_students']} students):")
        course_stability = {}
        
        for method in ["kmeans", "gmm", "agglomerative"]:
            stability = compute_clustering_stability(
                result["embeddings"], n_clusters=2, method=method, n_runs=NUM_SEEDS
            )
            course_stability[method] = {
                "mean_ari": stability["mean_ari"],
                "std_ari": stability["std_ari"],
                "min_ari": stability["min_ari"],
                "max_ari": stability["max_ari"],
                "n_valid_runs": stability["n_valid_runs"],
            }
            print(f"    {method:15s}: ARI={stability['mean_ari']:.3f} "
                  f"± {stability['std_ari']:.3f} "
                  f"[{stability['min_ari']:.3f}, {stability['max_ari']:.3f}]")
        
        # Also test stability on BoW
        scm, _ = read_class(filepath)
        bow_stability = compute_clustering_stability(
            scm, n_clusters=2, method="kmeans", n_runs=NUM_SEEDS
        )
        course_stability["bow_kmeans"] = {
            "mean_ari": bow_stability["mean_ari"],
            "std_ari": bow_stability["std_ari"],
        }
        print(f"    {'BoW kmeans':15s}: ARI={bow_stability['mean_ari']:.3f} "
              f"± {bow_stability['std_ari']:.3f}")
        
        clustering_stability[i] = course_stability
    
    # Summary
    print_subheader("Stability Summary")
    for method in ["kmeans", "gmm", "agglomerative"]:
        mean_aris = [clustering_stability[i][method]["mean_ari"] 
                     for i in FILE_INDICES if method in clustering_stability.get(i, {})]
        if mean_aris:
            print(f"  {method:15s}: mean ARI = {np.mean(mean_aris):.3f}")
    
    with open(os.path.join(RESULTS_DIR, "exp_D_clustering_stability.json"), "w") as f:
        json.dump(clustering_stability, f, indent=2)
    
    return clustering_stability


# ============================================================
# STEP 4E: Additional Clustering Metrics
# ============================================================
def experiment_additional_metrics():
    """Experiment E: Additional clustering metrics."""
    print_header("STEP 4E: Additional Clustering Metrics")
    
    all_results = []
    
    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        scm, _ = read_class(filepath)
        result = run_pipeline(filepath, seed=0)
        
        if not result["success"]:
            continue
        
        print(f"\n  Course {i} ({result['num_students']} students):")
        
        # Evaluate BoW
        bow_eval = evaluate_clustering(scm, n_clusters=2, random_state=0)
        
        # Evaluate DeepWalk embeddings
        emb_eval = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
        
        for method in ["kmeans", "affinity", "gmm", "agglomerative"]:
            bow_m = bow_eval.get(method, {}).get("metrics", {})
            emb_m = emb_eval.get(method, {}).get("metrics", {})
            
            all_results.append({
                "Course": i,
                "Method": f"DeepWalk {method}",
                "Silhouette": emb_m.get("silhouette", np.nan),
                "Davies-Bouldin": emb_m.get("davies_bouldin", np.nan),
                "Calinski-Harabasz": emb_m.get("calinski_harabasz", np.nan),
                "WCSS": emb_m.get("wcss", np.nan),
                "Balance": emb_m.get("balance", np.nan),
            })
            all_results.append({
                "Course": i,
                "Method": f"BoW {method}",
                "Silhouette": bow_m.get("silhouette", np.nan),
                "Davies-Bouldin": bow_m.get("davies_bouldin", np.nan),
                "Calinski-Harabasz": bow_m.get("calinski_harabasz", np.nan),
                "WCSS": bow_m.get("wcss", np.nan),
                "Balance": bow_m.get("balance", np.nan),
            })
    
    df_metrics = pd.DataFrame(all_results)
    
    # Print summary tables
    print_subheader("Silhouette Scores by Method")
    pivot_sil = df_metrics.pivot_table(
        index="Course", columns="Method", values="Silhouette"
    )
    print(pivot_sil.round(3).to_string())
    
    print_subheader("Davies-Bouldin Index by Method")
    pivot_db = df_metrics.pivot_table(
        index="Course", columns="Method", values="Davies-Bouldin"
    )
    print(pivot_db.round(3).to_string())
    
    print_subheader("Calinski-Harabasz Index by Method")
    pivot_ch = df_metrics.pivot_table(
        index="Course", columns="Method", values="Calinski-Harabasz"
    )
    print(pivot_ch.round(1).to_string())
    
    print_subheader("WCSS by Method")
    pivot_wcss = df_metrics.pivot_table(
        index="Course", columns="Method", values="WCSS"
    )
    print(pivot_wcss.round(1).to_string())
    
    # Save
    df_metrics.to_excel(os.path.join(RESULTS_DIR, "exp_E_additional_metrics.xlsx"), index=False)
    
    return df_metrics


# ============================================================
# STEP 4F: Embedding Visualization
# ============================================================
def experiment_visualization():
    """Experiment F: Embedding visualization with PCA and t-SNE."""
    print_header("STEP 4F: Embedding Visualization")
    
    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        scm, _ = read_class(filepath)
        result = run_pipeline(filepath, seed=0)
        
        if not result["success"]:
            continue
        
        embeddings = result["embeddings"]
        emb_labels = evaluate_clustering(embeddings, n_clusters=2, random_state=0)["kmeans"]["labels"]
        bow_labels = evaluate_clustering(scm, n_clusters=2, random_state=0)["kmeans"]["labels"]
        
        print(f"\n  Course {i}: {embeddings.shape[0]} students, {embeddings.shape[1]}D embeddings")
        
        # PCA of BoW
        bow_pca, bow_var = reduce_pca(scm, n_components=2)
        plot_embedding_visualization(
            scm, bow_labels,
            f"PCA of Student-Course Representation (Course {i})\n"
            f"Explained variance: {bow_var[0]:.1%}, {bow_var[1]:.1%}",
            f"exp_F_PCA_BoW_course{i}.png",
            reduced_2d=bow_pca,
        )
        
        # PCA of embeddings
        emb_pca, emb_var = reduce_pca(embeddings, n_components=2)
        plot_embedding_visualization(
            embeddings, emb_labels,
            f"PCA of DeepWalk Embeddings (Course {i})\n"
            f"Explained variance: {emb_var[0]:.1%}, {emb_var[1]:.1%}",
            f"exp_F_PCA_DeepWalk_course{i}.png",
            reduced_2d=emb_pca,
        )
        
        # t-SNE of BoW
        bow_tsne = reduce_tsne(scm.astype(float), perplexity=min(30, scm.shape[0]-1), random_state=42)
        plot_embedding_visualization(
            scm, bow_labels,
            f"t-SNE of Student-Course Representation (Course {i})",
            f"exp_F_tSNE_BoW_course{i}.png",
            reduced_2d=bow_tsne,
        )
        
        # t-SNE of embeddings
        emb_tsne = reduce_tsne(embeddings, perplexity=min(30, embeddings.shape[0]-1), random_state=42)
        plot_embedding_visualization(
            embeddings, emb_labels,
            f"t-SNE of DeepWalk Embeddings (Course {i})",
            f"exp_F_tSNE_DeepWalk_course{i}.png",
            reduced_2d=emb_tsne,
        )
        
        print(f"    Generated PCA and t-SNE figures for Course {i}")
    
    return True


# ============================================================
# STEP 5: Statistical Analysis
# ============================================================
def experiment_statistical_analysis(seed_stability):
    """Step 5: Statistical significance analysis."""
    print_header("STEP 5: Statistical Significance Analysis")
    
    stat_results = {}
    
    # Use seed stability data: DeepWalk scores vs BoW scores across seeds
    for i in FILE_INDICES:
        if i not in seed_stability:
            continue
        
        dw_scores = seed_stability[i]["silhouette"]["values"]
        bow_score = seed_stability[i]["silhouette"]["mean"]  # BoW is deterministic
        
        # Compare DeepWalk vs BoW for this course
        test = wilcoxon_signed_rank_test(
            np.array(dw_scores),
            np.full_like(dw_scores, bow_score),
            alternative='greater'
        )
        
        r = compute_effect_size_r(test["statistic"], test["n"])
        delta, delta_interp = compute_cliffs_delta(
            np.array(dw_scores), np.full_like(dw_scores, bow_score)
        )
        ci_mean, ci_lower, ci_upper = compute_bootstrap_ci(np.array(dw_scores) - bow_score)
        
        print(f"\n  Course {i}:")
        print(f"    DeepWalk: {np.mean(dw_scores):.3f} ± {np.std(dw_scores):.3f}")
        print(f"    BoW:      {bow_score:.3f} (deterministic)")
        print(f"    Wilcoxon: stat={test['statistic']:.1f}, p={test['p_value']:.6f}")
        print(f"    Effect size r={r:.3f} ({'Large' if r>=0.5 else 'Medium' if r>=0.3 else 'Small'})")
        print(f"    Cliff's delta={delta:.3f} ({delta_interp})")
        print(f"    95% CI for mean diff: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        stat_results[i] = {
            "deepwalk_mean": float(np.mean(dw_scores)),
            "deepwalk_std": float(np.std(dw_scores)),
            "bow_score": float(bow_score),
            "wilcoxon_stat": float(test["statistic"]),
            "wilcoxon_p": float(test["p_value"]),
            "effect_size_r": float(r),
            "cliffs_delta": float(delta),
            "cliffs_interp": delta_interp,
            "ci_mean_diff": float(ci_mean),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }
    
    # Aggregate across all courses
    print_subheader("Aggregate Statistical Analysis")
    all_dw_means = [stat_results[i]["deepwalk_mean"] for i in stat_results]
    all_bow_means = [stat_results[i]["bow_score"] for i in stat_results]
    all_differences = [stat_results[i]["deepwalk_mean"] - stat_results[i]["bow_score"] for i in stat_results]
    
    # Overall Wilcoxon test
    overall_test = wilcoxon_signed_rank_test(all_dw_means, all_bow_means, alternative='greater')
    overall_r = compute_effect_size_r(overall_test["statistic"], overall_test["n"])
    overall_delta, overall_delta_interp = compute_cliffs_delta(all_dw_means, all_bow_means)
    overall_ci_mean, overall_ci_lower, overall_ci_upper = compute_bootstrap_ci(all_differences)
    
    print(f"  Aggregate comparison ({len(all_dw_means)} courses):")
    print(f"    Mean DeepWalk: {np.mean(all_dw_means):.3f}")
    print(f"    Mean BoW:      {np.mean(all_bow_means):.3f}")
    print(f"    Mean improvement: +{np.mean(all_differences):.3f}")
    print(f"    Wilcoxon: p={overall_test['p_value']:.6f}")
    print(f"    Effect size r={overall_r:.3f}")
    print(f"    Cliff's delta={overall_delta:.3f} ({overall_delta_interp})")
    print(f"    95% CI for mean improvement: [{overall_ci_lower:.3f}, {overall_ci_upper:.3f}]")
    
    stat_results["aggregate"] = {
        "mean_deepwalk": float(np.mean(all_dw_means)),
        "mean_bow": float(np.mean(all_bow_means)),
        "mean_improvement": float(np.mean(all_differences)),
        "wilcoxon_p": float(overall_test["p_value"]),
        "effect_size_r": float(overall_r),
        "cliffs_delta": float(overall_delta),
        "cliffs_interp": overall_delta_interp,
        "ci_lower": float(overall_ci_lower),
        "ci_upper": float(overall_ci_upper),
    }
    
    with open(os.path.join(RESULTS_DIR, "exp_statistical_analysis.json"), "w") as f:
        json.dump(stat_results, f, indent=2)
    
    return stat_results


# ============================================================
# STEP 7: Practical Improvements
# ============================================================
def experiment_improvements():
    """Step 7: Test practical improvements to the pipeline."""
    print_header("STEP 7: Practical Improvements")
    
    improvement_results = []
    
    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        scm, _ = read_class(filepath)
        
        print(f"\n  Course {i} ({scm.shape[0]} students):")
        
        # Baseline: standard pipeline
        base_result = run_pipeline(filepath, seed=0)
        if not base_result["success"]:
            continue
        base_metrics = evaluate_clustering(base_result["embeddings"], n_clusters=2, random_state=0)
        base_sil = base_metrics["kmeans"]["metrics"]["silhouette"]
        
        improvement_results.append({
            "Course": i, "Variant": "Baseline (standard)", 
            "Silhouette": base_sil, "Type": "baseline"
        })
        print(f"    Baseline:          sil={base_sil:.3f}")
        
        # Improvement 1: Edge weight normalization (divide by max)
        G_norm = base_result["graph"].copy()
        max_w = max(d['weight'] for _, _, d in G_norm.edges(data=True)) if G_norm.number_of_edges() > 0 else 1
        for u, v, d in G_norm.edges(data=True):
            d['weight'] = d['weight'] / max_w
        
        walks_norm = generate_random_walks(G_norm, num_walks=80, walk_length=10, seed=0)
        w2v_norm = train_word2vec(walks_norm, vector_size=2, seed=0)
        if w2v_norm:
            emb_norm = w2v_norm.wv.vectors
            m_norm = evaluate_clustering(emb_norm, n_clusters=2, random_state=0)
            s_norm = m_norm["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({
                "Course": i, "Variant": "Normalized edge weights", 
                "Silhouette": s_norm, "Type": "improvement"
            })
            print(f"    Normalized edges:  sil={s_norm:.3f} ({'+' if s_norm > base_sil else ''}{(s_norm-base_sil):.3f})")
        
        # Improvement 2: Higher walks + longer walks
        result_more = run_pipeline(filepath, num_walks=160, walk_length=20, seed=0)
        if result_more["success"]:
            m_more = evaluate_clustering(result_more["embeddings"], n_clusters=2, random_state=0)
            s_more = m_more["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({
                "Course": i, "Variant": "More walks (160) + longer (20)", 
                "Silhouette": s_more, "Type": "improvement"
            })
            print(f"    More walks (160):  sil={s_more:.3f} ({'+' if s_more > base_sil else ''}{(s_more-base_sil):.3f})")
        
        # Improvement 3: d=1
        result_d1 = run_pipeline(filepath, vector_size=1, seed=0)
        if result_d1["success"]:
            m_d1 = evaluate_clustering(result_d1["embeddings"], n_clusters=2, random_state=0)
            s_d1 = m_d1["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({
                "Course": i, "Variant": "Vector size d=1", 
                "Silhouette": s_d1, "Type": "improvement"
            })
            print(f"    Vector size d=1:   sil={s_d1:.3f} ({'+' if s_d1 > base_sil else ''}{(s_d1-base_sil):.3f})")
        
        # Improvement 4: Cosine-similarity graph
        # Normalize rows before dot product
        # Instead of binary weight (shared courses), use cosine similarity
        scm_float = scm.astype(float)
        norms = np.linalg.norm(scm_float, axis=1, keepdims=True)
        norms[norms == 0] = 1
        scm_norm = scm_float / norms
        
        G_cos = nx.Graph()
        for k in range(scm_norm.shape[0]):
            G_cos.add_node(k)
        for k1 in range(scm_norm.shape[0]):
            for k2 in range(k1 + 1, scm_norm.shape[0]):
                sim = float(np.dot(scm_norm[k1], scm_norm[k2]))
                if sim > 0.3:  # threshold
                    G_cos.add_edge(k1, k2, weight=sim)
        
        walks_cos = generate_random_walks(G_cos, num_walks=80, walk_length=10, seed=0)
        w2v_cos = train_word2vec(walks_cos, vector_size=2, seed=0)
        if w2v_cos:
            emb_cos = w2v_cos.wv.vectors
            m_cos = evaluate_clustering(emb_cos, n_clusters=2, random_state=0)
            s_cos = m_cos["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({
                "Course": i, "Variant": "Cosine similarity graph", 
                "Silhouette": s_cos, "Type": "improvement"
            })
            print(f"    Cosine graph:     sil={s_cos:.3f} ({'+' if s_cos > base_sil else ''}{(s_cos-base_sil):.3f})")
        
        # Improvement 5: Higher epochs
        result_ep50 = run_pipeline(filepath, epochs=50, seed=0)
        if result_ep50["success"]:
            m_ep50 = evaluate_clustering(result_ep50["embeddings"], n_clusters=2, random_state=0)
            s_ep50 = m_ep50["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({
                "Course": i, "Variant": "50 epochs", 
                "Silhouette": s_ep50, "Type": "improvement"
            })
            print(f"    50 epochs:        sil={s_ep50:.3f} ({'+' if s_ep50 > base_sil else ''}{(s_ep50-base_sil):.3f})")
    
    df_improvements = pd.DataFrame(improvement_results)
    df_improvements.to_excel(os.path.join(RESULTS_DIR, "exp_F_improvements.xlsx"), index=False)
    
    # Print aggregate
    print_subheader("Improvement Summary (Mean Silhouette)")
    summary = df_improvements.groupby("Variant")["Silhouette"].agg(["mean", "std", "min", "max"])
    print(summary.round(3).to_string())
    
    return df_improvements


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    total_start = time.time()
    
    print("=" * 70)
    print("  DeepWalk-SSP Complete Experiment Suite")
    print("  Paper: DeepWalk for Student Sectioning")
    print("=" * 70)
    
    # Step 2: Reproduce
    df, graphs, timing = reproduce_existing()
    
    # Step 4A: Hyperparameter sensitivity
    hyper_results = experiment_hyperparameter_sensitivity(df, graphs)
    
    # Step 4B: Runtime analysis
    runtime_results = experiment_runtime()
    
    # Step 4C: Seed stability
    seed_results = experiment_seed_stability()
    
    # Step 4D: Clustering stability
    clustering_stab = experiment_clustering_stability()
    
    # Step 4E: Additional metrics
    df_metrics = experiment_additional_metrics()
    
    # Step 4F: Visualization
    experiment_visualization()
    
    # Step 5: Statistical analysis
    stat_results = experiment_statistical_analysis(seed_results)
    
    # Step 7: Improvements
    df_improvements = experiment_improvements()
    
    total_time = time.time() - total_start
    
    # Final report
    print_header("EXPERIMENT SUITE COMPLETE")
    print(f"\nTotal runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\nGenerated files in: {RESULTS_DIR}")
    
    generated = []
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith("exp_") or f.startswith("repro_"):
            generated.append(f)
            print(f"  - {f}")
    
    print(f"\nTotal generated files: {len(generated)}")
    
    # Print final summary
    print_subheader("Key Findings Summary")
    
    # Best vector size
    best_d = max(hyper_results["embedding_dim"], 
                 key=lambda d: hyper_results["embedding_dim"][d]["silhouette"])
    print(f"  Best embedding dimension: d={best_d} "
          f"(silhouette={hyper_results['embedding_dim'][best_d]['silhouette']:.3f})")
    
    # Best walk length
    best_wl = max(hyper_results["walk_length"],
                  key=lambda w: hyper_results["walk_length"][w]["silhouette"])
    print(f"  Best walk length: t={best_wl} "
          f"(silhouette={hyper_results['walk_length'][best_wl]['silhouette']:.3f})")
    
    # Best number of walks
    best_nw = max(hyper_results["num_walks"],
                  key=lambda n: hyper_results["num_walks"][n]["silhouette"])
    print(f"  Best num walks: gamma={best_nw} "
          f"(silhouette={hyper_results['num_walks'][best_nw]['silhouette']:.3f})")
    
    # Statistical significance
    if "aggregate" in stat_results:
        agg = stat_results["aggregate"]
        print(f"\n  Statistical significance (aggregate):")
        print(f"    Wilcoxon p-value: {agg['wilcoxon_p']:.6f}")
        print(f"    Effect size r:    {agg['effect_size_r']:.3f}")
        print(f"    Improvement:      +{agg['mean_improvement']:.3f}")

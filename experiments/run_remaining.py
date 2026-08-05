# -*- coding: utf-8 -*-
"""
Run remaining experiments: Statistical Analysis, Visualization, Improvements
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.config import *
from experiments.core import read_class, create_graph_from_bow, generate_random_walks, train_word2vec, run_pipeline
from experiments.evaluation import (
    evaluate_clustering, compute_clustering_stability,
    reduce_pca, reduce_tsne
)
from experiments.plotting import (
    plot_embedding_visualization, setup_style
)
from experiments.stats import (
    wilcoxon_signed_rank_test, compute_effect_size_r,
    compute_cliffs_delta, compute_bootstrap_ci
)

import networkx as nx

def header(t):
    print("\n" + "="*70)
    print(f"  {t}")
    print("="*70)

# ============================================================
# STEP 4F: Embedding Visualization (with fixed t-SNE)
# ============================================================
def run_visualization():
    header("STEP 4F: Embedding Visualization (PCA + t-SNE)")
    
    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        scm, _ = read_class(filepath)
        result = run_pipeline(filepath, seed=0)
        
        if not result["success"]:
            print(f"  Course {i}: FAILED")
            continue
        
        embeddings = result["embeddings"]
        emb_labels = evaluate_clustering(embeddings, n_clusters=2, random_state=0)["kmeans"]["labels"]
        bow_labels = evaluate_clustering(scm, n_clusters=2, random_state=0)["kmeans"]["labels"]
        
        print(f"\n  Course {i}: {embeddings.shape[0]} students, {embeddings.shape[1]}D embeddings")
        
        # PCA of BoW
        bow_pca, bow_var = reduce_pca(scm, n_components=2)
        plot_embedding_visualization(
            scm, bow_labels,
            f"PCA of Student-Course Representation (Course {i})",
            f"exp_F_PCA_BoW_course{i}.png",
            reduced_2d=bow_pca,
        )
        
        # PCA of embeddings
        emb_pca, emb_var = reduce_pca(embeddings, n_components=2)
        plot_embedding_visualization(
            embeddings, emb_labels,
            f"PCA of DeepWalk Embeddings (Course {i})",
            f"exp_F_PCA_DeepWalk_course{i}.png",
            reduced_2d=emb_pca,
        )
        
        # t-SNE of BoW
        try:
            bow_tsne = reduce_tsne(scm.astype(float), perplexity=min(20, scm.shape[0]-2), random_state=42)
            plot_embedding_visualization(
                scm, bow_labels,
                f"t-SNE of Student-Course Representation (Course {i})",
                f"exp_F_tSNE_BoW_course{i}.png",
                reduced_2d=bow_tsne,
            )
            print(f"    t-SNE BoW: OK")
        except Exception as e:
            print(f"    t-SNE BoW failed: {e}")
        
        # t-SNE of embeddings
        try:
            emb_tsne = reduce_tsne(embeddings, perplexity=min(20, embeddings.shape[0]-2), random_state=42)
            plot_embedding_visualization(
                embeddings, emb_labels,
                f"t-SNE of DeepWalk Embeddings (Course {i})",
                f"exp_F_tSNE_DeepWalk_course{i}.png",
                reduced_2d=emb_tsne,
            )
            print(f"    t-SNE DeepWalk: OK")
        except Exception as e:
            print(f"    t-SNE DeepWalk failed: {e}")
    
    return True


# ============================================================
# STEP 5: Statistical Analysis
# ============================================================
def run_statistical_analysis():
    header("STEP 5: Statistical Significance Analysis")
    
    # Load seed stability data
    stab_path = os.path.join(RESULTS_DIR, "exp_C_seed_stability.json")
    with open(stab_path) as f:
        seed_stability = json.load(f)
    
    stat_results = {}
    
    # Compute actual BoW silhouette scores (deterministic with fixed seed)
    bow_scores = {}
    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        scm, _ = read_class(filepath)
        bow_eval = evaluate_clustering(scm, n_clusters=2, random_state=0)
        bow_scores[i] = bow_eval["kmeans"]["metrics"]["silhouette"]
    print("  BoW silhouette scores:", bow_scores)
    
    for i_str, course_data in seed_stability.items():
        i = int(i_str)
        dw_scores = np.array(course_data["silhouette"]["values"])
        bow_score = bow_scores[i]  # Actual BoW silhouette score
        
        test = wilcoxon_signed_rank_test(
            dw_scores, np.full_like(dw_scores, bow_score), alternative='greater'
        )
        r = compute_effect_size_r(test.get("statistic", 0), test.get("n", 0))
        delta, delta_interp = compute_cliffs_delta(dw_scores, np.full_like(dw_scores, bow_score))
        ci_mean, ci_lower, ci_upper = compute_bootstrap_ci(dw_scores - bow_score)
        
        print(f"\n  Course {i}:")
        print(f"    DeepWalk: {np.mean(dw_scores):.3f} ± {np.std(dw_scores):.3f}")
        print(f"    BoW:      {bow_score:.3f}")
        print(f"    Wilcoxon: stat={test['statistic']:.1f}, p={test['p_value']:.6f}")
        print(f"    Effect size r={r:.3f} ({'Large' if r>=0.5 else 'Medium' if r>=0.3 else 'Small'})")
        print(f"    Cliff's delta={delta:.3f} ({delta_interp})")
        print(f"    95% CI for diff: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
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
    
    # Aggregate
    all_dw = [stat_results[int(k)]["deepwalk_mean"] for k in stat_results]
    all_bow = [stat_results[int(k)]["bow_score"] for k in stat_results]
    all_diff = [d - b for d, b in zip(all_dw, all_bow)]
    
    agg_test = wilcoxon_signed_rank_test(np.array(all_dw), np.array(all_bow), alternative='greater')
    agg_r = compute_effect_size_r(agg_test["statistic"], agg_test["n"])
    agg_delta, agg_delta_interp = compute_cliffs_delta(all_dw, all_bow)
    agg_ci_mean, agg_ci_lower, agg_ci_upper = compute_bootstrap_ci(all_diff)
    
    print(f"\n  AGGREGATE ({len(all_dw)} courses):")
    print(f"    Mean DeepWalk: {np.mean(all_dw):.3f}")
    print(f"    Mean BoW:      {np.mean(all_bow):.3f}")
    print(f"    Mean improvement: +{np.mean(all_diff):.3f}")
    print(f"    Wilcoxon: p={agg_test['p_value']:.6f}")
    print(f"    Effect size r={agg_r:.3f}")
    print(f"    Cliff's delta={agg_delta:.3f} ({agg_delta_interp})")
    print(f"    95% CI for mean improvement: [{agg_ci_lower:.3f}, {agg_ci_upper:.3f}]")
    
    stat_results["aggregate"] = {
        "mean_deepwalk": float(np.mean(all_dw)),
        "mean_bow": float(np.mean(all_bow)),
        "mean_improvement": float(np.mean(all_diff)),
        "wilcoxon_p": float(agg_test["p_value"]),
        "effect_size_r": float(agg_r),
        "cliffs_delta": float(agg_delta),
        "cliffs_interp": agg_delta_interp,
        "ci_lower": float(agg_ci_lower),
        "ci_upper": float(agg_ci_upper),
    }
    
    with open(os.path.join(RESULTS_DIR, "exp_statistical_analysis.json"), "w") as f:
        json.dump(stat_results, f, indent=2)
    
    return stat_results


# ============================================================
# STEP 7: Practical Improvements
# ============================================================
def run_improvements():
    header("STEP 7: Practical Improvements")
    
    improvement_results = []
    
    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        scm, _ = read_class(filepath)
        
        print(f"\n  Course {i} ({scm.shape[0]} students):")
        
        # Baseline
        base_result = run_pipeline(filepath, seed=0)
        if not base_result["success"]:
            continue
        base_metrics = evaluate_clustering(base_result["embeddings"], n_clusters=2, random_state=0)
        base_sil = base_metrics["kmeans"]["metrics"]["silhouette"]
        
        improvement_results.append({"Course": i, "Variant": "Baseline (standard)", "Silhouette": base_sil, "Type": "baseline"})
        print(f"    Baseline:          sil={base_sil:.3f}")
        
        # 1: Normalized edge weights
        G_norm = base_result["graph"].copy()
        max_w = max((d['weight'] for _, _, d in G_norm.edges(data=True)), default=1)
        for u, v, d in G_norm.edges(data=True):
            d['weight'] = d['weight'] / max_w
        walks_norm = generate_random_walks(G_norm, num_walks_per_node=80, walk_length=10, seed=0)
        w2v_norm = train_word2vec(walks_norm, vector_size=2, seed=0)
        if w2v_norm:
            m = evaluate_clustering(w2v_norm.wv.vectors, n_clusters=2, random_state=0)
            s = m["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({"Course": i, "Variant": "Normalized edge weights", "Silhouette": s, "Type": "improvement"})
            print(f"    Normalized edges:  sil={s:.3f} ({'+' if s > base_sil else ''}{(s-base_sil):.3f})")
        
        # 2: More walks + longer walks
        r2 = run_pipeline(filepath, num_walks=160, walk_length=20, seed=0)
        if r2["success"]:
            s2 = evaluate_clustering(r2["embeddings"], n_clusters=2, random_state=0)["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({"Course": i, "Variant": "More walks (160) + longer (20)", "Silhouette": s2, "Type": "improvement"})
            print(f"    More walks (160):  sil={s2:.3f} ({'+' if s2 > base_sil else ''}{(s2-base_sil):.3f})")
        
        # 3: d=1
        r3 = run_pipeline(filepath, vector_size=1, seed=0)
        if r3["success"]:
            s3 = evaluate_clustering(r3["embeddings"], n_clusters=2, random_state=0)["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({"Course": i, "Variant": "Vector size d=1", "Silhouette": s3, "Type": "improvement"})
            print(f"    Vector size d=1:   sil={s3:.3f} ({'+' if s3 > base_sil else ''}{(s3-base_sil):.3f})")
        
        # 4: Cosine similarity graph
        scm_f = scm.astype(float)
        norms = np.linalg.norm(scm_f, axis=1, keepdims=True)
        norms[norms == 0] = 1
        scm_n = scm_f / norms
        G_cos = nx.Graph()
        for k in range(scm_n.shape[0]):
            G_cos.add_node(k)
        for k1 in range(scm_n.shape[0]):
            for k2 in range(k1 + 1, scm_n.shape[0]):
                sim = float(np.dot(scm_n[k1], scm_n[k2]))
                if sim > 0.3:
                    G_cos.add_edge(k1, k2, weight=sim)
        walks_cos = generate_random_walks(G_cos, num_walks_per_node=80, walk_length=10, seed=0)
        w2v_cos = train_word2vec(walks_cos, vector_size=2, seed=0)
        if w2v_cos:
            s4 = evaluate_clustering(w2v_cos.wv.vectors, n_clusters=2, random_state=0)["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({"Course": i, "Variant": "Cosine similarity graph", "Silhouette": s4, "Type": "improvement"})
            print(f"    Cosine graph:     sil={s4:.3f} ({'+' if s4 > base_sil else ''}{(s4-base_sil):.3f})")
        
        # 5: 50 epochs
        r5 = run_pipeline(filepath, epochs=50, seed=0)
        if r5["success"]:
            s5 = evaluate_clustering(r5["embeddings"], n_clusters=2, random_state=0)["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({"Course": i, "Variant": "50 epochs", "Silhouette": s5, "Type": "improvement"})
            print(f"    50 epochs:        sil={s5:.3f} ({'+' if s5 > base_sil else ''}{(s5-base_sil):.3f})")
    
    df = pd.DataFrame(improvement_results)
    df.to_excel(os.path.join(RESULTS_DIR, "exp_F_improvements.xlsx"), index=False)
    
    print(f"\n  Summary (Mean Silhouette):")
    summary = df.groupby("Variant")["Silhouette"].agg(["mean", "std"])
    print(summary.round(3).to_string())
    
    return df


if __name__ == "__main__":
    total_start = time.time()
    
    # Run visualization
    run_visualization()
    
    # Run statistical analysis
    run_statistical_analysis()
    
    # Run improvements
    run_improvements()
    
    total_time = time.time() - total_start
    print(f"\nRemaining experiments completed in {total_time:.1f}s ({total_time/60:.1f} min)")

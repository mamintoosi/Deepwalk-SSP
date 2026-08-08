# -*- coding: utf-8 -*-
"""
Standalone visualization script for Course 3 and Course 5.
Generates figures comparing:
  (a) t-SNE of traditional Student-Course (BoW) representation
  (b) Direct 2D DeepWalk-SSP embeddings (no dimensionality reduction)

Usage:
    python -m experiments.visualize_courses
"""

import os
import sys
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.config import DATA_DIR, FIGURES_DIR, DEFAULT_PARAMS
from experiments.core import read_class, run_pipeline
from experiments.evaluation import evaluate_clustering, reduce_tsne
from experiments.plotting import (
    plot_embedding_visualization, setup_style, save_fig
)

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper")

# Courses to visualize (same as in the paper)
COURSES = [3, 5]

# Data dimensions (from actual data files)
DATA_DIMS = {
    3: (48, 28),   # 48 students x 28 courses
    5: (52, 32),   # 52 students x 32 courses
}


def generate_visualization(course_idx):
    """Generate side-by-side visualization for a single course.
    
    Left panel:  t-SNE of Student-Course (BoW) representation
    Right panel: Direct 2D DeepWalk-SSP embeddings (no t-SNE)
    """
    filepath = os.path.join(DATA_DIR, f"{course_idx}.txt")
    scm, _ = read_class(filepath)
    if scm is None:
        print(f"  ERROR: Could not load Course {course_idx}")
        return

    n_students, n_courses = scm.shape
    n_students_expected, n_courses_expected = DATA_DIMS[course_idx]
    assert n_students == n_students_expected, \
        f"Expected {n_students_expected} students, got {n_students}"
    assert n_courses == n_courses_expected, \
        f"Expected {n_courses_expected} courses, got {n_courses}"

    print(f"\n  Course {course_idx}: {n_students} students x {n_courses} courses")

    # Run DeepWalk pipeline with d=2 (same config as paper)
    result = run_pipeline(
        filepath,
        vector_size=DEFAULT_PARAMS["vector_size"],   # d=2
        walk_length=DEFAULT_PARAMS["walk_length"],    # t=10
        num_walks=DEFAULT_PARAMS["num_walks"],        # gamma=80
        window=DEFAULT_PARAMS["window"],              # w=5
        epochs=DEFAULT_PARAMS["epochs"],              # eps=30
        seed=0,
    )

    if not result["success"]:
        print(f"  ERROR: DeepWalk pipeline failed for Course {course_idx}")
        return

    embeddings = result["embeddings"]
    print(f"    Embeddings shape: {embeddings.shape} (expected ({n_students}, 2))")
    assert embeddings.shape == (n_students, 2), \
        f"Expected ({n_students}, 2), got {embeddings.shape}"

    # Cluster assignments
    emb_labels = evaluate_clustering(
        embeddings, n_clusters=2, random_state=0
    )["kmeans"]["labels"]

    bow_labels = evaluate_clustering(
        scm, n_clusters=2, random_state=0
    )["kmeans"]["labels"]

    # ── Panel (a): t-SNE of BoW (high-dimensional, needs projection) ──
    try:
        bow_tsne = reduce_tsne(
            scm.astype(float),
            perplexity=min(20, scm.shape[0] - 2),
            random_state=42
        )
        plot_embedding_visualization(
            scm, bow_labels,
            f"t-SNE of Student-Course Representation (Course {course_idx})",
            f"exp_F_tSNE_BoW_course{course_idx}.png",
            reduced_2d=bow_tsne,
        )
        print(f"    [a] t-SNE BoW ({n_students}x{n_courses}): OK")
    except Exception as e:
        print(f"    [a] t-SNE BoW failed: {e}")

    # ── Panel (b): Direct 2D DeepWalk embeddings (NO t-SNE) ──
    plot_embedding_visualization(
        embeddings, emb_labels,
        f"DeepWalk-SSP Embeddings (Course {course_idx})",
        f"exp_F_tSNE_DeepWalk_course{course_idx}.png",
        reduced_2d=embeddings,  # Already 2D — plot directly
        axis_labels=(
            'Embedding Dimension 1',
            'Embedding Dimension 2'
        ),
    )
    print(f"    [b] DeepWalk 2D direct ({embeddings.shape}): OK")

    return embeddings, emb_labels


def copy_figures_to_paper():
    """Copy generated figures to paper/ folder for LaTeX compilation."""
    if not os.path.isdir(PAPER_DIR):
        print(f"\n  WARNING: paper/ directory not found at {PAPER_DIR}")
        return

    copied = []
    for course_idx in COURSES:
        for suffix in ["BoW", "DeepWalk"]:
            fname = f"exp_F_tSNE_{suffix}_course{course_idx}.png"
            src = os.path.join(FIGURES_DIR, fname)
            dst = os.path.join(PAPER_DIR, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied.append(fname)
            else:
                print(f"  WARNING: {fname} not found in {FIGURES_DIR}")

    if copied:
        print(f"\n  Copied {len(copied)} figures to paper/:")
        for f in copied:
            print(f"    {f}")


def main():
    print("=" * 70)
    print("  DeepWalk-SSP Visualization: Course 3 and Course 5")
    print("=" * 70)
    print(f"\n  Configuration:")
    print(f"    Embedding dimension d = {DEFAULT_PARAMS['vector_size']}")
    print(f"    Walk length t = {DEFAULT_PARAMS['walk_length']}")
    print(f"    Number of walks gamma = {DEFAULT_PARAMS['num_walks']}")
    print(f"    Context window w = {DEFAULT_PARAMS['window']}")
    print(f"    Training epochs = {DEFAULT_PARAMS['epochs']}")
    print(f"    Random seed = 0")
    print(f"\n  DeepWalk: plotted DIRECTLY (no t-SNE, no PCA)")
    print(f"    x-axis: Embedding Dimension 1")
    print(f"    y-axis: Embedding Dimension 2")
    print(f"\n  BoW: plotted using t-SNE projection (high-dimensional)")

    for course_idx in COURSES:
        generate_visualization(course_idx)

    # Copy to paper/
    copy_figures_to_paper()

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

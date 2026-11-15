"""
Table ML EDA Package.
"""

from .audit import analyze_missing_and_types, load_and_audit
from .pipeline import generate_markdown_report, run_full_eda
from .plots import (
    plot_numeric_by_class,
    plot_pca_projection,
    plot_spearman_correlation,
    plot_target_distribution,
)
from .stats import (
    compute_numeric_and_codebook_thresholds,
    compute_target_and_quotas,
    rank_features_for_label_studio,
)

__all__ = [
    "load_and_audit",
    "analyze_missing_and_types",
    "compute_target_and_quotas",
    "rank_features_for_label_studio",
    "compute_numeric_and_codebook_thresholds",
    "plot_target_distribution",
    "plot_numeric_by_class",
    "plot_spearman_correlation",
    "plot_pca_projection",
    "generate_markdown_report",
    "run_full_eda",
]

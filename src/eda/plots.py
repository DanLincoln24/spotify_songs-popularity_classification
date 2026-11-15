from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


def plot_target_distribution(
    target_df: pd.DataFrame,
    target_col: str,
    total_rows: int,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Строит столбчатую диаграмму баланса классов."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=target_df,
        x="class_label",
        y="count",
        hue="class_label",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    for idx, row in target_df.iterrows():
        ax.text(
            idx,
            row["count"] + (total_rows * 0.01),
            f"{row['count']} ({row['share_pct']}%)",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title(f"Распределение целевой переменной: {target_col}")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Количество записей")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_numeric_by_class(
    df: pd.DataFrame,
    target_col: str,
    top_features: List[str],
    save_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """Строит boxplot-графики топ числовых признаков в разрезе классов."""
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != target_col
    ]
    top_num = [c for c in top_features if c in num_cols][:6] or num_cols[:6]
    if not top_num:
        return None

    ncols = 2
    nrows = (len(top_num) + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(top_num):
        sns.boxplot(
            data=df,
            x=target_col,
            y=col,
            hue=target_col,
            palette="Set2",
            legend=False,
            ax=axes[i],
        )
        axes[i].set_title(f"{col} в разрезе {target_col}")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_spearman_correlation(
    df: pd.DataFrame,
    target_col: str,
    save_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """Строит тепловую карту ранговых корреляций Спирмена."""
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != target_col
    ][:15]
    if len(num_cols) < 2:
        return None

    corr = df[num_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Матрица ранговых корреляций Спирмена (числовые признаки)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_pca_projection(
    df: pd.DataFrame,
    target_col: str,
    id_cols: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """Строит 2D PCA-проекцию для оценки перекрытия классов."""
    id_cols = id_cols or []
    num_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in id_cols + [target_col]
    ]
    if len(num_cols) < 2:
        return None

    X_num = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    X_scaled = StandardScaler().fit_transform(X_num)
    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    proj_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "target": df[target_col].astype(str),
    })

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=proj_df,
        x="PC1",
        y="PC2",
        hue="target",
        alpha=0.65,
        s=45,
        palette="deep",
        ax=ax,
    )
    ax.set_title("2D PCA-проекция признаков (оценка перекрытия классов / серых зон)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig

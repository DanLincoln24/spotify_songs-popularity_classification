from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def compute_target_and_quotas(
    df: pd.DataFrame,
    target_col: str,
    quota_per_class: int = 75,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Рассчитывает распределение классов, квоты на ручной сбор (50-100 на класс),
    стратификацию для пилота (>= 50 объектов) и проверяет риск парадокса Каппы.
    """
    counts = df[target_col].value_counts(dropna=False)
    shares = (df[target_col].value_counts(normalize=True, dropna=False) * 100).round(2)

    n_classes = len(counts)
    total_manual = n_classes * quota_per_class

    target_df = pd.DataFrame({
        "class_label": counts.index.astype(str),
        "count": counts.values,
        "share_pct": shares.values,
        "manual_quota_equal (50-100)": quota_per_class,
        "manual_quota_proportional": np.maximum(
            15, np.round((shares.values / 100.0) * total_manual).astype(int)
        ),
        "pilot_50_stratified": np.maximum(
            5, np.round((shares.values / 100.0) * 50).astype(int)
        ),
    })

    max_share = float(shares.max())
    min_share = float(shares.min())
    imbalance_ratio = round(max_share / max(min_share, 0.01), 2)
    kappa_paradox_risk = max_share >= 75.0 or imbalance_ratio >= 5.0

    meta = {
        "n_classes": n_classes,
        "imbalance_ratio": imbalance_ratio,
        "kappa_paradox_risk": kappa_paradox_risk,
    }
    return target_df, meta


def rank_features_for_label_studio(
    df: pd.DataFrame,
    target_col: str,
    id_cols: Optional[List[str]] = None,
    top_k: int = 8,
) -> pd.DataFrame:
    """Отбирает топ-K информативных признаков через Mutual Information для Label Studio."""
    id_cols = id_cols or []
    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]

    X = df[feature_cols].copy()
    y = LabelEncoder().fit_transform(df[target_col].astype(str))

    discrete_mask = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
            discrete_mask.append(False)
        else:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str).fillna("MISSING"))
            discrete_mask.append(True)

    mi_scores = mutual_info_classif(
        X, y, discrete_features=discrete_mask, random_state=42
    )
    ranking = pd.DataFrame({
        "feature": feature_cols,
        "dtype": [str(df[c].dtype) for c in feature_cols],
        "mi_score": np.round(mi_scores, 4),
        "n_unique": [df[c].nunique(dropna=True) for c in feature_cols],
    }).sort_values(by="mi_score", ascending=False).reset_index(drop=True)

    ranking["include_in_label_studio"] = ranking.index < top_k
    return ranking


def compute_numeric_and_codebook_thresholds(
    df: pd.DataFrame,
    target_col: str,
    top_features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Считает статистики числовых колонок и пороговые квантили по классам для Codebook."""
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != target_col
    ]
    if not num_cols:
        return pd.DataFrame(), pd.DataFrame()

    stats_list = []
    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        q05, q25, q50, q75, q95 = np.percentile(s, [5, 25, 50, 75, 95])
        iqr = q75 - q25
        outliers = int(((s < (q25 - 1.5 * iqr)) | (s > (q75 + 1.5 * iqr))).sum())
        stats_list.append({
            "feature": col,
            "mean": round(float(s.mean()), 3),
            "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3),
            "q05": round(float(q05), 3),
            "q25": round(float(q25), 3),
            "median": round(float(q50), 3),
            "q75": round(float(q75), 3),
            "q95": round(float(q95), 3),
            "max": round(float(s.max()), 3),
            "iqr_outliers_cnt": outliers,
            "iqr_outliers_pct": round(outliers / len(s) * 100, 2),
        })
    num_stats_df = pd.DataFrame(stats_list)

    top_num = [c for c in top_features if c in num_cols][:6] or num_cols[:6]
    codebook_rows = []
    for col in top_num:
        for cls_val, grp in df.groupby(target_col):
            s = grp[col].dropna()
            if s.empty:
                continue
            q25, q50, q75 = np.percentile(s, [25, 50, 75])
            codebook_rows.append({
                "feature": col,
                "class": str(cls_val),
                "q25 (нижняя граница нормы класса)": round(float(q25), 3),
                "median (типичное значение)": round(float(q50), 3),
                "q75 (верхняя граница нормы класса)": round(float(q75), 3),
            })
    codebook_thresholds_df = pd.DataFrame(codebook_rows)

    return num_stats_df, codebook_thresholds_df

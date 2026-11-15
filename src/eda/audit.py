from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HIDDEN_MISSING_TOKENS = {"?", "-", "unknown", "none", "null", "nan", "n/a", ""}


def load_and_audit(
    csv_path: str | Path,
    target_col: str,
    id_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Загружает CSV, заменяет текстовые заглушки на NaN, удаляет дубликаты
    и проверяет выполнение требования >= 1000 записей.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(
            f"Целевая колонка '{target_col}' не найдена. Доступные: {list(df.columns)}"
        )

    raw_rows, raw_cols = df.shape

    for col in df.select_dtypes(include=["object", "category"]).columns:
        mask = df[col].astype(str).str.strip().str.lower().isin(HIDDEN_MISSING_TOKENS)
        if mask.any():
            df.loc[mask, col] = np.nan

    id_cols = id_cols or []
    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]

    full_dups = int(df.duplicated().sum())
    feature_dups = int(df.duplicated(subset=feature_cols).sum()) if feature_cols else 0

    df_clean = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    clean_rows = len(df_clean)

    audit_info = {
        "file_name": path.name,
        "raw_rows": raw_rows,
        "raw_cols": raw_cols,
        "full_duplicates": full_dups,
        "feature_duplicates": feature_dups,
        "clean_rows": clean_rows,
        "passes_1000_threshold": clean_rows >= 1000,
        "constant_columns": constant_cols,
        "missing_total_cells": int(df_clean.isna().sum().sum()),
    }
    return df_clean, audit_info


def analyze_missing_and_types(df: pd.DataFrame) -> pd.DataFrame:
    """Сводная таблица типов данных, доли пропусков и кардинальности."""
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_cnt": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=True),
    })
    return summary.sort_values(by="missing_pct", ascending=False)

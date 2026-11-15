from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .audit import analyze_missing_and_types, load_and_audit
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


def generate_markdown_report(
    audit_info: Dict,
    target_df: pd.DataFrame,
    target_meta: Dict,
    ls_ranking: pd.DataFrame,
    codebook_thresholds_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Генерирует Markdown-сводку с результатами"""
    top_ls_cols = ls_ranking.loc[
        ls_ranking["include_in_label_studio"], "feature"
    ].tolist()

    status_1000 = (
        "✅ Выполнено (>= 1000 записей)"
        if audit_info["passes_1000_threshold"]
        else "❌ ВНИМАНИЕ: Менее 1000 записей после очистки!"
    )
    kappa_note = (
        "⚠️ **Высокий риск парадокса Каппы** (перекос классов). Для пилотной разметки (>= 50 строк) обязательно используйте стратифицированный сэмплинг!"
        if target_meta["kappa_paradox_risk"]
        else "✅ Классы сбалансированы умеренно, риск парадокса Каппы низкий."
    )

    md_lines = [
        f"# Итоги EDA табличного датасета (`{audit_info['file_name']}`)",
        "",
        "## 1. Паспорт и формальные требования",
        f"- **Исходный размер:** {audit_info['raw_rows']} строк × {audit_info['raw_cols']} столбцов",
        f"- **Полных дубликатов:** {audit_info['full_duplicates']} | **Дубликатов по признакам:** {audit_info['feature_duplicates']}",
        f"- **Чистый размер после удаления дублей:** **{audit_info['clean_rows']}** строк ({status_1000})",
        f"- **Всего ячеек с пропусками:** {audit_info['missing_total_cells']}",
        f"- **Константные колонки:** {audit_info['constant_columns'] or 'нет'}",
        "",
        "## 2. Баланс классов и квоты на ручной сбор (50–100 на класс)",
        kappa_note,
        "",
        target_df.to_markdown(index=False),
        "",
        "## 3. Топ информативных признаков для карточки Label Studio",
        f"Рекомендуемые колонки для отображения разметчикам (`{len(top_ls_cols)}` шт.): `{', '.join(top_ls_cols)}`",
        "",
        ls_ranking.head(10).to_markdown(index=False),
        "",
        "## 4. Ориентиры и «серые зоны» для Codebook (медианы и IQR по классам)",
        (
            codebook_thresholds_df.to_markdown(index=False)
            if not codebook_thresholds_df.empty
            else "_Числовые признаки отсутствуют_"
        ),
    ]

    report_path = output_dir / "eda_summary.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    return report_path


def run_full_eda(
    csv_path: str | Path,
    target_col: str,
    id_cols: Optional[List[str]] = None,
    output_dir: str | Path = "reports/eda",
    top_k_features: int = 8,
) -> Dict:
    """Запускает все шаги EDA, сохраняет CSV-таблицы, графики и Markdown-отчет."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_clean, audit_info = load_and_audit(csv_path, target_col, id_cols)
    missing_df = analyze_missing_and_types(df_clean)
    target_df, target_meta = compute_target_and_quotas(df_clean, target_col)
    ls_ranking = rank_features_for_label_studio(
        df_clean, target_col, id_cols, top_k=top_k_features
    )
    num_stats_df, codebook_thresholds_df = compute_numeric_and_codebook_thresholds(
        df_clean, target_col, top_features=ls_ranking["feature"].tolist()
    )

    # Генерация и сохранение графиков
    plt.close(
        plot_target_distribution(
            target_df, target_col, len(df_clean), out_path / "01_target_distribution.png"
        )
    )
    fig_num = plot_numeric_by_class(
        df_clean,
        target_col,
        ls_ranking["feature"].tolist(),
        out_path / "02_top_numeric_by_class.png",
    )
    if fig_num:
        plt.close(fig_num)

    fig_corr = plot_spearman_correlation(
        df_clean, target_col, out_path / "03_spearman_correlation.png"
    )
    if fig_corr:
        plt.close(fig_corr)

    fig_pca = plot_pca_projection(
        df_clean, target_col, id_cols, out_path / "04_pca_2d_projection.png"
    )
    if fig_pca:
        plt.close(fig_pca)

    # Сохранение таблиц
    missing_df.to_csv(out_path / "missing_and_types.csv")
    target_df.to_csv(out_path / "target_and_manual_quotas.csv", index=False)
    ls_ranking.to_csv(out_path / "feature_ranking_label_studio.csv", index=False)
    if not num_stats_df.empty:
        num_stats_df.to_csv(out_path / "numeric_stats.csv", index=False)
    if not codebook_thresholds_df.empty:
        codebook_thresholds_df.to_csv(out_path / "codebook_thresholds.csv", index=False)

    report_path = generate_markdown_report(
        audit_info, target_df, target_meta, ls_ranking, codebook_thresholds_df, out_path
    )

    return {
        "df_clean": df_clean,
        "audit_info": audit_info,
        "missing_df": missing_df,
        "target_df": target_df,
        "target_meta": target_meta,
        "ls_ranking": ls_ranking,
        "num_stats_df": num_stats_df,
        "codebook_thresholds_df": codebook_thresholds_df,
        "report_path": report_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск модульного пайплайна EDA")
    parser.add_argument("--data", type=str, required=True, help="Путь к CSV")
    parser.add_argument("--target", type=str, required=True, help="Целевая колонка")
    parser.add_argument("--id-cols", type=str, nargs="*", default=[])
    parser.add_argument("--out", type=str, default="reports/eda")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    run_full_eda(args.data, args.target, args.id_cols, args.out, args.top_k)

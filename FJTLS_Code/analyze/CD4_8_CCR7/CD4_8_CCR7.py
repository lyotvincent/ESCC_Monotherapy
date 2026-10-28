"""CD4/CD8 CCR7 double-positive raw sum/area response-group analysis."""

import argparse
import math
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

DEFAULT_MATRIX_ROOT = Path("data/cell_marker_matrix")
DEFAULT_CLINICAL_CSV = Path("data/clinical.csv")
DEFAULT_OUT_DIR = Path("analyze/CD4_8_CCR7")


def find_h5ad_files(matrix_root: Path) -> dict[str, Path]:
    return {
        path.parent.name: path
        for path in sorted(matrix_root.glob("*/*_cell_marker_matrix.h5ad"))
    }


def load_clinical(path: Path, sample_col: str, response_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in (sample_col, response_col) if col not in df.columns]
    if missing:
        raise ValueError(f"clinical CSV missing columns: {missing}")
    df = df.copy()
    df[sample_col] = df[sample_col].astype(str)
    df[response_col] = df[response_col].astype(str)
    return df


def fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else math.nan


def vector_from_x(adata: ad.AnnData, var_name: str) -> np.ndarray:
    var_names = list(map(str, adata.var_names))
    if var_name not in var_names:
        raise KeyError(f"{var_name} not found in adata.var_names")
    col_i = var_names.index(var_name)
    values = adata.X[:, col_i]
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32).reshape(-1)


def get_raw_sum_per_area(adata: ad.AnnData, marker: str) -> np.ndarray:
    if marker in set(map(str, adata.var_names)):
        return vector_from_x(adata, marker)
    legacy_name = f"{marker}_raw_sum_per_area"
    if legacy_name in set(map(str, adata.var_names)):
        return vector_from_x(adata, legacy_name)
    raise KeyError(f"cannot find {marker} raw_sum_per_area in adata.X")


def compute_sample_metrics(
    h5ad_path: Path,
    positive_threshold: float,
) -> dict[str, float | int | str]:
    columns = {
        "CD4": "CD4_log1p_zscore_mean",
        "CD8": "CD8_log1p_zscore_mean",
        "CCR7": "CCR7_log1p_zscore_mean",
    }
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        missing = [col for col in columns.values() if col not in adata.obs.columns]
        if missing:
            raise KeyError(f"missing obs columns: {missing}")

        obs = adata.obs[list(columns.values())].fillna(0)
        cd4_pos = obs[columns["CD4"]].to_numpy(dtype=np.float32) > positive_threshold
        cd8_pos = obs[columns["CD8"]].to_numpy(dtype=np.float32) > positive_threshold
        ccr7_pos = obs[columns["CCR7"]].to_numpy(dtype=np.float32) > positive_threshold
        ccr7_raw = get_raw_sum_per_area(adata, "CCR7")

        n_cells = int(obs.shape[0])
        if len(ccr7_raw) != n_cells:
            raise ValueError(f"length mismatch: raw={len(ccr7_raw)}, obs={n_cells}")
        n_cd4 = int(np.count_nonzero(cd4_pos))
        n_cd8 = int(np.count_nonzero(cd8_pos))
        n_ccr7 = int(np.count_nonzero(ccr7_pos))
        cd4_ccr7_pos = cd4_pos & ccr7_pos
        cd8_ccr7_pos = cd8_pos & ccr7_pos
        n_cd4_ccr7 = int(np.count_nonzero(cd4_ccr7_pos))
        n_cd8_ccr7 = int(np.count_nonzero(cd8_ccr7_pos))
        cd4_ccr7_raw = np.where(cd4_ccr7_pos, ccr7_raw, 0).astype(np.float32)
        cd8_ccr7_raw = np.where(cd8_ccr7_pos, ccr7_raw, 0).astype(np.float32)
        cd4_ccr7_raw_positive = cd4_ccr7_raw[cd4_ccr7_pos]
        cd8_ccr7_raw_positive = cd8_ccr7_raw[cd8_ccr7_pos]

        return {
            "status": "ok",
            "error": "",
            "n_cells": n_cells,
            "n_CD4_positive_cells": n_cd4,
            "n_CD8_positive_cells": n_cd8,
            "n_CCR7_positive_cells": n_ccr7,
            "n_CD4_CCR7_double_positive_cells": n_cd4_ccr7,
            "n_CD8_CCR7_double_positive_cells": n_cd8_ccr7,
            "CD4_positive_cell_fraction": fraction(n_cd4, n_cells),
            "CD8_positive_cell_fraction": fraction(n_cd8, n_cells),
            "CCR7_positive_cell_fraction": fraction(n_ccr7, n_cells),
            "CD4_CCR7_fraction_in_CD4": fraction(n_cd4_ccr7, n_cd4),
            "CD8_CCR7_fraction_in_CD8": fraction(n_cd8_ccr7, n_cd8),
            "CD4_CCR7_raw_sum_per_area_mean": (
                float(cd4_ccr7_raw.mean()) if n_cells else math.nan
            ),
            "CD8_CCR7_raw_sum_per_area_mean": (
                float(cd8_ccr7_raw.mean()) if n_cells else math.nan
            ),
            "CD4_CCR7_raw_sum_per_area_positive_mean": (
                float(cd4_ccr7_raw_positive.mean()) if n_cd4_ccr7 else 0.0
            ),
            "CD8_CCR7_raw_sum_per_area_positive_mean": (
                float(cd8_ccr7_raw_positive.mean()) if n_cd8_ccr7 else 0.0
            ),
        }
    finally:
        adata.file.close()


def p_text(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "p=NA"
    if pvalue < 1e-4:
        return f"p={pvalue:.1e}"
    return f"p={pvalue:.4f}"


def mannwhitney_p(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return math.nan
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def add_bracket(ax, x1: int, x2: int, y: float, h: float, label: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="black")
    ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom", fontsize=9)


def plot_metric_boxplot(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    order: list[str],
    title: str,
    ylabel: str,
    out_png: Path,
    out_pdf: Path,
    pairwise: bool,
) -> pd.DataFrame:
    plot_df = df[df[metric].notna() & df[group_col].notna()].copy()
    groups = [
        plot_df.loc[plot_df[group_col] == group, metric].to_numpy() for group in order
    ]

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.boxplot(
        groups,
        tick_labels=order,
        showfliers=False,
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#dbeafe", "edgecolor": "#1f2937"},
        medianprops={"color": "#dc2626", "linewidth": 1.6},
        whiskerprops={"color": "#1f2937"},
        capprops={"color": "#1f2937"},
    )

    rng = np.random.default_rng(20240618)
    for i, values in enumerate(groups, start=1):
        if len(values) == 0:
            continue
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), i, dtype=float) + jitter,
            values,
            s=28,
            alpha=0.8,
            color="#334155",
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)

    test_rows = []
    finite_values = plot_df[metric].to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    y_max = float(np.max(finite_values)) if len(finite_values) else 1.0
    y_min = float(np.min(finite_values)) if len(finite_values) else 0.0
    span = max(y_max - y_min, abs(y_max) * 0.1, 1e-6)
    h = span * 0.06
    y = y_max + span * 0.08
    if len(order) == 2:
        pvalue = mannwhitney_p(groups[0], groups[1])
        add_bracket(ax, 1, 2, y, h, p_text(pvalue))
        test_rows.append(
            {
                "metric": metric,
                "grouping": group_col,
                "test": "Mann-Whitney U",
                "group1": order[0],
                "group2": order[1],
                "pvalue": pvalue,
                "pvalue_adjusted": pvalue,
            }
        )
    elif len(order) >= 3:
        valid_groups = [values for values in groups if len(values) > 0]
        kruskal_p = (
            float(stats.kruskal(*valid_groups).pvalue)
            if len(valid_groups) >= 2
            else math.nan
        )
        ax.text(
            0.02,
            0.98,
            f"Kruskal-Wallis {p_text(kruskal_p)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        pairs = [(0, 1), (1, 2), (0, 2)] if pairwise else []
        raw_p = [mannwhitney_p(groups[i], groups[j]) for i, j in pairs]
        valid_count = sum(np.isfinite(p) for p in raw_p)
        for row_i, ((i, j), pvalue) in enumerate(zip(pairs, raw_p)):
            p_adj = min(pvalue * valid_count, 1.0) if np.isfinite(pvalue) else math.nan
            add_bracket(ax, i + 1, j + 1, y + row_i * h * 2.2, h, p_text(p_adj))
            test_rows.append(
                {
                    "metric": metric,
                    "grouping": group_col,
                    "test": "Mann-Whitney U Bonferroni",
                    "group1": order[i],
                    "group2": order[j],
                    "pvalue": pvalue,
                    "pvalue_adjusted": p_adj,
                }
            )
        test_rows.append(
            {
                "metric": metric,
                "grouping": group_col,
                "test": "Kruskal-Wallis",
                "group1": "|".join(order),
                "group2": "",
                "pvalue": kruskal_p,
                "pvalue_adjusted": kruskal_p,
            }
        )

    ax.set_ylim(y_min - span * 0.12, y + h * (7 if len(order) >= 3 else 3))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return pd.DataFrame(test_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--clinical-csv", type=Path, default=DEFAULT_CLINICAL_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--response-col", default="response")
    parser.add_argument("--positive-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    h5ad_files = find_h5ad_files(args.matrix_root)
    clinical = load_clinical(args.clinical_csv, args.sample_col, args.response_col)
    clinical_map = clinical.set_index(args.sample_col).to_dict("index")

    rows = []
    for sample_id in tqdm(sorted(h5ad_files), desc="CD4/CD8 CCR7 raw sum/area"):
        row = {
            "sample_id": sample_id,
            "h5ad_path": str(h5ad_files[sample_id]),
            "response": clinical_map.get(sample_id, {}).get(args.response_col, pd.NA),
        }
        if sample_id in clinical_map:
            for key, value in clinical_map[sample_id].items():
                if key != args.response_col:
                    row[key] = value
        try:
            row.update(
                compute_sample_metrics(
                    h5ad_path=h5ad_files[sample_id],
                    positive_threshold=args.positive_threshold,
                )
            )
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "n_cells": math.nan,
                    "n_CD4_positive_cells": math.nan,
                    "n_CD8_positive_cells": math.nan,
                    "n_CCR7_positive_cells": math.nan,
                    "n_CD4_CCR7_double_positive_cells": math.nan,
                    "n_CD8_CCR7_double_positive_cells": math.nan,
                    "CD4_positive_cell_fraction": math.nan,
                    "CD8_positive_cell_fraction": math.nan,
                    "CCR7_positive_cell_fraction": math.nan,
                    "CD4_CCR7_fraction_in_CD4": math.nan,
                    "CD8_CCR7_fraction_in_CD8": math.nan,
                    "CD4_CCR7_raw_sum_per_area_mean": math.nan,
                    "CD8_CCR7_raw_sum_per_area_mean": math.nan,
                    "CD4_CCR7_raw_sum_per_area_positive_mean": math.nan,
                    "CD8_CCR7_raw_sum_per_area_positive_mean": math.nan,
                }
            )
        rows.append(row)

    result_df = pd.DataFrame(rows)
    response_order_3 = ["pCR", "MPR", "Non-MPR"]
    result_df["response_2group"] = result_df["response"].map(
        {"pCR": "pCR/MPR", "MPR": "pCR/MPR", "Non-MPR": "Non-MPR"}
    )
    result_df["response_3group"] = pd.Categorical(
        result_df["response"],
        categories=response_order_3,
        ordered=True,
    )

    result_csv = args.out_dir / "CD4_8_CCR7_response_metrics.csv"
    result_df.to_csv(result_csv, index=False)

    metric_specs = [
        (
            "CD4_CCR7_raw_sum_per_area_mean",
            "CD4+CCR7+/CCR7+ raw sum/area by response",
            "CCR7 raw sum/area mean",
            "CD4_CCR7_raw_sum_per_area_mean",
        ),
        (
            "CD8_CCR7_raw_sum_per_area_mean",
            "CD8+CCR7+/CCR7+ raw sum/area by response",
            "CCR7 raw sum/area mean",
            "CD8_CCR7_raw_sum_per_area_mean",
        ),
    ]
    test_tables = []
    for metric, title, ylabel, name in metric_specs:
        test_tables.append(
            plot_metric_boxplot(
                result_df,
                metric=metric,
                group_col="response_2group",
                order=["pCR/MPR", "Non-MPR"],
                title=title,
                ylabel=ylabel,
                out_png=args.out_dir / f"{name}_boxplot_2group.png",
                out_pdf=args.out_dir / f"{name}_boxplot_2group.pdf",
                pairwise=False,
            )
        )
        test_tables.append(
            plot_metric_boxplot(
                result_df,
                metric=metric,
                group_col="response_3group",
                order=response_order_3,
                title=title,
                ylabel=ylabel,
                out_png=args.out_dir / f"{name}_boxplot_3group.png",
                out_pdf=args.out_dir / f"{name}_boxplot_3group.pdf",
                pairwise=True,
            )
        )

    tests_df = pd.concat(test_tables, ignore_index=True)
    tests_csv = args.out_dir / "CD4_8_CCR7_group_tests.csv"
    tests_df.to_csv(tests_csv, index=False)

    print(f"saved_result_csv: {result_csv}")
    print(f"saved_tests_csv: {tests_csv}")
    print(f"ok_samples: {(result_df['status'] == 'ok').sum()}/{len(result_df)}")
    print(f"samples_with_response: {result_df['response'].notna().sum()}")
    print(f"out_dir: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

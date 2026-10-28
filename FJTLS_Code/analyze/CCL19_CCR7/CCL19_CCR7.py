"""Batch spatial grid-level Spearman analysis for CCL19 and CCR7."""

import argparse
import math
import warnings
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
DEFAULT_OUT_DIR = Path("analyze/CCL19_CCR7")


def find_h5ad_files(matrix_root: Path) -> dict[str, Path]:
    paths = sorted(matrix_root.glob("*/*_cell_marker_matrix.h5ad"))
    out = {}
    for path in paths:
        sample_id = path.parent.name
        out[sample_id] = path
    return out


def load_clinical(path: Path, sample_col: str, response_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in (sample_col, response_col) if col not in df.columns]
    if missing:
        raise ValueError(f"clinical CSV missing columns: {missing}")
    df = df.copy()
    df[sample_col] = df[sample_col].astype(str)
    df[response_col] = df[response_col].astype(str)
    return df


def finite_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return math.nan, math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, pvalue = stats.spearmanr(x, y)
    return float(rho), float(pvalue)


def compute_sample_grid_spearman(
    h5ad_path: Path,
    marker_a: str,
    marker_b: str,
    mpp: float,
    grid_um: float,
    min_cells_per_grid: int,
    positive_fraction_threshold: float | None,
) -> dict[str, float | int | str]:
    marker_a_mean = f"{marker_a}_log1p_zscore_mean"
    marker_b_mean = f"{marker_b}_log1p_zscore_mean"
    marker_a_pos = f"{marker_a}_positive_fraction"
    marker_b_pos = f"{marker_b}_positive_fraction"

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        missing = [
            col
            for col in (
                marker_a_mean,
                marker_b_mean,
                marker_a_pos,
                marker_b_pos,
                "cell_id",
            )
            if col not in adata.obs.columns
        ]
        if missing:
            raise ValueError(f"missing obs columns: {missing}")
        if "spatial" not in adata.obsm:
            raise ValueError("missing adata.obsm['spatial']")

        obs = adata.obs.copy()
        n_cells_total = int(obs.shape[0])
        keep_values = None
        if positive_fraction_threshold is not None:
            pos_frac_cols = [c for c in obs.columns if c.endswith("_positive_fraction")]
            if pos_frac_cols:
                keep = (
                    obs[pos_frac_cols]
                    .fillna(0)
                    .ge(positive_fraction_threshold)
                    .any(axis=1)
                )
                keep_values = keep.to_numpy()
                obs = obs.loc[keep].copy()

        xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        if len(xy) != n_cells_total:
            raise ValueError(f"spatial length mismatch: {len(xy)} != {n_cells_total}")
        if keep_values is not None:
            xy = xy[keep_values]

        grid_size_px = int(grid_um / mpp)
        obs["x"] = xy[:, 0]
        obs["y"] = xy[:, 1]
        obs["grid_x"] = np.floor(obs["x"] / grid_size_px).astype(int)
        obs["grid_y"] = np.floor(obs["y"] / grid_size_px).astype(int)
        obs["grid_id"] = obs["grid_x"].astype(str) + "_" + obs["grid_y"].astype(str)

        grid_df = (
            obs.groupby("grid_id", observed=True)
            .agg(
                marker_a_mean=(marker_a_mean, "mean"),
                marker_b_mean=(marker_b_mean, "mean"),
                marker_a_pos_frac=(marker_a_pos, "mean"),
                marker_b_pos_frac=(marker_b_pos, "mean"),
                n_cells=("cell_id", "count"),
                x=("x", "mean"),
                y=("y", "mean"),
            )
            .reset_index()
        )
        n_grids_before_filter = int(grid_df.shape[0])
        grid_df = grid_df[grid_df["n_cells"] >= min_cells_per_grid].copy()
        rho, pvalue = finite_spearman(
            grid_df["marker_a_mean"], grid_df["marker_b_mean"]
        )

        return {
            "status": "ok",
            "error": "",
            "n_cells_total": n_cells_total,
            "n_cells_used": int(obs.shape[0]),
            "n_grids_before_filter": n_grids_before_filter,
            "n_grids": int(grid_df.shape[0]),
            "rho": rho,
            "pvalue": pvalue,
            "grid_size_px": grid_size_px,
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


def plot_boxplot(
    df: pd.DataFrame,
    group_col: str,
    order: list[str],
    title: str,
    out_png: Path,
    out_pdf: Path,
    pairwise: bool,
) -> pd.DataFrame:
    plot_df = df[df["rho"].notna() & df[group_col].notna()].copy()
    groups = [
        plot_df.loc[plot_df[group_col] == group, "rho"].to_numpy() for group in order
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
    rng = np.random.default_rng(20240616)
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
    ax.set_ylabel("Spatial grid-level Spearman rho")
    ax.set_title(title)
    ax.axhline(0, color="#9ca3af", lw=1, ls="--")

    test_rows = []
    finite_values = plot_df["rho"].to_numpy()
    if len(finite_values):
        y_max = float(np.nanmax(finite_values))
        y_min = float(np.nanmin(finite_values))
    else:
        y_max, y_min = 1.0, 0.0
    span = max(y_max - y_min, 0.1)
    h = span * 0.06
    y = y_max + span * 0.08

    if len(order) == 2:
        pvalue = mannwhitney_p(groups[0], groups[1])
        add_bracket(ax, 1, 2, y, h, p_text(pvalue))
        test_rows.append(
            {
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
                "grouping": group_col,
                "test": "Kruskal-Wallis",
                "group1": "|".join(order),
                "group2": "",
                "pvalue": kruskal_p,
                "pvalue_adjusted": kruskal_p,
            }
        )

    ax.set_ylim(y_min - span * 0.12, y + h * (7 if len(order) >= 3 else 3))
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
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
    parser.add_argument("--marker-a", default="CCL19")
    parser.add_argument("--marker-b", default="CCR7")
    parser.add_argument("--mpp", type=float, default=0.278)
    parser.add_argument("--grid-um", type=float, default=100.0)
    parser.add_argument("--min-cells-per-grid", type=int, default=10)
    parser.add_argument("--positive-fraction-threshold", type=float, default=0.1)
    parser.add_argument("--no-positive-fraction-qc", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    h5ad_files = find_h5ad_files(args.matrix_root)
    clinical = load_clinical(args.clinical_csv, args.sample_col, args.response_col)
    clinical_map = clinical.set_index(args.sample_col).to_dict("index")
    sample_ids = sorted(h5ad_files)
    qc_threshold = (
        None if args.no_positive_fraction_qc else args.positive_fraction_threshold
    )

    rows = []
    for sample_id in tqdm(sample_ids, desc="grid-level Spearman"):
        h5ad_path = h5ad_files[sample_id]
        row = {
            "sample_id": sample_id,
            "h5ad_path": str(h5ad_path),
            "response": clinical_map.get(sample_id, {}).get(args.response_col, pd.NA),
        }
        if sample_id in clinical_map:
            for key, value in clinical_map[sample_id].items():
                if key != args.response_col:
                    row[key] = value
        try:
            row.update(
                compute_sample_grid_spearman(
                    h5ad_path=h5ad_path,
                    marker_a=args.marker_a,
                    marker_b=args.marker_b,
                    mpp=args.mpp,
                    grid_um=args.grid_um,
                    min_cells_per_grid=args.min_cells_per_grid,
                    positive_fraction_threshold=qc_threshold,
                )
            )
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "n_cells_total": math.nan,
                    "n_cells_used": math.nan,
                    "n_grids_before_filter": math.nan,
                    "n_grids": math.nan,
                    "rho": math.nan,
                    "pvalue": math.nan,
                    "grid_size_px": int(args.grid_um / args.mpp),
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

    out_csv = (
        args.out_dir / f"{args.marker_a}_{args.marker_b}_spatial_grid_spearman.csv"
    )
    result_df.to_csv(out_csv, index=False)

    test_tables = []
    test_tables.append(
        plot_boxplot(
            result_df,
            group_col="response_2group",
            order=["pCR/MPR", "Non-MPR"],
            title=f"{args.marker_a}-{args.marker_b} grid Spearman by response",
            out_png=args.out_dir
            / f"{args.marker_a}_{args.marker_b}_spearman_boxplot_2group.png",
            out_pdf=args.out_dir
            / f"{args.marker_a}_{args.marker_b}_spearman_boxplot_2group.pdf",
            pairwise=False,
        )
    )
    test_tables.append(
        plot_boxplot(
            result_df,
            group_col="response_3group",
            order=response_order_3,
            title=f"{args.marker_a}-{args.marker_b} grid Spearman by response",
            out_png=args.out_dir
            / f"{args.marker_a}_{args.marker_b}_spearman_boxplot_3group.png",
            out_pdf=args.out_dir
            / f"{args.marker_a}_{args.marker_b}_spearman_boxplot_3group.pdf",
            pairwise=True,
        )
    )
    tests_df = pd.concat(test_tables, ignore_index=True)
    tests_csv = (
        args.out_dir / f"{args.marker_a}_{args.marker_b}_spearman_group_tests.csv"
    )
    tests_df.to_csv(tests_csv, index=False)

    print(f"saved_result_csv: {out_csv}")
    print(f"saved_tests_csv: {tests_csv}")
    print(f"ok_samples: {(result_df['status'] == 'ok').sum()}/{len(result_df)}")
    print(f"samples_with_response: {result_df['response'].notna().sum()}")
    print(f"out_dir: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate a large-font two-group CD4/CD8+CCR7 response boxplot."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = DEFAULT_DIR / "CD4_8_CCR7_response_metrics.csv"
GROUP_COL = "response_2group"
GROUP_ORDER = ["pCR/MPR", "Non-MPR"]

METRIC_LABELS = {
    "CD4_CCR7_raw_sum_per_area_mean": {
        "title": "CD4+CCR7+/CCR7+ raw sum/area by response",
        "ylabel": "CCR7 raw sum/area mean",
    },
    "CD8_CCR7_raw_sum_per_area_mean": {
        "title": "CD8+CCR7+/CCR7+ raw sum/area by response",
        "ylabel": "CCR7 raw sum/area mean",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--metric",
        nargs="+",
        choices=sorted(METRIC_LABELS),
        default=list(METRIC_LABELS),
        help="Metrics to plot; defaults to both CD4+CCR7 and CD8+CCR7.",
    )
    return parser.parse_args()


def format_pvalue(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "p=NA"
    if pvalue < 1e-4:
        return f"p={pvalue:.1e}"
    return f"p={pvalue:.4f}"


def plot_large_font_boxplot(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
) -> tuple[Path, Path, float, list[int]]:
    required = {metric, GROUP_COL}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"input CSV missing columns: {missing}")

    plot_df = df[df[metric].notna() & df[GROUP_COL].notna()].copy()
    groups = [
        plot_df.loc[plot_df[GROUP_COL] == group, metric].to_numpy(dtype=float)
        for group in GROUP_ORDER
    ]
    if any(len(values) == 0 for values in groups):
        counts = dict(zip(GROUP_ORDER, map(len, groups)))
        raise ValueError(f"response group has no valid samples: {counts}")

    pvalue = float(
        stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided").pvalue
    )

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.boxplot(
        groups,
        tick_labels=GROUP_ORDER,
        showfliers=False,
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#dbeafe", "edgecolor": "#1f2937"},
        medianprops={"color": "#dc2626", "linewidth": 1.6},
        whiskerprops={"color": "#1f2937"},
        capprops={"color": "#1f2937"},
    )

    rng = np.random.default_rng(20240618)
    for position, values in enumerate(groups, start=1):
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), position, dtype=float) + jitter,
            values,
            s=44,
            alpha=0.8,
            color="#334155",
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    labels = METRIC_LABELS[metric]
    ax.set_title(labels["title"])
    ax.set_ylabel(labels["ylabel"])
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)

    finite_values = plot_df[metric].to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values):
        y_max = float(np.max(finite_values))
        y_min = float(np.min(finite_values))
    else:
        y_max, y_min = 1.0, 0.0
    span = max(y_max - y_min, abs(y_max) * 0.1, 1e-6)
    bracket_height = span * 0.06
    bracket_y = y_max + span * 0.08
    ax.plot(
        [1, 1, 2, 2],
        [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
        linewidth=1.2,
        color="black",
    )
    ax.text(
        1.5,
        bracket_y + bracket_height,
        format_pvalue(pvalue),
        ha="center",
        va="bottom",
        fontsize=16,
    )
    ax.set_ylim(y_min - span * 0.12, bracket_y + bracket_height * 3)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = out_dir / f"{metric}_boxplot_2group_large_font"
    out_png = output_stem.with_suffix(".png")
    out_pdf = output_stem.with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf, pvalue, [len(values) for values in groups]


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    for metric in args.metric:
        result = plot_large_font_boxplot(
            df=df,
            metric=metric,
            out_dir=args.out_dir,
        )
        out_png, out_pdf, pvalue, counts = result
        print(f"metric: {metric}")
        print(f"groups: {GROUP_ORDER[0]}={counts[0]}, {GROUP_ORDER[1]}={counts[1]}")
        print(f"Mann-Whitney U pvalue: {pvalue:.8g}")
        print(f"saved_png: {out_png}")
        print(f"saved_pdf: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

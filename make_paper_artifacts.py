"""
Make paper-ready figures + LaTeX tables from sweep results.

Input: results.jsonl produced by run_experiments.py
Output:
  - PNG figures under outdir/figures/
  - LaTeX tables under outdir/tables/ (booktabs-ready)

This script is fast and CPU-only.

Run:
  python make_paper_artifacts.py \
    --results artifacts/results.jsonl \
    --outdir artifacts/paper
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Row:
    depth: int
    n_train: int
    train_mse: float
    test_mse: float
    gen_gap: float
    L_hat: float
    x_term: float


def load_rows(path: str) -> List[Row]:
    out: List[Row] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(
                Row(
                    depth=int(d["depth"]),
                    n_train=int(d["n_train"]),
                    train_mse=float(d["train_mse"]),
                    test_mse=float(d["test_mse"]),
                    gen_gap=float(d["gen_gap"]),
                    L_hat=float(d["L_hat"]),
                    x_term=float(d["x_term"]),
                )
            )
    return out


def _group(rows: List[Row]) -> Dict[Tuple[int, int], List[Row]]:
    g: Dict[Tuple[int, int], List[Row]] = {}
    for r in rows:
        g.setdefault((r.depth, r.n_train), []).append(r)
    return g


def _mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = x.astype(np.float64)
    return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else 0.0


def _savefig(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def fig_gap_vs_term(rows: List[Row], outpath: str) -> None:
    x = np.array([r.x_term for r in rows], dtype=np.float64)
    y = np.array([r.gen_gap for r in rows], dtype=np.float64)
    d = np.array([r.depth for r in rows], dtype=np.int32)

    plt.figure(figsize=(7.2, 5.4))
    for depth in sorted(set(d.tolist())):
        m = d == depth
        plt.scatter(x[m], y[m], s=18, alpha=0.8, label=f"depth={depth}")

    # Fit line through origin for visualization
    denom = float(np.sum(x * x)) + 1e-12
    slope = float(np.sum(x * y) / denom)
    xx = np.logspace(np.log10(max(x.min(), 1e-12)) - 0.1, np.log10(x.max()) + 0.1, 200)
    plt.plot(xx, slope * xx, linewidth=2.0, label=f"fit: gap ≈ {slope:.2e}·term")

    plt.xscale("log")
    plt.yscale("symlog", linthresh=1e-6)
    plt.xlabel(r"Theoretical term: $(\hat L^d)/\sqrt{n}$")
    plt.ylabel("Generalization gap (Test MSE − Train MSE)")
    plt.title("Generalization Gap vs. Lipschitz-Depth Complexity")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    _savefig(outpath)


def fig_scaling_by_depth(rows: List[Row], outpath: str) -> None:
    """
    Plot median |gap| vs n for each depth (with IQR bands).
    """
    plt.figure(figsize=(7.2, 5.4))
    depths = sorted({r.depth for r in rows})
    ns = sorted({r.n_train for r in rows})

    for depth in depths:
        meds = []
        q25 = []
        q75 = []
        for n in ns:
            vals = np.array([abs(r.gen_gap) for r in rows if r.depth == depth and r.n_train == n], dtype=np.float64)
            meds.append(float(np.median(vals)))
            q25.append(float(np.quantile(vals, 0.25)))
            q75.append(float(np.quantile(vals, 0.75)))
        ns_arr = np.array(ns, dtype=np.float64)
        plt.plot(ns_arr, meds, marker="o", linewidth=2.0, label=f"depth={depth}")
        plt.fill_between(ns_arr, q25, q75, alpha=0.15)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n (train samples)")
    plt.ylabel("Median |gap| (log scale)")
    plt.title("Generalization Gap Decays with n (Stratified by Depth)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    _savefig(outpath)


def fig_heatmap(rows: List[Row], outpath: str) -> None:
    """
    Heatmap: median log10 |gap| for each (depth, n).
    """
    depths = sorted({r.depth for r in rows})
    ns = sorted({r.n_train for r in rows})
    mat = np.zeros((len(depths), len(ns)), dtype=np.float64)
    for i, d in enumerate(depths):
        for j, n in enumerate(ns):
            vals = np.array([abs(r.gen_gap) for r in rows if r.depth == d and r.n_train == n], dtype=np.float64)
            mat[i, j] = float(np.log10(np.median(vals) + 1e-12))

    plt.figure(figsize=(7.2, 3.8))
    im = plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(im, label=r"$\log_{10}(\mathrm{median}\ |\mathrm{gap}|)$")
    plt.yticks(range(len(depths)), [f"d={d}" for d in depths])
    plt.xticks(range(len(ns)), [str(n) for n in ns])
    plt.xlabel("n")
    plt.ylabel("Depth")
    plt.title("Median |Gap| Across Depth and Sample Size")
    _savefig(outpath)


def fig_distribution_box(rows: List[Row], outpath: str) -> None:
    """
    Boxplot of log10 |gap| by depth.
    """
    depths = sorted({r.depth for r in rows})
    data = []
    for d in depths:
        vals = np.array([abs(r.gen_gap) for r in rows if r.depth == d], dtype=np.float64)
        data.append(np.log10(vals + 1e-12))
    plt.figure(figsize=(7.2, 4.2))
    plt.boxplot(data, tick_labels=[f"d={d}" for d in depths], showfliers=False)
    plt.ylabel(r"$\log_{10}|\mathrm{gap}|$")
    plt.title("Distribution of Generalization Gaps by Depth")
    plt.grid(True, axis="y", alpha=0.25)
    _savefig(outpath)


def write_table_summary(rows: List[Row], outpath: str) -> None:
    """
    Write appendix-friendly LaTeX table: per (depth,n) mean±std including train/test MSE.
    Uses resizebox so wide numeric columns fit single-column appendix pages.
    """
    groups = _group(rows)
    depths = sorted({d for d, _ in groups.keys()})
    ns = sorted({n for _, n in groups.keys()})

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Full sweep summary (mean$\pm$std over seeds).}")
    lines.append(r"\label{tab:sweep-summary}")
    lines.append(r"\begin{footnotesize}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{@{}ccccccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"$d$ & $n$ & Train MSE & Test MSE & $|\mathrm{gap}|$ & $\hat L$ & $(\hat L^{d})/\sqrt{n}$ \\"
    )
    lines.append(r"\midrule")
    for d in depths:
        for n in ns:
            rs = groups[(d, n)]
            tr = np.array([r.train_mse for r in rs], dtype=np.float64)
            te = np.array([r.test_mse for r in rs], dtype=np.float64)
            gap = np.array([abs(r.gen_gap) for r in rs], dtype=np.float64)
            Lh = np.array([r.L_hat for r in rs], dtype=np.float64)
            term = np.array([r.x_term for r in rs], dtype=np.float64)
            tr_m, tr_s = _mean_std(tr)
            te_m, te_s = _mean_std(te)
            gap_m, gap_s = _mean_std(gap)
            L_m, L_s = _mean_std(Lh)
            t_m, t_s = _mean_std(term)
            lines.append(
                f"{d} & {n} & {tr_m:.2e}$\\pm${tr_s:.1e} & {te_m:.2e}$\\pm${te_s:.1e} & "
                f"{gap_m:.2e}$\\pm${gap_s:.1e} & {L_m:.2e}$\\pm${L_s:.1e} & {t_m:.2e}$\\pm${t_s:.1e} \\\\"
            )
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{footnotesize}")
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_table_main_compact(rows: List[Row], outpath: str) -> None:
    """
    Write a main-body-friendly LaTeX table.

    We summarize by (depth, n) but only for the smallest and largest n to keep it compact,
    and we report median |gap| (robust) plus median term.
    """
    depths = sorted({r.depth for r in rows})
    ns = sorted({r.n_train for r in rows})
    if not ns:
        raise RuntimeError("No n values found.")
    n_small, n_large = ns[0], ns[-1]

    def med(vals: List[float]) -> float:
        a = np.array(vals, dtype=np.float64)
        return float(np.median(a))

    lines: List[str] = []
    # table* spans both columns in twocolumn mode (ICML main body).
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Compact sweep summary (median over seeds).}")
    lines.append(r"\label{tab:main-summary}")
    lines.append(r"\begin{small}")
    lines.append(r"\begin{tabular}{ccccc}")
    lines.append(r"\toprule")
    lines.append(r"Depth & $n$ & median $|\mathrm{gap}|$ & median $\hat L$ & median $(\hat L^d)/\sqrt{n}$ \\")
    lines.append(r"\midrule")
    for d in depths:
        for n in (n_small, n_large):
            rs = [r for r in rows if r.depth == d and r.n_train == n]
            gap_med = med([abs(r.gen_gap) for r in rs])
            L_med = med([r.L_hat for r in rs])
            term_med = med([r.x_term for r in rs])
            lines.append(f"{d} & {n} & {gap_med:.2e} & {L_med:.2e} & {term_med:.2e} \\\\")
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{small}")
    lines.append(r"\end{table*}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_table_landscape_grid(rows: List[Row], outpath: str) -> None:
    """
    Optional landscape-style grid: rows = depths, columns = n values, cell = median |gap|.
    Kept compact for appendix 'column-unfriendly' numeric overview.
    """
    depths = sorted({r.depth for r in rows})
    ns = sorted({r.n_train for r in rows})

    def med(vals: List[float]) -> float:
        return float(np.median(np.array(vals, dtype=np.float64)))

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Median $|\mathrm{gap}|$ (grid over seeds).}")
    lines.append(r"\label{tab:gap-grid}")
    lines.append(r"\begin{scriptsize}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\resizebox{0.58\linewidth}{!}{%")
    head = " & ".join([r"${}$".format(nn) for nn in ns])
    lines.append(r"\begin{tabular}{@{}r" + "c" * len(ns) + r"@{}}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{1}{r}{} & " + head + r" \\")
    lines.append(r"\midrule")
    for d in depths:
        cells = []
        for n in ns:
            rs = [r for r in rows if r.depth == d and r.n_train == n]
            cells.append(f"{med([abs(r.gen_gap) for r in rs]):.2e}")
        lines.append(f"{d} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{scriptsize}")
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    args = p.parse_args()

    rows = load_rows(args.results)
    if not rows:
        raise RuntimeError("No rows loaded.")

    figdir = os.path.join(args.outdir, "figures")
    tabdir = os.path.join(args.outdir, "tables")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(tabdir, exist_ok=True)

    fig_gap_vs_term(rows, os.path.join(figdir, "gap_vs_term.png"))
    fig_scaling_by_depth(rows, os.path.join(figdir, "gap_scaling_by_depth.png"))
    fig_heatmap(rows, os.path.join(figdir, "gap_heatmap.png"))
    fig_distribution_box(rows, os.path.join(figdir, "gap_box_by_depth.png"))
    write_table_summary(rows, os.path.join(tabdir, "sweep_summary.tex"))
    write_table_main_compact(rows, os.path.join(tabdir, "main_summary.tex"))
    write_table_landscape_grid(rows, os.path.join(tabdir, "gap_grid.tex"))


if __name__ == "__main__":
    main()


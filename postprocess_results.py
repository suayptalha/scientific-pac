"""
Post-processing experiments for ICML 2026 paper.

This script is intentionally lightweight: it does NOT retrain models.
It reads `results.jsonl` produced by `run_experiments.py` and generates:

1) Correlation + R^2 between generalization gap and (L^d)/sqrt(n)
2) A stratified plot of gap vs 1/sqrt(n) for each depth
3) A stratified plot of gap vs L^d for each n

Run:
  python postprocess_results.py --results icml2026/artifacts/results.jsonl --outdir icml2026/artifacts
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

# Force non-interactive backend for servers/headless runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Row:
    depth: int
    n_train: int
    gen_gap: float
    L_hat: float
    x_term: float


def load_rows(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append(
                Row(
                    depth=int(d["depth"]),
                    n_train=int(d["n_train"]),
                    gen_gap=float(d["gen_gap"]),
                    L_hat=float(d["L_hat"]),
                    x_term=float(d["x_term"]),
                )
            )
    return rows


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())) + 1e-12
    return float((x * y).sum() / denom)


def r2_linear_through_origin(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit y ≈ a x (least squares through origin) and return (a, R^2).
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    a = float((x @ y) / ((x @ x) + 1e-12))
    yhat = a * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return a, 1.0 - ss_res / ss_tot


def plot_gap_vs_inv_sqrt_n(rows: List[Row], outpath: str) -> None:
    plt.figure(figsize=(7.5, 5.5))
    for depth in sorted({r.depth for r in rows}):
        xs = []
        ys = []
        for r in rows:
            if r.depth != depth:
                continue
            xs.append(1.0 / math.sqrt(r.n_train))
            ys.append(r.gen_gap)
        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        plt.scatter(xs, ys, s=18, alpha=0.75, label=f"depth={depth}")
    plt.xscale("log")
    plt.yscale("symlog", linthresh=1e-6)
    plt.xlabel(r"$1/\sqrt{n}$ (log scale)")
    plt.ylabel("Generalization gap (symlog)")
    plt.title("Gap vs. Sample Size Scaling (Stratified by Depth)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_gap_vs_L_pow_d(rows: List[Row], outpath: str) -> None:
    plt.figure(figsize=(7.5, 5.5))
    for n in sorted({r.n_train for r in rows}):
        xs = []
        ys = []
        for r in rows:
            if r.n_train != n:
                continue
            xs.append((r.L_hat**r.depth))
            ys.append(r.gen_gap)
        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        plt.scatter(xs, ys, s=18, alpha=0.75, label=f"n={n}")
    plt.xscale("log")
    plt.yscale("symlog", linthresh=1e-6)
    plt.xlabel(r"$(\hat L)^d$ (log scale)")
    plt.ylabel("Generalization gap (symlog)")
    plt.title("Gap vs. Lipschitz-Depth Factor (Stratified by n)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, required=True, help="Path to results.jsonl")
    p.add_argument("--outdir", type=str, required=True, help="Output directory for plots/metrics")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_rows(args.results)
    if not rows:
        raise RuntimeError(f"No rows found in {args.results}")

    x = np.array([r.x_term for r in rows], dtype=np.float64)
    y = np.array([r.gen_gap for r in rows], dtype=np.float64)
    yabs = np.abs(y)

    corr_loglog = pearsonr(np.log10(x + 1e-12), np.log10(yabs + 1e-12))
    slope_lin0, r2_lin0 = r2_linear_through_origin(x, y)

    # Log-log regression on absolute gaps: log |gap| = alpha log term + c
    lx = np.log(x + 1e-12)
    ly = np.log(yabs + 1e-12)
    X = np.vstack([lx, np.ones_like(lx)]).T
    alpha, c = np.linalg.lstsq(X, ly, rcond=None)[0]
    pred = X @ np.array([alpha, c])
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum()) + 1e-12
    r2_loglog = 1.0 - ss_res / ss_tot

    metrics = {
        "n_rows": len(rows),
        "pearson_loglog_term_gapabs": corr_loglog,
        "slope_through_origin": slope_lin0,
        "r2_through_origin": r2_lin0,
        "loglog_alpha": float(alpha),
        "loglog_intercept": float(c),
        "loglog_r2": float(r2_loglog),
    }
    with open(os.path.join(args.outdir, "postprocess_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_gap_vs_inv_sqrt_n(rows, os.path.join(args.outdir, "gap_vs_inv_sqrt_n.png"))
    plot_gap_vs_L_pow_d(rows, os.path.join(args.outdir, "gap_vs_Lpowd.png"))


if __name__ == "__main__":
    main()


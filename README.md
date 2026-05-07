# Compositional operator-tree experiments

PyTorch utilities to train **differentiable compositional trees** (fixed skeletons, learned parameters—**not** MLPs) on **1D synthetic regression** targets whose ground truth is a closed-form expression of controlled **depth**. Runs sweep training sample sizes and depths, logs train/test MSE and a **Lipschitz proxy** $\hat L$, and studies the **generalization gap** against the complexity-style term $(\hat L^d)/\sqrt{n}$.

**Scope:** this isolates statistical scaling under composition; it does **not** run discrete symbolic regression or structure search.

## Requirements

- Python 3.10+ recommended  
- Dependencies: see [`requirements.txt`](requirements.txt) (`torch`, `numpy`, `matplotlib`, `tqdm`)

## Install

From the repository root (parent of `icml2026/`):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run training sweep

```bash
python run_experiments.py --outdir icml2026/artifacts --device cuda
```

Use `--device cpu` if you have no GPU. Defaults match the released configuration: depths `{1,2,3,4}`, training sizes `{50,100,500,1000,5000}`, five seeds per setting.

### Useful flags (`run_experiments.py`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--outdir` | `icml2026/artifacts` | Directory for all outputs |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--seeds` | `5` | Random seeds per `(depth, n)` |
| `--depths` | `1 2 3 4` | Ground-truth / model depths |
| `--n_train` | `50 100 500 1000 5000` | Training set sizes |
| `--lipschitz_method` | `grad` | `grad` (batch gradient norms) or `analytic` (composed bounds) |
| `--lr` | `0.01` | Adam learning rate |
| `--batch_size` | `256` | Training batch size |
| `--max_epochs` | `5000` | Maximum epochs per run |
| `--patience` | `200` | Early-stopping patience on training MSE |
| `--log_level` | `INFO` | `DEBUG` … `ERROR` |
| `--no_tqdm` | off | Disable progress bars |

### Outputs (`--outdir`)

| File | Description |
|------|-------------|
| `results.jsonl` | One JSON object per run: depth, `n_train`, MSEs, `gen_gap`, `L_hat`, `x_term` (= $(\hat L^d)/\sqrt{n}$), etc. |
| `gap_vs_term.png` | Scatter: generalization gap vs `x_term` |
| `summary.json` | Small metadata blob (paths, hyperparameters) |

## Post-process (no retraining)

Reads `results.jsonl` and writes extra diagnostics and plots:

```bash
python postprocess_results.py \
  --results icml2026/artifacts/results.jsonl \
  --outdir icml2026/artifacts
```

Adds `postprocess_metrics.json`, `gap_vs_inv_sqrt_n.png`, and `gap_vs_Lpowd.png`.

## Paper-style figures and LaTeX tables

From the same `results.jsonl`, generates higher-DPI figures and `booktabs`-style tables (for pasting into a LaTeX manuscript):

```bash
python make_paper_artifacts.py \
  --results icml2026/artifacts/results.jsonl \
  --outdir icml2026/artifacts/paper
```

Under `--outdir`:

- `figures/`: `gap_vs_term.png`, `gap_scaling_by_depth.png`, `gap_heatmap.png`, `gap_box_by_depth.png`
- `tables/`: `sweep_summary.tex`, `main_summary.tex`, `gap_grid.tex`

## Layout

If you run everything with `--outdir icml2026/artifacts`, a typical tree is:

```text
icml2026/artifacts/
  results.jsonl
  summary.json
  gap_vs_term.png
  postprocess_metrics.json      # after postprocess_results.py
  gap_vs_inv_sqrt_n.png
  gap_vs_Lpowd.png
  paper/                         # after make_paper_artifacts.py
    figures/
    tables/
```

Adjust paths if your artifact directory differs.

## Reproducibility

`run_experiments.py` fixes seeds per run (Python, NumPy, PyTorch). For bitwise-identical GPU runs across machines, additional determinism flags may be needed; CPU runs are the most portable baseline.

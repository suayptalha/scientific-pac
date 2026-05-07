"""
ICML 2026 Experiments: Sample Complexity of Scientific Discovery
---------------------------------------------------------------

This script generates synthetic "physics-like" regression problems whose
ground-truth functions are compositional expression trees of controlled depth.
It then trains a differentiable compositional tree model (NOT an MLP) and
empirically evaluates the generalization gap vs. the theoretical term
    (L^d) / sqrt(n),
where L is an (empirical, data-dependent) Lipschitz proxy and d is the depth.

Design choices (for reproducibility and clarity):
- We do NOT perform discrete structure search (symbolic regression). Instead, we
  fix a tree skeleton per depth and learn continuous parameters. This isolates
  the statistical question (generalization under composition) from the
  computational search problem.
- L is estimated as a proxy for the global Lipschitz constant using two methods:
  (i) gradient-norm sampling of ||∇_x f(x)||_2 on a large batch,
  (ii) optional per-operator analytic upper bounds composed along the tree.

Run:
  python run_experiments.py --outdir icml2026/artifacts

The script is CPU-friendly; on a server you can add --device cuda.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class GroundTruthSpec:
    """
    A fully specified ground-truth function f(x) with controlled depth.

    We restrict to 1D input for clean visualization and stable Lipschitz
    estimation. Extending to d-dimensional inputs is straightforward: treat x as
    vector and update Linear/grad computations accordingly.
    """

    depth: int
    name: str
    params: Dict[str, float]
    noise_std: float
    x_low: float = -1.0
    x_high: float = 1.0

    def f(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (N, 1)
        returns y: shape (N, 1)
        """
        # Depth 1: y = w*x + b
        if self.depth == 1:
            w = self.params["w"]
            b = self.params["b"]
            return w * x + b

        # Depth 2: y = sin(w*x + b0) + b1
        if self.depth == 2:
            w = self.params["w"]
            b0 = self.params["b0"]
            b1 = self.params["b1"]
            return torch.sin(w * x + b0) + b1

        # Depth 3: y = exp(-(a*x + b0)) * sin(w*x + b1)
        # (damped oscillator envelope)
        if self.depth == 3:
            a = self.params["a"]
            w = self.params["w"]
            b0 = self.params["b0"]
            b1 = self.params["b1"]
            env = torch.exp(-(a * x + b0))
            osc = torch.sin(w * x + b1)
            return env * osc

        # Depth 4: y = exp(-(a*x + b0)) * sin(w*x + b1) + (c*x + d)
        if self.depth == 4:
            a = self.params["a"]
            w = self.params["w"]
            b0 = self.params["b0"]
            b1 = self.params["b1"]
            c = self.params["c"]
            d = self.params["d"]
            env = torch.exp(-(a * x + b0))
            osc = torch.sin(w * x + b1)
            return env * osc + (c * x + d)

        raise ValueError(f"Unsupported depth: {self.depth}")


def sample_ground_truth(depth: int, seed: int) -> GroundTruthSpec:
    """
    Choose parameters that keep activations in a numerically stable range over
    x in [-1, 1] so that the Lipschitz proxies are meaningful and training is stable.
    """
    rng = np.random.default_rng(seed)
    if depth == 1:
        params = dict(
            w=float(rng.uniform(0.5, 2.0)),
            b=float(rng.uniform(-0.5, 0.5)),
        )
        return GroundTruthSpec(depth=1, name="linear", params=params, noise_std=0.01)
    if depth == 2:
        params = dict(
            w=float(rng.uniform(0.5, 3.0)),
            b0=float(rng.uniform(-0.5, 0.5)),
            b1=float(rng.uniform(-0.5, 0.5)),
        )
        return GroundTruthSpec(depth=2, name="sin_affine_plus_bias", params=params, noise_std=0.01)
    if depth == 3:
        params = dict(
            a=float(rng.uniform(0.2, 2.0)),
            w=float(rng.uniform(0.5, 6.0)),
            b0=float(rng.uniform(-0.2, 0.2)),
            b1=float(rng.uniform(-0.5, 0.5)),
        )
        return GroundTruthSpec(depth=3, name="damped_sine", params=params, noise_std=0.02)
    if depth == 4:
        params = dict(
            a=float(rng.uniform(0.2, 2.0)),
            w=float(rng.uniform(0.5, 6.0)),
            b0=float(rng.uniform(-0.2, 0.2)),
            b1=float(rng.uniform(-0.5, 0.5)),
            c=float(rng.uniform(-1.0, 1.0)),
            d=float(rng.uniform(-0.5, 0.5)),
        )
        return GroundTruthSpec(depth=4, name="damped_sine_plus_linear", params=params, noise_std=0.02)
    raise ValueError(f"Unsupported depth: {depth}")


class PhysicsTreeDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Synthetic dataset for a fixed ground-truth compositional function of depth d.
    Generates samples i.i.d. from x ~ Uniform([x_low, x_high]).
    """

    def __init__(
        self,
        spec: GroundTruthSpec,
        n: int,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.n = int(n)
        self.seed = int(seed)
        self.device = device

        g = torch.Generator()
        g.manual_seed(self.seed)
        x = (self.spec.x_high - self.spec.x_low) * torch.rand(self.n, 1, generator=g) + self.spec.x_low
        y = self.spec.f(x)
        if self.spec.noise_std > 0:
            y = y + self.spec.noise_std * torch.randn_like(y, generator=g)

        self.x = x.to(self.device)
        self.y = y.to(self.device)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class Linear1D(nn.Module):
    """Scalar affine: x -> a*x + b (applied elementwise)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * x + self.b

    def lipschitz_upper_bound(self) -> torch.Tensor:
        # For scalar affine, operator norm is |a|
        return self.a.abs()


UnaryOpName = Literal["sin", "cos", "exp", "linear"]
BinaryOpName = Literal["add", "mult"]


class UnaryOp(nn.Module):
    def __init__(self, op: UnaryOpName) -> None:
        super().__init__()
        self.op = op
        self.linear = Linear1D() if op == "linear" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.op == "sin":
            return torch.sin(x)
        if self.op == "cos":
            return torch.cos(x)
        if self.op == "exp":
            return torch.exp(x)
        if self.op == "linear":
            assert self.linear is not None
            return self.linear(x)
        raise ValueError(f"Unknown unary op: {self.op}")

    def lipschitz_upper_bound(self, x: torch.Tensor) -> torch.Tensor:
        """
        Data-dependent Lipschitz upper bound on the batch (local).
        For scalar->scalar:
        - sin, cos: |d/dx| <= 1
        - exp: |d/dx| = exp(x) so bound by max exp(x) on batch
        - linear: |a|
        Returns a scalar tensor.
        """
        if self.op in ("sin", "cos"):
            return torch.ones((), device=x.device, dtype=x.dtype)
        if self.op == "exp":
            return torch.exp(x).max()
        if self.op == "linear":
            assert self.linear is not None
            return self.linear.lipschitz_upper_bound()
        raise ValueError(f"Unknown unary op: {self.op}")


class BinaryOp(nn.Module):
    def __init__(self, op: BinaryOpName) -> None:
        super().__init__()
        self.op = op

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.op == "add":
            return a + b
        if self.op == "mult":
            return a * b
        raise ValueError(f"Unknown binary op: {self.op}")

    def lipschitz_upper_bound(
        self, a: torch.Tensor, b: torch.Tensor, La: torch.Tensor, Lb: torch.Tensor
    ) -> torch.Tensor:
        """
        Data-dependent Lipschitz upper bound wrt input(s), assuming upstream
        Lipschitz bounds La, Lb for the two children.

        For add: L <= La + Lb.
        For mult: if y=a*b, then |dy| <= |b|*|da| + |a|*|db| => L <= max|b|*La + max|a|*Lb.
        Returns a scalar tensor.
        """
        if self.op == "add":
            return La + Lb
        if self.op == "mult":
            return b.abs().max() * La + a.abs().max() * Lb
        raise ValueError(f"Unknown binary op: {self.op}")


class TreeNode(nn.Module):
    """
    Differentiable computation tree node.

    A node is either:
    - a leaf that forwards the input x through a unary operator (including linear),
    - or an internal node that combines two child nodes with a binary operator.
    """

    def __init__(
        self,
        *,
        unary: Optional[UnaryOpName] = None,
        binary: Optional[BinaryOpName] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
    ) -> None:
        super().__init__()
        if (unary is None) == (binary is None):
            raise ValueError("Specify exactly one of unary or binary.")
        self.unary_op = UnaryOp(unary) if unary is not None else None
        self.binary_op = BinaryOp(binary) if binary is not None else None
        self.left = left
        self.right = right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.unary_op is not None:
            return self.unary_op(x)
        assert self.binary_op is not None and self.left is not None and self.right is not None
        return self.binary_op(self.left(x), self.right(x))

    @torch.no_grad()
    def lipschitz_upper_bound(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a data-dependent Lipschitz upper bound for this subtree on the batch x.

        Returns scalar tensor L̂(x_batch) based on local derivative bounds.
        """
        if self.unary_op is not None:
            # Unary bound may depend on its input; we evaluate on current batch.
            u_in = x
            return self.unary_op.lipschitz_upper_bound(u_in)

        assert self.binary_op is not None and self.left is not None and self.right is not None
        # We need intermediate activations a(x), b(x) for multiplication bounds.
        a = self.left(x)
        b = self.right(x)
        La = self.left.lipschitz_upper_bound(x)
        Lb = self.right.lipschitz_upper_bound(x)
        return self.binary_op.lipschitz_upper_bound(a, b, La, Lb)


class CompositionalTreeModel(nn.Module):
    """
    Fixed-depth compositional model over {add, mult, sin, cos, exp, linear}.

    We provide depth-specific skeletons to match the synthetic data-generating
    process. This keeps experiments focused on the statistical behavior of
    compositional classes.
    """

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.depth = int(depth)
        self.root = self._build_tree(self.depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.root(x)

    def lipschitz_proxy(self, x: torch.Tensor, method: Literal["grad", "analytic"] = "grad") -> float:
        """
        Estimate a proxy for the global Lipschitz constant over the support of x.

        - grad: sample max ||∇_x f(x)||_2 over the given batch.
        - analytic: compute data-dependent composed upper bound via local derivatives.
        """
        if method == "analytic":
            with torch.no_grad():
                return float(self.root.lipschitz_upper_bound(x).detach().cpu().item())

        if method == "grad":
            # Ensure autograd is enabled even if caller is in no_grad.
            self.eval()
            with torch.enable_grad():
                x_req = x.detach().clone().requires_grad_(True)
                y = self.forward(x_req)  # (N,1)
                # For scalar output, Lipschitz at x is ||grad||_2
                grads = torch.autograd.grad(
                    outputs=y,
                    inputs=x_req,
                    grad_outputs=torch.ones_like(y),
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[0]
                # grads shape (N,1) => norm per sample is abs
                return float(grads.norm(p=2, dim=1).max().detach().cpu().item())

        raise ValueError(f"Unknown method: {method}")

    def _build_tree(self, depth: int) -> TreeNode:
        """
        Depth is defined as the maximum number of operator applications along any
        root-to-leaf path in the intended ground-truth family.

        We build skeletons:
        d=1: linear(x)
        d=2: add( sin( linear(x) ), linear(x_const) )  -> implement bias via Linear1D with a=0 learnable
        d=3: mult( exp( linear(x) ), sin( linear(x) ) )
        d=4: add( mult(exp(linear(x)), sin(linear(x))), linear(x) )
        """
        if depth == 1:
            return TreeNode(unary="linear")

        # Build explicitly without type-ignore confusion:
        if depth == 2:
            # Wrap input through a learnable affine before sin:
            affine1 = TreeNode(unary="linear")
            # Compose by making sin node consume affine output: implement via a small adapter node.
            # Simpler: encode as sin(linear(x)) by defining a custom node type is overkill.
            # We emulate by making a "linear leaf" and then applying sin at root via UnaryOp on its output.
            # For that, we use a binary add at root with children: sin(linear(x)) and bias(linear(x) with a=0).
            sin_leaf = _ComposeUnaryOverChild(unary="sin", child=affine1)
            bias = _BiasLeaf()
            return TreeNode(binary="add", left=sin_leaf, right=bias)

        if depth == 3:
            env = _ComposeUnaryOverChild(unary="exp", child=TreeNode(unary="linear"))
            osc = _ComposeUnaryOverChild(unary="sin", child=TreeNode(unary="linear"))
            return TreeNode(binary="mult", left=env, right=osc)

        if depth == 4:
            env = _ComposeUnaryOverChild(unary="exp", child=TreeNode(unary="linear"))
            osc = _ComposeUnaryOverChild(unary="sin", child=TreeNode(unary="linear"))
            prod = TreeNode(binary="mult", left=env, right=osc)
            lin = TreeNode(unary="linear")
            return TreeNode(binary="add", left=prod, right=lin)

        raise ValueError(f"Unsupported depth: {depth}")


class _ComposeUnaryOverChild(TreeNode):
    """
    A helper node representing unary(child(x)).
    It is still a tree node, but with an explicit child subtree.
    """

    def __init__(self, unary: UnaryOpName, child: TreeNode) -> None:
        super().__init__(unary=unary)
        self.child = child

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.unary_op is not None
        return self.unary_op(self.child(x))

    @torch.no_grad()
    def lipschitz_upper_bound(self, x: torch.Tensor) -> torch.Tensor:
        assert self.unary_op is not None
        child_out = self.child(x)
        L_child = self.child.lipschitz_upper_bound(x)
        L_unary = self.unary_op.lipschitz_upper_bound(child_out)
        return L_unary * L_child


class _BiasLeaf(TreeNode):
    """
    Constant leaf implemented as linear(x) with slope fixed to 0 by construction.
    We keep a trainable bias only, so that it is a true constant wrt x, which
    yields Lipschitz 0 for this branch.
    """

    def __init__(self) -> None:
        super().__init__(unary="linear")
        assert self.unary_op is not None and self.unary_op.linear is not None
        # Freeze slope to exactly 0; allow bias to train.
        with torch.no_grad():
            self.unary_op.linear.a.fill_(0.0)
        self.unary_op.linear.a.requires_grad_(False)

    @torch.no_grad()
    def lipschitz_upper_bound(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=x.device, dtype=x.dtype)


@dataclass(frozen=True)
class TrainConfig:
    depth: int
    n_train: int
    n_test: int = 100_000
    batch_size: int = 256
    lr: float = 1e-2
    max_epochs: int = 5000
    weight_decay: float = 0.0
    patience: int = 200
    tol: float = 1e-6
    lipschitz_method: Literal["grad", "analytic"] = "grad"
    lipschitz_batch: int = 8192


def mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    se_sum = 0.0
    n_sum = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            se_sum += torch.sum((pred - y) ** 2).item()
            n_sum += x.shape[0]
    return se_sum / max(1, n_sum)


def fit_model(
    cfg: TrainConfig,
    spec: GroundTruthSpec,
    seed: int,
    device: torch.device,
    *,
    logger: logging.Logger,
    show_progress: bool,
) -> Dict[str, float]:
    set_seed(seed)
    model = CompositionalTreeModel(depth=cfg.depth).to(device)

    train_ds = PhysicsTreeDataset(spec=spec, n=cfg.n_train, seed=seed + 10_000, device=device)
    test_ds = PhysicsTreeDataset(spec=spec, n=cfg.n_test, seed=seed + 20_000, device=device)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_train = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_steps = 0

    epoch_iter: Iterable[int] = range(cfg.max_epochs)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc=f"train d={cfg.depth} n={cfg.n_train}", leave=False)

    for epoch in epoch_iter:
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * x.shape[0]
            n_seen += x.shape[0]
        train_mse = epoch_loss / max(1, n_seen)

        if train_mse + cfg.tol < best_train:
            best_train = train_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_steps = 0
        else:
            bad_steps += 1

        if bad_steps >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_err = mse(model, train_loader, device=device)
    test_err = mse(model, test_loader, device=device)
    gap = test_err - train_err

    # Lipschitz proxy estimation on a large random batch from the test distribution.
    x_lip = (spec.x_high - spec.x_low) * torch.rand(cfg.lipschitz_batch, 1, device=device) + spec.x_low
    L_hat = model.lipschitz_proxy(x_lip, method=cfg.lipschitz_method)

    # Complexity term used in the plot: (L^d)/sqrt(n).
    x_term = (L_hat ** cfg.depth) / math.sqrt(cfg.n_train)

    logger.info(
        "done depth=%d n=%d seed=%d epochs=%d train_mse=%.6g test_mse=%.6g gap=%.6g L_hat=%.6g term=%.6g",
        cfg.depth,
        cfg.n_train,
        seed,
        epoch + 1,
        train_err,
        test_err,
        gap,
        L_hat,
        x_term,
    )

    return dict(
        depth=float(cfg.depth),
        n_train=float(cfg.n_train),
        train_mse=float(train_err),
        test_mse=float(test_err),
        gen_gap=float(gap),
        L_hat=float(L_hat),
        x_term=float(x_term),
        epochs=float(epoch + 1),
    )


def plot_gap_vs_term(rows: Sequence[Dict[str, float]], outpath: str) -> None:
    x = np.array([r["x_term"] for r in rows], dtype=np.float64)
    y = np.array([r["gen_gap"] for r in rows], dtype=np.float64)
    d = np.array([int(r["depth"]) for r in rows], dtype=np.int32)

    plt.figure(figsize=(7.5, 5.5))
    for depth in sorted(set(d.tolist())):
        m = d == depth
        plt.scatter(x[m], y[m], s=22, alpha=0.85, label=f"depth={depth}")

    # Fit a line through the origin as a visual "upper envelope" guide.
    # (We keep it simple; feel free to replace with quantile regression.)
    denom = float(np.sum(x * x)) + 1e-12
    slope = float(np.sum(x * y) / denom)
    xx = np.linspace(float(x.min()) * 0.9, float(x.max()) * 1.1, 200)
    plt.plot(xx, slope * xx, linewidth=2.0, label=f"fit: gap ≈ {slope:.3g}·term")

    plt.xscale("log")
    plt.yscale("symlog", linthresh=1e-6)
    plt.xlabel(r"Theoretical term: $(\hat L^d)/\sqrt{n}$ (log scale)")
    plt.ylabel(r"Generalization gap: Test MSE − Train MSE (symlog)")
    plt.title("Generalization Gap vs. Compositional Lipschitz Complexity")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="icml2026/artifacts")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--lipschitz_method", type=str, choices=["grad", "analytic"], default="grad")
    p.add_argument("--max_epochs", type=int, default=5000)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--n_train", type=int, nargs="+", default=[50, 100, 500, 1000, 5000])
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars.")
    args = p.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Logging: console + file in outdir.
    logger = logging.getLogger("icml2026.experiments")
    logger.setLevel(getattr(logging, args.log_level))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, args.log_level))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(outdir, "run.log"), mode="w", encoding="utf-8")
    fh.setLevel(getattr(logging, args.log_level))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    device = torch.device(args.device)
    show_progress = (not args.no_tqdm)

    logger.info("starting sweep device=%s seeds=%d depths=%s n_train=%s", args.device, args.seeds, args.depths, args.n_train)

    all_rows: List[Dict[str, float]] = []
    jobs: List[Tuple[int, int, int]] = [(d, n, s) for d in args.depths for n in args.n_train for s in range(args.seeds)]
    job_iter: Iterable[Tuple[int, int, int]] = jobs
    if show_progress:
        job_iter = tqdm(job_iter, total=len(jobs), desc="sweep", leave=True)

    for depth, n_train, s in job_iter:
                spec = sample_ground_truth(depth=depth, seed=12345 + 1000 * depth + s)
                cfg = TrainConfig(
                    depth=depth,
                    n_train=n_train,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    lipschitz_method=args.lipschitz_method,
                )
                row = fit_model(
                    cfg=cfg,
                    spec=spec,
                    seed=999 + s,
                    device=device,
                    logger=logger,
                    show_progress=show_progress,
                )
                row["gt_name"] = float(depth)  # kept numeric for simple JSON; see gt_params below
                row["gt_params_json"] = 0.0
                # Store params alongside in a separate JSONL to keep schema stable.
                all_rows.append({**row})

    # Save results (CSV-like JSON for simplicity).
    results_path = os.path.join(outdir, "results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    fig_path = os.path.join(outdir, "gap_vs_term.png")
    plot_gap_vs_term(all_rows, fig_path)

    # Small human-readable summary.
    summary = {
        "n_rows": len(all_rows),
        "depths": args.depths,
        "n_train": args.n_train,
        "seeds": args.seeds,
        "lipschitz_method": args.lipschitz_method,
        "results_jsonl": results_path,
        "figure_png": fig_path,
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("finished: wrote %s and %s", results_path, fig_path)


if __name__ == "__main__":
    main()

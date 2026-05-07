#!/usr/bin/env python3
"""
eval_tinystories33m_convergence_smalllr_1k_professional.py

A cleaner TinyStories convergence experiment with publication-quality plots.

This version keeps the same experimental logic as the small-LR 1k-universe script:
  - fixed train universe / fixed val universe
  - Original / Top-90% / Bottom-90% / Random-90%
  - same-order stream across subsets
  - same initialization for all runs
  - DEV-based early stopping

Main presentation change:
  - the primary figure is the TEST-loss curve truncated at each run's best DEV checkpoint
  - y-axis limits are automatically tightened to zoom into the visible differences
  - labels / titles / legend are made more professional
  - a caption text file is written for easy reuse in the paper
"""

from __future__ import annotations

import os

# ------------------------------------------------------------------
# Force HuggingFace caches under /deac
# ------------------------------------------------------------------
os.environ["HF_HOME"] = "xxxxxx"
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import csv
import json
import math
import time
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader

from load_model_tiny import Config as DataConfig, load_tinystories_33m


# =========================
# CONFIG
# =========================
@dataclass
class Cfg:
    # Data written by load_model_data_tinystories33m.py
    data_output_dir: str = "../output/tinystories33m_setup"
    train_subdir: str = "tinystories_train_blocks"
    val_subdir: str = "tinystories_val_blocks"

    # DuoShap output written by train_tinystories33m_soda_inrun.py
    shapley_dir: str = "../output/tinystories33m_soda_run"
    shapley_csv: str = "soda_inrun_shapley_values.csv"

    # Output
    out_dir: str = "../output/eval_tinystories33m_test_loss"

    # Smaller universe to better match the earlier convergence comparison
    train_universe: int = 1000
    val_universe: int = 1000

    # Subset construction
    subset_drop_frac: float = 0.10   # drop 10% => keep 90%

    # Dev/Test split from the fixed validation universe
    dev_size: int = 200
    test_size: int = 500
    split_seed: int = 777

    # Gentler continued-training setup
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0

    # Early-convergence focused budget
    target_epochs: float = 1.5
    max_steps_cap: int = 120
    eval_every_steps: int = 5

    # Early stopping on DEV only
    early_stop_patience_evals: int = 4
    early_stop_min_steps: int = 15

    # Track loss on a fixed small train-probe subset
    train_probe_size: int = 200

    # Plotting
    figure_dpi: int = 320
    y_zoom_pad_ratio: float = 0.12
    y_zoom_min_pad: float = 0.0015

    # Reproducibility
    seed: int = 42
    order_seed: int = 4242

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


DISPLAY_NAMES = {
    "original_100pct": "Original-100%",
    "top_keep90pct": "Top-90%",
    "bottom_keep90pct": "Bottom-90%",
    "random_keep90pct": "Random-90%",
}


# =========================
# Repro helpers
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# IO helpers
# =========================
def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def read_shapley_dict(csv_path: Path) -> Dict[int, float]:
    d: Dict[int, float] = {}
    with csv_path.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            d[int(r["train_id"])] = float(r["phi_avg"])
    if not d:
        raise RuntimeError(f"No Shapley rows read from {csv_path}")
    return d


def ensure_example_id(ds):
    if "example_id" in ds.column_names:
        return ds
    return ds.add_column("example_id", list(range(len(ds))))


def set_torch_format(ds):
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "example_id"],
    )
    return ds


def write_summary_csv(path: Path, runs: List[Dict[str, Any]]) -> None:
    fields = [
        "run_name",
        "train_size",
        "dev_size",
        "test_size",
        "target_epochs",
        "max_steps_used",
        "initial_train_probe_loss",
        "initial_test_loss",
        "best_dev_loss",
        "step_at_best_dev",
        "test_at_best_dev",
        "final_train_probe_loss",
        "final_test_loss",
        "elapsed_sec",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in runs:
            w.writerow({k: r[k] for k in fields})


def write_curves_csv(path: Path, runs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["run_name", "step", "train_probe_loss", "dev_loss", "test_loss", "wall_sec"],
        )
        w.writeheader()
        for r in runs:
            for p in r["curve"]:
                w.writerow(
                    {
                        "run_name": r["run_name"],
                        "step": p["step"],
                        "train_probe_loss": p["train_probe_loss"],
                        "dev_loss": p["dev_loss"],
                        "test_loss": p["test_loss"],
                        "wall_sec": p["wall_sec"],
                    }
                )


def write_caption_file(path: Path, cfg: Cfg) -> None:
    caption = (
        "Test loss versus optimization steps for TinyStories-33M retrained on filtered "
        "TinyStories subsets. We compare the original training set (Original-100%) with "
        "three 90% subsets: Top-90% (remove the lowest 10% by DuoShap score), Bottom-90% "
        "(remove the highest 10%), and Random-90% (remove a uniformly random 10%). "
        "All runs start from the same initialization, use the same optimizer settings, "
        "and are selected by dev-based early stopping. The main figure is truncated at the "
        "best dev checkpoint for each run to emphasize early convergence before late-stage "
        "overfitting."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption)


# =========================
# Same-order training stream
# =========================
class OrderStream:
    def __init__(self, N: int, keep_mask: np.ndarray, order_seed: int):
        self.N = int(N)
        self.keep_mask = keep_mask.astype(bool)
        self.order_seed = int(order_seed)

        if int(self.keep_mask.sum()) <= 0:
            raise RuntimeError("Empty subset keep_mask.")

        self.epoch = 0
        self.pos = 0
        self.perm = self._make_perm(self.epoch)

    def _make_perm(self, epoch: int) -> np.ndarray:
        rng = np.random.RandomState(self.order_seed + epoch)
        return rng.permutation(self.N)

    def next_batch(self, batch_size: int) -> List[int]:
        batch = []
        while len(batch) < batch_size:
            if self.pos >= self.N:
                self.epoch += 1
                self.pos = 0
                self.perm = self._make_perm(self.epoch)

            idx = int(self.perm[self.pos])
            self.pos += 1

            if self.keep_mask[idx]:
                batch.append(idx)

        return batch


# =========================
# Evaluation
# =========================
@torch.no_grad()
def eval_loss(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        bs = int(batch["input_ids"].shape[0])
        batch = {k: v.to(device) for k, v in batch.items() if k != "example_id"}

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        total_loss += float(out.loss.item()) * bs
        total_items += bs

    return total_loss / max(1, total_items)


def compute_max_steps_for_subset(cfg: Cfg, subset_size: int) -> int:
    steps = int(math.ceil(cfg.target_epochs * (subset_size / max(1, cfg.batch_size))))
    steps = max(1, steps)
    return int(min(cfg.max_steps_cap, steps))


# =========================
# One retraining run
# =========================
def train_run(
    run_name: str,
    cfg: Cfg,
    train_ds,
    keep_mask: np.ndarray,
    dev_ds,
    test_ds,
    init_state_dict_cpu: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    device = torch.device(cfg.device)

    data_cfg = DataConfig()
    _, model = load_tinystories_33m(data_cfg)
    model.load_state_dict({k: v.to(device) for k, v in init_state_dict_cpu.items()})
    model.to(device)

    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # fixed train-probe subset for debugging whether train-side loss is going down
    train_probe_n = min(cfg.train_probe_size, len(train_ds))
    train_probe_ds = set_torch_format(train_ds.select(range(train_probe_n)))
    train_probe_loader = DataLoader(train_probe_ds, batch_size=cfg.batch_size, shuffle=False)

    opt = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    subset_size = int(keep_mask.sum())
    max_steps = compute_max_steps_for_subset(cfg, subset_size)
    stream = OrderStream(N=len(train_ds), keep_mask=keep_mask, order_seed=cfg.order_seed)

    curve: List[Dict[str, Any]] = []
    start = time.time()

    # Step 0 evaluation
    train0 = eval_loss(model, train_probe_loader, device)
    dev0 = eval_loss(model, dev_loader, device)
    test0 = eval_loss(model, test_loader, device)
    curve.append(
        {
            "step": 0,
            "train_probe_loss": float(train0),
            "dev_loss": float(dev0),
            "test_loss": float(test0),
            "wall_sec": 0.0,
        }
    )

    best_dev = float(dev0)
    step_at_best = 0
    test_at_best = float(test0)
    bad_evals = 0

    for step in range(1, max_steps + 1):
        idxs = stream.next_batch(cfg.batch_size)
        items = [train_ds[i] for i in idxs]

        batch = {
            "input_ids": torch.stack([x["input_ids"] for x in items]).to(device),
            "attention_mask": torch.stack([x["attention_mask"] for x in items]).to(device),
            "labels": torch.stack([x["labels"] for x in items]).to(device),
        }

        model.train()
        opt.zero_grad(set_to_none=True)

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = out.loss
        loss.backward()

        if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        opt.step()

        if cfg.eval_every_steps > 0 and (step % cfg.eval_every_steps == 0 or step == max_steps):
            trainv = eval_loss(model, train_probe_loader, device)
            devv = eval_loss(model, dev_loader, device)
            testv = eval_loss(model, test_loader, device)

            curve.append(
                {
                    "step": int(step),
                    "train_probe_loss": float(trainv),
                    "dev_loss": float(devv),
                    "test_loss": float(testv),
                    "wall_sec": float(time.time() - start),
                }
            )

            improved = (devv < best_dev - 1e-6)
            if improved:
                best_dev = float(devv)
                step_at_best = int(step)
                test_at_best = float(testv)
                bad_evals = 0
            else:
                if step >= int(cfg.early_stop_min_steps):
                    bad_evals += 1

            if bad_evals >= int(cfg.early_stop_patience_evals):
                break

    final_train = float(curve[-1]["train_probe_loss"])
    final_dev = float(curve[-1]["dev_loss"])
    final_test = float(curve[-1]["test_loss"])

    return {
        "run_name": run_name,
        "train_size": int(subset_size),
        "dev_size": int(len(dev_ds)),
        "test_size": int(len(test_ds)),
        "target_epochs": float(cfg.target_epochs),
        "max_steps_used": int(curve[-1]["step"]),
        "eval_every_steps": int(cfg.eval_every_steps),
        "initial_train_probe_loss": float(curve[0]["train_probe_loss"]),
        "initial_dev_loss": float(curve[0]["dev_loss"]),
        "initial_test_loss": float(curve[0]["test_loss"]),
        "best_dev_loss": float(best_dev),
        "step_at_best_dev": int(step_at_best),
        "test_at_best_dev": float(test_at_best),
        "final_train_probe_loss": float(final_train),
        "final_dev_loss": float(final_dev),
        "final_test_loss": float(final_test),
        "elapsed_sec": float(time.time() - start),
        "curve": curve,
    }


# =========================
# Plotting helpers
# =========================
def _display_name(run_name: str) -> str:
    return DISPLAY_NAMES.get(run_name, run_name)


def _prepare_points(runs: List[Dict[str, Any]], truncate_at_best_dev: bool) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    out = []
    for r in runs:
        pts = r["curve"]
        if truncate_at_best_dev:
            best_step = int(r["step_at_best_dev"])
            pts = [p for p in pts if int(p["step"]) <= best_step]
        out.append((r, pts))
    return out


def _set_zoomed_ylim(ax, values: List[float], cfg: Cfg) -> None:
    if not values:
        return
    ymin = min(values)
    ymax = max(values)
    yrange = ymax - ymin
    pad = max(cfg.y_zoom_min_pad, cfg.y_zoom_pad_ratio * max(yrange, 1e-8))
    ax.set_ylim(ymin - pad, ymax + pad)


def _style_axis(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_test_loss_vs_step(
    out_path: Path,
    runs: List[Dict[str, Any]],
    cfg: Cfg,
    title: str,
    subtitle: str,
    logy: bool,
    truncate_at_best_dev: bool,
    zoom_y: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.8))

    plotted_values: List[float] = []
    prepared = _prepare_points(runs, truncate_at_best_dev)

    for r, pts in prepared:
        xs = [p["step"] for p in pts]
        ys = [p["test_loss"] for p in pts]
        if logy:
            ys = [max(y, 1e-10) for y in ys]

        plotted_values.extend(ys)
        label = f'{_display_name(r["run_name"])} (n={r["train_size"]}, best dev @ {r["step_at_best_dev"]})'
        ax.plot(xs, ys, linewidth=2.2, label=label)
        ax.scatter([r["step_at_best_dev"]], [r["test_at_best_dev"]], s=28, zorder=3)

    ax.set_xlabel("Optimization step", fontsize=12)
    ax.set_ylabel("Test loss", fontsize=12)
    ax.set_title(title, fontsize=14, pad=14)
    fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)

    if logy:
        ax.set_yscale("log")

    if zoom_y and not logy:
        _set_zoomed_ylim(ax, plotted_values, cfg)

    _style_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="best")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_train_probe_loss_vs_step(
    out_path: Path,
    runs: List[Dict[str, Any]],
    cfg: Cfg,
    title: str,
    subtitle: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.8))

    plotted_values: List[float] = []
    for r in runs:
        pts = r["curve"]
        xs = [p["step"] for p in pts]
        ys = [p["train_probe_loss"] for p in pts]
        plotted_values.extend(ys)
        ax.plot(xs, ys, linewidth=2.2, label=f'{_display_name(r["run_name"])} (n={r["train_size"]})')

    ax.set_xlabel("Optimization step", fontsize=12)
    ax.set_ylabel("Train-probe loss", fontsize=12)
    ax.set_title(title, fontsize=14, pad=14)
    fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)

    _set_zoomed_ylim(ax, plotted_values, cfg)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="best")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    cfg = Cfg()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (0.0 < cfg.subset_drop_frac < 1.0):
        raise ValueError("subset_drop_frac must be in (0, 1).")

    set_seed(cfg.seed)

    # -------------------------
    # Load fixed blocked datasets
    # -------------------------
    root = Path(cfg.data_output_dir)
    train_ds = ensure_example_id(load_from_disk(str(root / cfg.train_subdir)))
    val_ds = ensure_example_id(load_from_disk(str(root / cfg.val_subdir)))

    train_ds = train_ds.select(range(min(cfg.train_universe, len(train_ds))))
    val_ds = val_ds.select(range(min(cfg.val_universe, len(val_ds))))

    required_cols = {"input_ids", "attention_mask", "labels", "example_id"}
    if not required_cols.issubset(set(train_ds.column_names)):
        raise RuntimeError(f"Train dataset missing columns: {required_cols - set(train_ds.column_names)}")
    if not required_cols.issubset(set(val_ds.column_names)):
        raise RuntimeError(f"Val dataset missing columns: {required_cols - set(val_ds.column_names)}")

    train_ds = set_torch_format(train_ds)
    val_ds = set_torch_format(val_ds)

    train_ids = [int(x) for x in train_ds["example_id"]]

    # -------------------------
    # Load Shapley and enforce universe alignment
    # -------------------------
    shapley_path = Path(cfg.shapley_dir) / cfg.shapley_csv
    phi = read_shapley_dict(shapley_path)

    missing = [tid for tid in train_ids if tid not in phi]
    if missing:
        raise RuntimeError(
            f"Shapley CSV missing {len(missing)} / {len(train_ids)} training ids. "
            f"Example missing ids: {missing[:10]}. "
            f"Please make train_universe exactly match your DuoShap run."
        )

    # -------------------------
    # Build subsets
    # -------------------------
    rows = [(tid, phi[tid]) for tid in train_ids]
    rows_sorted = sorted(rows, key=lambda x: x[1])  # ascending by phi_avg

    N = len(train_ids)
    drop_n = int(round(cfg.subset_drop_frac * N))
    drop_n = max(1, min(drop_n, N - 1))
    keep_n = N - drop_n

    top_ids = set(tid for (tid, _) in rows_sorted[drop_n:])
    bottom_ids = set(tid for (tid, _) in rows_sorted[:keep_n])

    rng = np.random.RandomState(cfg.seed + 999)
    random_ids = set(rng.choice(train_ids, size=keep_n, replace=False).tolist())

    # -------------------------
    # Dev/Test split from fixed val universe
    # -------------------------
    n_val = len(val_ds)
    need = min(n_val, cfg.dev_size + cfg.test_size)

    rngs = np.random.RandomState(cfg.split_seed)
    chosen = rngs.choice(n_val, size=need, replace=False).tolist()

    dev_idx = chosen[: min(cfg.dev_size, len(chosen))]
    test_idx = chosen[min(cfg.dev_size, len(chosen)) : min(cfg.dev_size + cfg.test_size, len(chosen))]

    dev_ds = set_torch_format(val_ds.select(dev_idx))
    test_ds = set_torch_format(val_ds.select(test_idx))

    write_json(
        out_dir / "split_info.json",
        {
            "split_seed": int(cfg.split_seed),
            "val_universe": int(n_val),
            "dev_size": int(len(dev_idx)),
            "test_size": int(len(test_idx)),
            "dev_indices": dev_idx,
            "test_indices": test_idx,
        },
    )

    # -------------------------
    # Same initialization for all runs
    # -------------------------
    data_cfg = DataConfig()
    _, init_model = load_tinystories_33m(data_cfg)
    init_state_dict_cpu = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
    del init_model

    # -------------------------
    # ID -> row map, then masks
    # -------------------------
    id_to_row = {tid: i for i, tid in enumerate(train_ids)}

    def mask_from_ids(idset: set[int]) -> np.ndarray:
        m = np.zeros(N, dtype=bool)
        for tid in idset:
            m[id_to_row[tid]] = True
        return m

    keep_full = np.ones(N, dtype=bool)
    keep_top = mask_from_ids(top_ids)
    keep_bottom = mask_from_ids(bottom_ids)
    keep_rand = mask_from_ids(random_ids)

    write_json(
        out_dir / "subset_info.json",
        {
            "train_universe": int(N),
            "subset_drop_frac": float(cfg.subset_drop_frac),
            "drop_n": int(drop_n),
            "keep_n": int(keep_n),
            "original_kept": int(keep_full.sum()),
            "top_kept": int(keep_top.sum()),
            "bottom_kept": int(keep_bottom.sum()),
            "random_kept": int(keep_rand.sum()),
        },
    )
    write_json(out_dir / "config.json", asdict(cfg))
    write_caption_file(out_dir / "figure_caption.txt", cfg)

    # -------------------------
    # Run experiments
    # -------------------------
    runs: List[Dict[str, Any]] = []

    runs.append(train_run("original_100pct", cfg, train_ds, keep_full, dev_ds, test_ds, init_state_dict_cpu))
    write_json(out_dir / "run_original_100pct.json", runs[-1])

    runs.append(train_run("top_keep90pct", cfg, train_ds, keep_top, dev_ds, test_ds, init_state_dict_cpu))
    write_json(out_dir / "run_top_keep90pct.json", runs[-1])

    runs.append(train_run("bottom_keep90pct", cfg, train_ds, keep_bottom, dev_ds, test_ds, init_state_dict_cpu))
    write_json(out_dir / "run_bottom_keep90pct.json", runs[-1])

    runs.append(train_run("random_keep90pct", cfg, train_ds, keep_rand, dev_ds, test_ds, init_state_dict_cpu))
    write_json(out_dir / "run_random_keep90pct.json", runs[-1])

    # -------------------------
    # Save tables + curves
    # -------------------------
    write_summary_csv(out_dir / "summary.csv", runs)
    write_curves_csv(out_dir / "curves.csv", runs)

    # -------------------------
    # Plots
    # -------------------------
    title_main = "TinyStories-33M Convergence on Filtered TinyStories Subsets"
    subtitle_main = (
        f"Test loss vs. optimization step | train universe={len(train_ds)}, "
        f"dev={len(dev_ds)}, test={len(test_ds)} | dev-based early stopping"
    )

    plot_test_loss_vs_step(
        out_dir / "figure_main_test_loss_bestdev_zoom.png",
        runs,
        cfg,
        title=title_main,
        subtitle=subtitle_main + " | main figure truncated at each run's best dev step",
        logy=False,
        truncate_at_best_dev=True,
        zoom_y=True,
    )

    plot_test_loss_vs_step(
        out_dir / "figure_full_test_loss_zoom.png",
        runs,
        cfg,
        title=title_main,
        subtitle=subtitle_main + " | full curves",
        logy=False,
        truncate_at_best_dev=False,
        zoom_y=True,
    )

    plot_test_loss_vs_step(
        out_dir / "figure_full_test_loss_logy.png",
        runs,
        cfg,
        title=title_main,
        subtitle=subtitle_main + " | full curves, log-scale y-axis",
        logy=True,
        truncate_at_best_dev=False,
        zoom_y=False,
    )

    plot_train_probe_loss_vs_step(
        out_dir / "figure_train_probe_loss_zoom.png",
        runs,
        cfg,
        title="TinyStories-33M Optimization Behavior on a Fixed Train-Probe Subset",
        subtitle=f"Train-probe loss vs. optimization step | probe size={cfg.train_probe_size}",
    )

    print(f"[DONE] outputs in: {out_dir}")
    print(f"  - {out_dir / 'summary.csv'}")
    print(f"  - {out_dir / 'curves.csv'}")
    print(f"  - {out_dir / 'figure_main_test_loss_bestdev_zoom.png'}")
    print(f"  - {out_dir / 'figure_caption.txt'}")


if __name__ == "__main__":
    main()
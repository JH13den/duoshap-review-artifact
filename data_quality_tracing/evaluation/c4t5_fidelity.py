#!/usr/bin/env python3
"""
eval_inrun_fidelity_multibatch_shard.py

Multi-batch fidelity check for in-run data attribution.

Main logic:
- Keep EACH batch as its own local Shapley game.
- For each batch:
    (A) compute MC permutation Shapley on the SAME local utility U(S)
    (B) compute DuoShap / tabular SODA antithetic estimate on the SAME U(S)
- Pool all (MC, DuoShap) point pairs across many batches for denser visualization.

Important update in this version:
- By default, batches are DISJOINT (no repeated training examples across batches).
- We shuffle the train dataset once, then split it into fixed non-overlapping chunks.

Parallelization:
- Use sharding across global batch IDs:
    global_batch_id % num_shards == shard_id
- Run several shards in parallel (e.g. Slurm array jobs), then merge results later.
"""

from __future__ import annotations

import os
import csv
import json
import time
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from load_model_data import Config as DataConfig, load_t5_small
from soda_data_shapely import (
    SODAInRunConfig,
    set_seed,
    load_tokenized_c4_subsets,
    backup_params,
    restore_params,
    compute_val_loss,
    simulate_one_subset_update,
)


# -----------------------------
# Determinism helpers
# -----------------------------
def set_full_seed(seed: int) -> None:
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def disable_all_dropout(model: torch.nn.Module) -> List[Tuple[torch.nn.Module, float]]:
    changed = []
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            changed.append((m, m.p))
            m.p = 0.0
    return changed


def restore_dropout(changed: List[Tuple[torch.nn.Module, float]]) -> None:
    for m, p in changed:
        m.p = p


# -----------------------------
# Config
# -----------------------------
@dataclass
class FidelityEvalConfig:
    output_dir: str = "../output/eval_exp2_fidelity_multibatch_111"
    seed: int = 42

    # Fidelity experiment size
    batch_size: int = 32
    num_batches_total: int = 300
    max_train_examples: int = 10000
    max_val_examples: int = 1000

    # Use disjoint non-overlapping batches by default
    disjoint_batches: bool = True

    # Validation subset used by local utility U(S)
    max_val_for_local_utility: int = 64
    randomize_val_for_utility: bool = True
    val_utility_seed_offset: int = 2026
    per_batch_randomize_val_utility: bool = False
    # False = one shared fixed random val subset across all batches in this run
    # True  = each batch gets its own random val subset

    # Utility step size
    sgd_proxy_lr: float = 1e-4

    # Estimator budgets
    N_per_i: int = 10
    mc_num_permutations: int = 1000

    # Plot formatting (for optional shard-level plots only)
    force_sci_exp: int = -3
    paper_like_lim: float = 2.5e-3
    zoom_quantile: float = 0.995

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Sharding
    shard_id: int = 0
    num_shards: int = 1


# -----------------------------
# Batch construction utilities
# -----------------------------
def sample_random_batch_from_dataset(
    dataset,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Uniformly sample `batch_size` distinct examples from a torch-formatted HF dataset.
    This is only used when disjoint_batches=False.
    """
    n = len(dataset)
    if batch_size > n:
        raise ValueError(f"batch_size={batch_size} > dataset size={n}")

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=batch_size, replace=False)

    samples = [dataset[int(i)] for i in idx]
    keys = samples[0].keys()

    batch: Dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [s[k] for s in samples]

        if k == "example_id":
            ids = []
            for v in vals:
                if torch.is_tensor(v):
                    ids.append(int(v.item()) if v.numel() == 1 else int(v.view(-1)[0].item()))
                else:
                    ids.append(int(v))
            batch[k] = torch.tensor(ids, dtype=torch.long, device=device)
        else:
            tens = []
            for v in vals:
                if not torch.is_tensor(v):
                    v = torch.tensor(v)
                tens.append(v.to(device))
            batch[k] = torch.stack(tens, dim=0)

    return batch


def stack_examples_from_indices(
    dataset,
    indices: List[int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Create a batch dict from a fixed list of dataset indices.
    Used for the disjoint no-repeat batch plan.
    """
    samples = [dataset[int(i)] for i in indices]
    keys = samples[0].keys()

    batch: Dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [s[k] for s in samples]

        if k == "example_id":
            ids = []
            for v in vals:
                if torch.is_tensor(v):
                    ids.append(int(v.item()) if v.numel() == 1 else int(v.view(-1)[0].item()))
                else:
                    ids.append(int(v))
            batch[k] = torch.tensor(ids, dtype=torch.long, device=device)
        else:
            tens = []
            for v in vals:
                if not torch.is_tensor(v):
                    v = torch.tensor(v)
                tens.append(v.to(device))
            batch[k] = torch.stack(tens, dim=0)

    return batch


def build_disjoint_batch_plan(
    dataset_size: int,
    batch_size: int,
    num_batches_total: int,
    seed: int,
) -> List[List[int]]:
    """
    Shuffle the dataset ONCE and split into disjoint non-overlapping full batches.

    Example:
      dataset_size = 10000
      batch_size = 32
      max disjoint full batches = 312
    """
    max_full_batches = dataset_size // batch_size
    if num_batches_total > max_full_batches:
        raise ValueError(
            f"num_batches_total={num_batches_total} exceeds max disjoint full batches={max_full_batches} "
            f"for dataset_size={dataset_size}, batch_size={batch_size}"
        )

    rng = np.random.RandomState(seed + 12345)
    perm = rng.permutation(dataset_size)

    plan: List[List[int]] = []
    for b in range(num_batches_total):
        start = b * batch_size
        end = (b + 1) * batch_size
        plan.append(perm[start:end].tolist())

    return plan


def sample_random_val_inputs(
    val_dataset,
    n_val: int,
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Build val_inputs dict used by compute_val_loss / simulate_one_subset_update.
    """
    n = len(val_dataset)
    n_val = min(n_val, n)

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=n_val, replace=False)

    samples = [val_dataset[int(i)] for i in idx]
    return {
        "input_ids": torch.stack([s["input_ids"].to(device) for s in samples], dim=0),
        "attention_mask": torch.stack([s["attention_mask"].to(device) for s in samples], dim=0),
        "labels": torch.stack([s["labels"].to(device) for s in samples], dim=0),
    }


# -----------------------------
# Utility cache (bitmask-based)
# -----------------------------
class UtilityCache:
    """
    Cache U(S) for subsets S of one fixed batch, keyed by bitmask int.
    U(empty)=0.0.

    Valid because:
      - all U(S) are evaluated at the same w_t
      - simulate_one_subset_update() restores to backup
      - dropout is disabled
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        val_inputs: Dict[str, torch.Tensor],
        sgd_proxy_lr: float,
    ):
        self.model = model
        self.batch = batch
        self.val_inputs = val_inputs
        self.lr = float(sgd_proxy_lr)

        self.B = int(batch["input_ids"].shape[0])
        self.full_mask = (1 << self.B) - 1

        self.backup = backup_params(model)
        self.base_val_loss = compute_val_loss(model, val_inputs)

        self.cache: Dict[int, float] = {0: 0.0}

    def mask_to_indices(self, mask: int) -> List[int]:
        idx = []
        b = 0
        while mask:
            if mask & 1:
                idx.append(b)
            mask >>= 1
            b += 1
        return idx

    def U(self, mask: int) -> float:
        mask = int(mask)
        if mask in self.cache:
            return self.cache[mask]

        subset_indices = self.mask_to_indices(mask)
        val = float(
            simulate_one_subset_update(
                model=self.model,
                batch=self.batch,
                subset_indices=subset_indices,
                val_inputs=self.val_inputs,
                base_val_loss=self.base_val_loss,
                learning_rate=self.lr,
                backup=self.backup,
            )
        )
        self.cache[mask] = val
        return val


# -----------------------------
# Estimators
# -----------------------------
def estimate_phi_mc_permutation(U: UtilityCache, n_perm: int, seed: int) -> np.ndarray:
    """
    MC permutation Shapley on the SAME local utility U(S).
    """
    rng = np.random.RandomState(seed)
    B = U.B
    phi = np.zeros(B, dtype=np.float64)

    all_idx = list(range(B))
    for _ in range(int(n_perm)):
        perm = all_idx[:]
        rng.shuffle(perm)

        S_mask = 0
        U_S = 0.0
        for i in perm:
            S_plus = S_mask | (1 << i)
            U_Splus = U.U(S_plus)
            phi[i] += (U_Splus - U_S)
            S_mask = S_plus
            U_S = U_Splus

    return (phi / float(n_perm)).astype(np.float64)


def estimate_phi_soda_tabular_antithetic_per_i(U: UtilityCache, N_per_i: int, seed: int) -> np.ndarray:
    """
    Tabular DuoShap / SODA estimator with per-i antithetic grouping.

    Uses the group statistic:
        g(S) = U(S) - U(M \\ S)

    For each i:
      repeat N_per_i times:
        sample K ~ Uniform{1, ..., B-1}
        sample S of size K such that i in S
        define antithetic A = (M \\ S) ∪ {i}
        accumulate g(S), g(A)
      phi_i = average over 2 * N_per_i group values
    """
    rng = np.random.RandomState(seed)
    B = U.B
    full = U.full_mask

    phi = np.zeros(B, dtype=np.float64)
    cnt = np.zeros(B, dtype=np.int64)

    def g(mask_S: int) -> float:
        mask_S = int(mask_S)
        mask_Sc = full & (~mask_S)
        return float(U.U(mask_S) - U.U(mask_Sc))

    for i in range(B):
        bit_i = 1 << i
        for _ in range(int(N_per_i)):
            k = int(rng.randint(1, B))  # 1..B-1

            if k == 1:
                mask_S = bit_i
            else:
                pool = [j for j in range(B) if j != i]
                others = rng.choice(pool, size=k - 1, replace=False).tolist()
                mask_S = bit_i
                for j in others:
                    mask_S |= (1 << int(j))

            mask_Sc = full & (~mask_S)
            mask_A = mask_Sc | bit_i

            phi[i] += g(mask_S)
            phi[i] += g(mask_A)
            cnt[i] += 2

    return (phi / np.maximum(cnt, 1)).astype(np.float64)


# -----------------------------
# Metrics / helpers
# -----------------------------
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def assigned_global_batch_ids(num_batches_total: int, shard_id: int, num_shards: int) -> List[int]:
    return [b for b in range(num_batches_total) if (b % num_shards) == shard_id]


def ensure_csv_with_header(path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)


def choose_point_style(n_points: int) -> Tuple[float, float]:
    if n_points <= 256:
        return 18.0, 0.55
    if n_points <= 512:
        return 12.0, 0.42
    if n_points <= 1000:
        return 8.0, 0.30
    return 6.0, 0.22


def plot_scatter_paper_like(
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    rmse_val: float,
    corr_val: float,
    n_batches: int,
    exp: int = -3,
    paper_like_lim: float = 2.5e-3,
    mode: str = "full",
    zoom_quantile: float = 0.995,
) -> None:
    """
    Optional shard-level plot only.
    Final paper plot should still come from the merge script.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0:
        return

    scale = 10 ** (-exp)
    xs = x * scale
    ys = y * scale

    abs_all = np.abs(np.concatenate([x, y]))
    max_abs = float(np.max(abs_all))

    if mode == "full":
        lim = paper_like_lim if max_abs <= paper_like_lim else 1.05 * max_abs
    elif mode == "zoom":
        q = float(np.quantile(abs_all, zoom_quantile))
        lim = max(paper_like_lim, 1.10 * q)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    lims = lim * scale
    point_size, point_alpha = choose_point_style(len(x))

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(
        xs,
        ys,
        s=point_size,
        alpha=point_alpha,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([-lims, lims], [-lims, lims], linestyle="--", linewidth=1.5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lims, lims)
    ax.set_ylim(-lims, lims)
    ax.grid(True, alpha=0.22)

    title_suffix = "full range" if mode == "full" else f"zoom ({zoom_quantile:.3f} quantile)"
    ax.set_title(
        f"DuoShap vs MC (pooled across {n_batches} batches, {title_suffix})\n"
        f"RMSE={rmse_val:.3g}, Corr={corr_val:.3g}",
        fontsize=12,
    )

    ax.set_xlabel(r"MC permutation Shapley ($\times 10^{-3}$)", labelpad=10)
    ax.set_ylabel(r"DuoShap estimate ($\times 10^{-3}$)", labelpad=10)

    fig.subplots_adjust(left=0.18, bottom=0.16, right=0.98, top=0.87)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args() -> FidelityEvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="../output/eval_exp2_fidelity_multibatch")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_batches_total", type=int, default=300)
    parser.add_argument("--max_train_examples", type=int, default=10000)
    parser.add_argument("--max_val_examples", type=int, default=1000)

    parser.add_argument("--disjoint_batches", action="store_true")
    parser.add_argument("--allow_repeat_batches", dest="disjoint_batches", action="store_false")
    parser.set_defaults(disjoint_batches=True)

    parser.add_argument("--max_val_for_local_utility", type=int, default=64)
    parser.add_argument("--randomize_val_for_utility", action="store_true")
    parser.add_argument("--no_randomize_val_for_utility", dest="randomize_val_for_utility", action="store_false")
    parser.set_defaults(randomize_val_for_utility=True)

    parser.add_argument("--per_batch_randomize_val_utility", action="store_true")
    parser.add_argument("--sgd_proxy_lr", type=float, default=1e-4)

    parser.add_argument("--N_per_i", type=int, default=10)
    parser.add_argument("--mc_num_permutations", type=int, default=1000)

    parser.add_argument("--force_sci_exp", type=int, default=-3)
    parser.add_argument("--paper_like_lim", type=float, default=2.5e-3)
    parser.add_argument("--zoom_quantile", type=float, default=0.995)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    args = parser.parse_args()
    return FidelityEvalConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_batches_total=args.num_batches_total,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
        disjoint_batches=args.disjoint_batches,
        max_val_for_local_utility=args.max_val_for_local_utility,
        randomize_val_for_utility=args.randomize_val_for_utility,
        per_batch_randomize_val_utility=args.per_batch_randomize_val_utility,
        sgd_proxy_lr=args.sgd_proxy_lr,
        N_per_i=args.N_per_i,
        mc_num_permutations=args.mc_num_permutations,
        force_sci_exp=args.force_sci_exp,
        paper_like_lim=args.paper_like_lim,
        zoom_quantile=args.zoom_quantile,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    E = parse_args()
    set_full_seed(E.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    out_dir = Path(E.output_dir)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    points_path = shard_dir / f"shard_{E.shard_id:03d}_points.csv"
    batch_summary_path = shard_dir / f"shard_{E.shard_id:03d}_batch_summary.csv"
    shard_meta_path = shard_dir / f"shard_{E.shard_id:03d}_meta.json"

    point_fields = [
        "global_batch_id",
        "pos_in_batch",
        "train_id",
        "dataset_index",
        "phi_gt_mc",
        "phi_est_soda_antithetic",
        "diff",
        "batch_seed",
        "mc_seed",
        "duoshap_seed",
    ]
    batch_fields = [
        "global_batch_id",
        "batch_seed",
        "mc_seed",
        "duoshap_seed",
        "rmse",
        "corr",
        "sum_phi_gt",
        "sum_phi_est",
        "phi_gt_std",
        "phi_est_std",
        "U_M",
        "base_val_loss",
        "utility_cache_size",
        "elapsed_sec",
    ]

    ensure_csv_with_header(points_path, point_fields)
    ensure_csv_with_header(batch_summary_path, batch_fields)

    device = torch.device(E.device)
    print(f"[INFO] device={device}")
    print(f"[INFO] shard_id={E.shard_id} / {E.num_shards}")

    batch_ids = assigned_global_batch_ids(E.num_batches_total, E.shard_id, E.num_shards)
    print(f"[INFO] assigned global batch IDs: {batch_ids}")

    # Build cfg used by existing data loader
    cfg = SODAInRunConfig()
    cfg.max_train_examples = E.max_train_examples
    cfg.max_val_examples = E.max_val_examples
    cfg.max_val_for_local_utility = E.max_val_for_local_utility
    cfg.sgd_proxy_lr = E.sgd_proxy_lr
    cfg.device = E.device

    # Load model/tokenizer/data once per shard
    data_cfg = DataConfig()
    tokenizer, model = load_t5_small(data_cfg)
    model.to(device)

    dropout_backup = disable_all_dropout(model)
    model.eval()

    train_loader, val_loader, val_inputs_default = load_tokenized_c4_subsets(cfg, tokenizer, device)
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    print(f"[INFO] train_dataset_size={len(train_dataset)}")
    print(f"[INFO] val_dataset_size={len(val_dataset)}")

    # Build a global disjoint batch plan once
    if E.disjoint_batches:
        batch_plan = build_disjoint_batch_plan(
            dataset_size=len(train_dataset),
            batch_size=E.batch_size,
            num_batches_total=E.num_batches_total,
            seed=E.seed,
        )
        max_full_batches = len(train_dataset) // E.batch_size
        print(
            f"[INFO] Using DISJOINT batch plan: num_batches_total={E.num_batches_total}, "
            f"batch_size={E.batch_size}, max_full_batches={max_full_batches}"
        )
    else:
        batch_plan = None
        print("[INFO] Using independently sampled random batches (repeats allowed)")

    # Shared val subset for all batches in this shard (default)
    if E.randomize_val_for_utility and not E.per_batch_randomize_val_utility:
        shared_val_inputs = sample_random_val_inputs(
            val_dataset=val_dataset,
            n_val=E.max_val_for_local_utility,
            seed=E.seed + E.val_utility_seed_offset,
            device=device,
        )
        print(f"[INFO] Using ONE shared random val utility subset of size={shared_val_inputs['input_ids'].shape[0]}")
    elif not E.randomize_val_for_utility:
        shared_val_inputs = val_inputs_default
        print(f"[INFO] Using DEFAULT val utility subset of size={shared_val_inputs['input_ids'].shape[0]}")
    else:
        shared_val_inputs = None
        print("[INFO] Using a DIFFERENT random val utility subset for each batch")

    pooled_x: List[float] = []
    pooled_y: List[float] = []
    pooled_corrs: List[float] = []
    pooled_rmses: List[float] = []
    total_start = time.time()

    for global_batch_id in batch_ids:
        batch_seed = E.seed + 100000 + global_batch_id
        mc_seed = E.seed + 200000 + global_batch_id
        duoshap_seed = E.seed + 300000 + global_batch_id

        print("=" * 90)
        print(f"[INFO] Processing global_batch_id={global_batch_id}")

        if E.disjoint_batches:
            batch_indices = batch_plan[global_batch_id]
            batch = stack_examples_from_indices(
                dataset=train_dataset,
                indices=batch_indices,
                device=device,
            )
        else:
            batch_indices = None
            batch = sample_random_batch_from_dataset(
                dataset=train_dataset,
                batch_size=E.batch_size,
                seed=batch_seed,
                device=device,
            )

        B = int(batch["input_ids"].shape[0])
        assert B == E.batch_size

        if shared_val_inputs is not None:
            val_inputs = shared_val_inputs
        else:
            val_inputs = sample_random_val_inputs(
                val_dataset=val_dataset,
                n_val=E.max_val_for_local_utility,
                seed=E.seed + E.val_utility_seed_offset + global_batch_id,
                device=device,
            )

        t0 = time.time()
        U = UtilityCache(
            model=model,
            batch=batch,
            val_inputs=val_inputs,
            sgd_proxy_lr=E.sgd_proxy_lr,
        )
        U_M = U.U(U.full_mask)

        phi_gt = estimate_phi_mc_permutation(U, n_perm=E.mc_num_permutations, seed=mc_seed)
        phi_est = estimate_phi_soda_tabular_antithetic_per_i(U, N_per_i=E.N_per_i, seed=duoshap_seed)

        e = rmse(phi_est, phi_gt)
        corr = safe_corr(phi_est, phi_gt)
        elapsed = time.time() - t0

        print(f"[RESULT] batch={global_batch_id}  RMSE={e:.10g}  Corr={corr:.6f}")
        print(f"[SANITY] sum(phi_gt)={float(phi_gt.sum()):.6g}  sum(phi_est)={float(phi_est.sum()):.6g}  U(M)={float(U_M):.6g}")
        print(f"[INFO] cache_size={len(U.cache)}  elapsed_sec={elapsed:.3f}")

        ids = batch["example_id"].detach().cpu().numpy().astype(int)
        point_rows = []
        for i in range(B):
            point_rows.append({
                "global_batch_id": int(global_batch_id),
                "pos_in_batch": int(i),
                "train_id": int(ids[i]),
                "dataset_index": int(batch_indices[i]) if batch_indices is not None else "",
                "phi_gt_mc": float(phi_gt[i]),
                "phi_est_soda_antithetic": float(phi_est[i]),
                "diff": float(phi_est[i] - phi_gt[i]),
                "batch_seed": int(batch_seed),
                "mc_seed": int(mc_seed),
                "duoshap_seed": int(duoshap_seed),
            })

        batch_row = {
            "global_batch_id": int(global_batch_id),
            "batch_seed": int(batch_seed),
            "mc_seed": int(mc_seed),
            "duoshap_seed": int(duoshap_seed),
            "rmse": float(e),
            "corr": float(corr) if np.isfinite(corr) else "",
            "sum_phi_gt": float(np.sum(phi_gt)),
            "sum_phi_est": float(np.sum(phi_est)),
            "phi_gt_std": float(np.std(phi_gt)),
            "phi_est_std": float(np.std(phi_est)),
            "U_M": float(U_M),
            "base_val_loss": float(U.base_val_loss),
            "utility_cache_size": int(len(U.cache)),
            "elapsed_sec": float(elapsed),
        }

        append_csv_rows(points_path, point_fields, point_rows)
        append_csv_rows(batch_summary_path, batch_fields, [batch_row])

        pooled_x.extend(phi_gt.tolist())
        pooled_y.extend(phi_est.tolist())
        if np.isfinite(corr):
            pooled_corrs.append(float(corr))
        pooled_rmses.append(float(e))

        restore_params(model, U.backup)

    total_elapsed = time.time() - total_start
    restore_dropout(dropout_backup)

    pooled_x_arr = np.asarray(pooled_x, dtype=np.float64)
    pooled_y_arr = np.asarray(pooled_y, dtype=np.float64)

    pooled_rmse = rmse(pooled_y_arr, pooled_x_arr) if len(pooled_x_arr) > 0 else float("nan")
    pooled_corr = safe_corr(pooled_y_arr, pooled_x_arr) if len(pooled_x_arr) > 0 else float("nan")

    shard_meta = {
        "config": asdict(E),
        "assigned_batch_ids": batch_ids,
        "num_batches_processed": len(batch_ids),
        "num_points_processed": int(len(pooled_x_arr)),
        "pooled_rmse": float(pooled_rmse) if np.isfinite(pooled_rmse) else None,
        "pooled_corr": float(pooled_corr) if np.isfinite(pooled_corr) else None,
        "mean_batch_rmse": float(np.mean(pooled_rmses)) if pooled_rmses else None,
        "std_batch_rmse": float(np.std(pooled_rmses)) if pooled_rmses else None,
        "mean_batch_corr": float(np.mean(pooled_corrs)) if pooled_corrs else None,
        "std_batch_corr": float(np.std(pooled_corrs)) if pooled_corrs else None,
        "total_elapsed_sec": float(total_elapsed),
        "device": str(device),
        "disjoint_batches": bool(E.disjoint_batches),
    }
    shard_meta_path.write_text(json.dumps(shard_meta, indent=2))
    print(f"[INFO] Wrote {shard_meta_path}")

    if len(pooled_x_arr) > 0:
        plot_scatter_paper_like(
            x=pooled_x_arr,
            y=pooled_y_arr,
            out_path=shard_dir / f"shard_{E.shard_id:03d}_scatter_full.pdf",
            rmse_val=pooled_rmse,
            corr_val=pooled_corr,
            n_batches=len(batch_ids),
            exp=E.force_sci_exp,
            paper_like_lim=E.paper_like_lim,
            mode="full",
            zoom_quantile=E.zoom_quantile,
        )
        plot_scatter_paper_like(
            x=pooled_x_arr,
            y=pooled_y_arr,
            out_path=shard_dir / f"shard_{E.shard_id:03d}_scatter_zoom.pdf",
            rmse_val=pooled_rmse,
            corr_val=pooled_corr,
            n_batches=len(batch_ids),
            exp=E.force_sci_exp,
            paper_like_lim=E.paper_like_lim,
            mode="zoom",
            zoom_quantile=E.zoom_quantile,
        )
        print(f"[INFO] Wrote shard scatter plots in {shard_dir}")

    print("[DONE] Multi-batch shard finished successfully.")


if __name__ == "__main__":
    main()
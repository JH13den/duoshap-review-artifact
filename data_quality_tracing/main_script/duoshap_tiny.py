#!/usr/bin/env python3
"""
train_tinystories33m_soda_inrun.py

Train TinyStories-33M on a fixed TinyStories subset, and compute In-Run Data Shapley
using the same corrected DuoShap/SODA estimator as your previous pairs.

Utility (per step t, for subset S of the current batch M):
  U^(t)(S) = base_val_loss(w_t) - val_loss(w_t - eta * sum_{z in S} grad loss(z))

Corrected tabular DuoShap / SODA (per example i in the batch):
  Repeat N times:
    - sample k ~ Uniform{1..B-1} (or fixed k via subset_size)
    - sample S of size k that MUST contain i
    - define complement Sc = M \\ S
    - define antithetic group A = Sc ∪ {i}
    - proxy group statistic: g(S) = U(S) - U(Sc)
      and (if antithetic): g(A) = U(A) - U(M\\A)
    - accumulate g(S) and g(A) into phi_i
  phi_i = average over the sampled (1 or 2) groups per repetition

We accumulate per training example_id:
  sum_phi[id] += contributions
  count[id]   += number of contributions
  phi_avg[id] = sum_phi[id] / count[id]

Outputs:
  - training_log.json
  - soda_inrun_shapley_values.csv
  - soda_inrun_shapley_stats.json
"""

from __future__ import annotations

import os

os.environ["HF_HOME"] = "xxxxxxxxx"
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import json
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torch.optim import AdamW

from load_model_tiny import Config as DataConfig, load_tinystories_33m


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

@dataclass
class SODAInRunConfig:
    # Where load_model_data_tinystories33m.py wrote the subsets
    data_output_dir: str = "../output/tinystories33m_setup"
    train_subdir: str = "tinystories_train_blocks"
    val_subdir: str = "tinystories_val_blocks"

    # Use only a fixed subset for experiments
    max_train_examples: int = 10000
    max_val_examples: int = 1000

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-4

    # Virtual update step size for utility U(S)
    sgd_proxy_lr: float = 1e-4

    # ---------- DuoShap / corrected tabular SODA settings ----------
    soda_N_per_i: int = 3

    # subset_size:
    #   0  -> choose k ~ Uniform{1, ..., batch_size-1}
    #   >0 -> fixed subset size k (clamped to [1, batch_size-1])
    subset_size: int = 0

    # Antithetic pairing: A = (M\\S) ∪ {i}
    use_antithetic: bool = True

    # Utility val subset used inside U (fixed once at startup)
    max_val_for_local_utility: int = 64
    randomize_val_for_local_utility: bool = True
    val_utility_seed: int = 2026

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "../output/tinystories33m_soda_run"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def backup_params(model: torch.nn.Module):
    """Clone all parameters at w_t (same device)."""
    return [p.detach().clone() for p in model.parameters()]


def restore_params(model: torch.nn.Module, backup) -> None:
    """Restore parameters from cloned tensors."""
    with torch.no_grad():
        for p, b in zip(model.parameters(), backup):
            p.copy_(b)


# ---------------------------------------------------------------------
# Dataset prep: load fixed blocked TinyStories subsets
# ---------------------------------------------------------------------

def load_blocked_tinystories_subsets(
    cfg: SODAInRunConfig,
    device: torch.device,
):
    root = Path(cfg.data_output_dir)
    train_dir = root / cfg.train_subdir
    val_dir = root / cfg.val_subdir

    print(f"[INFO] Loading train dataset from {train_dir}")
    train_ds = load_from_disk(str(train_dir))

    print(f"[INFO] Loading val dataset   from {val_dir}")
    val_ds = load_from_disk(str(val_dir))

    needed_cols = {"input_ids", "attention_mask", "labels", "example_id"}
    assert needed_cols.issubset(set(train_ds.column_names)), \
        f"Train dataset must contain {needed_cols}, got {train_ds.column_names}"
    assert needed_cols.issubset(set(val_ds.column_names)), \
        f"Val dataset must contain {needed_cols}, got {val_ds.column_names}"

    # Restrict sizes
    if cfg.max_train_examples is not None:
        train_ds = train_ds.select(range(min(cfg.max_train_examples, len(train_ds))))
    if cfg.max_val_examples is not None:
        val_ds = val_ds.select(range(min(cfg.max_val_examples, len(val_ds))))

    print(f"[INFO] Train size: {len(train_ds)}")
    print(f"[INFO] Val size  : {len(val_ds)}")

    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "example_id"],
    )
    val_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "example_id"],
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # Build ONE fixed utility val batch on device
    n_val_util = min(cfg.max_val_for_local_utility, len(val_ds))
    if cfg.randomize_val_for_local_utility:
        rng = np.random.RandomState(cfg.val_utility_seed)
        idx = rng.choice(len(val_ds), size=n_val_util, replace=False).tolist()
    else:
        idx = list(range(n_val_util))

    util_subset = val_ds.select(idx)
    util_loader = DataLoader(util_subset, batch_size=n_val_util, shuffle=False)
    val_all = next(iter(util_loader))

    val_inputs = {
        "input_ids": val_all["input_ids"].to(device),
        "attention_mask": val_all["attention_mask"].to(device),
        "labels": val_all["labels"].to(device),
    }

    return train_loader, val_loader, val_inputs


# ---------------------------------------------------------------------
# Local utility and DuoShap within one SGD step
# ---------------------------------------------------------------------

def compute_val_loss(model, val_inputs) -> float:
    """Validation loss on the fixed utility val subset."""
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=val_inputs["input_ids"],
            attention_mask=val_inputs["attention_mask"],
            labels=val_inputs["labels"],
        )
        return float(out.loss.item())


def simulate_one_subset_update(
    model,
    batch,
    subset_indices: List[int],
    val_inputs,
    base_val_loss: float,
    learning_rate: float,
    backup,
) -> float:
    """
    Deterministic virtual-update utility (positive = improves val):
      U(S) = base_val_loss - val_loss_after_update_on_S

    We compute subset gradients with dropout OFF (eval mode) for stability.
    """
    if subset_indices is None or len(subset_indices) == 0:
        return 0.0

    restore_params(model, backup)
    model.eval()
    model.zero_grad(set_to_none=True)

    input_ids = batch["input_ids"][subset_indices]
    attn_mask = batch["attention_mask"][subset_indices]
    labels = batch["labels"][subset_indices]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=labels,
    )
    loss = outputs.loss
    loss.backward()

    # Virtual SGD step
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.add_(-learning_rate * p.grad)

    new_val_loss = compute_val_loss(model, val_inputs)
    return float(base_val_loss - new_val_loss)


def soda_inrun_for_batch(
    model,
    batch: Dict[str, torch.Tensor],
    val_inputs: Dict[str, torch.Tensor],
    cfg: SODAInRunConfig,
    global_sum_phi: dict,
    global_count: dict,
):
    """
    Corrected tabular DuoShap/SODA per-i (with antithetic), proxy g(S)=U(S)-U(M\\S).

    Updates global_sum_phi/count for the example_id of each i in the batch.
    """
    B = int(batch["input_ids"].shape[0])
    if B <= 1:
        return

    backup = backup_params(model)
    base_val_loss = compute_val_loss(model, val_inputs)

    ids = batch["example_id"]  # tensor [B]
    full_mask = (1 << B) - 1

    cache: Dict[int, float] = {0: 0.0}  # U(empty)=0

    def mask_to_indices(mask: int) -> List[int]:
        return [j for j in range(B) if (mask & (1 << j)) != 0]

    def U_mask(mask: int) -> float:
        mask = int(mask)
        if mask in cache:
            return cache[mask]

        subset_idx = mask_to_indices(mask)
        val = float(
            simulate_one_subset_update(
                model=model,
                batch=batch,
                subset_indices=subset_idx,
                val_inputs=val_inputs,
                base_val_loss=base_val_loss,
                learning_rate=cfg.sgd_proxy_lr,
                backup=backup,
            )
        )
        cache[mask] = val
        return val

    def g(mask_S: int) -> float:
        mask_S = int(mask_S)
        mask_Sc = full_mask & (~mask_S)
        return U_mask(mask_S) - U_mask(mask_Sc)

    rng = np.random.RandomState(cfg.seed + int(ids[0].item()) if B > 0 else cfg.seed)

    for i in range(B):
        bit_i = 1 << i
        train_id_i = int(ids[i].item())

        for _ in range(int(cfg.soda_N_per_i)):
            # choose k
            if cfg.subset_size > 0:
                k = max(1, min(int(cfg.subset_size), B - 1))
            else:
                k = int(rng.randint(1, B))  # Uniform{1..B-1}

            # sample S (size k) containing i
            if k == 1:
                mask_S = bit_i
            else:
                pool = [j for j in range(B) if j != i]
                others = rng.choice(pool, size=k - 1, replace=False).tolist()

                mask_S = bit_i
                for j in others:
                    mask_S |= (1 << int(j))

            mask_Sc = full_mask & (~mask_S)

            if cfg.use_antithetic:
                # A = (M\\S) ∪ {i}
                mask_A = mask_Sc | bit_i

                contrib = g(mask_S) + g(mask_A)
                global_sum_phi[train_id_i] += float(contrib)
                global_count[train_id_i] += 2
            else:
                contrib = g(mask_S)
                global_sum_phi[train_id_i] += float(contrib)
                global_count[train_id_i] += 1

    restore_params(model, backup)


# ---------------------------------------------------------------------
# Main training loop with In-Run DuoShap
# ---------------------------------------------------------------------

def main():
    cfg = SODAInRunConfig()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"[INFO] Using device: {device}")

    # Load model/tokenizer
    data_cfg = DataConfig()
    tokenizer, model = load_tinystories_33m(data_cfg)
    model.to(device)

    # Data
    train_loader, val_loader, val_inputs = load_blocked_tinystories_subsets(cfg, device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)

    # Global Shapley trackers
    global_sum_phi = defaultdict(float)
    global_count = defaultdict(int)

    # Train/val log
    log: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.num_epochs} =====")
        model.train()
        running_train_loss = 0.0
        n_train_steps = 0

        for batch in train_loader:
            batch = {
                k: (v.to(device) if k != "example_id" else v)
                for k, v in batch.items()
            }

            # ---- In-Run DuoShap at w_t ----
            soda_inrun_for_batch(
                model=model,
                batch=batch,
                val_inputs=val_inputs,
                cfg=cfg,
                global_sum_phi=global_sum_phi,
                global_count=global_count,
            )

            # ---- Actual SGD step on full batch ----
            model.train()
            optimizer.zero_grad(set_to_none=True)

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss
            loss.backward()
            optimizer.step()

            running_train_loss += float(loss.item())
            n_train_steps += 1

        avg_train_loss = running_train_loss / max(1, n_train_steps)

        # ----- Validation loss -----
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for vb in val_loader:
                vb = {
                    k: (v.to(device) if k != "example_id" else v)
                    for k, v in vb.items()
                }
                out = model(
                    input_ids=vb["input_ids"],
                    attention_mask=vb["attention_mask"],
                    labels=vb["labels"],
                )
                val_loss_sum += float(out.loss.item())
                val_steps += 1

        avg_val_loss = val_loss_sum / max(1, val_steps)

        print(
            f"[RESULT] Epoch {epoch}: "
            f"train loss = {avg_train_loss:.4f}, val loss = {avg_val_loss:.4f}"
        )
        log.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
        )

    # -----------------------------------------------------------------
    # Post-processing
    # -----------------------------------------------------------------
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))

    # Build final Shapley table
    records = []
    for train_id, sphi in global_sum_phi.items():
        c = global_count[train_id]
        phi_avg = (sphi / c) if c > 0 else 0.0
        records.append(
            {
                "train_id": int(train_id),
                "sum_phi": float(sphi),
                "count": int(c),
                "phi_avg": float(phi_avg),
            }
        )

    records.sort(key=lambda r: r["phi_avg"], reverse=True)

    csv_path = out_dir / "soda_inrun_shapley_values.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["train_id", "sum_phi", "count", "phi_avg"],
        )
        w.writeheader()
        w.writerows(records)

    print(f"[INFO] Wrote Shapley values to {csv_path}")

    # Stats
    if records:
        phi_avgs = np.array([r["phi_avg"] for r in records], dtype=float)
        stats = {
            "n_points": int(len(phi_avgs)),
            "mean": float(phi_avgs.mean()),
            "std": float(phi_avgs.std()),
            "min": float(phi_avgs.min()),
            "max": float(phi_avgs.max()),
            "p10": float(np.percentile(phi_avgs, 10)),
            "p25": float(np.percentile(phi_avgs, 25)),
            "p50": float(np.percentile(phi_avgs, 50)),
            "p75": float(np.percentile(phi_avgs, 75)),
            "p90": float(np.percentile(phi_avgs, 90)),
        }
        (out_dir / "soda_inrun_shapley_stats.json").write_text(
            json.dumps(stats, indent=2)
        )
        print("[INFO] Wrote stats to soda_inrun_shapley_stats.json")
    else:
        print("[WARN] No Shapley records collected.")


if __name__ == "__main__":
    main()
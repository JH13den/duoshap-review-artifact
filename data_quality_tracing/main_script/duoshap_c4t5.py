#!/usr/bin/env python3
"""
train_t5_soda_inrun_logical128_micro32.py

Train t5-small on a fixed C4 subset, and compute In-Run Data Shapley using
a corrected tabular SODA estimator with antithetic pairing and proxy form.

This version keeps the semantic alignment that the user wanted:
- each cooperative game is defined on ONE logical training batch
- the SAME logical batch is then used for the real optimizer update
- the real optimizer update is executed with smaller microbatches for memory safety

Concretely:
- logical_batch_size = number of players in the DuoShap game
- train_microbatch_size = chunk size used during the actual backward pass

So if logical_batch_size=128 and train_microbatch_size=32, each step is:
1) sample one 128-example logical batch
2) run DuoShap on those 128 players at checkpoint w_t
3) restore the model to w_t
4) do the real update on those SAME 128 examples via 4 accumulated microbatches of 32

Outputs:
  - training_log.json
  - soda_inrun_shapley_values.csv
  - soda_inrun_shapley_stats.json
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader

from load_model_data import Config as DataConfig, load_t5_small


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

@dataclass
class SODAInRunConfig:
    # Where load_model_data.py wrote the C4 subsets
    data_output_dir: str = "../output/t5_c4_setup"
    train_subdir: str = "c4_train"
    val_subdir: str = "c4_val"

    # Use only a small subset for initial experiments
    max_train_examples: int = 10000
    max_val_examples: int = 1000

    # Training hyperparameters
    num_epochs: int = 3
    logical_batch_size: int = 128          # number of players per DuoShap game
    train_microbatch_size: int = 32        # memory-safe chunk size for real training update
    val_batch_size: int = 32               # standard validation batch size
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Virtual update step size for utility U(S)
    sgd_proxy_lr: float = 1e-4

    # ---------- Corrected tabular SODA settings ----------
    # N samples per example i in the batch (antithetic -> 2N contributions per i)
    soda_N_per_i: int = 5

    # subset_size:
    #   0  -> choose k ~ Uniform{1, ..., B-1}
    #   >0 -> fixed subset size k (clamped to [1, B-1])
    subset_size: int = 0

    # Antithetic pairing: A = (M\S) ∪ {i}
    use_antithetic: bool = True

    # Utility val subset used inside U (fixed once at startup)
    max_val_for_local_utility: int = 32
    randomize_val_for_local_utility: bool = True
    val_utility_seed: int = 2026

    # -------- T5 denoising (span corruption) task ----------
    max_length: int = 128
    noise_density: float = 0.15
    mean_span_length: float = 3.0

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "../output/t5_c4_soda_run_logical128_micro32"

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


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tensor fields to device, keep example_id as-is if already tensor on CPU."""
    out = {}
    for k, v in batch.items():
        if k == "example_id":
            out[k] = v
        else:
            out[k] = v.to(device)
    return out


def count_valid_label_tokens(labels: torch.Tensor) -> int:
    """Count non-ignored target tokens for stable weighted loss logging."""
    return int((labels != -100).sum().item())


def iter_microbatches(batch: Dict[str, torch.Tensor], microbatch_size: int):
    """Yield contiguous microbatches from one logical batch."""
    B = int(batch["input_ids"].shape[0])
    for start in range(0, B, microbatch_size):
        end = min(start + microbatch_size, B)
        yield {
            "input_ids": batch["input_ids"][start:end],
            "attention_mask": batch["attention_mask"][start:end],
            "labels": batch["labels"][start:end],
            "example_id": batch["example_id"][start:end],
        }


# ---------------------------------------------------------------------
# Dataset prep: load C4 subsets, restrict, tokenize
# ---------------------------------------------------------------------

def t5_span_corrupt_example(
    input_ids: List[int],
    attention_mask: List[int],
    tokenizer,
    rng: np.random.RandomState,
    noise_density: float = 0.15,
    mean_span_length: float = 3.0,
) -> Tuple[List[int], List[int]]:
    """
    Build T5-style span corruption:
      encoder input: tokens with <extra_id_k> replacing masked spans
      decoder labels: <extra_id_k> + span tokens, ..., ending with eos
    Returns: (corrupted_input_ids, labels_ids)
    """
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # Effective length (non-pad)
    L = int(sum(attention_mask))
    if L <= 2:
        labels = [eos_id] if eos_id is not None else [pad_id]
        return list(input_ids), labels

    content = list(input_ids[:L])

    # avoid corrupting the final EOS token if present
    has_eos = (eos_id is not None and len(content) > 0 and content[-1] == eos_id)
    if has_eos:
        content = content[:-1]

    num_noise = int(round(len(content) * noise_density))
    num_noise = max(1, min(num_noise, len(content) - 1))

    num_spans = int(round(num_noise / mean_span_length))
    num_spans = max(1, num_spans)

    noise_span_lengths = rng.multinomial(num_noise - num_spans, [1.0 / num_spans] * num_spans) + 1
    num_nonnoise = len(content) - num_noise
    nonnoise_span_lengths = rng.multinomial(num_nonnoise, [1.0 / (num_spans + 1)] * (num_spans + 1))

    corrupted: List[int] = []
    labels: List[int] = []
    cursor = 0

    for s in range(num_spans):
        keep_len = int(nonnoise_span_lengths[s])
        if keep_len > 0:
            corrupted.extend(content[cursor:cursor + keep_len])
            cursor += keep_len

        sentinel = tokenizer.convert_tokens_to_ids(f"<extra_id_{s}>")
        corrupted.append(sentinel)

        span_len = int(noise_span_lengths[s])
        span_tokens = content[cursor:cursor + span_len]
        cursor += span_len

        labels.append(sentinel)
        labels.extend(span_tokens)

    tail_len = int(nonnoise_span_lengths[-1])
    if tail_len > 0:
        corrupted.extend(content[cursor:cursor + tail_len])
        cursor += tail_len

    if has_eos and eos_id is not None:
        corrupted.append(eos_id)

    if eos_id is not None:
        labels.append(eos_id)

    return corrupted, labels


def load_tokenized_c4_subsets(
    cfg: SODAInRunConfig,
    tokenizer,
    device: torch.device,
):
    root = Path(cfg.data_output_dir)
    train_dir = root / cfg.train_subdir
    val_dir = root / cfg.val_subdir

    print(f"[INFO] Loading train dataset from {train_dir}")
    train_ds = load_from_disk(str(train_dir))
    print(f"[INFO] Loading val dataset   from {val_dir}")
    val_ds = load_from_disk(str(val_dir))

    assert "example_id" in train_ds.column_names, "Train dataset must have 'example_id'."
    assert "text" in train_ds.column_names, "Train dataset must have 'text'."
    assert "text" in val_ds.column_names, "Val dataset must have 'text'."
    assert "example_id" in val_ds.column_names, "Val dataset must have 'example_id'."

    if cfg.max_train_examples is not None:
        train_ds = train_ds.select(range(min(cfg.max_train_examples, len(train_ds))))
    if cfg.max_val_examples is not None:
        val_ds = val_ds.select(range(min(cfg.max_val_examples, len(val_ds))))

    print(f"[INFO] Train raw size: {len(train_ds)}")
    print(f"[INFO] Val raw size  : {len(val_ds)}")

    if len(train_ds) < cfg.logical_batch_size:
        raise ValueError(
            f"Need at least logical_batch_size={cfg.logical_batch_size} training examples, "
            f"but only found {len(train_ds)} after restriction."
        )
    if cfg.train_microbatch_size > cfg.logical_batch_size:
        raise ValueError(
            f"train_microbatch_size={cfg.train_microbatch_size} cannot exceed "
            f"logical_batch_size={cfg.logical_batch_size}."
        )

    max_length = int(cfg.max_length)

    def tokenize_batch_train(batch):
        enc = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        out_input_ids, out_attn, out_labels = [], [], []

        for ex_id, ids, attn in zip(batch["example_id"], enc["input_ids"], enc["attention_mask"]):
            rng = np.random.RandomState(cfg.seed + int(ex_id))
            corrupted, labels = t5_span_corrupt_example(
                input_ids=list(ids),
                attention_mask=list(attn),
                tokenizer=tokenizer,
                rng=rng,
                noise_density=cfg.noise_density,
                mean_span_length=cfg.mean_span_length,
            )

            pad_id = tokenizer.pad_token_id
            corrupted = (corrupted[:max_length] + [pad_id] * max_length)[:max_length]
            labels = (labels[:max_length] + [pad_id] * max_length)[:max_length]

            attn2 = [0 if t == pad_id else 1 for t in corrupted]
            labels = [(t if t != pad_id else -100) for t in labels]

            out_input_ids.append(corrupted)
            out_attn.append(attn2)
            out_labels.append(labels)

        return {
            "input_ids": out_input_ids,
            "attention_mask": out_attn,
            "labels": out_labels,
            "example_id": batch["example_id"],
        }

    def tokenize_batch_val(batch):
        enc = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        out_input_ids, out_attn, out_labels = [], [], []

        for ex_id, ids, attn in zip(batch["example_id"], enc["input_ids"], enc["attention_mask"]):
            rng = np.random.RandomState(cfg.seed + 1_000_000 + int(ex_id))
            corrupted, labels = t5_span_corrupt_example(
                input_ids=list(ids),
                attention_mask=list(attn),
                tokenizer=tokenizer,
                rng=rng,
                noise_density=cfg.noise_density,
                mean_span_length=cfg.mean_span_length,
            )

            pad_id = tokenizer.pad_token_id
            corrupted = (corrupted[:max_length] + [pad_id] * max_length)[:max_length]
            labels = (labels[:max_length] + [pad_id] * max_length)[:max_length]

            attn2 = [0 if t == pad_id else 1 for t in corrupted]
            labels = [(t if t != pad_id else -100) for t in labels]

            out_input_ids.append(corrupted)
            out_attn.append(attn2)
            out_labels.append(labels)

        return {
            "input_ids": out_input_ids,
            "attention_mask": out_attn,
            "labels": out_labels,
            "example_id": batch["example_id"],
        }

    print("[INFO] Tokenizing train dataset...")
    train_tok = train_ds.map(
        tokenize_batch_train,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    print("[INFO] Tokenizing val dataset...")
    val_tok = val_ds.map(
        tokenize_batch_val,
        batched=True,
        remove_columns=val_ds.column_names,
    )

    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "example_id"])
    val_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "example_id"])

    # Important semantic choice:
    # - one logical training batch = one 128-player cooperative game
    # - the SAME 128 examples are used in the real optimizer update afterward
    train_loader = DataLoader(
        train_tok,
        batch_size=cfg.logical_batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_tok,
        batch_size=cfg.val_batch_size,
        shuffle=False,
    )

    # Build ONE fixed utility val batch (optionally randomized) on device
    n_val_util = min(cfg.max_val_for_local_utility, len(val_tok))
    if cfg.randomize_val_for_local_utility:
        rng = np.random.RandomState(cfg.val_utility_seed)
        idx = rng.choice(len(val_tok), size=n_val_util, replace=False).tolist()
    else:
        idx = list(range(n_val_util))

    util_subset = val_tok.select(idx)
    util_loader = DataLoader(util_subset, batch_size=n_val_util, shuffle=False)
    val_all = next(iter(util_loader))

    val_inputs = {
        "input_ids": val_all["input_ids"].to(device),
        "attention_mask": val_all["attention_mask"].to(device),
        "labels": val_all["labels"].to(device),
    }

    return train_loader, val_loader, val_inputs


# ---------------------------------------------------------------------
# Local utility and SODA within one SGD step
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

    outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
    loss = outputs.loss
    loss.backward()

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
    Corrected tabular SODA per-i (with antithetic), proxy g(S)=U(S)-U(M\\S).

    Updates global_sum_phi/count for the example_id of each i in the batch.
    """
    B = int(batch["input_ids"].shape[0])
    if B <= 1:
        return

    backup = backup_params(model)
    base_val_loss = compute_val_loss(model, val_inputs)

    ids = batch["example_id"]
    full_mask = (1 << B) - 1

    cache: Dict[int, float] = {0: 0.0}

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
            if cfg.subset_size > 0:
                k = max(1, min(int(cfg.subset_size), B - 1))
            else:
                k = int(rng.randint(1, B))

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
# Real training step on one logical batch via gradient accumulation
# ---------------------------------------------------------------------

def training_step_with_microbatches(
    model,
    optimizer,
    logical_batch: Dict[str, torch.Tensor],
    microbatch_size: int,
) -> float:
    """
    Perform ONE real optimizer step on the SAME logical batch used by DuoShap,
    but split into smaller microbatches for memory safety.

    Returns a weighted average training loss for logging.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_weighted_loss = 0.0
    total_weight = 0

    microbatches = list(iter_microbatches(logical_batch, microbatch_size))
    num_microbatches = len(microbatches)
    if num_microbatches == 0:
        raise RuntimeError("No microbatches were created from the logical batch.")

    for micro in microbatches:
        out = model(
            input_ids=micro["input_ids"],
            attention_mask=micro["attention_mask"],
            labels=micro["labels"],
        )
        raw_loss = out.loss

        # Scale for gradient accumulation so total gradient matches one logical batch update.
        (raw_loss / num_microbatches).backward()

        weight = count_valid_label_tokens(micro["labels"])
        if weight <= 0:
            weight = int(micro["input_ids"].shape[0])
        total_weighted_loss += float(raw_loss.item()) * weight
        total_weight += weight

    optimizer.step()
    return total_weighted_loss / max(1, total_weight)


# ---------------------------------------------------------------------
# Main training loop with SODA In-Run Data Shapley
# ---------------------------------------------------------------------

def main():
    cfg = SODAInRunConfig()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] logical_batch_size   = {cfg.logical_batch_size}")
    print(f"[INFO] train_microbatch_size = {cfg.train_microbatch_size}")
    print(
        f"[INFO] gradient accumulation microbatches per step = "
        f"{math.ceil(cfg.logical_batch_size / cfg.train_microbatch_size)}"
    )

    data_cfg = DataConfig()
    tokenizer, model = load_t5_small(data_cfg)
    model.to(device)

    train_loader, val_loader, val_inputs = load_tokenized_c4_subsets(cfg, tokenizer, device)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    global_sum_phi = defaultdict(float)
    global_count = defaultdict(int)
    log: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.num_epochs} =====")
        running_train_loss = 0.0
        n_train_steps = 0

        for logical_batch in train_loader:
            logical_batch = move_batch_to_device(logical_batch, device)

            # 1) Estimate in-run DuoShap on the current logical batch at checkpoint w_t.
            soda_inrun_for_batch(
                model=model,
                batch=logical_batch,
                val_inputs=val_inputs,
                cfg=cfg,
                global_sum_phi=global_sum_phi,
                global_count=global_count,
            )

            # 2) Real update on the SAME logical batch via microbatches.
            batch_loss = training_step_with_microbatches(
                model=model,
                optimizer=optimizer,
                logical_batch=logical_batch,
                microbatch_size=cfg.train_microbatch_size,
            )

            running_train_loss += float(batch_loss)
            n_train_steps += 1

        avg_train_loss = running_train_loss / max(1, n_train_steps)

        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for vb in val_loader:
                vb = move_batch_to_device(vb, device)
                out = model(
                    input_ids=vb["input_ids"],
                    attention_mask=vb["attention_mask"],
                    labels=vb["labels"],
                )
                val_loss_sum += float(out.loss.item())
                val_steps += 1
        avg_val_loss = val_loss_sum / max(1, val_steps)

        print(f"[RESULT] Epoch {epoch}: train loss = {avg_train_loss:.4f}, val loss = {avg_val_loss:.4f}")
        log.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))

    records = []
    for train_id, sphi in global_sum_phi.items():
        c = global_count[train_id]
        phi_avg = (sphi / c) if c > 0 else 0.0
        records.append({
            "train_id": int(train_id),
            "sum_phi": float(sphi),
            "count": int(c),
            "phi_avg": float(phi_avg),
        })

    records.sort(key=lambda r: r["phi_avg"], reverse=True)

    csv_path = out_dir / "soda_inrun_shapley_values.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["train_id", "sum_phi", "count", "phi_avg"])
        w.writeheader()
        w.writerows(records)

    print(f"[INFO] Wrote Shapley values to {csv_path}")

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
        (out_dir / "soda_inrun_shapley_stats.json").write_text(json.dumps(stats, indent=2))
        print("[INFO] Wrote stats to soda_inrun_shapley_stats.json")
    else:
        print("[WARN] No Shapley records collected.")


if __name__ == "__main__":
    main()

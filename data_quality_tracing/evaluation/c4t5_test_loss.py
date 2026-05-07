import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import csv
import json
import time
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datasets import load_from_disk
from torch.optim import AdamW

from load_model_data import Config as DataConfig, load_t5_small  # also sets HF caches


# =========================
# CONFIG
# =========================
@dataclass
class Cfg:
    # Data
    data_output_dir: str = "../output/t5_c4_setup"
    train_subdir: str = "c4_train"
    val_subdir: str = "c4_val"

    # Shapley
    shapley_dir: str = "../output/t5_c4_soda_run_update"
    shapley_csv: str = "soda_inrun_shapley_values.csv"

    # Output
    out_dir: str = "../output/eval_exp3_final_2"

    # MUST match Shapley universe
    train_universe: int = 1000
    val_universe: int = 1000  # just to bound loading

   # Subsets (keep 90% for diversity)
    subset_drop_frac: float = 0.10       # drop 10% (=> keep 90%)
    include_keep_phi_ge_0: bool = False  # keep plots to 4 lines only


    # Dev/Test (paper-like: random heldout sets), disjoint, fixed seeds
    dev_size: int = 200
    test_size: int = 500
    split_seed: int = 777  # controls dev/test sampling from val

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0

    # Training budget CONTROL (epoch-based, prevents subset overfitting)
    target_epochs: float = 20.0          # per-run epochs relative to that run’s subset size
    max_steps_cap: int = 900             # safety cap
    eval_every_steps: int = 10

    # Early stopping on DEV
    early_stop_patience_evals: int = 8   # number of eval points without improvement
    early_stop_min_steps: int = 50       # don’t stop too early before model moves

    # Denoising task (must match main Shapley script)
    max_length: int = 128
    noise_density: float = 0.15
    mean_span_length: float = 3.0
    seed: int = 42        # for corruption streams + random subsets
    order_seed: int = 4242  # for same-order global permutation

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Plotting / zoom-in inset
    inset_enabled: bool = True
    inset_loc: str = "upper right"
    inset_width: str = "40%"
    inset_height: str = "42%"
    inset_borderpad: float = 1.1
    inset_x_pad_left: int = 25
    inset_x_pad_right: int = 15
    inset_y_pad_frac: float = 0.15
    inset_y_pad_min: float = 0.002


# =========================
# Repro
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Shapley IO
# =========================
def read_shapley_dict(csv_path: Path) -> Dict[int, float]:
    d = {}
    with csv_path.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            d[int(r["train_id"])] = float(r["phi_avg"])
    if not d:
        raise RuntimeError(f"No shapley rows read from {csv_path}")
    return d


# =========================
# T5 span corruption (same as main)
# =========================
def t5_span_corrupt_example(
    input_ids: List[int],
    attention_mask: List[int],
    tokenizer,
    rng: np.random.RandomState,
    noise_density: float,
    mean_span_length: float,
) -> Tuple[List[int], List[int]]:
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    L = int(sum(attention_mask))
    if L <= 2:
        labels = [eos_id] if eos_id is not None else [pad_id]
        return list(input_ids), labels

    content = list(input_ids[:L])

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

    corrupted, labels = [], []
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


def ensure_example_id(ds):
    if "example_id" in ds.column_names:
        return ds
    return ds.add_column("example_id", list(range(len(ds))))


def tokenize_denoise(ds, tokenizer, cfg: Cfg, stream_offset: int):
    max_length = int(cfg.max_length)
    pad_id = tokenizer.pad_token_id

    def tok(batch):
        enc = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        out_input_ids, out_attn, out_labels, out_eids = [], [], [], []

        for ex_id, ids, attn in zip(batch["example_id"], enc["input_ids"], enc["attention_mask"]):
            ex_id = int(ex_id)
            rng = np.random.RandomState(cfg.seed + stream_offset + ex_id)

            corrupted, labels = t5_span_corrupt_example(
                input_ids=list(ids),
                attention_mask=list(attn),
                tokenizer=tokenizer,
                rng=rng,
                noise_density=cfg.noise_density,
                mean_span_length=cfg.mean_span_length,
            )

            corrupted = (corrupted[:max_length] + [pad_id] * max_length)[:max_length]
            labels = (labels[:max_length] + [pad_id] * max_length)[:max_length]

            attn2 = [0 if t == pad_id else 1 for t in corrupted]
            labels = [(t if t != pad_id else -100) for t in labels]

            out_input_ids.append(corrupted)
            out_attn.append(attn2)
            out_labels.append(labels)
            out_eids.append(ex_id)

        return {
            "input_ids": out_input_ids,
            "attention_mask": out_attn,
            "labels": out_labels,
            "example_id": out_eids,
        }

    tok_ds = ds.map(tok, batched=True, remove_columns=ds.column_names)
    tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "example_id"])
    return tok_ds


# =========================
# Same-order stream (global permutation filtered by subset)
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
# Eval (weighted by batch size)
# =========================
@torch.no_grad()
def eval_loss(model, loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        bs = batch["input_ids"].shape[0]
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
    return int(min(cfg.max_steps_cap, max(1, steps)))


def train_run(
    run_name: str,
    cfg: Cfg,
    train_tok_full,
    keep_mask: np.ndarray,
    dev_tok,
    test_tok,
    init_state_dict_cpu: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    device = torch.device(cfg.device)

    # fresh model with identical init weights
    data_cfg = DataConfig()
    _, model = load_t5_small(data_cfg)
    model.load_state_dict({k: v.to(device) for k, v in init_state_dict_cpu.items()})
    model.to(device)

    dev_loader = torch.utils.data.DataLoader(dev_tok, batch_size=cfg.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_tok, batch_size=cfg.batch_size, shuffle=False)

    opt = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    subset_size = int(keep_mask.sum())
    max_steps = compute_max_steps_for_subset(cfg, subset_size)
    stream = OrderStream(N=len(train_tok_full), keep_mask=keep_mask, order_seed=cfg.order_seed)

    curve = []
    start = time.time()

    # step 0 evals
    dev0 = eval_loss(model, dev_loader, device)
    test0 = eval_loss(model, test_loader, device)
    curve.append({"step": 0, "dev_loss": float(dev0), "test_loss": float(test0), "wall_sec": 0.0})

    best_dev = float(dev0)
    step_at_best = 0
    test_at_best = float(test0)

    # early stop bookkeeping (count evals without improvement)
    bad_evals = 0

    for step in range(1, max_steps + 1):
        idxs = stream.next_batch(cfg.batch_size)
        items = [train_tok_full[i] for i in idxs]
        batch = {
            "input_ids": torch.stack([b["input_ids"] for b in items]).to(device),
            "attention_mask": torch.stack([b["attention_mask"] for b in items]).to(device),
            "labels": torch.stack([b["labels"] for b in items]).to(device),
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
            devv = eval_loss(model, dev_loader, device)
            testv = eval_loss(model, test_loader, device)
            curve.append({
                "step": int(step),
                "dev_loss": float(devv),
                "test_loss": float(testv),
                "wall_sec": float(time.time() - start),
            })

            improved = (devv < best_dev - 1e-6)
            if improved:
                best_dev = float(devv)
                step_at_best = int(step)
                test_at_best = float(testv)
                bad_evals = 0
            else:
                # only start counting patience after min_steps
                if step >= int(cfg.early_stop_min_steps):
                    bad_evals += 1

            if bad_evals >= int(cfg.early_stop_patience_evals):
                break

    # final eval (at last step in curve)
    final_dev = float(curve[-1]["dev_loss"])
    final_test = float(curve[-1]["test_loss"])

    return {
        "run_name": run_name,
        "train_size": int(subset_size),
        "dev_size": int(len(dev_tok)),
        "test_size": int(len(test_tok)),
        "target_epochs": float(cfg.target_epochs),
        "max_steps_used": int(curve[-1]["step"]),
        "eval_every_steps": int(cfg.eval_every_steps),
        "initial_dev_loss": float(curve[0]["dev_loss"]),
        "initial_test_loss": float(curve[0]["test_loss"]),
        "best_dev_loss": float(best_dev),
        "step_at_best_dev": int(step_at_best),
        "test_at_best_dev": float(test_at_best),
        "final_dev_loss": float(final_dev),
        "final_test_loss": float(final_test),
        "elapsed_sec": float(time.time() - start),
        "curve": curve,
    }


# =========================
# IO + Plotting
# =========================
def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def write_summary_csv(path: Path, runs: List[Dict[str, Any]]):
    fields = [
        "run_name","train_size","dev_size","test_size",
        "target_epochs","max_steps_used",
        "initial_test_loss",
        "best_dev_loss","step_at_best_dev","test_at_best_dev",
        "final_test_loss","elapsed_sec"
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in runs:
            w.writerow({k: r[k] for k in fields})


def write_curves_csv(path: Path, runs: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_name","step","dev_loss","test_loss","wall_sec"])
        w.writeheader()
        for r in runs:
            for p in r["curve"]:
                w.writerow({
                    "run_name": r["run_name"],
                    "step": p["step"],
                    "dev_loss": p["dev_loss"],
                    "test_loss": p["test_loss"],
                    "wall_sec": p["wall_sec"],
                })


def pretty_run_label(run_name: str) -> str:
    mapping = {
        "original_100pct": "Original 100%",
        "top_keep90pct": "Top 90%",
        "bottom_keep90pct": "Bottom 90%",
        "random_keep90pct": "Random 90%",
    }
    return mapping.get(run_name, run_name)


def plot_test_loss_vs_step(
    out_path: Path,
    runs: List[Dict[str, Any]],
    title: str,
    logy: bool,
    truncate_at_best_dev: bool,
    cfg: Cfg,
):
    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    plotted_series = []
    for r in runs:
        pts = r["curve"]
        if truncate_at_best_dev:
            best_step = int(r["step_at_best_dev"])
            pts = [p for p in pts if int(p["step"]) <= best_step]

        xs = [int(p["step"]) for p in pts]
        ys = [float(p["test_loss"]) for p in pts]
        if logy:
            ys = [max(y, 1e-10) for y in ys]

        plotted_series.append((r, xs, ys))
        ax.plot(
            xs,
            ys,
            linewidth=1.8,
            label=pretty_run_label(r["run_name"]),
        )
        ax.scatter([r["step_at_best_dev"]], [r["test_at_best_dev"]], s=22, zorder=3)

    ax.set_xlabel("Optimizer steps")
    ax.set_ylabel("Test loss" + (" (log)" if logy else ""))
    ax.set_title(title)

    if logy:
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
    else:
        ax.grid(True, alpha=0.25)

    ax.legend(loc="lower left", fontsize=8)

    # Add a zoomed inset only for linear-scale plots.
    # We avoid mark_inset/connector lines because they made the figure look strange.
    # if cfg.inset_enabled and (not logy) and len(plotted_series) > 0:
    #     best_steps = [int(r["step_at_best_dev"]) for r, _, _ in plotted_series]
    #     zoom_x1 = max(0, min(best_steps) - int(cfg.inset_x_pad_left))
    #     zoom_x2 = max(best_steps) + int(cfg.inset_x_pad_right)

    #     window_ys = []
    #     for _, xs, ys in plotted_series:
    #         for x, y in zip(xs, ys):
    #             if zoom_x1 <= x <= zoom_x2:
    #                 window_ys.append(float(y))

    if cfg.inset_enabled and (not logy) and len(plotted_series) > 0:
        # fixed inset window
        zoom_x1 = 50
        zoom_x2 = 120

        window_ys = []
        for _, xs, ys in plotted_series:
            for x, y in zip(xs, ys):
                if zoom_x1 <= x <= zoom_x2:
                    window_ys.append(float(y))

        if window_ys:
            y_min = min(window_ys)
            y_max = max(window_ys)
            y_span = max(y_max - y_min, float(cfg.inset_y_pad_min))
            y_pad = max(float(cfg.inset_y_pad_min), y_span * float(cfg.inset_y_pad_frac))
            zoom_y1 = y_min - y_pad
            zoom_y2 = y_max + y_pad

            axins = inset_axes(
                ax,
                width=cfg.inset_width,
                height=cfg.inset_height,
                loc=cfg.inset_loc,
                borderpad=cfg.inset_borderpad,
            )

            for _, xs, ys in plotted_series:
                axins.plot(xs, ys, linewidth=1.5)

            axins.set_xlim(zoom_x1, zoom_x2)
            axins.set_ylim(zoom_y1, zoom_y2)
            axins.grid(True, alpha=0.20)
            axins.tick_params(axis="both", labelsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
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

    # Load datasets
    root = Path(cfg.data_output_dir)
    train_ds = ensure_example_id(load_from_disk(str(root / cfg.train_subdir)))
    val_ds   = ensure_example_id(load_from_disk(str(root / cfg.val_subdir)))

    train_ds = train_ds.select(range(min(cfg.train_universe, len(train_ds))))
    val_ds   = val_ds.select(range(min(cfg.val_universe, len(val_ds))))

    train_ids = [int(x) for x in train_ds["example_id"]]

    # Load Shapley and enforce universe alignment
    shapley_path = Path(cfg.shapley_dir) / cfg.shapley_csv
    phi = read_shapley_dict(shapley_path)

    missing = [tid for tid in train_ids if tid not in phi]
    if missing:
        raise RuntimeError(
            f"Shapley CSV missing {len(missing)} / {len(train_ids)} training ids. "
            f"Example missing ids: {missing[:10]}. "
            f"Fix by recomputing Shapley on the same train_universe."
        )

    # Build subsets within this exact universe (keep 90% for diversity)
    rows = [(tid, phi[tid]) for tid in train_ids]
    rows_sorted = sorted(rows, key=lambda x: x[1])  # ascending by phi

    N = len(train_ids)
    if not (0.0 < cfg.subset_drop_frac < 1.0):
        raise ValueError("subset_drop_frac must be in (0, 1).")

    drop_n = int(round(cfg.subset_drop_frac * N))
    drop_n = max(1, min(drop_n, N - 1))   # ensure we keep at least 1 point
    keep_n = N - drop_n                   # e.g., 1000 -> keep 900

    # "top" keeps top 90% (drop most negative 10%)
    top_ids = set(tid for (tid, _) in rows_sorted[drop_n:])

    # "bottom" keeps bottom 90% (drop most positive 10%)
    bottom_ids = set(tid for (tid, _) in rows_sorted[:keep_n])

    # random keeps 90%
    rng = np.random.RandomState(cfg.seed + 999)
    random_ids = set(rng.choice(train_ids, size=keep_n, replace=False).tolist())


    # Optional: keep all phi >= 0
    ge0_ids = set([tid for (tid, ph) in rows if ph >= 0.0])
    ge0_size = int(len(ge0_ids))
    if cfg.include_keep_phi_ge_0 and ge0_size <= 0:
        print("[WARN] keep_phi_ge_0 is empty; skipping it.")
        cfg.include_keep_phi_ge_0 = False

    # Size-matched random control for keep_phi_ge_0
    if cfg.include_keep_phi_ge_0:
        rng2 = np.random.RandomState(cfg.seed + 12345)
        random_ge0match_ids = set(rng2.choice(train_ids, size=ge0_size, replace=False).tolist())

    # Dev/Test split from val (disjoint)
    n_val = len(val_ds)
    need = min(n_val, cfg.dev_size + cfg.test_size)
    rngs = np.random.RandomState(cfg.split_seed)
    chosen = rngs.choice(n_val, size=need, replace=False).tolist()
    dev_idx = chosen[: min(cfg.dev_size, len(chosen))]
    test_idx = chosen[min(cfg.dev_size, len(chosen)) : min(cfg.dev_size + cfg.test_size, len(chosen))]

    dev_ds = val_ds.select(dev_idx)
    test_ds = val_ds.select(test_idx)

    write_json(out_dir / "split_info.json", {
        "split_seed": int(cfg.split_seed),
        "val_universe": int(n_val),
        "dev_size": int(len(dev_idx)),
        "test_size": int(len(test_idx)),
        "dev_indices": dev_idx,
        "test_indices": test_idx,
    })

    # Tokenize once (same tokenizer used for all)
    data_cfg = DataConfig()
    tokenizer, init_model = load_t5_small(data_cfg)

    # Freeze an identical init state dict for all runs (stronger than just set_seed)
    init_state_dict_cpu = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
    del init_model

    train_tok_full = tokenize_denoise(train_ds, tokenizer, cfg, stream_offset=0)
    dev_tok        = tokenize_denoise(dev_ds, tokenizer, cfg, stream_offset=1_000_000)
    test_tok       = tokenize_denoise(test_ds, tokenizer, cfg, stream_offset=2_000_000)

    # Masks over ROW indices 0..train_universe-1
    id_to_row = {tid: i for i, tid in enumerate(train_ids)}
    N = len(train_ids)

    def mask_from_ids(idset: set) -> np.ndarray:
        m = np.zeros(N, dtype=bool)
        for tid in idset:
            m[id_to_row[tid]] = True
        return m

    keep_full   = np.ones(N, dtype=bool)
    keep_top    = mask_from_ids(top_ids)
    keep_bottom = mask_from_ids(bottom_ids)
    keep_rand   = mask_from_ids(random_ids)

    subset_meta = {
    "train_universe": int(N),
    "subset_drop_frac": float(cfg.subset_drop_frac),
    "drop_n": int(drop_n),
    "keep_n": int(keep_n),
    "top_kept": int(keep_top.sum()),
    "bottom_kept": int(keep_bottom.sum()),
    "random_kept": int(keep_rand.sum()),
    "keep_phi_ge_0_size": int(ge0_size),
}

    if cfg.include_keep_phi_ge_0:
        subset_meta["random_size_matched_to_phi_ge_0"] = int(len(random_ge0match_ids))

    write_json(out_dir / "subset_info.json", subset_meta)
    write_json(out_dir / "config.json", asdict(cfg))

    # Run experiments
    runs: List[Dict[str, Any]] = []

    runs.append(train_run("original_100pct", cfg, train_tok_full, keep_full, dev_tok, test_tok, init_state_dict_cpu))
    write_json(out_dir / "run_original_100pct.json", runs[-1])

    runs.append(train_run("top_keep90pct", cfg, train_tok_full, keep_top, dev_tok, test_tok, init_state_dict_cpu))
    write_json(out_dir / "run_top_keep90pct.json", runs[-1])

    runs.append(train_run("bottom_keep90pct", cfg, train_tok_full, keep_bottom, dev_tok, test_tok, init_state_dict_cpu))
    write_json(out_dir / "run_bottom_keep90pct.json", runs[-1])

    runs.append(train_run("random_keep90pct", cfg, train_tok_full, keep_rand, dev_tok, test_tok, init_state_dict_cpu))
    write_json(out_dir / "run_random_keep90pct.json", runs[-1])


    if cfg.include_keep_phi_ge_0:
        keep_ge0 = mask_from_ids(ge0_ids)
        keep_ge0match = mask_from_ids(random_ge0match_ids)

        runs.append(train_run("keep_phi_ge_0", cfg, train_tok_full, keep_ge0, dev_tok, test_tok, init_state_dict_cpu))
        write_json(out_dir / "run_keep_phi_ge_0.json", runs[-1])

        runs.append(train_run("random_size_matched_to_phi_ge_0", cfg, train_tok_full, keep_ge0match, dev_tok, test_tok, init_state_dict_cpu))
        write_json(out_dir / "run_random_size_matched_to_phi_ge_0.json", runs[-1])

    # Save summary/curves
    write_summary_csv(out_dir / "summary.csv", runs)
    write_curves_csv(out_dir / "curves.csv", runs)

    # Plots
    plot_test_loss_vs_step(
        out_dir / "plot_test_loss_step.png",
        runs,
        title="Test loss vs steps",
        logy=False,
        truncate_at_best_dev=False,
        cfg=cfg,
    )
    plot_test_loss_vs_step(
        out_dir / "plot_test_loss_step_trunc_bestdev.png",
        runs,
        title="Test loss vs steps",
        logy=False,
        truncate_at_best_dev=True,
        cfg=cfg,
    )
    plot_test_loss_vs_step(
        out_dir / "plot_test_loss_step_logy.png",
        runs,
        title="Test loss vs steps",
        logy=True,
        truncate_at_best_dev=False,
        cfg=cfg,
    )

    print(f"[DONE] outputs in: {out_dir}")
    print(f"  - {out_dir/'summary.csv'}")
    print(f"  - {out_dir/'plot_test_loss_step_trunc_bestdev.png'}")


if __name__ == "__main__":
    main()






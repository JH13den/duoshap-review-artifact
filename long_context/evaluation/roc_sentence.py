#!/usr/bin/env python3
"""
Plot global sentence-level ROC curves from all finished examples.

Matched to global sentence output structure:

    OUT_DIR / RUN_NAME / TASK / example_XXXX /
        global_sentence_structure.json
        shards/sent_shard_*.csv

Each shard CSV should contain:
    raw_phi
    global_sentence_idx
    unit_k
    is_support_unit
    is_support_sentence
    metric_valid
    text

Default label rule:
    passage_retrieval_en:
        positive = sentences inside gold paragraph
        label column = is_support_unit

    hotpotqa / 2wikimqa:
        positive = sentence directly containing answer string
        label column = is_support_sentence

This script reads all COMPLETE finished examples.
Incomplete examples are skipped, which is useful while jobs are still running.

Clean figure style:
    - title only shows task name
    - legend only shows Random and DuoShap ROC, AUC = ...
    - shaded std region is shown but not included in legend
    - STD_SCALE controls shaded-region width
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    OUT_DIR: Path = Path(
        "xxxxxx"
        "global_sentence_longbench"
    )

    # Change this if your official/flash-attention run saved under a different run name.
    RUN_NAME: str = "official200_fast_flash_all3"

    TASKS: tuple[str, ...] = (
        "passage_retrieval_en",
        "hotpotqa",
        "2wikimqa",
    )

    # None means use all complete finished examples.
    MAX_EXAMPLES_PER_TASK: Optional[int] = None

    # Task-specific label modes.
    TASK_LABEL_MODE: Dict[str, str] = field(
        default_factory=lambda: {
            "passage_retrieval_en": "support_unit",
            "hotpotqa": "support_sentence",
            "2wikimqa": "support_sentence",
        }
    )

    # raw_phi scale can differ by example.
    SCORE_NORMALIZATION: str = "zscore_by_example"  # "none", "zscore_by_example", "minmax_by_example"

    # True: skip examples whose shards do not cover all sentence players.
    # False: use partial rows. For paper results, keep True.
    REQUIRE_COMPLETE_EXAMPLE: bool = True

    OUT_SUBDIR: str = "roc_all_finished_clean"

    FIGSIZE: tuple[float, float] = (8.6, 8.4)
    COMBINED_FIGSIZE: tuple[float, float] = (18.0, 5.4)

    AXIS_LABEL_SIZE: int = 20
    TICK_SIZE: int = 16
    TITLE_SIZE: int = 18
    LEGEND_SIZE: int = 13

    GRID_SIZE: int = 1001

    RANDOM_LINE_WIDTH: float = 1.4
    ROC_LINE_WIDTH: float = 2.6

    # STD_ALPHA controls transparency.
    # STD_SCALE controls width: 1.0 = ±1 std, 0.5 = ±0.5 std, 0.35 = narrower.
    STD_ALPHA: float = 0.12
    STD_SCALE: float = 0.50


# ============================================================
# ARGUMENTS
# ============================================================

def parse_optional_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"none", "null", "-1", ""}:
        return None
    return int(s)


def parse_args() -> Config:
    cfg = Config()

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=cfg.OUT_DIR)
    p.add_argument("--run-name", type=str, default=cfg.RUN_NAME)
    p.add_argument(
        "--tasks",
        type=str,
        default=",".join(cfg.TASKS),
        help="Comma-separated task list.",
    )
    p.add_argument(
        "--max-examples-per-task",
        type=parse_optional_int,
        default=cfg.MAX_EXAMPLES_PER_TASK,
        help="Use None or -1 for all finished examples.",
    )
    p.add_argument(
        "--score-normalization",
        type=str,
        default=cfg.SCORE_NORMALIZATION,
        choices=["none", "zscore_by_example", "minmax_by_example"],
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Use incomplete examples. Not recommended for final paper results.",
    )
    p.add_argument(
        "--passage-label",
        type=str,
        default=cfg.TASK_LABEL_MODE["passage_retrieval_en"],
        choices=[
            "support_unit",
            "support_sentence",
            "paragraph_top1",
            "paragraph_top3",
            "paragraph_top5",
        ],
    )
    p.add_argument(
        "--qa-label",
        type=str,
        default="support_sentence",
        choices=[
            "support_unit",
            "support_sentence",
            "paragraph_top1",
            "paragraph_top3",
            "paragraph_top5",
        ],
    )
    p.add_argument("--out-subdir", type=str, default=cfg.OUT_SUBDIR)
    p.add_argument("--std-alpha", type=float, default=cfg.STD_ALPHA)
    p.add_argument("--std-scale", type=float, default=cfg.STD_SCALE)

    args = p.parse_args()

    cfg.OUT_DIR = args.out_dir
    cfg.RUN_NAME = args.run_name
    cfg.TASKS = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    cfg.MAX_EXAMPLES_PER_TASK = args.max_examples_per_task
    cfg.SCORE_NORMALIZATION = args.score_normalization
    cfg.REQUIRE_COMPLETE_EXAMPLE = not args.allow_partial
    cfg.OUT_SUBDIR = args.out_subdir
    cfg.STD_ALPHA = args.std_alpha
    cfg.STD_SCALE = args.std_scale

    cfg.TASK_LABEL_MODE = {
        "passage_retrieval_en": args.passage_label,
        "hotpotqa": args.qa_label,
        "2wikimqa": args.qa_label,
    }

    return cfg


# ============================================================
# BASIC IO
# ============================================================

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)


def parse_example_index(example_dir: Path) -> int:
    m = re.search(r"example_(\d+)", example_dir.name)
    if not m:
        raise ValueError(f"Cannot parse example index from {example_dir}")
    return int(m.group(1))


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


# ============================================================
# LABEL / SCORE HELPERS
# ============================================================

def label_col_from_mode(label_mode: str) -> str:
    if label_mode == "support_unit":
        return "is_support_unit"

    if label_mode == "support_sentence":
        return "is_support_sentence"

    if label_mode == "paragraph_top1":
        return "is_in_paragraph_level_top1"

    if label_mode == "paragraph_top3":
        return "is_in_paragraph_level_top3"

    if label_mode == "paragraph_top5":
        return "is_in_paragraph_level_top5"

    raise ValueError(f"Unsupported label_mode={label_mode}")


def get_score_from_row(row: Dict[str, str]) -> float:
    if "raw_phi" in row and row["raw_phi"] != "":
        return float(row["raw_phi"])
    if "phi" in row and row["phi"] != "":
        return float(row["phi"])
    raise KeyError("Could not find score column: expected raw_phi or phi")


def normalize_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    scores = scores.astype(float)

    if mode == "none":
        return scores

    if mode == "zscore_by_example":
        mu = float(np.mean(scores))
        sd = float(np.std(scores))
        if sd < 1e-12:
            return np.zeros_like(scores)
        return (scores - mu) / sd

    if mode == "minmax_by_example":
        lo = float(np.min(scores))
        hi = float(np.max(scores))
        if abs(hi - lo) < 1e-12:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    raise ValueError(f"Unsupported normalization mode={mode}")


# ============================================================
# EXAMPLE LOADING
# ============================================================

def find_task_example_dirs(
    run_root: Path,
    task: str,
    max_examples: Optional[int],
) -> List[Path]:
    task_dir = run_root / task

    if not task_dir.exists():
        raise FileNotFoundError(f"Missing task directory: {task_dir}")

    example_dirs = sorted(
        [p for p in task_dir.glob("example_*") if p.is_dir()],
        key=parse_example_index,
    )

    if max_examples is not None:
        example_dirs = example_dirs[:max_examples]

    return example_dirs


def load_one_example(
    *,
    example_dir: Path,
    label_mode: str,
    score_normalization: str,
    require_complete: bool,
) -> Optional[Dict[str, Any]]:
    structure_path = example_dir / "global_sentence_structure.json"
    shards_dir = example_dir / "shards"

    if not structure_path.exists():
        print(f"[SKIP] Missing structure: {structure_path}")
        return None

    if not shards_dir.exists():
        print(f"[SKIP] Missing shards dir: {shards_dir}")
        return None

    structure = read_json(structure_path)

    task = str(structure.get("task", ""))
    example_index = int(structure.get("example_index", parse_example_index(example_dir)))
    metric_valid = bool(structure.get("metric_valid", False))
    metric_reason = str(structure.get("metric_reason", ""))

    num_sentence_players = safe_int(structure.get("num_sentence_players", -1), default=-1)

    if not metric_valid:
        print(
            f"[SKIP] task={task} example={example_index:04d} "
            f"metric_valid=False reason={metric_reason}"
        )
        return None

    label_col = label_col_from_mode(label_mode)
    shard_paths = sorted(shards_dir.glob("sent_shard_*_players_*.csv"))

    if not shard_paths:
        print(f"[SKIP] No shard CSV files found: {shards_dir}")
        return None

    by_idx: Dict[int, Dict[str, Any]] = {}

    for shard_path in shard_paths:
        rows = read_csv_rows(shard_path)

        for row in rows:
            if label_col not in row:
                print(
                    f"[SKIP] Missing label column '{label_col}' in {shard_path}. "
                    f"Available columns: {list(row.keys())}"
                )
                return None

            try:
                idx = int(row["global_sentence_idx"])
                unit_k = int(row["unit_k"])
                raw_phi = get_score_from_row(row)
                label = int(row[label_col])
                metric_valid_row = safe_int(row.get("metric_valid", "1"), default=1)
            except Exception as e:
                raise ValueError(f"Bad row in {shard_path}: {row}") from e

            if metric_valid_row != 1:
                continue

            if idx in by_idx:
                old_phi = float(by_idx[idx]["raw_phi"])
                if abs(old_phi - raw_phi) > 1e-9:
                    print(
                        f"[WARN] Duplicate sentence idx with different phi: "
                        f"{example_dir.name} idx={idx} old={old_phi} new={raw_phi}"
                    )
                continue

            by_idx[idx] = {
                "task": task,
                "example_index": example_index,
                "global_sentence_idx": idx,
                "unit_k": unit_k,
                "raw_phi": raw_phi,
                "label": label,
                "label_mode": label_mode,
                "text": row.get("text", ""),
            }

    if not by_idx:
        print(f"[SKIP] No usable sentence rows: {example_dir}")
        return None

    if num_sentence_players <= 0:
        num_sentence_players = max(by_idx.keys()) + 1

    present = set(by_idx.keys())
    expected = set(range(num_sentence_players))
    missing = sorted(expected - present)
    extra = sorted(present - expected)

    if missing:
        msg = (
            f"[INCOMPLETE] task={task} example={example_index:04d}: "
            f"loaded={len(present)} expected={num_sentence_players} "
            f"missing_count={len(missing)}"
        )

        if require_complete:
            print(msg + " -> skipping")
            return None

        print(msg + " -> using available rows")

    if extra:
        print(
            f"[WARN] task={task} example={example_index:04d}: "
            f"extra sentence ids outside expected range: {extra[:10]}"
        )

    valid_indices = sorted(i for i in by_idx.keys() if 0 <= i < num_sentence_players)
    rows_sorted = [by_idx[i] for i in valid_indices]

    scores_raw = np.array([r["raw_phi"] for r in rows_sorted], dtype=float)
    labels = np.array([r["label"] for r in rows_sorted], dtype=np.int64)

    num_total = int(len(labels))
    num_pos = int(labels.sum())
    num_neg = num_total - num_pos

    if num_total == 0:
        print(f"[SKIP] Empty example: task={task} example={example_index:04d}")
        return None

    if num_pos == 0:
        print(
            f"[SKIP] No positives under label_mode={label_mode}: "
            f"task={task} example={example_index:04d}"
        )
        return None

    if num_neg == 0:
        print(
            f"[SKIP] No negatives under label_mode={label_mode}: "
            f"task={task} example={example_index:04d}"
        )
        return None

    scores_norm = normalize_scores(scores_raw, score_normalization)

    for r, score in zip(rows_sorted, scores_norm):
        r["score"] = float(score)

    return {
        "task": task,
        "example_index": example_index,
        "example_dir": str(example_dir),
        "label_mode": label_mode,
        "num_sentence_players": num_sentence_players,
        "num_loaded_sentences": len(rows_sorted),
        "num_positive": num_pos,
        "num_negative": num_neg,
        "metric_reason": metric_reason,
        "support_unit_ids_1based": structure.get("support_unit_ids_1based", []),
        "support_sentence_indices": structure.get("support_sentence_indices", []),
        "rows": rows_sorted,
        "scores": scores_norm,
        "labels": labels,
    }


def collect_task_examples(cfg: Config, task: str) -> List[Dict[str, Any]]:
    run_root = cfg.OUT_DIR / cfg.RUN_NAME
    label_mode = cfg.TASK_LABEL_MODE.get(task, "support_sentence")

    example_dirs = find_task_example_dirs(
        run_root=run_root,
        task=task,
        max_examples=cfg.MAX_EXAMPLES_PER_TASK,
    )

    loaded: List[Dict[str, Any]] = []

    print("=" * 100)
    print(f"[TASK] {task}")
    print(f"[LABEL] {label_mode}")
    print(f"[INFO] Candidate example dirs: {len(example_dirs)}")
    print("=" * 100)

    for ex_dir in example_dirs:
        try:
            ex = load_one_example(
                example_dir=ex_dir,
                label_mode=label_mode,
                score_normalization=cfg.SCORE_NORMALIZATION,
                require_complete=cfg.REQUIRE_COMPLETE_EXAMPLE,
            )
        except Exception as e:
            print(f"[SKIP_ERROR] {ex_dir}: {type(e).__name__}: {e}")
            continue

        if ex is not None:
            loaded.append(ex)
            print(
                f"[OK] example={ex['example_index']:04d} "
                f"sentences={ex['num_loaded_sentences']} "
                f"pos={ex['num_positive']} neg={ex['num_negative']}"
            )

    if not loaded:
        raise ValueError(f"No valid complete examples loaded for task={task}")

    return loaded


# ============================================================
# ROC COMPUTATION
# ============================================================

def roc_from_scores(
    scores: np.ndarray,
    labels: np.ndarray,
) -> List[Dict[str, Any]]:
    labels = labels.astype(np.int64)
    scores = scores.astype(float)

    P = int(labels.sum())
    N = int(len(labels) - P)

    if P == 0 or N == 0:
        raise ValueError(f"ROC requires both positives and negatives, got P={P}, N={N}")

    order = np.lexsort((np.arange(len(scores)), -scores))
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    roc_rows: List[Dict[str, Any]] = []

    tp = 0
    fp = 0

    roc_rows.append(
        {
            "rank_cutoff": 0,
            "threshold_score": math.inf,
            "tpr": 0.0,
            "fpr": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": P,
            "tn": N,
        }
    )

    for k, y in enumerate(sorted_labels, start=1):
        if y == 1:
            tp += 1
        else:
            fp += 1

        fn = P - tp
        tn = N - fp

        roc_rows.append(
            {
                "rank_cutoff": k,
                "threshold_score": float(sorted_scores[k - 1]),
                "tpr": float(tp / P),
                "fpr": float(fp / N),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )

    return roc_rows


def auc_trapezoid(roc_rows: List[Dict[str, Any]]) -> float:
    fpr = np.array([r["fpr"] for r in roc_rows], dtype=float)
    tpr = np.array([r["tpr"] for r in roc_rows], dtype=float)

    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def auc_pairwise_tie_aware(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.astype(float)
    labels = labels.astype(np.int64)

    pos = scores[labels == 1]
    neg = scores[labels == 0]

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    wins = 0.0

    for ps in pos:
        wins += np.sum(ps > neg)
        wins += 0.5 * np.sum(ps == neg)

    return float(wins / (len(pos) * len(neg)))


def tpr_on_grid(
    roc_rows: List[Dict[str, Any]],
    fpr_grid: np.ndarray,
) -> np.ndarray:
    fpr = np.array([r["fpr"] for r in roc_rows], dtype=float)
    tpr = np.array([r["tpr"] for r in roc_rows], dtype=float)

    order = np.lexsort((tpr, fpr))
    fpr = fpr[order]
    tpr = tpr[order]

    unique_fpr = []
    max_tpr = []

    for val in np.unique(fpr):
        mask = fpr == val
        unique_fpr.append(float(val))
        max_tpr.append(float(np.max(tpr[mask])))

    unique_fpr = np.array(unique_fpr, dtype=float)
    max_tpr = np.array(max_tpr, dtype=float)

    interp = np.interp(fpr_grid, unique_fpr, max_tpr)
    interp[0] = 0.0
    interp[-1] = 1.0

    return interp


# ============================================================
# TASK ROC
# ============================================================

def flatten_sentence_rows(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for ex in examples:
        for r in ex["rows"]:
            out.append(
                {
                    "task": ex["task"],
                    "example_index": ex["example_index"],
                    "global_sentence_idx": r["global_sentence_idx"],
                    "unit_k": r["unit_k"],
                    "raw_phi": r["raw_phi"],
                    "score_normalized": r["score"],
                    "label": r["label"],
                    "label_mode": ex["label_mode"],
                    "text": r.get("text", ""),
                }
            )

    return out


def build_task_roc(
    examples: List[Dict[str, Any]],
    cfg: Config,
) -> Dict[str, Any]:
    all_scores = np.concatenate([ex["scores"] for ex in examples], axis=0)
    all_labels = np.concatenate([ex["labels"] for ex in examples], axis=0)

    micro_roc = roc_from_scores(all_scores, all_labels)
    micro_auc = auc_trapezoid(micro_roc)
    micro_auc_pairwise = auc_pairwise_tie_aware(all_scores, all_labels)

    fpr_grid = np.linspace(0.0, 1.0, cfg.GRID_SIZE)

    example_summaries: List[Dict[str, Any]] = []
    interp_tprs = []

    for ex in examples:
        ex_roc = roc_from_scores(ex["scores"], ex["labels"])
        ex_auc = auc_trapezoid(ex_roc)
        ex_auc_pairwise = auc_pairwise_tie_aware(ex["scores"], ex["labels"])
        interp = tpr_on_grid(ex_roc, fpr_grid)
        interp_tprs.append(interp)

        example_summaries.append(
            {
                "task": ex["task"],
                "example_index": ex["example_index"],
                "label_mode": ex["label_mode"],
                "num_sentences": ex["num_loaded_sentences"],
                "num_positive": ex["num_positive"],
                "num_negative": ex["num_negative"],
                "auc_trapezoid": ex_auc,
                "auc_pairwise_tie_aware": ex_auc_pairwise,
                "metric_reason": ex["metric_reason"],
                "support_unit_ids_1based": ex["support_unit_ids_1based"],
                "support_sentence_count": len(ex["support_sentence_indices"]),
                "example_dir": ex["example_dir"],
            }
        )

    tpr_matrix = np.stack(interp_tprs, axis=0)
    macro_mean_tpr = np.mean(tpr_matrix, axis=0)
    macro_std_tpr = np.std(tpr_matrix, axis=0)

    macro_auc = float(np.trapz(macro_mean_tpr, fpr_grid))
    mean_example_auc = float(np.mean([r["auc_trapezoid"] for r in example_summaries]))
    std_example_auc = float(np.std([r["auc_trapezoid"] for r in example_summaries]))

    macro_rows = []

    for fpr, mean_tpr, std_tpr in zip(fpr_grid, macro_mean_tpr, macro_std_tpr):
        scaled_std = cfg.STD_SCALE * std_tpr

        macro_rows.append(
            {
                "fpr": float(fpr),
                "mean_tpr": float(mean_tpr),
                "std_tpr": float(std_tpr),
                "std_scale": float(cfg.STD_SCALE),
                "tpr_lower": float(max(0.0, mean_tpr - scaled_std)),
                "tpr_upper": float(min(1.0, mean_tpr + scaled_std)),
            }
        )

    task = examples[0]["task"]
    label_mode = examples[0]["label_mode"]

    return {
        "task": task,
        "label_mode": label_mode,
        "num_examples": len(examples),
        "num_total_sentences": int(len(all_labels)),
        "num_total_positive": int(all_labels.sum()),
        "num_total_negative": int(len(all_labels) - all_labels.sum()),
        "micro_roc": micro_roc,
        "micro_auc": micro_auc,
        "micro_auc_pairwise_tie_aware": micro_auc_pairwise,
        "macro_rows": macro_rows,
        "macro_auc": macro_auc,
        "mean_example_auc": mean_example_auc,
        "std_example_auc": std_example_auc,
        "example_summaries": example_summaries,
        "sentence_rows": flatten_sentence_rows(examples),
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_one_task(
    task_result: Dict[str, Any],
    out_base: Path,
    cfg: Config,
) -> None:
    macro = task_result["macro_rows"]

    fpr = np.array([r["fpr"] for r in macro], dtype=float)
    mean_tpr = np.array([r["mean_tpr"] for r in macro], dtype=float)
    lower = np.array([r["tpr_lower"] for r in macro], dtype=float)
    upper = np.array([r["tpr_upper"] for r in macro], dtype=float)

    task = task_result["task"]

    plt.figure(figsize=cfg.FIGSIZE)

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=cfg.RANDOM_LINE_WIDTH,
        color="gray",
        label="Random",
        zorder=1,
    )

    plt.fill_between(
        fpr,
        lower,
        upper,
        alpha=cfg.STD_ALPHA,
        label="_nolegend_",
        zorder=2,
    )

    plt.plot(
        fpr,
        mean_tpr,
        linewidth=cfg.ROC_LINE_WIDTH,
        label=f"DuoShap ROC, AUC = {task_result['macro_auc']:.3f}",
        zorder=3,
    )

    plt.xlabel("False Positive Rate", fontsize=cfg.AXIS_LABEL_SIZE)
    plt.ylabel("True Positive Rate", fontsize=cfg.AXIS_LABEL_SIZE)

    # Only task name in title
    plt.title(task, fontsize=cfg.TITLE_SIZE)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)

    plt.xticks(fontsize=cfg.TICK_SIZE)
    plt.yticks(fontsize=cfg.TICK_SIZE)

    plt.legend(
        fontsize=cfg.LEGEND_SIZE,
        loc="lower right",
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.savefig(out_base.with_suffix(".pdf"))
    plt.close()


def plot_combined(
    all_task_results: List[Dict[str, Any]],
    out_base: Path,
    cfg: Config,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(all_task_results),
        figsize=cfg.COMBINED_FIGSIZE,
        sharex=True,
        sharey=True,
    )

    if len(all_task_results) == 1:
        axes = [axes]

    for ax, task_result in zip(axes, all_task_results):
        macro = task_result["macro_rows"]

        fpr = np.array([r["fpr"] for r in macro], dtype=float)
        mean_tpr = np.array([r["mean_tpr"] for r in macro], dtype=float)
        lower = np.array([r["tpr_lower"] for r in macro], dtype=float)
        upper = np.array([r["tpr_upper"] for r in macro], dtype=float)

        task = task_result["task"]

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.2,
            color="gray",
            label="Random",
            zorder=1,
        )

        ax.fill_between(
            fpr,
            lower,
            upper,
            alpha=cfg.STD_ALPHA,
            label="_nolegend_",
            zorder=2,
        )

        ax.plot(
            fpr,
            mean_tpr,
            linewidth=2.4,
            label=f"DuoShap ROC, AUC = {task_result['macro_auc']:.3f}",
            zorder=3,
        )

        # Only task name in title
        ax.set_title(task, fontsize=cfg.TITLE_SIZE)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.tick_params(axis="both", labelsize=cfg.TICK_SIZE)
        ax.legend(fontsize=cfg.LEGEND_SIZE, loc="lower right", frameon=False)

    axes[0].set_ylabel("True Positive Rate", fontsize=cfg.AXIS_LABEL_SIZE)

    for ax in axes:
        ax.set_xlabel("False Positive Rate", fontsize=cfg.AXIS_LABEL_SIZE)

    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    cfg = parse_args()

    out_dir = cfg.OUT_DIR / cfg.RUN_NAME / cfg.OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    summary: Dict[str, Any] = {
        "config": asdict(cfg),
        "tasks": {},
    }

    print("=" * 100)
    print("[START] Plot ROC for all finished global sentence examples")
    print(f"[RUN_ROOT] {cfg.OUT_DIR / cfg.RUN_NAME}")
    print(f"[OUT_DIR]  {out_dir}")
    print(f"[TASKS]    {cfg.TASKS}")
    print(f"[COMPLETE] require_complete={cfg.REQUIRE_COMPLETE_EXAMPLE}")
    print(f"[STD] alpha={cfg.STD_ALPHA}, scale={cfg.STD_SCALE}")
    print("=" * 100)

    for task in cfg.TASKS:
        examples = collect_task_examples(cfg, task)
        task_result = build_task_roc(examples, cfg)
        all_results.append(task_result)

        task_base = out_dir / f"{task}_roc"

        plot_one_task(
            task_result=task_result,
            out_base=task_base,
            cfg=cfg,
        )

        write_csv(
            out_dir / f"{task}_macro_roc_points.csv",
            task_result["macro_rows"],
            fieldnames=[
                "fpr",
                "mean_tpr",
                "std_tpr",
                "std_scale",
                "tpr_lower",
                "tpr_upper",
            ],
        )

        write_csv(
            out_dir / f"{task}_micro_roc_points.csv",
            task_result["micro_roc"],
            fieldnames=[
                "rank_cutoff",
                "threshold_score",
                "tpr",
                "fpr",
                "tp",
                "fp",
                "fn",
                "tn",
            ],
        )

        write_csv(
            out_dir / f"{task}_example_summary.csv",
            task_result["example_summaries"],
            fieldnames=[
                "task",
                "example_index",
                "label_mode",
                "num_sentences",
                "num_positive",
                "num_negative",
                "auc_trapezoid",
                "auc_pairwise_tie_aware",
                "metric_reason",
                "support_unit_ids_1based",
                "support_sentence_count",
                "example_dir",
            ],
        )

        write_csv(
            out_dir / f"{task}_sentence_scores_used.csv",
            task_result["sentence_rows"],
            fieldnames=[
                "task",
                "example_index",
                "global_sentence_idx",
                "unit_k",
                "raw_phi",
                "score_normalized",
                "label",
                "label_mode",
                "text",
            ],
        )

        summary["tasks"][task] = {
            "label_mode": task_result["label_mode"],
            "num_examples": task_result["num_examples"],
            "num_total_sentences": task_result["num_total_sentences"],
            "num_total_positive": task_result["num_total_positive"],
            "num_total_negative": task_result["num_total_negative"],
            "macro_auc": task_result["macro_auc"],
            "mean_example_auc": task_result["mean_example_auc"],
            "std_example_auc": task_result["std_example_auc"],
            "micro_auc": task_result["micro_auc"],
            "micro_auc_pairwise_tie_aware": task_result["micro_auc_pairwise_tie_aware"],
        }

        print("=" * 100)
        print(f"[DONE] {task}")
        print(f"[LABEL] {task_result['label_mode']}")
        print(f"[INFO] examples:             {task_result['num_examples']}")
        print(f"[INFO] total sentences:      {task_result['num_total_sentences']}")
        print(f"[INFO] positives:            {task_result['num_total_positive']}")
        print(f"[INFO] negatives:            {task_result['num_total_negative']}")
        print(f"[INFO] macro AUC:            {task_result['macro_auc']:.4f}")
        print(f"[INFO] mean example AUC:     {task_result['mean_example_auc']:.4f}")
        print(f"[INFO] micro pooled AUC:     {task_result['micro_auc']:.4f}")
        print(f"[SAVE] {task_base.with_suffix('.png')}")
        print(f"[SAVE] {task_base.with_suffix('.pdf')}")
        print("=" * 100)

    combined_base = out_dir / "combined_roc_all_tasks"

    plot_combined(
        all_task_results=all_results,
        out_base=combined_base,
        cfg=cfg,
    )

    write_json(out_dir / "summary.json", summary)

    print("=" * 100)
    print("[DONE] All ROC curves generated")
    print(f"[OUT_DIR] {out_dir}")
    print(f"[SAVE] {combined_base.with_suffix('.png')}")
    print(f"[SAVE] {combined_base.with_suffix('.pdf')}")
    print(f"[SAVE] {out_dir / 'summary.json'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
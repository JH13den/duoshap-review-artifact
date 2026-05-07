#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ABLATION_ROOT = Path(
    "xxxxxx"
)

RUN_NAME = "paragraph_ablation_qwen25_7b_official"

TASKS = [
    "passage_retrieval_en",
    "hotpotqa",
    "2wikimqa",
]

TASK_DISPLAY = {
    "passage_retrieval_en": "Passage Retrieval",
    "hotpotqa": "HotpotQA",
    "2wikimqa": "2WikiMQA",
}

OUT_DIR = ABLATION_ROOT / RUN_NAME / "final_pubstyle_ablation_plots"

# ============================================================
# X-axis range
# ============================================================

TASK_XMIN = {
    "passage_retrieval_en": 0.0,
    "hotpotqa": 0.0,
    "2wikimqa": 0.0,
}

TASK_XMAX = {
    "passage_retrieval_en": 60.0,
    "hotpotqa": 90.0,
    "2wikimqa": 90.0,
}

# Exclude 100% because it is the trivial empty-context endpoint.
INCLUDE_100_PERCENT = False

# Padding so endpoint markers are fully visible.
TASK_XPAD = {
    "passage_retrieval_en": 1.5,
    "hotpotqa": 2.5,
    "2wikimqa": 2.5,
}


# ============================================================
# Filtering
# ============================================================

FILTER_METRIC_VALID = True
FILTER_NORMALIZATION_VALID = True


# ============================================================
# Summary and band style
# ============================================================

LINE_STAT = "median"       # "median" or "mean"
ERROR_STAT = "sem"         # "sem", "std", or "iqr"

# Controls shaded band size.
# Increase to 0.30 if too small; decrease to 0.15 if too large.
ERROR_SCALE = 0.20

# Controls shaded band darkness.
# Increase to 0.22 if too light; decrease to 0.12 if too dark.
BAND_ALPHA = 0.20


# ============================================================
# Plot style
# ============================================================

DPI = 300

FIGSIZE_SINGLE = (5.8, 4.1)
FIGSIZE_PANEL = (15.0, 4.2)

LINEWIDTH = 2.2
MARKERSIZE = 6

STYLE = {
    "top": {
        "label": "High DuoShap",
        "linestyle": "-",
        "marker": "o",
    },
    "random": {
        "label": "Random",
        "linestyle": "--",
        "marker": "s",
    },
    "bottom": {
        "label": "Low DuoShap",
        "linestyle": "-.",
        "marker": "^",
    },
}

TASK_YLIM_VALUE = {
    "passage_retrieval_en": (-0.9, 1.25),
    "hotpotqa": (-0.45, 1.25),
    "2wikimqa": (-0.55, 1.65),
}

TASK_YTICKS = {
    "passage_retrieval_en": [-0.8, -0.4, 0.0, 0.5, 1.0],
    "hotpotqa": [-0.4, 0.0, 0.5, 1.0],
    "2wikimqa": [-0.5, 0.0, 0.5, 1.0, 1.5],
}

TASK_LEGEND_LOC = {
    "passage_retrieval_en": "lower left",
    "hotpotqa": "lower left",
    "2wikimqa": "lower left",
}

SHOW_REFERENCE_LINES = True


# ============================================================
# BASIC HELPERS
# ============================================================

def parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def safe_float(x) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return np.nan
        return v
    except Exception:
        return np.nan


def safe_int(x) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Could not parse {path}:{line_no}: {e}")

    return rows


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        keys = set()
        for r in rows:
            keys.update(r.keys())
        fieldnames = sorted(keys)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# COLLECT OFFICIAL RESULTS
# ============================================================

def flatten_example_result(r: dict) -> dict:
    return {
        "task": r.get("task"),
        "example_index": r.get("example_index"),
        "status": r.get("status"),
        "metric_type": r.get("metric_type"),
        "metric_valid": r.get("metric_valid"),
        "metric_reason": r.get("metric_reason"),
        "normalization_valid": r.get("normalization_valid"),
        "num_players": r.get("num_players"),
        "support_ids_1based": json.dumps(r.get("support_ids_1based", []), ensure_ascii=False),
        "top_player_1based": r.get("top_player_1based"),
        "bottom_player_1based": r.get("bottom_player_1based"),
        "top1_hit": r.get("top1_hit"),
        "f_empty": r.get("f_empty"),
        "f_full": r.get("f_full"),
        "f_full_minus_empty": r.get("f_full_minus_empty"),
        "num_curve_points": r.get("num_curve_points"),
        "random_repeats": r.get("random_repeats"),
        "runtime_sec": r.get("runtime_sec"),
        "value_cache_size": r.get("value_cache_size"),
        "shapley_csv": r.get("shapley_csv"),
        "ablation_curves_csv": r.get("ablation_curves_csv"),
        "random_repeats_csv": r.get("random_repeats_csv"),
        "ranking_csv": r.get("ranking_csv"),
        "_source_shard": r.get("_source_shard"),
        "_result_path": r.get("_result_path"),
        "_curves_path": r.get("_curves_path"),
    }


def clean_curve_row(row: dict) -> dict:
    normalized_top = safe_float(row.get("normalized_remove_top"))
    normalized_bottom = safe_float(row.get("normalized_remove_bottom"))
    normalized_random = safe_float(row.get("normalized_remove_random_mean"))

    return {
        "task": row.get("task"),
        "example_index": safe_int(row.get("example_index")),
        "percent_removed": safe_float(row.get("percent_removed")),
        "k_removed": safe_int(row.get("k_removed")),
        "num_players": safe_int(row.get("num_players")),
        "num_players_example": safe_int(row.get("num_players_example")),

        "metric_valid": parse_bool(row.get("metric_valid")),
        "normalization_valid": parse_bool(row.get("normalization_valid_example")),

        "f_empty": safe_float(row.get("f_empty")),
        "f_full": safe_float(row.get("f_full")),
        "f_full_minus_empty": safe_float(row.get("f_full_minus_empty")),

        "f_remove_top": safe_float(row.get("f_remove_top")),
        "f_remove_bottom": safe_float(row.get("f_remove_bottom")),
        "f_remove_random_mean": safe_float(row.get("f_remove_random_mean")),

        "delta_remove_top": safe_float(row.get("delta_remove_top")),
        "delta_remove_bottom": safe_float(row.get("delta_remove_bottom")),
        "delta_remove_random_mean": safe_float(row.get("delta_remove_random_mean")),

        "normalized_remove_top": normalized_top,
        "normalized_remove_bottom": normalized_bottom,
        "normalized_remove_random_mean": normalized_random,

        "normalized_drop_top": 1.0 - normalized_top if not np.isnan(normalized_top) else np.nan,
        "normalized_drop_bottom": 1.0 - normalized_bottom if not np.isnan(normalized_bottom) else np.nan,
        "normalized_drop_random": 1.0 - normalized_random if not np.isnan(normalized_random) else np.nan,

        "_source_shard": row.get("_source_shard"),
        "_source_file": row.get("_source_file"),
    }


def collect_task_outputs(task: str):
    task_dir = ABLATION_ROOT / RUN_NAME / task

    all_curve_rows = []
    all_example_rows = []
    all_error_rows = []

    if not task_dir.exists():
        print(f"[WARN] Missing task dir: {task_dir}")
        return all_curve_rows, all_example_rows, all_error_rows

    shard_dirs = sorted(task_dir.glob("shard_*_of_*"))

    print("\n" + "=" * 100)
    print(f"[TASK] {task}")
    print(f"[DIR]  {task_dir}")
    print(f"[INFO] Found {len(shard_dirs)} shard dirs")

    for shard_dir in shard_dirs:
        results_jsonl = shard_dir / "results.jsonl"
        errors_jsonl = shard_dir / "errors.jsonl"

        result_rows = read_jsonl(results_jsonl)
        error_rows = read_jsonl(errors_jsonl)

        print(
            f"  {shard_dir.name}: "
            f"results={len(result_rows)}, errors={len(error_rows)}"
        )

        for err in error_rows:
            err["_source_shard"] = shard_dir.name
            err["_source_file"] = str(errors_jsonl)
            all_error_rows.append(err)

        examples_dir = shard_dir / "examples"
        if not examples_dir.exists():
            continue

        for example_dir in sorted(examples_dir.glob("example_*")):
            result_path = example_dir / "ablation_result.json"
            curves_path = example_dir / "ablation_curves.csv"

            if not result_path.exists() or not curves_path.exists():
                continue

            try:
                result = read_json(result_path)
            except Exception as e:
                print(f"[WARN] Could not read {result_path}: {e}")
                continue

            result["_source_shard"] = shard_dir.name
            result["_result_path"] = str(result_path)
            result["_curves_path"] = str(curves_path)
            all_example_rows.append(flatten_example_result(result))

            if result.get("status") != "ok":
                continue

            try:
                curves = read_csv(curves_path)
            except Exception as e:
                print(f"[WARN] Could not read {curves_path}: {e}")
                continue

            example_index = int(result["example_index"])

            for row in curves:
                row["_source_shard"] = shard_dir.name
                row["_source_file"] = str(curves_path)
                row["task"] = task
                row["example_index"] = example_index
                row["metric_valid"] = parse_bool(result.get("metric_valid", False))
                row["normalization_valid_example"] = parse_bool(
                    result.get("normalization_valid", False)
                )
                row["num_players_example"] = result.get("num_players", "")

                all_curve_rows.append(clean_curve_row(row))

    return all_curve_rows, all_example_rows, all_error_rows


# ============================================================
# FILTER + SUMMARY
# ============================================================

def include_curve_row(row: dict) -> bool:
    if FILTER_METRIC_VALID and not row.get("metric_valid", False):
        return False

    if FILTER_NORMALIZATION_VALID and not row.get("normalization_valid", False):
        return False

    p = row.get("percent_removed")
    if p is None or np.isnan(p):
        return False

    task = row["task"]

    if p < TASK_XMIN[task]:
        return False

    if p > TASK_XMAX[task]:
        return False

    if not INCLUDE_100_PERCENT and p >= 100.0:
        return False

    return True


def summarize_curves(curve_rows: List[dict]) -> List[dict]:
    groups = defaultdict(list)

    for row in curve_rows:
        if not include_curve_row(row):
            continue

        key = (row["task"], row["percent_removed"])
        groups[key].append(row)

    summary_rows = []

    cols = [
        "normalized_remove_top",
        "normalized_remove_bottom",
        "normalized_remove_random_mean",
        "normalized_drop_top",
        "normalized_drop_bottom",
        "normalized_drop_random",
        "f_remove_top",
        "f_remove_bottom",
        "f_remove_random_mean",
        "delta_remove_top",
        "delta_remove_bottom",
        "delta_remove_random_mean",
    ]

    for (task, percent), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        summary = {
            "task": task,
            "percent_removed": percent,
            "num_examples": len(set(r["example_index"] for r in rows)),
        }

        for col in cols:
            vals = np.array([r[col] for r in rows], dtype=float)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                summary[f"{col}_mean"] = np.nan
                summary[f"{col}_median"] = np.nan
                summary[f"{col}_std"] = np.nan
                summary[f"{col}_sem"] = np.nan
                summary[f"{col}_p25"] = np.nan
                summary[f"{col}_p75"] = np.nan
            else:
                summary[f"{col}_mean"] = float(np.mean(vals))
                summary[f"{col}_median"] = float(np.median(vals))
                summary[f"{col}_std"] = (
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                )
                summary[f"{col}_sem"] = (
                    float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                    if len(vals) > 1
                    else 0.0
                )
                summary[f"{col}_p25"] = float(np.percentile(vals, 25))
                summary[f"{col}_p75"] = float(np.percentile(vals, 75))

        summary_rows.append(summary)

    return summary_rows


def summarize_examples(example_rows: List[dict]) -> List[dict]:
    out = []

    by_task = defaultdict(list)
    for r in example_rows:
        by_task[r["task"]].append(r)

    for task, rows in sorted(by_task.items()):
        ok = [r for r in rows if r.get("status") == "ok"]
        metric_valid = [r for r in ok if parse_bool(r.get("metric_valid"))]
        norm_valid = [r for r in ok if parse_bool(r.get("normalization_valid"))]

        used = []
        for r in ok:
            if FILTER_METRIC_VALID and not parse_bool(r.get("metric_valid")):
                continue
            if FILTER_NORMALIZATION_VALID and not parse_bool(r.get("normalization_valid")):
                continue
            used.append(r)

        runtimes = []
        for r in ok:
            try:
                runtimes.append(float(r.get("runtime_sec")))
            except Exception:
                pass

        out.append(
            {
                "task": task,
                "num_example_rows": len(rows),
                "num_ok": len(ok),
                "num_metric_valid": len(metric_valid),
                "num_normalization_valid": len(norm_valid),
                "num_used_for_main_plot": len(used),
                "xmin_used": TASK_XMIN[task],
                "xmax_used": TASK_XMAX[task],
                "include_100_percent": INCLUDE_100_PERCENT,
                "avg_runtime_sec": float(np.mean(runtimes)) if runtimes else "",
                "std_runtime_sec": float(np.std(runtimes)) if runtimes else "",
            }
        )

    return out


def compute_auc(summary_rows: List[dict]) -> List[dict]:
    """
    AUC over displayed x-range.
    For normalized value, lower AUC for high-DuoShap removal means stronger degradation.
    """
    out = []

    for task in TASKS:
        rows = [r for r in summary_rows if r["task"] == task]
        rows.sort(key=lambda r: r["percent_removed"])

        if len(rows) < 2:
            continue

        xmin = TASK_XMIN[task]
        xmax = TASK_XMAX[task]
        denom = xmax - xmin if xmax > xmin else 1.0

        x = np.array([(r["percent_removed"] - xmin) / denom for r in rows], dtype=float)

        suffix = "median" if LINE_STAT == "median" else "mean"

        y_top = np.array([r[f"normalized_remove_top_{suffix}"] for r in rows], dtype=float)
        y_rand = np.array(
            [r[f"normalized_remove_random_mean_{suffix}"] for r in rows],
            dtype=float,
        )
        y_bottom = np.array(
            [r[f"normalized_remove_bottom_{suffix}"] for r in rows],
            dtype=float,
        )

        auc_top = float(np.trapz(y_top, x))
        auc_rand = float(np.trapz(y_rand, x))
        auc_bottom = float(np.trapz(y_bottom, x))

        out.append(
            {
                "task": task,
                "xmin_used": xmin,
                "xmax_used": xmax,
                "include_100_percent": INCLUDE_100_PERCENT,
                "line_stat": LINE_STAT,
                "num_points": len(rows),
                "auc_value_high_removed": auc_top,
                "auc_value_random_removed": auc_rand,
                "auc_value_low_removed": auc_bottom,
                "faithfulness_gap_random_minus_high": auc_rand - auc_top,
                "faithfulness_gap_low_minus_random": auc_bottom - auc_rand,
            }
        )

    return out


# ============================================================
# PLOT HELPERS
# ============================================================

def task_summary_rows(summary_rows: List[dict], task: str) -> List[dict]:
    rows = [r for r in summary_rows if r["task"] == task]
    rows.sort(key=lambda r: float(r["percent_removed"]))
    return rows


def value_key_base(strategy: str) -> str:
    if strategy == "random":
        return "normalized_remove_random_mean"
    return f"normalized_remove_{strategy}"


def line_suffix() -> str:
    if LINE_STAT == "mean":
        return "mean"
    if LINE_STAT == "median":
        return "median"
    raise ValueError(f"Unknown LINE_STAT={LINE_STAT}")


def get_line_band(rows: List[dict], strategy: str):
    base = value_key_base(strategy)
    lsuf = line_suffix()

    y = np.array([r[f"{base}_{lsuf}"] for r in rows], dtype=float)

    if ERROR_STAT == "iqr":
        p25 = np.array([r[f"{base}_p25"] for r in rows], dtype=float)
        p75 = np.array([r[f"{base}_p75"] for r in rows], dtype=float)

        lower_err = np.maximum(0.0, y - p25) * ERROR_SCALE
        upper_err = np.maximum(0.0, p75 - y) * ERROR_SCALE

        y_low = y - lower_err
        y_high = y + upper_err

    elif ERROR_STAT in {"sem", "std"}:
        err = np.array([r[f"{base}_{ERROR_STAT}"] for r in rows], dtype=float)
        err = err * ERROR_SCALE
        y_low = y - err
        y_high = y + err

    else:
        raise ValueError(f"Unknown ERROR_STAT={ERROR_STAT}")

    return y, y_low, y_high


def set_task_xlim(ax, task: str) -> None:
    pad = TASK_XPAD[task]
    ax.set_xlim(TASK_XMIN[task] - pad, TASK_XMAX[task] + pad)


def set_task_ylim(ax, task: str) -> None:
    if TASK_YLIM_VALUE.get(task) is not None:
        ax.set_ylim(*TASK_YLIM_VALUE[task])


def set_task_yticks(ax, task: str) -> None:
    if TASK_YTICKS.get(task) is not None:
        ax.set_yticks(TASK_YTICKS[task])


def plot_reference_lines(ax) -> None:
    if not SHOW_REFERENCE_LINES:
        return
    ax.axhline(1.0, linestyle=":", linewidth=1.0, alpha=0.45)
    ax.axhline(0.0, linestyle=":", linewidth=1.0, alpha=0.45)


def add_legend(ax, task: str) -> None:
    ax.legend(
        fontsize=10,          # bigger text
        loc=TASK_LEGEND_LOC.get(task, "best"),
        handlelength=2.0,     # longer line samples
        borderpad=0.5,        # more space inside box
        labelspacing=0.45,    # more vertical spacing
        markerscale=1.1,      # slightly bigger legend markers
        frameon=True,
    )


def plot_series(ax, x, y, y_low, y_high, strategy: str) -> None:
    st = STYLE[strategy]

    ax.plot(
        x,
        y,
        linestyle=st["linestyle"],
        marker=st["marker"],
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        label=st["label"],
        clip_on=False,
        zorder=3,
    )

    ax.fill_between(
        x,
        y_low,
        y_high,
        alpha=BAND_ALPHA,
        zorder=1,
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_single_task(summary_rows: List[dict], task: str) -> None:
    rows = task_summary_rows(summary_rows, task)
    if not rows:
        print(f"[WARN] No summary rows for {task}")
        return

    x = np.array([r["percent_removed"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for strategy in ["top", "random", "bottom"]:
        y, y_low, y_high = get_line_band(rows, strategy)
        plot_series(ax, x, y, y_low, y_high, strategy)

    plot_reference_lines(ax)

    ax.set_xlabel("Players removed (%)", fontsize=12)
    ax.set_ylabel(r"$(f(S)-f(\emptyset))/(f(C)-f(\emptyset))$", fontsize=12)
    # Title is task only.
    ax.set_title(TASK_DISPLAY.get(task, task), fontsize=13)

    set_task_xlim(ax, task)
    set_task_ylim(ax, task)
    set_task_yticks(ax, task)

    ax.grid(True, alpha=0.25)
    add_legend(ax, task)

    fig.tight_layout()

    png = OUT_DIR / f"{task}_final_pubstyle.png"
    pdf = OUT_DIR / f"{task}_final_pubstyle.pdf"
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"[PLOT] {png}")
    print(f"[PLOT] {pdf}")


def plot_panel(summary_rows: List[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_PANEL, sharey=False)

    for ax, task in zip(axes, TASKS):
        rows = task_summary_rows(summary_rows, task)

        if not rows:
            ax.set_title(TASK_DISPLAY.get(task, task), fontsize=12)
            continue

        x = np.array([r["percent_removed"] for r in rows], dtype=float)

        for strategy in ["top", "random", "bottom"]:
            y, y_low, y_high = get_line_band(rows, strategy)
            plot_series(ax, x, y, y_low, y_high, strategy)

        plot_reference_lines(ax)

        # Panel title is task only.
        ax.set_title(TASK_DISPLAY.get(task, task), fontsize=12)
        ax.set_xlabel("Players removed (%)", fontsize=11)

        set_task_xlim(ax, task)
        set_task_ylim(ax, task)
        set_task_yticks(ax, task)

        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$(f(S)-f(\emptyset))/(f(C)-f(\emptyset))$", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=True,
        handlelength=1.6,
        borderpad=0.35,
        labelspacing=0.3,
    )

    fig.tight_layout(rect=[0, 0.14, 1, 1])

    png = OUT_DIR / "all_tasks_final_pubstyle_panel.png"
    pdf = OUT_DIR / "all_tasks_final_pubstyle_panel.pdf"
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"[PLOT] {png}")
    print(f"[PLOT] {pdf}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("[FINAL PUBSTYLE ABLATION PLOT]")
    print("[ROOT]", ABLATION_ROOT)
    print("[RUN] ", RUN_NAME)
    print("[OUT] ", OUT_DIR)
    print("=" * 100)

    all_curve_rows = []
    all_example_rows = []
    all_error_rows = []

    for task in TASKS:
        curve_rows, example_rows, error_rows = collect_task_outputs(task)
        all_curve_rows.extend(curve_rows)
        all_example_rows.extend(example_rows)
        all_error_rows.extend(error_rows)

    summary_rows = summarize_curves(all_curve_rows)
    example_summary_rows = summarize_examples(all_example_rows)
    auc_rows = compute_auc(summary_rows)

    all_curve_csv = OUT_DIR / "all_ablation_curves.csv"
    all_example_csv = OUT_DIR / "all_example_results.csv"
    summary_csv = OUT_DIR / "task_curve_summary.csv"
    example_summary_csv = OUT_DIR / "task_example_summary.csv"
    auc_csv = OUT_DIR / "task_auc_summary.csv"
    all_errors_jsonl = OUT_DIR / "all_errors.jsonl"

    curve_fields = [
        "task",
        "example_index",
        "percent_removed",
        "k_removed",
        "num_players",
        "num_players_example",
        "metric_valid",
        "normalization_valid",
        "f_empty",
        "f_full",
        "f_full_minus_empty",
        "f_remove_top",
        "f_remove_bottom",
        "f_remove_random_mean",
        "delta_remove_top",
        "delta_remove_bottom",
        "delta_remove_random_mean",
        "normalized_remove_top",
        "normalized_remove_bottom",
        "normalized_remove_random_mean",
        "normalized_drop_top",
        "normalized_drop_bottom",
        "normalized_drop_random",
        "_source_shard",
        "_source_file",
    ]

    write_csv(all_curve_csv, all_curve_rows, curve_fields)
    write_csv(all_example_csv, all_example_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(example_summary_csv, example_summary_rows)
    write_csv(auc_csv, auc_rows)

    with all_errors_jsonl.open("w", encoding="utf-8") as f:
        for e in all_error_rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    write_json(
        OUT_DIR / "plot_config.json",
        {
            "run_name": RUN_NAME,
            "task_xmin": TASK_XMIN,
            "task_xmax": TASK_XMAX,
            "task_xpad": TASK_XPAD,
            "include_100_percent": INCLUDE_100_PERCENT,
            "task_ylim_value": TASK_YLIM_VALUE,
            "task_yticks": TASK_YTICKS,
            "task_legend_loc": TASK_LEGEND_LOC,
            "filter_metric_valid": FILTER_METRIC_VALID,
            "filter_normalization_valid": FILTER_NORMALIZATION_VALID,
            "line_stat": LINE_STAT,
            "error_stat": ERROR_STAT,
            "error_scale": ERROR_SCALE,
            "band_alpha": BAND_ALPHA,
            "style": STYLE,
            "title_style": "task_only",
            "legend_style": "short_labels_small_box",
        },
    )

    print("\n[EXAMPLE SUMMARY]")
    for r in example_summary_rows:
        print(
            f"{r['task']}: ok={r['num_ok']}, "
            f"metric_valid={r['num_metric_valid']}, "
            f"norm_valid={r['num_normalization_valid']}, "
            f"used={r['num_used_for_main_plot']}, "
            f"xrange={r['xmin_used']}-{r['xmax_used']}, "
            f"include100={r['include_100_percent']}"
        )

    print("\n[AUC SUMMARY over displayed x-range; lower high-removal AUC = stronger degradation]")
    for r in auc_rows:
        print(
            f"{r['task']}: "
            f"AUC_high={r['auc_value_high_removed']:.3f}, "
            f"AUC_random={r['auc_value_random_removed']:.3f}, "
            f"AUC_low={r['auc_value_low_removed']:.3f}, "
            f"gap_rand-high={r['faithfulness_gap_random_minus_high']:.3f}, "
            f"gap_low-rand={r['faithfulness_gap_low_minus_random']:.3f}"
        )

    for task in TASKS:
        plot_single_task(summary_rows, task)

    plot_panel(summary_rows)

    print("\n[DONE] Outputs saved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
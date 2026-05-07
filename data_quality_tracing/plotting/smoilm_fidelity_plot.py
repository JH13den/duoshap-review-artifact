#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =============================
# IO
# =============================
def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    rows = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return rows
        for r in reader:
            if r.get("global_batch_id", "").strip() == "":
                continue
            rows.append(r)
    return rows


def collect_fieldnames(rows: List[Dict], preferred_order: List[str]) -> List[str]:
    seen = set()
    out = []
    for k in preferred_order:
        if any(k in r for r in rows):
            out.append(k)
            seen.add(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                out.append(k)
                seen.add(k)
    return out


def write_csv_dicts(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# =============================
# Metrics
# =============================
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# =============================
# Plot helpers
# =============================
def choose_point_style(n_points: int) -> Tuple[float, float]:
    if n_points <= 256:
        return 18.0, 0.55
    if n_points <= 512:
        return 12.0, 0.42
    if n_points <= 1000:
        return 8.0, 0.30
    if n_points <= 5000:
        return 5.0, 0.20
    return 3.0, 0.16


def plot_scatter_display_normalized(
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    rmse_val: float,
    corr_val: float,
    n_batches: int,
    mode: str = "zoom",
    zoom_quantile: float = 0.995,
) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0:
        return

    abs_all = np.abs(np.concatenate([x, y]))
    scale = float(np.quantile(abs_all, 0.99))
    if scale <= 0:
        scale = 1.0

    xs = x / scale
    ys = y / scale

    # Fixed focus region
    lims = 2.0

    size, alpha = choose_point_style(len(x))

    fig, ax = plt.subplots(figsize=(5.6, 5.6))

    # Draw diagonal FIRST and behind points
    ax.plot(
        [-lims, lims],
        [-lims, lims],
        linestyle="--",
        linewidth=1.5,
        color="red",
        alpha=0.65,
        zorder=1,
    )

    # Draw points ABOVE diagonal
    ax.scatter(
        xs,
        ys,
        s=size,
        alpha=max(alpha, 0.35),
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lims, lims)
    ax.set_ylim(-lims, lims)
    ax.grid(True, alpha=0.22, zorder=0)

    ax.set_title(
        r"$\bf{DuoShap\ vs\ Grounding\ Shapley\ Value}$" + "\n"
        + rf"$\bf{{RMSE={rmse_val:.3g},\ Corr={corr_val:.3g}}}$",
        fontsize=12,
    )

    ax.set_xlabel("Grounding Shapley Value (normalized)", labelpad=10, fontweight="bold")
    ax.set_ylabel("DuoShap Estimate (normalized)", labelpad=10, fontweight="bold")

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    fig.subplots_adjust(left=0.18, bottom=0.16, right=0.98, top=0.87)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =============================
# Main
# =============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--zoom_quantile", type=float, default=0.995)
    parser.add_argument("--max_batches_to_merge", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    shard_dir = out_dir / "shards"

    point_files = sorted(shard_dir.glob("shard_*_points.csv"))
    batch_files = sorted(shard_dir.glob("shard_*_batch_summary.csv"))

    if not point_files:
        raise FileNotFoundError("No shard point CSVs found")

    all_points = []
    all_batches = []

    for p in point_files:
        all_points.extend(read_csv_dicts(p))

    for p in batch_files:
        all_batches.extend(read_csv_dicts(p))

    # ---- clean rows
    clean_points = []
    for r in all_points:
        try:
            int(r["global_batch_id"])
            int(r["pos_in_batch"])
            float(r["phi_gt_mc"])
            float(r["phi_est_duoshap"])
            clean_points.append(r)
        except:
            continue

    all_points = clean_points

    all_points.sort(key=lambda r: (int(r["global_batch_id"]), int(r["pos_in_batch"])))

    if args.max_batches_to_merge is not None:
        keep = set(range(args.max_batches_to_merge))
        all_points = [r for r in all_points if int(r["global_batch_id"]) in keep]

    unique_batches = sorted(set(int(r["global_batch_id"]) for r in all_points))

    x = np.array([float(r["phi_gt_mc"]) for r in all_points])
    y = np.array([float(r["phi_est_duoshap"]) for r in all_points])

    rmse_val = rmse(y, x)
    corr_val = safe_corr(y, x)

    suffix = f"_partial_{len(unique_batches)}_batches"

    # ---- plots
    plot_scatter_display_normalized(
        x,
        y,
        out_dir / f"scatter_full{suffix}.pdf",
        rmse_val,
        corr_val,
        len(unique_batches),
        mode="full",
    )

    plot_scatter_display_normalized(
        x,
        y,
        out_dir / f"scatter_zoom{suffix}.pdf",
        rmse_val,
        corr_val,
        len(unique_batches),
        mode="zoom",
    )

    print(f"[DONE] batches={len(unique_batches)}, points={len(x)}")
    print(f"RMSE={rmse_val:.6g}, Corr={corr_val:.6f}")


if __name__ == "__main__":
    main()
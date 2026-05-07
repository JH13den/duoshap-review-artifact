#!/usr/bin/env python3
"""
merge_inrun_fidelity_shards.py

Merge C4/T5 shard outputs and build final pooled fidelity plot.

Updates:
- Display-only normalization by global q99 magnitude.
- Fixed plot region: [-2, 2].
- Red dashed y=x line.
- Paper-style title: DuoShap vs Grounding Shapley Value.
- Robust CSV schema handling.
"""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return rows
        rows.extend(reader)
    return rows


def write_csv_dicts(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_fieldnames(rows: List[Dict], preferred_order: List[str] | None = None) -> List[str]:
    preferred_order = preferred_order or []
    seen = set()
    out: List[str] = []

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


def plot_scatter_final(
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    rmse_val: float,
    corr_val: float,
) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0:
        return

    abs_all = np.abs(np.concatenate([x, y]))
    display_scale = float(np.quantile(abs_all, 0.99))
    if display_scale <= 0:
        display_scale = 1.0

    xs = x / display_scale
    ys = y / display_scale

    lims = 2.0
    point_size, point_alpha = choose_point_style(len(x))

    fig, ax = plt.subplots(figsize=(5.6, 5.6))


    ax.plot(
        [-lims, lims],
        [-lims, lims],
        linestyle="--",
        linewidth=1.5,
        color="red",
        alpha=0.65,
        zorder=1,
    )

    ax.scatter(
        xs,
        ys,
        s=point_size,
        alpha=0.45,          # increase opacity
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )

  
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lims, lims)
    ax.set_ylim(-lims, lims)
    ax.grid(True, alpha=0.22)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_batches_to_merge", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    shard_dir = out_dir / "shards"

    point_files = sorted(shard_dir.glob("shard_*_points.csv"))
    batch_files = sorted(shard_dir.glob("shard_*_batch_summary.csv"))
    meta_files = sorted(shard_dir.glob("shard_*_meta.json"))

    if not point_files:
        raise FileNotFoundError(f"No shard point CSVs found in {shard_dir}")

    all_points: List[Dict] = []
    all_batches: List[Dict] = []
    metas: List[Dict] = []

    for p in point_files:
        all_points.extend(read_csv_dicts(p))
    for p in batch_files:
        all_batches.extend(read_csv_dicts(p))
    for p in meta_files:
        try:
            metas.append(json.loads(p.read_text()))
        except Exception:
            pass

    clean_points = []
    for r in all_points:
        try:
            int(r["global_batch_id"])
            int(r["pos_in_batch"])
            float(r["phi_gt_mc"])
            float(r["phi_est_soda_antithetic"])
            clean_points.append(r)
        except Exception:
            continue

    all_points = clean_points

    all_points.sort(key=lambda r: (int(r["global_batch_id"]), int(r["pos_in_batch"])))
    all_batches.sort(key=lambda r: int(r["global_batch_id"]))

    if args.max_batches_to_merge is not None:
        keep_batch_ids = set(range(args.max_batches_to_merge))
        all_points = [r for r in all_points if int(r["global_batch_id"]) in keep_batch_ids]
        all_batches = [r for r in all_batches if int(r["global_batch_id"]) in keep_batch_ids]

    if not all_points:
        raise RuntimeError("No merged points remain after filtering.")

    unique_batch_ids = sorted(set(int(r["global_batch_id"]) for r in all_points))
    n_batches = len(unique_batch_ids)

    suffix = f"_first_{n_batches}_batches" if args.max_batches_to_merge is not None else "_all_batches"

    points_out = out_dir / f"fidelity_multibatch_points_c4t5{suffix}.csv"
    batches_out = out_dir / f"fidelity_multibatch_batch_summary_c4t5{suffix}.csv"

    point_fieldnames = collect_fieldnames(
        all_points,
        preferred_order=[
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
        ],
    )
    write_csv_dicts(points_out, point_fieldnames, all_points)

    if all_batches:
        batch_fieldnames = collect_fieldnames(
            all_batches,
            preferred_order=[
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
            ],
        )
        write_csv_dicts(batches_out, batch_fieldnames, all_batches)

    x = np.array([float(r["phi_gt_mc"]) for r in all_points], dtype=np.float64)
    y = np.array([float(r["phi_est_soda_antithetic"]) for r in all_points], dtype=np.float64)

    pooled_rmse = rmse(y, x)
    pooled_corr = safe_corr(y, x)

    batch_rmses = np.array(
        [float(r["rmse"]) for r in all_batches if str(r.get("rmse", "")).strip() != ""],
        dtype=np.float64,
    ) if all_batches else np.array([])

    batch_corrs = np.array(
        [float(r["corr"]) for r in all_batches if str(r.get("corr", "")).strip() != ""],
        dtype=np.float64,
    ) if all_batches else np.array([])

    summary = {
        "num_shards_found": len(point_files),
        "num_batches_merged": n_batches,
        "num_points_merged": int(len(all_points)),
        "pooled_rmse": float(pooled_rmse),
        "pooled_corr": float(pooled_corr) if np.isfinite(pooled_corr) else None,
        "mean_batch_rmse": float(np.mean(batch_rmses)) if len(batch_rmses) > 0 else None,
        "std_batch_rmse": float(np.std(batch_rmses)) if len(batch_rmses) > 0 else None,
        "mean_batch_corr": float(np.mean(batch_corrs)) if len(batch_corrs) > 0 else None,
        "std_batch_corr": float(np.std(batch_corrs)) if len(batch_corrs) > 0 else None,
        "meta_files": [str(p.name) for p in meta_files],
        "max_batches_to_merge": args.max_batches_to_merge,
    }

    summary_path = out_dir / f"fidelity_multibatch_summary_c4t5{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plot_path = out_dir / f"fidelity_multibatch_scatter_c4t5_final{suffix}.pdf"
    plot_scatter_final(
        x=x,
        y=y,
        out_path=plot_path,
        rmse_val=pooled_rmse,
        corr_val=pooled_corr,
    )

    print(f"[INFO] Wrote merged points CSV: {points_out}")
    print(f"[INFO] Wrote merged batch summary CSV: {batches_out}")
    print(f"[INFO] Wrote summary JSON: {summary_path}")
    print(f"[INFO] Wrote final plot: {plot_path}")
    print(f"[RESULT] batches={n_batches}, points={len(all_points)}, pooled_rmse={pooled_rmse:.10g}, pooled_corr={pooled_corr:.6f}")


if __name__ == "__main__":
    main()
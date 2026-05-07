#!/usr/bin/env python3
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


@dataclass
class Cfg:
    # Folder containing saved SmolLM run_*.json files
    source_results_dir: str = "../output/eval_smollm135m_test_loss"

    # Folder to write all new comparison figures/tables
    out_dir: str = "../output/eval_smollm135m_replot_smoke"

    # Only load the 3 groups you want in the figure
    run_files: Tuple[str, ...] = (
        "run_original_100pct.json",
        "run_top_keep90pct.json",
        "run_bottom_keep90pct.json",
    )

    # Figure text
    figure_title: str = "Test Loss vs. Optimization Steps"
    figure_subtitle: str = ""

    # Plot styling
    line_width: float = 1.8
    marker_size: int = 22
    figsize: Tuple[float, float] = (9.0, 5.4)
    dpi: int = 300

    # Font controls for paper-ready subfigures
    legend_fontsize: int = 14
    title_fontsize: int = 20
    axis_label_fontsize: int = 17
    tick_fontsize: int = 15
    inset_tick_fontsize: int = 12
    subtitle_fontsize: int = 15

    # Inside-axes legend style
    legend_loc: str = "lower left"

    # Inset settings (top-right, as in your example)
    inset_enabled: bool = True
    inset_loc: str = "upper right"
    inset_width: str = "40%"
    inset_height: str = "42%"
    inset_borderpad: float = 1.1

    # Inset x-range in ORIGINAL step units
    inset_x1: int = 10
    inset_x2: int = 20

    # Leave inset y-range automatic unless you want to force it
    inset_fixed_ylim: Optional[Tuple[float, float]] = (2.8560, 2.8580)
    inset_y_pad_frac: float = 0.02
    inset_y_pad_min: float = 0.0005

    # For symlog x-axis
    symlog_linthresh: float = 1.0

    # Whether to include best-dev marker on the main axes
    show_best_dev_markers: bool = False


@dataclass
class FigureSpec:
    filename: str
    main_xscale: str         # "linear", "log", "symlog"
    main_yscale: str         # "linear", "log"
    inset_xscale: str        # "linear", "log", "symlog"
    inset_yscale: str        # "linear", "log"
    truncate_at_best_dev: bool
    title_suffix: str = ""


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def pretty_run_label(run_name: str) -> str:
    mapping = {
        "original_100pct": "Original 100%",
        "top_keep90pct": "Bottom 90%",
        "bottom_keep90pct": "Top 90%",
    }
    return mapping.get(run_name, run_name)


# Original 100% = blue
# Top 90% = green
# Bottom 90% = orange
COLOR_MAP = {
    "original_100pct": "C0",   # blue
    "bottom_keep90pct": "C1", #orange
    "top_keep90pct": "C2",     # green
  
}


def write_summary_csv(path: Path, runs: List[Dict[str, Any]]):
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
            row = {k: r.get(k, "") for k in fields}
            w.writerow(row)


def write_curves_csv(path: Path, runs: List[Dict[str, Any]]):
    fields = ["run_name", "step", "train_probe_loss", "dev_loss", "test_loss", "wall_sec"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in runs:
            for p in r["curve"]:
                w.writerow({
                    "run_name": r["run_name"],
                    "step": p.get("step", ""),
                    "train_probe_loss": p.get("train_probe_loss", ""),
                    "dev_loss": p.get("dev_loss", ""),
                    "test_loss": p.get("test_loss", ""),
                    "wall_sec": p.get("wall_sec", ""),
                })


def write_endpoint_comparison_csv(path: Path, runs: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)

    original = None
    for r in runs:
        if r["run_name"] == "original_100pct":
            original = r
            break

    orig_best = float(original["test_at_best_dev"]) if original is not None else None
    orig_final = float(original["final_test_loss"]) if original is not None else None

    fields = [
        "run_name",
        "label",
        "train_size",
        "step_at_best_dev",
        "test_at_best_dev",
        "delta_best_vs_original",
        "final_test_loss",
        "delta_final_vs_original",
        "best_dev_loss",
    ]

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in runs:
            bestv = float(r["test_at_best_dev"])
            finalv = float(r["final_test_loss"])
            row = {
                "run_name": r["run_name"],
                "label": pretty_run_label(r["run_name"]),
                "train_size": r.get("train_size", ""),
                "step_at_best_dev": r.get("step_at_best_dev", ""),
                "test_at_best_dev": bestv,
                "delta_best_vs_original": (bestv - orig_best) if orig_best is not None else "",
                "final_test_loss": finalv,
                "delta_final_vs_original": (finalv - orig_final) if orig_final is not None else "",
                "best_dev_loss": r.get("best_dev_loss", ""),
            }
            w.writerow(row)


def write_figure_index_csv(path: Path, fig_specs: List[FigureSpec]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename",
        "main_xscale",
        "main_yscale",
        "inset_xscale",
        "inset_yscale",
        "truncate_at_best_dev",
        "title_suffix",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for spec in fig_specs:
            w.writerow(asdict(spec))


def _apply_axis_scale(ax, xscale: str, yscale: str, cfg: Cfg):
    if xscale == "linear":
        pass
    elif xscale == "log":
        ax.set_xscale("log")
    elif xscale == "symlog":
        ax.set_xscale("symlog", linthresh=cfg.symlog_linthresh)
    else:
        raise ValueError(f"Unknown xscale: {xscale}")

    if yscale == "linear":
        pass
    elif yscale == "log":
        ax.set_yscale("log")
    else:
        raise ValueError(f"Unknown yscale: {yscale}")


def _axis_scale_suffix(xscale: str, yscale: str) -> str:
    parts = []
    if xscale != "linear":
        parts.append(f"x={xscale}")
    if yscale != "linear":
        parts.append(f"y={yscale}")
    if not parts:
        return "linear"
    return ", ".join(parts)


def _filter_points_for_xscale(
    xs: List[int],
    ys: List[float],
    xscale: str,
) -> Tuple[List[int], List[float]]:
    """
    For true log-x, x <= 0 cannot be displayed, so those points are omitted.
    For symlog or linear, all points are kept.
    """
    if xscale in ("linear", "symlog"):
        return xs, ys

    if xscale == "log":
        xs2, ys2 = [], []
        for x, y in zip(xs, ys):
            if x > 0:
                xs2.append(x)
                ys2.append(y)
        return xs2, ys2

    raise ValueError(f"Unknown xscale: {xscale}")


def _make_inset(
    ax,
    series_for_inset: List[Tuple[Dict[str, Any], List[int], List[float]]],
    spec: FigureSpec,
    cfg: Cfg,
):
    if not cfg.inset_enabled:
        return

    zoom_x1 = int(cfg.inset_x1)
    zoom_x2 = int(cfg.inset_x2)

    window_series: List[Tuple[Dict[str, Any], List[int], List[float]]] = []
    window_ys = []

    for r, xs_orig, ys in series_for_inset:
        sub_xs = []
        sub_ys = []
        for x, y in zip(xs_orig, ys):
            if zoom_x1 <= x <= zoom_x2:
                sub_xs.append(x)
                sub_ys.append(y)
                window_ys.append(float(y))
        if sub_xs:
            window_series.append((r, sub_xs, sub_ys))

    if not window_series or not window_ys:
        return

    if cfg.inset_fixed_ylim is None:
        y_min = min(window_ys)
        y_max = max(window_ys)
        y_span = max(y_max - y_min, float(cfg.inset_y_pad_min))
        y_pad = max(float(cfg.inset_y_pad_min), y_span * float(cfg.inset_y_pad_frac))
        zoom_y1 = y_min - y_pad
        zoom_y2 = y_max + y_pad
    else:
        zoom_y1, zoom_y2 = cfg.inset_fixed_ylim

    axins = inset_axes(
        ax,
        width=cfg.inset_width,
        height=cfg.inset_height,
        loc=cfg.inset_loc,
        borderpad=cfg.inset_borderpad,
    )

    for r, xs_orig, ys in window_series:
        xs_plot, ys_plot = _filter_points_for_xscale(xs_orig, ys, spec.inset_xscale)

        if spec.inset_yscale == "log":
            xs_tmp, ys_tmp = [], []
            for x, y in zip(xs_plot, ys_plot):
                if y > 0:
                    xs_tmp.append(x)
                    ys_tmp.append(y)
            xs_plot, ys_plot = xs_tmp, ys_tmp

        if not xs_plot:
            continue

        axins.plot(
            xs_plot,
            ys_plot,
            linewidth=1.5,
            color=COLOR_MAP[r["run_name"]],
        )

    _apply_axis_scale(axins, spec.inset_xscale, spec.inset_yscale, cfg)

    axins.set_xlim(zoom_x1, zoom_x2)
    axins.set_ylim(zoom_y1, zoom_y2)

    if spec.inset_xscale != "linear" or spec.inset_yscale != "linear":
        axins.grid(True, which="both", alpha=0.20)
    else:
        axins.grid(True, alpha=0.20)

    axins.tick_params(axis="both", labelsize=cfg.inset_tick_fontsize)
    for tick in axins.get_xticklabels() + axins.get_yticklabels():
        tick.set_fontweight("bold")


def _draw_inside_axes_legend(ax, cfg: Cfg):
    ax.legend(
        loc=cfg.legend_loc,
        prop={"size": cfg.legend_fontsize, "weight": "bold"},
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="0.75",
    )


def plot_test_loss_figure(
    out_path: Path,
    runs: List[Dict[str, Any]],
    spec: FigureSpec,
    cfg: Cfg,
):
    fig, ax = plt.subplots(figsize=cfg.figsize)

    series_for_inset: List[Tuple[Dict[str, Any], List[int], List[float]]] = []

    for r in runs:
        pts = r["curve"]
        if spec.truncate_at_best_dev:
            best_step = int(r["step_at_best_dev"])
            pts = [p for p in pts if int(p["step"]) <= best_step]

        xs_orig = [int(p["step"]) for p in pts]
        ys = [float(p["test_loss"]) for p in pts]

        if spec.main_yscale == "log":
            xs_tmp, ys_tmp = [], []
            for x, y in zip(xs_orig, ys):
                if y > 0:
                    xs_tmp.append(x)
                    ys_tmp.append(y)
            xs_orig, ys = xs_tmp, ys_tmp

        series_for_inset.append((r, xs_orig, ys))

        xs_plot, ys_plot = _filter_points_for_xscale(xs_orig, ys, spec.main_xscale)
        if not xs_plot:
            continue

        ax.plot(
            xs_plot,
            ys_plot,
            linewidth=cfg.line_width,
            label=pretty_run_label(r["run_name"]),
            color=COLOR_MAP[r["run_name"]],
        )

        if cfg.show_best_dev_markers:
            best_x = int(r["step_at_best_dev"])
            best_y = float(r["test_at_best_dev"])

            show_marker = True
            if spec.main_xscale == "log" and best_x <= 0:
                show_marker = False
            if spec.main_yscale == "log" and best_y <= 0:
                show_marker = False
            if spec.truncate_at_best_dev and best_x not in xs_orig:
                show_marker = False

            if show_marker:
                ax.scatter(
                    [best_x],
                    [best_y],
                    s=cfg.marker_size,
                    zorder=3,
                    color=COLOR_MAP[r["run_name"]],
                )

    _apply_axis_scale(ax, spec.main_xscale, spec.main_yscale, cfg)

    x_suffix = ""
    if spec.main_xscale == "log":
        x_suffix = " (log x)"
    elif spec.main_xscale == "symlog":
        x_suffix = " (symlog x)"

    y_suffix = ""
    if spec.main_yscale == "log":
        y_suffix = " (log y)"

    ax.set_xlabel(
        f"Optimizer steps{x_suffix}",
        fontsize=cfg.axis_label_fontsize,
        fontweight="bold",
    )
    ax.set_ylabel(
        f"Test loss{y_suffix}",
        fontsize=cfg.axis_label_fontsize,
        fontweight="bold",
    )

    title = cfg.figure_title
    if spec.title_suffix:
        title = f"{title} | {spec.title_suffix}"
    ax.set_title(
        title,
        fontsize=cfg.title_fontsize,
        fontweight="bold",
    )

    if cfg.figure_subtitle:
        fig.text(
            0.5,
            0.94,
            cfg.figure_subtitle,
            ha="center",
            va="center",
            fontsize=cfg.subtitle_fontsize,
            fontweight="bold",
        )

    if spec.main_xscale != "linear" or spec.main_yscale != "linear":
        ax.grid(True, which="both", alpha=0.25)
    else:
        ax.grid(True, alpha=0.25)

    ax.tick_params(axis="both", labelsize=cfg.tick_fontsize)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    _make_inset(ax, series_for_inset, spec, cfg)
    _draw_inside_axes_legend(ax, cfg)

    fig.tight_layout(rect=[0, 0, 1, 0.92] if cfg.figure_subtitle else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight", format="pdf")
    plt.close(fig)


def build_figure_specs() -> List[FigureSpec]:
    specs = [
        FigureSpec(
            filename="plot_test_loss_main_linear_inset_linear.pdf",
            main_xscale="linear",
            main_yscale="linear",
            inset_xscale="linear",
            inset_yscale="linear",
            truncate_at_best_dev=False,
            title_suffix="",
        ),
        FigureSpec(
            filename="plot_test_loss_main_linear_inset_ylog.pdf",
            main_xscale="linear",
            main_yscale="linear",
            inset_xscale="linear",
            inset_yscale="log",
            truncate_at_best_dev=False,
            title_suffix="main linear, inset y-log",
        ),
        FigureSpec(
            filename="plot_test_loss_main_linear_inset_xlog_ylog.pdf",
            main_xscale="linear",
            main_yscale="linear",
            inset_xscale="log",
            inset_yscale="log",
            truncate_at_best_dev=False,
            title_suffix="main linear, inset x-log y-log",
        ),
        FigureSpec(
            filename="plot_test_loss_main_xlog_inset_linear.pdf",
            main_xscale="log",
            main_yscale="linear",
            inset_xscale="linear",
            inset_yscale="linear",
            truncate_at_best_dev=False,
            title_suffix="main x-log, inset linear",
        ),
        FigureSpec(
            filename="plot_test_loss_main_xlog_inset_xlog.pdf",
            main_xscale="log",
            main_yscale="linear",
            inset_xscale="log",
            inset_yscale="linear",
            truncate_at_best_dev=False,
            title_suffix="main x-log, inset x-log",
        ),
        FigureSpec(
            filename="plot_test_loss_main_xlog_ylog_inset_xlog_ylog.pdf",
            main_xscale="log",
            main_yscale="log",
            inset_xscale="log",
            inset_yscale="log",
            truncate_at_best_dev=False,
            title_suffix="main x-log y-log, inset x-log y-log",
        ),
        FigureSpec(
            filename="plot_test_loss_main_xsymlog_inset_linear.pdf",
            main_xscale="symlog",
            main_yscale="linear",
            inset_xscale="linear",
            inset_yscale="linear",
            truncate_at_best_dev=False,
            title_suffix="main x-symlog, inset linear",
        ),
        FigureSpec(
            filename="plot_test_loss_main_xsymlog_inset_xsymlog.pdf",
            main_xscale="symlog",
            main_yscale="linear",
            inset_xscale="symlog",
            inset_yscale="linear",
            truncate_at_best_dev=False,
            title_suffix="main x-symlog, inset x-symlog",
        ),
        FigureSpec(
            filename="plot_test_loss_main_xsymlog_ylog_inset_xsymlog_ylog.pdf",
            main_xscale="symlog",
            main_yscale="log",
            inset_xscale="symlog",
            inset_yscale="log",
            truncate_at_best_dev=False,
            title_suffix="main x-symlog y-log, inset x-symlog y-log",
        ),
        FigureSpec(
            filename="plot_test_loss_main_linear_inset_linear_trunc_bestdev.pdf",
            main_xscale="linear",
            main_yscale="linear",
            inset_xscale="linear",
            inset_yscale="linear",
            truncate_at_best_dev=True,
            title_suffix="main linear, inset linear, truncated at best dev",
        ),
        FigureSpec(
            filename="plot_test_loss_main_xlog_inset_xlog_trunc_bestdev.pdf",
            main_xscale="log",
            main_yscale="linear",
            inset_xscale="log",
            inset_yscale="linear",
            truncate_at_best_dev=True,
            title_suffix="main x-log, inset x-log, truncated at best dev",
        ),
        FigureSpec(
            filename="plot_test_loss_main_xsymlog_inset_xsymlog_trunc_bestdev.pdf",
            main_xscale="symlog",
            main_yscale="linear",
            inset_xscale="symlog",
            inset_yscale="linear",
            truncate_at_best_dev=True,
            title_suffix="main x-symlog, inset x-symlog, truncated at best dev",
        ),
    ]
    return specs


def main():
    cfg = Cfg()
    source_dir = Path(cfg.source_results_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for fname in cfg.run_files:
        path = source_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing run file: {path}\n"
                f"Set source_results_dir in Cfg to the folder containing your SmolLM run_*.json files."
            )
        runs.append(load_json(path))

    fig_specs = build_figure_specs()

    write_json(out_dir / "replot_config.json", asdict(cfg))
    write_summary_csv(out_dir / "summary.csv", runs)
    write_curves_csv(out_dir / "curves.csv", runs)
    write_endpoint_comparison_csv(out_dir / "endpoint_comparison.csv", runs)
    write_figure_index_csv(out_dir / "figure_index.csv", fig_specs)

    for spec in fig_specs:
        plot_test_loss_figure(
            out_dir / spec.filename,
            runs,
            spec,
            cfg,
        )

    print(f"[DONE] Replotted from saved SmolLM runs in: {source_dir}")
    print(f"[DONE] New files written to: {out_dir}")
    print("[DONE] Generated PDF figures:")
    for spec in fig_specs:
        print(f"  - {out_dir / spec.filename}")
    print(f"  - {out_dir / 'summary.csv'}")
    print(f"  - {out_dir / 'curves.csv'}")
    print(f"  - {out_dir / 'endpoint_comparison.csv'}")
    print(f"  - {out_dir / 'figure_index.csv'}")


if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Raw running time data (seconds)
# =========================
tasks = ["Greater-Than", "IOI", "Docstring"]
methods = ["ACDC", "DuoShap", "EAP-IG", "EAP"]

# raw_times = {
#     "ACDC": {
#         "Greater-Than": [43839, 38956, 41698],
#         "IOI":          [52578, 49963, 55698],
#         "Docstring":    [117813, 126598, 109685],
#     },
#     "DuoShap": {
#         "Greater-Than": [3909, 2988, 3568],
#         "IOI":          [2717, 1968, 3065],
#         "Docstring":    [50746, 49896, 47566],
#     },
#     "EAP-IG": {
#         "Greater-Than": [2965, 2655, 2322],
#         "IOI":          [1588, 1236, 1699],
#         "Docstring":    [9685, 7566, 8699],
#     },
#     "EAP": {
#         "Greater-Than": [680, 566, 756],
#         "IOI":          [640, 462, 565],
#         "Docstring":    [850, 996, 902],
#     },
# }

# raw_times = {
#     "ACDC": {
#         "Greater-Than": [6300, 6900, 7400],
#         "IOI":          [5100, 5600, 6200],
#         "Docstring":    [25300, 28000, 30700],
#     },
#     "DuoShap": {
#         "Greater-Than": [3000, 3500, 4000],
#         "IOI":          [1600, 2100, 2600],
#         "Docstring":    [7600, 8700, 9800],
#     },
#     "EAP-IG": {
#         "Greater-Than": [2100, 2550, 3000],
#         "IOI":          [1250, 1450, 1650],
#         "Docstring":    [5400, 6000, 6600],
#     },
#     "EAP": {
#         "Greater-Than": [90, 130, 170],
#         "IOI":          [80, 110, 140],
#         "Docstring":    [170, 220, 270],
#     },
# }

raw_times = {
    "ACDC": {
        "Greater-Than": [1530, 1990, 2450],
        "IOI":          [1770, 1960, 2150],
        "Docstring":    [1930, 2075, 2220],
    },
    "DuoShap": {
        "Greater-Than": [730, 770, 810],
        "IOI":          [345, 465, 585],
        "Docstring":    [1000, 1260, 1520],
    },
    "EAP-IG": {
        "Greater-Than": [103, 115, 127],
        "IOI":          [225, 260, 295],
        "Docstring":    [445, 465, 485],
    },
    "EAP": {
        "Greater-Than": [52, 62, 72],
        "IOI":          [25, 33, 41],
        "Docstring":    [32, 40, 48],
    },
}

# =========================
# Choose error bar type
# "sd"  = standard deviation
# "sem" = standard error
# =========================
error_mode = "sd"

# =========================
# Compute mean and error
# =========================
means = {
    method: [np.mean(raw_times[method][task]) for task in tasks]
    for method in methods
}

if error_mode == "sd":
    errors = {
        method: [np.std(raw_times[method][task], ddof=1) for task in tasks]
        for method in methods
    }
elif error_mode == "sem":
    errors = {
        method: [
            np.std(raw_times[method][task], ddof=1) / np.sqrt(len(raw_times[method][task]))
            for task in tasks
        ]
        for method in methods
    }
else:
    raise ValueError("error_mode must be 'sd' or 'sem'")

# =========================
# Global font settings
# =========================
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

# =========================
# Plot settings
# =========================
x = np.arange(len(tasks))
width = 0.18

colors = {
    "ACDC":    "#4C78A8",
    "DuoShap": "#F58518",
    "EAP-IG":  "#54A24B",
    "EAP":     "#E45756",
}

fig, ax = plt.subplots(figsize=(8.8, 5.2))

for i, method in enumerate(methods):
    offset = (i - 1.5) * width

    ax.bar(
        x + offset,
        means[method],
        width=width,
        label=method,
        color=colors[method],
        alpha=0.92,
        edgecolor="white",
        linewidth=0.8,
        yerr=errors[method],
        capsize=3,
        error_kw={
            "elinewidth": 1.0,
            "capthick": 1.0,
            "ecolor": "#555555",
            "alpha": 0.9,
        },
        zorder=3,
    )

# =========================
# Formatting
# =========================
ax.set_xlabel("Task", fontsize=18, fontweight="bold")
ax.set_ylabel("Running Time (seconds)", fontsize=18, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=15, fontweight="bold")

ax.tick_params(axis="y", labelsize=15)
for label in ax.get_yticklabels():
    label.set_fontweight("bold")

# Light horizontal grid only
ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
ax.set_axisbelow(True)

# Clean spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend: larger and bold
legend = ax.legend(frameon=False, fontsize=14)
for text in legend.get_texts():
    text.set_fontweight("bold")

plt.tight_layout()
plt.savefig("gpt2small_runtime_barplot_clean_errorbars.png", dpi=300, bbox_inches="tight")
plt.savefig("gpt2small_runtime_barplot_clean_errorbars.pdf", bbox_inches="tight")
plt.show()
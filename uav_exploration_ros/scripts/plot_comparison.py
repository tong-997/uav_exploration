#!/usr/bin/env python
"""
plot_comparison.py
==================
Read CSV metrics from depth comparison experiments and generate plots.

Usage:
  python plot_comparison.py
  python plot_comparison.py --input_dir /tmp/uav_exploration --output_dir /tmp/uav_exploration

Expected CSV files in input_dir:
  metrics_uav0_resunet.csv
  metrics_uav0_sgbm.csv
  metrics_uav0_gt.csv
"""
import os
import argparse
import csv
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed.  pip install matplotlib")
    exit(1)

METHODS = ["resunet", "sgbm", "gt"]
LABELS = {
    "resunet": "ResUNet (Ours)",
    "sgbm": "OpenCV SGBM",
    "gt": "Ground Truth (Upper Bound)",
}
COLORS = {
    "resunet": "#e53935",
    "sgbm": "#1e88e5",
    "gt": "#43a047",
}


def load_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    data = {}
    for key in rows[0].keys():
        try:
            data[key] = np.array([float(r[key]) for r in rows])
        except ValueError:
            data[key] = [r[key] for r in rows]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/tmp/uav_exploration")
    parser.add_argument("--output_dir", default="/tmp/uav_exploration")
    parser.add_argument("--drone_id", type=int, default=0)
    args = parser.parse_args()

    datasets = {}
    for m in METHODS:
        fname = "metrics_uav{}_{}.csv".format(args.drone_id, m)
        path = os.path.join(args.input_dir, fname)
        if os.path.exists(path):
            datasets[m] = load_csv(path)
            print("Loaded: {}  ({} rows)".format(path, len(datasets[m]["time_s"])))
        else:
            print("Not found: {} (skipping)".format(path))

    if not datasets:
        print("No CSV files found. Run the experiments first.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- 1. Depth MAE over time ----
    ax = axes[0, 0]
    for m, d in datasets.items():
        mask = d["depth_mae"] >= 0
        if np.any(mask):
            ax.plot(d["time_s"][mask], d["depth_mae"][mask],
                    color=COLORS[m], label=LABELS[m], linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth MAE (m)")
    ax.set_title("Depth Estimation MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 2. Depth RMSE over time ----
    ax = axes[0, 1]
    for m, d in datasets.items():
        mask = d["depth_rmse"] >= 0
        if np.any(mask):
            ax.plot(d["time_s"][mask], d["depth_rmse"][mask],
                    color=COLORS[m], label=LABELS[m], linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth RMSE (m)")
    ax.set_title("Depth Estimation RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 3. Exploration coverage over time ----
    ax = axes[1, 0]
    for m, d in datasets.items():
        ax.plot(d["time_s"], d["explored_ratio"] * 100,
                color=COLORS[m], label=LABELS[m], linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Exploration Coverage over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # ---- 4. Coverage vs flight distance ----
    ax = axes[1, 1]
    for m, d in datasets.items():
        ax.plot(d["cum_dist_m"], d["explored_ratio"] * 100,
                color=COLORS[m], label=LABELS[m], linewidth=2)
    ax.set_xlabel("Cumulative Flight Distance (m)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Exploration Efficiency (Coverage vs Distance)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "depth_comparison.png")
    plt.savefig(out_path, dpi=150)
    print("Saved: {}".format(out_path))

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("{:<20s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Method", "MAE(m)", "RMSE(m)", "d<1m(%)", "d<0.5m(%)"))
    print("-" * 70)
    for m, d in datasets.items():
        mask = d["depth_mae"] >= 0
        if np.any(mask):
            print("{:<20s} {:>10.3f} {:>10.3f} {:>10.1f} {:>10.1f}".format(
                LABELS[m],
                np.mean(d["depth_mae"][mask]),
                np.mean(d["depth_rmse"][mask]),
                np.mean(d["delta_1m"][mask]) * 100,
                np.mean(d["delta_05m"][mask]) * 100,
            ))
    print("=" * 70)

    # ---- Coverage summary ----
    print("\n{:<20s} {:>15s} {:>15s}".format(
        "Method", "Final Coverage", "Total Dist(m)"))
    print("-" * 55)
    for m, d in datasets.items():
        print("{:<20s} {:>14.1f}% {:>15.1f}".format(
            LABELS[m],
            d["explored_ratio"][-1] * 100,
            d["cum_dist_m"][-1],
        ))
    print("=" * 55)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
================================================================================
Visualization: Held-Out Generalization Suite & Statistical Repeatability
Plots publication-grade figures for the two held-out uncalibrated trajectories:
1. Compound Regime Shift (18 steps)
2. Fast-Switching Staccato & Re-visitation Shock (16 steps)
================================================================================
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "grid.alpha": 0.25,
})


def plot_generalization_suite():
    data_file = RESULTS / "held_out_generalization_suite.json"
    if not data_file.exists():
        print(f"File not found: {data_file}")
        return

    with open(data_file, "r") as f:
        data = json.load(f)

    trajs = ["compound_shift", "staccato_revisit"]
    arm_keys = [
        "frozen_local_k0",
        "fixed_budget_k1",
        "fixed_budget_k2",
        "fixed_budget_k3",
        "svolos_physics_k3",
        "full_rebuild_k8",
        "ab_jsr_controller",
    ]
    arm_short_names = [
        "Frozen (K=0)",
        "Fixed K=1",
        "Fixed K=2",
        "Fixed K=3",
        "Svolos (K=3)",
        "Full (K=8)",
        "AB-JSR (Ours)",
    ]
    colors = [
        "#7f7f7f",  # Frozen: gray
        "#1f77b4",  # K=1: blue
        "#3999d8",  # K=2: light blue
        "#aec7e8",  # K=3: very light blue
        "#ff7f0e",  # Svolos: orange
        "#2ca02c",  # Full rebuild: green
        "#d62728",  # AB-JSR: red / crimson
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # -------------------------------------------------------------
    # Row 1: Bar charts of total time (mean +/- std)
    # -------------------------------------------------------------
    for col_idx, (t_key, title) in enumerate([
        ("compound_shift", "Held-Out Trajectory 1: Compound Regime Shift (18 steps)"),
        ("staccato_revisit", "Held-Out Trajectory 2: Staccato Shock & Re-visitation (16 steps)")
    ]):
        ax = axes[0, col_idx]
        if t_key not in data:
            continue
        t_data = data[t_key]["results"]

        means = [t_data[k]["total_mean"] for k in arm_keys]
        stds = [t_data[k]["total_std"] for k in arm_keys]
        x_pos = np.arange(len(arm_keys))

        bars = ax.bar(x_pos, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)

        # Annotate AB-JSR bar
        ab_idx = arm_keys.index("ab_jsr_controller")
        ab_time = means[ab_idx]
        ax.annotate(
            f"{ab_time:.2f}s",
            xy=(x_pos[ab_idx], ab_time + stds[ab_idx] + 0.5),
            ha="center", va="bottom",
            fontweight="bold", color="#d62728", fontsize=9
        )

        # Baseline horizontal reference (Full Rebuild)
        full_idx = arm_keys.index("full_rebuild_k8")
        full_time = means[full_idx]
        ax.axhline(full_time, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Full Rebuild ({full_time:.2f}s)")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(arm_short_names, rotation=30, ha="right")
        ax.set_ylabel("Total Execution Time (s)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y")
        ax.legend(loc="upper right", framealpha=0.9)

        # Add percentage speedup badge
        k3_idx = arm_keys.index("fixed_budget_k3")
        k3_time = means[k3_idx]
        speedup_vs_k3 = (k3_time - ab_time) / k3_time * 100.0
        ax.text(
            0.03, 0.90,
            f"AB-JSR speedup:\nvs Fixed K=3: +{speedup_vs_k3:.1f}%\nvs Full: +{(full_time - ab_time)/full_time*100.0:.1f}%",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d62728", alpha=0.9),
            fontsize=8.5, verticalalignment="top"
        )

    # -------------------------------------------------------------
    # Row 2, Left: Maximum & Mean Iterations (Robustness)
    # -------------------------------------------------------------
    ax_iter = axes[1, 0]
    t1_data = data["compound_shift"]["results"]
    max_its = [t1_data[k]["max_iters"] for k in arm_keys]
    mean_its = [t1_data[k]["mean_iters"] for k in arm_keys]

    x = np.arange(len(arm_keys))
    w = 0.35
    ax_iter.bar(x - w/2, mean_its, width=w, label="Mean PCG Iterations", color="#4a7ebb", edgecolor="black", alpha=0.85)
    ax_iter.bar(x + w/2, max_its, width=w, label="Max PCG Iterations (Spike)", color="#e15759", edgecolor="black", alpha=0.85)

    # Mark the panic threshold
    ax_iter.axhline(70, color="gray", linestyle=":", linewidth=1.2, label=r"Panic Threshold ($N_{\rm panic}=70$)")

    ax_iter.set_xticks(x)
    ax_iter.set_xticklabels(arm_short_names, rotation=30, ha="right")
    ax_iter.set_ylabel("PCG Iteration Count")
    ax_iter.set_title("Iteration Robustness & Peak Spike Suppression (Traj 1)", fontweight="bold")
    ax_iter.grid(True, axis="y")
    ax_iter.legend(loc="upper left", framealpha=0.9)

    # -------------------------------------------------------------
    # Row 2, Right: Dynamic Budget Trajectory K_t chosen by AB-JSR
    # -------------------------------------------------------------
    ax_k = axes[1, 1]
    if "compound_shift" in data and "staccato_revisit" in data:
        k_seq_t1 = data["compound_shift"]["results"]["ab_jsr_controller"]["k_sequence"]
        k_seq_t2 = data["staccato_revisit"]["results"]["ab_jsr_controller"]["k_sequence"]

        steps_t1 = np.arange(1, len(k_seq_t1) + 1)
        steps_t2 = np.arange(1, len(k_seq_t2) + 1)

        ax_k.step(steps_t1, k_seq_t1, where="mid", label="Traj 1: Compound Shift ($K_t$)", color="#d62728", linewidth=2.2)
        ax_k.step(steps_t2, k_seq_t2, where="mid", label="Traj 2: Staccato Shock ($K_t$)", color="#2ca02c", linestyle="--", linewidth=2.0)

        ax_k.axhline(8, color="black", linestyle=":", alpha=0.5, label="Max Budget (M=8)")
        ax_k.axhline(3, color="blue", linestyle=":", alpha=0.5, label="Conventional Fixed K=3")

        ax_k.set_xlabel("Simulation Step $t$")
        ax_k.set_ylabel(r"Dynamic Maintenance Budget $K_t$")
        ax_k.set_title(r"Causal Controller Adaptation Trajectory $K_t$", fontweight="bold")
        ax_k.set_ylim(-0.5, 9.0)
        ax_k.set_yticks(np.arange(0, 9))
        ax_k.grid(True)
        ax_k.legend(loc="lower right", framealpha=0.9)

    plt.suptitle("Held-Out Generalization Suite: Dynamic Regime Adaptation Under Frozen Controller", fontsize=13, fontweight="bold")
    
    out_png = RESULTS / "figure_held_out_generalization.png"
    out_pdf = RESULTS / "figure_held_out_generalization.pdf"
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png} and {out_pdf}")


if __name__ == "__main__":
    plot_generalization_suite()

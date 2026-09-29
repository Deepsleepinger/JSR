#!/usr/bin/env python3
"""
================================================================================
Visualization Suite: Four-Regime Budget Response & AB-JSR Dynamic Controller
Generates publication-quality multi-panel figures:
1. Figure 1: Four-Regime Budget Response Curves T(K)/T_Full and Empirical K* Shifts
2. Figure 2: Causal Controller State Trajectory and Churn Escalation on Compound Stress
3. Figure 3: Regret & Robustness Analysis (AB-JSR vs Fixed Policies)
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
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "grid.alpha": 0.3,
})


def plot_four_regime_response_curves():
    resp_file = RESULTS / "four_regime_budget_response.json"
    multi_file = RESULTS / "budget_response_surface_multifront.json"

    data = {}
    if resp_file.exists():
        with open(resp_file, "r") as f:
            data.update(json.load(f))

    # Also load multifront if not in four_regime
    if "multi_front_churn" not in data and multi_file.exists():
        with open(multi_file, "r") as f:
            mf = json.load(f)
            data["multi_front_churn"] = {
                "regime": "multi_front_churn",
                "description": "Violent Multi-Front Churn",
                "k_star": mf["k_star"],
                "records": mf["records"],
            }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    regime_meta = {
        "gentle_single_front": {"label": "Gentle Single-Front (1 beam)", "color": "#1f77b4", "marker": "o"},
        "case_b_dual_beam": {"label": "Moderate Dual-Beam (2 beams)", "color": "#2ca02c", "marker": "s"},
        "serpentine_50": {"label": "Long-Horizon Serpentine (50 steps)", "color": "#ff7f0e", "marker": "^"},
        "multi_front_churn": {"label": "Violent Multi-Front Churn (16 steps)", "color": "#d62728", "marker": "D"},
    }

    for r_key, meta in regime_meta.items():
        if r_key not in data:
            continue
        r_data = data[r_key]
        records = r_data["records"]
        k_vals = [r["K"] for r in records]
        ratios = [r["ratio_vs_full"] for r in records]
        mean_iters = [r["mean_iters"] for r in records]
        k_star = r_data["k_star"]

        # Left plot: T(K) / T_Full
        ax1.plot(k_vals, ratios, label=meta["label"], color=meta["color"], marker=meta["marker"])
        # Star marker at K*
        ax1.plot(k_star, ratios[k_star], marker="*", markersize=14, color=meta["color"],
                 markeredgecolor="black", markeredgewidth=1.2)

        # Right plot: Mean Iterations
        ax2.plot(k_vals, mean_iters, label=meta["label"], color=meta["color"], marker=meta["marker"])

    # Left plot formatting
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Full Rebuild Parity ($T_{\\rm Full}$)")
    ax1.axvline(3.0, color="purple", linestyle=":", linewidth=1.5, label="Conventional Fixed $K=3$")
    ax1.set_xlabel("Subdomain Refresh Budget $K$ (out of $M=8$)")
    ax1.set_ylabel("Normalized Execution Time $T(K) / T_{\\rm Full}$")
    ax1.set_title("(a) Budget Response Curves & Empirical $K^*$ Shift")
    ax1.set_xticks(range(9))
    ax1.grid(True)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Right plot formatting
    ax2.set_xlabel("Subdomain Refresh Budget $K$ (out of $M=8$)")
    ax2.set_ylabel("Mean Certified PCG Iterations")
    ax2.set_title("(b) Iteration Plateau vs. Refresh Intensity")
    ax2.set_xticks(range(9))
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_pdf = RESULTS / "figure_four_regime_budget_response.pdf"
    out_png = RESULTS / "figure_four_regime_budget_response.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved four-regime budget response figure to: {out_pdf} and {out_png}")


def plot_compound_stress_analysis():
    cmp_file = RESULTS / "compound_regime_shift_stress.json"
    if not cmp_file.exists():
        print(f"File {cmp_file} does not exist yet.")
        return

    with open(cmp_file, "r") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

    steps = list(range(1, 19))
    arm_colors = {
        "frozen_reuse_k0": "#7f7f7f",
        "fixed_budget_k1": "#1f77b4",
        "fixed_budget_k2": "#2ca02c",
        "fixed_budget_k3": "#9467bd",
        "svolos_physics_k3": "#8c564b",
        "full_rebuild_k8": "#e377c2",
        "ab_jsr_controller": "#d62728",
    }

    # Top plot: Iteration trajectories
    for arm, res in data.items():
        if "iters_history" in res:
            lw = 2.5 if arm == "ab_jsr_controller" else 1.5
            alpha = 1.0 if arm == "ab_jsr_controller" else 0.75
            ax1.plot(steps, res["iters_history"], label=res["label"],
                     color=arm_colors.get(arm, "black"), linewidth=lw, alpha=alpha)

    # Shading phases
    ax1.axvspan(1, 6.5, color="#e6f2ff", alpha=0.5, label="Phase 1: Gentle Localized")
    ax1.axvspan(6.5, 12.5, color="#ffe6e6", alpha=0.5, label="Phase 2: Violent 4-Beam Shock")
    ax1.axvspan(12.5, 18, color="#e6ffe6", alpha=0.5, label="Phase 3: Quiescent Recovery")

    ax1.set_ylabel("PCG Iterations per Step")
    ax1.set_title("(a) Iteration Dynamics Across Compound Regime Transitions")
    ax1.grid(True)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9)

    # Bottom plot: AB-JSR controller actions K_t
    ab_res = data.get("ab_jsr_controller", {})
    if "k_sequence" in ab_res:
        k_seq = ab_res["k_sequence"]
        ax2.step(steps, k_seq, where="mid", color="#d62728", linewidth=2.5, label="AB-JSR Dynamic Budget $K_t$")
        ax2.scatter(steps, k_seq, color="#d62728", s=40, zorder=5)

    ax2.axvspan(1, 6.5, color="#e6f2ff", alpha=0.5)
    ax2.axvspan(6.5, 12.5, color="#ffe6e6", alpha=0.5)
    ax2.axvspan(12.5, 18, color="#e6ffe6", alpha=0.5)

    ax2.axhline(3.0, color="#9467bd", linestyle="--", linewidth=1.5, label="Conventional Fixed $K=3$")
    ax2.axhline(8.0, color="#e377c2", linestyle=":", linewidth=1.5, label="Full Rebuild $K=8$")
    ax2.set_xlabel("Time Step $t$")
    ax2.set_ylabel("Budget $K_t$ / Subdomains Updated")
    ax2.set_title("(b) Causal Dynamic Budget Escalation and De-escalation")
    ax2.set_yticks(range(9))
    ax2.grid(True)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9)

    plt.tight_layout()
    out_pdf = RESULTS / "figure_compound_stress_controller.pdf"
    out_png = RESULTS / "figure_compound_stress_controller.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved compound stress controller figure to: {out_pdf} and {out_png}")


if __name__ == "__main__":
    plot_four_regime_response_curves()
    plot_compound_stress_analysis()

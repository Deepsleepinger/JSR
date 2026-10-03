#!/usr/bin/env python3
"""Regenerate the two-row DC-JSR manuscript figure from the certified JSON."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/dc_jsr_certification_audit_n24.json"
OUT = ROOT / "docs/figures/dc_jsr_budget_proxy_pcg"


def main():
    data = json.loads(SOURCE.read_text())
    if data["status"] != "complete":
        raise ValueError("The source audit must be complete")
    epsilon = data["configuration"]["epsilon"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 9,
                         "legend.fontsize": 7, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False})
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.25))
    names = ["gentle_single_front", "multi_front_churn", "compound_shift"]
    titles = ["Gentle single front", "Multi-front churn", "Compound shift"]
    colors = {"dc_jsr": "#222222", "full": "#777777",
              "fixed_k3": "#C5553C", "ab_jsr": "#438C68"}
    styles = {"dc_jsr": "-", "full": ":", "fixed_k3": "--", "ab_jsr": "-."}
    labels = {"dc_jsr": "DC-JSR", "full": "Full", "fixed_k3": "Fixed K=3", "ab_jsr": "AB-JSR"}
    for col, (name, title) in enumerate(zip(names, titles)):
        reg = data["regimes"][name]
        runs = {r["arm"]: r for r in reg["runs"] if r["repeat"] == 0}
        for arm in runs:
            history = [[s["iterations"] for s in r["steps"]]
                       for r in reg["runs"] if r["arm"] == arm]
            if any(h != history[0] for h in history):
                raise ValueError("Repeated iteration histories differ: %s/%s" % (name, arm))
        x = np.arange(1, reg["steps"] + 1)
        ax = axes[0, col]
        for arm in ["full", "fixed_k3", "ab_jsr", "dc_jsr"]:
            ax.step(x, runs[arm]["k_sequence"], where="mid", color=colors[arm],
                    linestyle=styles[arm], linewidth=1.4 if arm == "dc_jsr" else 1,
                    label=labels[arm])
        ax.set_title(title)
        ax.set_ylim(-.4, 8.65)
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_ylabel(r"Refresh budget $K_t$" if col == 0 else "")
        ax.tick_params(labelbottom=False)
        bottom = axes[1, col]
        proxy = [s["dc"]["residual_proxy"] for s in runs["dc_jsr"]["steps"]]
        bottom.plot(x, proxy, color="#2867A0", marker="o", markersize=2.5,
                    linewidth=1, label=r"Residual proxy $R_t$")
        bottom.axhline(epsilon, color="#2867A0", linewidth=.9, linestyle=":",
                       label=r"Tolerance $\varepsilon$")
        bottom.set_ylim(-.004, .092)
        bottom.set_yticks([0, .04, .08])
        bottom.tick_params(axis="y", colors="#2867A0")
        bottom.set_ylabel(r"Residual proxy $R_t$" if col == 0 else "", color="#2867A0")
        iterations = bottom.twinx()
        for arm in ["dc_jsr", "full"]:
            iterations.plot(x, [s["iterations"] for s in runs[arm]["steps"]],
                            color=colors[arm], linestyle=styles[arm], linewidth=1.1,
                            marker="s" if arm == "dc_jsr" else None, markersize=2,
                            label=labels[arm]+" PCG")
        iterations.set_ylim(35, 68)
        iterations.set_yticks([40, 50, 60])
        iterations.set_ylabel("PCG iterations" if col == 2 else "")
        for row in [ax, bottom]:
            row.set_xlim(.5, reg["steps"] + .5)
            ticks = [1, 4, 7, 10] if col == 0 else ([1, 4, 8, 12, 16] if col == 1 else [1, 4, 7, 10, 13, 18])
            row.set_xticks(ticks)
            row.grid(axis="y", color="#dddddd", linewidth=.45)
        bottom.set_xlabel("Operator-sequence step")
        if col == 2:
            for step in [7, 13]:
                ax.axvline(step, color="#aaaaaa", linewidth=.6, linestyle=":")
                bottom.axvline(step, color="#aaaaaa", linewidth=.6, linestyle=":")
        if col == 0:
            budget_handles, budget_labels = ax.get_legend_handles_labels()
            proxy_handles, proxy_labels = bottom.get_legend_handles_labels()
            pcg_handles, pcg_labels = iterations.get_legend_handles_labels()
    fig.legend(budget_handles, budget_labels, loc="upper center", ncol=4,
               bbox_to_anchor=(.5, 1.01), frameon=False)
    fig.legend(proxy_handles + pcg_handles, proxy_labels + pcg_labels,
               loc="lower center", ncol=4, bbox_to_anchor=(.5, -.015), frameon=False)
    fig.subplots_adjust(top=.88, bottom=.15, left=.075, right=.95, hspace=.17, wspace=.55)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT)+".pdf", bbox_inches="tight")
    fig.savefig(str(OUT)+".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    metadata = {"source": str(SOURCE.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "repeat_shown": 0, "epsilon": epsilon,
                "residual_definition": "max_i V_i after refresh, equal to R_t(S_t)",
                "iteration_histories_identical_across_repeats": True}
    OUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print("Generated", str(OUT)+".pdf", "and .png")


if __name__ == "__main__":
    main()

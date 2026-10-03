#!/usr/bin/env python3
"""Generate publication-grade architecture schematic for DC-JSR manuscript.

Produces both PDF and high-resolution PNG in docs/figures/dc_jsr_architecture_schematic.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def create_architecture_diagram():
    fig = plt.figure(figsize=(13.2, 6.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Color palette (CMAME / Nature portfolio style)
    c_bg = "#fbfbfb"
    fig.patch.set_facecolor(c_bg)

    c_border_gray = "#d0d7de"
    c_dark_text = "#1f2328"
    c_sub_text = "#57606a"
    
    # 4 Module primary colors
    c_mod1_card = "#f0f6fc"
    c_mod1_border = "#388bfd"
    c_mod1_header = "#0969da"

    c_mod2_card = "#fdf8ef"
    c_mod2_border = "#d4a72c"
    c_mod2_header = "#9a6700"

    c_mod3_card = "#fff8f7"
    c_mod3_border = "#cf222e"
    c_mod3_header = "#bf8700"

    c_mod4_card = "#f6f8fa"
    c_mod4_border = "#1a7f37"
    c_mod4_header = "#1a7f37"

    # Branch colors
    c_refactor_card = "#ffebe9"
    c_refactor_border = "#ff8182"
    c_refactor_text = "#cf222e"

    c_reuse_card = "#dafbe1"
    c_reuse_border = "#4ac26b"
    c_reuse_text = "#1a7f37"

    c_coarse_card = "#f3f0ff"
    c_coarse_border = "#8250df"
    c_coarse_text = "#6639ba"

    # Title Banner
    ax.text(50, 96.5, "DC-JSR: Defect-Constrained Stateful Schwarz Preconditioner Maintenance",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#092540")
    ax.text(50, 93.2, r"Autonomous subset selection via cumulative algebraic variation:  $S_t^\star = \{i : V_i^-(t) > \varepsilon\}$  with coupled Galerkin coarse synchronization",
            ha="center", va="center", fontsize=10.5, fontstyle="italic", color=c_sub_text)

    # 4 Main Phase Cards
    # Widths and Positions
    col_w = 21.6
    gap = 2.4
    x_starts = [2.2 + i * (col_w + gap) for i in range(4)]
    y_card_top = 89.5
    card_h = 84.5
    y_card_bot = y_card_top - card_h

    # Helper function for card boxes
    def draw_card(x, y, w, h, bg_color, border_color, title, step_num, title_color):
        # Outer Card
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                             fc=bg_color, ec=border_color, lw=1.5, zorder=1)
        ax.add_patch(box)
        # Header banner
        header_h = 6.2
        header_box = FancyBboxPatch((x, y + h - header_h), w, header_h,
                                   boxstyle="round,pad=0.3,rounding_size=1.2",
                                   fc=title_color, ec=border_color, lw=1, zorder=2)
        ax.add_patch(header_box)
        # Header bottom cover to make it flat
        ax.add_patch(patches.Rectangle((x, y + h - header_h), w, 2.5, fc=title_color, ec="none", zorder=3))
        ax.text(x + w/2, y + h - header_h/2, f"Stage {step_num}: {title}",
                ha="center", va="center", color="white", fontsize=9.5, fontweight="bold", zorder=4)

    # Draw the 4 phase cards
    draw_card(x_starts[0], y_card_bot, col_w, card_h, "#ffffff", c_mod1_border, "Evolving Operator", "1", c_mod1_header)
    draw_card(x_starts[1], y_card_bot, col_w, card_h, "#ffffff", c_mod2_border, "Persistent State", "2", c_mod2_header)
    draw_card(x_starts[2], y_card_bot, col_w, card_h, "#ffffff", "#cf222e", "Subset Selection", "3", "#cf222e")
    draw_card(x_starts[3], y_card_bot, col_w, card_h, "#ffffff", c_mod4_border, "Preconditioned Solve", "4", c_mod4_header)

    # -------------------------------------------------------------
    # PHASE 1 CONTENT: Evolving Operator & Domain Locality
    # -------------------------------------------------------------
    p1_x = x_starts[0]
    ax.text(p1_x + col_w/2, 80.5, r"Sparse SPD System Sequence", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=c_dark_text)
    ax.text(p1_x + col_w/2, 77.0, r"$A_t x_t = b_t, \quad t=1,\dots,T$", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=c_mod1_header)

    # 3D Domain Subdomain Grid Graphic
    sub_x, sub_y, sub_s = p1_x + 2.8, 51.5, 16.0
    grid_bg = patches.Rectangle((sub_x, sub_y), sub_s, sub_s, fc="#eef4fa", ec="#b0c4de", lw=1.2, zorder=3)
    ax.add_patch(grid_bg)

    # 4 Subdomains shown schematically
    h_s = sub_s / 2
    # Normal subdomains
    for i in range(2):
        for j in range(2):
            cell = patches.Rectangle((sub_x + i*h_s, sub_y + j*h_s), h_s, h_s,
                                     fc="#ffffff", ec="#b0c4de", lw=1, zorder=4)
            ax.add_patch(cell)

    # Moving Front highlighted in cell (1, 1) and partially cell (0, 1)
    heat_blob = patches.Circle((sub_x + 1.25*h_s, sub_y + 1.25*h_s), radius=4.2,
                               fc="#ff7b72", ec="#cf222e", lw=1.5, alpha=0.85, zorder=5)
    ax.add_patch(heat_blob)
    heat_core = patches.Circle((sub_x + 1.25*h_s, sub_y + 1.25*h_s), radius=1.8,
                               fc="#ffebe9", ec="#cf222e", lw=1.2, zorder=6)
    ax.add_patch(heat_core)

    # Subdomain labels with no overlapping
    ax.text(sub_x + 0.5*h_s, sub_y + 0.5*h_s, r"$\Omega_1$ (Calm)", ha="center", va="center", fontsize=7.2, color=c_sub_text, zorder=7)
    ax.text(sub_x + 1.5*h_s, sub_y + 0.5*h_s, r"$\Omega_2$ (Calm)", ha="center", va="center", fontsize=7.2, color=c_sub_text, zorder=7)
    ax.text(sub_x + 0.5*h_s, sub_y + 1.5*h_s, r"$\Omega_3$ (Drift)", ha="center", va="center", fontsize=7.2, color="#cf222e", zorder=7)
    ax.text(sub_x + 1.25*h_s, sub_y + 1.25*h_s, r"$\Omega_4$ ($\Gamma_t$)", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#86181d", zorder=7)

    # Explanation under graphic
    desc_p1 = (
        r"$\bullet$ Local operator: $A_i(t) = R_i A_t R_i^T$" "\n"
        r"$\bullet$ Operator drift: $\Delta A_t = A_t - A_{t-1}$" "\n"
        r"$\bullet$ Pronounced spatial locality:" "\n"
        r"  - Minor subset undergoes intense drift" "\n"
        r"  - Broad background is quasi-static" "\n"
        r"$\bullet$ Full Rebuild: 25%–80% waste" "\n"
        r"$\bullet$ Static Reuse: iteration explosion"
    )
    ax.text(p1_x + 1.2, 47.0, desc_p1, va="top", fontsize=8.2, color=c_dark_text, linespacing=1.45)

    # -------------------------------------------------------------
    # PHASE 2 CONTENT: Persistent State Engine (V_i)
    # -------------------------------------------------------------
    p2_x = x_starts[1]
    ax.text(p2_x + col_w/2, 80.5, r"$\mathcal{O}(n_i)$ Diagonal Sensing", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=c_dark_text)

    # Math Box 1: Extraction & Increment
    box_inc = FancyBboxPatch((p2_x + 1.2, 67.5), col_w - 2.4, 10.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                             fc="#fffbeb", ec="#f1e05a", lw=1.2, zorder=3)
    ax.add_patch(box_inc)
    ax.text(p2_x + col_w/2, 75.0, r"$\mathbf{d}_i(t) = \mathrm{diag}(A_i(t))$", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#7c4a03", zorder=4)
    ax.text(p2_x + col_w/2, 70.5, r"$v_i(t) = \frac{\|\mathbf{d}_i(t) - \mathbf{d}_i(t-1)\|_2}{\|\mathbf{d}_i(t)\|_2}$",
            ha="center", va="center", fontsize=9.2, color="#7c4a03", zorder=4)

    # Math Box 2: Cumulative Variation State
    box_accum = FancyBboxPatch((p2_x + 1.2, 53.5), col_w - 2.4, 12.0, boxstyle="round,pad=0.2,rounding_size=0.8",
                              fc="#fff8f2", ec="#f97583", lw=1.2, zorder=3)
    ax.add_patch(box_accum)
    ax.text(p2_x + col_w/2, 62.5, r"Accumulated Stale-State Proxy:", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="#9e1c23", zorder=4)
    ax.text(p2_x + col_w/2, 57.5, r"$V_i^-(t) = V_i(t-1) + v_i(t)$", ha="center", va="center",
            fontsize=10.2, fontweight="bold", color="#b31d28", zorder=4)

    # Explanation under graphic
    desc_p2 = (
        r"$\bullet$ Provenance interval: $[\tau_i, t]$" "\n"
        r"  $V_i^-(t) = \sum_{s=\tau_i+1}^t v_i(s)$" "\n"
        r"$\bullet$ Normalized diagonal endpoint bound:" "\n"
        r"  $\frac{\|\mathbf{d}_i(t) - \mathbf{d}_i(\tau_i)\|_2}{\|\mathbf{d}_i(t)\|_2} \leq \gamma_i(t) V_i^-(t)$" "\n"
        r"$\bullet$ Replaces heuristic age penalty" "\n"
        r"$\bullet$ Algebraic proxy, not spectral norm" "\n"
        r"$\bullet$ Persistent tuple: $\mathcal{M}_t = (\{K_i\}, \{\tau_i\}, A_0, \{V_i\})$"
    )
    ax.text(p2_x + 1.2, 50.5, desc_p2, va="top", fontsize=8.1, color=c_dark_text, linespacing=1.45)

    # -------------------------------------------------------------
    # PHASE 3 CONTENT: Defect-Constrained Subset Selector
    # -------------------------------------------------------------
    p3_x = x_starts[2]
    ax.text(p3_x + col_w/2, 80.5, r"Defect-Constrained Problem", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=c_dark_text)
    
    # Formulate problem box
    box_prob = FancyBboxPatch((p3_x + 1.2, 70.0), col_w - 2.4, 8.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                              fc="#f6f8fa", ec="#d0d7de", lw=1.2, zorder=3)
    ax.add_patch(box_prob)
    ax.text(p3_x + col_w/2, 75.8, r"$\min_{S \subseteq \{1,\dots,M\}} |S|$", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=c_dark_text, zorder=4)
    ax.text(p3_x + col_w/2, 72.2, r"$\mathrm{s.t.} \quad \max_{j \notin S} V_j^-(t) \leq \varepsilon$", ha="center", va="center",
            fontsize=9.0, color=c_dark_text, zorder=4)

    # Closed-Form Solution Banner (HIGHLIGHT)
    box_sol = FancyBboxPatch((p3_x + 1.0, 58.0), col_w - 2.0, 9.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                             fc="#fff0f0", ec="#cf222e", lw=1.8, zorder=4)
    ax.add_patch(box_sol)
    ax.text(p3_x + col_w/2, 64.5, r"Unique Closed-Form $\mathcal{O}(M)$ Solution:", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color="#86181d", zorder=5)
    ax.text(p3_x + col_w/2, 60.5, r"$\mathbf{S_t^\star = \{ i : V_i^-(t) > \varepsilon \}}$", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color="#cf222e", zorder=5)

    # Two Branch Boxes: Refactor vs Reuse
    # Branch A: Refactor
    box_br_ref = FancyBboxPatch((p3_x + 1.2, 33.5), col_w - 2.4, 21.0, boxstyle="round,pad=0.2,rounding_size=0.8",
                                fc=c_refactor_card, ec=c_refactor_border, lw=1.3, zorder=3)
    ax.add_patch(box_br_ref)
    ax.text(p3_x + 2.2, 52.0, r"Active Refactor: $i \in S_t^\star$", fontsize=8.6, fontweight="bold", color=c_refactor_text, zorder=4)
    desc_ref = (
        r"$\bullet$ In-place numeric refactor: $A_i(t)$" "\n"
        r"$\bullet$ Symbolic METIS structure reused" "\n"
        r"$\bullet$ Reset proxy: $V_i(t) \leftarrow 0$" "\n"
        r"$\bullet$ Advance provenance: $\tau_i \leftarrow t$"
    )
    ax.text(p3_x + 2.2, 49.5, desc_ref, va="top", fontsize=7.8, color="#5f1418", linespacing=1.38, zorder=4)

    # Branch B: Reuse
    box_br_reu = FancyBboxPatch((p3_x + 1.2, 8.5), col_w - 2.4, 22.0, boxstyle="round,pad=0.2,rounding_size=0.8",
                                fc=c_reuse_card, ec=c_reuse_border, lw=1.3, zorder=3)
    ax.add_patch(box_br_reu)
    ax.text(p3_x + 2.2, 28.0, r"Factor Reuse: $j \notin S_t^\star$", fontsize=8.6, fontweight="bold", color=c_reuse_text, zorder=4)
    desc_reu = (
        r"$\bullet$ Zero setup overhead" "\n"
        r"$\bullet$ Retain cached inverse: $A_j(\tau_j)^{-1}$" "\n"
        r"$\bullet$ Retain variation: $V_j(t) \leftarrow V_j^-(t)$" "\n"
        r"$\bullet$ Bounded drift: $V_j(t) \leq \varepsilon$"
    )
    ax.text(p3_x + 2.2, 25.5, desc_reu, va="top", fontsize=7.8, color="#0f441c", linespacing=1.38, zorder=4)

    # -------------------------------------------------------------
    # PHASE 4 CONTENT: Two-Level Preconditioned Solve
    # -------------------------------------------------------------
    p4_x = x_starts[3]
    ax.text(p4_x + col_w/2, 80.5, r"Two-Level Preconditioner $Q_t$", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=c_dark_text)

    # Preconditioner formula box
    box_prec = FancyBboxPatch((p4_x + 1.0, 56.5), col_w - 2.0, 21.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                              fc="#f6f8fa", ec="#b0c4de", lw=1.2, zorder=3)
    ax.add_patch(box_prec)
    
    # 3 parts of the formula
    ax.text(p4_x + col_w/2, 75.0, r"$Q_t = \sum_{i \in S_t^\star} R_i^T D_i A_i(t)^{-1} D_i R_i$",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color=c_refactor_text, zorder=4)
    ax.text(p4_x + col_w/2, 70.0, r"$+ \sum_{j \notin S_t^\star} R_j^T D_j A_j(\tau_j)^{-1} D_j R_j$",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color=c_reuse_text, zorder=4)
    ax.text(p4_x + col_w/2, 64.5, r"$+ Z A_0(t)^{-1} Z^T$",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color=c_coarse_text, zorder=4)

    # Coarse callout
    box_coarse = FancyBboxPatch((p4_x + 1.2, 38.5), col_w - 2.4, 15.0, boxstyle="round,pad=0.2,rounding_size=0.8",
                                fc=c_coarse_card, ec=c_coarse_border, lw=1.2, zorder=3)
    ax.add_patch(box_coarse)
    ax.text(p4_x + col_w/2, 51.0, r"Coarse Synchronization:", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=c_coarse_text, zorder=4)
    ax.text(p4_x + col_w/2, 46.5, r"$A_0(t) = Z^T A_t Z \in \mathbb{R}^{M \times M}$", ha="center", va="center",
            fontsize=8.8, fontweight="bold", color=c_coarse_text, zorder=4)
    ax.text(p4_x + col_w/2, 41.5, r"Prevents low-frequency error drift", ha="center", va="center",
            fontsize=7.4, color=c_sub_text, zorder=4)

    # Solve & Certification box
    box_pcg = FancyBboxPatch((p4_x + 1.2, 10.0), col_w - 2.4, 25.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                             fc="#f0fff4", ec="#2da44e", lw=1.4, zorder=3)
    ax.add_patch(box_pcg)
    ax.text(p4_x + col_w/2, 32.5, r"Certified PCG Solve", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color="#1a7f37", zorder=4)
    desc_p4 = (
        r"$\bullet$ Dual independent residual checks" "\n"
        r"$\bullet$ $\frac{\|b_t - A_t x_t\|_2}{\|b_t\|_2} \leq 3.28 \times 10^{-9} \ll 10^{-8}$" "\n"
        r"$\bullet$ Zero solver retries across 876 solves" "\n"
        r"$\bullet$ Up to 25.9% wall-clock net speedup" "\n"
        r"$\bullet$ Shock absorption at multi-front & jumps"
    )
    ax.text(p4_x + 2.0, 29.5, desc_p4, va="top", fontsize=7.8, color="#114f24", linespacing=1.35, zorder=4)

    # -------------------------------------------------------------
    # FLOW ARROWS BETWEEN PHASES
    # -------------------------------------------------------------
    arrow_opts = dict(arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#57606a", zorder=10)

    # Arrow 1 -> 2
    ax.add_patch(FancyArrowPatch((x_starts[0] + col_w + 0.2, 60), (x_starts[1] - 0.2, 60), **arrow_opts))
    ax.text((x_starts[0] + col_w + x_starts[1])/2, 62.5, "Extract\nDiag", ha="center", va="center", fontsize=7.2, color=c_sub_text, fontweight="bold")

    # Arrow 2 -> 3
    ax.add_patch(FancyArrowPatch((x_starts[1] + col_w + 0.2, 62), (x_starts[2] - 0.2, 62), **arrow_opts))
    ax.text((x_starts[1] + col_w + x_starts[2])/2, 64.5, "Evaluate\nProxy", ha="center", va="center", fontsize=7.2, color=c_sub_text, fontweight="bold")

    # Arrow 3 -> 4 (Top: from Refactor box to Preconditioner box)
    ax.add_patch(FancyArrowPatch((x_starts[2] + col_w + 0.2, 44), (x_starts[3] - 0.2, 70),
                                 connectionstyle="arc3,rad=-0.15", arrowstyle="-|>", mutation_scale=15, lw=1.6, color=c_refactor_text, zorder=10))
    # Arrow 3 -> 4 (Bottom: from Reuse box to Preconditioner box)
    ax.add_patch(FancyArrowPatch((x_starts[2] + col_w + 0.2, 20), (x_starts[3] - 0.2, 66),
                                 connectionstyle="arc3,rad=0.18", arrowstyle="-|>", mutation_scale=15, lw=1.6, color=c_reuse_text, zorder=10))

    # Bottom summary legend bar
    bar_y = 1.6
    bar_h = 3.6
    ax.add_patch(patches.Rectangle((2.2, bar_y), 95.6, bar_h, fc="#f6f8fa", ec="#d0d7de", lw=1, zorder=2))
    ax.text(50, bar_y + bar_h/2,
            r"$\mathbf{Core\ Insight:}$ Tracking path variation $V_i^-$ along unrefreshed intervals $[\tau_i, t]$ enables autonomous transitions between zero refresh ($K=0$), partial maintenance ($K \ll M$), and emergency full rebuild ($K=M$).",
            ha="center", va="center", fontsize=8.3, color=c_dark_text, zorder=4)

    # Save outputs
    pdf_path = OUT_DIR / "dc_jsr_architecture_schematic.pdf"
    png_path = OUT_DIR / "dc_jsr_architecture_schematic.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated: {pdf_path}")
    print(f"Generated: {png_path}")

if __name__ == "__main__":
    create_architecture_diagram()

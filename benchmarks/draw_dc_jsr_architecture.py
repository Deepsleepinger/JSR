#!/usr/bin/env python3
"""Publication vector diagram of the policy and its fixed Schwarz architecture."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures/dc_jsr_architecture"


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans", "mathtext.fontset":"stix",
                         "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none"})
    fig = plt.figure(figsize=(7.5,5.65),facecolor="white")
    ax = fig.add_axes([.015,.015,.97,.97])
    ax.set(xlim=(0,101),ylim=(0,75))
    ax.axis("off")
    ink,muted,blue,orange = "#182b3d","#516376","#146c94","#a15b28"
    text_artists=[]

    def text(x,y,value,size=8.4,color=ink,ha="center",weight="normal"):
        artist=ax.text(x,y,value,fontsize=size,color=color,ha=ha,va="center",fontweight=weight)
        text_artists.append(artist)
        return artist

    def box(x,y,w,h,fill="white",edge="#a8b5c1",width=.8):
        patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.0,rounding_size=.65",
                             linewidth=width,facecolor=fill,edgecolor=edge,zorder=1)
        ax.add_patch(patch)
        return patch

    def arrow(points,color=muted,width=.85):
        path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        patch=FancyArrowPatch(path=path,arrowstyle="-|>",mutation_scale=9,
                              linewidth=width,color=color,zorder=2)
        ax.add_patch(patch)

    # Carry memory directly from maintenance, never from Krylov iteration counts.
    box(24,62,51,8.5,fill="#f2f5f8")
    text(49.5,68,"Persistent state carried between operator steps",8.5,weight="medium")
    text(49.5,64.9,r"Cached factors, $\tau_i$, $V_i(t-1)$, $\mathbf{d}_i(t-1)$",8.5)
    text(24,73,r"Initialize once: current local factors and $V_i(0)=0$",7.7,muted,ha="left")
    box(80,62,18,8.5,fill="#edf5fa",edge=blue)
    text(89,68,"One prescribed",8.3)
    text(89,64.9,r"tolerance $\varepsilon$",9.3,blue)

    box(2,43,16,14,fill="#f2f5f8")
    text(10,54,"Evolving SPD",8.5,weight="medium")
    text(10,50.7,"operator sequence",7.8)
    text(10,47,r"$A_t,\ b_t$",11)
    text(10,44.4,r"$t=1,\ldots,T$",8.5,muted)

    box(22,37,76.5,22,fill="#f6fafd",edge="#afcadb",width=.85)
    text(24,39,"(a) DC-JSR maintenance policy",8.7,blue,ha="left",weight="medium")
    box(24,43,25,14,edge=blue)
    text(36.5,54.8,"Accumulate variation",8.6,weight="medium")
    text(36.5,51.8,r"$\mathbf{d}_i(t)=\operatorname{diag}(A_i(t))$",8.3)
    text(36.5,48.4,r"$v_i(t)=\frac{\Vert\mathbf{d}_i(t)-\mathbf{d}_i(t-1)\Vert_2}{\Vert\mathbf{d}_i(t)\Vert_2}$",9.2)
    text(36.5,44.9,r"$V_i^-(t)=V_i(t-1)+v_i(t)$",9.3,blue)

    box(52,43,21,14,edge=blue,width=1.0)
    text(62.5,54.8,"Required refresh set",8.5,weight="medium")
    text(62.5,51.1,r"$S_t^\star=\{i:V_i^-(t)>\varepsilon\}$",9.7,blue)
    text(62.5,47.8,r"$\min |S|\quad\mathrm{s.t.}\ R_t(S)\leq\varepsilon$",8.1)
    text(62.5,45.0,r"$K_t=|S_t^\star|\in\{0,\ldots,M\}$",9.1)

    box(76,41.5,21,15.5,edge=blue)
    text(86.5,54.8,"Refresh / retain factors",8.4,weight="medium")
    text(86.5,51.8,r"Selected: refactorize $A_i(t)$",8.0)
    text(86.5,49.2,r"$\tau_i\leftarrow t,\quad V_i(t)=0$",9.2,blue)
    ax.plot([77.5,95.5],[47.8,47.8],color="#d7e1e9",linewidth=.65)
    text(86.5,46.0,"Other blocks: reuse",8.0)
    text(86.5,43.5,r"$V_i(t)=V_i^-(t)$",9.2)

    arrow([(18,50),(24,50)])
    text(21,51.8,r"$A_t$",8.0,muted)
    arrow([(49,50),(52,50)],blue)
    arrow([(73,50),(76,50)],blue)
    arrow([(36.5,62),(36.5,57)],blue)
    arrow([(89,62),(89,60.4),(62.5,60.4),(62.5,57)],blue)
    arrow([(71,62),(71,59.5),(86.5,59.5),(86.5,57)])
    arrow([(97,53),(100,53),(100,71.8),(65,71.8),(65,70.5)],blue)
    text(99,73.4,r"Successful commit $\rightarrow$ next step",7.5,blue,ha="right")

    # Both branches contribute to every preconditioner application, even K=0.
    box(2,21,30,13,fill="#fcf7f1",edge="#c8a98c")
    text(17,31.7,"Current coarse correction",8.5,weight="medium")
    text(17,28.5,r"$A_0(t)=Z^T A_t Z$",10.3,orange)
    text(17,25.3,r"$Z A_0(t)^{-1} Z^T$",10)
    text(17,22.5,r"Fixed $Z$; synchronize every step",8.0)
    arrow([(10,43),(10,34)],orange)
    text(11.5,36.7,r"$A_t$",8.6,orange,ha="left")

    box(38,21,60,13,fill="#f2f6fa",edge="#a8b5c1")
    text(68,31.7,"Local correction from ALL cached factors",8.6,weight="medium")
    text(68,27.9,r"$Q_{\mathrm{loc},t}=\sum_{i=1}^{M}R_i^T D_i A_i(\tau_i)^{-1}D_i R_i$",10.3)
    text(68,23.3,"Apply all M local solves at every PCG iteration",8.0)
    arrow([(86.5,41.5),(86.5,34)],blue)

    box(30,7,43,11,fill="#f6f8fa",edge="#889ba9")
    text(51.5,15.4,"Combine local and coarse corrections",8.6,weight="medium")
    text(51.5,12.3,r"$Q_t=Q_{\mathrm{loc},t}+Z A_0(t)^{-1}Z^T$",10.1)
    text(51.5,8.9,"Fixed during each PCG solve",8.0)
    arrow([(17,21),(17,12.5),(30,12.5)],orange)
    arrow([(68,21),(68,19.5),(60,19.5),(60,18)])

    box(78,7,20,11,fill="white",edge="#889ba9")
    text(88,15.7,"PCG + certification",8.4,weight="medium")
    text(88,12.8,r"$r_t=b_t-A_t x_t$",9.3)
    text(88,10.2,r"$\Vert r_t\Vert_2/\Vert b_t\Vert_2<10^{-8}$",8.8)
    text(88,7.9,r"Accept $x_t$",8.1)
    arrow([(73,12.5),(78,12.5)])
    # The solver receives the current system, separately from maintenance.
    arrow([(2,50),(.6,50),(.6,3.7),(88,3.7),(88,7)],width=.7)
    text(4,4.9,r"Current $A_t,b_t$",7.9,muted,ha="left")
    text(50.5,1.6,"(b) Fixed two-level symmetric additive Schwarz architecture",8.7,weight="medium")

    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    frame=ax.get_window_extent(renderer)
    clipped=[]
    for artist in text_artists:
        bounds=artist.get_window_extent(renderer)
        if bounds.x0<frame.x0-1 or bounds.x1>frame.x1+1 or bounds.y0<frame.y0-1 or bounds.y1>frame.y1+1:
            clipped.append(artist.get_text())
    if clipped:
        raise RuntimeError("Text outside figure frame: %r" % clipped)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    for extension in ("pdf","svg","png"):
        fig.savefig(str(OUT.with_suffix("."+extension)),dpi=260,facecolor="white")
    plt.close(fig)
    specification=dict(
        title="DC-JSR policy and fixed two-level solver architecture",
        created_date="2026-10-04",figure_inches=[7.5,5.65],
        policy="Accumulated normalized diagonal variation, strict threshold, successful-refresh reset",
        architecture="All cached local actions plus current Galerkin coarse action; fixed within PCG",
        state_feedback="Factor maintenance to next operator step; no Krylov trigger in the policy",
        rendering_checks=dict(text_within_canvas=True),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={e:hashlib.sha256(OUT.with_suffix("."+e).read_bytes()).hexdigest() for e in ("pdf","svg","png")})
    OUT.with_suffix(".json").write_text(json.dumps(specification,indent=2)+"\n")
    print("Saved publication PDF, editable SVG and PNG:",OUT,flush=True)


if __name__=="__main__":
    main()

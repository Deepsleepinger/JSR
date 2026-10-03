#!/usr/bin/env python3
"""Render a portable GIF of actual FEM slices synchronized with measured logs."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation

ROOT = Path(__file__).resolve().parents[1]


def coefficient(nodes, z, sources, parameters):
    values = np.full(len(nodes), parameters["k_solid"])
    for x,y,z0,q in sources:
        distance = (nodes[:,0]-x)**2 + (nodes[:,1]-y)**2 + (z-z0)**2
        values += (parameters["k_melt"]-parameters["k_solid"])*np.exp(-distance/(2*parameters["r0"]**2))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-n", type=int, default=48)
    args = parser.parse_args()
    data = json.loads((ROOT / "results/dc_jsr_long_horizon_audit/animation_data.json").read_text())
    c = next(c for c in data["configurations"] if c["mesh_n"]==args.mesh_n)
    fig = plt.figure(figsize=(11,7.5))
    grid = fig.add_gridspec(3,3,height_ratios=[2.1,1,1],hspace=.42,wspace=.30)
    spatial_axes = [fig.add_subplot(grid[0,j]) for j in range(3)]
    fields, projections, highlights, labels = [], [], [], []
    for axis,geo in zip(spatial_axes,c["geometry"]["slices"]):
        nodes = np.array(geo["nodes"]);triangles = np.array(geo["triangles"])
        tri = Triangulation(nodes[:,0],nodes[:,1],triangles)
        values = coefficient(nodes,geo["z"],c["sources"][1],c["coefficient"])
        fields.append(axis.tripcolor(tri,facecolors=values[triangles].mean(axis=1),
                                     cmap="viridis",vmin=1,vmax=39,edgecolors=(.5,.5,.5,.28),linewidth=.10))
        projections.append(axis.scatter([],[],s=40,c="#d92f84",marker="x",linewidths=1.5))
        rects, texts = [], []
        for iy in range(2):
            for ix in range(2):
                rect = Rectangle((ix*.5,iy*.5),.5,.5,facecolor="none",edgecolor="#eb6c25",linewidth=1.5)
                axis.add_patch(rect);rects.append(rect)
                texts.append(axis.text(ix*.5+.25,iy*.5+.25,"",ha="center",va="center",fontsize=8,
                                       bbox=dict(facecolor="white",alpha=.7,edgecolor="none",pad=1)))
        highlights.append(rects);labels.append(texts)
        axis.set(xlim=(0,1),ylim=(0,1),xlabel="x",ylabel="y",title="Actual FEM slice z=%.2f" % geo["z"])
        axis.set_aspect("equal")
    fig.colorbar(fields[-1],ax=spatial_axes,fraction=.025,pad=.025,label="Prescribed coefficient kappa (not temperature)")
    budget = fig.add_subplot(grid[1,:]);iteration = fig.add_subplot(grid[2,:],sharex=budget)
    x = np.arange(1,65)
    budget.step(x,[s["k"] for s in c["steps"]],where="mid",color="#087f8c",label="DC-JSR K")
    budget.axhline(8,color="#9d5722",linestyle="--",label="Full K=8")
    budget.set(ylabel="Refreshed factors",ylim=(0,8.7),xlim=(.5,64.5))
    iteration.plot(x,[s["dc_iterations"] for s in c["steps"]],color="#087f8c",label="DC-JSR")
    iteration.plot(x,[s["full_iterations"] for s in c["steps"]],color="#9d5722",linestyle="--",label="Full")
    iteration.set(ylabel="PCG iterations",xlabel="Time step (four-run mean curves)")
    markers=[]
    for axis in (budget,iteration):
        for bound in (16.5,32.5,48.5):axis.axvline(bound,color=".6",linewidth=.8)
        for start in (8.5,24.5,40.5,56.5):axis.axvspan(start,start+4,color=".5",alpha=.08)
        markers.append(axis.axvline(1,color="#d92f84",linewidth=1.5))
        axis.grid(alpha=.15);axis.legend(loc="upper left",fontsize=8,frameon=False,ncol=2)
    header=fig.suptitle("",fontsize=11,y=.98)
    fig.subplots_adjust(left=.065,right=.94,bottom=.07,top=.88)
    def update(j):
        step=j+1;record=c["steps"][j]
        for plane,(geo,field,projection,rects,texts) in enumerate(zip(c["geometry"]["slices"],fields,projections,highlights,labels)):
            nodes=np.array(geo["nodes"]);triangles=np.array(geo["triangles"])
            values=coefficient(nodes,geo["z"],c["sources"][step],c["coefficient"])
            field.set_array(values[triangles].mean(axis=1))
            projection.set_offsets(np.array([s[:2] for s in c["sources"][step]]))
            for quadrant,(rect,text) in enumerate(zip(rects,texts)):
                ids=[quadrant+4*(geo["z"]>=.5)]
                if geo["z"]==.5:ids.append(quadrant)
                selected=[i for i in ids if i in record["selected"]]
                rect.set_visible(bool(selected));text.set_text("/".join(map(str,sorted(ids)))+(" R" if selected else ""))
        for marker in markers:marker.set_xdata([step,step])
        header.set_text("N=%d | %s DOFs | 4 MPI ranks | step %d/64 | K=%d | Vmax=%.4f\n" %
                        (c["mesh_n"],format(c["n_dofs"],","),step,record["k"],record["residual_proxy"])+
                        "R: refreshed core; pink x: source projection; factors include one overlap layer")
        if step%16==0:print("Rendered step",step,flush=True)
        return fields+projections+markers+[header]
    animation=FuncAnimation(fig,update,frames=64,interval=350,blit=False,repeat=False)
    output=ROOT / "docs/figures" / ("dc_jsr_long_horizon_mesh_n%d.gif" % args.mesh_n)
    animation.save(str(output),writer=PillowWriter(fps=3),dpi=95)
    update(8);fig.savefig(output.with_suffix(".png"),dpi=150)
    plt.close(fig)
    print("Saved",output,flush=True)


if __name__=="__main__":
    main()

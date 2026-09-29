#!/usr/bin/env python3
"""
================================================================================
CMAME Crucial Experiment 3: Long-Horizon Sequence Benchmark (T = 50 Steps)
Multi-Track Serpentine Laser Scanning on 3D Continuous Galerkin FEM
================================================================================

Evaluates the long-term temporal stability and cumulative wall-clock economics
of stateful preconditioner maintenance over a realistic 50-step multi-pass
reciprocating laser scan trajectory:

  Track 1 (Steps 1-16):  Forward pass along y = 0.25 (x: 0.20 -> 0.80)
  Track 2 (Steps 17-33): Backward pass along y = 0.50 (x: 0.80 -> 0.20)
  Track 3 (Steps 34-50): Forward pass along y = 0.75 (x: 0.20 -> 0.80)

Evaluates 5 strategic maintenance paradigms:
1. Full Rebuild: 100% factor reassembly at every step (linear cumulative baseline).
2. Frozen Static Reuse: 0% factor updates (exposing chronic spectral breakdown).
3. AsRAS-style Cyclic: Blind round-robin partial updates.
4. Svolos-style Physics-Aware: Updates subdomains containing active physical front.
5. JSR Stateful Algebraic: Proposed drift-proxy + persistent age + mass budget + joint coarse.

Outputs cumulative time trajectory T_cum(t) and iteration stability curves.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

os.environ.setdefault("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
os.environ.setdefault("PKG_CONFIG_PATH", "/mnt/h/CodexLinux/stokes-r3/envs/stokes-fenics-2020-abi6-r3/lib/pkgconfig")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dolfin as d
from petsc4py import PETSc
from jsr import partition as part_module
from jsr import backend_mumps as mumps_module
from benchmarks.run_same_budget_selector_ablation import refresh_coarse_vectorized


class PythonPC(object):
    def __init__(self, backend_ctx: mumps_module.OverlappingMUMPSSchwarzBackend):
        self.backend = backend_ctx

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        x_arr = x.getArray(readonly=True)
        y_arr = y.getArray()
        self.backend.apply(x_arr, y_arr)


def solve_pcg(
    mat: PETSc.Mat,
    backend_ctx: mumps_module.OverlappingMUMPSSchwarzBackend,
    b_vec: PETSc.Vec,
    x_vec: PETSc.Vec,
    rtol: float = 1.0e-8,
    max_it: int = 150,
) -> Tuple[int, float, float]:
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    ksp.setOperators(mat)
    ksp.setType("cg")
    ksp.setTolerances(rtol=rtol, atol=1.0e-14, max_it=max_it)

    pc = ksp.getPC()
    pc.setType("python")
    pc.setPythonContext(PythonPC(backend_ctx))
    pc.setUp()

    t0 = time.perf_counter()
    ksp.solve(b_vec, x_vec)
    t_solve = time.perf_counter() - t0
    its = int(ksp.getIterationNumber())

    r_vec = b_vec.duplicate()
    mat.mult(x_vec, r_vec)
    r_vec.aypx(-1.0, b_vec)
    norm_res = float(r_vec.norm(PETSc.NormType.NORM_2))
    norm_b = float(b_vec.norm(PETSc.NormType.NORM_2))
    rel_res = norm_res / max(norm_b, 1.0e-14)

    r_vec.destroy()
    ksp.destroy()
    return its, t_solve, rel_res


def get_laser_position(step: int, total_steps: int = 50) -> Tuple[float, float, float]:
    """Calculate (xc, yc, zc) for 3-pass serpentine scan track."""
    if step <= 16:
        # Track 1: y = 0.25, x moves 0.20 -> 0.80
        s = float(step) / 16.0
        xc = 0.20 + 0.60 * s
        yc = 0.25
    elif step <= 33:
        # Track 2: y = 0.50, x moves 0.80 -> 0.20 (backward)
        s = float(step - 17) / 16.0
        xc = 0.80 - 0.60 * s
        yc = 0.50
    else:
        # Track 3: y = 0.75, x moves 0.20 -> 0.80
        s = float(step - 34) / 16.0
        xc = 0.20 + 0.60 * s
        yc = 0.75
    zc = 1.0
    return xc, yc, zc


def assemble_step(
    mesh: d.Mesh,
    V: d.FunctionSpace,
    step: int,
    total_steps: int,
    dt: float,
    u_prev: d.Function,
    rho_cp: float = 10.0,
    k_solid: float = 1.0,
    k_melt: float = 10.0,
    r0: float = 0.10,
    d_pen: float = 0.18,
    q0: float = 60.0,
) -> Tuple[PETSc.Mat, PETSc.Vec, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = d.TrialFunction(V)
    v = d.TestFunction(V)

    xc, yc, zc = get_laser_position(step, total_steps)

    kappa_expr = d.Expression(
        "k_solid + (k_melt - k_solid) * exp(-((x[0]-xc)*(x[0]-xc) + (x[1]-yc)*(x[1]-yc))/(2.0*r0*r0)) * exp(-(1.0 - x[2])/d_pen)",
        k_solid=k_solid,
        k_melt=k_melt,
        xc=xc,
        yc=yc,
        r0=r0,
        d_pen=d_pen,
        degree=2,
    )

    heat_expr = d.Expression(
        "q0 * exp(-((x[0]-xc)*(x[0]-xc) + (x[1]-yc)*(x[1]-yc))/(r0*r0)) * exp(-(1.0 - x[2])/d_pen)",
        q0=q0,
        xc=xc,
        yc=yc,
        r0=r0,
        d_pen=d_pen,
        degree=2,
    )

    a = (kappa_expr * d.inner(d.grad(u), d.grad(v)) + d.Constant(rho_cp / dt) * u * v) * d.dx
    L = (d.Constant(rho_cp / dt) * u_prev + heat_expr) * v * d.dx

    bc = d.DirichletBC(V, d.Constant(0.0), "on_boundary && x[2] < 1.0e-4")

    A_dolfin, b_dolfin = d.assemble_system(a, L, bc)
    mat = d.as_backend_type(A_dolfin).mat()
    b_vec = d.as_backend_type(b_dolfin).vec()

    indptr, indices, data = mat.getValuesCSR()

    dof_coords = V.tabulate_dof_coordinates()
    r_sq = (dof_coords[:, 0] - xc) ** 2 + (dof_coords[:, 1] - yc) ** 2
    z_dist = 1.0 - dof_coords[:, 2]
    k_vals = k_solid + (k_melt - k_solid) * np.exp(-r_sq / (2.0 * r0 * r0)) * np.exp(-z_dist / d_pen)

    return mat, b_vec, indptr, indices, data, k_vals


def run_long_horizon(
    n_mesh: int = 28,
    total_steps: int = 50,
    sub_per_axis: int = 2,
    overlap: int = 1,
    dt: float = 0.05,
    budget_k: int = 3,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 100)
    print(f"   CMAME EXPERIMENT 3: LONG-HORIZON SEQUENCE BENCHMARK (T = {total_steps} STEPS)")
    print(f"   Mesh: UnitCubeMesh({n_mesh}, {n_mesh}, {n_mesh}) | Multi-Track Serpentine Scan")
    print(f"   Subdomains: {sub_per_axis**3} | Budget K: {budget_k} | dt: {dt}")
    print("=" * 100)

    mesh = d.UnitCubeMesh(n_mesh, n_mesh, n_mesh)
    V = d.FunctionSpace(mesh, "Lagrange", 1)
    n_dofs = V.dim()
    coords = V.tabulate_dof_coordinates()

    part = part_module.create_3d_overlapping_partition(
        coordinates=coords,
        grid=(sub_per_axis, sub_per_axis, sub_per_axis),
        overlap_layers=overlap,
        mesh_n=n_mesh,
    )
    n_sub = int(part["subdomain_count"])
    print(f"✓ DOFs: {n_dofs:,} | Subdomains: {n_sub} (mean size: {np.mean([len(idx) for idx in part['local_indices']]):.0f} DOFs)")

    arms = [
        "full_rebuild",
        "frozen_reuse",
        "cyclic_asras_style",
        "physics_svolos_style",
        "jsr_stateful",
    ]

    arm_descriptions = {
        "full_rebuild": "Full Rebuild (100% updates, standard practice)",
        "frozen_reuse": "Frozen Static Reuse (0% updates)",
        "cyclic_asras_style": f"Cyclic Partial Update (AsRAS-style, budget K={budget_k})",
        "physics_svolos_style": "Physics-Aware Localized Update (Svolos-style front threshold)",
        "jsr_stateful": "JSR Stateful Maintenance (Algebraic drift + Persistent age + Mass-alpha budget)",
    }

    all_trajectories: Dict[str, Dict[str, Any]] = {}

    for arm in arms:
        print(f"\n[{arm.upper()}]: {arm_descriptions[arm]}")
        backend = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)

        u_state = d.Function(V)
        u_state.vector()[:] = 0.0

        # Step 0 initial assembly
        mat0, b_vec0, ind0, indx0, d0, k0 = assemble_step(mesh, V, 0, total_steps, dt, u_state)
        diag0 = mat0.getDiagonal().getArray().copy()

        backend.refresh_local(mat0, 0, range(n_sub))
        refresh_coarse_vectorized(backend, ind0, indx0, d0, n_sub)

        sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)
        sol_vec.set(0.0)
        solve_pcg(mat0, backend, b_vec0, sol_vec)

        prev_mat = mat0
        prev_diag = diag0
        mat0.destroy()
        b_vec0.destroy()

        local_ages = [0] * n_sub
        cyclic_ptr = 0

        times_setup: List[float] = []
        times_solve: List[float] = []
        times_total: List[float] = []
        cum_times: List[float] = []
        iters_list: List[int] = []
        relres_list: List[float] = []
        updates_list: List[int] = []

        running_cum_time = 0.0

        for step in range(1, total_steps + 1):
            mat_t, b_t, ind_t, indx_t, d_t, k_t = assemble_step(mesh, V, step, total_steps, dt, u_state)
            diag_t = mat_t.getDiagonal().getArray().copy()

            t0 = time.perf_counter()

            # Strategy selection
            if arm == "full_rebuild":
                selected = list(range(n_sub))
            elif arm == "frozen_reuse":
                selected = []
            elif arm == "cyclic_asras_style":
                selected = [(cyclic_ptr + i) % n_sub for i in range(budget_k)]
                cyclic_ptr = (cyclic_ptr + budget_k) % n_sub
            elif arm == "physics_svolos_style":
                k_tol = 1.0 + 0.15 * (10.0 - 1.0)
                selected = [cid for cid, idx in enumerate(part["local_indices"]) if float(np.max(k_t[idx])) > k_tol]
            elif arm == "jsr_stateful":
                delta_diag = diag_t - prev_diag
                delta_norm = [float(np.linalg.norm(delta_diag[idx])) for idx in part["local_indices"]]
                scores = [delta_norm[cid] * (1.0 + 0.15 * local_ages[cid]) for cid in range(n_sub)]
                tot_score = sum(scores)
                ranked = sorted(range(n_sub), key=lambda i: -scores[i])
                selected = set()
                cum = 0.0
                for cid in ranked:
                    selected.add(cid)
                    cum += scores[cid]
                    if cum >= 0.85 * tot_score or len(selected) >= 4:
                        break
                # Anti-starvation
                if len(selected) < 4:
                    for cid in range(n_sub):
                        if local_ages[cid] >= 6 and cid not in selected:
                            selected.add(cid)
                            break
                selected = sorted(selected)

            updated = sorted(selected)
            t_loc = backend.refresh_local(mat_t, step, updated) if updated else 0.0

            if arm != "frozen_reuse":
                t_crs = refresh_coarse_vectorized(backend, ind_t, indx_t, d_t, n_sub)
            else:
                t_crs = 0.0

            t_setup = time.perf_counter() - t0

            for cid in range(n_sub):
                if cid in selected:
                    local_ages[cid] = 0
                else:
                    local_ages[cid] += 1

            sol_vec.set(0.0)
            its, t_sol, rel_res = solve_pcg(mat_t, backend, b_t, sol_vec)
            t_step = time.perf_counter() - t0
            running_cum_time += t_step

            times_setup.append(float(t_setup))
            times_solve.append(float(t_sol))
            times_total.append(float(t_step))
            cum_times.append(float(running_cum_time))
            iters_list.append(int(its))
            relres_list.append(float(rel_res))
            updates_list.append(int(len(updated)))

            prev_diag = diag_t
            mat_t.destroy()
            b_t.destroy()

            if step % 10 == 0 or step == total_steps:
                print(f"  Step {step:2d}/{total_steps}: CumTime={running_cum_time:.2f}s | "
                      f"LastStep: Setup={t_setup:.3f}s, Solve={t_sol:.3f}s, Iters={its:2d}, Updated={len(updated)}/{n_sub}")

        backend.destroy()

        all_trajectories[arm] = {
            "description": arm_descriptions[arm],
            "times_setup": times_setup,
            "times_solve": times_solve,
            "times_total": times_total,
            "cum_times": cum_times,
            "iters": iters_list,
            "relres": relres_list,
            "updates": updates_list,
            "total_wall_clock": running_cum_time,
            "mean_step_time": float(np.mean(times_total)),
            "mean_iters": float(np.mean(iters_list)),
            "max_iters": int(np.max(iters_list)),
        }

    # Summary table
    ref_total = all_trajectories["full_rebuild"]["total_wall_clock"]
    print("\n" + "=" * 105)
    print(f"{'Strategy Policy':<26} | {'Total Wall-Clock':<18} | {'Mean Step Time':<16} | {'Mean Iters':<12} | {'Max Iter':<10} | {'Speedup':<8}")
    print("-" * 105)
    for arm in arms:
        traj = all_trajectories[arm]
        spd = ref_total / traj["total_wall_clock"]
        traj["speedup"] = float(spd)
        print(f"{arm:<26} | {traj['total_wall_clock']:<18.2f}s | {traj['mean_step_time']:<16.4f}s | {traj['mean_iters']:<12.1f} | {traj['max_iters']:<10d} | {spd:<7.2f}x")
    print("=" * 105)

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump({
                "mesh_n": int(n_mesh),
                "total_dofs": int(n_dofs),
                "subdomains": int(n_sub),
                "total_steps": int(total_steps),
                "trajectories": all_trajectories,
            }, fp, indent=2)
        print(f"\n✓ Saved 50-step long-horizon results to: {out_file.resolve()}")

    sol_vec.destroy()
    return all_trajectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMAME Long-Horizon Sequence Benchmark")
    parser.add_argument("--mesh", type=int, default=28, help="Mesh resolution (default: 28 = 24,389 DOFs)")
    parser.add_argument("--steps", type=int, default=50, help="Total steps in serpentine trajectory (default: 50)")
    parser.add_argument("--subdomains", type=int, default=2, help="Subdomains per axis (default: 2 -> 8 subdomains)")
    parser.add_argument("--budget", type=int, default=3, help="Budget K for selective updates (default: 3)")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step size dt (default: 0.05)")
    parser.add_argument("--out", type=str, default="results/long_horizon_50steps_n28.json", help="Output path")
    args = parser.parse_args()

    run_long_horizon(
        n_mesh=args.mesh,
        total_steps=args.steps,
        sub_per_axis=args.subdomains,
        budget_k=args.budget,
        dt=args.dt,
        output_path=args.out,
    )

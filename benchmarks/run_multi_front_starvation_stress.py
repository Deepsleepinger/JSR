#!/usr/bin/env python3
"""
================================================================================
CMAME Anti-Starvation & Multi-Front Stress Benchmark: Worst-Case Tail Latency
Evaluates Stateful Preconditioner Maintenance under Multi-Beam Competition,
Thermal Re-visitation, and Starvation Risk Control.
================================================================================

Research Objective:
Demonstrate that memoryless/instantaneous selectors (such as instantaneous physics
peaks or memoryless algebraic drift) suffer severe tail-latency spikes (exceeding
100+ iterations) when thermal fronts cross, alternate, or re-visit previously
softened zones. In contrast, JSR's stateful persistence and age damping act as a
rigorous tail-latency risk control mechanism, bounding max_t N_iter(t).
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
from benchmarks.run_monitor_family_ablation import compute_monitor_scores
from benchmarks.run_subdomain_scaling_pressure import PythonPC, solve_pcg_certified


def multi_front_trajectory_sources(step: int, total_steps: int = 16) -> List[Tuple[float, float, float, float]]:
    """
    4-phase alternating multi-beam trajectory with periodic re-visitation:
    - Phase 1 (Steps 1-4): Laser 1 scans lower layer (z=0.25, x: 0.2 -> 0.8)
    - Phase 2 (Steps 5-8): Laser 2 scans upper layer (z=0.75, x: 0.8 -> 0.2)
    - Phase 3 (Steps 9-12): Intersecting dual-beam confrontation with central crossing
    - Phase 4 (Steps 13-16): Rapid corner pulses re-visiting previously softened zones
    """
    srcs = []
    if step <= 4:
        s = step / 4.0
        srcs.append((0.2 + 0.6 * s, 0.25 + 0.5 * s, 0.25, 90.0))
    elif step <= 8:
        s = (step - 4) / 4.0
        srcs.append((0.8 - 0.6 * s, 0.75 - 0.5 * s, 0.75, 90.0))
    elif step <= 12:
        s = (step - 8) / 4.0
        srcs.append((0.2 + 0.6 * s, 0.2 + 0.6 * s, 0.5, 75.0))
        srcs.append((0.8 - 0.6 * s, 0.2 + 0.6 * s, 0.5, 75.0))
    else:
        # Rapid alternating corner pulses
        if step % 2 == 1:
            srcs.append((0.2, 0.25, 0.25, 95.0))
            srcs.append((0.8, 0.75, 0.75, 95.0))
        else:
            srcs.append((0.2, 0.75, 0.75, 95.0))
            srcs.append((0.8, 0.25, 0.25, 95.0))
    return srcs


def assemble_multi_front_fem_step(
    mesh: d.Mesh,
    V: d.FunctionSpace,
    step: int,
    total_steps: int,
    dt: float,
    u_prev: d.Function,
    rho_cp: float = 1.0,
    k_solid: float = 1.0,
    k_melt: float = 20.0,
    r0: float = 0.12,
) -> Tuple[PETSc.Mat, PETSc.Vec, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = d.TrialFunction(V)
    v = d.TestFunction(V)
    srcs = multi_front_trajectory_sources(step, total_steps)

    kappa_parts = []
    heat_parts = []
    for (xc, yc, zc, q0) in srcs:
        dist_sq = f"((x[0]-{xc})*(x[0]-{xc}) + (x[1]-{yc})*(x[1]-{yc}) + (x[2]-{zc})*(x[2]-{zc}))"
        kappa_parts.append(f"exp(-{dist_sq}/(2.0*{r0*r0}))")
        heat_parts.append(f"{q0} * exp(-{dist_sq}/({r0*r0}))")

    k_expr_str = f"{k_solid} + ({k_melt} - {k_solid}) * (" + " + ".join(kappa_parts) + ")"
    h_expr_str = " + ".join(heat_parts)

    kappa_expr = d.Expression(k_expr_str, degree=2)
    heat_expr = d.Expression(h_expr_str, degree=2)

    a = (kappa_expr * d.inner(d.grad(u), d.grad(v)) + d.Constant(rho_cp / dt) * u * v) * d.dx
    L = (d.Constant(rho_cp / dt) * u_prev + heat_expr) * v * d.dx
    bc = d.DirichletBC(V, d.Constant(0.0), "on_boundary && (x[0] < 1e-4 || x[1] < 1e-4)")

    A_d, b_d = d.assemble_system(a, L, bc)
    mat = d.as_backend_type(A_d).mat()
    b_vec = d.as_backend_type(b_d).vec()
    indptr, indices, data = mat.getValuesCSR()

    dof_coords = V.tabulate_dof_coordinates()
    k_vals = np.full(V.dim(), k_solid)
    for (xc, yc, zc, q0) in srcs:
        r_sq = (dof_coords[:, 0] - xc)**2 + (dof_coords[:, 1] - yc)**2 + (dof_coords[:, 2] - zc)**2
        k_vals += (k_melt - k_solid) * np.exp(-r_sq / (2.0 * r0 * r0))

    return mat, b_vec, indptr, indices, data, k_vals


def run_multi_front_benchmark(
    mesh_n: int = 24,
    total_steps: int = 16,
    dt: float = 0.05,
    rho_cp: float = 1.0,
    budget_k: int = 3,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 115)
    print(f"   CMAME ANTI-STARVATION & MULTI-FRONT STRESS BENCHMARK (T = {total_steps} STEPS)")
    print(f"   Mesh: UnitCubeMesh({mesh_n}, {mesh_n}, {mesh_n}) | Budget K: {budget_k}/8 | dt: {dt} | rho_cp: {rho_cp}")
    print(f"   Certified True Residual: ||b - A x|| / ||b|| < 1.0e-8")
    print("=" * 115)

    mesh = d.UnitCubeMesh(mesh_n, mesh_n, mesh_n)
    V = d.FunctionSpace(mesh, "Lagrange", 1)
    n_dofs = V.dim()
    coords = V.tabulate_dof_coordinates()

    print(f"--> Preassembling all {total_steps} multi-front FEM steps (total DOFs = {n_dofs})...")
    mats, b_vecs, csrs, diags, k_fields = [], [], [], [], []
    u_prev = d.Function(V)
    u_prev.vector()[:] = 0.0

    t_asm_start = time.perf_counter()
    for step in range(total_steps + 1):
        mat, b_vec, indptr, indices, data, k_vals = assemble_multi_front_fem_step(
            mesh, V, step, total_steps, dt, u_prev, rho_cp=rho_cp
        )
        diag = mat.getDiagonal().getArray().copy()
        mats.append(mat)
        b_vecs.append(b_vec)
        csrs.append((indptr, indices, data))
        diags.append(diag)
        k_fields.append(k_vals)
    print(f"    Assembled {total_steps + 1} steps in {time.perf_counter() - t_asm_start:.2f}s.\n")

    part = part_module.create_3d_overlapping_partition(
        coordinates=coords,
        grid=(2, 2, 2),
        overlap_layers=1,
        mesh_n=mesh_n,
    )
    n_sub = int(part["subdomain_count"])

    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)

    arms = [
        ("svolos_physics_peak", "Svolos-Style Physics Peak (Memoryless)", "physics", False),
        ("jsr_memoryless_drift", "Memoryless Drift-Only (lambda = 0)", "drift_step", False),
        ("jsr_stateful_step", "JSR Stateful Step Drift (lambda = 0.15)", "drift_step", True),
        ("jsr_stateful_cum", "JSR Stateful Cumulative Drift (lambda = 0.15)", "drift_cum", True),
        ("full_rebuild", "Full Rebuild Baseline (100% every step)", "full", False),
    ]

    records = []

    print("-" * 115)
    print(f"{'Method':<40}{'Mean Iters':<12}{'Max Iters':<12}{'Max Age':<10}{'T_solve(s)':<12}{'T_total(s)':<12}{'RelRes':<10}")
    print("-" * 115)

    for arm_id, arm_label, mode, use_age in arms:
        backend = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
        backend.refresh_local(mats[0], 0, range(n_sub))
        ind0, indx0, d0 = csrs[0]
        refresh_coarse_vectorized(backend, ind0, indx0, d0, n_sub)

        ages = [0] * n_sub
        factor_diags = [diags[0][idx].copy() for idx in part["local_indices"]]

        times_setup, times_solve, times_total = [], [], []
        iters_list, res_list, max_ages_list = [], [], []

        for step in range(1, total_steps + 1):
            mat_t = mats[step]
            b_t = b_vecs[step]
            ind_t, indx_t, d_t = csrs[step]
            diag_t = diags[step]
            prev_diag = diags[step - 1]
            delta_diag = diag_t - prev_diag

            t0 = time.perf_counter()

            if mode == "full":
                selected = list(range(n_sub))
            elif mode == "physics":
                raw = [float(np.max(k_fields[step][idx])) for idx in part["local_indices"]]
                selected = sorted(sorted(range(n_sub), key=lambda i: -raw[i])[:budget_k])
            elif mode == "drift_step":
                raw = compute_monitor_scores("jsr_rel_l2_k3", delta_diag, diag_t, prev_diag, part["local_indices"], None)
                if use_age:
                    scores = [raw[i] * (1.0 + 0.15 * ages[i]) for i in range(n_sub)]
                else:
                    scores = raw
                selected = sorted(sorted(range(n_sub), key=lambda i: -scores[i])[:budget_k])
            elif mode == "drift_cum":
                scores = []
                for i, idx in enumerate(part["local_indices"]):
                    diff = np.abs(diag_t[idx] - factor_diags[i])
                    denom = np.maximum(np.abs(factor_diags[i]), 1e-12)
                    cum_d = float(np.linalg.norm(diff / denom) / np.sqrt(len(idx)))
                    if use_age:
                        scores.append(cum_d * (1.0 + 0.15 * ages[i]))
                    else:
                        scores.append(cum_d)
                selected = sorted(sorted(range(n_sub), key=lambda i: -scores[i])[:budget_k])
            else:
                raise ValueError(f"Unknown mode: {mode}")

            assert len(selected) == (n_sub if mode == "full" else budget_k)

            # Update age and persistent factor snapshots
            for cid in range(n_sub):
                if cid in selected:
                    ages[cid] = 0
                    factor_diags[cid] = diag_t[part["local_indices"][cid]].copy()
                else:
                    ages[cid] += 1

            t_loc = backend.refresh_local(mat_t, step, selected)
            t_crs = refresh_coarse_vectorized(backend, ind_t, indx_t, d_t, n_sub)
            t_setup = time.perf_counter() - t0

            sol_vec.set(0.0)
            its, t_sol, rel_res = solve_pcg_certified(mat_t, backend, b_t, sol_vec, max_it=400)
            assert rel_res < 1.0e-8, f"Residual certification failure: {rel_res} >= 1.0e-8"

            t_step = t_setup + t_sol
            times_setup.append(float(t_setup))
            times_solve.append(float(t_sol))
            times_total.append(float(t_step))
            iters_list.append(int(its))
            res_list.append(float(rel_res))
            max_ages_list.append(int(max(ages)))

        backend.destroy()

        mean_iters = float(np.mean(iters_list))
        max_iters = int(np.max(iters_list))
        max_age = int(np.max(max_ages_list))
        tot_solve = float(np.sum(times_solve))
        tot_setup = float(np.sum(times_setup))
        tot_time = float(np.sum(times_total))
        max_res = float(np.max(res_list))

        rec = {
            "arm_id": arm_id,
            "label": arm_label,
            "mean_iters": mean_iters,
            "max_iters": max_iters,
            "max_age": max_age,
            "iters_history": iters_list,
            "total_solve_sec": tot_solve,
            "total_setup_sec": tot_setup,
            "total_wall_sec": tot_time,
            "mean_solve_sec": float(np.mean(times_solve)),
            "max_rel_res": max_res,
        }
        records.append(rec)

        print(
            f"{arm_label:<40}{mean_iters:<12.1f}{max_iters:<12}{max_age:<10}"
            f"{tot_solve:<12.3f}{tot_time:<12.3f}{max_res:<10.2e}"
        )

    print("-" * 115)
    sol_vec.destroy()
    for m in mats:
        m.destroy()
    for b in b_vecs:
        b.destroy()

    if output_path is None:
        output_path = str(ROOT / "results" / "multi_front_starvation_stress.json")
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n✓ Multi-front anti-starvation stress records successfully saved to:\n  {out_file}")
    return {"records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMAME Multi-Front Anti-Starvation Stress Benchmark")
    parser.add_argument("--mesh-n", type=int, default=24, help="Grid resolution per axis")
    parser.add_argument("--steps", type=int, default=16, help="Number of transient steps")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step size")
    parser.add_argument("--rho-cp", type=float, default=1.0, help="Volumetric heat capacity")
    parser.add_argument("--budget-k", type=int, default=3, help="Subdomain refresh budget per step")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    run_multi_front_benchmark(
        mesh_n=args.mesh_n,
        total_steps=args.steps,
        dt=args.dt,
        rho_cp=args.rho_cp,
        budget_k=args.budget_k,
        output_path=args.output,
    )

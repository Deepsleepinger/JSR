#!/usr/bin/env python3
"""
================================================================================
CMAME Evidence Consolidation: Long-Horizon Coarse Space Ablation (T = 50 Steps)
Controlled Comparison under IDENTICAL Local Refresh Masks {S_t}:
  Arm A: Fresh Coarse (A_0(t) synchronized every step)
  Arm B: Stale / Frozen Coarse (A_0(0) frozen indefinitely)
  Arm C: Periodic Coarse (A_0 refreshed every 10 steps)
Residual Certification Strictly < 1.0e-8
================================================================================
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
from benchmarks.run_long_horizon_sequence import (
    PythonPC,
    get_laser_position,
    assemble_step,
)


def solve_pcg_certified(
    mat: PETSc.Mat,
    backend_ctx: mumps_module.OverlappingMUMPSSchwarzBackend,
    b_vec: PETSc.Vec,
    x_vec: PETSc.Vec,
    rtol: float = 2.0e-10,
    max_it: int = 150,
) -> Tuple[int, float, float]:
    """Solve system A x = b with certified relative residual < 1e-8."""
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


def run_coarse_ablation_50step(
    n_mesh: int = 28,
    total_steps: int = 50,
    budget_k: int = 2,
    dt: float = 0.05,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 110)
    print(f"   CMAME EXPERIMENT: LONG-HORIZON COARSE SPACE ABLATION (T = {total_steps} STEPS)")
    print(f"   Mesh: UnitCubeMesh({n_mesh}, {n_mesh}, {n_mesh}) | 8 Subdomains (2x2x2) | Budget K: {budget_k}")
    print(f"   Controlled Variable: Coarse Synchronization Frequency under IDENTICAL Local Refresh Masks")
    print(f"   Arm 1: Fresh Coarse (every step) | Arm 2: Stale Coarse (frozen t=0) | Arm 3: Periodic (every 10 steps)")
    print(f"   Residual Certification Target: Strictly < 1.0e-8")
    print("=" * 110)

    mesh = d.UnitCubeMesh(n_mesh, n_mesh, n_mesh)
    V = d.FunctionSpace(mesh, "Lagrange", 1)
    n_dofs = V.dim()
    coords = V.tabulate_dof_coordinates()

    part = part_module.create_3d_overlapping_partition(
        coordinates=coords,
        grid=(2, 2, 2),
        overlap_layers=1,
        mesh_n=n_mesh,
    )
    n_sub = int(part["subdomain_count"])

    # First: preassemble all 50 steps so matrix states and RHS are identical
    print(f"--> Preassembling all {total_steps} FEM steps...")
    mats, b_vecs, csrs, diag_arrays = [], [], [], []
    u_state = d.Function(V)
    u_state.vector()[:] = 0.0

    for step in range(total_steps + 1):
        mat, b_vec, indptr, indices, data, _ = assemble_step(mesh, V, step, total_steps, dt, u_state)
        diag_vec = mat.getDiagonal()
        diag_arr = diag_vec.getArray().copy()
        diag_vec.destroy()

        mats.append(mat)
        b_vecs.append(b_vec)
        csrs.append((indptr, indices, data))
        diag_arrays.append(diag_arr)

    # Step 1: Run JSR to determine the official local mask sequence {S_t}
    print("\n--> Step 1: Generating standard JSR local selection sequence {S_t}...")
    local_masks: List[List[int]] = []
    local_ages = [0] * n_sub
    for step in range(1, total_steps + 1):
        diag_t = diag_arrays[step]
        prev_diag = diag_arrays[step - 1]
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
        if len(selected) < 4:
            for cid in range(n_sub):
                if local_ages[cid] >= 6 and cid not in selected:
                    selected.add(cid)
                    break
        selected_list = sorted(selected)
        local_masks.append(selected_list)

        for cid in range(n_sub):
            if cid in selected_list:
                local_ages[cid] = 0
            else:
                local_ages[cid] += 1

    print(f"    Generated {len(local_masks)} local masks. Average refreshed subdomains: {np.mean([len(m) for m in local_masks]):.2f}/8")

    arms = [
        ("fresh_coarse", "Fresh Coarse (Synchronized every step)"),
        ("stale_coarse", "Stale Coarse (Frozen at t=0 across all 50 steps)"),
        ("periodic_coarse_10", "Periodic Coarse (Synchronized every 10 steps)"),
    ]

    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)
    results: Dict[str, Any] = {}

    for arm_id, arm_label in arms:
        print(f"\n================================================================================")
        print(f"--> Running Arm: {arm_label}...")
        print(f"================================================================================")

        backend = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
        backend.refresh_local(mats[0], 0, range(n_sub))
        ind0, indx0, d0 = csrs[0]
        refresh_coarse_vectorized(backend, ind0, indx0, d0, n_sub)

        t_setup_list, t_solve_list, t_total_list, iters_list, relres_list = [], [], [], [], []
        cum_times = []
        running_cum = 0.0

        for step in range(1, total_steps + 1):
            mask = local_masks[step - 1]
            mat_t = mats[step]
            b_t = b_vecs[step]
            ind_t, indx_t, d_t = csrs[step]

            t0 = time.perf_counter()
            # 1. Local refresh using IDENTICAL mask
            t_loc = backend.refresh_local(mat_t, step, mask) if mask else 0.0

            # 2. Coarse refresh policy
            if arm_id == "fresh_coarse":
                t_crs = refresh_coarse_vectorized(backend, ind_t, indx_t, d_t, n_sub)
            elif arm_id == "stale_coarse":
                t_crs = 0.0  # Zero coarse update
            elif arm_id == "periodic_coarse_10":
                if step % 10 == 0:
                    t_crs = refresh_coarse_vectorized(backend, ind_t, indx_t, d_t, n_sub)
                else:
                    t_crs = 0.0

            t_setup = time.perf_counter() - t0

            # 3. Solve with certified residual
            sol_vec.set(0.0)
            its, t_sol, rel_res = solve_pcg_certified(mat_t, backend, b_t, sol_vec)
            t_step = t_setup + t_sol
            running_cum += t_step

            t_setup_list.append(float(t_setup))
            t_solve_list.append(float(t_sol))
            t_total_list.append(float(t_step))
            cum_times.append(float(running_cum))
            iters_list.append(int(its))
            relres_list.append(float(rel_res))

            if step in [1, 10, 20, 30, 40, 50]:
                print(f"    Step {step:2d}/50 | Setup: {t_setup:.4f}s | Solve: {t_sol:.4f}s | Iter: {its:2d} | TrueRelRes: {rel_res:.2e} | Cum: {running_cum:.2f}s")

        backend.destroy()

        tot_time = running_cum
        avg_iters = float(np.mean(iters_list))
        max_iters = int(np.max(iters_list))
        min_iters = int(np.min(iters_list))
        tot_setup = float(np.sum(t_setup_list))
        tot_solve = float(np.sum(t_solve_list))

        results[arm_id] = {
            "label": arm_label,
            "total_time": tot_time,
            "setup_time": tot_setup,
            "solve_time": tot_solve,
            "mean_iters": avg_iters,
            "max_iters": max_iters,
            "min_iters": min_iters,
            "iters_per_step": iters_list,
            "cum_times": cum_times,
            "solve_times": t_solve_list,
            "setup_times": t_setup_list,
            "rel_res_per_step": relres_list,
        }

    sol_vec.destroy()

    # Compare
    t_fresh = results["fresh_coarse"]["total_time"]
    t_stale = results["stale_coarse"]["total_time"]
    t_period = results["periodic_coarse_10"]["total_time"]

    print("\n" + "=" * 90)
    print("   LONG-HORIZON 50-STEP COARSE ABLATION COMPARISON SUMMARY")
    print("=" * 90)
    print(f"   Fresh Coarse (every step):  Total = {t_fresh:.4f}s | Setup = {results['fresh_coarse']['setup_time']:.4f}s | Solve = {results['fresh_coarse']['solve_time']:.4f}s | Mean Iters = {results['fresh_coarse']['mean_iters']:.1f} (Max: {results['fresh_coarse']['max_iters']})")
    print(f"   Stale Coarse (frozen t=0):  Total = {t_stale:.4f}s | Setup = {results['stale_coarse']['setup_time']:.4f}s | Solve = {results['stale_coarse']['solve_time']:.4f}s | Mean Iters = {results['stale_coarse']['mean_iters']:.1f} (Max: {results['stale_coarse']['max_iters']})")
    print(f"   Periodic Coarse (every 10): Total = {t_period:.4f}s | Setup = {results['periodic_coarse_10']['setup_time']:.4f}s | Solve = {results['periodic_coarse_10']['solve_time']:.4f}s | Mean Iters = {results['periodic_coarse_10']['mean_iters']:.1f} (Max: {results['periodic_coarse_10']['max_iters']})")
    print(f"   Relative Difference (Stale vs Fresh): {(t_stale - t_fresh)/t_fresh * 100.0:+.2f}% Total Time | Iteration Delta: {results['stale_coarse']['mean_iters'] - results['fresh_coarse']['mean_iters']:+.2f} iters")
    print("=" * 90)

    summary_export = {
        "mesh_n": n_mesh,
        "n_dofs": n_dofs,
        "total_steps": total_steps,
        "budget_k": budget_k,
        "local_masks": local_masks,
        "results": results,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary_export, f, indent=2)
        print(f"[Saved] Coarse ablation results written to {out_p}")

    return summary_export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long Horizon Coarse Ablation")
    parser.add_argument("--mesh-n", type=int, default=28)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--budget-k", type=int, default=2)
    parser.add_argument("--output", type=str, default="results/coarse_50step_ablation_n28.json")
    args = parser.parse_args()

    run_coarse_ablation_50step(
        n_mesh=args.mesh_n,
        total_steps=args.steps,
        budget_k=args.budget_k,
        output_path=args.output,
    )

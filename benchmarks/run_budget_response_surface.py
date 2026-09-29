#!/usr/bin/env python3
"""
================================================================================
Stage A Experiment: Budget Response Surface T(K) on Multi-Front Stress
Sweeps K in {0, 1, 2, 3, 4, 5, 6, 7, 8} on the identical 16-step
multi-beam alternating scan trajectory (N=24, 8 subdomains).

Identifies the Empirical Budget Oracle:
    K* = argmin_K T(K)
and isolates the exact transition from starvation under-budgeting to
full rebuild parity under certified true residual < 1.0e-8.
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
from benchmarks.run_monitor_family_ablation import compute_monitor_scores
from benchmarks.run_subdomain_scaling_pressure import solve_pcg_certified
from benchmarks.run_multi_front_starvation_stress import (
    multi_front_trajectory_sources,
    assemble_multi_front_fem_step,
)


def run_budget_sweep(
    mesh_n: int = 24,
    total_steps: int = 16,
    dt: float = 0.05,
    rho_cp: float = 1.0,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 115)
    print(f"   STAGE A: BUDGET RESPONSE SURFACE T(K) SWEEP ON MULTI-FRONT STRESS")
    print(f"   Mesh: UnitCubeMesh({mesh_n}, {mesh_n}, {mesh_n}) | Steps: {total_steps} | dt: {dt} | rho_cp: {rho_cp}")
    print(f"   Budget Sweep: K in {{0, 1, 2, 3, 4, 5, 6, 7, 8}} (K=0: Static, K=8: Full Rebuild)")
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
    assert n_sub == 8

    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)

    k_values = list(range(n_sub + 1))  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    records = []

    print("-" * 115)
    print(f"{'K':<5}{'Update %':<11}{'Mean Iters':<12}{'Max Iters':<12}{'Max Age':<10}{'T_setup(s)':<12}{'T_solve(s)':<12}{'T_total(s)':<12}{'T(K)/T_Full':<12}{'RelRes':<10}")
    print("-" * 115)

    # First get T_Full for reference
    t_full_ref = None

    for k in k_values:
        backend = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
        backend.refresh_local(mats[0], 0, range(n_sub))
        ind0, indx0, d0 = csrs[0]
        refresh_coarse_vectorized(backend, ind0, indx0, d0, n_sub)

        ages = [0] * n_sub
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

            if k == 0:
                selected = []
            elif k == n_sub:
                selected = list(range(n_sub))
            else:
                raw = compute_monitor_scores("jsr_rel_l2_k3", delta_diag, diag_t, prev_diag, part["local_indices"], None)
                scores = [raw[i] * (1.0 + 0.15 * ages[i]) for i in range(n_sub)]
                selected = sorted(sorted(range(n_sub), key=lambda i: -scores[i])[:k])
                assert len(selected) == k

            for cid in range(n_sub):
                if cid in selected:
                    ages[cid] = 0
                else:
                    ages[cid] += 1

            t_loc = backend.refresh_local(mat_t, step, selected)
            t_crs = refresh_coarse_vectorized(backend, ind_t, indx_t, d_t, n_sub)
            t_setup = time.perf_counter() - t0

            sol_vec.set(0.0)
            its, t_sol, rel_res = solve_pcg_certified(mat_t, backend, b_t, sol_vec, max_it=500)
            assert rel_res < 1.0e-8, f"Residual certification failure at K={k}, step={step}: {rel_res} >= 1.0e-8"

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
        tot_setup = float(np.sum(times_setup))
        tot_solve = float(np.sum(times_solve))
        tot_time = float(np.sum(times_total))
        max_res = float(np.max(res_list))

        if k == n_sub:
            t_full_ref = tot_time

        rec = {
            "K": k,
            "refresh_ratio": k / float(n_sub),
            "mean_iters": mean_iters,
            "max_iters": max_iters,
            "max_age": max_age,
            "iters_history": iters_list,
            "total_setup_sec": tot_setup,
            "total_solve_sec": tot_solve,
            "total_wall_sec": tot_time,
            "mean_solve_sec": float(np.mean(times_solve)),
            "mean_setup_sec": float(np.mean(times_setup)),
            "max_rel_res": max_res,
        }
        records.append(rec)

    # Compute T(K)/T_Full ratio
    for rec in records:
        ratio = rec["total_wall_sec"] / t_full_ref
        rec["ratio_vs_full"] = ratio
        pct_update = f"{rec['K']}/8 ({rec['refresh_ratio']*100:.1f}%)"
        print(
            f"{rec['K']:<5}{pct_update:<11}{rec['mean_iters']:<12.1f}{rec['max_iters']:<12}{rec['max_age']:<10}"
            f"{rec['total_setup_sec']:<12.3f}{rec['total_solve_sec']:<12.3f}{rec['total_wall_sec']:<12.3f}"
            f"{ratio:<12.3f}{rec['max_rel_res']:<10.2e}"
        )

    # Find empirical budget oracle K*
    best_rec = min(records, key=lambda r: r["total_wall_sec"])
    k_star = best_rec["K"]
    print("-" * 115)
    print(f"==> EMPIRICAL BUDGET ORACLE on Multi-Front Stress: K* = {k_star}/8 (T = {best_rec['total_wall_sec']:.3f}s)")
    print(f"    Fixed K=3 penalty: T(3) = {records[3]['total_wall_sec']:.3f}s vs T(K*) = {best_rec['total_wall_sec']:.3f}s (+{(records[3]['total_wall_sec']/best_rec['total_wall_sec'] - 1.0)*100:.1f}% overhead)")
    print(f"    Full Rebuild:      T(8) = {records[8]['total_wall_sec']:.3f}s")
    print("-" * 115)

    sol_vec.destroy()
    for m in mats:
        m.destroy()
    for b in b_vecs:
        b.destroy()

    if output_path is None:
        output_path = str(ROOT / "results" / "budget_response_surface_multifront.json")
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"k_star": k_star, "records": records}, f, indent=2)

    print(f"\n✓ Budget response surface records archived to: {out_file}\n")
    return {"k_star": k_star, "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Budget Response Surface T(K) Sweep")
    parser.add_argument("--mesh-n", type=int, default=24)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--rho-cp", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_budget_sweep(
        mesh_n=args.mesh_n,
        total_steps=args.steps,
        dt=args.dt,
        rho_cp=args.rho_cp,
        output_path=args.output,
    )

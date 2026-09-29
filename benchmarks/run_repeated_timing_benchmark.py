#!/usr/bin/env python3
"""
================================================================================
CMAME Evidence Consolidation: Statistical Repeatability Benchmark
Evaluates Multi-Run Wall-Clock Variance (Mean +/- Std) under
Strictly Equalized Refresh-Count Budget K = 3 and Certified Residual < 1.0e-8

Arms Evaluated:
1. Svolos-Inspired Physics Peak Baseline (instantaneous peak selector max_{Omega_i} kappa)
2. JSR-L2 (Absolute Euclidean accumulation ||Delta diag(A_i)||_2)
3. JSR-relL2 (Normalized relative distributed drift, Proposed Flagship)
4. JSR-relLinf-sym (Symmetric relative localized peak)
5. JSR-Hybrid (Dual-channel peak + distributed drift)
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
from benchmarks.run_long_horizon_sequence import assemble_step as assemble_serpentine_step
from benchmarks.run_case_b_complex_trajectory import assemble_dual_beam_step
from benchmarks.run_monitor_family_ablation import solve_pcg_certified, compute_monitor_scores


def run_repeated_trials(
    benchmark_type: str = "case_b_dual_beam",
    n_mesh: int = 28,
    total_steps: int = 12,
    budget_k: int = 3,
    n_repeats: int = 5,
    dt: float = 0.05,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 110)
    print(f"   CMAME STATISTICAL REPEATABILITY SUITE ({benchmark_type.upper()})")
    print(f"   Mesh: UnitCubeMesh({n_mesh}, {n_mesh}, {n_mesh}) | Steps: {total_steps} | Repeats: {n_repeats}")
    print(f"   Protocol: Equalized Refresh-Count Budget K = {budget_k}/8 | Certified True Residual < 1.0e-8")
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

    print(f"--> Preassembling all {total_steps} FEM steps for {benchmark_type}...")
    mats, b_vecs, csrs, diag_arrays, k_fields = [], [], [], [], []
    u_state = d.Function(V)
    u_state.vector()[:] = 0.0

    for step in range(total_steps + 1):
        if benchmark_type == "serpentine_50":
            mat, b_vec, indptr, indices, data, k_vals = assemble_serpentine_step(
                mesh, V, step, total_steps, dt, u_state
            )
        elif benchmark_type == "case_b_dual_beam":
            mat, b_vec, indptr, indices, data, k_vals = assemble_dual_beam_step(
                mesh, V, step, total_steps, dt, u_state
            )
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_type}")

        diag_vec = mat.getDiagonal()
        diag_arr = diag_vec.getArray().copy()
        diag_vec.destroy()

        mats.append(mat)
        b_vecs.append(b_vec)
        csrs.append((indptr, indices, data))
        diag_arrays.append(diag_arr)
        k_fields.append(k_vals)

    monitors = [
        ("svolos_physics_k3", "Svolos-Inspired Physics Peak Baseline"),
        ("jsr_l2_k3", "JSR-L2 (Absolute Euclidean Accumulation)"),
        ("jsr_rel_l2_k3", "JSR-relL2 (Normalized Relative Distributed Drift)"),
        ("jsr_rel_linf_sym_k3", "JSR-relLinf-sym (Symmetric Relative Peak)"),
        ("jsr_hybrid_k3", "JSR-Hybrid (Dual-Channel Peak + Distributed)"),
    ]

    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)
    results: Dict[str, Any] = {}

    for arm_id, arm_label in monitors:
        print(f"\n--> Evaluating arm: {arm_label} across {n_repeats} independent trials...")
        run_totals, run_setups, run_solves = [], [], []
        run_mean_iters, run_max_iters = [], []

        for trial in range(1, n_repeats + 1):
            backend = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
            backend.refresh_local(mats[0], 0, range(n_sub))
            ind0, indx0, d0 = csrs[0]
            refresh_coarse_vectorized(backend, ind0, indx0, d0, n_sub)

            local_ages = [0] * n_sub
            times_setup, times_solve, times_total = [], [], []
            iters_list, masks_history = [], []

            for step in range(1, total_steps + 1):
                mat_t = mats[step]
                b_t = b_vecs[step]
                ind_t, indx_t, d_t = csrs[step]
                diag_t = diag_arrays[step]
                prev_diag = diag_arrays[step - 1]
                delta_diag = diag_t - prev_diag
                k_t = k_fields[step]

                t0 = time.perf_counter()

                # 1. Compute monitor scores
                raw_scores = compute_monitor_scores(
                    arm_id, delta_diag, diag_t, prev_diag, part["local_indices"], k_t
                )

                # 2. Selection under equalized budget K = 3
                if arm_id == "svolos_physics_k3":
                    selected = sorted(sorted(range(n_sub), key=lambda i: -raw_scores[i])[:budget_k])
                else:
                    scores = [raw_scores[i] * (1.0 + 0.15 * local_ages[i]) for i in range(n_sub)]
                    selected = sorted(sorted(range(n_sub), key=lambda i: -scores[i])[:budget_k])

                # Rigorous check: enforce |S_t| == budget_k
                assert len(selected) == budget_k, f"Violation: |S_t| = {len(selected)} != {budget_k}"
                masks_history.append(selected)

                # 3. Factorization and coarse synchronization
                t_loc = backend.refresh_local(mat_t, step, selected)
                t_crs = refresh_coarse_vectorized(backend, ind_t, indx_t, d_t, n_sub)
                t_setup = time.perf_counter() - t0

                # 4. Age update
                for cid in range(n_sub):
                    if cid in selected:
                        local_ages[cid] = 0
                    else:
                        local_ages[cid] += 1

                # 5. PCG certified solve (< 1e-8)
                sol_vec.set(0.0)
                its, t_sol, rel_res = solve_pcg_certified(mat_t, backend, b_t, sol_vec)
                assert rel_res < 1.0e-8, f"Residual certification failure: {rel_res} >= 1.0e-8"

                t_step = t_setup + t_sol
                times_setup.append(float(t_setup))
                times_solve.append(float(t_sol))
                times_total.append(float(t_step))
                iters_list.append(int(its))

            backend.destroy()

            tot_time = float(sum(times_total))
            tot_setup = float(sum(times_setup))
            tot_solve = float(sum(times_solve))
            mean_its = float(np.mean(iters_list))
            max_its = int(np.max(iters_list))

            run_totals.append(tot_time)
            run_setups.append(tot_setup)
            run_solves.append(tot_solve)
            run_mean_iters.append(mean_its)
            run_max_iters.append(max_its)

            print(f"    Trial {trial}/{n_repeats}: Total = {tot_time:.4f}s | Setup = {tot_setup:.4f}s | Solve = {tot_solve:.4f}s | Iters = {mean_its:.2f} (Max: {max_its})")

        mean_tot = float(np.mean(run_totals))
        std_tot = float(np.std(run_totals, ddof=1)) if n_repeats > 1 else 0.0
        mean_set = float(np.mean(run_setups))
        std_set = float(np.std(run_setups, ddof=1)) if n_repeats > 1 else 0.0
        mean_sol = float(np.mean(run_solves))
        std_sol = float(np.std(run_solves, ddof=1)) if n_repeats > 1 else 0.0

        results[arm_id] = {
            "label": arm_label,
            "n_repeats": n_repeats,
            "mean_total_time": mean_tot,
            "std_total_time": std_tot,
            "min_total_time": float(np.min(run_totals)),
            "max_total_time": float(np.max(run_totals)),
            "all_total_times": run_totals,
            "mean_setup_time": mean_set,
            "std_setup_time": std_set,
            "mean_solve_time": mean_sol,
            "std_solve_time": std_sol,
            "mean_iters": float(np.mean(run_mean_iters)),
            "max_iters": int(np.max(run_max_iters)),
            "selected_counts": [budget_k] * total_steps,
            "audit_equalized_budget_verified": True,
        }

        print(f"  ==> Summary [{arm_id}]: Total = {mean_tot:.4f} +/- {std_tot:.4f}s | Mean Iters = {results[arm_id]['mean_iters']:.2f} | Max Iters = {results[arm_id]['max_iters']}")

    sol_vec.destroy()

    summary_export = {
        "benchmark_type": benchmark_type,
        "mesh_n": n_mesh,
        "n_dofs": n_dofs,
        "total_steps": total_steps,
        "budget_k": budget_k,
        "n_repeats": n_repeats,
        "results": results,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary_export, f, indent=2)
        print(f"\n[Saved] Statistical repeatability results exported to {out_p}")

    return summary_export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMAME Statistical Repeatability Benchmark")
    parser.add_argument("--benchmark", type=str, default="case_b_dual_beam", choices=["serpentine_50", "case_b_dual_beam"])
    parser.add_argument("--mesh-n", type=int, default=28)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--budget-k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/repeated_timing_{args.benchmark}_n{args.mesh_n}.json"

    run_repeated_trials(
        benchmark_type=args.benchmark,
        n_mesh=args.mesh_n,
        total_steps=args.steps,
        budget_k=args.budget_k,
        n_repeats=args.repeats,
        output_path=args.output,
    )

#!/usr/bin/env python3
"""
================================================================================
Stage A & B Benchmark: Four-Regime Budget Response Surface & Empirical K* Oracle
Evaluates the complete budget spectrum K in {0, 1, 2, 3, 4, 5, 6, 7, 8} across
all four characteristic physics regimes:
1. Gentle Single-Front (10 steps, 1 localized front)
2. Moderate Dual-Beam (Case B, 10 steps, 2 crossing fronts)
3. Long-Horizon Serpentine (50 steps, 3-pass continuous zigzag)
4. Violent Multi-Front Churn (16 steps, 4-phase alternating non-local churn)

Identifies:
- Regime-dependent Empirical Oracle: K*_r = argmin_K T_r(K)
- Fixed-K=3 Regret / Overhead across regimes
- Iteration plateau vs. setup cost trade-off curves
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
from benchmarks.run_long_horizon_sequence import assemble_step as assemble_serpentine_step
from benchmarks.run_case_b_complex_trajectory import assemble_dual_beam_step
from benchmarks.run_multi_front_starvation_stress import assemble_multi_front_fem_step


def assemble_regime_trajectory(
    regime: str,
    mesh: d.Mesh,
    V: d.FunctionSpace,
    total_steps: int,
    dt: float,
) -> Tuple[List[PETSc.Mat], List[PETSc.Vec], List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[np.ndarray]]:
    """Assemble all time steps for a chosen regime."""
    mats, b_vecs, csrs, diags = [], [], [], []
    u_prev = d.Function(V)
    u_prev.vector()[:] = 0.0

    for step in range(total_steps + 1):
        if regime == "gentle_single_front":
            mat, b_vec, indptr, indices, data, _ = assemble_serpentine_step(
                mesh, V, step, total_steps, dt, u_prev, rho_cp=2.0
            )
        elif regime == "case_b_dual_beam":
            mat, b_vec, indptr, indices, data, _ = assemble_dual_beam_step(
                mesh, V, step, total_steps, dt, u_prev
            )
        elif regime == "serpentine_50":
            mat, b_vec, indptr, indices, data, _ = assemble_serpentine_step(
                mesh, V, step, total_steps, dt, u_prev, rho_cp=10.0
            )
        elif regime == "multi_front_churn":
            mat, b_vec, indptr, indices, data, _ = assemble_multi_front_fem_step(
                mesh, V, step, total_steps, dt, u_prev, rho_cp=1.0
            )
        else:
            raise ValueError(f"Unknown regime: {regime}")

        mats.append(mat)
        b_vecs.append(b_vec)
        csrs.append((indptr, indices, data))
        diags.append(mat.getDiagonal().getArray().copy())

    return mats, b_vecs, csrs, diags


def evaluate_regime_sweep(
    regime: str,
    mesh_n: int = 24,
    total_steps: Optional[int] = None,
    dt: float = 0.05,
) -> Dict[str, Any]:
    if regime == "gentle_single_front":
        if total_steps is None: total_steps = 10
        desc = "Gentle Single-Front (1 beam linear translation)"
    elif regime == "case_b_dual_beam":
        if total_steps is None: total_steps = 10
        desc = "Moderate Dual-Beam (2 beams crossing diagonally)"
    elif regime == "serpentine_50":
        if total_steps is None: total_steps = 50
        desc = "Long-Horizon Serpentine (3-pass continuous zigzag)"
    elif regime == "multi_front_churn":
        if total_steps is None: total_steps = 16
        desc = "Violent Multi-Front Churn (4-phase alternating jumps)"
    else:
        raise ValueError(f"Unknown regime: {regime}")

    print("=" * 115)
    print(f"   REGIME: {desc.upper()}")
    print(f"   Mesh: UnitCubeMesh({mesh_n}, {mesh_n}, {mesh_n}) | Steps: {total_steps} | dt: {dt}")
    print(f"   Budget Sweep: K in {{0, 1, 2, 3, 4, 5, 6, 7, 8}} | Certified Res < 1.0e-8")
    print("=" * 115)

    mesh = d.UnitCubeMesh(mesh_n, mesh_n, mesh_n)
    V = d.FunctionSpace(mesh, "Lagrange", 1)
    n_dofs = V.dim()
    coords = V.tabulate_dof_coordinates()

    part = part_module.create_3d_overlapping_partition(
        coordinates=coords,
        grid=(2, 2, 2),
        overlap_layers=1,
        mesh_n=mesh_n,
    )
    n_sub = int(part["subdomain_count"])
    assert n_sub == 8

    print(f"--> Preassembling all {total_steps} steps...")
    t_asm_start = time.perf_counter()
    mats, b_vecs, csrs, diags = assemble_regime_trajectory(regime, mesh, V, total_steps, dt)
    print(f"    Assembled {total_steps + 1} steps in {time.perf_counter() - t_asm_start:.2f}s.\n")

    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)
    records = []

    print("-" * 115)
    print(f"{'K':<5}{'Update %':<11}{'Mean Iters':<12}{'Max Iters':<12}{'Max Age':<10}{'T_setup(s)':<12}{'T_solve(s)':<12}{'T_total(s)':<12}{'T(K)/T_Full':<12}{'RelRes':<10}")
    print("-" * 115)

    for k in range(n_sub + 1):
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

            backend.refresh_local(mat_t, step, selected)
            refresh_coarse_vectorized(backend, ind_t, indx_t, d_t, n_sub)
            t_setup = time.perf_counter() - t0

            sol_vec.set(0.0)
            its, t_sol, rel_res = solve_pcg_certified(mat_t, backend, b_t, sol_vec, max_it=500)
            assert rel_res < 1.0e-8, f"Certification failure at K={k}, step={step}: {rel_res}"

            times_setup.append(float(t_setup))
            times_solve.append(float(t_sol))
            times_total.append(float(t_setup + t_sol))
            iters_list.append(int(its))
            res_list.append(float(rel_res))
            max_ages_list.append(int(max(ages)))

        backend.destroy()

        tot_setup = float(np.sum(times_setup))
        tot_solve = float(np.sum(times_solve))
        tot_time = float(np.sum(times_total))

        rec = {
            "K": k,
            "refresh_ratio": k / float(n_sub),
            "mean_iters": float(np.mean(iters_list)),
            "max_iters": int(np.max(iters_list)),
            "max_age": int(np.max(max_ages_list)),
            "iters_history": iters_list,
            "total_setup_sec": tot_setup,
            "total_solve_sec": tot_solve,
            "total_wall_sec": tot_time,
            "max_rel_res": float(np.max(res_list)),
        }
        records.append(rec)

    t_full = records[n_sub]["total_wall_sec"]
    for rec in records:
        ratio = rec["total_wall_sec"] / t_full
        rec["ratio_vs_full"] = float(ratio)
        pct = f"{rec['K']}/8 ({rec['refresh_ratio']*100:.1f}%)"
        print(
            f"{rec['K']:<5}{pct:<11}{rec['mean_iters']:<12.1f}{rec['max_iters']:<12}{rec['max_age']:<10}"
            f"{rec['total_setup_sec']:<12.3f}{rec['total_solve_sec']:<12.3f}{rec['total_wall_sec']:<12.3f}"
            f"{ratio:<12.3f}{rec['max_rel_res']:<10.2e}"
        )

    best_rec = min(records, key=lambda r: r["total_wall_sec"])
    k_star = best_rec["K"]
    print("-" * 115)
    print(f"==> EMPIRICAL ORACLE for {regime}: K* = {k_star}/8 (T = {best_rec['total_wall_sec']:.3f}s, {best_rec['ratio_vs_full']:.3f}x T_Full)")
    k3_rec = records[3]
    k3_regret = (k3_rec["total_wall_sec"] - best_rec["total_wall_sec"]) / best_rec["total_wall_sec"] * 100.0
    print(f"    Fixed K=3: T(3) = {k3_rec['total_wall_sec']:.3f}s (Regret vs K*: +{k3_regret:.1f}%)")
    print("-" * 115 + "\n")

    sol_vec.destroy()
    for m in mats: m.destroy()
    for b in b_vecs: b.destroy()

    return {
        "regime": regime,
        "description": desc,
        "mesh_n": mesh_n,
        "total_steps": total_steps,
        "k_star": k_star,
        "t_full": t_full,
        "k3_regret_pct": k3_regret,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Four-Regime Budget Response Surface")
    parser.add_argument("--regime", type=str, default="all",
                        choices=["gentle_single_front", "case_b_dual_beam", "serpentine_50", "multi_front_churn", "all"])
    parser.add_argument("--mesh-n", type=int, default=24)
    parser.add_argument("--out", type=str, default="results/four_regime_budget_response.json")
    args = parser.parse_args()

    regimes_to_run = (
        ["gentle_single_front", "case_b_dual_beam", "serpentine_50", "multi_front_churn"]
        if args.regime == "all"
        else [args.regime]
    )

    results = {}
    for r in regimes_to_run:
        res = evaluate_regime_sweep(r, mesh_n=args.mesh_n)
        results[r] = res

    out_file = ROOT / args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n✓ All regime results archived to: {out_file.resolve()}")


if __name__ == "__main__":
    main()

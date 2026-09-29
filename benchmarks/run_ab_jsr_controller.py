#!/usr/bin/env python3
"""
================================================================================
Stage C Benchmark: Adaptive-Budget Stateful Schwarz Maintenance (AB-JSR)
Implements and evaluates the Causal Adaptive Budget Controller:
    K_t = C(s(t), G_t, N_{iter}(t-1)) in {0, 1, ..., M}

Core Mechanisms:
1. Pre-solve Algebraic Sensing:
   - Normalized relative distributed drift s_i(t) = d_{i,2}^{rel}(t) * (1 + beta * age_i)
   - Risk concentration Lorenz curve Q_k(t) = sum_{j=1}^k s_{(j)}(t) / sum_i s_i(t)
   - Global churn intensity G_t = sqrt(1/M sum_i (||Delta d_i|| / ||d_i||)^2)
2. Closed-Loop Decision Policy:
   - Full Rebuild Escalation: G_t >= tau_rebuild OR N_{iter}(t-1) >= N_panic -> K_t = M
   - Frozen Reuse: S_tot(t) <= tau_quiesce -> K_t = 0
   - Risk Catchment: K_t = min { k in {1, ..., M-1} : Q_k(t) >= alpha }
3. Strict Causality:
   - Uses only pre-solve matrix diagonal and delayed step (t-1) solver feedback.
   - Zero forward information leakage.
   - Certified PCG True Residual < 1.0e-8.
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
from benchmarks.run_four_regime_budget_response import assemble_regime_trajectory


class AdaptiveBudgetController:
    """
    Causal controller for dynamically selecting subdomain maintenance budget K_t.
    """
    def __init__(
        self,
        n_sub: int = 8,
        alpha_catchment: float = 0.80,
        tau_rebuild: float = 0.22,
        tau_quiesce: float = 1.0e-4,
        n_panic: int = 70,
        beta_age: float = 0.15,
        k_min: int = 1,
    ):
        self.n_sub = n_sub
        self.alpha_catchment = alpha_catchment
        self.tau_rebuild = tau_rebuild
        self.tau_quiesce = tau_quiesce
        self.n_panic = n_panic
        self.beta_age = beta_age
        self.k_min = k_min

    def decide_budget_and_subdomains(
        self,
        raw_scores: List[float],
        rel_drifts: List[float],
        ages: List[int],
        prev_iters: Optional[int],
    ) -> Tuple[int, List[int], str, Dict[str, float]]:
        """
        Decides K_t and returns (K_t, selected_subdomains, decision_reason, metrics).
        Strictly causal: uses pre-solve drifts and previous step's iteration count.
        """
        M = self.n_sub
        stateful_scores = [raw_scores[i] * (1.0 + self.beta_age * ages[i]) for i in range(M)]
        s_tot = float(np.sum(stateful_scores))
        g_t = float(np.sqrt(np.mean(np.array(rel_drifts) ** 2)))

        sorted_indices = sorted(range(M), key=lambda i: -stateful_scores[i])
        sorted_scores = [stateful_scores[i] for i in sorted_indices]

        # Metric dictionary for diagnostic telemetry
        metrics = {
            "s_tot": s_tot,
            "G_t": g_t,
            "top1_score": sorted_scores[0],
            "top2_score": sorted_scores[1] if M > 1 else 0.0,
        }

        # Case 1: Quiescence -> Frozen Reuse
        if s_tot <= self.tau_quiesce:
            return 0, [], "quiescent_reuse", metrics

        # Case 2: Severe global churn -> Full Rebuild
        if g_t >= self.tau_rebuild:
            metrics["q_val"] = 1.0
            return M, list(range(M)), f"global_churn_escalation (G_t={g_t:.3f}>={self.tau_rebuild})", metrics

        # Case 3: Solver distress feedback from step t-1 -> Full Rebuild or Boost
        if prev_iters is not None and prev_iters >= self.n_panic:
            metrics["q_val"] = 1.0
            return M, list(range(M)), f"solver_distress_escalation (iter_prev={prev_iters}>={self.n_panic})", metrics

        # Case 4: Risk concentration catchment Q_k >= alpha
        cum_s = np.cumsum(sorted_scores)
        q_fractions = cum_s / max(s_tot, 1.0e-14)

        k_chosen = M
        for k in range(1, M):
            if q_fractions[k - 1] >= self.alpha_catchment:
                k_chosen = k
                break

        k_chosen = max(self.k_min, min(M, k_chosen))
        selected = sorted(sorted_indices[:k_chosen])
        metrics["q_val"] = float(q_fractions[k_chosen - 1]) if k_chosen <= M else 1.0

        reason = f"risk_catchment (K={k_chosen}/{M}, Q={metrics['q_val']:.2f}>={self.alpha_catchment})"
        return k_chosen, selected, reason, metrics


def run_adaptive_budget_regime(
    regime: str,
    mesh_n: int = 24,
    total_steps: Optional[int] = None,
    dt: float = 0.05,
    controller_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if regime == "gentle_single_front":
        if total_steps is None: total_steps = 10
        desc = "Gentle Single-Front"
    elif regime == "case_b_dual_beam":
        if total_steps is None: total_steps = 10
        desc = "Moderate Dual-Beam (Case B)"
    elif regime == "serpentine_50":
        if total_steps is None: total_steps = 50
        desc = "Long-Horizon Serpentine"
    elif regime == "multi_front_churn":
        if total_steps is None: total_steps = 16
        desc = "Violent Multi-Front Churn"
    else:
        raise ValueError(f"Unknown regime: {regime}")

    c_params = {
        "alpha_catchment": 0.80,
        "tau_rebuild": 0.22,
        "tau_quiesce": 1.0e-4,
        "n_panic": 70,
        "beta_age": 0.15,
        "k_min": 1,
    }
    if controller_params:
        c_params.update(controller_params)

    print("=" * 115)
    print(f"   AB-JSR CAUSAL CONTROLLER EVALUATION: {desc.upper()}")
    print(f"   Mesh: UnitCubeMesh({mesh_n}) | Steps: {total_steps} | dt: {dt}")
    print(f"   Controller: alpha={c_params['alpha_catchment']} | tau_rebuild={c_params['tau_rebuild']} | n_panic={c_params['n_panic']}")
    print(f"   Certified PCG Residual < 1.0e-8")
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

    controller = AdaptiveBudgetController(
        n_sub=n_sub,
        alpha_catchment=c_params["alpha_catchment"],
        tau_rebuild=c_params["tau_rebuild"],
        tau_quiesce=c_params["tau_quiesce"],
        n_panic=c_params["n_panic"],
        beta_age=c_params["beta_age"],
        k_min=c_params["k_min"],
    )

    print(f"--> Preassembling all {total_steps} steps...")
    t_asm_start = time.perf_counter()
    mats, b_vecs, csrs, diags = assemble_regime_trajectory(regime, mesh, V, total_steps, dt)
    print(f"    Assembled {total_steps + 1} steps in {time.perf_counter() - t_asm_start:.2f}s.\n")

    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)

    backend = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
    backend.refresh_local(mats[0], 0, range(n_sub))
    ind0, indx0, d0 = csrs[0]
    refresh_coarse_vectorized(backend, ind0, indx0, d0, n_sub)

    ages = [0] * n_sub
    prev_iters = None

    history = []
    times_setup, times_solve, times_monitor, times_total = [], [], [], []
    iters_list, res_list, k_history = [], [], []

    print("-" * 115)
    print(f"{'Step':<6}{'K_t':<6}{'Decision Reason':<40}{'G_t':<10}{'Iters':<8}{'T_mon(ms)':<11}{'T_set(s)':<11}{'T_sol(s)':<11}{'T_tot(s)':<11}")
    print("-" * 115)

    for step in range(1, total_steps + 1):
        mat_t = mats[step]
        b_t = b_vecs[step]
        ind_t, indx_t, d_t = csrs[step]
        diag_t = diags[step]
        prev_diag = diags[step - 1]
        delta_diag = diag_t - prev_diag

        # 1. Pre-solve sensing & decision
        t_m0 = time.perf_counter()
        raw_scores = compute_monitor_scores("jsr_rel_l2_k3", delta_diag, diag_t, prev_diag, part["local_indices"], None)
        rel_drifts = [
            float(np.linalg.norm(delta_diag[idx])) / max(float(np.linalg.norm(diag_t[idx])), 1.0e-14)
            for idx in part["local_indices"]
        ]
        k_t, selected, reason, metrics = controller.decide_budget_and_subdomains(
            raw_scores, rel_drifts, ages, prev_iters
        )
        t_monitor = time.perf_counter() - t_m0

        # 2. Preconditioner maintenance
        t_s0 = time.perf_counter()
        for cid in range(n_sub):
            if cid in selected:
                ages[cid] = 0
            else:
                ages[cid] += 1

        backend.refresh_local(mat_t, step, selected)
        refresh_coarse_vectorized(backend, ind_t, indx_t, d_t, n_sub)
        t_setup = time.perf_counter() - t_s0

        # 3. Certified PCG solve
        sol_vec.set(0.0)
        its, t_sol, rel_res = solve_pcg_certified(mat_t, backend, b_t, sol_vec, max_it=500)
        assert rel_res < 1.0e-8, f"Residual failure at step {step}: {rel_res}"
        prev_iters = its

        t_total = t_monitor + t_setup + t_sol

        times_monitor.append(float(t_monitor))
        times_setup.append(float(t_setup))
        times_solve.append(float(t_sol))
        times_total.append(float(t_total))
        iters_list.append(int(its))
        res_list.append(float(rel_res))
        k_history.append(int(k_t))

        short_reason = reason if len(reason) <= 38 else reason[:35] + "..."
        print(
            f"{step:<6}{k_t:<6}{short_reason:<40}{metrics['G_t']:<10.3f}{its:<8}"
            f"{t_monitor*1000:<11.2f}{t_setup:<11.3f}{t_sol:<11.3f}{t_total:<11.3f}"
        )

        history.append({
            "step": step,
            "K_t": k_t,
            "selected": selected,
            "reason": reason,
            "metrics": metrics,
            "iters": int(its),
            "rel_res": float(rel_res),
            "t_monitor": float(t_monitor),
            "t_setup": float(t_setup),
            "t_solve": float(t_sol),
            "t_total": float(t_total),
        })

    backend.destroy()
    sol_vec.destroy()
    for m in mats: m.destroy()
    for b in b_vecs: b.destroy()

    tot_mon = float(np.sum(times_monitor))
    tot_set = float(np.sum(times_setup))
    tot_sol = float(np.sum(times_solve))
    tot_time = float(np.sum(times_total))

    mean_k = float(np.mean(k_history))
    mean_its = float(np.mean(iters_list))
    max_its = int(np.max(iters_list))

    print("-" * 115)
    print(f"AB-JSR Summary for {regime}:")
    print(f"  Total Wall-Clock: {tot_time:.3f} s  (Setup: {tot_set:.3f} s, Solve: {tot_sol:.3f} s, Monitor: {tot_mon*1000:.1f} ms [{tot_mon/tot_time*100:.2f}%])")
    print(f"  Mean Budget K_t:  {mean_k:.2f}/8  ({mean_k/8.0*100:.1f}% average refresh ratio)")
    print(f"  Mean Iterations:  {mean_its:.1f}  (Max: {max_its})")
    print(f"  Budget Sequence:  {k_history}")
    print("-" * 115 + "\n")

    return {
        "regime": regime,
        "description": desc,
        "mesh_n": mesh_n,
        "total_steps": total_steps,
        "controller_params": c_params,
        "total_wall_sec": tot_time,
        "total_setup_sec": tot_set,
        "total_solve_sec": tot_sol,
        "total_monitor_sec": tot_mon,
        "monitor_overhead_pct": tot_mon / tot_time * 100.0,
        "mean_k": mean_k,
        "k_history": k_history,
        "mean_iters": mean_its,
        "max_iters": max_its,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="AB-JSR Causal Controller Benchmark")
    parser.add_argument("--regime", type=str, default="all",
                        choices=["gentle_single_front", "case_b_dual_beam", "serpentine_50", "multi_front_churn", "all"])
    parser.add_argument("--mesh-n", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--tau-rebuild", type=float, default=0.22)
    parser.add_argument("--n-panic", type=int, default=70)
    parser.add_argument("--out", type=str, default="results/ab_jsr_controller_evaluation.json")
    args = parser.parse_args()

    c_params = {
        "alpha_catchment": args.alpha,
        "tau_rebuild": args.tau_rebuild,
        "tau_quiesce": 1.0e-4,
        "n_panic": args.n_panic,
        "beta_age": 0.15,
        "k_min": 1,
    }

    regimes_to_run = (
        ["gentle_single_front", "case_b_dual_beam", "serpentine_50", "multi_front_churn"]
        if args.regime == "all"
        else [args.regime]
    )

    results = {}
    for r in regimes_to_run:
        res = run_adaptive_budget_regime(r, mesh_n=args.mesh_n, controller_params=c_params)
        results[r] = res

    out_file = ROOT / args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n✓ AB-JSR controller results archived to: {out_file.resolve()}")


if __name__ == "__main__":
    main()

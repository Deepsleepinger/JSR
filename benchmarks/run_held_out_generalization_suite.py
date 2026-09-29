#!/usr/bin/env python3
"""
================================================================================
CMAME Generalization Suite: Held-Out Trajectory Validation & Repeated Timing
Evaluates the FROZEN AB-JSR causal controller across two completely held-out,
uncalibrated trajectories under n=5 repeated timing trials:
1. Held-out Trajectory 1: Compound Regime Shift (18 steps)
2. Held-out Trajectory 2: Fast-Switching Staccato & Re-visitation Shock (16 steps)

Protocol:
- Calibration Set: {Gentle Single-Front, Case B Dual-Beam, Serpentine 50}
- Frozen Controller Parameters:
  alpha=0.80, tau_up=0.22, tau_down=0.15, cooldown_steps=2, n_panic=70, beta=0.15
- Comparison Arms:
  1. Frozen Local Factors (Sync Coarse, K=0)
  2. Fixed Low Budget (K=1)
  3. Fixed Mid-Low Budget (K=2)
  4. Conventional Fixed Budget (K=3)
  5. Svolos-Style Physics Peak Selector (K=3)
  6. Full Rebuild (K=8)
  7. AB-JSR Causal Controller (Proposed)
- Certified PCG True Residual < 1.0e-8
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
from benchmarks.run_ab_jsr_controller import AdaptiveBudgetController
from benchmarks.run_compound_regime_shift_stress import (
    ParameterizedCompoundSource,
    assemble_compound_fem_step,
)


class ParameterizedStaccatoSource:
    """Parameterized fast-switching staccato shock source."""
    def __init__(self, k_solid: float = 1.0, k_melt: float = 20.0, r0: float = 0.12):
        self.k_expr = d.Expression(
            "k_solid + (k_melt - k_solid) * ("
            "w1 * exp(-((x[0]-x1)*(x[0]-x1) + (x[1]-y1)*(x[1]-y1) + (x[2]-z1)*(x[2]-z1))/(2.0*r0*r0)) + "
            "w2 * exp(-((x[0]-x2)*(x[0]-x2) + (x[1]-y2)*(x[1]-y2) + (x[2]-z2)*(x[2]-z2))/(2.0*r0*r0)) + "
            "w3 * exp(-((x[0]-x3)*(x[0]-x3) + (x[1]-y3)*(x[1]-y3) + (x[2]-z3)*(x[2]-z3))/(2.0*r0*r0)) + "
            "w4 * exp(-((x[0]-x4)*(x[0]-x4) + (x[1]-y4)*(x[1]-y4) + (x[2]-z4)*(x[2]-z4))/(2.0*r0*r0))"
            ")",
            k_solid=k_solid, k_melt=k_melt, r0=r0,
            x1=0.0, y1=0.0, z1=0.0, w1=0.0,
            x2=0.0, y2=0.0, z2=0.0, w2=0.0,
            x3=0.0, y3=0.0, z3=0.0, w3=0.0,
            x4=0.0, y4=0.0, z4=0.0, w4=0.0,
            degree=2,
        )
        self.h_expr = d.Expression(
            "q1 * exp(-((x[0]-x1)*(x[0]-x1) + (x[1]-y1)*(x[1]-y1) + (x[2]-z1)*(x[2]-z1))/(r0*r0)) + "
            "q2 * exp(-((x[0]-x2)*(x[0]-x2) + (x[1]-y2)*(x[1]-y2) + (x[2]-z2)*(x[2]-z2))/(r0*r0)) + "
            "q3 * exp(-((x[0]-x3)*(x[0]-x3) + (x[1]-y3)*(x[1]-y3) + (x[2]-z3)*(x[2]-z3))/(r0*r0)) + "
            "q4 * exp(-((x[0]-x4)*(x[0]-x4) + (x[1]-y4)*(x[1]-y4) + (x[2]-z4)*(x[2]-z4))/(r0*r0))",
            r0=r0,
            x1=0.0, y1=0.0, z1=0.0, q1=0.0,
            x2=0.0, y2=0.0, z2=0.0, q2=0.0,
            x3=0.0, y3=0.0, z3=0.0, q3=0.0,
            x4=0.0, y4=0.0, z4=0.0, q4=0.0,
            degree=2,
        )

    def update(self, step: int):
        self.k_expr.w1 = self.k_expr.w2 = self.k_expr.w3 = self.k_expr.w4 = 0.0
        self.h_expr.q1 = self.h_expr.q2 = self.h_expr.q3 = self.h_expr.q4 = 0.0

        if step <= 4:
            # Phase 1: Lower-left corner pulse
            s = float(step) / 4.0
            self.k_expr.x1, self.k_expr.y1, self.k_expr.z1, self.k_expr.w1 = 0.20 + 0.10 * s, 0.25, 0.25, 1.0
            self.h_expr.x1, self.h_expr.y1, self.h_expr.z1, self.h_expr.q1 = self.k_expr.x1, 0.25, 0.25, 110.0
        elif step <= 8:
            # Phase 2: Upper-right corner pulse
            s = float(step - 4) / 4.0
            self.k_expr.x1, self.k_expr.y1, self.k_expr.z1, self.k_expr.w1 = 0.80 - 0.10 * s, 0.75, 0.75, 1.0
            self.h_expr.x1, self.h_expr.y1, self.h_expr.z1, self.h_expr.q1 = self.k_expr.x1, 0.75, 0.75, 110.0
        elif step <= 12:
            # Phase 3: Dual intersecting confrontation
            s = float(step - 8) / 4.0
            self.k_expr.x1, self.k_expr.y1, self.k_expr.z1, self.k_expr.w1 = 0.25 + 0.25 * s, 0.50, 0.50, 1.0
            self.h_expr.x1, self.h_expr.y1, self.h_expr.z1, self.h_expr.q1 = self.k_expr.x1, 0.50, 0.50, 125.0

            self.k_expr.x2, self.k_expr.y2, self.k_expr.z2, self.k_expr.w2 = 0.75 - 0.25 * s, 0.50, 0.50, 1.0
            self.h_expr.x2, self.h_expr.y2, self.h_expr.z2, self.h_expr.q2 = self.k_expr.x2, 0.50, 0.50, 125.0
        else:
            # Phase 4: Fast-alternating re-visitation corner shocks
            if step % 2 == 1:
                self.k_expr.x1, self.k_expr.y1, self.k_expr.z1, self.k_expr.w1 = 0.20, 0.25, 0.25, 1.0
                self.h_expr.x1, self.h_expr.y1, self.h_expr.z1, self.h_expr.q1 = 0.20, 0.25, 0.25, 120.0
                self.k_expr.x2, self.k_expr.y2, self.k_expr.z2, self.k_expr.w2 = 0.80, 0.75, 0.75, 1.0
                self.h_expr.x2, self.h_expr.y2, self.h_expr.z2, self.h_expr.q2 = 0.80, 0.75, 0.75, 120.0
            else:
                self.k_expr.x1, self.k_expr.y1, self.k_expr.z1, self.k_expr.w1 = 0.20, 0.75, 0.75, 1.0
                self.h_expr.x1, self.h_expr.y1, self.h_expr.z1, self.h_expr.q1 = 0.20, 0.75, 0.75, 120.0
                self.k_expr.x2, self.k_expr.y2, self.k_expr.z2, self.k_expr.w2 = 0.80, 0.25, 0.25, 1.0
                self.h_expr.x2, self.h_expr.y2, self.h_expr.z2, self.h_expr.q2 = 0.80, 0.25, 0.25, 120.0


def preassemble_trajectory(
    traj_name: str,
    mesh: d.Mesh,
    V: d.FunctionSpace,
    total_steps: int,
    dt: float = 0.05,
) -> Tuple[List[PETSc.Mat], List[PETSc.Vec], List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[np.ndarray], List[np.ndarray]]:
    mats, b_vecs, csrs, diags, k_fields = [], [], [], [], []
    u_prev = d.Function(V)
    u_prev.vector()[:] = 0.0

    if traj_name == "compound_shift":
        src_ctx = ParameterizedCompoundSource()
    elif traj_name == "staccato_revisit":
        src_ctx = ParameterizedStaccatoSource()
    else:
        raise ValueError(traj_name)

    for s in range(total_steps + 1):
        mat, b_vec, indptr, indices, data = assemble_compound_fem_step(
            mesh, V, s, dt, u_prev, src_ctx, rho_cp=1.0
        )
        k_func = d.interpolate(src_ctx.k_expr, V)
        mats.append(mat)
        b_vecs.append(b_vec)
        csrs.append((indptr, indices, data))
        diags.append(mat.getDiagonal().getArray().copy())
        k_fields.append(k_func.vector().get_local().copy())

    return mats, b_vecs, csrs, diags, k_fields


def evaluate_single_run(
    arm: str,
    mats: List[PETSc.Mat],
    b_vecs: List[PETSc.Vec],
    csrs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    diags: List[np.ndarray],
    k_fields: List[np.ndarray],
    part: Dict[str, Any],
    n_sub: int,
    n_dofs: int,
    total_steps: int,
    controller_frozen: AdaptiveBudgetController,
) -> Dict[str, Any]:
    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)
    backend = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
    backend.refresh_local(mats[0], 0, range(n_sub))
    ind0, indx0, d0 = csrs[0]
    refresh_coarse_vectorized(backend, ind0, indx0, d0, n_sub)

    ages = [0] * n_sub
    prev_iters = None
    controller_frozen.reset()

    times_setup, times_solve, times_monitor, times_total = [], [], [], []
    iters_list, k_list = [], []

    for step in range(1, total_steps + 1):
        mat_t = mats[step]
        b_t = b_vecs[step]
        ind_t, indx_t, d_t = csrs[step]
        diag_t = diags[step]
        prev_diag = diags[step - 1]
        delta_diag = diag_t - prev_diag

        t_m0 = time.perf_counter()
        if arm == "frozen_local_k0":
            k_t = 0
            selected = []
        elif arm == "fixed_budget_k1":
            k_t = 1
            raw = compute_monitor_scores("jsr_rel_l2_k3", delta_diag, diag_t, prev_diag, part["local_indices"], None)
            scores = [raw[i] * (1.0 + 0.15 * ages[i]) for i in range(n_sub)]
            selected = sorted(sorted(range(n_sub), key=lambda i: -scores[i])[:1])
        elif arm == "fixed_budget_k2":
            k_t = 2
            raw = compute_monitor_scores("jsr_rel_l2_k3", delta_diag, diag_t, prev_diag, part["local_indices"], None)
            scores = [raw[i] * (1.0 + 0.15 * ages[i]) for i in range(n_sub)]
            selected = sorted(sorted(range(n_sub), key=lambda i: -scores[i])[:2])
        elif arm == "fixed_budget_k3":
            k_t = 3
            raw = compute_monitor_scores("jsr_rel_l2_k3", delta_diag, diag_t, prev_diag, part["local_indices"], None)
            scores = [raw[i] * (1.0 + 0.15 * ages[i]) for i in range(n_sub)]
            selected = sorted(sorted(range(n_sub), key=lambda i: -scores[i])[:3])
        elif arm == "svolos_physics_k3":
            k_t = 3
            raw = [float(np.max(k_fields[step][idx])) for idx in part["local_indices"]]
            selected = sorted(sorted(range(n_sub), key=lambda i: -raw[i])[:3])
        elif arm == "full_rebuild_k8":
            k_t = 8
            selected = list(range(n_sub))
        elif arm == "ab_jsr_controller":
            raw_scores = compute_monitor_scores("jsr_rel_l2_k3", delta_diag, diag_t, prev_diag, part["local_indices"], None)
            rel_drifts = [
                float(np.linalg.norm(delta_diag[idx])) / max(float(np.linalg.norm(diag_t[idx])), 1.0e-14)
                for idx in part["local_indices"]
            ]
            k_t, selected, reason, metrics = controller_frozen.decide_budget_and_subdomains(
                raw_scores, rel_drifts, ages, prev_iters
            )
        else:
            raise ValueError(arm)

        t_mon = time.perf_counter() - t_m0

        t_s0 = time.perf_counter()
        for cid in range(n_sub):
            if cid in selected: ages[cid] = 0
            else: ages[cid] += 1
        backend.refresh_local(mat_t, step, selected)
        refresh_coarse_vectorized(backend, ind_t, indx_t, d_t, n_sub)
        t_set = time.perf_counter() - t_s0

        sol_vec.set(0.0)
        its, t_sol, rel_res = solve_pcg_certified(mat_t, backend, b_t, sol_vec, max_it=500)
        assert rel_res < 1.0e-8, f"Residual failure at step {step}: {rel_res}"
        prev_iters = its

        t_tot = t_mon + t_set + t_sol
        times_setup.append(float(t_set))
        times_solve.append(float(t_sol))
        times_monitor.append(float(t_mon))
        times_total.append(float(t_tot))
        iters_list.append(int(its))
        k_list.append(int(k_t))

    backend.destroy()
    sol_vec.destroy()

    return {
        "total_time": float(np.sum(times_total)),
        "setup_time": float(np.sum(times_setup)),
        "solve_time": float(np.sum(times_solve)),
        "monitor_time": float(np.sum(times_monitor)),
        "mean_iters": float(np.mean(iters_list)),
        "max_iters": int(np.max(iters_list)),
        "mean_k": float(np.mean(k_list)),
        "iters_history": iters_list,
        "k_sequence": k_list,
    }


def run_held_out_trajectory(
    traj_name: str,
    mesh_n: int = 24,
    total_steps: int = 18,
    n_repeats: int = 5,
    dt: float = 0.05,
) -> Dict[str, Any]:
    desc = "Held-Out Trajectory 1: Compound Regime Shift" if traj_name == "compound_shift" else "Held-Out Trajectory 2: Staccato Shock & Re-visitation"
    print("=" * 115)
    print(f"   {desc.upper()}")
    print(f"   Mesh: UnitCubeMesh({mesh_n}) | Steps: {total_steps} | Repeats: {n_repeats} | Certified Res < 1.0e-8")
    print("   Frozen Controller: alpha=0.80 | tau_up=0.22 | tau_down=0.15 | cooldown=2 | n_panic=70")
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

    print(f"--> Preassembling all {total_steps} steps of {traj_name}...")
    t_asm_start = time.perf_counter()
    mats, b_vecs, csrs, diags, k_fields = preassemble_trajectory(traj_name, mesh, V, total_steps, dt)
    print(f"    Preassembled in {time.perf_counter() - t_asm_start:.2f}s.\n")

    frozen_controller = AdaptiveBudgetController(
        n_sub=n_sub,
        alpha_catchment=0.80,
        tau_up=0.22,
        tau_down=0.15,
        cooldown_steps=2,
        tau_quiesce=1.0e-4,
        n_panic=70,
        beta_age=0.15,
        k_min=1,
    )

    arms = [
        "frozen_local_k0",
        "fixed_budget_k1",
        "fixed_budget_k2",
        "fixed_budget_k3",
        "svolos_physics_k3",
        "full_rebuild_k8",
        "ab_jsr_controller",
    ]

    arm_labels = {
        "frozen_local_k0": "Frozen Local Factors (Sync Coarse, K=0)",
        "fixed_budget_k1": "Low Budget JSR (Fixed K=1)",
        "fixed_budget_k2": "Mid-Low Budget JSR (Fixed K=2)",
        "fixed_budget_k3": "Conventional JSR (Fixed K=3)",
        "svolos_physics_k3": "Svolos-Style Physics Peak (Fixed K=3)",
        "full_rebuild_k8": "Full Rebuild (K=8, 100% updates)",
        "ab_jsr_controller": "AB-JSR Causal Controller (Proposed)",
    }

    print("-" * 115)
    print(f"{'Strategy Policy':<42}{'Total Time (s)':<22}{'Setup (s)':<14}{'Solve (s)':<14}{'Mean It':<10}{'Max It':<10}{'Mean K':<8}")
    print("-" * 115)

    traj_results = {}

    for arm in arms:
        trials = []
        for rep in range(n_repeats):
            res = evaluate_single_run(
                arm, mats, b_vecs, csrs, diags, k_fields, part, n_sub, n_dofs, total_steps, frozen_controller
            )
            trials.append(res)

        total_times = [t["total_time"] for t in trials]
        setup_times = [t["setup_time"] for t in trials]
        solve_times = [t["solve_time"] for t in trials]
        mon_times = [t["monitor_time"] for t in trials]

        t_mean, t_std = float(np.mean(total_times)), float(np.std(total_times))
        s_mean, s_std = float(np.mean(setup_times)), float(np.std(setup_times))
        sol_mean, sol_std = float(np.mean(solve_times)), float(np.std(solve_times))

        mean_it = float(trials[0]["mean_iters"])
        max_it = int(trials[0]["max_iters"])
        mean_k = float(trials[0]["mean_k"])

        time_str = f"{t_mean:.3f} +/- {t_std:.3f}"
        print(
            f"{arm_labels[arm]:<42}{time_str:<22}{s_mean:<14.3f}{sol_mean:<14.3f}"
            f"{mean_it:<10.1f}{max_it:<10}{mean_k:<8.2f}"
        )

        traj_results[arm] = {
            "label": arm_labels[arm],
            "total_mean": t_mean,
            "total_std": t_std,
            "setup_mean": s_mean,
            "solve_mean": sol_mean,
            "monitor_mean": float(np.mean(mon_times)),
            "mean_iters": mean_it,
            "max_iters": max_it,
            "mean_k": mean_k,
            "raw_trials": trials,
            "k_sequence": trials[0]["k_sequence"],
            "iters_history": trials[0]["iters_history"],
        }

    # Summary comparisons
    t_ab_mean = traj_results["ab_jsr_controller"]["total_mean"]
    print("-" * 115)
    for arm in arms:
        if arm == "ab_jsr_controller": continue
        t_arm_mean = traj_results[arm]["total_mean"]
        diff_pct = (t_arm_mean - t_ab_mean) / t_ab_mean * 100.0
        print(f"--> AB-JSR ({t_ab_mean:.3f}s) vs {arm_labels[arm]} ({t_arm_mean:.3f}s): {diff_pct:+.1f}% overhead for baseline")
    print("-" * 115 + "\n")

    for m in mats: m.destroy()
    for b in b_vecs: b.destroy()

    return {
        "traj_name": traj_name,
        "description": desc,
        "mesh_n": mesh_n,
        "total_steps": total_steps,
        "n_repeats": n_repeats,
        "results": traj_results,
    }


def main():
    parser = argparse.ArgumentParser(description="CMAME Generalization Suite")
    parser.add_argument("--mesh-n", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=str, default="results/held_out_generalization_suite.json")
    args = parser.parse_args()

    suite_results = {}

    # Held-out Trajectory 1: Compound Shift (18 steps)
    res_cmp = run_held_out_trajectory(
        "compound_shift", mesh_n=args.mesh_n, total_steps=18, n_repeats=args.repeats
    )
    suite_results["compound_shift"] = res_cmp

    # Held-out Trajectory 2: Staccato Shock & Re-visitation (16 steps)
    res_stc = run_held_out_trajectory(
        "staccato_revisit", mesh_n=args.mesh_n, total_steps=16, n_repeats=args.repeats
    )
    suite_results["staccato_revisit"] = res_stc

    out_file = ROOT / args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(suite_results, fp, indent=2)
    print(f"✓ Complete Generalization Suite results archived to: {out_file.resolve()}\n")


if __name__ == "__main__":
    main()

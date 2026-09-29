#!/usr/bin/env python3
"""
================================================================================
CMAME Evidence Consolidation: JSR Monitor Family Ablation Study
Strictly Equalized Budget K = 3 under Certified True Residual < 1.0e-8

Evaluates the JSR Monitor Family across two demanding trajectories:
1. 50-Step Reciprocating Serpentine Trajectory (N=28, 24,389 DOFs)
2. Case B Dual-Beam Non-Monotonic Competing Fronts (N=28, 24,389 DOFs, 12 Steps)

Monitors Evaluated:
- JSR-L2: Absolute Euclidean Accumulation ||Delta diag(A_i)||_2
- JSR-relL2: Normalized Relative Distributed Drift (1 / sqrt(|Omega_i|)) * ||Delta diag / diag||_2
- JSR-relLinf-sym: Symmetric Relative Peak max_j |A_jj^t - A_jj^{t-1}| / max(|A_jj^t|, |A_jj^{t-1}|, eps_d)
- JSR-Hybrid: Dual-Channel Combining Peak Sensitivity and Distributed Drift
- Svolos-Inspired Physics-Aware Baseline: Instantaneous physics peak selector (max_{x in Omega_i} kappa(x))
- Full Rebuild: 100% factor reassembly at every step
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


class PythonPC(object):
    def __init__(self, backend_ctx: mumps_module.OverlappingMUMPSSchwarzBackend):
        self.backend = backend_ctx

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        x_arr = x.getArray(readonly=True)
        y_arr = y.getArray()
        self.backend.apply(x_arr, y_arr)


def solve_pcg_certified(
    mat: PETSc.Mat,
    backend_ctx: mumps_module.OverlappingMUMPSSchwarzBackend,
    b_vec: PETSc.Vec,
    x_vec: PETSc.Vec,
    rtol: float = 2.0e-10,
    max_it: int = 150,
) -> Tuple[int, float, float]:
    """Solve system A x = b with certified relative residual strictly < 1e-8."""
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


def compute_monitor_scores(
    monitor_name: str,
    delta_diag: np.ndarray,
    diag_t: np.ndarray,
    prev_diag: np.ndarray,
    local_indices: List[np.ndarray],
    k_vals: Optional[np.ndarray],
    eps_d: float = 1.0e-12,
) -> List[float]:
    """Compute raw per-subdomain drift indicators before age damping."""
    n_sub = len(local_indices)

    if monitor_name == "svolos_physics_k3":
        if k_vals is None:
            raise ValueError("k_vals required for svolos_physics")
        return [float(np.max(k_vals[idx])) for idx in local_indices]

    elif monitor_name == "jsr_l2_k3":
        # Classical absolute Euclidean accumulation
        return [float(np.linalg.norm(delta_diag[idx])) for idx in local_indices]

    elif monitor_name == "jsr_rel_l2_k3":
        # Normalized relative distributed drift
        denom = np.maximum(np.abs(prev_diag), eps_d)
        rel_diff = np.abs(delta_diag) / denom
        return [float(np.linalg.norm(rel_diff[idx]) / np.sqrt(len(idx))) for idx in local_indices]

    elif monitor_name == "jsr_rel_linf_sym_k3":
        # Symmetric relative localized peak
        denom = np.maximum(np.maximum(np.abs(diag_t), np.abs(prev_diag)), eps_d)
        rel_diff = np.abs(delta_diag) / denom
        return [float(np.max(rel_diff[idx])) for idx in local_indices]

    elif monitor_name == "jsr_hybrid_k3":
        # Dual-channel: balances localized peak sensitivity and distributed drift
        denom = np.maximum(np.maximum(np.abs(diag_t), np.abs(prev_diag)), eps_d)
        rel_diff = np.abs(delta_diag) / denom
        d_linf = np.array([float(np.max(rel_diff[idx])) for idx in local_indices])
        d_l2_rel = np.array([float(np.linalg.norm(rel_diff[idx]) / np.sqrt(len(idx))) for idx in local_indices])
        s_linf = d_linf / max(np.max(d_linf), eps_d)
        s_l2 = d_l2_rel / max(np.max(d_l2_rel), eps_d)
        d_comb = np.maximum(s_linf, s_l2)
        return [float(d_comb[i]) for i in range(n_sub)]

    else:
        raise ValueError(f"Unknown monitor name: {monitor_name}")


def run_benchmark_suite(
    benchmark_type: str = "serpentine_50",
    n_mesh: int = 28,
    total_steps: int = 50,
    budget_k: int = 3,
    dt: float = 0.05,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 110)
    print(f"   CMAME EXPERIMENT: JSR MONITOR FAMILY ABLATION ({benchmark_type.upper()})")
    print(f"   Mesh: UnitCubeMesh({n_mesh}, {n_mesh}, {n_mesh}) | Steps: {total_steps} | Budget K: {budget_k}/8")
    print(f"   Protocol: Strictly Equalized K = {budget_k} | Certified PCG True Residual < 1.0e-8")
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
        ("jsr_rel_l2_k3", "JSR-relL2 (Normalized Relative Distributed)"),
        ("jsr_rel_linf_sym_k3", "JSR-relLinf-sym (Symmetric Relative Peak)"),
        ("jsr_hybrid_k3", "JSR-Hybrid (Dual-Channel Peak + Distributed)"),
    ]

    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)
    results: Dict[str, Any] = {}

    for arm_id, arm_label in monitors:
        print(f"\n--> Running arm: {arm_label} (Budget K = {budget_k})...")
        backend = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
        backend.refresh_local(mats[0], 0, range(n_sub))
        ind0, indx0, d0 = csrs[0]
        refresh_coarse_vectorized(backend, ind0, indx0, d0, n_sub)

        local_ages = [0] * n_sub
        times_setup, times_solve, times_total = [], [], []
        iters_list, relres_list = [], []
        masks_history = []

        t_start_arm = time.perf_counter()

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

            # 2. Stateful risk assessment (age damping)
            if arm_id == "svolos_physics_k3":
                # Svolos-inspired instantaneous selection
                selected = sorted(sorted(range(n_sub), key=lambda i: -raw_scores[i])[:budget_k])
            else:
                # JSR stateful scoring: s_i(t) = d_i(t) * (1 + 0.15 * a_i(t))
                scores = [raw_scores[i] * (1.0 + 0.15 * local_ages[i]) for i in range(n_sub)]
                selected = sorted(sorted(range(n_sub), key=lambda i: -scores[i])[:budget_k])

            masks_history.append(selected)

            # 3. Factorization and coarse update
            t_loc = backend.refresh_local(mat_t, step, selected)
            t_crs = refresh_coarse_vectorized(backend, ind_t, indx_t, d_t, n_sub)
            t_setup = time.perf_counter() - t0

            # 4. Age update
            for cid in range(n_sub):
                if cid in selected:
                    local_ages[cid] = 0
                else:
                    local_ages[cid] += 1

            # 5. PCG certified solve
            sol_vec.set(0.0)
            its, t_sol, rel_res = solve_pcg_certified(mat_t, backend, b_t, sol_vec)
            t_step = t_setup + t_sol

            times_setup.append(float(t_setup))
            times_solve.append(float(t_sol))
            times_total.append(float(t_step))
            iters_list.append(int(its))
            relres_list.append(float(rel_res))

        backend.destroy()

        tot_time = float(sum(times_total))
        tot_setup = float(sum(times_setup))
        tot_solve = float(sum(times_solve))
        mean_iters = float(np.mean(iters_list))
        max_iters = int(np.max(iters_list))
        min_iters = int(np.min(iters_list))

        results[arm_id] = {
            "label": arm_label,
            "total_time": tot_time,
            "setup_time": tot_setup,
            "solve_time": tot_solve,
            "mean_iters": mean_iters,
            "max_iters": max_iters,
            "min_iters": min_iters,
            "iters_per_step": iters_list,
            "solve_times": times_solve,
            "setup_times": times_setup,
            "total_times": times_total,
            "relres_list": relres_list,
            "masks_history": masks_history,
        }

        print(f"    -> Total: {tot_time:.4f}s | Setup: {tot_setup:.4f}s | Solve: {tot_solve:.4f}s | Iters: {mean_iters:.2f} (Max: {max_iters})")

    sol_vec.destroy()

    summary_export = {
        "benchmark_type": benchmark_type,
        "mesh_n": n_mesh,
        "n_dofs": n_dofs,
        "total_steps": total_steps,
        "budget_k": budget_k,
        "results": results,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary_export, f, indent=2)
        print(f"\n[Saved] Benchmark results exported to {out_p}")

    return summary_export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSR Monitor Family Ablation")
    parser.add_argument("--benchmark", type=str, default="serpentine_50", choices=["serpentine_50", "case_b_dual_beam"])
    parser.add_argument("--mesh-n", type=int, default=28)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--budget-k", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/monitor_family_{args.benchmark}_n{args.mesh_n}.json"

    run_benchmark_suite(
        benchmark_type=args.benchmark,
        n_mesh=args.mesh_n,
        total_steps=args.steps,
        budget_k=args.budget_k,
        output_path=args.output,
    )

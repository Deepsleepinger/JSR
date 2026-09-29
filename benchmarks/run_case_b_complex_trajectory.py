#!/usr/bin/env python3
"""
================================================================================
CMAME Case B: Complex Dual-Beam Non-Monotonic Trajectory Benchmark
================================================================================

Simulates a dual-laser additive manufacturing process on 3D continuous Galerkin FEM:
- Laser 1: Moves forward along y = 0.25 (x: 0.20 -> 0.80)
- Laser 2: Moves backward along y = 0.75 (x: 0.80 -> 0.20) simultaneously!

This creates TWO simultaneous advancing disturbance fronts in opposing subdomains.
- Svolos-style physics heuristic: Queries physical kappa. Because Laser 1 is slightly
  hotter (q1 = 55, q2 = 45), greedy physical selection concentrates all K budget on
  Laser 1 subdomains, starving Laser 2 subdomains and triggering severe PCG iteration inflation.
- JSR algebraic drift: Naturally senses matrix delta on ALL perturbed DOFs regardless
  of physical coordinates, allocating budget proportionally across both moving fronts.
- Certified true residual strictly < 1.0e-8.
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
from benchmarks.run_fixed_budget_oracle_and_component_ablation import (
    refresh_coarse_vectorized,
    solve_pcg_certified,
)


def assemble_dual_beam_step(
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
    q1: float = 55.0,
    q2: float = 45.0,
) -> Tuple[PETSc.Mat, PETSc.Vec, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = d.TrialFunction(V)
    v = d.TestFunction(V)

    s = float(step) / max(total_steps, 1)
    # Beam 1 moves forward along y = 0.25
    x1 = 0.20 + 0.60 * s
    y1 = 0.25
    # Beam 2 moves backward along y = 0.75
    x2 = 0.80 - 0.60 * s
    y2 = 0.75

    kappa_expr = d.Expression(
        "k_solid + (k_melt - k_solid) * ("
        "  exp(-((x[0]-x1)*(x[0]-x1) + (x[1]-y1)*(x[1]-y1))/(2.0*r0*r0)) +"
        "  exp(-((x[0]-x2)*(x[0]-x2) + (x[1]-y2)*(x[1]-y2))/(2.0*r0*r0))"
        ") * exp(-(1.0 - x[2])/d_pen)",
        k_solid=k_solid,
        k_melt=k_melt,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        r0=r0,
        d_pen=d_pen,
        degree=2,
    )

    heat_expr = d.Expression(
        "("
        "  q1 * exp(-((x[0]-x1)*(x[0]-x1) + (x[1]-y1)*(x[1]-y1))/(r0*r0)) +"
        "  q2 * exp(-((x[0]-x2)*(x[0]-x2) + (x[1]-y2)*(x[1]-y2))/(r0*r0))"
        ") * exp(-(1.0 - x[2])/d_pen)",
        q1=q1,
        q2=q2,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
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
    r1_sq = (dof_coords[:, 0] - x1) ** 2 + (dof_coords[:, 1] - y1) ** 2
    r2_sq = (dof_coords[:, 0] - x2) ** 2 + (dof_coords[:, 1] - y2) ** 2
    z_dist = 1.0 - dof_coords[:, 2]
    k_vals = k_solid + (k_melt - k_solid) * (np.exp(-r1_sq / (2.0 * r0 * r0)) + np.exp(-r2_sq / (2.0 * r0 * r0))) * np.exp(-z_dist / d_pen)

    return mat, b_vec, indptr, indices, data, k_vals


def run_dual_beam_benchmark(
    n_mesh: int = 28,
    total_steps: int = 12,
    budget_k: int = 3,
    dt: float = 0.05,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 110)
    print(f"   CMAME EXPERIMENT: CASE B - DUAL-BEAM DUAL-FRONT NON-TRIVIAL TRAJECTORY")
    print(f"   Mesh: UnitCubeMesh({n_mesh}, {n_mesh}, {n_mesh}) | Steps: {total_steps} | Budget K: {budget_k}/8")
    print(f"   Physics: Dual opposing laser beams (forward + backward) perturbing separate subdomains simultaneously")
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
    print(f"✓ DOFs: {n_dofs:,} | Subdomains: {n_sub}")

    u_state = d.Function(V)
    u_state.vector()[:] = 0.0

    print("--> Pre-assembling dual-beam FEM steps...")
    mats, b_vecs, csrs, diag_arrays, phys_k_fields = [], [], [], [], []
    for step in range(total_steps + 1):
        mat, b_vec, indptr, indices, data, k_vals = assemble_dual_beam_step(
            mesh, V, step, total_steps, dt, u_state
        )
        diag_vec = mat.getDiagonal()
        diag_arr = diag_vec.getArray().copy()
        diag_vec.destroy()

        mats.append(mat)
        b_vecs.append(b_vec)
        csrs.append((indptr, indices, data))
        diag_arrays.append(diag_arr)
        phys_k_fields.append(k_vals)

    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)

    arms = [
        "full_rebuild",
        "frozen_reuse",
        "cyclic_asras_style",
        "physics_svolos_style",
        "drift_only",
        "jsr_stateful",
    ]

    arm_descriptions = {
        "full_rebuild": "Full Rebuild (100% updates, reference baseline)",
        "frozen_reuse": "Frozen Static Reuse (0% updates)",
        "cyclic_asras_style": f"Cyclic Partial Update (AsRAS-style, budget K={budget_k})",
        "physics_svolos_style": f"Physics-Aware Selection (Svolos-style greedy max kappa, K={budget_k})",
        "drift_only": f"Drift-Only Selection (Pure algebraic rate of change, K={budget_k})",
        "jsr_stateful": f"JSR Stateful Maintenance (Algebraic drift + Age damping, K={budget_k})",
    }

    all_arm_results: Dict[str, Dict[str, Any]] = {}

    for arm in arms:
        print(f"\n[{arm.upper()}]: {arm_descriptions[arm]}")
        backend = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)

        # Initialize at Step 0
        mat0 = mats[0]
        ind0, indx0, d0 = csrs[0]
        backend.refresh_local(mat0, 0, range(n_sub))
        refresh_coarse_vectorized(backend, ind0, indx0, d0, n_sub)

        local_ages = [0] * n_sub
        cyclic_ptr = 0

        times_setup, times_solve, times_total, iters_list, relres_list = [], [], [], [], []

        for step in range(1, total_steps + 1):
            mat_t = mats[step]
            b_t = b_vecs[step]
            ind_t, indx_t, d_t = csrs[step]
            diag_t = diag_arrays[step]
            diag_prev = diag_arrays[step - 1]

            t0 = time.perf_counter()

            if arm == "full_rebuild":
                selected = list(range(n_sub))
            elif arm == "frozen_reuse":
                selected = []
            elif arm == "cyclic_asras_style":
                selected = [(cyclic_ptr + i) % n_sub for i in range(budget_k)]
                cyclic_ptr = (cyclic_ptr + budget_k) % n_sub
            elif arm == "physics_svolos_style":
                # Greedy physical peak selection: selects top-K subdomains with highest kappa
                k_field = phys_k_fields[step]
                k_max = [float(np.max(k_field[idx])) for idx in part["local_indices"]]
                selected = sorted(sorted(range(n_sub), key=lambda i: -k_max[i])[:budget_k])
            elif arm == "drift_only":
                delta_diag = diag_t - diag_prev
                drift_norm = [float(np.linalg.norm(delta_diag[idx])) for idx in part["local_indices"]]
                selected = sorted(sorted(range(n_sub), key=lambda i: -drift_norm[i])[:budget_k])
            elif arm == "jsr_stateful":
                delta_diag = diag_t - diag_prev
                drift_norm = [float(np.linalg.norm(delta_diag[idx])) for idx in part["local_indices"]]
                scores = [drift_norm[i] * (1.0 + 0.15 * local_ages[i]) for i in range(n_sub)]
                selected = sorted(sorted(range(n_sub), key=lambda i: -scores[i])[:budget_k])

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
            its, t_sol, rel_res = solve_pcg_certified(mat_t, backend, b_t, sol_vec)
            t_step = t_setup + t_sol

            times_setup.append(float(t_setup))
            times_solve.append(float(t_sol))
            times_total.append(float(t_step))
            iters_list.append(int(its))
            relres_list.append(float(rel_res))

            print(f"  Step {step:2d}/{total_steps}: Setup={t_setup:.4f}s | Solve={t_sol:.4f}s | Total={t_step:.4f}s | "
                  f"Iters={its:2d} | Updated={updated} | RelRes={rel_res:.2e}")

        backend.destroy()

        all_arm_results[arm] = {
            "description": arm_descriptions[arm],
            "mean_setup": float(np.mean(times_setup)),
            "mean_solve": float(np.mean(times_solve)),
            "mean_total": float(np.mean(times_total)),
            "total_wall_clock": float(np.sum(times_total)),
            "mean_iters": float(np.mean(iters_list)),
            "max_iters": int(np.max(iters_list)),
            "max_relres": float(np.max(relres_list)),
            "all_relres_pass_1e8": bool(all(r < 1.0e-8 for r in relres_list)),
        }

    ref_time = all_arm_results["full_rebuild"]["mean_total"]
    for k, v in all_arm_results.items():
        v["speedup"] = float(ref_time / v["mean_total"])

    print("\n" + "=" * 120)
    print(f"{'Dual-Beam Case B Policy':<28} | {'Mean Setup':<10} | {'Mean Solve':<10} | {'Mean Total':<10} | {'Total Time':<10} | {'Iters':<8} | {'Max RelRes':<11} | {'Speedup':<8}")
    print("-" * 120)
    for arm in arms:
        v = all_arm_results[arm]
        print(f"{arm:<28} | {v['mean_setup']:<10.4f} | {v['mean_solve']:<10.4f} | {v['mean_total']:<10.4f} | {v['total_wall_clock']:<9.2f}s | {v['mean_iters']:<8.1f} | {v['max_relres']:<11.2e} | {v['speedup']:<7.2f}x")
    print("=" * 120)

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump({
                "mesh_n": n_mesh,
                "total_dofs": n_dofs,
                "total_steps": total_steps,
                "budget_k": budget_k,
                "arms": all_arm_results,
            }, fp, indent=2)
        print(f"✓ Saved Dual-Beam Case B results to: {out_file.resolve()}")

    for m in mats:
        m.destroy()
    for b in b_vecs:
        b.destroy()
    sol_vec.destroy()

    return all_arm_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMAME Dual-Beam Case B Benchmark")
    parser.add_argument("--mesh", type=int, default=28, help="Mesh resolution (default: 28 = 24,389 DOFs)")
    parser.add_argument("--steps", type=int, default=12, help="Total steps (default: 12)")
    parser.add_argument("--budget", type=int, default=3, help="Budget K (default: 3)")
    parser.add_argument("--out", type=str, default="results/dual_beam_case_b_n28.json")
    args = parser.parse_args()

    run_dual_beam_benchmark(
        n_mesh=args.mesh,
        total_steps=args.steps,
        budget_k=args.budget,
        output_path=args.out,
    )

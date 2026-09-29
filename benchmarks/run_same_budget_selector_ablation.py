#!/usr/bin/env python3
"""
================================================================================
CMAME Crucial Experiment 1: Same-Budget Selector Ablation Benchmark
================================================================================

Under a strictly matched, identical subdomain refresh budget (e.g. exactly K=3
or K=4 subdomains refreshed per time step), we evaluate how DIFFERENT selection
policies affect Krylov convergence, solve time, and residual correctness:

1. Random Selection: Randomly picks K subdomains (stochastic baseline).
2. Cyclic Selection: Mechanical round-robin picks K subdomains (AsRAS-style).
3. Age-Only Selection: Picks the K oldest unrefreshed subdomains (FIFO aging).
4. Drift-Only Selection: Picks the K subdomains with largest instantaneous ||Delta diag||.
5. JSR Selection: Picks the K subdomains based on stateful risk s_i = Delta_i * (1 + beta * tau_i).
6. Physics-Aware Selection: Picks the K subdomains with highest physical conductivity/temperature (Svolos-style).
7. Full Rebuild: Unconditional 100% refresh (reference upper bound on setup).
8. Frozen Reuse: 0% refresh (reference lower bound on setup).

Because all selective arms update EXACTLY the same number of subdomains K,
their Setup costs are identical. All differences in Solve time and PCG iterations
are PURELY attributable to the mathematical quality of the selection decision!
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
from benchmarks.run_cmame_fem_moving_laser_benchmark import (
    PythonPC,
    assemble_transient_fem_step,
    solve_pcg,
)


def refresh_coarse_vectorized(
    backend: mumps_module.OverlappingMUMPSSchwarzBackend,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_sub: int,
) -> float:
    """Vectorized assembly and refactorization of the Galerkin coarse matrix via np.bincount."""
    t0 = time.perf_counter()
    row_core = np.repeat(backend.core_ids, np.diff(indptr))
    col_core = backend.core_ids[indices]
    dense_coarse = np.bincount(
        row_core * n_sub + col_core, weights=data, minlength=n_sub * n_sub
    ).reshape(n_sub, n_sub)
    dense_coarse = 0.5 * (dense_coarse + dense_coarse.T)

    ia = np.zeros(n_sub + 1, dtype=np.int64)
    rows, cols, vals = [], [], []
    for i in range(n_sub):
        nz = np.flatnonzero(np.abs(dense_coarse[i]) > 0.0)
        rows.append(np.full(nz.size, i, dtype=np.int64))
        cols.append(nz.astype(np.int64))
        vals.append(dense_coarse[i, nz])
        ia[i + 1] = ia[i] + nz.size
    ja = np.concatenate(cols).astype(np.int64)
    aa = np.concatenate(vals).astype(np.float64)

    new_coarse = mumps_module.create_petsc_aij_matrix(ia, ja, aa, n_sub)
    if backend.coarse_ksp is None:
        backend.coarse_ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        backend.coarse_ksp.setType("preonly")
        pc = backend.coarse_ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        backend.coarse_ksp.setOperators(new_coarse)
        backend.coarse_ksp.setUp()
        backend.coarse_rhs = new_coarse.createVecRight()
        backend.coarse_sol = new_coarse.createVecRight()
    else:
        backend.coarse_ksp.setOperators(new_coarse)
        backend.coarse_ksp.setUp()
    new_coarse.destroy()
    return time.perf_counter() - t0


def run_ablation(
    n_mesh: int = 48,
    n_steps: int = 6,
    sub_per_axis: int = 2,
    overlap: int = 1,
    budget_k: int = 3,
    dt: float = 0.05,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 96)
    print(f"   CMAME EXPERIMENT 1: SAME-BUDGET SELECTOR ABLATION")
    print(f"   Mesh: UnitCubeMesh({n_mesh}, {n_mesh}, {n_mesh}) | Steps: {n_steps} | Subdomains: {sub_per_axis**3}")
    print(f"   Fixed Budget K: {budget_k}/{sub_per_axis**3} ({budget_k / sub_per_axis**3 * 100:.1f}%) | Seed: {seed}")
    print("=" * 96)

    np.random.seed(seed)

    # 1. Finite Element Space and Overlapping Partition
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

    # 2. Pre-assemble transient systems
    print(f"--> Pre-assembling {n_steps + 1} FEM time steps...")
    u_state = d.Function(V)
    u_state.vector()[:] = 0.0

    mats: List[PETSc.Mat] = []
    b_vecs: List[PETSc.Vec] = []
    csrs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    phys_k_fields: List[np.ndarray] = []
    diag_arrays: List[np.ndarray] = []

    for step in range(n_steps + 1):
        t_val = float(step) / max(n_steps, 1)
        mat, b_vec, indptr, indices, data, k_vals = assemble_transient_fem_step(
            mesh, V, t_val, dt, u_state
        )
        diag_vec = mat.getDiagonal()
        diag_arr = diag_vec.getArray().copy()
        diag_vec.destroy()

        mats.append(mat)
        b_vecs.append(b_vec)
        csrs.append((indptr, indices, data))
        phys_k_fields.append(k_vals)
        diag_arrays.append(diag_arr)

    # 3. Strategy Arms under Identical Budget K
    arms = [
        "full_rebuild",
        "frozen_reuse",
        "random",
        "cyclic_asras_style",
        "age_only",
        "drift_only",
        "physics_svolos_style",
        "jsr_stateful",
    ]

    arm_descriptions = {
        "full_rebuild": "Full Rebuild (100% updates, reference baseline)",
        "frozen_reuse": "Frozen Static Reuse (0% updates, reference baseline)",
        "random": f"Random Selector (Stochastic, budget K={budget_k})",
        "cyclic_asras_style": f"Cyclic Selector (AsRAS-style Round-Robin, budget K={budget_k})",
        "age_only": f"Age-Only Selector (Oldest unrefreshed, budget K={budget_k})",
        "drift_only": f"Drift-Only Selector (Instantaneous ||Delta diag||, budget K={budget_k})",
        "physics_svolos_style": f"Physics-Aware Selector (Svolos-style max physical kappa, budget K={budget_k})",
        "jsr_stateful": f"JSR Selector (Stateful Drift + Age ranking, budget K={budget_k})",
    }

    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)
    all_results: Dict[str, Dict[str, Any]] = {}

    for arm in arms:
        print(f"\n[{arm.upper()}]: {arm_descriptions[arm]}")
        backend = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)

        # Initial Step 0 setup
        backend.refresh_local(mats[0], 0, range(n_sub))
        ind0, indx0, d0 = csrs[0]
        refresh_coarse_vectorized(backend, ind0, indx0, d0, n_sub)

        # Warmup solve
        sol_vec.set(0.0)
        solve_pcg(mats[0], backend, b_vecs[0], sol_vec)

        local_ages = [0] * n_sub
        cyclic_ptr = 0

        step_setup: List[float] = []
        step_solve: List[float] = []
        step_total: List[float] = []
        step_iters: List[int] = []
        step_relres: List[float] = []
        step_updates: List[List[int]] = []

        for step in range(1, n_steps + 1):
            mat_t = mats[step]
            b_t = b_vecs[step]
            ind_t, indx_t, d_t = csrs[step]
            t0 = time.perf_counter()

            # Selection Decision
            if arm == "full_rebuild":
                selected = list(range(n_sub))
            elif arm == "frozen_reuse":
                selected = []
            elif arm == "random":
                selected = list(np.random.choice(n_sub, size=budget_k, replace=False))
            elif arm == "cyclic_asras_style":
                selected = [(cyclic_ptr + i) % n_sub for i in range(budget_k)]
                cyclic_ptr = (cyclic_ptr + budget_k) % n_sub
            elif arm == "age_only":
                ranked = sorted(range(n_sub), key=lambda i: -local_ages[i])
                selected = ranked[:budget_k]
            elif arm == "drift_only":
                delta_diag = diag_arrays[step] - diag_arrays[step - 1]
                delta_norm = [float(np.linalg.norm(delta_diag[idx])) for idx in part["local_indices"]]
                ranked = sorted(range(n_sub), key=lambda i: -delta_norm[i])
                selected = ranked[:budget_k]
            elif arm == "physics_svolos_style":
                k_field = phys_k_fields[step]
                k_max = [float(np.max(k_field[idx])) for idx in part["local_indices"]]
                ranked = sorted(range(n_sub), key=lambda i: -k_max[i])
                selected = ranked[:budget_k]
            elif arm == "jsr_stateful":
                delta_diag = diag_arrays[step] - diag_arrays[step - 1]
                delta_norm = [float(np.linalg.norm(delta_diag[idx])) for idx in part["local_indices"]]
                # Joint risk score combining algebraic drift and persistent stateful age
                scores = [delta_norm[cid] * (1.0 + 0.15 * local_ages[cid]) for cid in range(n_sub)]
                ranked = sorted(range(n_sub), key=lambda i: -scores[i])
                selected = ranked[:budget_k]

            updated = sorted(selected)
            t_local = backend.refresh_local(mat_t, step, updated) if updated else 0.0

            # Coarse update: synchronized for all updating arms (None for frozen)
            if arm != "frozen_reuse":
                t_coarse = refresh_coarse_vectorized(backend, ind_t, indx_t, d_t, n_sub)
            else:
                t_coarse = 0.0

            t_setup = time.perf_counter() - t0

            # Age state maintenance
            for cid in range(n_sub):
                if cid in selected:
                    local_ages[cid] = 0
                else:
                    local_ages[cid] += 1

            # Solve step
            sol_vec.set(0.0)
            its, t_sol, rel_res = solve_pcg(mat_t, backend, b_t, sol_vec)
            t_tot = time.perf_counter() - t0

            step_setup.append(float(t_setup))
            step_solve.append(float(t_sol))
            step_total.append(float(t_tot))
            step_iters.append(int(its))
            step_relres.append(float(rel_res))
            step_updates.append([int(x) for x in updated])

            print(f"  Step {step}: Setup={t_setup:.4f}s | Solve={t_sol:.4f}s | Total={t_tot:.4f}s | "
                  f"Iters={its:2d} | Updated={updated} | RelRes={rel_res:.2e}")

        backend.destroy()

        all_results[arm] = {
            "description": arm_descriptions[arm],
            "setup": step_setup,
            "solve": step_solve,
            "total": step_total,
            "iters": step_iters,
            "rel_res": step_relres,
            "updates": step_updates,
            "mean_setup": float(np.mean(step_setup)),
            "mean_solve": float(np.mean(step_solve)),
            "mean_total": float(np.mean(step_total)),
            "mean_iters": float(np.mean(step_iters)),
            "max_rel_res": float(np.max(step_relres)),
            "updated_count": int(len(updated)),
        }

    # Speedups relative to Full Rebuild
    ref_tot = all_results["full_rebuild"]["mean_total"]
    for arm in arms:
        all_results[arm]["speedup"] = float(ref_tot / max(all_results[arm]["mean_total"], 1.0e-9))

    # Display Clean Ablation Table
    print("\n" + "=" * 110)
    print(f"{'Selector Policy':<24} | {'Mean Setup':<10} | {'Mean Solve':<10} | {'Mean Total':<10} | {'Iters':<6} | {'Max RelRes':<10} | {'Speedup':<8}")
    print("-" * 110)
    for arm in arms:
        res = all_results[arm]
        print(f"{arm:<24} | {res['mean_setup']:<10.4f} | {res['mean_solve']:<10.4f} | {res['mean_total']:<10.4f} | "
              f"{res['mean_iters']:<6.1f} | {res['max_rel_res']:<10.2e} | {res['speedup']:<7.2f}x")
    print("=" * 110)

    # Save structured JSON
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump({
                "mesh_n": int(n_mesh),
                "total_dofs": int(n_dofs),
                "subdomains": int(n_sub),
                "budget_k": int(budget_k),
                "steps": int(n_steps),
                "results": all_results,
            }, fp, indent=2)
        print(f"\n✓ Saved ablation results to: {out_file.resolve()}")

    for m in mats:
        m.destroy()
    for b in b_vecs:
        b.destroy()
    sol_vec.destroy()

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMAME Same-Budget Selector Ablation Benchmark")
    parser.add_argument("--mesh", type=int, default=48, help="Mesh resolution (default: 48)")
    parser.add_argument("--steps", type=int, default=6, help="Time steps (default: 6)")
    parser.add_argument("--subdomains", type=int, default=2, help="Subdomains per axis (default: 2 -> 8 subdomains)")
    parser.add_argument("--budget", type=int, default=3, help="Fixed subdomain budget K per step (default: 3)")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step size dt (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out", type=str, default="results/same_budget_selector_ablation_n48.json", help="Output path")
    args = parser.parse_args()

    run_ablation(
        n_mesh=args.mesh,
        n_steps=args.steps,
        sub_per_axis=args.subdomains,
        budget_k=args.budget,
        dt=args.dt,
        seed=args.seed,
        output_path=args.out,
    )

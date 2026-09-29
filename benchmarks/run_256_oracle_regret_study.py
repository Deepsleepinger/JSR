#!/usr/bin/env python3
"""
================================================================================
CMAME Crucial Experiment 2: 256-Oracle Offline Enumeration and Regret Analysis
================================================================================

For an 8-subdomain partition, the powerset of all possible selective maintenance
actions has exact cardinality 2^8 = 256. At a representative time step where
the moving front causes localized operator perturbation, we enumerate ALL 256
possible subdomain refresh masks S subset of {0, ..., 7}.

For each mask S in {0, ..., 255}:
1. Selectively refactorize subdomains in S via MUMPS and update the coarse grid.
2. Solve the linear system using PCG to certified residual <= 1e-8.
3. Record exact T_setup(S), T_solve(S), T_total(S), and PCG iterations K(S).

We identify the true global empirical oracle:
    S* = argmin_{S} T_total(S),  T* = min_{S} T_total(S)

We then evaluate the algorithmic Regret of JSR and other baseline policies:
    Regret(Policy) = (T_total(Policy) - T*) / T* * 100%

This provides an exact, combinatorial ground-truth benchmark proving how close
JSR's algebraic selection gets to the theoretical optimal subset without needing
to solve a combinatorial search.
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
from benchmarks.run_cmame_fem_moving_laser_benchmark import (
    PythonPC,
    assemble_transient_fem_step,
    solve_pcg,
)
from benchmarks.run_same_budget_selector_ablation import refresh_coarse_vectorized


def run_oracle_study(
    n_mesh: int = 24,
    target_step: int = 2,
    total_steps: int = 4,
    dt: float = 0.05,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 96)
    print(f"   CMAME EXPERIMENT 2: 256-ORACLE OFFLINE GLOBAL OPTIMAL & REGRET STUDY")
    print(f"   Mesh: UnitCubeMesh({n_mesh}, {n_mesh}, {n_mesh}) | Target Step: {target_step}/{total_steps}")
    print(f"   Subdomains: 8 -> Exact 2^8 = 256 Combinatorial Search")
    print("=" * 96)

    # 1. Finite Element Space and Overlapping Partition
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

    # 2. Assemble up to target step
    u_state = d.Function(V)
    u_state.vector()[:] = 0.0

    mats, b_vecs, csrs, diag_arrays, phys_k_fields = [], [], [], [], []
    for step in range(target_step + 1):
        t_val = float(step) / max(total_steps, 1)
        mat, b_vec, indptr, indices, data, k_vals = assemble_transient_fem_step(
            mesh, V, t_val, dt, u_state
        )
        diag_vec = mat.getDiagonal()
        diag_arr = diag_vec.getArray().copy()
        diag_vec.destroy()

        mats.append(mat)
        b_vecs.append(b_vec)
        csrs.append((indptr, indices, data))
        diag_arrays.append(diag_arr)
        phys_k_fields.append(k_vals)

    mat_t = mats[target_step]
    b_t = b_vecs[target_step]
    ind_t, indx_t, d_t = csrs[target_step]
    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)

    # Compute selector decisions at target step
    # 1. JSR decision
    delta_diag = diag_arrays[target_step] - diag_arrays[target_step - 1]
    delta_norm = [float(np.linalg.norm(delta_diag[idx])) for idx in part["local_indices"]]
    # At step 2, assuming 3 subdomains were refreshed at step 1
    scores = delta_norm.copy()
    tot_score = sum(scores)
    ranked_jsr = sorted(range(n_sub), key=lambda i: -scores[i])
    cum = 0.0
    jsr_selected = set()
    for cid in ranked_jsr:
        jsr_selected.add(cid)
        cum += scores[cid]
        if cum >= 0.85 * tot_score:
            break
    jsr_mask = sum(1 << cid for cid in jsr_selected)

    # 2. Physics-aware (Svolos-style) decision
    k_field = phys_k_fields[target_step]
    k_tol = 1.0 + 0.15 * (10.0 - 1.0)
    svolos_selected = [cid for cid, idx in enumerate(part["local_indices"]) if float(np.max(k_field[idx])) > k_tol]
    svolos_mask = sum(1 << cid for cid in svolos_selected)

    # 3. Cyclic mask (e.g. [0, 1, 2])
    cyclic_mask = (1 << 0) | (1 << 1) | (1 << 2)

    print(f"\n--> JSR chosen subset: {sorted(jsr_selected)} (mask: {jsr_mask})")
    print(f"--> Svolos-style chosen subset: {sorted(svolos_selected)} (mask: {svolos_mask})")

    # 3. Exhaustive 256 Enumeration
    print(f"\n--> Enumerating all 256 subsets on Step {target_step}...")
    all_evals = []
    t_start_all = time.perf_counter()

    for mask in range(256):
        sub_list = [i for i in range(8) if (mask & (1 << i))]

        # Fresh backend setup from step 0 baseline
        backend = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
        backend.refresh_local(mats[0], 0, range(n_sub))
        ind0, indx0, d0 = csrs[0]
        refresh_coarse_vectorized(backend, ind0, indx0, d0, n_sub)

        # Apply mask on target step
        t0 = time.perf_counter()
        t_loc = backend.refresh_local(mat_t, target_step, sub_list) if sub_list else 0.0
        t_crs = refresh_coarse_vectorized(backend, ind_t, indx_t, d_t, n_sub)
        t_setup = time.perf_counter() - t0

        sol_vec.set(0.0)
        its, t_sol, rel_res = solve_pcg(mat_t, backend, b_t, sol_vec)
        t_tot = t_setup + t_sol

        backend.destroy()

        entry = {
            "mask": mask,
            "subdomains": sub_list,
            "cardinality": len(sub_list),
            "setup": t_setup,
            "solve": t_sol,
            "total": t_tot,
            "iters": its,
            "rel_res": rel_res,
        }
        all_evals.append(entry)

        if (mask + 1) % 64 == 0:
            print(f"    Completed {mask + 1}/256 evaluations ({time.perf_counter() - t_start_all:.1f}s elapsed)...")

    # Identify Oracle
    all_evals.sort(key=lambda x: x["total"])
    oracle = all_evals[0]
    full_rebuild_eval = [e for e in all_evals if e["mask"] == 255][0]
    frozen_eval = [e for e in all_evals if e["mask"] == 0][0]
    jsr_eval = [e for e in all_evals if e["mask"] == jsr_mask][0]
    svolos_eval = [e for e in all_evals if e["mask"] == svolos_mask][0]
    cyclic_eval = [e for e in all_evals if e["mask"] == cyclic_mask][0]

    # Calculate Regrets
    t_opt = oracle["total"]
    for e in all_evals:
        e["regret_pct"] = (e["total"] - t_opt) / t_opt * 100.0

    print("\n" + "=" * 100)
    print(f"{'Strategy / Mask':<26} | {'Subset':<16} | {'Setup (s)':<10} | {'Solve (s)':<10} | {'Total (s)':<10} | {'Iters':<6} | {'Regret':<8}")
    print("-" * 100)
    print(f"{'TRUE GLOBAL ORACLE':<26} | {str(oracle['subdomains']):<16} | {oracle['setup']:<10.4f} | {oracle['solve']:<10.4f} | {oracle['total']:<10.4f} | {oracle['iters']:<6d} | {oracle['regret_pct']:<7.1f}%")
    print(f"{'JSR Algebraic Selection':<26} | {str(jsr_eval['subdomains']):<16} | {jsr_eval['setup']:<10.4f} | {jsr_eval['solve']:<10.4f} | {jsr_eval['total']:<10.4f} | {jsr_eval['iters']:<6d} | {jsr_eval['regret_pct']:<7.1f}%")
    print(f"{'Svolos-style Physics':<26} | {str(svolos_eval['subdomains']):<16} | {svolos_eval['setup']:<10.4f} | {svolos_eval['solve']:<10.4f} | {svolos_eval['total']:<10.4f} | {svolos_eval['iters']:<6d} | {svolos_eval['regret_pct']:<7.1f}%")
    print(f"{'Cyclic (AsRAS-style)':<26} | {str(cyclic_eval['subdomains']):<16} | {cyclic_eval['setup']:<10.4f} | {cyclic_eval['solve']:<10.4f} | {cyclic_eval['total']:<10.4f} | {cyclic_eval['iters']:<6d} | {cyclic_eval['regret_pct']:<7.1f}%")
    print(f"{'Full Rebuild (100%)':<26} | {str(full_rebuild_eval['subdomains']):<16} | {full_rebuild_eval['setup']:<10.4f} | {full_rebuild_eval['solve']:<10.4f} | {full_rebuild_eval['total']:<10.4f} | {full_rebuild_eval['iters']:<6d} | {full_rebuild_eval['regret_pct']:<7.1f}%")
    print(f"{'Frozen Static Reuse (0%)':<26} | {str(frozen_eval['subdomains']):<16} | {frozen_eval['setup']:<10.4f} | {frozen_eval['solve']:<10.4f} | {frozen_eval['total']:<10.4f} | {frozen_eval['iters']:<6d} | {frozen_eval['regret_pct']:<7.1f}%")
    print("=" * 100)

    # Save structured results
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump({
                "mesh_n": n_mesh,
                "total_dofs": n_dofs,
                "target_step": target_step,
                "oracle": oracle,
                "jsr": jsr_eval,
                "svolos": svolos_eval,
                "cyclic": cyclic_eval,
                "full_rebuild": full_rebuild_eval,
                "frozen": frozen_eval,
                "all_256_evals": all_evals,
            }, fp, indent=2)
        print(f"\n✓ Saved 256-oracle evaluation to: {out_file.resolve()}")

    for m in mats:
        m.destroy()
    for b in b_vecs:
        b.destroy()
    sol_vec.destroy()

    return {
        "oracle": oracle,
        "jsr": jsr_eval,
        "svolos": svolos_eval,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMAME 256-Oracle Offline Enumeration Benchmark")
    parser.add_argument("--mesh", type=int, default=24, help="Mesh resolution (default: 24)")
    parser.add_argument("--target_step", type=int, default=2, help="Target evaluation step (default: 2)")
    parser.add_argument("--steps", type=int, default=4, help="Total steps trajectory (default: 4)")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step size dt (default: 0.05)")
    parser.add_argument("--out", type=str, default="results/oracle_256_regret_study_n24.json", help="Output path")
    args = parser.parse_args()

    run_oracle_study(
        n_mesh=args.mesh,
        target_step=args.target_step,
        total_steps=args.steps,
        dt=args.dt,
        output_path=args.out,
    )

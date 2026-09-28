#!/usr/bin/env python3
"""
================================================================================
Comprehensive Multi-Baseline Benchmark for CMAME Submission:
Head-to-Head Comparison: Full Rebuild, Blind Reuse, AsRAS (Berenguer 2015),
Svolos Physical Selection (Svolos 2020), and JSR (Proposed Stateful Maintenance).
================================================================================

This benchmark evaluates 6 distinct preconditioner maintenance strategies for
sequences of linear systems arising from 3-D transient PDEs with localized moving fronts:

1. FULL_REBUILD: Standard industry practice; refactorizes all subdomains and coarse grid (100% setup).
2. BLIND_REUSE: Static freeze; zero setup past t=0, but suffers spectral degradation.
3. ASRAS_CYCLIC: Berenguer & Tromeur-Dervout (Computers & Fluids 2015); blind round-robin partial updates.
4. SVOLOS_1LEVEL: Svolos et al. (JCP 2020); physical-event front selection on single-level Schwarz (no coarse grid).
5. SVOLOS_2LEVEL: Svolos-style physical front selection extended with coarse grid, but unbudgeted and memoryless.
6. JSR_ADAPTIVE: Proposed; physics-agnostic diagonal drift proxy + mass-alpha budget + joint coarse synchronization + stateful age tracking.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from petsc4py import PETSc
from jsr import partition as part_module
from jsr import backend_mumps as mumps_module


def build_3d_laplacian_csr(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a 7-point 3-D Laplacian grid on [0, 1]^3 with Dirichlet boundary."""
    N = int(n) ** 3
    indptr = np.zeros(N + 1, dtype=np.int64)
    indices = []
    values = []
    diagonal = 6.0
    h = 1.0 / float(n)

    coords = np.zeros((N, 3), dtype=np.float64)
    for k in range(n):
        for j in range(n):
            for i in range(n):
                row = i + n * (j + n * k)
                coords[row] = [(i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h]
                entries = [(row, diagonal)]
                if i > 0:
                    entries.append((row - 1, -1.0))
                if i + 1 < n:
                    entries.append((row + 1, -1.0))
                if j > 0:
                    entries.append((row - n, -1.0))
                if j + 1 < n:
                    entries.append((row + n, -1.0))
                if k > 0:
                    entries.append((row - n * n, -1.0))
                if k + 1 < n:
                    entries.append((row + n * n, -1.0))
                entries.sort(key=lambda pair: pair[0])
                indices.extend(pair[0] for pair in entries)
                values.extend(pair[1] for pair in entries)
                indptr[row + 1] = len(values)

    return (
        coords,
        indptr,
        np.asarray(indices, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


def apply_moving_front(
    coords: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    base_values: np.ndarray,
    t: float,
    amplitude: float = 8.0,
    width: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modulate diagonal entries based on a localized moving Gaussian thermal/conductivity front.
    Returns (perturbed_values, perturbation_field).
    """
    center = np.array([0.20 + 0.60 * t, 0.20 + 0.60 * t, 0.20 + 0.60 * t], dtype=np.float64)
    dist_sq = np.sum(((coords - center) / width) ** 2, axis=1)
    perturbation = amplitude * np.exp(-dist_sq)

    vals = base_values.copy()
    for row in range(indptr.size - 1):
        lo, hi = int(indptr[row]), int(indptr[row + 1])
        for p in range(lo, hi):
            if indices[p] == row:
                vals[p] = base_values[p] + perturbation[row]
    return vals, perturbation


class PythonPC(object):
    """Bridge PETSc KSP to our Two-Level / One-Level Schwarz Backend."""

    def __init__(self, backend_ctx: mumps_module.OverlappingMUMPSSchwarzBackend):
        self.backend = backend_ctx

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        x_arr = x.getArray(readonly=True)
        y_arr = y.getArray()
        self.backend.apply(x_arr, y_arr)


def solve_pcg(
    mat: PETSc.Mat,
    backend_ctx: mumps_module.OverlappingMUMPSSchwarzBackend,
    b_vec: PETSc.Vec,
    x_vec: PETSc.Vec,
    rtol: float = 1.0e-8,
    max_it: int = 150,
) -> Tuple[int, float, float]:
    """Solve system A x = b using preconditioned CG with external residual verification."""
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    ksp.setOperators(mat)
    ksp.setType("cg")
    ksp.setTolerances(rtol=rtol, atol=1.0e-14, max_it=max_it)

    pc = ksp.getPC()
    pc.setType("python")
    py_pc = PythonPC(backend_ctx)
    pc.setPythonContext(py_pc)
    pc.setUp()

    t0 = time.perf_counter()
    ksp.solve(b_vec, x_vec)
    t_solve = time.perf_counter() - t0

    its = int(ksp.getIterationNumber())

    # Certified true residual computation: ||b - A x|| / ||b||
    r_vec = b_vec.duplicate()
    mat.mult(x_vec, r_vec)
    r_vec.aypx(-1.0, b_vec)
    norm_res = float(r_vec.norm(PETSc.NormType.NORM_2))
    norm_b = float(b_vec.norm(PETSc.NormType.NORM_2))
    rel_res = norm_res / max(norm_b, 1.0e-14)

    r_vec.destroy()
    ksp.destroy()
    return its, t_solve, rel_res


def run_benchmark(
    n_mesh: int = 24,
    n_steps: int = 4,
    sub_per_axis: int = 2,
    overlap: int = 1,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    n_dofs = n_mesh ** 3
    print("=" * 84)
    print(f"   CMAME MULTI-BASELINE BENCHMARK: PRIOR ART HEAD-TO-HEAD COMPARISON")
    print(f"   Mesh: {n_mesh}x{n_mesh}x{n_mesh} = {n_dofs} DOFs | "
          f"Subdomains: {sub_per_axis}^3 = {sub_per_axis**3} | Overlap: {overlap} | Steps: {n_steps}")
    print("=" * 84)

    # 1. Geometry and Partition
    coords, indptr, indices, base_values = build_3d_laplacian_csr(n_mesh)
    part = part_module.create_3d_overlapping_partition(
        coordinates=coords,
        grid=(sub_per_axis, sub_per_axis, sub_per_axis),
        overlap_layers=overlap,
        mesh_n=n_mesh,
    )
    n_sub = int(part["subdomain_count"])
    print(f"✓ Overlapping partition: {n_sub} subdomains, mean size: {np.mean([len(idx) for idx in part['local_indices']]):.0f} DOFs")

    # Locate diagonal offsets in CSR
    diag_ptrs = np.zeros(n_dofs, dtype=np.int64)
    for row in range(n_dofs):
        lo, hi = int(indptr[row]), int(indptr[row + 1])
        for p in range(lo, hi):
            if indices[p] == row:
                diag_ptrs[row] = p
                break

    # 2. Pre-generate trajectory of evolving linear systems
    print(f"--> Generating moving front trajectory across {n_steps + 1} steps...")
    states_data = []
    perturbations = []
    for step in range(n_steps + 1):
        t_val = float(step) / max(n_steps, 1)
        vals, pert = apply_moving_front(coords, indptr, indices, base_values, t_val)
        states_data.append(vals)
        perturbations.append(pert)

    rhs_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)
    rhs_vec.set(1.0)
    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)

    # 3. Define the Comparative Strategy Arms
    arms = [
        "full_rebuild",
        "blind_reuse",
        "asras_cyclic",
        "svolos_1level",
        "svolos_2level",
        "jsr_adaptive",
    ]

    arm_descriptions = {
        "full_rebuild": "Full Rebuild (Industry standard, 100% updates, 2-Level)",
        "blind_reuse": "Blind Reuse (Frozen preconditioner past t=0, 2-Level)",
        "asras_cyclic": "AsRAS-like Cyclic (Berenguer 2015, Round-Robin 25-30% updates, 2-Level)",
        "svolos_1level": "Svolos-like 1-Level (Svolos 2020, Physical front selection, NO coarse space)",
        "svolos_2level": "Svolos-like 2-Level (Svolos 2020 + Coarse, Unbudgeted physical front selection)",
        "jsr_adaptive": "JSR (Proposed, Physics-agnostic drift + Mass-alpha budget + Joint Coarse)",
    }

    all_results: Dict[str, Dict[str, Any]] = {}

    for arm in arms:
        print(f"\n[{arm.upper()}] : {arm_descriptions[arm]}")
        backend_ctx = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)

        # Step 0 initial factorization
        mat0 = mumps_module.create_petsc_aij_matrix(indptr, indices, states_data[0], n_dofs)
        backend_ctx.refresh_local(mat0, 0, range(n_sub))
        if arm != "svolos_1level":
            backend_ctx.refresh_coarse(indptr, indices, states_data[0], 0)

        # Warmup solve
        sol_vec.set(0.0)
        solve_pcg(mat0, backend_ctx, rhs_vec, sol_vec)
        mat0.destroy()

        step_setup: List[float] = []
        step_solve: List[float] = []
        step_total: List[float] = []
        step_iters: List[int] = []
        step_relres: List[float] = []
        step_updated_subdomains: List[int] = []

        # Cyclic index pointer for AsRAS
        cyclic_ptr = 0
        cyclic_k = max(1, int(math.ceil(0.25 * n_sub)))

        # Age counters for JSR
        local_ages = [0] * n_sub
        coarse_age = 0

        for step in range(1, n_steps + 1):
            mat_t = mumps_module.create_petsc_aij_matrix(indptr, indices, states_data[step], n_dofs)
            t_step_start = time.perf_counter()
            t_setup = 0.0
            updated_subdomains = []

            if arm == "full_rebuild":
                updated_subdomains = list(range(n_sub))
                t_setup += backend_ctx.refresh_local(mat_t, step, updated_subdomains)
                t_setup += backend_ctx.refresh_coarse(indptr, indices, states_data[step], step)

            elif arm == "blind_reuse":
                updated_subdomains = []
                t_setup = 0.0

            elif arm == "asras_cyclic":
                # Round-robin selection: update cyclic_k subdomains in order
                updated_subdomains = [
                    (cyclic_ptr + i) % n_sub for i in range(cyclic_k)
                ]
                cyclic_ptr = (cyclic_ptr + cyclic_k) % n_sub
                t_setup += backend_ctx.refresh_local(mat_t, step, updated_subdomains)
                # In standard cyclic AsRAS, coarse grid is either lagged or rebuilt
                t_setup += backend_ctx.refresh_coarse(indptr, indices, states_data[step], step)

            elif arm == "svolos_1level":
                # Svolos 2020: Physical crack/front intersection selection, 1-Level (NO coarse grid)
                pert_field = perturbations[step]
                pert_max = float(np.max(pert_field))
                active_tol = 0.10 * pert_max
                updated_subdomains = []
                for cid, idx in enumerate(part["local_indices"]):
                    if float(np.max(pert_field[idx])) > active_tol:
                        updated_subdomains.append(cid)
                t_setup += backend_ctx.refresh_local(mat_t, step, updated_subdomains)
                # Notice: NO refresh_coarse()! coarse_ksp remains None (true 1-level)

            elif arm == "svolos_2level":
                # Svolos 2020 concept + Coarse grid, but unbudgeted physical threshold & no age lifecycle
                pert_field = perturbations[step]
                pert_max = float(np.max(pert_field))
                active_tol = 0.10 * pert_max
                updated_subdomains = []
                for cid, idx in enumerate(part["local_indices"]):
                    if float(np.max(pert_field[idx])) > active_tol:
                        updated_subdomains.append(cid)
                t_setup += backend_ctx.refresh_local(mat_t, step, updated_subdomains)
                t_setup += backend_ctx.refresh_coarse(indptr, indices, states_data[step], step)

            elif arm == "jsr_adaptive":
                # JSR: Algebraic drift proxy + mass-alpha budget (90% Pareto) + age-2 ceiling + joint coarse sync
                t_mon_start = time.perf_counter()
                delta_diag = (states_data[step] - states_data[step - 1])[diag_ptrs]
                delta_norm = []
                for cid, idx in enumerate(part["local_indices"]):
                    d_norm = float(np.linalg.norm(delta_diag[idx]))
                    delta_norm.append(d_norm)
                t_mon = time.perf_counter() - t_mon_start

                tot_drift = sum(delta_norm)
                selected = set()
                if tot_drift > 1.0e-12:
                    ranked = sorted(range(n_sub), key=lambda i: -delta_norm[i])
                    cum = 0.0
                    for cid in ranked:
                        selected.add(cid)
                        cum += delta_norm[cid]
                        if cum >= 0.90 * tot_drift:
                            break

                # Enforce stateful age <= 2 (prevent chronic drift accumulation)
                for cid in range(n_sub):
                    if local_ages[cid] >= 2:
                        selected.add(cid)

                updated_subdomains = sorted(selected)
                t_setup += t_mon
                t_setup += backend_ctx.refresh_local(mat_t, step, updated_subdomains)
                t_setup += backend_ctx.refresh_coarse(indptr, indices, states_data[step], step)

                # Update lifecycle ages
                for cid in range(n_sub):
                    if cid in selected:
                        local_ages[cid] = 0
                    else:
                        local_ages[cid] += 1
                coarse_age = 0

            # Solve preconditioned system
            sol_vec.set(0.0)
            its, t_solve, rel_res = solve_pcg(mat_t, backend_ctx, rhs_vec, sol_vec)
            t_total = time.perf_counter() - t_step_start

            step_setup.append(t_setup)
            step_solve.append(t_solve)
            step_total.append(t_total)
            step_iters.append(its)
            step_relres.append(rel_res)
            step_updated_subdomains.append(len(updated_subdomains))

            mat_t.destroy()

        backend_ctx.destroy()

        all_results[arm] = {
            "description": arm_descriptions[arm],
            "setup": step_setup,
            "solve": step_solve,
            "total": step_total,
            "iters": step_iters,
            "rel_res": step_relres,
            "updated_count": step_updated_subdomains,
            "mean_setup": float(np.mean(step_setup)),
            "mean_solve": float(np.mean(step_solve)),
            "mean_total": float(np.mean(step_total)),
            "mean_iters": float(np.mean(step_iters)),
            "max_rel_res": float(np.max(step_relres)),
            "mean_updated_ratio": float(np.mean(step_updated_subdomains)) / float(n_sub),
        }

        print(f"    Mean Setup: {all_results[arm]['mean_setup']:.4f}s | "
              f"Mean Solve: {all_results[arm]['mean_solve']:.4f}s | "
              f"Mean Total: {all_results[arm]['mean_total']:.4f}s | "
              f"Mean Iter: {all_results[arm]['mean_iters']:.1f} | "
              f"Updated: {all_results[arm]['mean_updated_ratio']*100:.1f}% | "
              f"Max RelRes: {all_results[arm]['max_rel_res']:.2e}")

    # Calculate Speedups relative to Full Rebuild
    ref_tot = all_results["full_rebuild"]["mean_total"]
    for arm in arms:
        all_results[arm]["speedup"] = float(ref_tot / max(all_results[arm]["mean_total"], 1.0e-9))

    # Display Executive Summary
    print("\n" + "=" * 100)
    print(f"{'Arm':<20} | {'Mean Setup':<10} | {'Mean Solve':<10} | {'Mean Total':<10} | {'Mean Iter':<9} | {'Update %':<8} | {'Speedup':<8}")
    print("-" * 100)
    for arm in arms:
        res = all_results[arm]
        print(f"{arm:<20} | {res['mean_setup']:<10.4f} | {res['mean_solve']:<10.4f} | {res['mean_total']:<10.4f} | {res['mean_iters']:<9.1f} | {res['mean_updated_ratio']*100:<7.1f}% | {res['speedup']:<8.2f}x")
    print("=" * 100)

    # Save structured JSON
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "mesh_n": n_mesh,
            "total_dofs": n_dofs,
            "subdomains": n_sub,
            "overlap": overlap,
            "steps": n_steps,
            "results": all_results,
        }
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump(summary_payload, fp, indent=2)
        print(f"\n✓ Detailed results saved to: {out_file.resolve()}")

    rhs_vec.destroy()
    sol_vec.destroy()
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMAME Multi-Baseline Prior Art Benchmark")
    parser.add_argument("--mesh", type=int, default=24, help="Grid points per axis (default: 24 = 13824 DOFs)")
    parser.add_argument("--steps", type=int, default=4, help="Time steps (default: 4)")
    parser.add_argument("--subdomains", type=int, default=2, help="Subdomains per axis (default: 2 -> 8 subdomains)")
    parser.add_argument("--overlap", type=int, default=1, help="Overlap layers (default: 1)")
    parser.add_argument("--out", type=str, default="results/cmame_prior_art_baselines_results.json", help="Output JSON path")
    args = parser.parse_args()

    run_benchmark(
        n_mesh=args.mesh,
        n_steps=args.steps,
        sub_per_axis=args.subdomains,
        overlap=args.overlap,
        output_path=args.out,
    )

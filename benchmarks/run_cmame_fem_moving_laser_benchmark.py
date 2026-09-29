#!/usr/bin/env python3
"""
================================================================================
CMAME Realistic 3D Finite Element (FEM) Multi-Baseline Benchmark
Transient Heat Conduction with Moving Laser Melt Pool / Thermal Phase Change
================================================================================

Head-to-head comparison of 6 preconditioner maintenance strategies for sequences
of linear systems arising from genuine 3D continuous Galerkin finite element
discretizations (assembled via FEniCS / PETSc / MUMPS):

1. FULL_REBUILD: Standard industry practice; refactorizes all subdomains and coarse grid (100% setup).
2. BLIND_REUSE: Static freeze; zero setup past t=0, but suffers spectral degradation.
3. ASRAS_CYCLIC: Berenguer & Tromeur-Dervout (Computers & Fluids 2015); blind round-robin partial updates.
4. SVOLOS_1LEVEL: Svolos et al. (JCP 2020); physical melt-pool front selection on single-level Schwarz (no coarse grid).
5. SVOLOS_2LEVEL: Svolos-style physical front selection extended with coarse grid, but unbudgeted and memoryless.
6. JSR_ADAPTIVE: Proposed; physics-agnostic diagonal drift proxy + mass-alpha budget + joint coarse synchronization + stateful age tracking.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

# Ensure FEniCS ABI and dynamic linker paths are set
os.environ.setdefault("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
os.environ.setdefault("PKG_CONFIG_PATH", "/mnt/h/CodexLinux/stokes-r3/envs/stokes-fenics-2020-abi6-r3/lib/pkgconfig")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dolfin as d
from petsc4py import PETSc
from jsr import partition as part_module
from jsr import backend_mumps as mumps_module


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


def assemble_transient_fem_step(
    mesh: d.Mesh,
    V: d.FunctionSpace,
    t_val: float,
    dt: float,
    u_prev: d.Function,
    rho_cp: float = 10.0,
    k_solid: float = 1.0,
    k_melt: float = 10.0,
    r0: float = 0.12,
    d_pen: float = 0.18,
    q0: float = 50.0,
) -> Tuple[PETSc.Mat, PETSc.Vec, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble the 3D continuous Galerkin finite element system at time step t_val:
    rho_cp / dt * (u - u_prev) - div(k(x, t) grad(u)) = q(x, t).
    """
    u = d.TrialFunction(V)
    v = d.TestFunction(V)

    # Laser beam coordinates moving along diagonal scan track
    xc = 0.20 + 0.60 * t_val
    yc = 0.20 + 0.60 * t_val
    zc = 1.0

    # Nonlinear / heterogeneous thermal conductivity in melt pool
    # k(x, t) = k_solid + (k_melt - k_solid) * exp(-dist^2 / (2 r0^2)) * exp(-(1-z)/d_pen)
    kappa_expr = d.Expression(
        "k_solid + (k_melt - k_solid) * exp(-((x[0]-xc)*(x[0]-xc) + (x[1]-yc)*(x[1]-yc))/(2.0*r0*r0)) * exp(-(1.0 - x[2])/d_pen)",
        k_solid=k_solid,
        k_melt=k_melt,
        xc=xc,
        yc=yc,
        r0=r0,
        d_pen=d_pen,
        degree=2,
    )

    # Volumetric laser heat source
    heat_expr = d.Expression(
        "q0 * exp(-((x[0]-xc)*(x[0]-xc) + (x[1]-yc)*(x[1]-yc))/(r0*r0)) * exp(-(1.0 - x[2])/d_pen)",
        q0=q0,
        xc=xc,
        yc=yc,
        r0=r0,
        d_pen=d_pen,
        degree=2,
    )

    a = (kappa_expr * d.inner(d.grad(u), d.grad(v)) + d.Constant(rho_cp / dt) * u * v) * d.dx
    L = (d.Constant(rho_cp / dt) * u_prev + heat_expr) * v * d.dx

    # Substrate heat sink at bottom surface z = 0
    bc = d.DirichletBC(V, d.Constant(0.0), "on_boundary && x[2] < 1.0e-4")

    A_dolfin, b_dolfin = d.assemble_system(a, L, bc)
    mat = d.as_backend_type(A_dolfin).mat()
    b_vec = d.as_backend_type(b_dolfin).vec()

    # Extract CSR representation for coarse grid Galerkin assembly
    indptr, indices, data = mat.getValuesCSR()

    # Evaluate physical conductivity at DOF coordinates for Svolos physical detection
    dof_coords = V.tabulate_dof_coordinates()
    r_sq = (dof_coords[:, 0] - xc) ** 2 + (dof_coords[:, 1] - yc) ** 2
    z_dist = 1.0 - dof_coords[:, 2]
    k_vals = k_solid + (k_melt - k_solid) * np.exp(-r_sq / (2.0 * r0 * r0)) * np.exp(-z_dist / d_pen)

    return mat, b_vec, indptr, indices, data, k_vals


def run_fem_benchmark(
    n_mesh: int = 48,
    n_steps: int = 6,
    sub_per_axis: int = 2,
    overlap: int = 1,
    dt: float = 0.05,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 88)
    print("   CMAME REALISTIC 3D FINITE ELEMENT (FEM) MULTI-BASELINE BENCHMARK")
    print(f"   Mesh: UnitCubeMesh({n_mesh}, {n_mesh}, {n_mesh}) | Steps: {n_steps} | dt: {dt}")
    print(f"   Subdomains: {sub_per_axis}^3 = {sub_per_axis**3} | Overlap: {overlap} layer(s)")
    print("=" * 88)

    # 1. Finite Element Space and Coordinates
    t0_mesh = time.perf_counter()
    mesh = d.UnitCubeMesh(n_mesh, n_mesh, n_mesh)
    V = d.FunctionSpace(mesh, "Lagrange", 1)
    n_dofs = V.dim()
    coords = V.tabulate_dof_coordinates()
    print(f"✓ FEniCS 3D Lagrange P1 Space: {n_dofs:,} DOFs assembled in {time.perf_counter() - t0_mesh:.3f}s")

    # 2. Domain Partitioning
    part = part_module.create_3d_overlapping_partition(
        coordinates=coords,
        grid=(sub_per_axis, sub_per_axis, sub_per_axis),
        overlap_layers=overlap,
        mesh_n=n_mesh,
    )
    n_sub = int(part["subdomain_count"])
    mean_sub_dofs = np.mean([len(idx) for idx in part["local_indices"]])
    print(f"✓ Overlapping partition: {n_sub} subdomains, mean size: {mean_sub_dofs:.0f} DOFs/subdomain")

    # 3. Assemble trajectory of transient systems
    print(f"\n--> Pre-assembling {n_steps + 1} transient FEM time steps (Moving Laser Melt Pool)...")
    u_state = d.Function(V)
    u_state.vector()[:] = 0.0  # Initial ambient temperature T=0

    mats: List[PETSc.Mat] = []
    b_vecs: List[PETSc.Vec] = []
    csrs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    phys_k_fields: List[np.ndarray] = []
    diag_arrays: List[np.ndarray] = []

    for step in range(n_steps + 1):
        t_val = float(step) / max(n_steps, 1)
        t_asm_0 = time.perf_counter()
        mat, b_vec, indptr, indices, data, k_vals = assemble_transient_fem_step(
            mesh, V, t_val, dt, u_state
        )
        t_asm = time.perf_counter() - t_asm_0

        # Extract diagonal vector for JSR drift proxy
        diag_vec = mat.getDiagonal()
        diag_arr = diag_vec.getArray().copy()
        diag_vec.destroy()

        mats.append(mat)
        b_vecs.append(b_vec)
        csrs.append((indptr, indices, data))
        phys_k_fields.append(k_vals)
        diag_arrays.append(diag_arr)

        if step == 0:
            nz = int(mat.getInfo()["nz_used"])
            print(f"    Step 0: {n_dofs:,} DOFs, {nz:,} nonzeros, assembled in {t_asm:.3f}s")
        else:
            print(f"    Step {step}/{n_steps}: t={t_val:.2f}, laser moving, assembled in {t_asm:.3f}s")

    # 4. Define Comparison Arms
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
        "asras_cyclic": "AsRAS-like Cyclic (Berenguer 2015, Round-Robin 25% updates, 2-Level)",
        "svolos_1level": "Svolos-like 1-Level (Svolos 2020, Physical melt-pool selection, NO coarse space)",
        "svolos_2level": "Svolos-like 2-Level (Svolos 2020 + Coarse, Unbudgeted physical selection)",
        "jsr_adaptive": "JSR (Proposed, Physics-agnostic drift + Mass-alpha budget + Joint Coarse)",
    }

    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)
    all_results: Dict[str, Dict[str, Any]] = {}

    for arm in arms:
        print(f"\n================================================================================")
        print(f"[{arm.upper()}] : {arm_descriptions[arm]}")
        print(f"================================================================================")

        backend_ctx = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)

        # Step 0 initial factorization
        mat0 = mats[0]
        indptr0, indices0, data0 = csrs[0]
        backend_ctx.refresh_local(mat0, 0, range(n_sub))
        if arm != "svolos_1level":
            backend_ctx.refresh_coarse(indptr0, indices0, data0, 0)

        # Warmup solve on step 0
        sol_vec.set(0.0)
        solve_pcg(mat0, backend_ctx, b_vecs[0], sol_vec)

        step_setup: List[float] = []
        step_solve: List[float] = []
        step_total: List[float] = []
        step_iters: List[int] = []
        step_relres: List[float] = []
        step_updated_subdomains: List[int] = []

        # Cyclic index pointer for AsRAS
        cyclic_ptr = 0
        cyclic_k = max(1, int(math.ceil(0.25 * n_sub)))

        # Stateful age tracking for JSR
        local_ages = [0] * n_sub

        for step in range(1, n_steps + 1):
            mat_t = mats[step]
            b_vec_t = b_vecs[step]
            indptr_t, indices_t, data_t = csrs[step]

            t_step_start = time.perf_counter()
            t_setup = 0.0
            updated_subdomains: List[int] = []

            if arm == "full_rebuild":
                updated_subdomains = list(range(n_sub))
                t_setup += backend_ctx.refresh_local(mat_t, step, updated_subdomains)
                t_setup += backend_ctx.refresh_coarse(indptr_t, indices_t, data_t, step)

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
                t_setup += backend_ctx.refresh_coarse(indptr_t, indices_t, data_t, step)

            elif arm == "svolos_1level":
                # Svolos 2020: Physical melt pool threshold selection, 1-Level (NO coarse space)
                k_field = phys_k_fields[step]
                k_active_tol = 1.0 + 0.15 * (10.0 - 1.0)  # active if kappa elevated > 15% of melt jump
                updated_subdomains = []
                for cid, idx in enumerate(part["local_indices"]):
                    if float(np.max(k_field[idx])) > k_active_tol:
                        updated_subdomains.append(cid)
                t_setup += backend_ctx.refresh_local(mat_t, step, updated_subdomains)
                # NO coarse grid in Svolos 2020 1-level ASM

            elif arm == "svolos_2level":
                # Svolos 2020 physical selection + Coarse grid, unbudgeted and memoryless
                k_field = phys_k_fields[step]
                k_active_tol = 1.0 + 0.15 * (10.0 - 1.0)
                updated_subdomains = []
                for cid, idx in enumerate(part["local_indices"]):
                    if float(np.max(k_field[idx])) > k_active_tol:
                        updated_subdomains.append(cid)
                t_setup += backend_ctx.refresh_local(mat_t, step, updated_subdomains)
                t_setup += backend_ctx.refresh_coarse(indptr_t, indices_t, data_t, step)

            elif arm == "jsr_adaptive":
                # Proposed JSR: Physics-agnostic algebraic drift proxy + mass-alpha budget + age ceiling
                t_mon_start = time.perf_counter()
                delta_diag = diag_arrays[step] - diag_arrays[step - 1]
                delta_norm = []
                for cid, idx in enumerate(part["local_indices"]):
                    d_norm = float(np.linalg.norm(delta_diag[idx]))
                    delta_norm.append(d_norm)
                t_mon = time.perf_counter() - t_mon_start

                # Score combines drift proxy with stateful age: s_i = delta_i * (1 + 0.25 * age_i)
                scores = [delta_norm[cid] * (1.0 + 0.25 * local_ages[cid]) for cid in range(n_sub)]
                tot_score = sum(scores)
                selected = set()
                if tot_score > 1.0e-12:
                    ranked = sorted(range(n_sub), key=lambda i: -scores[i])
                    cum = 0.0
                    for cid in ranked:
                        selected.add(cid)
                        cum += scores[cid]
                        if cum >= 0.85 * tot_score:
                            break

                # Anti-starvation age ceiling (tau_max = 4)
                for cid in range(n_sub):
                    if local_ages[cid] >= 4:
                        selected.add(cid)

                updated_subdomains = sorted(selected)
                t_setup += t_mon
                t_setup += backend_ctx.refresh_local(mat_t, step, updated_subdomains)
                t_setup += backend_ctx.refresh_coarse(indptr_t, indices_t, data_t, step)

                # Update lifecycle ages
                for cid in range(n_sub):
                    if cid in selected:
                        local_ages[cid] = 0
                    else:
                        local_ages[cid] += 1

            # Solve preconditioned system
            sol_vec.set(0.0)
            its, t_solve, rel_res = solve_pcg(mat_t, backend_ctx, b_vec_t, sol_vec)
            t_total = time.perf_counter() - t_step_start

            step_setup.append(t_setup)
            step_solve.append(t_solve)
            step_total.append(t_total)
            step_iters.append(its)
            step_relres.append(rel_res)
            step_updated_subdomains.append(len(updated_subdomains))

            print(f"  Step {step}: Setup={t_setup:.4f}s | Solve={t_solve:.4f}s | Total={t_total:.4f}s | "
                  f"Iters={its:2d} | Updated={len(updated_subdomains)}/{n_sub} | RelRes={rel_res:.2e}")

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

    # Clean up pre-assembled matrices and vectors
    for m in mats:
        m.destroy()
    for b in b_vecs:
        b.destroy()
    sol_vec.destroy()

    # Calculate Speedups relative to Full Rebuild
    ref_tot = all_results["full_rebuild"]["mean_total"]
    for arm in arms:
        all_results[arm]["speedup"] = float(ref_tot / max(all_results[arm]["mean_total"], 1.0e-9))

    # Display Executive Summary Table
    print("\n" + "=" * 105)
    print(f"{'Strategy Arm':<18} | {'Setup (s)':<10} | {'Solve (s)':<10} | {'Total (s)':<10} | {'Iter':<6} | {'Update %':<9} | {'Speedup':<8}")
    print("-" * 105)
    for arm in arms:
        res = all_results[arm]
        print(f"{arm:<18} | {res['mean_setup']:<10.4f} | {res['mean_solve']:<10.4f} | {res['mean_total']:<10.4f} | "
              f"{res['mean_iters']:<6.1f} | {res['mean_updated_ratio']*100:<8.1f}% | {res['speedup']:<7.2f}x")
    print("=" * 105)

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
            "dt": dt,
            "physics": "3D Transient Heat Conduction with Moving Laser Melt Pool / Thermal Phase Change",
            "element": "Lagrange P1 Tetrahedral Elements",
            "results": all_results,
        }
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump(summary_payload, fp, indent=2)
        print(f"\n✓ Detailed benchmark results saved to: {out_file.resolve()}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realistic 3D FEM Multi-Baseline Benchmark")
    parser.add_argument("--mesh", type=int, default=48, help="Mesh resolution (default: 48 = 117,649 DOFs)")
    parser.add_argument("--steps", type=int, default=6, help="Transient time steps (default: 6)")
    parser.add_argument("--subdomains", type=int, default=2, help="Subdomains per axis (default: 2 -> 8 subdomains)")
    parser.add_argument("--overlap", type=int, default=1, help="Overlap layers (default: 1)")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step size dt (default: 0.05)")
    parser.add_argument("--out", type=str, default="results/cmame_fem_laser_benchmark_n48.json", help="Output JSON path")
    args = parser.parse_args()

    run_fem_benchmark(
        n_mesh=args.mesh,
        n_steps=args.steps,
        sub_per_axis=args.subdomains,
        overlap=args.overlap,
        dt=args.dt,
        output_path=args.out,
    )

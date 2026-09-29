#!/usr/bin/env python3
"""
================================================================================
CMAME Structural Scaling Study: Subdomain Granularity Pressure Benchmark
Compares One-Level Selective Schwarz Baseline vs. Two-Level JSR-relL2
across Decomposition Granularities N_sub in {8, 27, 64, 125}.
================================================================================

Research Objective:
Demonstrate the fundamental structural bifurcation between 1-level and 2-level
preconditioner maintenance:
- 1-Level condition number deteriorates as O(H^{-1}), causing iteration counts
  and solve times to blow up as the subdomain diameter H shrinks (N_sub grows).
- 2-Level JSR with persistent coarse state remains bounded across all granularities,
  preserving optimal tail latency and bounded PCG iterations.
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
from benchmarks.run_long_horizon_sequence import assemble_step
from benchmarks.run_monitor_family_ablation import compute_monitor_scores


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
    rtol: float = 5.0e-11,
    max_it: int = 300,
    target_rel_res: float = 1.0e-8,
) -> Tuple[int, float, float]:
    """Solve system A x = b with certified relative true residual strictly < 1e-8."""
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

    if rel_res >= target_rel_res:
        ksp.setTolerances(rtol=1.0e-13, max_it=max_it + 100)
        t1 = time.perf_counter()
        ksp.solve(b_vec, x_vec)
        t_solve += (time.perf_counter() - t1)
        its += int(ksp.getIterationNumber())
        mat.mult(x_vec, r_vec)
        r_vec.aypx(-1.0, b_vec)
        norm_res = float(r_vec.norm(PETSc.NormType.NORM_2))
        rel_res = norm_res / max(norm_b, 1.0e-14)

    r_vec.destroy()
    ksp.destroy()
    return its, t_solve, rel_res


def run_subdomain_scaling(
    mesh_n: int = 30,
    total_steps: int = 6,
    dt: float = 0.05,
    rho_cp: float = 2.0,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 115)
    print(f"   CMAME STRUCTURAL SCALING PRESSURE: N_sub in {{8, 27, 64, 125}}")
    print(f"   Mesh: UnitCubeMesh({mesh_n}, {mesh_n}, {mesh_n}) | Steps: {total_steps} | dt: {dt} | rho_cp: {rho_cp}")
    print(f"   Certified True Residual: ||b - A x|| / ||b|| < 1.0e-8")
    print("=" * 115)

    mesh = d.UnitCubeMesh(mesh_n, mesh_n, mesh_n)
    V = d.FunctionSpace(mesh, "Lagrange", 1)
    n_dofs = V.dim()
    coords = V.tabulate_dof_coordinates()

    print(f"--> Preassembling all {total_steps} transient FEM steps (total DOFs = {n_dofs})...")
    mats, b_vecs, csrs, diags = [], [], [], []
    u_prev = d.Function(V)
    u_prev.vector()[:] = 0.0

    t_asm_start = time.perf_counter()
    for step in range(total_steps + 1):
        mat, b_vec, indptr, indices, data, _ = assemble_step(
            mesh, V, step, total_steps, dt, u_prev, rho_cp=rho_cp
        )
        diag = mat.getDiagonal().getArray().copy()
        mats.append(mat)
        b_vecs.append(b_vec)
        csrs.append((indptr, indices, data))
        diags.append(diag)
    print(f"    Assembled {total_steps + 1} steps in {time.perf_counter() - t_asm_start:.2f}s.\n")

    grids: List[Tuple[int, int, int]] = [(2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)]
    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)

    records = []

    print("-" * 115)
    print(f"{'N_sub':<7}{'Grid':<9}{'Budget K':<10}{'Mean SubDOF':<13}{'Method':<24}{'Mean Iters':<12}{'Max Iters':<11}{'T_solve(s)':<12}{'T_total(s)':<12}{'RelRes':<10}")
    print("-" * 115)

    for grid in grids:
        kx, ky, kz = grid
        n_sub = kx * ky * kz
        budget_k = max(1, int(round(0.375 * n_sub)))

        part = part_module.create_3d_overlapping_partition(
            coordinates=coords,
            grid=grid,
            overlap_layers=1,
            mesh_n=mesh_n,
        )
        mean_sub_dof = int(np.mean([idx.size for idx in part["local_indices"]]))

        arms = [
            ("one_level_selective", "1-Level Selective Schwarz", False, False),
            ("two_level_jsr", "2-Level JSR-relL2", True, False),
            ("one_level_full", "1-Level Full Rebuild", False, True),
            ("two_level_full", "2-Level Full Rebuild", True, True),
        ]

        grid_rec: Dict[str, Any] = {
            "n_sub": n_sub,
            "grid": f"{kx}x{ky}x{kz}",
            "budget_k": budget_k,
            "mean_sub_dof": mean_sub_dof,
            "arms": {},
        }

        for arm_id, arm_label, use_coarse, is_full in arms:
            backend = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
            backend.refresh_local(mats[0], 0, range(n_sub))
            if use_coarse:
                ind0, indx0, d0 = csrs[0]
                refresh_coarse_vectorized(backend, ind0, indx0, d0, n_sub)

            local_ages = [0] * n_sub
            times_setup, times_solve, times_total = [], [], []
            iters_list, res_list = [], []

            for step in range(1, total_steps + 1):
                mat_t = mats[step]
                b_t = b_vecs[step]
                ind_t, indx_t, d_t = csrs[step]
                diag_t = diags[step]
                prev_diag = diags[step - 1]
                delta_diag = diag_t - prev_diag

                t0 = time.perf_counter()

                if is_full:
                    selected = list(range(n_sub))
                else:
                    raw_scores = compute_monitor_scores(
                        "jsr_rel_l2_k3", delta_diag, diag_t, prev_diag, part["local_indices"], None
                    )
                    scores = [raw_scores[i] * (1.0 + 0.15 * local_ages[i]) for i in range(n_sub)]
                    selected = sorted(sorted(range(n_sub), key=lambda i: -scores[i])[:budget_k])
                    assert len(selected) == budget_k

                t_loc = backend.refresh_local(mat_t, step, selected)
                t_crs = 0.0
                if use_coarse:
                    t_crs = refresh_coarse_vectorized(backend, ind_t, indx_t, d_t, n_sub)
                t_setup = time.perf_counter() - t0

                for cid in range(n_sub):
                    if cid in selected:
                        local_ages[cid] = 0
                    else:
                        local_ages[cid] += 1

                sol_vec.set(0.0)
                its, t_sol, rel_res = solve_pcg_certified(mat_t, backend, b_t, sol_vec, max_it=400)
                assert rel_res < 1.0e-8, f"Certification failure: {rel_res} >= 1.0e-8"

                t_step = t_setup + t_sol
                times_setup.append(float(t_setup))
                times_solve.append(float(t_sol))
                times_total.append(float(t_step))
                iters_list.append(int(its))
                res_list.append(float(rel_res))

            backend.destroy()

            mean_iters = float(np.mean(iters_list))
            max_iters = int(np.max(iters_list))
            min_iters = int(np.min(iters_list))
            tot_solve = float(np.sum(times_solve))
            tot_setup = float(np.sum(times_setup))
            tot_time = float(np.sum(times_total))
            max_res = float(np.max(res_list))

            arm_data = {
                "label": arm_label,
                "use_coarse": use_coarse,
                "is_full": is_full,
                "mean_iters": mean_iters,
                "max_iters": max_iters,
                "min_iters": min_iters,
                "iters_history": iters_list,
                "total_solve_sec": tot_solve,
                "total_setup_sec": tot_setup,
                "total_wall_sec": tot_time,
                "mean_solve_sec": float(np.mean(times_solve)),
                "mean_setup_sec": float(np.mean(times_setup)),
                "max_rel_res": max_res,
            }
            grid_rec["arms"][arm_id] = arm_data

            print(
                f"{n_sub:<7}{f'{kx}x{ky}x{kz}':<9}{budget_k:<10}{mean_sub_dof:<13}{arm_label:<24}"
                f"{mean_iters:<12.1f}{max_iters:<11}{tot_solve:<12.3f}{tot_time:<12.3f}{max_res:<10.2e}"
            )

        # Bifurcation metrics between 1-level selective and 2-level JSR
        iter_bifurcation = grid_rec["arms"]["one_level_selective"]["mean_iters"] / grid_rec["arms"]["two_level_jsr"]["mean_iters"]
        solve_bifurcation = grid_rec["arms"]["one_level_selective"]["total_solve_sec"] / grid_rec["arms"]["two_level_jsr"]["total_solve_sec"]
        grid_rec["iter_bifurcation_ratio"] = iter_bifurcation
        grid_rec["solve_bifurcation_ratio"] = solve_bifurcation
        print(f"      ==> Bifurcation (1-Level / 2-Level JSR): Iterations x{iter_bifurcation:.2f} | Solve Time x{solve_bifurcation:.2f}")
        print("-" * 115)

        records.append(grid_rec)

    sol_vec.destroy()
    for m in mats:
        m.destroy()
    for b in b_vecs:
        b.destroy()

    if output_path is None:
        output_path = str(ROOT / "results" / "subdomain_scaling_pressure.json")
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n✓ Structural Subdomain Granularity Scaling records successfully saved to:\n  {out_file}")
    return {"records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMAME Structural Subdomain Scaling Pressure Benchmark")
    parser.add_argument("--mesh-n", type=int, default=30, help="Grid resolution per axis")
    parser.add_argument("--steps", type=int, default=6, help="Number of transient steps")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step size")
    parser.add_argument("--rho-cp", type=float, default=2.0, help="Volumetric heat capacity")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    run_subdomain_scaling(
        mesh_n=args.mesh_n,
        total_steps=args.steps,
        dt=args.dt,
        rho_cp=args.rho_cp,
        output_path=args.output,
    )

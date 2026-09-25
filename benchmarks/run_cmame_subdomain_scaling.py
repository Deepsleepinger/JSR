#!/usr/bin/env python3
"""
Supporting Study for CMAME: Subdomain Granularity Scaling (N_sub = 8, 27, 64).

Investigates:
- How selective maintenance behaves across decomposition granularities (8 -> 27 -> 64 subdomains);
- Ratio of refactored subdomains |S_t| / N_sub as granularity refines;
- Setup savings, solve time, iteration counts, and end-to-end speedup.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from petsc4py import PETSc
from jsr import partition as part_module
from jsr import backend_mumps as mumps_module


def build_3d_laplacian_csr(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

                indptr[row + 1] = indptr[row] + len(entries)
                for col, val in entries:
                    indices.append(col)
                    values.append(val)

    return coords, indptr, np.array(indices, dtype=np.int64), np.array(values, dtype=np.float64)


def apply_moving_front(
    coords: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    base_vals: np.ndarray,
    t_norm: float,
    amplitude: float = 8.0,
    width: float = 0.12,
) -> np.ndarray:
    n_dofs = coords.shape[0]
    vals = np.copy(base_vals)
    c_start = np.array([0.20, 0.20, 0.20])
    c_end = np.array([0.80, 0.80, 0.80])
    c_curr = c_start + t_norm * (c_end - c_start)

    dists_sq = np.sum((coords - c_curr) ** 2, axis=1)
    kappa = 1.0 + amplitude * np.exp(-dists_sq / (width ** 2))

    for row in range(n_dofs):
        k_row = kappa[row]
        for p in range(indptr[row], indptr[row + 1]):
            col = indices[p]
            if row == col:
                continue
            k_col = kappa[col]
            k_edge = 2.0 * (k_row * k_col) / (k_row + k_col + 1.0e-14)
            vals[p] = -k_edge

    for row in range(n_dofs):
        diag_sum = 0.0
        diag_idx = -1
        for p in range(indptr[row], indptr[row + 1]):
            col = indices[p]
            if row == col:
                diag_idx = p
            else:
                diag_sum += abs(vals[p])
        if diag_idx >= 0:
            vals[diag_idx] = diag_sum + 0.1

    return vals


class PythonPC(object):
    def __init__(self, backend_ctx: mumps_module.OverlappingMUMPSSchwarzBackend):
        self.ctx = backend_ctx

    def apply(self, pc, x_vec, y_vec):
        xa = x_vec.getArray(readonly=True)
        ya = y_vec.getArray()
        self.ctx.apply(xa, ya)


def solve_pcg(
    petsc_mat: PETSc.Mat,
    backend_ctx: mumps_module.OverlappingMUMPSSchwarzBackend,
    rhs_vec: PETSc.Vec,
    sol_vec: PETSc.Vec,
    rtol: float = 1.0e-11,
    max_it: int = 300,
    target_rel_res: float = 1.0e-8,
) -> Tuple[int, float, float]:
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    ksp.setType("cg")
    ksp.setTolerances(rtol=rtol, atol=1.0e-14, max_it=max_it)

    pc = ksp.getPC()
    pc.setType("python")
    py_pc = PythonPC(backend_ctx)
    pc.setPythonContext(py_pc)

    ksp.setOperators(petsc_mat)
    ksp.setUp()

    t0 = time.perf_counter()
    ksp.solve(rhs_vec, sol_vec)
    solve_seconds = time.perf_counter() - t0

    its = ksp.getIterationNumber()
    sol_arr = sol_vec.getArray(readonly=True)
    rhs_arr = rhs_vec.getArray(readonly=True)

    res_vec = petsc_mat.createVecRight()
    petsc_mat.mult(sol_vec, res_vec)
    res_arr = rhs_arr - res_vec.getArray(readonly=True)
    res_norm = float(np.linalg.norm(res_arr))
    rhs_norm = max(float(np.linalg.norm(rhs_arr)), 1.0)
    rel_res = res_norm / rhs_norm

    if rel_res > target_rel_res:
        ksp.setTolerances(rtol=1.0e-13, max_it=max_it + 50)
        ksp.solve(rhs_vec, sol_vec)
        solve_seconds = time.perf_counter() - t0
        its = ksp.getIterationNumber()
        petsc_mat.mult(sol_vec, res_vec)
        res_arr = rhs_arr - res_vec.getArray(readonly=True)
        rel_res = float(np.linalg.norm(res_arr)) / rhs_norm

    res_vec.destroy()
    ksp.destroy()
    return its, solve_seconds, rel_res


def run_subdomain_granularity_study(
    mesh_n: int = 32,
    grids: List[Tuple[int, int, int]] = [(2, 2, 2), (3, 3, 3), (4, 4, 4)],
    n_steps: int = 3,
    overlap: int = 1,
) -> Dict[str, object]:
    coords, indptr, indices, base_vals = build_3d_laplacian_csr(mesh_n)
    n_dofs = coords.shape[0]

    diag_ptrs = np.zeros(n_dofs, dtype=np.int64)
    for row in range(n_dofs):
        for p in range(indptr[row], indptr[row + 1]):
            if indices[p] == row:
                diag_ptrs[row] = p
                break

    states_data = []
    for step in range(n_steps + 1):
        t = step / float(n_steps)
        vals_t = apply_moving_front(coords, indptr, indices, base_vals, t)
        states_data.append(vals_t)

    rhs_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)
    rhs_vec.set(1.0)
    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)

    records = []

    print("=" * 80)
    print(f"  CMAME SUPPORTING STUDY: Subdomain Granularity Scaling (Mesh {mesh_n}^3 = {n_dofs} DOFs)")
    print(f"  Granularities: {[g[0]*g[1]*g[2] for g in grids]} subdomains | Overlap: {overlap} layer")
    print("=" * 80)
    print(f"{'N_sub':<7}{'Grid':<9}{'Sub_DOFs':<10}{'|S_t|/Nsub':<15}{'T_full(s)':<11}{'T_JSR(s)':<11}{'Speedup':<9}{'Savings':<10}{'RelRes':<10}")
    print("-" * 80)

    for grid in grids:
        kx, ky, kz = grid
        n_sub = kx * ky * kz

        part = part_module.create_3d_overlapping_partition(
            coordinates=coords,
            grid=grid,
            overlap_layers=overlap,
            mesh_n=mesh_n,
        )
        mean_sub_dof = int(np.mean([idx.size for idx in part['local_indices']]))

        run_data = {
            "full": {"setup": [], "solve": [], "total": [], "iters": [], "rel_res": []},
            "jsr": {"setup": [], "solve": [], "total": [], "iters": [], "rel_res": [], "refactored": []},
        }

        # Arm 1: Full Rebuild
        backend_full = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
        mat0 = mumps_module.create_petsc_aij_matrix(indptr, indices, states_data[0], n_dofs)
        backend_full.refresh_local(mat0, 0, range(n_sub))
        backend_full.refresh_coarse(indptr, indices, states_data[0], 0)
        sol_vec.set(0.0)
        solve_pcg(mat0, backend_full, rhs_vec, sol_vec)
        mat0.destroy()

        for step in range(1, n_steps + 1):
            mat_t = mumps_module.create_petsc_aij_matrix(indptr, indices, states_data[step], n_dofs)
            t_start = time.perf_counter()
            t_setup = 0.0
            t_setup += backend_full.refresh_local(mat_t, step, range(n_sub))
            t_setup += backend_full.refresh_coarse(indptr, indices, states_data[step], step)

            sol_vec.set(0.0)
            its, t_solve, rel_res = solve_pcg(mat_t, backend_full, rhs_vec, sol_vec)
            t_total = time.perf_counter() - t_start

            run_data["full"]["setup"].append(t_setup)
            run_data["full"]["solve"].append(t_solve)
            run_data["full"]["total"].append(t_total)
            run_data["full"]["iters"].append(its)
            run_data["full"]["rel_res"].append(rel_res)
            mat_t.destroy()
        backend_full.destroy()

        # Arm 2: JSR Adaptive
        backend_jsr = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
        mat0 = mumps_module.create_petsc_aij_matrix(indptr, indices, states_data[0], n_dofs)
        backend_jsr.refresh_local(mat0, 0, range(n_sub))
        backend_jsr.refresh_coarse(indptr, indices, states_data[0], 0)
        sol_vec.set(0.0)
        solve_pcg(mat0, backend_jsr, rhs_vec, sol_vec)
        mat0.destroy()

        for step in range(1, n_steps + 1):
            mat_t = mumps_module.create_petsc_aij_matrix(indptr, indices, states_data[step], n_dofs)
            t_start = time.perf_counter()

            # Diagonal drift proxy
            t_mon_start = time.perf_counter()
            delta_diag = (states_data[step] - states_data[step - 1])[diag_ptrs]
            delta_norm = []
            for cid, idx in enumerate(part["local_indices"]):
                d_norm = float(np.linalg.norm(delta_diag[idx]))
                delta_norm.append(d_norm)
            t_mon = time.perf_counter() - t_mon_start

            tot_drift = sum(delta_norm)
            if tot_drift > 1.0e-12:
                ranked = sorted(range(len(delta_norm)), key=lambda i: -delta_norm[i])
                cum = 0.0
                selected = []
                for cid in ranked:
                    selected.append(cid)
                    cum += delta_norm[cid]
                    if cum >= 0.95 * tot_drift:
                        break
            else:
                selected = []

            t_setup = t_mon
            t_setup += backend_jsr.refresh_local(mat_t, step, selected)
            t_setup += backend_jsr.refresh_coarse(indptr, indices, states_data[step], step)

            sol_vec.set(0.0)
            its, t_solve, rel_res = solve_pcg(mat_t, backend_jsr, rhs_vec, sol_vec)
            t_total = time.perf_counter() - t_start

            run_data["jsr"]["setup"].append(t_setup)
            run_data["jsr"]["solve"].append(t_solve)
            run_data["jsr"]["total"].append(t_total)
            run_data["jsr"]["iters"].append(its)
            run_data["jsr"]["rel_res"].append(rel_res)
            run_data["jsr"]["refactored"].append(len(selected))
            mat_t.destroy()
        backend_jsr.destroy()

        mean_full_setup = float(np.mean(run_data["full"]["setup"]))
        mean_full_solve = float(np.mean(run_data["full"]["solve"]))
        mean_full_total = float(np.mean(run_data["full"]["total"]))
        mean_full_iter = float(np.mean(run_data["full"]["iters"]))

        mean_jsr_setup = float(np.mean(run_data["jsr"]["setup"]))
        mean_jsr_solve = float(np.mean(run_data["jsr"]["solve"]))
        mean_jsr_total = float(np.mean(run_data["jsr"]["total"]))
        mean_jsr_iter = float(np.mean(run_data["jsr"]["iters"]))
        mean_refact = float(np.mean(run_data["jsr"]["refactored"]))
        refact_ratio = mean_refact / float(n_sub)

        speedup = mean_full_total / mean_jsr_total
        saving_pct = ((mean_full_total - mean_jsr_total) / mean_full_total) * 100.0
        max_relres = float(max(np.max(run_data["full"]["rel_res"]), np.max(run_data["jsr"]["rel_res"])))

        rec = {
            "n_sub": n_sub,
            "grid": f"{kx}x{ky}x{kz}",
            "mean_sub_dof": mean_sub_dof,
            "refact_subdomains": mean_refact,
            "refact_ratio": refact_ratio,
            "t_setup_full": mean_full_setup,
            "t_setup_jsr": mean_jsr_setup,
            "t_solve_full": mean_full_solve,
            "t_solve_jsr": mean_jsr_solve,
            "t_total_full": mean_full_total,
            "t_total_jsr": mean_jsr_total,
            "speedup": speedup,
            "saving_pct": saving_pct,
            "iter_full": mean_full_iter,
            "iter_jsr": mean_jsr_iter,
            "max_relres": max_relres,
        }
        records.append(rec)

        print(f"{n_sub:<7}{rec['grid']:<9}{mean_sub_dof:<10}{mean_refact:.1f}/{n_sub} ({refact_ratio*100:.1f}%){'':<2}"
              f"{mean_full_total:<11.4f}{mean_jsr_total:<11.4f}{speedup:<9.2f}x{saving_pct:<9.2f}%{max_relres:<10.2e}")

    rhs_vec.destroy()
    sol_vec.destroy()

    out_file = ROOT.parent / "results" / "cmame_3d_five_pillars_v1" / "subdomain_granularity_study.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)

    print("-" * 80)
    print(f"✓ Subdomain granularity records archived to: {out_file}\n")
    return {"records": records}


if __name__ == "__main__":
    run_subdomain_granularity_study()

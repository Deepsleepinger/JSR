#!/usr/bin/env python3
"""
Shadow Prototype: Defect-Constrained JSR (DC-JSR)
Pure defect-constrained stateful maintenance comparison vs Current AB-JSR and baselines.
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dolfin as d
from petsc4py import PETSc
from jsr import partition as part_module
from jsr import backend_mumps as mumps_module
from benchmarks.run_same_budget_selector_ablation import refresh_coarse_vectorized
from benchmarks.run_subdomain_scaling_pressure import solve_pcg_certified
from benchmarks.run_four_regime_budget_response import assemble_regime_trajectory


class DC_JSR_Controller:
    """
    Defect-Constrained JSR Controller.
    mode: 'max' (R = max_{i not in S} V_i <= eps) or 'sum' (R = sum_{i not in S} V_i <= eps)
    metric_type: 'path' (accumulated variation) or 'endpoint' (net drift from cached factor)
    """
    def __init__(self, n_sub: int = 8, epsilon: float = 0.08, mode: str = "max", metric_type: str = "path"):
        self.n_sub = n_sub
        self.epsilon = epsilon
        self.mode = mode
        self.metric_type = metric_type
        self.V = np.zeros(n_sub, dtype=float)
        self.cached_diags = [None] * n_sub
        self.tau = np.zeros(n_sub, dtype=int)

    def reset(self):
        self.V.fill(0.0)
        self.cached_diags = [None] * self.n_sub
        self.tau.fill(0)

    def update_and_decide(
        self,
        step: int,
        diag_t: np.ndarray,
        prev_diag: np.ndarray,
        local_indices: List[np.ndarray],
    ) -> Tuple[int, List[int], Dict[str, float]]:
        M = self.n_sub
        delta_diag = diag_t - prev_diag

        for i in range(M):
            idx = local_indices[i]
            norm_diag_t = max(float(np.linalg.norm(diag_t[idx])), 1e-14)

            if self.cached_diags[i] is None:
                self.cached_diags[i] = np.copy(prev_diag[idx])

            if self.metric_type == "path":
                # Path length increment: ||diag(A_i(t)) - diag(A_i(t-1))|| / ||diag(A_i(t))||
                v_inc = float(np.linalg.norm(delta_diag[idx])) / norm_diag_t
                self.V[i] += v_inc
            elif self.metric_type == "endpoint":
                # Net drift from cached state: ||diag(A_i(t)) - diag(A_i(tau_i))|| / ||diag(A_i(t))||
                net_delta = diag_t[idx] - self.cached_diags[i]
                self.V[i] = float(np.linalg.norm(net_delta)) / norm_diag_t
            else:
                raise ValueError(self.metric_type)

        # Sorted threshold selection
        sorted_indices = sorted(range(M), key=lambda i: -self.V[i])
        sorted_V = [self.V[i] for i in sorted_indices]

        # Find smallest k in {0, ..., M} such that R_t(S) <= epsilon
        k_selected = M
        for k in range(M + 1):
            if k == M:
                res_defect = 0.0
            else:
                remaining = sorted_V[k:]
                res_defect = float(np.max(remaining)) if self.mode == "max" else float(np.sum(remaining))
            if res_defect <= self.epsilon:
                k_selected = k
                break

        selected = sorted(sorted_indices[:k_selected])

        metrics = {
            "V_total": float(np.sum(self.V)),
            "V_max": float(np.max(self.V)),
            "k_t": k_selected,
        }

        # Reset state for selected subdomains
        for i in selected:
            self.V[i] = 0.0
            idx = local_indices[i]
            self.cached_diags[i] = np.copy(diag_t[idx])
            self.tau[i] = step

        return k_selected, selected, metrics


def run_experiment(regime: str, mesh_n: int = 24, total_steps: int = 10, dt: float = 0.05):
    print("=" * 90)
    print(f"   SHADOW TEST ON REGIME: {regime.upper()} (Steps: {total_steps}, Mesh: N={mesh_n})")
    print("=" * 90)

    mesh = d.UnitCubeMesh(mesh_n, mesh_n, mesh_n)
    V = d.FunctionSpace(mesh, "Lagrange", 1)
    coords = V.tabulate_dof_coordinates()
    n_dofs = V.dim()

    part = part_module.create_3d_overlapping_partition(coords, grid=(2, 2, 2), overlap_layers=1, mesh_n=mesh_n)
    n_sub = int(part["subdomain_count"])

    mats, b_vecs, csrs, diags = assemble_regime_trajectory(regime, mesh, V, total_steps, dt)
    sol_vec = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)

    # 1. Full Rebuild (K=8)
    backend_full = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
    backend_full.refresh_local(mats[0], 0, range(n_sub))
    t_full_start = time.perf_counter()
    full_iters = []
    for step in range(1, total_steps + 1):
        backend_full.refresh_local(mats[step], step, range(n_sub))
        refresh_coarse_vectorized(backend_full, csrs[step][0], csrs[step][1], csrs[step][2], n_sub)
        sol_vec.set(0.0)
        its, _, rel_res = solve_pcg_certified(mats[step], backend_full, b_vecs[step], sol_vec)
        assert rel_res < 1e-8
        full_iters.append(its)
    t_full = time.perf_counter() - t_full_start
    backend_full.destroy()

    # 2. Fixed Budget (K=3)
    backend_k3 = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
    backend_k3.refresh_local(mats[0], 0, range(n_sub))
    t_k3_start = time.perf_counter()
    k3_iters = []
    for step in range(1, total_steps + 1):
        delta = diags[step] - diags[step-1]
        drifts = [float(np.linalg.norm(delta[part["local_indices"][i]])) / max(float(np.linalg.norm(diags[step][part["local_indices"][i]])), 1e-14) for i in range(n_sub)]
        sel_k3 = sorted(sorted(range(n_sub), key=lambda i: -drifts[i])[:3])
        backend_k3.refresh_local(mats[step], step, sel_k3)
        refresh_coarse_vectorized(backend_k3, csrs[step][0], csrs[step][1], csrs[step][2], n_sub)
        sol_vec.set(0.0)
        its, _, rel_res = solve_pcg_certified(mats[step], backend_k3, b_vecs[step], sol_vec)
        assert rel_res < 1e-8
        k3_iters.append(its)
    t_k3 = time.perf_counter() - t_k3_start
    backend_k3.destroy()

    print(f"Baseline Full Rebuild (K=8): {t_full:.3f}s | Mean Iters: {np.mean(full_iters):.1f} | Max Iters: {max(full_iters)}")
    print(f"Baseline Fixed Budget (K=3): {t_k3:.3f}s | Mean Iters: {np.mean(k3_iters):.1f} | Max Iters: {max(k3_iters)}")
    print("-" * 90)

    # 3. Test DC-JSR Prototype across epsilons
    for mode in ["max", "sum"]:
        for metric in ["path", "endpoint"]:
            for eps in [0.03, 0.05, 0.08, 0.12]:
                controller = DC_JSR_Controller(n_sub=n_sub, epsilon=eps, mode=mode, metric_type=metric)
                backend = mumps_module.OverlappingMUMPSSchwarzBackend(n_dofs, part)
                backend.refresh_local(mats[0], 0, range(n_sub))
                for i in range(n_sub):
                    controller.cached_diags[i] = np.copy(diags[0][part["local_indices"][i]])

                t_start = time.perf_counter()
                k_list = []
                iters = []
                for step in range(1, total_steps + 1):
                    k_t, selected, m = controller.update_and_decide(
                        step, diags[step], diags[step-1], part["local_indices"]
                    )
                    k_list.append(k_t)
                    backend.refresh_local(mats[step], step, selected)
                    refresh_coarse_vectorized(backend, csrs[step][0], csrs[step][1], csrs[step][2], n_sub)
                    sol_vec.set(0.0)
                    its, _, rel_res = solve_pcg_certified(mats[step], backend, b_vecs[step], sol_vec)
                    assert rel_res < 1e-8
                    iters.append(its)
                t_total = time.perf_counter() - t_start
                backend.destroy()

                ratio = t_total / t_full
                print(f"DC-JSR [{metric:>8}, {mode:>3}, eps={eps:<4}]: {t_total:.3f}s ({ratio:.3f}x Full) | Mean K: {np.mean(k_list):.2f} | K seq: {k_list[:8]}... | Max Iter: {max(iters)}")

    sol_vec.destroy()
    for m in mats: m.destroy()
    for b in b_vecs: b.destroy()


if __name__ == "__main__":
    reg = sys.argv[1] if len(sys.argv) > 1 else "gentle_single_front"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_experiment(reg, mesh_n=24, total_steps=n_steps)

#!/usr/bin/env python3
"""Frozen DC-JSR replication with balanced ordering and stepwise certification.

Uses the existing shadow controller without modifying it. Timed intervals include
sensing, maintenance, CG, and true-residual certification. Extra state/coarse/CSR
audits are separately accounted for. Assembly and initialization are excluded.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import dolfin as d
from petsc4py import PETSc
from jsr import partition, backend_mumps
from benchmarks.run_dc_jsr_shadow_prototype import DC_JSR_Controller
from benchmarks.run_ab_jsr_controller import AdaptiveBudgetController
from benchmarks.run_four_regime_budget_response import assemble_regime_trajectory
from benchmarks.run_held_out_generalization_suite import preassemble_trajectory
from benchmarks.run_monitor_family_ablation import compute_monitor_scores
from benchmarks.run_same_budget_selector_ablation import refresh_coarse_vectorized
from benchmarks.run_cmame_fem_moving_laser_benchmark import PythonPC

TARGET = 1e-8
ARMS = ["full", "fixed_k3", "dc_jsr", "ab_jsr"]
STEPS = {"gentle_single_front": 10, "multi_front_churn": 16,
         "compound_shift": 18, "staccato_revisit": 16}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def certified_cg(mat, b, x, backend):
    """Match shadow tolerances, recording both attempts and convergence reasons."""
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    residual = b.duplicate()
    try:
        ksp.setOperators(mat)
        ksp.setType("cg")
        ksp.getPC().setType("python")
        ksp.getPC().setPythonContext(PythonPC(backend))
        ksp.getPC().setUp()
        norm_b = float(b.norm())
        require(np.isfinite(norm_b) and norm_b > 0, "Invalid RHS norm")
        attempts = []
        for rtol, max_it in [(5e-11, 300), (1e-13, 400)]:
            ksp.setTolerances(rtol=rtol, atol=1e-14, max_it=max_it)
            ksp.solve(b, x)
            mat.mult(x, residual)
            residual.aypx(-1, b)
            rel = float(residual.norm()) / norm_b
            attempts.append({"iterations": int(ksp.getIterationNumber()),
                             "reason": int(ksp.getConvergedReason()),
                             "rtol": rtol, "true_relative_residual": rel})
            require(np.isfinite(rel), "Nonfinite true residual")
            if rel < TARGET:
                break
        require(rel < TARGET, "PCG true residual failed: %r" % attempts)
        require(attempts[-1]["reason"] > 0, "CG did not report convergence")
        return {"iterations": sum(a["iterations"] for a in attempts),
                "petsc_true_relative_residual": rel, "attempts": attempts}
    finally:
        residual.destroy()
        ksp.destroy()


def run_arm(arm, mats, rhs, csrs, diags, part, epsilon):
    n_sub = part["subdomain_count"]
    n_dofs = len(diags[0])
    controller = DC_JSR_Controller(n_sub=n_sub, epsilon=epsilon)
    ab = AdaptiveBudgetController(n_sub=n_sub)
    backend = backend_mumps.OverlappingMUMPSSchwarzBackend(n_dofs, part)
    x = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)
    ages = [0] * n_sub
    prev_iters = None
    rows = []
    t_init = time.perf_counter()
    try:
        backend.refresh_local(mats[0], 0, range(n_sub))
        for i, idx in enumerate(part["local_indices"]):
            controller.cached_diags[i] = diags[0][idx].copy()
        initial_seconds = time.perf_counter() - t_init
        for step in range(1, len(mats)):
            t0 = time.perf_counter()
            proposal = None
            telemetry = {}
            if arm == "full":
                selected = list(range(n_sub))
            elif arm == "dc_jsr":
                # Commit the proposed controller state only after local refresh
                # succeeds. A partial backend failure aborts the entire run.
                proposal = copy.deepcopy(controller)
                _, selected, telemetry = proposal.update_and_decide(
                    step, diags[step], diags[step - 1], part["local_indices"])
            else:
                delta = diags[step] - diags[step - 1]
                drifts = [float(np.linalg.norm(delta[idx])) /
                          max(float(np.linalg.norm(diags[step][idx])), 1e-14)
                          for idx in part["local_indices"]]
                if arm == "fixed_k3":
                    # Exactly the shadow baseline: instantaneous drift, no Age.
                    selected = sorted(sorted(range(n_sub), key=lambda i: -drifts[i])[:3])
                else:
                    raw = compute_monitor_scores("jsr_rel_l2_k3", delta,
                                                 diags[step], diags[step - 1],
                                                 part["local_indices"], None)
                    _, selected, reason, telemetry = ab.decide_budget_and_subdomains(
                        raw, drifts, ages, prev_iters)
                    telemetry = dict(telemetry, reason=reason)
            t_monitor = time.perf_counter() - t0
            t1 = time.perf_counter()
            backend.refresh_local(mats[step], step, selected)
            if proposal is not None:
                committed_old = controller
                controller = proposal
            t_local = time.perf_counter() - t1
            t2 = time.perf_counter()
            refresh_coarse_vectorized(backend, *csrs[step], n_sub)
            t_coarse = time.perf_counter() - t2
            for i in range(n_sub):
                ages[i] = 0 if i in selected else ages[i] + 1
            x.set(0)
            t3 = time.perf_counter()
            solve = certified_cg(mats[step], rhs[step], x, backend)
            t_solve = time.perf_counter() - t3
            timed_seconds = time.perf_counter() - t0
            prev_iters = solve["iterations"]

            t_audit = time.perf_counter()
            indptr, indices, data = csrs[step]
            row_indices = np.repeat(np.arange(n_dofs), np.diff(indptr))
            b_np = rhs[step].getArray(readonly=True)
            ax = np.bincount(row_indices, weights=data * x.getArray(readonly=True)[indices], minlength=n_dofs)
            independent_rel = float(np.linalg.norm(b_np - ax) /
                                    np.linalg.norm(b_np))
            require(np.isfinite(independent_rel) and independent_rel < TARGET,
                    "Independent CSR certification failed")
            builds = [int(f.build_state) for f in backend.factors]
            if rows:
                expected_builds = rows[-1]["factor_build_states"].copy()
            else:
                expected_builds = [0] * n_sub
            for i in selected:
                expected_builds[i] = step
            require(builds == expected_builds, "Factor build-state mismatch")
            core = backend.core_ids
            # Form A Z column by column from CSR, then restrict with Z^T.
            # This uses a different accumulation order from the backend's
            # one-pass grouping by coarse (row, column) pairs.
            coarse = np.empty((n_sub, n_sub))
            for j in range(n_sub):
                az_j = np.bincount(row_indices, weights=data * (core[indices] == j), minlength=n_dofs)
                coarse[:, j] = np.bincount(core, weights=az_j, minlength=n_sub)
            actual = backend.coarse_ksp.getOperators()[0]
            actual_dense = actual.getValues(np.arange(n_sub, dtype=PETSc.IntType),
                                            np.arange(n_sub, dtype=PETSc.IntType))
            coarse_error = float(np.linalg.norm(actual_dense - coarse) /
                                 max(np.linalg.norm(coarse), 1e-14))
            require(coarse_error < 1e-12, "Coarse Galerkin mismatch")
            dc_trace = {}
            if proposal is not None:
                delta = diags[step] - diags[step - 1]
                increments = np.array([np.linalg.norm(delta[idx]) /
                                       max(np.linalg.norm(diags[step][idx]), 1e-14)
                                       for idx in part["local_indices"]])
                before = committed_old.V + increments
                required = np.flatnonzero(before > epsilon).tolist()
                require(selected == required, "Max-defect minimum-cardinality violation")
                expected = before.copy()
                expected[selected] = 0
                require(np.allclose(controller.V, expected, rtol=1e-13, atol=1e-14),
                        "Path evolution/reset mismatch")
                require(controller.tau.tolist() == builds, "DC state/factor mismatch")
                for i in selected:
                    require(np.array_equal(controller.cached_diags[i], diags[step][part["local_indices"][i]]),
                            "DC cached diagonal mismatch")
                residual_proxy = float(np.max(controller.V))
                require(np.isfinite(residual_proxy) and residual_proxy <= epsilon,
                        "Defect constraint violated")
                dc_trace = {"increment": increments.tolist(), "V_before": before.tolist(),
                            "V_after": controller.V.tolist(), "residual_proxy": residual_proxy}
            rows.append(dict(step=step, selected=selected, k=len(selected),
                             factor_build_states=builds, coarse_relative_error=coarse_error,
                             independent_csr_relative_residual=independent_rel,
                             monitor_seconds=t_monitor, local_seconds=t_local,
                             coarse_seconds=t_coarse, solve_certification_seconds=t_solve,
                             timed_seconds=timed_seconds,
                             audit_seconds=time.perf_counter() - t_audit,
                             dc=dc_trace, telemetry=telemetry, **solve))
        return {"arm": arm, "epsilon": epsilon if arm == "dc_jsr" else None,
                "initialization_seconds": initial_seconds,
                "total_seconds": sum(r["timed_seconds"] for r in rows),
                "extra_audit_seconds": sum(r["audit_seconds"] for r in rows),
                "max_iterations": max(r["iterations"] for r in rows),
                "max_true_relative_residual": max(max(r["petsc_true_relative_residual"],
                                                       r["independent_csr_relative_residual"]) for r in rows),
                "retry_count": sum(len(r["attempts"]) - 1 for r in rows),
                "k_sequence": [r["k"] for r in rows], "steps": rows}
    finally:
        backend.destroy()
        x.destroy()


def summarize(runs):
    summary = {}
    for arm in dict.fromkeys(r["arm"] for r in runs):
        subset = [r for r in runs if r["arm"] == arm]
        vals = np.array([r["total_seconds"] for r in subset])
        summary[arm] = {"mean_seconds": float(vals.mean()), "median_seconds": float(np.median(vals)),
                        "std_seconds": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                        "min_seconds": float(vals.min()), "max_seconds": float(vals.max()),
                        "max_iterations": max(r["max_iterations"] for r in subset),
                        "max_true_relative_residual": max(r["max_true_relative_residual"] for r in subset),
                        "repeat_k_equal": all(r["k_sequence"] == subset[0]["k_sequence"] for r in subset),
                        "k_sequence": subset[0]["k_sequence"]}
        if arm != "full" and any(r["arm"] == "full" for r in runs):
            full_by_rep = {r["repeat"]: r for r in runs if r["arm"] == "full"}
            ratios = [r["total_seconds"] / full_by_rep[r["repeat"]]["total_seconds"] for r in subset]
            summary[arm]["paired_ratios_vs_full"] = ratios
            summary[arm]["paired_wins_vs_full"] = sum(r < 1 for r in ratios)
            summary[arm]["ratio_of_mean_times_vs_full"] = float(vals.mean() / np.mean(
                [full_by_rep[r["repeat"]]["total_seconds"] for r in subset]))
    dc_runs = [r for r in runs if r["arm"] == "dc_jsr"]
    for other in ["full", "ab_jsr"]:
        other_runs = {r["repeat"]: r for r in runs if r["arm"] == other}
        if dc_runs and other_runs:
            summary["dc_jsr"]["iterations_equal_%s_every_step" % other] = all(
                [s["iterations"] for s in r["steps"]] ==
                [s["iterations"] for s in other_runs[r["repeat"]]["steps"]] for r in dc_runs)
    return summary


def save(output, result):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-n", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=.08)
    parser.add_argument("--regimes", nargs="+", choices=list(STEPS),
                        default=["gentle_single_front", "multi_front_churn", "compound_shift"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--output", type=Path, default=ROOT / "results/dc_jsr_certification_audit_n24.json")
    args = parser.parse_args()
    require(args.mesh_n > 0 and args.repeats > 0 and np.isfinite(args.epsilon) and args.epsilon >= 0,
            "Invalid mesh, repeats, or epsilon")
    require(len(set(args.arms)) == len(args.arms), "Duplicate arms")
    sources = sorted(set(ROOT.glob("benchmarks/*.py")) | set(ROOT.glob("jsr/*.py")))
    result = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "configuration": dict(mesh_n=args.mesh_n, repeats=args.repeats, epsilon=args.epsilon,
                                    regimes=args.regimes, arms=args.arms, dt=.05, subdomain_grid=[2, 2, 2]),
              "protocol": {"timing": "monitor + local + coarse + CG + PETSc certification; excludes assembly, initialization and extra audit",
                           "ordering": "cyclic arm rotation across repeats; four repeats balance four positions",
                           "target": TARGET, "fixed_k3_selector": "instantaneous relative diagonal l2 drift, no age",
                           "claim_scope": "preassembled evolving FEM operator sequences; u_prev remains zero; not a coupled SLM time integration",
                           "epsilon_status": "frozen at handoff value; previously inspected trajectories are replication, not DC blind tests"},
              "environment": {"python": sys.executable, "python_version": sys.version,
                              "platform": platform.platform(), "numpy": np.__version__,
                              "dolfin": d.__version__, "petsc": PETSc.Sys.getVersion(),
                              "thread_environment": {k: os.environ.get(k) for k in
                                  ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LD_PRELOAD", "PKG_CONFIG_PATH"]},
                              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                              "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)},
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "regimes": {}}
    save(args.output, result)
    for regime in args.regimes:
        print("Assembling", regime, flush=True)
        t_assembly = time.perf_counter()
        mesh = d.UnitCubeMesh(args.mesh_n, args.mesh_n, args.mesh_n)
        space = d.FunctionSpace(mesh, "Lagrange", 1)
        part = partition.create_3d_overlapping_partition(space.tabulate_dof_coordinates(),
                                                       grid=(2, 2, 2), overlap_layers=1, mesh_n=args.mesh_n)
        if regime in ["compound_shift", "staccato_revisit"]:
            mats, rhs, csrs, diags, fields = preassemble_trajectory(regime, mesh, space, STEPS[regime], .05)
        else:
            mats, rhs, csrs, diags = assemble_regime_trajectory(regime, mesh, space, STEPS[regime], .05)
        fingerprint = hashlib.sha256()
        for csr, b in zip(csrs, rhs):
            for array in list(csr) + [b.getArray(readonly=True)]:
                fingerprint.update(np.ascontiguousarray(array).tobytes())
        record = {"n_dofs": space.dim(), "steps": STEPS[regime], "assembly_seconds": time.perf_counter() - t_assembly,
                  "trajectory_sha256": fingerprint.hexdigest(), "runs": []}
        result["regimes"][regime] = record
        try:
            for rep in range(args.repeats):
                shift = rep % len(args.arms)
                order = args.arms[shift:] + args.arms[:shift]
                for position, arm in enumerate(order):
                    run = run_arm(arm, mats, rhs, csrs, diags, part, args.epsilon)
                    run.update(repeat=rep, position=position, order=order)
                    record["runs"].append(run)
                    record["summary"] = summarize(record["runs"]) if len(record["runs"]) % len(args.arms) == 0 else record.get("summary", {})
                    save(args.output, result)
                    print("%s rep=%d %s %.3fs max_it=%d max_res=%.2e K=%s" %
                          (regime, rep, arm, run["total_seconds"], run["max_iterations"],
                           run["max_true_relative_residual"], run["k_sequence"]), flush=True)
        finally:
            for mat in mats:
                mat.destroy()
            for b in rhs:
                b.destroy()
    result["status"] = "complete"
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.output, result)
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()

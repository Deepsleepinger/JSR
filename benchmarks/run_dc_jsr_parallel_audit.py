#!/usr/bin/env python3
"""Real MPI replay: fixed eight subdomains, Full versus frozen-epsilon DC-JSR."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Bind before importing numeric libraries. Every rank uses one logical CPU.
from mpi4py import MPI
COMM = MPI.COMM_WORLD
AVAILABLE_CPUS = sorted(os.sched_getaffinity(0))
if COMM.size > len(AVAILABLE_CPUS):
    raise RuntimeError("Oversubscribed run rejected; available CPUs=%r" % AVAILABLE_CPUS)
os.sched_setaffinity(0, {AVAILABLE_CPUS[COMM.rank]})
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    if os.environ.get(name) != "1":
        raise RuntimeError("Set %s=1 before launching MPI." % name)

import numpy as np
from petsc4py import PETSc
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jsr.backend_mpi import DistributedSchwarz, selected_csr_rows
from jsr.backend_mumps import OverlappingMUMPSSchwarzBackend, create_petsc_aij_matrix
from benchmarks.run_same_budget_selector_ablation import refresh_coarse_vectorized
from benchmarks.run_dc_jsr_shadow_prototype import DC_JSR_Controller

TARGET = 1e-8


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, result):
    if COMM.rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        tmp.replace(path)


def merged_dict(local):
    return {k: v for shard in COMM.allgather(local) for k, v in shard.items()}


def csr_coarse_reference(csr, core_ids, m):
    """Independent column-wise A*Z, rather than row/column bin-pair assembly."""
    ia, ja, values = csr
    row = np.repeat(np.arange(len(core_ids)), np.diff(ia))
    coarse = np.zeros((m, m))
    for col in range(m):
        az = np.bincount(row, weights=values * (core_ids[ja] == col), minlength=len(core_ids))
        coarse[:, col] = np.bincount(core_ids, weights=az, minlength=m)
    return coarse


def prepare(path):
    t0 = time.perf_counter()
    global_data = None
    if COMM.rank == 0:
        manifest = json.loads(path.with_suffix(".json").read_text())
        require(hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"], "Input hash mismatch")
        with np.load(str(path), allow_pickle=False) as archive:
            global_data = {k: archive[k].copy() for k in archive.files}
        metadata = dict(core_ids=global_data["core_ids"], weights=global_data["weights"],
                        local_indices=[global_data["indices_%d" % i] for i in range(8)], subdomain_count=8)
    else:
        manifest = metadata = None
    manifest = COMM.bcast(manifest, root=0)
    metadata = COMM.bcast(metadata, root=0)
    n = manifest["n_dofs"]
    counts = np.array([n // COMM.size + (r < n % COMM.size) for r in range(COMM.size)], dtype=np.int64)
    bounds = np.concatenate(([0], np.cumsum(counts)))
    local_steps = []
    for step in range(manifest["steps"] + 1):
        payloads = None
        if COMM.rank == 0:
            csr = tuple(global_data["%s_%d" % (key, step)] for key in ["ia", "ja", "values"])
            payloads = []
            for rank in range(COMM.size):
                owned = list(range(rank, 8, COMM.size))
                union = np.unique(np.concatenate([metadata["local_indices"][c] for c in owned]))
                first, last = bounds[rank:rank + 2]
                payloads.append(dict(csr=selected_csr_rows(csr, np.arange(first, last)),
                                     overlap_csr=selected_csr_rows(csr, union),
                                     diag=global_data["diag_%d" % step][union],
                                     rhs=global_data["rhs_%d" % step][first:last]))
        payload = COMM.scatter(payloads, root=0)
        local_n = int(counts[COMM.rank])
        ia, ja, values = payload["csr"]
        matrix = PETSc.Mat().createAIJ(size=((local_n, n), (local_n, n)),
                                      csr=(ia, ja, values), comm=PETSc.COMM_WORLD)
        matrix.assemble()
        b = matrix.createVecRight()
        require(b.getOwnershipRange() == tuple(bounds[COMM.rank:COMM.rank + 2]), "Unexpected PETSc row layout")
        b.getArray()[:] = payload["rhs"]
        payload.update(matrix=matrix, b=b, row=np.repeat(np.arange(local_n), np.diff(ia)))
        if step:
            require(np.array_equal(payload["overlap_csr"][0], local_steps[0]["overlap_csr"][0]) and
                    np.array_equal(payload["overlap_csr"][1], local_steps[0]["overlap_csr"][1]),
                    "Changing sparsity requires rebuilding static extraction maps")
        local_steps.append(payload)
    COMM.Barrier()
    elapsed = COMM.allreduce(time.perf_counter() - t0, op=MPI.MAX)
    return manifest, metadata, global_data, local_steps, elapsed


def certify(matrix, b, x, pc_context):
    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    residual = b.duplicate()
    try:
        ksp.setOperators(matrix)
        ksp.setType("cg")
        ksp.getPC().setType("python")
        ksp.getPC().setPythonContext(pc_context)
        ksp.getPC().setUp()
        norm_b = float(b.norm())
        require(norm_b > 0 and np.isfinite(norm_b), "Invalid RHS")
        attempts = []
        for rtol, max_it in [(5e-11, 300), (1e-13, 400)]:
            ksp.setTolerances(rtol=rtol, atol=1e-14, max_it=max_it)
            ksp.solve(b, x)
            matrix.mult(x, residual)
            residual.aypx(-1, b)
            rel = float(residual.norm()) / norm_b
            attempts.append(dict(iterations=int(ksp.getIterationNumber()),
                                 reason=int(ksp.getConvergedReason()), rtol=rtol,
                                 petsc_true_relative_residual=rel))
            if np.isfinite(rel) and rel < TARGET:
                break
        require(rel < TARGET and attempts[-1]["reason"] > 0, "Uncertified solve: %r" % attempts)
        return dict(iterations=sum(a["iterations"] for a in attempts),
                    petsc_true_relative_residual=rel, attempts=attempts)
    finally:
        ksp.destroy()
        residual.destroy()


def independent_residual(payload, x):
    shards = COMM.allgather(x.getArray(readonly=True).copy())
    global_x = np.concatenate(shards)
    ia, ja, values = payload["csr"]
    ax = np.bincount(payload["row"], weights=values * global_x[ja], minlength=len(payload["rhs"]))
    numerator = COMM.allreduce(float(np.sum((payload["rhs"] - ax) ** 2)), op=MPI.SUM)
    denominator = COMM.allreduce(float(np.sum(payload["rhs"] ** 2)), op=MPI.SUM)
    relative = float(np.sqrt(numerator / denominator))
    require(np.isfinite(relative) and relative < TARGET, "Independent CSR residual failed")
    return relative


def reference_apply_check(backend, x, metadata, global_data, current, versions):
    """Compare distributed action with the original serial backend at same state."""
    rng = np.random.RandomState(817)
    probe = rng.standard_normal(backend.n)
    x.getArray()[:] = probe[backend.first:backend.last]
    y = x.duplicate()
    backend.apply(None, x, y)
    actual = np.concatenate(COMM.allgather(y.getArray(readonly=True).copy()))
    y.destroy()
    error = None
    if COMM.rank == 0:
        reference = OverlappingMUMPSSchwarzBackend(backend.n, metadata)
        try:
            for step in sorted(set(versions.values())):
                csr = tuple(global_data["%s_%d" % (key, step)] for key in ["ia", "ja", "values"])
                matrix = create_petsc_aij_matrix(*csr, backend.n)
                reference.refresh_local(matrix, step, [c for c, version in versions.items() if version == step])
                matrix.destroy()
            csr = tuple(global_data["%s_%d" % (key, current)] for key in ["ia", "ja", "values"])
            refresh_coarse_vectorized(reference, *csr, backend.m)
            expected = np.empty_like(probe)
            reference.apply(probe, expected)
            error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        finally:
            reference.destroy()
    error = COMM.bcast(error, root=0)
    require(error < 1e-10, "Serial/distributed preconditioner mismatch: %g" % error)
    return error


def run_arm(arm, repeat, local_steps, metadata, global_data, epsilon, reference_checks):
    x = local_steps[0]["b"].duplicate()
    COMM.Barrier()
    t0 = time.perf_counter()
    backend = DistributedSchwarz(x, metadata, local_steps[0]["overlap_csr"][:2])
    static_setup = time.perf_counter() - t0
    t1 = time.perf_counter()
    backend.refresh_local(local_steps[0]["overlap_csr"][2], 0, backend.owned)
    backend.refresh_coarse(local_steps[0]["csr"], 0)
    initialization = time.perf_counter() - t1
    init_times = COMM.allgather(dict(static_setup=static_setup, factors_and_coarse=initialization,
                                   total=time.perf_counter() - t0))
    versions = merged_dict(backend.versions)
    reference_errors = []
    if reference_checks:
        reference_errors.append(dict(step=0, relative_error=reference_apply_check(
            backend, x, metadata, global_data, 0, versions)))
    oracle = DC_JSR_Controller(8, epsilon=epsilon) if COMM.rank == 0 else None
    rows = []
    try:
        for step in range(1, len(local_steps)):
            payload = local_steps[step]
            backend.apply_calls = 0
            backend.apply_seconds = dict(overlap=0.0, local=0.0, coarse=0.0)
            COMM.Barrier()
            start = time.perf_counter()
            selected = backend.decide(payload["diag"], local_steps[step - 1]["diag"], epsilon,
                                      full=arm == "full")
            monitor_end = time.perf_counter()
            backend.refresh_local(payload["overlap_csr"][2], step, selected)
            local_end = time.perf_counter()
            backend.refresh_coarse(payload["csr"], step)
            coarse_end = time.perf_counter()
            x.set(0.0)
            solve = certify(payload["matrix"], payload["b"], x, backend)
            end = time.perf_counter()
            # No telemetry communication is inside the timed interval.
            rank_times = COMM.allgather(dict(monitor=monitor_end - start,
                                            local=local_end - monitor_end,
                                            coarse=coarse_end - local_end,
                                            solve_and_certification=end - coarse_end,
                                            total=end - start, apply_calls=backend.apply_calls,
                                            apply_seconds=backend.apply_seconds.copy()))
            audit_start = time.perf_counter()
            selected_global = sorted(c for shard in COMM.allgather(selected) for c in shard)
            versions = merged_dict(backend.versions)
            states = merged_dict(backend.state)
            pre_states = merged_dict(backend.pre_state)
            audit = None
            if COMM.rank == 0:
                diag = global_data["diag_%d" % step]
                prev = global_data["diag_%d" % (step - 1)]
                _, expected_selected, _ = oracle.update_and_decide(step, diag, prev, metadata["local_indices"])
                if arm == "full":
                    expected_selected = list(range(8))
                require(selected_global == expected_selected, "Threshold subset differs from shadow policy")
                require(all(versions[c] == (step if c in selected_global else rows[-1]["factor_versions"][c]
                                           if rows else 0) for c in range(8)), "Factor version mismatch")
                require(max(states.values()) <= epsilon + 1e-14, "Residual proxy constraint failed")
                if arm == "dc_jsr":
                    require(np.allclose([states[c] for c in range(8)], oracle.V, rtol=0, atol=1e-14),
                            "Accumulated state differs from serial shadow controller")
                csr = tuple(global_data["%s_%d" % (key, step)] for key in ["ia", "ja", "values"])
                coarse_ref = csr_coarse_reference(csr, metadata["core_ids"], 8)
                coarse_error = float(np.linalg.norm(backend.coarse_dense - coarse_ref) / np.linalg.norm(coarse_ref))
                require(coarse_error < 1e-10, "Coarse Galerkin audit failed")
                audit = dict(coarse_relative_error=coarse_error)
            audit = COMM.bcast(audit, root=0)
            residual = independent_residual(payload, x)
            if reference_checks and step == len(local_steps) - 1:
                reference_errors.append(dict(step=step, relative_error=reference_apply_check(
                    backend, x, metadata, global_data, step, versions)))
            audit_seconds = COMM.allreduce(time.perf_counter() - audit_start, op=MPI.MAX)
            rows.append(dict(step=step, selected=selected_global, k=len(selected_global),
                             pre_proxy=[pre_states[c] for c in range(8)],
                             residual_proxy=max(states.values()), factor_versions=[versions[c] for c in range(8)],
                             rank_times=rank_times, total_seconds=max(r["total"] for r in rank_times),
                             stage_maxima={key: max(r[key] for r in rank_times)
                                           for key in ["monitor", "local", "coarse", "solve_and_certification"]},
                             independent_true_relative_residual=residual,
                             audit_seconds=audit_seconds, **solve, **audit))
        record = dict(arm=arm, repeat=repeat, steps=rows, initialization_rank_times=init_times,
                      static_setup_seconds=max(r["static_setup"] for r in init_times),
                      initial_factor_coarse_seconds=max(r["factors_and_coarse"] for r in init_times),
                      initialization_seconds=max(r["total"] for r in init_times),
                      total_seconds=sum(r["total_seconds"] for r in rows),
                      max_iterations=max(r["iterations"] for r in rows),
                      max_true_residual=max(r["independent_true_relative_residual"] for r in rows),
                      reference_apply_errors=reference_errors)
        record["total_with_initialization_seconds"] = record["total_seconds"] + record["initialization_seconds"]
        return record
    finally:
        x.destroy()
        backend.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--epsilon", type=float, default=.08)
    p.add_argument("--skip-warmup", action="store_true", help="Smoke tests only")
    a = p.parse_args()
    require(a.repeats > 0 and np.isfinite(a.epsilon) and a.epsilon >= 0, "Invalid arguments")
    manifest, metadata, global_data, local_steps, preparation_seconds = prepare(a.input)
    result = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                  input_manifest=manifest, mpi_processes=COMM.size, epsilon=a.epsilon, repeats=a.repeats,
                  preparation_seconds=preparation_seconds,
                  protocol=dict(subdomain_owner="i mod P", fine_ownership="contiguous rows",
                                ordering="Full/DC alternates positions across paired repeats",
                                timing="sum of per-step maximum rank wall time, monitoring+local+coarse+CG+true residual",
                                excluded="FEM generation, input loading/distribution, PETSc global matrix staging, extra audits",
                                initialization="reported separately and added in secondary metric",
                                coarse="8-dimensional Galerkin, Allreduce assembly/RHS, redundant exact LU",
                                scope="single-node fixed-eight-subdomain strong scaling; not multi-node/application end-to-end"),
                  rank_environment=COMM.allgather(dict(rank=COMM.rank, hostname=platform.node(),
                                                      cpu_affinity=sorted(os.sched_getaffinity(0)),
                                                      python=sys.executable, petsc=PETSc.Sys.getVersion(),
                                                      mpi_library=MPI.Get_library_version().splitlines()[0],
                                                      thread_environment={key: os.environ[key] for key in
                                                          ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]})),
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__).resolve(), ROOT / "jsr/backend_mpi.py", ROOT / "jsr/backend_mumps.py",
                                              ROOT / "benchmarks/run_dc_jsr_shadow_prototype.py"]},
                  warmup_runs=[], runs=[])
    save(a.output, result)
    if not a.skip_warmup:
        for arm in ["full", "dc_jsr"]:
            run = run_arm(arm, -1, local_steps, metadata, global_data, a.epsilon, True)
            result["warmup_runs"].append(run)
            save(a.output, result)
            if COMM.rank == 0:
                print("Warmup", manifest["regime"], "P", COMM.size, arm,
                      "seconds", round(run["total_seconds"], 3), "reference verified", flush=True)
    for repeat in range(a.repeats):
        for arm in (["full", "dc_jsr"] if repeat % 2 == 0 else ["dc_jsr", "full"]):
            run = run_arm(arm, repeat, local_steps, metadata, global_data, a.epsilon,
                          a.skip_warmup and repeat == 0)
            result["runs"].append(run)
            save(a.output, result)
            if COMM.rank == 0:
                print("Repeat", repeat, manifest["regime"], "P", COMM.size, arm,
                      "seconds", round(run["total_seconds"], 3), "maxiter", run["max_iterations"], flush=True)
    result["summary"] = {}
    full = {r["repeat"]: r for r in result["runs"] if r["arm"] == "full"}
    for arm in ["full", "dc_jsr"]:
        runs = [r for r in result["runs"] if r["arm"] == arm]
        values = np.array([r["total_seconds"] for r in runs])
        ratios = [r["total_seconds"] / full[r["repeat"]]["total_seconds"] for r in runs]
        summary = dict(mean_seconds=float(values.mean()), sd_seconds=float(values.std(ddof=1)) if len(values) > 1 else None,
                       ratio_of_mean_times_vs_full=float(values.mean() / np.mean([r["total_seconds"] for r in full.values()])),
                       paired_ratios_vs_full=ratios, paired_wins_vs_full=sum(r < 1 for r in ratios),
                       max_iterations=max(r["max_iterations"] for r in runs),
                       max_true_residual=max(r["max_true_residual"] for r in runs),
                       mean_total_with_initialization_seconds=float(np.mean([r["total_with_initialization_seconds"] for r in runs])),
                       mean_stage_maxima={key: float(np.mean([sum(s["stage_maxima"][key] for s in r["steps"]) for r in runs]))
                                           for key in ["monitor", "local", "coarse", "solve_and_certification"]})
        result["summary"][arm] = summary
    result["status"] = "complete"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(a.output, result)
    for step in local_steps:
        step["matrix"].destroy()
        step["b"].destroy()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        COMM.Abort(1)

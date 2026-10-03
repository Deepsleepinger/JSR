#!/usr/bin/env python3
"""Export the existing FEM sequences unchanged for deterministic MPI replay."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import dolfin as d
from mpi4py import MPI
from jsr.partition import create_3d_overlapping_partition
from benchmarks.run_four_regime_budget_response import assemble_regime_trajectory

STEPS = {"gentle_single_front": 10, "multi_front_churn": 16}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh-n", type=int, default=24)
    p.add_argument("--regimes", nargs="+", choices=list(STEPS), default=list(STEPS))
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/dc_jsr_parallel_inputs")
    p.add_argument("--verify-existing", action="store_true", help="Reassemble, compare every array, preserve frozen files")
    a = p.parse_args()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Prepare the common input once in one process.")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    verification = []
    for regime in a.regimes:
        start = time.perf_counter()
        mesh = d.UnitCubeMesh(d.MPI.comm_self, a.mesh_n, a.mesh_n, a.mesh_n)
        space = d.FunctionSpace(mesh, "Lagrange", 1)
        coords = space.tabulate_dof_coordinates()
        part = create_3d_overlapping_partition(coords, (2, 2, 2), 1, a.mesh_n)
        mats, rhs, csrs, diags = assemble_regime_trajectory(regime, mesh, space, STEPS[regime], .05)
        assembly_seconds = time.perf_counter() - start
        arrays = dict(coordinates=coords, core_ids=part["core_ids"], weights=part["weights"])
        for cid, indices in enumerate(part["local_indices"]):
            arrays["indices_%d" % cid] = indices
        for step, ((ia, ja, values), b, diag) in enumerate(zip(csrs, rhs, diags)):
            for key, value in [("ia", ia), ("ja", ja), ("values", values),
                               ("rhs", b.getArray(readonly=True)), ("diag", diag)]:
                arrays["%s_%d" % (key, step)] = np.array(value, copy=True)
        output = a.output_dir / ("%s_n%d.npz" % (regime, a.mesh_n))
        if a.verify_existing:
            with np.load(str(output), allow_pickle=False) as stored:
                if set(stored.files) != set(arrays):
                    raise RuntimeError("Input array keys differ")
                for key, value in arrays.items():
                    if not np.array_equal(value, stored[key]):
                        raise RuntimeError("Reassembled input differs: %s" % key)
            fingerprint = hashlib.sha256()
            for step in range(STEPS[regime] + 1):
                for key in ["ia", "ja", "values", "rhs"]:
                    fingerprint.update(np.ascontiguousarray(arrays["%s_%d" % (key, step)]).tobytes())
            reference = ROOT / "results/dc_jsr_certification_audit_n24.json"
            old_match = None
            if a.mesh_n == 24 and reference.exists():
                expected = json.loads(reference.read_text())["regimes"][regime]["trajectory_sha256"]
                old_match = fingerprint.hexdigest() == expected
                if not old_match:
                    raise RuntimeError("N24 input does not match the original 876-solve audit")
            verification.append(dict(regime=regime, mesh_n=a.mesh_n, assembly_seconds=assembly_seconds,
                                     all_arrays_identical=True, original_n24_audit_identical=old_match,
                                     trajectory_sha256=fingerprint.hexdigest(),
                                     frozen_npz_sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
            print("Verified all input arrays", output.name, "assembly seconds", round(assembly_seconds, 3), flush=True)
            for mat, b in zip(mats, rhs):
                mat.destroy()
                b.destroy()
            continue
        np.savez_compressed(str(output), **arrays)
        manifest = dict(regime=regime, mesh_n=a.mesh_n, steps=STEPS[regime], dt=.05,
                        n_dofs=space.dim(), subdomain_count=8, overlap_layers=1,
                        sha256=hashlib.sha256(output.read_bytes()).hexdigest(), assembly_seconds=assembly_seconds,
                        source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                       for f in sorted(ROOT.glob("benchmarks/*.py"))
                                       if "parallel" not in f.name},
                        scope="Existing preassembled FEM operator replay; u_prev=0; not new PDE time integration")
        output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("Prepared", output, space.dim(), "DOFs", flush=True)
        for mat, b in zip(mats, rhs):
            mat.destroy()
            b.destroy()
    if a.verify_existing:
        target = ROOT / "results/dc_jsr_parallel_audit" / ("input_reassembly_n%d.json" % a.mesh_n)
        target.write_text(json.dumps(verification, indent=2) + "\n")


if __name__ == "__main__":
    main()

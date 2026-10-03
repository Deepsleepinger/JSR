#!/usr/bin/env python3
"""Repeat the complete existing 16-step source cycle, without resetting factors."""
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
from benchmarks.run_multi_front_starvation_stress import multi_front_trajectory_sources


def cycle_step(step):
    return 0 if step == 0 else (step - 1) % 16 + 1


def slice_geometry(mesh, space):
    coords = space.tabulate_dof_coordinates()
    cells = d.vertex_to_dof_map(space)[mesh.cells()]
    slices = []
    for z in (.25, .5, .75):
        ids = np.flatnonzero(np.isclose(coords[:, 2], z, atol=1e-12, rtol=0))
        hit = np.isin(cells, ids)
        candidate = cells[hit.sum(axis=1) >= 3]
        faces = []
        for omitted in range(4):
            face = candidate[:, [k for k in range(4) if k != omitted]]
            face = face[np.all(np.isin(face, ids), axis=1)]
            faces.append(np.sort(face, axis=1))
        faces = np.unique(np.concatenate(faces), axis=0)
        triangles = np.searchsorted(ids, faces)
        assert np.all(ids[triangles] == faces)
        area = np.abs(np.cross(coords[faces[:, 1], :2] - coords[faces[:, 0], :2],
                               coords[faces[:, 2], :2] - coords[faces[:, 0], :2])).sum() / 2
        assert abs(area - 1) < 1e-10
        slices.append(dict(z=z, nodes=np.round(coords[ids, :2], 8).tolist(),
                           triangles=triangles.tolist(), area=float(area)))
    return slices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-n", required=True, type=int)
    parser.add_argument("--steps", default=64, type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/dc_jsr_long_horizon_inputs")
    args = parser.parse_args()
    assert MPI.COMM_WORLD.size == 1
    assert args.mesh_n % 4 == 0 and args.steps > 16 and args.steps % 16 == 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / ("multi_front_churn_n%d_t%d.npz" % (args.mesh_n, args.steps))
    if path.exists() or path.with_suffix(".json").exists():
        raise RuntimeError("Refusing to replace frozen inputs: %s" % path)
    start = time.perf_counter()
    mesh = d.UnitCubeMesh(d.MPI.comm_self, args.mesh_n, args.mesh_n, args.mesh_n)
    space = d.FunctionSpace(mesh, "Lagrange", 1)
    coords = space.tabulate_dof_coordinates()
    part = create_3d_overlapping_partition(coords, (2, 2, 2), 1, args.mesh_n)
    mats, rhs, csrs, diags = assemble_regime_trajectory("multi_front_churn", mesh, space, 16, .05)
    arrays = dict(coordinates=coords, core_ids=part["core_ids"], weights=part["weights"])
    arrays.update({"indices_%d" % cid: indices for cid, indices in enumerate(part["local_indices"])})
    fingerprint = hashlib.sha256()
    for step in range(args.steps + 1):
        phase = cycle_step(step)
        ia, ja, values = csrs[phase]
        for key, value in [("ia", ia), ("ja", ja), ("values", values),
                           ("rhs", rhs[phase].getArray(readonly=True)), ("diag", diags[phase])]:
            arrays["%s_%d" % (key, step)] = value
            if key != "diag":
                fingerprint.update(np.ascontiguousarray(value).tobytes())
    slices = slice_geometry(mesh, space)
    # Confirm periodic sharing represents identical prescribed matrices and RHS.
    assert all(arrays["values_%d" % s] is arrays["values_%d" % cycle_step(s)]
               for s in range(1, args.steps + 1))
    np.savez_compressed(str(path), **arrays)
    sources = [multi_front_trajectory_sources(cycle_step(s), 16) for s in range(args.steps + 1)]
    manifest = dict(regime="multi_front_churn", mesh_n=args.mesh_n, steps=args.steps,
                    dt=.05, n_dofs=space.dim(), tetrahedra=mesh.num_cells(),
                    subdomain_count=8, overlap_layers=1, source_cycle_steps=16,
                    source_cycle_repetitions=args.steps // 16,
                    state_reset_between_cycles=False, sources=sources,
                    coefficient=dict(k_solid=1., k_melt=20., r0=.12),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    trajectory_sha256=fingerprint.hexdigest(),
                    preparation_seconds=time.perf_counter()-start,
                    source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in [Path(__file__).resolve(), ROOT / "jsr/partition.py",
                                             ROOT / "benchmarks/run_four_regime_budget_response.py",
                                             ROOT / "benchmarks/run_multi_front_starvation_stress.py"]},
                    scope="64-step prescribed operator replay with complete 16-step periodic source cycles; "
                          "identical PDE assembly and u_prev=0; no claim of 64 distinct load states or transient temperature")
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    visual = dict(mesh_n=args.mesh_n, n_dofs=space.dim(), tetrahedra=mesh.num_cells(),
                  slices=slices, sources=sources, coefficient=manifest["coefficient"],
                  geometry_source="Actual tetrahedral mesh faces, mapped to P1 DOF coordinates")
    path.with_suffix(".geometry.json").write_text(json.dumps(visual, separators=(",", ":")) + "\n")
    for matrix, b in zip(mats, rhs):
        matrix.destroy()
        b.destroy()
    print("Prepared", path.name, space.dim(), "DOFs; actual slices", [len(s["triangles"]) for s in slices], flush=True)


if __name__ == "__main__":
    main()

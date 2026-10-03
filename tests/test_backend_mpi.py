"""MPI integration check against an independent dense two-level reference.

Run with the scientific Python via mpiexec -n 1/2/4. Exercises non-neighbour
matrix entries, overlap accumulation, stale factors and current coarse state.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jsr.backend_mpi import DistributedSchwarz, selected_csr_rows


def dense_csr(a):
    rows, cols = np.nonzero(a)
    ia = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=len(a))))).astype(PETSc.IntType)
    return ia, cols.astype(PETSc.IntType), a[rows, cols]


class DistributedReferenceTest(unittest.TestCase):
    def test_overlap_staleness_and_coarse(self):
        comm = MPI.COMM_WORLD
        self.assertIn(comm.size, [1, 2, 4])
        n, m = 17, 4
        a = np.diag(np.full(n, 4.0)) + np.diag(np.full(n-1, -1.0), 1) + np.diag(np.full(n-1, -1.0), -1)
        a[2, 10] = a[10, 2] = -.15
        indices = [np.arange(0, 6), np.arange(4, 10), np.arange(8, 14), np.arange(12, 17)]
        memberships = sum(np.isin(np.arange(n), idx).astype(int) for idx in indices)
        core = np.minimum(np.arange(n) // 4, m-1)
        weights = 1.0 / memberships
        metadata = dict(local_indices=indices, core_ids=core, weights=weights, subdomain_count=m)
        counts = [n // comm.size + (rank < n % comm.size) for rank in range(comm.size)]
        first, last = sum(counts[:comm.rank]), sum(counts[:comm.rank+1])
        csr = dense_csr(a)
        local = selected_csr_rows(csr, np.arange(first, last))
        matrix = PETSc.Mat().createAIJ(size=((last-first, n), (last-first, n)), csr=local, comm=PETSc.COMM_WORLD)
        matrix.assemble()
        x = matrix.createVecRight()
        y = x.duplicate()
        owned = range(comm.rank, m, comm.size)
        union = np.unique(np.concatenate([indices[c] for c in owned]))
        overlap = selected_csr_rows(csr, union)
        backend = DistributedSchwarz(x, metadata, overlap[:2])
        built = [a.copy() for _ in range(m)]
        z = np.eye(m)[core]
        probe = np.random.RandomState(52).standard_normal(n)
        try:
            for step in [0, 1]:
                if step:
                    # Only block 0 is rebuilt; other factors intentionally stale.
                    a = a.copy()
                    a[np.diag_indices(n)] += np.linspace(.1, .9, n)
                    csr = dense_csr(a)
                    local = selected_csr_rows(csr, np.arange(first, last))
                    overlap = selected_csr_rows(csr, union)
                    selected = [0] if 0 in backend.owned else []
                    built[0] = a.copy()
                else:
                    selected = list(backend.owned)
                backend.refresh_local(overlap[2], step, selected)
                backend.refresh_coarse(local, step)
                x.getArray()[:] = probe[first:last]
                backend.apply(None, x, y)
                actual = np.concatenate(comm.allgather(y.getArray(readonly=True).copy()))
                expected = z.dot(np.linalg.solve(z.T.dot(a).dot(z), z.T.dot(probe)))
                for cid, idx in enumerate(indices):
                    expected[idx] += weights[idx] * np.linalg.solve(built[cid][np.ix_(idx, idx)], weights[idx]*probe[idx])
                self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected), 1e-12)
                self.assertGreater(float(probe.dot(actual)), 0.0)
                np.testing.assert_allclose(backend.coarse_dense, z.T.dot(a).dot(z), atol=1e-12)
        finally:
            backend.close()
            x.destroy()
            y.destroy()
            matrix.destroy()


if __name__ == "__main__":
    unittest.main()

"""Unit tests for overlapping domain decomposition and MUMPS backend."""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jsr import partition, backend_mumps


class TestOverlappingMUMPS(unittest.TestCase):
    def test_3d_overlapping_partition_geometry(self):
        """Verify 3D partition covers all nodes and establishes valid overlap."""
        N = 12
        x = np.linspace(0.0, 1.0, N)
        xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
        coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        part = partition.create_3d_overlapping_partition(
            coordinates=coords,
            grid=(2, 2, 2),
            overlap_layers=1,
            mesh_n=N,
        )

        self.assertEqual(part["subdomain_count"], 8)
        self.assertEqual(len(part["local_indices"]), 8)
        # Every node must be covered by at least 1 subdomain
        self.assertTrue(np.all(part["memberships"] >= 1))
        # Boundary overlapping nodes must belong to multiple subdomains
        self.assertTrue(np.any(part["memberships"] > 1))
        # Partition of unity weights must sum correctly per node
        weights = part["weights"]
        memberships = part["memberships"]
        np.testing.assert_allclose(weights * memberships, 1.0)

    @unittest.skipUnless(backend_mumps and backend_mumps.HAS_PETSC, "PETSc not available")
    def test_mumps_factor_lifecycle(self):
        """Verify LocalMUMPSFactor creates, factorizes, and solves correctly."""
        # Simple 3D Laplacian submatrix
        from petsc4py import PETSc
        N = 100
        indptr = np.arange(N + 1, dtype=np.int64) * 3
        # Ensure tridiagonal structure
        indptr[0] = 0
        cols = []
        vals = []
        for i in range(N):
            cols.append(i)
            vals.append(4.0)
            if i > 0:
                cols.append(i - 1)
                vals.append(-1.0)
            if i < N - 1:
                cols.append(i + 1)
                vals.append(-1.0)
        # Sort cols
        ia = np.zeros(N + 1, dtype=np.int64)
        ja = []
        aa = []
        for i in range(N):
            row_c = []
            row_v = []
            row_c.append(i)
            row_v.append(4.0)
            if i > 0:
                row_c.append(i - 1)
                row_v.append(-1.0)
            if i < N - 1:
                row_c.append(i + 1)
                row_v.append(-1.0)
            order = np.argsort(row_c)
            ja.extend(np.array(row_c)[order])
            aa.extend(np.array(row_v)[order])
            ia[i + 1] = len(ja)

        mat = backend_mumps.create_petsc_aij_matrix(ia, ja, aa, N)
        indices = np.arange(N, dtype=np.int64)
        weights = np.ones(N, dtype=np.float64)

        factor = backend_mumps.LocalMUMPSFactor(0, indices, weights)
        factor.refresh(mat, state_index=0)

        self.assertIsNotNone(factor.ksp)
        self.assertEqual(factor.refresh_count, 1)

        x = np.ones(N, dtype=np.float64)
        y = np.zeros(N, dtype=np.float64)
        factor.apply_weighted(x, y)

        self.assertTrue(np.all(y > 0.0))
        factor.destroy()
        mat.destroy()


if __name__ == "__main__":
    unittest.main()

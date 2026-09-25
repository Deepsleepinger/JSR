"""
HPC-grade Two-Level Overlapping Schwarz Preconditioner Backend with PETSc/MUMPS.

Implements true Overlapping Restricted Additive Schwarz (RAS) and Symmetric Weighted
Additive Schwarz with local sparse direct factorizations via PETSc/MUMPS and coupled
exact Galerkin coarse correction.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

try:
    from petsc4py import PETSc
    HAS_PETSC = True
except ImportError:
    PETSc = None
    HAS_PETSC = False


def create_petsc_aij_matrix(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_rows: int,
    n_cols: Optional[int] = None,
) -> PETSc.Mat:
    """Instantiate an assembled PETSc AIJ sparse matrix from CSR arrays."""
    if not HAS_PETSC:
        raise RuntimeError("petsc4py is required for PETScMUMPS backend.")
    if n_cols is None:
        n_cols = n_rows

    mat = PETSc.Mat().createAIJ(
        size=(int(n_rows), int(n_cols)),
        csr=(
            np.asarray(indptr, dtype=PETSc.IntType),
            np.asarray(indices, dtype=PETSc.IntType),
            np.asarray(data, dtype=PETSc.ScalarType),
        ),
        comm=PETSc.COMM_SELF,
    )
    mat.assemble()
    return mat


class LocalMUMPSFactor:
    """
    Persistent local subdomain solver using PETSc/MUMPS sparse direct factorization.

    Maintains a persistent KSP/PC handle across time steps. When the local operator
    is updated, setting the operator on the same handle and calling setUp() triggers
    MUMPS numeric refactorization while reusing symbolic analysis where possible.
    """

    def __init__(self, cid: int, indices: np.ndarray, weights: np.ndarray):
        self.cid = int(cid)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.local_size = self.indices.size

        self.ksp: Optional[PETSc.KSP] = None
        self.matrix: Optional[PETSc.Mat] = None
        self.rhs_vec: Optional[PETSc.Vec] = None
        self.sol_vec: Optional[PETSc.Vec] = None

        self.build_state: int = -1
        self.refresh_count: int = 0
        self.last_setup_seconds: float = 0.0

    def refresh(self, global_matrix: PETSc.Mat, state_index: int) -> float:
        """Extract overlapping submatrix and perform MUMPS sparse direct factorization."""
        t0 = time.perf_counter()
        iset = PETSc.IS().createGeneral(
            self.indices.astype(PETSc.IntType), comm=PETSc.COMM_SELF
        )
        new_sub = global_matrix.createSubMatrix(iset, iset)
        iset.destroy()

        if self.ksp is None:
            self.ksp = PETSc.KSP().create(PETSc.COMM_SELF)
            self.ksp.setType("preonly")
            pc = self.ksp.getPC()
            pc.setType("lu")
            pc.setFactorSolverType("mumps")
            self.ksp.setOperators(new_sub)
            self.ksp.setUp()
        else:
            # Persistent KSP handle: update operator and perform numeric refactorization
            self.ksp.setOperators(new_sub)
            self.ksp.setUp()

        reason = int(self.ksp.getConvergedReason())
        if reason < 0:
            raise RuntimeError(f"MUMPS setup failed for subdomain {self.cid}, reason: {reason}")

        if self.matrix is not None:
            self.matrix.destroy()
        if self.rhs_vec is None:
            self.rhs_vec = new_sub.createVecRight()
            self.sol_vec = new_sub.createVecRight()

        self.matrix = new_sub
        self.build_state = int(state_index)
        self.refresh_count += 1
        elapsed = time.perf_counter() - t0
        self.last_setup_seconds = elapsed
        return elapsed

    def apply_weighted(self, x_global: np.ndarray, y_global: np.ndarray) -> None:
        """
        Symmetric weighted Additive Schwarz local solve:
        y += R_i^T D_i A_i^{-1} D_i R_i x
        """
        xa = x_global[self.indices]
        wa = self.weights[self.indices]

        # Restrict with weight: D_i R_i x
        self.rhs_vec.getArray()[:] = wa * xa
        self.ksp.solve(self.rhs_vec, self.sol_vec)

        # Prolongate with weight: R_i^T D_i (A_i^{-1} D_i R_i x)
        ya = self.sol_vec.getArray(readonly=True)
        y_global[self.indices] += wa * ya

    def apply_ras(self, x_global: np.ndarray, y_global: np.ndarray, is_core_mask: np.ndarray) -> None:
        """
        Restricted Additive Schwarz (RAS) local solve:
        Restricts from overlapping subdomain, extends strictly to non-overlapping core:
        y += \tilde{R}_i^T A_i^{-1} R_i x
        """
        xa = x_global[self.indices]
        self.rhs_vec.getArray()[:] = xa
        self.ksp.solve(self.rhs_vec, self.sol_vec)

        ya = self.sol_vec.getArray(readonly=True)
        core_locs = np.flatnonzero(is_core_mask[self.indices])
        core_global = self.indices[core_locs]
        y_global[core_global] += ya[core_locs]

    def destroy(self) -> None:
        for obj in (self.rhs_vec, self.sol_vec, self.matrix, self.ksp):
            if obj is not None:
                try:
                    obj.destroy()
                except Exception:
                    pass


class OverlappingMUMPSSchwarzBackend:
    """
    Two-Level Overlapping Schwarz Preconditioner Context with MUMPS.

    Supports selective maintenance of local subdomain direct factors
    and adaptive coarse Galerkin update across time steps.
    """

    def __init__(
        self,
        n_dofs: int,
        partition: Dict[str, object],
        use_ras: bool = False,
    ):
        if not HAS_PETSC:
            raise RuntimeError("petsc4py is required for OverlappingMUMPSSchwarzBackend.")

        self.n_dofs = int(n_dofs)
        self.partition = partition
        self.use_ras = bool(use_ras)

        self.local_indices: List[np.ndarray] = partition["local_indices"]
        self.weights: np.ndarray = partition["weights"]
        self.core_ids: np.ndarray = partition["core_ids"]
        self.subdomain_count = len(self.local_indices)

        self.factors: List[LocalMUMPSFactor] = [
            LocalMUMPSFactor(cid, idx, self.weights)
            for cid, idx in enumerate(self.local_indices)
        ]

        # Coarse space objects
        self.coarse_matrix: Optional[PETSc.Mat] = None
        self.coarse_ksp: Optional[PETSc.KSP] = None
        self.coarse_rhs: Optional[PETSc.Vec] = None
        self.coarse_sol: Optional[PETSc.Vec] = None
        self.coarse_state: int = -1
        self.coarse_refresh_count: int = 0

        # Timing and accounting
        self.total_setup_seconds: float = 0.0
        self.total_apply_seconds: float = 0.0
        self.apply_call_count: int = 0

    def refresh_local(
        self,
        global_petsc_matrix: PETSc.Mat,
        state_index: int,
        selected_subdomains: Iterable[int],
    ) -> float:
        """Selectively refactorize only the specified subdomains using MUMPS."""
        t0 = time.perf_counter()
        for cid in selected_subdomains:
            self.factors[cid].refresh(global_petsc_matrix, state_index)
        elapsed = time.perf_counter() - t0
        self.total_setup_seconds += elapsed
        return elapsed

    def refresh_coarse(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        state_index: int,
    ) -> float:
        """
        Assemble and factorize the exact Galerkin coarse matrix A_0 = Z^T A Z.
        Here Z is the standard piecewise-constant core partition indicator matrix.
        """
        t0 = time.perf_counter()
        nc = self.subdomain_count
        dense_coarse = np.zeros((nc, nc), dtype=np.float64)

        # Vectorized assembly of coarse entries: A_0[ci, cj] = sum_{i in core_ci, j in core_cj} A[i, j]
        for row in range(self.n_dofs):
            ci = int(self.core_ids[row])
            lo = int(indptr[row])
            hi = int(indptr[row + 1])
            cols = indices[lo:hi]
            vals = data[lo:hi]
            np.add.at(dense_coarse[ci], self.core_ids[cols], vals)

        # Symmetrize numerical roundoff
        dense_coarse = 0.5 * (dense_coarse + dense_coarse.T)

        # Convert to CSR PETSc matrix
        rows, cols, vals = [], [], []
        for i in range(nc):
            nz = np.flatnonzero(np.abs(dense_coarse[i]) > 0.0)
            rows.append(np.full(nz.size, i, dtype=np.int64))
            cols.append(nz.astype(np.int64))
            vals.append(dense_coarse[i, nz])

        ia = np.zeros(nc + 1, dtype=np.int64)
        for i in range(nc):
            ia[i + 1] = ia[i] + rows[i].size
        ja = np.concatenate(cols).astype(np.int64) if cols else np.empty(0, dtype=np.int64)
        aa = np.concatenate(vals).astype(np.float64) if vals else np.empty(0, dtype=np.float64)

        new_coarse = create_petsc_aij_matrix(ia, ja, aa, nc)

        if self.coarse_ksp is None:
            self.coarse_ksp = PETSc.KSP().create(PETSc.COMM_SELF)
            self.coarse_ksp.setType("preonly")
            pc = self.coarse_ksp.getPC()
            pc.setType("lu")
            pc.setFactorSolverType("mumps")
            self.coarse_ksp.setOperators(new_coarse)
            self.coarse_ksp.setUp()
            self.coarse_rhs = new_coarse.createVecRight()
            self.coarse_sol = new_coarse.createVecRight()
        else:
            self.coarse_ksp.setOperators(new_coarse)
            self.coarse_ksp.setUp()

        if self.coarse_matrix is not None:
            self.coarse_matrix.destroy()
        self.coarse_matrix = new_coarse
        self.coarse_state = int(state_index)
        self.coarse_refresh_count += 1

        elapsed = time.perf_counter() - t0
        self.total_setup_seconds += elapsed
        return elapsed

    def apply(self, x_in: np.ndarray, y_out: np.ndarray) -> None:
        """
        Apply the Two-Level Schwarz preconditioner:
        y = M^{-1} x = \sum_i R_i^T D_i A_i^{-1} D_i R_i x + R_0^T A_0^{-1} R_0 x
        """
        t0 = time.perf_counter()
        y_out.fill(0.0)

        # 1. Level 1: Subdomain local solves
        for factor in self.factors:
            factor.apply_weighted(x_in, y_out)

        # 2. Level 2: Coarse Galerkin solve
        if self.coarse_ksp is not None:
            nc = self.subdomain_count
            coarse_rhs_arr = np.zeros(nc, dtype=np.float64)
            # Restriction to coarse: R_0 x = sum_{i in core_c} x[i]
            np.add.at(coarse_rhs_arr, self.core_ids, x_in)

            self.coarse_rhs.getArray()[:] = coarse_rhs_arr
            self.coarse_ksp.solve(self.coarse_rhs, self.coarse_sol)

            coarse_sol_arr = self.coarse_sol.getArray(readonly=True)
            # Prolongation from coarse: y += R_0^T (coarse_sol)
            y_out += coarse_sol_arr[self.core_ids]

        elapsed = time.perf_counter() - t0
        self.total_apply_seconds += elapsed
        self.apply_call_count += 1

    def destroy(self) -> None:
        for factor in self.factors:
            factor.destroy()
        for obj in (self.coarse_rhs, self.coarse_sol, self.coarse_matrix, self.coarse_ksp):
            if obj is not None:
                try:
                    obj.destroy()
                except Exception:
                    pass

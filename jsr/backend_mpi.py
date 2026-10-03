"""Distributed symmetric two-level Schwarz for the fixed-operator replay audit.

Global matrix/vector storage and Krylov operations use COMM_WORLD. Local LU
factors use COMM_SELF only on their owning rank; overlap vectors use collective
PETSc VecScatter. The small fixed coarse problem is redundantly solved on ranks.
"""
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from jsr.backend_mumps import create_petsc_aij_matrix


def selected_csr_rows(csr, rows):
    """Copy specified CSR rows, retaining global column numbering."""
    ia, ja, values = csr
    rows = np.asarray(rows, dtype=np.int64)
    counts = ia[rows + 1] - ia[rows]
    out_ia = np.concatenate(([0], np.cumsum(counts))).astype(PETSc.IntType)
    offsets = np.repeat(ia[rows] - out_ia[:-1], counts)
    positions = np.arange(int(out_ia[-1])) + offsets
    return out_ia, ja[positions].astype(PETSc.IntType), values[positions].copy()


class DistributedSchwarz:
    def __init__(self, template, metadata, overlap_pattern, comm=MPI.COMM_WORLD):
        self.comm = comm
        self.rank = comm.rank
        self.n = template.getSize()
        self.first, self.last = template.getOwnershipRange()
        self.core_ids = metadata["core_ids"]
        self.m = int(metadata["subdomain_count"])
        self.owned = list(range(self.rank, self.m, comm.size))
        self.indices = metadata["local_indices"]
        self.union = np.unique(np.concatenate([self.indices[c] for c in self.owned]))
        self.union_weights = metadata["weights"][self.union]
        self.slots = {c: np.searchsorted(self.union, self.indices[c]) for c in self.owned}
        self.norm_denominators = {}
        self.factors = {}
        self.versions = {c: -1 for c in self.owned}
        self.state = {c: 0.0 for c in self.owned}
        self.pre_state = self.state.copy()
        self.patterns = {}
        ia, ja = overlap_pattern
        row = np.repeat(np.arange(len(self.union)), np.diff(ia))
        for cid in self.owned:
            idx = self.indices[cid]
            local_col = np.searchsorted(idx, ja)
            valid_col = local_col < len(idx)
            valid_col[valid_col] &= idx[local_col[valid_col]] == ja[valid_col]
            valid_row = np.isin(row, self.slots[cid])
            positions = np.flatnonzero(valid_row & valid_col)
            local_rows = np.searchsorted(self.slots[cid], row[positions])
            counts = np.bincount(local_rows, minlength=len(idx))
            sub_ia = np.concatenate(([0], np.cumsum(counts))).astype(PETSc.IntType)
            self.patterns[cid] = (sub_ia, local_col[positions].astype(PETSc.IntType), positions)
        self.overlap_x = PETSc.Vec().createSeq(len(self.union), comm=PETSc.COMM_SELF)
        self.overlap_y = self.overlap_x.duplicate()
        src_is = PETSc.IS().createGeneral(self.union.astype(PETSc.IntType), comm=PETSc.COMM_SELF)
        dst_is = PETSc.IS().createStride(len(self.union), first=0, step=1, comm=PETSc.COMM_SELF)
        self.scatter = PETSc.Scatter().create(template, src_is, self.overlap_x, dst_is)
        src_is.destroy()
        dst_is.destroy()
        self.coarse = None
        self.coarse_dense = None
        self.coarse_state = -1
        self.apply_calls = 0
        self.apply_seconds = dict(overlap=0.0, local=0.0, coarse=0.0)

    def decide(self, diag, previous, epsilon, full=False):
        if full:
            self.pre_state = {cid: 0.0 for cid in self.owned}
            return list(self.owned)
        selected = []
        for cid in self.owned:
            slots = self.slots[cid]
            increment = float(np.linalg.norm(diag[slots] - previous[slots])) / max(
                float(np.linalg.norm(diag[slots])), 1e-14)
            self.pre_state[cid] = self.state[cid] + increment
            if self.pre_state[cid] > epsilon:
                selected.append(cid)
        return selected

    def refresh_local(self, values, step, selected):
        for cid in selected:
            ia, ja, positions = self.patterns[cid]
            matrix = create_petsc_aij_matrix(ia, ja, values[positions], len(self.indices[cid]))
            if cid not in self.factors:
                ksp = PETSc.KSP().create(PETSc.COMM_SELF)
                ksp.setType("preonly")
                ksp.getPC().setType("lu")
                ksp.getPC().setFactorSolverType("mumps")
                b = matrix.createVecRight()
                x = b.duplicate()
                self.factors[cid] = dict(ksp=ksp, b=b, x=x, matrix=None)
            factor = self.factors[cid]
            factor["ksp"].setOperators(matrix)
            factor["ksp"].setUp()
            if factor["matrix"] is not None:
                factor["matrix"].destroy()
            factor["matrix"] = matrix
            self.versions[cid] = step
        # Commit only after all local setup calls succeed. A rank error aborts.
        for cid in self.owned:
            self.state[cid] = 0.0 if cid in selected else self.pre_state[cid]

    def refresh_coarse(self, csr, step):
        ia, ja, values = csr
        row_core = np.repeat(self.core_ids[self.first:self.last], np.diff(ia))
        local = np.bincount(row_core * self.m + self.core_ids[ja], weights=values,
                            minlength=self.m * self.m).reshape(self.m, self.m)
        dense = np.empty_like(local)
        self.comm.Allreduce(local, dense, op=MPI.SUM)
        dense = .5 * (dense + dense.T)
        ia0 = np.arange(0, self.m * self.m + 1, self.m, dtype=PETSc.IntType)
        ja0 = np.tile(np.arange(self.m, dtype=PETSc.IntType), self.m)
        matrix = create_petsc_aij_matrix(ia0, ja0, dense.ravel(), self.m)
        if self.coarse is None:
            ksp = PETSc.KSP().create(PETSc.COMM_SELF)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            b = matrix.createVecRight()
            self.coarse = dict(ksp=ksp, b=b, x=b.duplicate(), matrix=None)
        self.coarse["ksp"].setOperators(matrix)
        self.coarse["ksp"].setUp()
        if self.coarse["matrix"] is not None:
            self.coarse["matrix"].destroy()
        self.coarse["matrix"] = matrix
        self.coarse_dense = dense
        self.coarse_state = step

    def apply(self, pc, x, y):
        t0 = time.perf_counter()
        self.scatter.scatter(x, self.overlap_x, addv=PETSc.InsertMode.INSERT_VALUES,
                             mode=PETSc.ScatterMode.FORWARD)
        xa = self.overlap_x.getArray(readonly=True)
        ya = self.overlap_y.getArray()
        ya.fill(0.0)
        t1 = time.perf_counter()
        for cid in self.owned:
            factor = self.factors[cid]
            slots = self.slots[cid]
            weights = self.union_weights[slots]
            factor["b"].getArray()[:] = weights * xa[slots]
            factor["ksp"].solve(factor["b"], factor["x"])
            ya[slots] += weights * factor["x"].getArray(readonly=True)
        t2 = time.perf_counter()
        y.set(0.0)
        self.scatter.scatter(self.overlap_y, y, addv=PETSc.InsertMode.ADD_VALUES,
                             mode=PETSc.ScatterMode.REVERSE)
        t3 = time.perf_counter()
        rhs_local = np.bincount(self.core_ids[self.first:self.last],
                                weights=x.getArray(readonly=True), minlength=self.m)
        rhs = np.empty_like(rhs_local)
        self.comm.Allreduce(rhs_local, rhs, op=MPI.SUM)
        self.coarse["b"].getArray()[:] = rhs
        self.coarse["ksp"].solve(self.coarse["b"], self.coarse["x"])
        y.getArray()[:] += self.coarse["x"].getArray(readonly=True)[self.core_ids[self.first:self.last]]
        t4 = time.perf_counter()
        self.apply_seconds["overlap"] += (t1 - t0) + (t3 - t2)
        self.apply_seconds["local"] += t2 - t1
        self.apply_seconds["coarse"] += t4 - t3
        self.apply_calls += 1

    def close(self):
        self.scatter.destroy()
        self.overlap_x.destroy()
        self.overlap_y.destroy()
        for factor in list(self.factors.values()) + ([self.coarse] if self.coarse else []):
            for key in ["b", "x", "ksp", "matrix"]:
                if factor[key] is not None:
                    factor[key].destroy()

"""
JSR: Stateful Joint Maintenance of Multilevel Schwarz Preconditioners
=====================================================================

A stateful, causal, stale-aware domain decomposition preconditioning framework
for evolving sparse linear systems arising from transient PDE simulations.

Modules:
- `partition`: Physical coordinate-based overlapping domain decomposition (RAS & AS).
- `backend_mumps`: Production HPC backend using overlapping subdomains and PETSc/MUMPS.
- `backend`: Dual-engine / dense fallback backend for zero-dependency testing.
- `monitor`: Spatial Frobenius drift sensing and factor age monitoring.
- `selector`: Causal action selector based on mass95 Pareto energy truncation.
"""

from . import partition
from . import monitor
from . import selector
from . import backend

try:
    from . import backend_mumps
    HAS_MUMPS = backend_mumps.HAS_PETSC
except Exception:
    backend_mumps = None
    HAS_MUMPS = False

__version__ = "1.0.0"
__author__ = "Stateful Schwarz JSR Research Group"
__all__ = [
    "partition",
    "monitor",
    "selector",
    "backend",
    "backend_mumps",
    "HAS_MUMPS",
    "__version__",
]

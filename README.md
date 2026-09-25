# Stateful Selective Maintenance of Two-Level Schwarz Preconditioners (JSR)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Backend: PETSc / MUMPS / NumPy](https://img.shields.io/badge/Backend-PETSc%20%7C%20MUMPS%20%7C%20NumPy-green.svg)](https://petsc.org/)
[![Status: Certified](https://img.shields.io/badge/Residual%20Certificate-Passed%20(1e--8)-brightgreen.svg)]()

> **Official Open-Source Research Codebase** accompanying the manuscript:  
> *"Stateful Selective Maintenance of Two-Level Schwarz Preconditioners for Evolving Sparse Linear Systems"*

---

## 📌 1. Overview & Motivation

In transient physical simulations—such as moving phase-change interfaces, thermal softening localization, non-Newtonian flow fronts, and dynamic crack damage—numerical discretizations yield a sequence of large, sparse, symmetric positive definite (SPD) linear systems:
$$A(t) x(t) = b(t), \quad t = 1, 2, \dots, T$$

Traditional domain decomposition solvers (e.g., Two-Level Additive / Restricted Additive Schwarz) face an expensive dilemma:
1. **Always Full Rebuild**: Recomputing all subdomain sparse factorizations and global coarse operators at every time step keeps Krylov (PCG) iteration counts minimal, but **preconditioner setup consumes 70%–90% of total simulation time**.
2. **Blind Static Reuse**: Keeping the initial preconditioner fixed costs zero setup time, but localized physical evolution causes severe spectral mismatch, resulting in **exponential iteration runaway or outright solver divergence**.

### The JSR Paradigm
**Joint Stateful Repair (JSR)** exploits the fundamental property of **spatial locality** in physical evolution:
* The discrete operator changes intensely only across a localized sub-region (typically 5%–15% of the domain), while the vast background remains quasi-static.
* By treating local subdomain solvers and the coarse Galerkin space as **persistent stateful objects**, JSR selectively refactorizes only the physically drifted subdomains while adaptively updating the global coarse projection.
* **Result**: Substantial reduction in preconditioner setup overhead, preserving near-optimal Krylov iteration counts and translating into **certified end-to-end wall-clock net speedup**.

---

## 🚀 2. Key Architectural Features

- **Dual-Level Stateful Abstraction**: Subdomain factors ($A_i^{-1}$) and coarse projection ($A_0^{-1} = (Z^T A Z)^{-1}$) carry explicit state metadata (age, cumulative drift, factorization handles).
- **Causal Pareto Truncation (`mass95`)**: Evaluates localized Frobenius operator drift and isolates the top 95% perturbation energy under an 80/20 law. Strictly causal with **zero future-information leakage**.
- **Independent True Residual Certification**: Bypasses internal Krylov estimates and independently computes the algebraic CSR residual:
  $$\|b - A x\|_2 / \|b\|_2 \le 1.0 \times 10^{-8}$$
- **Atomic Rollback & Escalation**: In the event of sudden physical shocks, transactional snapshots allow instant rollback and automatic state escalation (`reuse` $\to$ `local_partial` $\to$ `joint_partial` $\to$ `full_rebuild`).
- **Dual-Engine Execution**: Built-in adapter seamlessly runs on **PETSc/MUMPS** (for HPC performance) or **pure NumPy/SciPy** (for zero-dependency demonstration and education).

---

## 📂 3. Repository Directory Structure

```text
stateful_schwarz_jsr/
├── README.md                       # Comprehensive guide and documentation
├── LICENSE                         # Open-source MIT License
├── pyproject.toml                  # Modern Python packaging configuration
├── requirements.txt                # Core dependencies
│
├── jsr/                            # Core JSR Python Package (`import jsr`)
│   ├── __init__.py                 # Package exports and versioning
│   ├── monitor.py                  # Stateful drift sensing & risk monitoring (Frobenius norm, age penalty)
│   ├── selector.py                 # Causal action selector (mass95 Pareto truncation & action evaluation)
│   ├── backend.py                  # Two-level Schwarz preconditioner execution core & transactional snapshots
│   ├── changing_basis.py           # Adaptive spectral basis analysis
│   └── changing_backend.py         # Dynamic spectral coarse space backend
│
├── examples/                       # Runnable demonstration scripts
│   └── demo_7step_walkthrough.py   # ★ One-click 7-step slow-motion documentary walkthrough
│
├── web_lab/                        # Standalone interactive web animation laboratory
│   ├── index.html                  # Interactive browser UI
│   ├── app.js                      # Real-time simulation and canvas renderer
│   └── style.css                   # Modern styling
│
├── docs/                           # Methodological and theoretical documentation
│   ├── JSR_REAL_CASE_WALKTHROUGH.md # In-depth step-by-step mathematical dissection
│   ├── JSR_STUDY_EXPLANATION_GUIDE.md # Comprehensive study manual
│   └── FULL_RESEARCH_AND_EXPERIMENT_JOURNEY.md # Full research and ablation chronicle
│
├── benchmarks/                     # Paper benchmark reproduction scripts
│   ├── run_phase2_2_comparative.py # 6-arm comparative rollout experiment
│   ├── run_phase4_fixed_basis.py   # Coarse space ablation benchmark
│   └── run_phase5a_sparse_backend.py # Sparse GAMG baseline comparison
│
└── tests/                          # Automated test suite
    ├── __init__.py
    └── test_core_lifecycle.py      # Core lifecycle & causal property unit tests
```

---

## 💻 4. Installation & Environment

### Prerequisites
- Python 3.8 or newer
- NumPy $\ge$ 1.19.0
- (Optional, recommended for high performance) `petsc4py` with MUMPS support

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/deepsleepinger/stateful-schwarz-jsr.git
cd stateful-schwarz-jsr
pip install -r requirements.txt
```

To install the package in editable mode:
```bash
pip install -e .
```

---

## ⚡ 5. Quick Start: The 7-Step Slow-Motion Walkthrough

Experience the complete lifecycle of JSR on a realistic 144-subdomain, 14,400-DOF transient mesh evolution:
```bash
python examples/demo_7step_walkthrough.py
```

### What You Will See:
1. **Step 1: Input & Problem Setup** (Loading CSR matrices and subdomains);
2. **Step 2: Monitor Risk Sensing** (Computing Frobenius matrix drift and age penalties);
3. **Step 3: Selector Decision** (`mass95` Pareto truncation selecting the 15 damaged subdomains);
4. **Step 4: Backend Refactorization** (Updating local Cholesky/LU factors and coarse Galerkin triplet product);
5. **Step 5: Two-Level PCG Solve** (Rapid convergence in 9 iterations);
6. **Step 6: Residual Certification** (Independent CSR $\|b - Ax\|_2 \le 10^{-8}$ verification);
7. **Step 7: Ledger State Commit** (Cryptographic SHA-256 state ledger persistence).

---

## 🌐 6. Interactive Web Visualization Lab

We provide a zero-dependency, browser-based real-time simulation laboratory:
- Simply open `web_lab/index.html` in any modern web browser.
- Interactively watch a moving phase interface traverse a 2D domain.
- Visualize real-time subdomain heatmaps, Frobenius drift vectors, selector decisions, and PCG convergence history.

---

## 🧪 7. Running Unit Tests

Verify the codebase integrity using Python's standard `unittest`:
```bash
python -m unittest discover tests/
```

---

## 📜 8. Citation

If you find this work or codebase helpful in your research, please cite:
```bibtex
@article{stateful_schwarz_jsr_2026,
  title={Stateful Selective Maintenance of Two-Level Schwarz Preconditioners for Evolving Sparse Linear Systems},
  author={JSR Research Group},
  journal={Computer Methods in Applied Mechanics and Engineering},
  year={2026},
  note={Under review}
}
```

---

## 📄 9. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

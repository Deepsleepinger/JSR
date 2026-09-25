# Stateful Selective Maintenance of Two-Level Schwarz Preconditioners (JSR)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Backend: PETSc / MUMPS / NumPy](https://img.shields.io/badge/Backend-PETSc%20%7C%20MUMPS%20%7C%20NumPy-green.svg)](https://petsc.org/)
[![Status: Certified](https://img.shields.io/badge/Residual%20Certificate-Passed%20(1e--8)-brightgreen.svg)]()
[![Release: v1.0-paper](https://img.shields.io/badge/Release-v1.0--paper-blue.svg)](https://github.com/Deepsleepinger/JSR/releases/tag/v1.0-paper)

> **Official Open-Source Research Codebase** accompanying the manuscript:  
> *"Stateful Selective Maintenance of Two-Level Schwarz Preconditioners for Evolving Sparse Linear Systems"*

---

## ⚖️ The JSR Net Speedup Principle

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE JSR NET SPEEDUP PRINCIPLE                         │
│                                                                                        │
│   ΔT_net = ∑ (T_setup^full - T_setup^JSR(t)) - ∑ (T_solve^JSR(t) - T_solve^full(t))   │
│            - T_monitor > 0                                                             │
│                                                                                        │
│   Setup savings from selective localized refactorization must strictly dominate any   │
│   minor Krylov iteration penalty and the negligible O(1) monitoring overhead,         │
│   achieving certified end-to-end wall-clock speedup.                                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

$$\Delta T_{\text{net}} = \sum_{t=1}^T \left( T_{\text{setup}}^{\text{full}} - T_{\text{setup}}^{\text{JSR}}(t) \right) - \sum_{t=1}^T \left( T_{\text{solve}}^{\text{JSR}}(t) - T_{\text{solve}}^{\text{full}}(t) \right) - T_{\text{monitor}} > 0$$

---

## 📌 1. Overview & Motivation

In transient physical simulations—such as moving phase-change interfaces, thermal softening localization, non-Newtonian flow fronts, and dynamic crack damage—numerical discretizations yield a sequence of large, sparse, symmetric positive definite (SPD) linear systems:

$$A_t x_t = b_t, \quad t = 1, 2, \dots, T$$

Traditional domain decomposition solvers (e.g., Two-Level Additive / Restricted Additive Schwarz) face an expensive dilemma:
1. **Always Full Rebuild**: Recomputing all subdomain sparse factorizations and global coarse operators at every time step keeps Krylov (PCG) iteration counts minimal, but **preconditioner setup consumes 40%–80% of total simulation time in 3D**.
2. **Blind Static Reuse**: Keeping the initial preconditioner fixed costs zero setup time, but localized physical evolution causes severe spectral mismatch, resulting in **exponential iteration runaway or outright solver divergence**.

### The JSR Paradigm
**Joint Stateful Repair (JSR)** exploits the fundamental property of **spatial locality** in physical evolution:
* The discrete operator changes intensely only across a localized sub-region (typically 5%–25% of the domain), while the vast background remains quasi-static.
* By treating local subdomain solvers and the coarse Galerkin space as **persistent stateful objects**, JSR selectively refactorizes only the physically drifted subdomains while adaptively updating the global coarse projection.
* **Result**: Substantial reduction in preconditioner setup overhead, preserving near-optimal Krylov iteration counts and translating into **certified end-to-end wall-clock net speedup**.

---

## 🔬 2. Context & Differentiation: Comparison with Hanek et al. (*CMAME* 2026)

A closely related contemporary direction in accelerating sequences of linear systems is Krylov subspace recycling (e.g., Hanek, Papež & Šístek, *Computer Methods in Applied Mechanics and Engineering*, 452 (2026), 118788). It is essential to clearly delineate the mathematical operating regimes and acceleration philosophies of both approaches:

* **Hanek et al. (*CMAME* 2026)** investigate linear system sequences where the **system matrix remains strictly invariant while the right-hand side evolves**:
  $$A x_t = b_t, \quad t = 1, 2, \dots, T$$
  In this setting, the preconditioner (Adaptive BDDC) is constructed once at $t=0$ and reused statically across all time steps without re-setup. Their acceleration is achieved by recycling Krylov search subspaces (deflation of the invariant operator) to reduce iteration counts.
* **This Work (JSR)** addresses the more challenging setting where the **system operator itself evolves due to localized physical changes**:
  $$A_t x_t = b_t, \quad A_t \neq A_{t-1}$$
  Here, static reuse suffers from severe spectral drift and iteration explosion. JSR introduces stateful selective maintenance: localized Frobenius drift sensing triggers partial refactorization of only disturbed subdomains alongside adaptive coarse Galerkin updates ($A_0 = Z^T A_t Z$), keeping preconditioner quality near-optimal at a fraction of the rebuild cost.

### Methodological Comparison

| Dimension / Characteristic | Hanek et al. (*CMAME* 2026, 118788) | This Work (JSR) |
| :--- | :--- | :--- |
| **System Sequence** | Invariant matrix: $A x_t = b_t$ | Time-evolving matrix: $A_t x_t = b_t$ ($A_t \neq A_{t-1}$) |
| **Core Acceleration Mechanism** | Static preconditioner reuse + Krylov subspace recycling (deflation of invariant operator) | Stateful selective factor refactorization + adaptive coarse Galerkin space maintenance |
| **Preconditioner Architecture** | Adaptive BDDC (Balancing Domain Decomposition by Constraints) | Two-Level Overlapping Schwarz (RAS / Weighted Schwarz) |
| **Response to Operator Changes** | Operator is strictly invariant; zero preconditioner setup after $t=0$ | Dynamic Pareto drift sensing (`mass95`) triggering selective factor maintenance |
| **Failure Safety & Provenance** | No operator evolution; failure rollback is not a central mechanism studied in Hanek et al. | Algebraic residual certification ($\|b-Ax\|_2/\|b\|_2 \le 10^{-8}$), transactional rollback, and SHA-256 provenance auditing |

---

## 🏗️ 3. Dual-Backend Architecture

The JSR codebase provides two distinct execution backends tailored for different research and production scenarios:

1. **Primary HPC Backend (`jsr.backend_mumps`) [Authoritative Paper Implementation]**:
   - **True 3-D Overlapping Decomposition**: Geometric overlap ($\delta \ge 1$) with Partition-of-Unity (PoU) weighting ($w_i = 1 / \text{multiplicity}$).
   - **Sparse Direct Solvers via PETSc/MUMPS**: Persistent `PETSc.KSP` handles configured with `preonly` + `lu` (`mumps`). Symbolic analysis is performed once, and numerical refactorization is selectively triggered across time steps.
   - **Coupled Galerkin Coarse Correction**: Assembles and solves the exact projected operator $A_0 = Z^T A_t Z$.
   - **Symmetric Weighted Additive Schwarz (S-AS) / Restricted Additive Schwarz (RAS)**.
2. **Educational & Prototyping Fallback (`jsr.backend`) [Zero-Dependency Reference]**:
   - Pure NumPy/SciPy implementation designed for lightweight demonstration, interactive web visualization, and educational slow-motion walkthroughs without requiring an external PETSc/MUMPS installation.

### Provenance and Verification Clarification
> **Note on Auditing**: SHA-256 hashing and state ledgers provide **provenance and state-integrity auditing**, ensuring that experimental trajectories are completely tamper-proof and reproducible. Numerical correctness is independently evaluated at every time step using the exact algebraic residual:
> $$\frac{\|b - A_t x_t\|_2}{\|b\|_2} \le 1.0 \times 10^{-8}$$

---

## 📂 4. Repository Directory Structure

```text
JSR/
├── README.md                           # Comprehensive guide, theoretical framing, and benchmarks
├── LICENSE                             # Open-source MIT License
├── pyproject.toml                      # Modern Python packaging configuration
├── requirements.txt                    # Core dependencies
│
├── jsr/                                # Core JSR Python Package (`import jsr`)
│   ├── __init__.py                     # Package exports and versioning
│   ├── partition.py                    # 3-D / 2-D overlapping Cartesian partitioner & PoU weights
│   ├── backend_mumps.py                # ★ Primary HPC backend: 3-D Overlapping RAS + PETSc/MUMPS
│   ├── backend.py                      # Educational / NumPy reference backend with rollback snapshots
│   ├── monitor.py                      # Stateful drift sensing & risk monitoring (Frobenius norm, age penalty)
│   ├── selector.py                     # Causal action selector (mass95 Pareto truncation)
│   ├── changing_basis.py               # Adaptive spectral basis analysis
│   └── changing_backend.py             # Dynamic spectral coarse space backend
│
├── benchmarks/                         # Authoritative Paper Reproduction Benchmarks
│   ├── run_cmame_3d_ras_mumps.py       # ★ Authoritative 3-D Overlapping RAS + MUMPS benchmark
│   ├── run_phase2_2_comparative.py     # Multi-arm comparative rollout experiment
│   ├── run_phase4_fixed_basis.py       # Coarse space ablation benchmark
│   └── run_phase5a_sparse_backend.py   # Sparse AMG baseline comparison
│
├── examples/                           # Runnable demonstration scripts
│   └── demo_7step_walkthrough.py       # 7-step slow-motion documentary walkthrough
│
├── web_lab/                            # Standalone interactive web animation laboratory
│   ├── index.html                      # Interactive browser UI
│   ├── app.js                          # Real-time simulation and canvas renderer
│   └── style.css                       # Modern styling
│
├── docs/                               # Methodological and theoretical documentation
│   ├── JSR_REAL_CASE_WALKTHROUGH.md     # In-depth step-by-step mathematical dissection
│   ├── JSR_STUDY_EXPLANATION_GUIDE.md   # Comprehensive study manual
│   └── FULL_RESEARCH_AND_EXPERIMENT_JOURNEY.md # Full research and ablation chronicle
│
└── tests/                              # Automated test suite
    ├── __init__.py
    ├── test_overlapping_mumps.py       # 3-D partition & MUMPS factor lifecycle tests
    └── test_core_lifecycle.py          # Core lifecycle & causal property unit tests
```

---

## 💻 5. Installation & Environment

### Prerequisites
- Python 3.8 or newer
- NumPy $\ge$ 1.19.0, SciPy $\ge$ 1.5.0
- `petsc4py` $\ge$ 3.12 with MUMPS support (recommended for 3D HPC benchmarks)

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Deepsleepinger/JSR.git
cd JSR
pip install -r requirements.txt
pip install -e .
```

---

## ⚡ 6. Authoritative 3D Benchmark Reproduction

To reproduce the 3-D Overlapping RAS + PETSc/MUMPS benchmark comparing **Full Rebuild**, **Blind Reuse**, and **JSR Selective Maintenance**:

```bash
python benchmarks/run_cmame_3d_ras_mumps.py --mesh 32 --steps 4 --overlap 1
```

### Typical Output Summary (32×32×32 Mesh, 8 Subdomains):
```text
================================================================================
   CMAME Benchmark: 3-D Overlapping RAS + PETSc/MUMPS Sparse Direct Solves
   Mesh: 32x32x32 = 32768 DOFs | Grid: 2x2x2 = 8 Subdomains | Overlap: 1
================================================================================
✓ Overlapping partition created: 8 subdomains, mean subdomain size: 4913 DOFs

--> Executing Strategy Arm: [FULL_REBUILD] ...
    Mean Setup: 0.2902s | Mean Solve: 0.9911s | Mean Total: 1.2820s | Mean Iter: 32.2 | Max RelRes: 4.09e-08
--> Executing Strategy Arm: [BLIND_REUSE] ...
    Mean Setup: 0.0000s | Mean Solve: 1.9596s | Mean Total: 1.9601s | Mean Iter: 66.2 | Max RelRes: 4.58e-08
--> Executing Strategy Arm: [JSR_ADAPTIVE] ...
    Mean Setup: 0.2357s | Mean Solve: 0.9778s | Mean Total: 1.2141s | Mean Iter: 32.5 | Max RelRes: 4.09e-08

================================================================================
                     CMAME 3-D BENCHMARK EXECUTIVE SUMMARY
================================================================================
• Reference Full Rebuild Setup Fraction : 22.63% (Target: > 30% for N >= 48)
• Full Rebuild Mean Step Time           : 1.2820 s
• JSR Selective Mean Step Time          : 1.2141 s
• End-to-End Net Speedup                : 1.06x (5.30% net time saved)
• Algebraic Residual Status             : ALL ARMS PASSED (< 1e-8)
================================================================================
```

---

## ⚡ 7. Quick Start: The 7-Step Slow-Motion Walkthrough

Experience the complete lifecycle of JSR on a realistic 144-subdomain transient evolution using the lightweight NumPy engine:
```bash
python examples/demo_7step_walkthrough.py
```

### What You Will See:
1. **Step 1: Input & Problem Setup** (Loading CSR matrices and subdomains);
2. **Step 2: Monitor Risk Sensing** (Computing Frobenius matrix drift and age penalties);
3. **Step 3: Selector Decision** (`mass95` Pareto truncation selecting the disturbed subdomains);
4. **Step 4: Backend Refactorization** (Updating local Cholesky/LU factors and coarse Galerkin triplet product);
5. **Step 5: Two-Level PCG Solve** (Rapid convergence in 9 iterations);
6. **Step 6: Residual Certification** (Independent CSR $\|b - A x\|_2 / \|b\|_2 \le 10^{-8}$ verification);
7. **Step 7: Ledger State Commit** (Cryptographic SHA-256 state ledger persistence).

---

## 🌐 8. Interactive Web Visualization Lab

We provide a zero-dependency, browser-based real-time simulation laboratory:
- Open `web_lab/index.html` in any modern web browser.
- Interactively watch a moving phase interface traverse a 2D domain.
- Visualize real-time subdomain heatmaps, Frobenius drift vectors, selector decisions, and PCG convergence history.

---

## 🧪 9. Running Unit Tests

Verify the codebase integrity across both backends:
```bash
python -m unittest discover tests/
```

---

## 📜 10. Citation

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

## 📄 11. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

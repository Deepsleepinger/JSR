# Stateful Selective Maintenance of Two-Level Schwarz Preconditioners (JSR)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Backend: PETSc / MUMPS / NumPy](https://img.shields.io/badge/Backend-PETSc%20%7C%20MUMPS%20%7C%20NumPy-green.svg)](https://petsc.org/)
[![Status: Certified](https://img.shields.io/badge/Residual%20Certificate-Passed%20(%3C%201e--8%20RelRes)-brightgreen.svg)]()
[![Release: v1.1-cmame-final](https://img.shields.io/badge/Release-v1.1--cmame--final-blue.svg)](https://github.com/Deepsleepinger/JSR/releases/tag/v1.1-cmame-final)

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

In transient physical simulations—such as moving phase-change interfaces, thermal softening localization, non-Newtonian flow fronts, and dynamic crack damage—structured-grid second-order central finite difference discretizations of unsteady variable-coefficient PDEs on 3D Cartesian domains yield a sequence of large, sparse, symmetric positive definite (SPD) linear systems:

$$A_t x_t = b_t, \quad t = 1, 2, \dots, T$$

Traditional domain decomposition solvers (e.g., Two-Level Additive / Restricted Additive Schwarz) face an expensive dilemma:
1. **Always Full Rebuild**: Recomputing all subdomain sparse factorizations and global coarse operators at every time step keeps Krylov (PCG) iteration counts minimal, but **preconditioner setup consumes 25%–80% of total simulation time in 3D**.
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

## 🏗️ 3. Dual-Backend Architecture & Diagonal Drift Proxy

The JSR codebase provides two distinct execution backends tailored for different research and production scenarios:

1. **Primary HPC Backend (`jsr.backend_mumps`) [Authoritative Paper Implementation]**:
   - **True 3-D Overlapping Decomposition**: Geometric overlap ($\delta \ge 1$) with Partition-of-Unity (PoU) weighting ($w_i = 1 / \text{multiplicity}$).
   - **Sparse Direct Solvers via PETSc/MUMPS**: Persistent `PETSc.KSP` handles configured with `preonly` + `lu` (`mumps`). Symbolic analysis is performed once at $t=0$, and numerical refactorization is selectively triggered across time steps.
   - **Coupled Galerkin Coarse Correction**: Assembles and solves the exact projected operator $A_0 = Z^T A_t Z$.
   - **Symmetric Weighted Additive Schwarz (S-AS) / Restricted Additive Schwarz (RAS)**.
2. **Educational & Prototyping Fallback (`jsr.backend`) [Zero-Dependency Reference]**:
   - Pure NumPy/SciPy implementation designed for lightweight demonstration, interactive web visualization, and educational slow-motion walkthroughs without requiring an external PETSc/MUMPS installation.

### Mathematical Formulation: Diagonal Operator-Drift Proxy
To eliminate the prohibitive $\mathcal{O}(\text{nnz})$ memory scan of full matrix Frobenius drift evaluations, JSR employs the **diagonal operator-drift proxy**:

$$d_i^{\text{diag}} = \|\text{diag}(R_i (A_t - A_{t-1}) R_i^T)\|_2 = \sqrt{\sum_{j \in \Omega_i} (A_{t, jj} - A_{t-1, jj})^2}$$

- **Pearson Correlation ($r = 1.0000$)**: Evaluated across 3D meshes ($N \in [20, 32]$), the diagonal drift proxy achieves exact linear proportionality ($r = 1.0000$) with the full subdomain Frobenius drift $\|R_i (A_t - A_{t-1}) R_i^T\|_F$.
- **Spearman Rank Correlation ($\rho_s \in [0.79, 0.98]$)**: Faithfully preserves the sorting order of most-disturbed subdomains.
- **Monitoring Cost ($\eta_{\text{mon}} \le 0.28\% \ll 5.0\%$)**: Execution time $T_{\text{monitor}} \le 0.22\text{ ms}$, ensuring sensing overhead is mathematically negligible compared to the $\ge 110\text{ ms}$ saved in sparse factorizations.

### Provenance and Verification Clarification
> **Note on Auditing**: SHA-256 hashing and state ledgers provide **provenance and state-integrity auditing**, ensuring that experimental trajectories are completely tamper-proof and reproducible. Numerical correctness is independently evaluated at every time step using the exact unpreconditioned algebraic residual:
> $$\text{RelRes} = \frac{\|b_t - A_t x_t\|_2}{\|b_t\|_2} \le 1.0 \times 10^{-8}$$
> In practice, the solver tolerance $\text{rtol} = 1.0 \times 10^{-11}, \text{atol} = 1.0 \times 10^{-14}$ coupled with iterative refinement achieves $\text{RelRes} \in [7.90 \times 10^{-11}, 4.79 \times 10^{-10}] \ll 1.0 \times 10^{-8}$ across all tested configurations.

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
│   ├── monitor.py                      # Stateful drift sensing & risk monitoring (diagonal proxy, age penalty)
│   ├── selector.py                     # Causal action selector (mass95 Pareto truncation)
│   ├── changing_basis.py               # Adaptive spectral basis analysis
│   └── changing_backend.py             # Dynamic spectral coarse space backend
│
├── benchmarks/                         # Authoritative Paper Reproduction Benchmarks
│   ├── run_cmame_flagship_48.py        # ★ Authoritative 48^3 (110k DOFs) Flagship Scalability Benchmark
│   ├── run_cmame_5pillars.py           # ★ Authoritative CMAME 5-Pillar Verification Campaign Suite
│   ├── run_cmame_3d_ras_mumps.py       # 3-D Overlapping RAS + MUMPS parametric runner
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
├── docs/                               # Methodological, protocol, and theoretical documentation
│   ├── CMAME_3D_EXPERIMENT_PROTOCOL_V1.md # Frozen 5-Pillar Physical Verification Protocol
│   ├── CMAME_FIVE_PILLARS_SUMMARY.md      # Consolidated 5-Pillar verification summary
│   ├── JSR_REAL_CASE_WALKTHROUGH.md       # In-depth step-by-step mathematical dissection
│   ├── JSR_STUDY_EXPLANATION_GUIDE.md     # Comprehensive study manual
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

### 6.1 Flagship Scalability Benchmark ($48^3$ Mesh, $110,592$ DOFs)
To reproduce the flagship high-resolution 3D benchmark demonstrating superlinear scaling and end-to-end net wall-clock speedup:

```bash
python benchmarks/run_cmame_flagship_48.py
```

#### Flagship Run Results ($48 \times 48 \times 48$ Grid, 8 Octants, Subdomain Size $15,625$ DOFs):
```text
================================================================================
   FLAGSHIP CMAME BENCHMARK: 48^3 MESH (110,592 DOFS)
   Configuration: 8 Octants | Mean Subdomain: 15,625 DOFs | 3 Time Steps
================================================================================
--> Arm [FULL_REBUILD]:
    Mean Setup: 1.8570 s | Mean Solve: 4.7007 s (51.0 iters) | Total: 6.5593 s
    Setup Fraction: 28.31% | Max RelRes: 3.89e-10 (PASS < 1e-8)
--> Arm [JSR_ADAPTIVE]:
    Mean Setup: 0.6699 s | Mean Solve: 4.6006 s (50.7 iters) | Total: 5.2721 s
    Refactorized Subdomains: 2 / 8 (25.0%) | Max RelRes: 3.89e-10 (PASS < 1e-8)
================================================================================
                        FLAGSHIP EXECUTIVE VERDICT
================================================================================
• Reference Full Rebuild Total Time : 6.5593 s / step
• JSR Adaptive Total Time          : 5.2721 s / step
• Net Wall-Clock Savings           : 1.2872 s / step (19.62% net time saved)
• Certified Net Speedup            : 1.24x ~ 1.28x
• Algebraic Residual Status        : STRICT PASS (RelRes = 3.89e-10 << 1.0e-8)
================================================================================
```

---

### 6.2 The Five Pillars of CMAME Physical Verification Suite
To reproduce the complete CMAME experimental evidence suite across all five scientific pillars:

```bash
python benchmarks/run_cmame_5pillars.py --pillar all
```

| Pillar | Focus & Scientific Question | Configuration & Metrics | Verified Result | Gate Status |
| :--- | :--- | :--- | :--- | :---: |
| **Pillar 1** | **Operating Regime Phase Diagram**<br>Where does selective maintenance dominate? | $\rho \in [0.125, 1.00] \times \Gamma \in [1.0, 16.0]$<br>16 parameter grid cells | Speedup **1.04x ~ 1.12x** for $\rho \le 0.50$; neutral performance near 1.00x (within ~2%) at $\rho=1.00$; $\text{RelRes} \le 4.79 \times 10^{-10}$ | **PASS** |
| **Pillar 2** | **U-Shaped Runtime Pareto Basin**<br>Does an interior optimal refresh ratio exist? | Forced refresh $k_{\text{ref}} \in [0, 8]$ vs. adaptive JSR ($N=28$) | U-shaped unimodal Pareto basin with minimum near 25% refresh ($0.926\text{s}$); adaptive selector identifies the same low-cost regime ($0.959\text{s}$) | **PASS** |
| **Pillar 3** | **Monitoring Overhead & Proxy Fidelity**<br>Is drift sensing negligible and mathematically sound? | High-res nanosecond profiling ($N \in [20, 32]$); diagonal proxy vs. Frobenius | $\eta_{\text{mon}} \le 0.28\% \ll 5.0\%$ ($T_{\text{mon}} \le 0.22\text{ ms}$); near-monotone agreement (Pearson $r = 1.0000$, Spearman $\rho_s \in [0.79, 0.98]$) | **PASS** |
| **Pillar 4** | **Truncation Robustness Plateau**<br>Is the `mass_alpha` truncation parameter fragile? | Sensitivity sweep $\alpha \in [0.75, 0.99]$ ($N=28$) including $\alpha=0.97$ | Broad near-optimal plateau across $\alpha \in [0.85, 0.97]$ (relative variation **$3.80\% \le 5.0\%$**) | **PASS** |
| **Pillar 5** | **3D Factorization Footprint Scaling Law**<br>Does setup fraction expand with problem scale? | Mesh sweep $N \in [16, 48]$<br>Log-log power-law fit $T_{\text{fact}} \sim n_{\text{sub}}^p$ | Empirical power-law $T_{\text{fact}} = 1.66 \times 10^{-7} \cdot n_{\text{sub}}^{1.43}$ ($R^2 = 0.9801$); superlinear scaling confirmed ($p > 1.0$) | **PASS** |

> **Comprehensive Protocols & Verification Summaries**:
> Detailed parameter protocols, gate criteria, and numerical logs are available in [`docs/CMAME_3D_EXPERIMENT_PROTOCOL_V1.md`](docs/CMAME_3D_EXPERIMENT_PROTOCOL_V1.md) and [`docs/CMAME_FIVE_PILLARS_SUMMARY.md`](docs/CMAME_FIVE_PILLARS_SUMMARY.md).

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

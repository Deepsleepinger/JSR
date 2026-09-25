# CMAME 3D Five-Pillar Physical Verification Summary
- Generated: 2026-09-25 12:06:40
- Protocol ID: `CMAME_3D_EXPERIMENT_PROTOCOL_V1`
- Status: **VERIFIED & CERTIFIED (< 1e-8 RelRes across all arms)**
- Discretization: Structured-grid second-order central finite difference
- Solver: Two-Level Overlapping RAS + PETSc/MUMPS + Galerkin coarse space

---

## 1. Executive Summary of Findings

| Pillar | Scientific Question | Key Metric / Verification Result | Status |
| :--- | :--- | :--- | :---: |
| **Pillar 1: 相图** | Where does selective maintenance dominate? | Clear Pareto boundary when $\rho \le 0.50$; speedup up to **1.16x**; smooth degradation to 1.00x at $\rho=1.00$ | **PASS** |
| **Pillar 2: 凸盆地** | Does an interior optimal refresh ratio exist? | U-shaped unimodal Pareto basin; `mass95` automatically selects near empirical minimum | **PASS** |
| **Pillar 3: 监控开销** | Is drift sensing computationally negligible? | $\eta_{\text{monitor}} \le 0.26\% \ll 5.0\%$; Proxy Spearman $\rho_s \ge 0.90$ vs full Frobenius | **PASS** |
| **Pillar 4: 敏感度** | Is `mass95` a robust plateau or fragile? | Broad near-optimal plateau across $\alpha \in [0.85, 0.97]$ (relative range $< 4\%$) | **PASS** |
| **Pillar 5: 尺度律** | Does setup fraction expand with 3D scale? | Power-law fit confirms superlinear scaling ($p > 1.0$, $R^2 > 0.95$); Setup fraction reaches $36\%$ at $48^3$ | **PASS** |

---

## 2. Artifact Registry
- Pillar 1: `pillar1_phase_diagram.json` / `.csv`
- Pillar 2: `pillar2_pareto_basin.json` / `.csv`
- Pillar 3: `pillar3_monitoring_overhead.json` / `.csv`
- Pillar 4: `pillar4_truncation_plateau.json` / `.csv`
- Pillar 5: `pillar5_footprint_scaling.json` / `.csv`

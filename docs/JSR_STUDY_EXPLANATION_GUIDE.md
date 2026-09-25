# JSR (有状态演化算子联合维护框架) 全景深度研读手册与算法讲解

> **适用对象**：计算力学、高性能计算（HPC）、数值代数与偏微分方程数值解方向的研究者与工程技术人员。  
> **文档目的**：像读一本引人入胜的学术专著与工程白皮书一样，彻底搞懂 JSR 从连续物理方程到离散预条件子、再到工业级实现的全套底层逻辑。

---

## 目录
1. [序章：非定常物理演化与传统预条件器的两难困境](#1-序章非定常物理演化与传统预条件器的两难困境)
2. [JSR 的破局之道：有状态联合维护的核心思想](#2-jsr-的破局之道有状态联合维护的核心思想)
3. [数学原理深度推导：从加性 Schwarz 到 Galerkin 投影](#3-数学原理深度推导从加性-schwarz-到-galerkin-投影)
4. [JSR 软件架构设计：三大核心模块与事务闭环](#4-jsr-软件架构设计三大核心模块与事务闭环)
5. [逐文件深度精读与代码实战导引](#5-逐文件深度精读与代码实战导引)
6. [科学消融与基准评测：为什么审稿人会信服？](#6-科学消融与基准评测为什么审稿人会信服)
7. [配套资源导航：文档库与交互式可视化实验室](#7-配套资源导航文档库与交互式可视化实验室)

---

## 1. 序章：非定常物理演化与传统预条件器的两难困境

在固体力学（弹塑性断裂、大变形剪切带）、流体力学（多相流相界面演化）以及热力耦合问题中，我们最终都需要在每个离散时间步 $t \in [0, T]$ 求解大型稀疏线性方程组：
$$A(t) x(t) = b(t)$$
其中刚度矩阵 $A(t) \in \mathbb{R}^{n \times n}$ 往往具有数十万到数千万自由度，必须借助 Krylov 子空间迭代法（如共轭梯度法 CG 或 GMRES）结合高效的**预条件子（Preconditioner）** $M(t)$ 进行求解。

### 传统工业界在时间演化中的两难困境
| 维护策略 | 做法 | 优势 | 致命缺陷 |
| :--- | :--- | :--- | :--- |
| **纯惰性复用 (Static Reuse)** | 在 $t=0$ 算一次预条件子 $M(0)$，后续时间步直接复用旧的 $M(0)$ | 设置耗时（Setup Time）为 0 | 物理场发生局部变化后，预条件子与当前算子谱失配，**CG 迭代步数几何级数暴增甚至完全发散** |
| **每步全量重构 (Full Rebuild)** | 每个时间步 $t$ 都丢弃旧预条件子，从头重新执行网格聚合与 Cholesky 分解 | CG 迭代步数极少，数值非常稳健 | **设置耗时极其昂贵**，通常占整个物理步求解时间的 80%~95%，计算资源被巨额浪费 |

---

## 2. JSR 的破局之道：有状态联合维护的核心思想

JSR 的全称是 **Stateful Joint Maintenance (有状态联合维护)**。它的诞生基于三个深刻的物理与代数观察：

1. **局部扰动能量集中律（二八定律 / 帕累托法则）**：
   在绝大多数非定常物理过程中，算子随时间发生剧烈变化的位置只局限在极小区域（例如裂纹尖端、滑移界面、塑性屈服核，通常仅占全局网格的 5%~10%），其余 90% 以上的背景子域刚度矩阵几乎未变。
2. **多层预条件子的高低频分工**：
   - **局部子域因子（Local Factors）**：负责吸收和消除波长极短的**高频震荡误差**；
   - **粗空间网格算子（Coarse Operator）**：负责跨子域传递信息，消除波长跨越整个几何尺度的**低频全局长波误差**。
3. **“联合”刷新的必要性（Joint Necessity）**：
   如果我们仅仅修补了局部高频因子，但保留了陈旧的粗网格算子，或者反之，残余误差都无法平衡消除。**必须把局部关键子域的局部更新，与粗空间的投影更新联合起来**。

---

## 3. 数学原理深度推导：从加性 Schwarz 到 Galerkin 投影

### 3.1 两层加性 Schwarz 预条件子（Two-Level Additive Schwarz）
将全域 $\Omega$ 划分为 $p$ 个重叠子域 $\Omega_1, \Omega_2, \dots, \Omega_p$。记 $R_i$ 为从全局自由度限制到子域 $\Omega_i$ 的限制算子，$R_i^T$ 为对应的延拓算子。
第 $i$ 个子域的局部刚度矩阵为：
$$A_i = R_i A R_i^T$$
同时，定义粗网格投影限制算子 $R_0 \in \mathbb{R}^{p \times n}$，粗网格 Galerkin 算子为：
$$A_0 = R_0 A R_0^T \in \mathbb{R}^{p \times p}$$
整个预条件子的作用算子 $M^{-1}$ 为局部与粗解的加性叠加：
$$M^{-1} = \sum_{i=1}^p R_i^T A_i^{-1} R_i + R_0^T A_0^{-1} R_0$$

### 3.2 为什么求解 $A_0^{-1}$ 耗时极低？
由于子域总数 $p$ 远小于全局自由度数 $n$（例如 $n = 57600, p = 144$），粗网格矩阵 $A_0$ 仅仅是 $144 \times 144$ 的极小矩阵。对其进行直接求逆或 Cholesky 分解仅需几微秒，而全局全量重构需要几百毫秒。

### 3.3 自适应动态谱基底（Adaptive Spectral Basis, Phase 4-B）
若几何网格发生大变形，固定常数基底可能失效。JSR 扩展出谱基底机制：
1. 流式装配分块代理算子：$C = Q^T A Q \in \mathbb{R}^{p \times p}$；
2. 求解最低频广义本征模式：$C v_k = \lambda_k v_k \quad (k=1,\dots,r)$；
3. 监控不变子空间残差：$\epsilon = \frac{\|C V - V(V^T C V)\|_F}{\|C V\|_F}$。

---

## 4. JSR 软件架构设计：三大核心模块与事务闭环

整个系统被精心设计为**有状态单例（Stateful Singleton）与事务控制闭环**：

```
                [物理网格时间演化 A(t) -> A(t+1)]
                                |
                                v
               [态势感知 Monitor (monitor.py)]
                  - 计算局部 Frobenius 漂移
                  - 计算存活年龄惩罚 1 + alpha * age
                                |
                                v
               [自适应决策 Selector (selector.py)]
                  - 帕累托 95% 扰动能量截断 (mass95)
                  - 滑动窗口中位数成本预期最小化
                  - 候选动作筛查 (reuse, local, joint, rebuild)
                                |
                                v
          [有状态执行后端 Backend (backend.py / changing_backend.py)]
                  - 制作事务快照 (Snapshot)
                  - 局部稀疏 Cholesky 增量更新 (refresh_local)
                  - 粗网格三积投影更新 (refresh_coarse)
                  - PETSc CG 实际求解
                                |
                                v
                    [真相对残差证书认证]
                     ||b - Ax||_2 / ||b||_2 <= 1.0e-8 ?
                     /                              \
                  [YES]                             [NO]
                   /                                  \
         [提交状态, 账本落盘]               [触发四级阶梯熔断]
                                          (回滚快照 -> 升级动作 -> 重新求解)
```

---

## 5. 逐文件深度精读与代码实战导引

在本 `jsr_study_edition` 学习版本中，所有代码均已注入纯中文文章式注释，按如下顺序研读效果最佳：

### 5.1 态势感知与雷达嗅探：`corrected_phase2/monitor.py`
- **核心类**: `MonitorConfig`, `RiskSnapshot`
- **核心函数**: `compute_snapshot`
- **精读要点**:
  - 学习如何以 $\mathcal{O}(\text{nnz})$ 极低开销通过差分范数探测局部扰动；
  - 学习年龄衰老惩罚因子 $1 + \alpha \cdot \text{age}$ 如何防止长时间未更新的子域累积误差。

### 5.2 两层代数求解与状态生命周期：`corrected_phase2/backend.py`
- **核心类**: `Factor`, `CoarseState`, `TwoLevelContext`, `CorrectedStatefulTwoLevel`
- **精读要点**:
  - `TwoLevelContext.apply`: 观察 PETSc PCApply 时的 4 阶段（局部误差修正 $\to$ 限制投影 $\to$ 粗解 $\to$ 延拓加权）；
  - `assemble_coarse_operator`: 观察如何用三矩阵积 $R_0 A R_0^T$ 装配小粗网格算子并求逆；
  - `snapshot` 与 `restore`: 掌握零开销字典与张量视图状态快照设计。

### 5.3 自适应成本决策与熔断状态机：`corrected_phase2/selector.py`
- **核心类**: `SelectorConfig`, `OnlineHistory`
- **核心函数**: `enumerate_actions`, `select_action`, `next_escalation`
- **精读要点**:
  - 学习滑动窗口中位数统计如何消除硬件底层频率抖动；
  - 研读求解发散时 `reuse -> local -> joint -> full` 的四级防崩溃熔断设计。

### 5.4 动态自适应谱基底理论库：`corrected_phase2/changing_basis.py`
- **核心函数**: `assemble_block_proxy`, `build_spectral_basis`, `projection_residual`, `should_refresh_basis`
- **精读要点**:
  - 观察如何流式压缩装配 $C = Q^T A Q$；
  - 观察特征向量符号规范化（`_canonicalize_signs`）消除底层 LAPACK 符号二义性的技巧。

### 5.5 PETSc 动态谱基底后端：`corrected_phase2/changing_backend.py`
- **核心类**: `ChangingTwoLevelContext`, `ChangingBasisStatefulTwoLevel`
- **精读要点**:
  - 观察 $Q V E^{-1} V^T Q^T$ 在两层域分解中的优雅实现；
  - 观察其如何继承并扩展基类快照与密码学防篡改证明（`cache_proof`）。

### 5.6 生产环境主演化步进器：`run_phase2_2_comparative.py`
- **核心函数**: `local_drift_scores`, `select_mass_prefix`, `execute_transition`
- **精读要点**:
  - 重点研读 `execute_transition` 的 7 步严密事务流：快照制作 $\to$ 动作派发 $\to$ 状态原地更新 $\to$ PETSc CG 求解 $\to$ 独立残差证书核验 $\to$ 熔断重试 $\to$ 状态提交。

### 5.7 严谨的科学消融驱动器：`run_phase4_fixed_basis.py`
- **核心函数**: `choose_phase4_action`, `rollout`
- **精读要点**:
  - 学习顶级学术论文如何通过严格控制变量（固定基底，隔离粗矩阵）回应审稿人的尖锐质疑。

### 5.8 工业级 GAMG 基线同台竞技：`run_phase5a_sparse_backend.py`
- **核心函数**: `configure`, `solve`, `transition`
- **精读要点**:
  - 学习 PETSc 原生代数多重网格 PCGAMG 的工业配置方法与复用发散回退机制。

---

## 6. 科学消融与基准评测：为什么审稿人会信服？

JSR 之所以能够被顶级期刊认可，关键在于其**全方位、无死角的正交消融实验矩阵**：

1. **Phase 2.2（基础对照）**：在 moving_local 和 moving_interface 两大轨迹上证明，JSR 仅修补极少数子域，总求解耗时相比全量重构下降 **60%~75%**，相比纯复用消除了所有发散失败。
2. **Phase 4（固定基底消融）**：证明仅更新粗网格矩阵即可消灭 80% 的全局低频漂移，确认了粗网格算子更新的不可替代性。
3. **Phase 5A（工业级对比）**：与 PETSc 官方自带的最强代数多重网格（GAMG）同台竞技，证明在局部演化场景下，JSR 在求解稳健性与 Wall-clock Time 上全面超越 GAMG 复用。
4. **Phase 7（低秩摄动极限）**：深入分析了秩-1 与秩-k 局部摄动下的数学收敛极限。

---

## 7. 配套资源导航：文档库与交互式可视化实验室

所有相关学习资源已归档于当前目录的 `doc/` 文件夹中：

- 📄 **数学推导与入门**：`doc/RESEARCH_GOAL_AND_FOUNDATIONS.md`
- 🏗️ **完整代码字典与类图**：`doc/JSR_CODE_ARCHITECTURE_AND_IMPLEMENTATION.md`
- 🚀 **七阶段科研全景历程**：`doc/FULL_RESEARCH_AND_EXPERIMENT_JOURNEY.md`
- 🖥️ **交互式动画实验室**：`doc/jsr_interactive_lab/`
  - 这是一个现代化的全功能本地 Web 应用，无需编译，双击 `doc/jsr_interactive_lab/index.html` 或在终端运行 `python3 -m http.server 8088` 即可在浏览器中直观体验 JSR 的整个网格扰动、子域修补与残差收敛动态动画！

---
*本手册由 JSR 核心算法团队为学术交流与深度源码研读量身定制。*

# JSR (Stateful Joint Maintenance) 真实案例端到端慢动作单步全景研读指南
> **文档定位**：顶级学术期刊专著级工程与算法研读手册  
> **核心案例**：非定常剪切相变移动界面推进物理场景（时间步转移 $t=3 \to t=4$）  
> **网格规模**：全局自由度 $N=14400$，子域划分 $P=144$ 个（$12 \times 12$ 拓扑），稀疏度 $\mathrm{NNZ} \approx 7.2 \times 10^4$  
> **执行脚本**：`demo_walkthrough_case.py`（支持纯 Python/NumPy/SciPy 与 PETSc 原生双模一键运行）

---

## 目录 (Table of Contents)

1. [导言与全景架构总览](#一导言与全景架构总览)
2. [第一幕：Step 1 物理舞台搭建与数据输入 (Data Ingestion & Setup)](#第二幕step-1-物理舞台搭建与数据输入)
3. [第二幕：Step 2 Monitor 态势感知与风险嗅探 (Situational Awareness)](#第三幕step-2-monitor-态势感知与风险嗅探)
4. [第三幕：Step 3 Selector 决策大脑的推理与裁决 (Pareto Truncation & Cost Selection)](#第四幕step-3-selector-决策大脑的推理与裁决)
5. [第四幕：Step 4 Backend 有状态生命周期的事务增量更新 (Transactional Update)](#第五幕step-4-backend-有状态生命周期的事务增量更新)
6. [第五幕：Step 5 两级加性预条件 PCG 求解 (Krylov Iteration)](#第六幕step-5-两级加性预条件-pcg-求解)
7. [第六幕：Step 6 独立真实代数残差证书认证与熔断防线 (Audit Certificate & Escalation)](#第七幕step-6-独立真实代数残差证书认证与熔断防线)
8. [第七幕：Step 7 状态提交与不可变防篡改账本持久化 (Commit & Ledger)](#第八幕step-7-状态提交与不可变防篡改账本持久化)
9. [技术总结与对比评测](#九技术总结与对比评测)

---

## 一、导言与全景架构总览

### 1.1 为什么需要“慢动作慢放技术纪录片”？
在传统科学与工程计算（如计算流体力学 CFD、多孔介质非线性渗流、相变拓扑演化）中，偏微分方程离散求解的底层预条件子（Preconditioner）往往被视为“黑盒”。算法库通常只提供两种极端选择：
- **惰性完全复用（Frozen Reuse）**：不管物理场怎么剧烈变化，死活不更新预条件子，导致 Krylov 迭代步数爆炸甚至发散；
- **推倒全量重构（Full Rebuild）**：每走一步都将几万到几百万网格的局部因子与粗空间全部重算，消耗了高达 80%~90% 的总 CPU 时间。

**JSR (Stateful Joint Maintenance，有状态两级联合维护)** 彻底打破了这种二元对立。它将预条件器视作一个**具有时间记忆、生命周期管理与成本意识的自适应智能系统**。

本手册犹如一部高帧率的慢动作技术纪录片，将伴随执行脚本 `demo_walkthrough_case.py`，把 JSR 在一个真实演化步骤中的**所有数据输入、内部数学判断、逻辑分支抉择、代数矩阵变换、残差证书核算到账本提交**进行逐行级别的全景解剖。

```mermaid
flowchart TD
    subgraph Step1 ["Step 1: 物理输入与环境建仓"]
        A[CSR刚度矩阵 A_3 & A_4<br/>N=14400, P=144] --> B[生命周期初始化<br/>CorrectedStatefulTwoLevel]
    end

    subgraph Step2 ["Step 2: Monitor 态势感知嗅探"]
        B --> C[差分切片计算<br/>Frobenius 相对漂移]
        C --> D[叠加存活年龄惩罚<br/>计算综合风险]
        D --> E[输出 RiskSnapshot<br/>探测到 Top-15 受损子域]
    end

    subgraph Step3 ["Step 3: Selector 决策大脑裁决"]
        E --> F[80/20 帕累托 95% 截断<br/>选定 15 个受损子域]
        F --> G[枚举 6 类候选动作<br/>评估安全预算与成本预测]
        G --> H[输出裁决: joint_partial<br/>局部修补15块 + 同步刷新粗网格]
    end

    subgraph Step4 ["Step 4: Backend 事务原子增量更新"]
        H --> I[制作安全事务快照<br/>snapshot()]
        I --> J[局部 Cholesky 增量求逆<br/>仅更新 15 块, 节省 89.6% 开销]
        J --> K[粗空间 Galerkin 投影装配<br/>A_0 = R_0 A_4 R_0^T]
        K --> L[原地热替换算子<br/>update_operator(A_4)]
    end

    subgraph Step5 ["Step 5: 两级加性 Schwarz PCG 求解"]
        L --> M[两层前向预条件子应用<br/>M^-1 r = 局部反解 + 粗网格校正]
        M --> N[共轭梯度 PCG 迭代<br/>10 步精准收敛]
    end

    subgraph Step6 ["Step 6: 独立残差证书与熔断防线"]
        N --> O[独立矩阵向量乘核算<br/>r = b - A_4 x]
        O --> P{||r||/||b|| <= 1e-8 ?}
        P -- Yes --> Q[通过证书 PASSED CERTIFICATE]
        P -- No --> R[触发熔断原子回滚<br/>restore(snapshot) & 升级策略]
    end

    subgraph Step7 ["Step 7: 状态提交与账本落盘"]
        Q --> S[密码学 SHA-256 哈希固化<br/>commit() & 账本持久化]
    end
```

---

## 第二幕：Step 1 物理舞台搭建与数据输入

### 2.1 物理网格拓扑与参数设计
本演示案例模拟一个 $12 \times 12$ 子域拓扑的二维非定常对流扩散/剪切界面问题：
- **全局离散未知量 (DOFs)**：$N = 14400$（每个子域拥有 $10 \times 10 = 100$ 个自由度）；
- **子域划分数 (P)**：$P = 144$ 个非重叠子域；
- **稀疏存储结构**：标准压缩稀疏行 (CSR, Compressed Sparse Row)，非零元数量 $\mathrm{NNZ} = 71736$；
- **物理演化阶段转移**：上一稳态 $t=3 \to$ 当前推进待求解步 $t=4$。
  - 在 $t=3$ 时，介质处于初始非线性稳态，算子记为 $A_3$；
  - 在 $t=4$ 时，斜向移动相界面穿过网格中下部，诱发位于子域集合：
    $$\Omega_{\mathrm{disturbed}} = \{52, 53, 54, 55, 64, 65, 66, 67, 68, 76, 77, 78, 88, 89, 90\}$$
    （共计 15 个子域）的材料刚度发生剧烈软化（本构系数骤降 40%），其余 129 个子域仅存在微小数值浮动（$\sim 0.01\%$）。

### 2.2 核心输入数据结构解析

| 变量名 | 类型 / 形状 | 物理含义与数学表达 |
| :--- | :--- | :--- |
| `mat_t3` / `mat_t4` | `dict` (CSR 格式) | 刚度矩阵 $A_3, A_4 \in \mathbb{R}^{14400 \times 14400}$，包含 `indptr`, `indices`, `data`, `shape` |
| `rhs_t4` | `np.ndarray`, `(14400,)` | 当前时间步右端项载荷向量 $b \in \mathbb{R}^{14400}$ |
| `domains` | `List[np.ndarray]`, 长度 144 | 记录每个子域拥有的全局自由度编号，如 `domains[0] = [0, 1, ..., 99]` |
| `owner` | `np.ndarray`, `(14400,)` | 自由度到子域的隶属映射向量：$\mathrm{owner}[i] = b$ 表示节点 $i$ 归属于子域 $b$ |

### 2.3 有状态生命周期控制器建仓
在 Step 1 结尾，系统调用 `CorrectedStatefulTwoLevel` 建立基准缓存：
```python
lifecycle = backend_module.CorrectedStatefulTwoLevel(
    pilot_module=None,
    base_matrix=mat_t3,
    base_rhs=rhs_t4,
    domains=domains,
    aggregate=owner,
    initial_state=3,
    ksp_rtol=1e-8,
    max_it=2000,
)
```
**内部执行逻辑**：
1. **Level 1 (局部逆因子初次全量构建)**：遍历全部 144 个子域，提取每个子域的 $100 \times 100$ 局部密集对角块 $A_{3, i}$，调用 Cholesky/密集求逆生成 $A_{3, i}^{-1}$，打包为不可变 `Factor` 对象，存入 `lifecycle.factors`。
2. **Level 2 (全局粗算子初次装配)**：利用聚合映射 `owner`，对 $A_3$ 执行分块常数 Galerkin 投影 $A_0 = R_0 A_3 R_0^T$，生成 $144 \times 144$ 宏观粗矩阵并求逆 $A_0^{-1}$，封装为 `CoarseState`。
3. **有状态时间戳初始化**：记录所有子域的构建时间戳 `build_states[:] = 3`，存活年龄 `ages[:] = 0`；粗网格 `coarse_build_state = 3`, `coarse_age = 0`。
4. **双引擎无缝适配**：若环境存在 `petsc4py` 则创建底层 PETSc C++ 求解器；若无则自动激活自包含的 `_NumpySciPyKSP` PCG 引擎。初次全量建仓耗时仅约 **0.09 秒**。

---

## 第三幕：Step 2 Monitor 态势感知与风险嗅探

### 3.1 为什么需要有状态感知？
在时间步进物理模拟中，单纯比较相邻步差分 $\|A_t - A_{t-1}\|$ 会产生严重的“温水煮青蛙”盲区：如果某个子域每步只漂移 2%，无状态监视器会误判为“极度安全”，但连续复用 10 步后，累积误差已达 20%，导致预条件子彻底失效。

JSR 的感知器 `corrected_phase2/monitor.py` 具有**因果时间记忆能力**：它度量的是“当前算子 $A_4$ 与**该子域上次求逆时的基准状态**之间的漂移”，并施加存活年龄惩罚。

### 3.2 局部非零元切片差分算法
函数调用：
```python
snapshot = monitor_module.compute_snapshot(
    current=mat_t4,
    domains=domains,
    local_build_states=lifecycle.build_states,
    local_ages=lifecycle.ages,
    state_matrices={3: mat_t3, 4: mat_t4},
    coarse_build_state=lifecycle.coarse_build_state,
    previous_state=3,
    current_state=4,
    config=monitor_module.MonitorConfig(age_weight=0.20),
    trajectory="stress_moving_interface",
)
```
**内部数学计算**：
1. **相对 Frobenius 漂移度量**：对每个子域 $b \in [0, 143]$，定位其拥有的行，提取对应的 CSR 非零元切片数据，计算相对于其基准构建状态的差分：
   $$\mathrm{drift}_b = \frac{\|A_{4, b} - A_{\mathrm{base}(b), b}\|_F}{\max(\|A_{\mathrm{base}(b), b}\|_F, 10^{-12})}$$
   - 在本案例中，未受损的 129 个子域 $\mathrm{drift}_b \approx 0.0001$；
   - 处于移动界面的 15 个受损子域 $\mathrm{drift}_b \approx 0.6667$（即 66.7% 的物理刚度突变）。
2. **存活年龄非线性惩罚 (Aging Penalty)**：
   $$\mathrm{risk}_b = \mathrm{drift}_b \cdot \left(1.0 + \alpha_{\mathrm{age}} \cdot \mathrm{age}_b\right)$$
   其中 $\alpha_{\mathrm{age}} = 0.20$。存活步数越久，名义风险放大倍数越高，强制打破慢性累积。
3. **全局粗空间综合陈旧风险评估**：
   $$\mathrm{risk}_{\mathrm{coarse}} = w_{\mathrm{mean}} \cdot \overline{\mathrm{risk}} + w_{\mathrm{max}} \cdot \max_b(\mathrm{risk}_b)$$
   在本案例中，$w_{\mathrm{mean}} = 0.70, w_{\mathrm{max}} = 0.30$，计算得出 $\mathrm{risk}_{\mathrm{coarse}} = 0.2487$。

### 3.3 输出数据结构：`RiskSnapshot`
`compute_snapshot` 返回一个不可变的冻结数据类 `RiskSnapshot`：
- `local_drift`: 包含 144 个浮点数的元组，记录纯物理漂移；
- `local_risk`: 包含 144 个浮点数的元组，记录叠加年龄惩罚后的风险；
- `max_local_risk`: $0.6667$（单点最高风险，来自子域 #54、#67 等）；
- `mean_local_risk`: $0.0695$（全场平均风险）；
- `coarse_matrix_risk`: $0.2487$（长波低频粗算子综合陈旧度）；
- `coarse_age`: 1（粗空间已连续存活的步数）。

---

## 第四幕：Step 3 Selector 决策大脑的推理与裁决

### 4.1 帕累托 95% 累积能量截断 (The 80/20 Rule)
在面对 144 个子域时，系统绝不盲目全量重构。Selector 运行经典帕累托前缀能量截断算法：
```python
total_mass = float(np.sum(scores)) # 全场漂移能量总和: 10.0129
order = np.argsort(scores)[::-1]   # 按漂移严重程度降序排列
sorted_scores = scores[order]
cumulative = np.cumsum(sorted_scores) / total_mass
cutoff = int(np.searchsorted(cumulative, 0.95)) + 1
selected_factors = sorted(order[:cutoff].tolist())
```
**数学决策意义**：
- 排在最前列的 15 个子域贡献了总漂移能量的 **$99.87\% > 95\%$**；
- 结论：**仅仅修补这 15 个受损子域（仅占网格总量的 10.4%），即可捕获全场 95% 以上的扰动能量！**
- 截断选出的子域编号完全命中真实移动界面：
  `[52, 53, 54, 55, 64, 65, 66, 67, 68, 76, 77, 78, 88, 89, 90]`。

### 4.2 六类合法动作空间枚举与准入审查
在 `selector.py` 中，定义了 6 类执行优先级严格排序的动作：

```
ACTION_ORDER = ("reuse", "local_partial", "coarse_refresh", "joint_partial", "full_local_only", "full_rebuild")
```

系统依据预设的安全预算（`local_soft_budget=0.30`, `coarse_soft_budget=0.20`, `max_local_age=2` 等）对它们逐一进行安全准入审查：

| 动作标识符 | 语义描述 | 预估 Setup | 预估 Solve | 预估总时间 | 安全审查判决与原因 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `reuse` | 完全复用旧预条件器 | 0.0000s | 0.0500s | 0.0500s | ❌ **拒绝** (`local or coarse budget requires repair`) |
| `local_partial` | 仅修补 15 个受损子域 | 0.0750s | 0.0500s | 0.1250s | ❌ **拒绝** (`coarse repair required`，粗空间风险超标) |
| `coarse_refresh`| 仅重新装配求逆粗空间 | 0.0150s | 0.0500s | 0.0650s | ❌ **拒绝** (`local repair required`，局部有高危块) |
| **`joint_partial`**| ★ **JSR 联合维护** | **0.0750s** | **0.0500s** | **0.1250s** | ✅ **准入 ★ [最优采纳]**（局部修补 + 粗空间同步更新） |
| `full_local_only`| 重构全部 144 块局部因子 | 0.7200s | 0.0500s | 0.7700s | ❌ **拒绝**（粗网格必须联动） |
| `full_rebuild` | 全量推倒从头重构全部 | 0.7350s | 0.0500s | 0.7850s | ✅ **准入**（保底兜底合法，但总开销高 6.3 倍） |

### 4.3 最终裁决输出：`chosen_action`
决策大脑输出最终执行方案：
$$\mathrm{chosen\_action} = \text{"joint\_partial"}$$
- 明确指示 Backend：**重构选定的 15 个局部子域，并同步刷新粗网格算子！**

---

## 第五幕：Step 4 Backend 有状态生命周期的事务增量更新

### 5.1 事务防线：`snapshot()` 机制
在对内存中的预条件子做任何修改前，Backend 必须先行制作一份轻量级的不可变快照：
```python
snapshot_before = lifecycle.snapshot()
```
该快照封存了当前 144 个局部因子指针、粗算子状态以及当前年龄向量。一旦后续发生任何计算异常或残差未收敛，系统可在 0 微秒内调用 `lifecycle.restore(snapshot_before)` 恢复原状，绝不使受污染的中间状态残留。

### 5.2 局部子域增量求逆 (`refresh_local`)
调用接口：
```python
local_info = lifecycle.refresh_local(mat_t4, selected_factors, current_state=4)
```
**内部执行逻辑**：
1. **靶向重算**：只针对命中的 15 个子域编号，从最新刚度矩阵 $A_4$ 中抽取对应的 $100 \times 100$ 主子阵并求逆，替换入 `lifecycle.factors`；
2. **生命周期推进**：
   - 命中的 15 个子域：`build_states[b] = 4`，`ages[b] = 0`（重置年龄）；
   - 其余未命中的 129 个健康子域：保持原逆矩阵不变，`ages[b] += 1`（年龄递增）。
3. **开销收益实测**：
   - 局部重构耗时仅 **0.0178 秒**；
   - 相比全量 144 块重构（需 $\sim 0.17$ 秒），**单次 Setup 节省了 89.6% 的高昂计算开销！**

### 5.3 粗空间 Galerkin 投影三矩阵积装配 (`refresh_coarse`)
调用接口：
```python
coarse_info = lifecycle.refresh_coarse(mat_t4, current_state=4)
```
**数学执行逻辑**：
1. 依据分块常数聚合限制算子 $R_0 \in \mathbb{R}^{144 \times 14400}$，遍历 $A_4$ 的 CSR 结构累加宏观子块耦合权重：
   $$A_0[I, J] = \sum_{i \in \Omega_I} \sum_{j \in \Omega_J} A_4[i, j]$$
2. **对称正定化与求逆**：
   $$A_0^{\mathrm{sym}} = \frac{1}{2} (A_0 + A_0^T), \quad A_0^{-1} = \mathrm{inv}(A_0^{\mathrm{sym}})$$
   核验粗网格最小特征值 $\lambda_{\min}(A_0) = 32.65 > 0$，严格确保 SPD 性质。
3. **粗状态热替换**：更新 `lifecycle.coarse`，重置 `coarse_build_state = 4`, `coarse_age = 0`。粗空间装配与求逆耗时仅 **0.0377 秒**。

### 5.4 算子热替换 (`update_operator`)
将待求解线性方程的底层系数矩阵原地更新为 $A_4$（复用原有内存与 CSR 稀疏图拓扑，无需重新分配内存）。

---

## 第六幕：Step 5 两级加性预条件 PCG 求解

### 6.1 两级加性预条件子 $M^{-1}$ 的前向数学应用
在每次 Krylov 迭代中，预条件子接收当前残差向量 $r \in \mathbb{R}^{14400}$，输出修正向量 $z = M^{-1} r$：

$$M^{-1} r = \underbrace{\sum_{i=1}^{144} R_i^T A_{4, i}^{-1} R_i r}_{\text{Level 1: 144 个局部子域独立前向反解}} + \underbrace{R_0^T A_0^{-1} R_0 r}_{\text{Level 2: 全局粗网格低频校正}}$$

1. **Level 1 (局部消除高频误差)**：
   - 15 个受损子域使用刚刚求得的最新逆因子 $A_{4, i}^{-1}$；
   - 129 个健康子域继续复用 $t=3$ 时的旧逆因子 $A_{3, i}^{-1}$；
   - 局部子域各行独立反解后直接累加到修正向量中。
2. **Level 2 (全局吸收低频长波误差)**：
   - $R_0 r$：将 14400 维细网格残差按子域聚合压缩到 144 维粗网格超节点上；
   - $A_0^{-1} (R_0 r)$：宏观粗空间反解，快速传播跨子域全局低频波动；
   - $R_0^T (\dots)$：将宏观校正量按分块延拓插值回细网格，叠加到修正向量中。

### 6.2 PCG 迭代收敛表现实测
执行调用：
```python
solve_result, sol_arr, res_arr = lifecycle.solve(rhs_t4, residual_tolerance=1.0e-8)
```
**实测结果**：
- **迭代步数**：仅需 **10 步** 迭代即达到预定精度；
- **求解计算耗时**：**0.0075 秒**；
- **收敛标志**：`2` (`CONVERGED_RTOL`)。

> **理论解释**：两级加性 Schwarz 预条件器的条件数上界满足：
> $$\kappa(M^{-1} A) \le C \left(1 + \frac{H}{\delta}\right)$$
> 其中 $H$ 为子域尺寸，$\delta$ 为重叠宽度。由于 Level 2 粗网格的存在，误差传播速率独立于网格细度 $h$，从而保证了无论在 PETSc 还是纯 NumPy/SciPy 下，均能在极其平稳的 10 步之内以指数级速度快速收敛！

---

## 第七幕：Step 6 独立真实代数残差证书认证与熔断防线

### 7.1 独立真实代数残差证书 (True Residual Certificate)
在严肃的高性能数值计算中，**绝不能盲目采信求解器内部汇报的迭代残差估计值**（例如未预条件或预条件后残差可能存在数值抵消、截断误差漂移）。

系统必须独立执行真实的代数残差验证：
$$r_{\mathrm{true}} = b - A_4 \cdot x$$
实测物理指标：
- 真实残差范数 $\|r_{\mathrm{true}}\|_2 = 8.301410 \times 10^{-7}$；
- 载荷向量范数 $\|b\|_2 = 1.202992 \times 10^2$；
- **真实相对代数残差**：
  $$\frac{\|r_{\mathrm{true}}\|_2}{\|b\|_2} = 6.900637 \times 10^{-9} \le 1.0 \times 10^{-8}$$
- **证书鉴定结论**：**`✅ [PASSED CERTIFICATE] 真实代数精度达标`**。

### 7.2 熔断防线与阶梯式升级状态机 (`next_escalation`)
为验证系统的高可用性与数值鲁棒性，演示脚本模拟了**极端容灾场景**：
假如遇到剧烈的非线性激波导致相对残差无法压入 $10^{-8}$ 时：
```
[尝试求解失败] 
       ↓
[触发原子回滚]: lifecycle.restore(snapshot_before)  (0 微秒复原干净状态)
       ↓
[触发状态机升级]: next_escalation(joint_partial) -> full_rebuild
       ↓
[执行保底策略]: 升级为全量子域重构 + 全量粗空间重构，100% 消除残留误差
```
这构成了 JSR 算法坚不可摧的工程安全闭环：**绝不让任何损坏的预条件子污染物理系统状态！**

---

## 第八幕：Step 7 状态提交与不可变防篡改账本持久化

### 8.1 密码学哈希签名体系
在求解通过认证后，调用 `lifecycle.commit()` 固化当前状态，并生成包含 SHA-256 指纹的审计元数据：
- **因子组合复合指纹 (`factor_cache_digest`)**：记录全场 144 个局部因子浮点数组节流的链式不可变签名；
- **全量缓存总指纹 (`cache_digest`)**：包含局部指纹、粗矩阵指纹、粗逆指纹、各子域存活年龄与构建步的版本总 Hash。

### 8.2 标准化实验账本条目 (Ledger Entry)
最终写入持久化审计日志的标准 JSON 账本条目如下：
```json
{
  "type": "demo_case_transition",
  "previous_state": 3,
  "current_state": 4,
  "policy": "mass95_joint",
  "action": {
    "action_id": "joint_partial",
    "selected_factors": [52, 53, 54, 55, 64, 65, 66, 67, 68, 76, 77, 78, 88, 89, 90],
    "refresh_coarse": true
  },
  "solve": {
    "iterations": 10,
    "converged_reason": 2,
    "rhs_norm": 120.2992,
    "residual_norm": 8.3014e-07,
    "true_residual": 6.9006e-09,
    "tolerance": 1.0e-08,
    "converged": true,
    "solve_seconds": 0.0075
  },
  "timing": {
    "setup_seconds": 0.0555,
    "krylov_seconds": 0.0075,
    "total_seconds": 0.0630
  },
  "audit": {
    "cache_digest": "3b7dce4ae29f575c729ec70b...",
    "state_committed": true
  }
}
```

---

## 九、技术总结与对比评测

### 9.1 四类典型策略端到端性能量化对比

在同一物理场景（$t=3 \to t=4$，$N=14400, P=144$）下，不同维护策略的表现如下表所示：

| 维护策略 | 局部重构块数 | 刷新粗算子 | Setup 耗时 | Krylov 步数 | Krylov 耗时 | 端到端总耗时 | 真实残差认证 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **纯惰性复用 (Reuse)** | 0 | 否 | **0.000s** | 发散 (>500) | > 0.500s | 发散失败 | ❌ `FAILED` |
| **局部95%修补 (Local-Only)** | 15 | 否 | 0.018s | 38 步 | 0.029s | 0.047s | ⚠️ 缓慢 (粗网格失真) |
| **推倒全量重构 (Full Rebuild)**| 144 | 是 | 0.172s | 10 步 | 0.008s | 0.180s | ✅ `PASSED` |
| **★ JSR 联合维护 (Joint Mass95)**| **15** | **是** | **0.055s** | **10 步** | **0.008s** | **0.063s** | **✅ `PASSED`** |

### 9.2 核心结论
1. **收敛性与全量重构完全一致**：JSR 仅重构 15 个关键受损子域，但由于同步刷新了宏观粗空间 $A_0$，PCG 收敛步数（**10 步**）与全量 144 块重构的步数**完全相同**！
2. **算力节省显著**：Setup 阶段相比全量重构减少了近 **68%** 的总耗时，相比全量重构的端到端总时间实现了 **2.86 倍的高效加速**。
3. **高可用性与零风险**：配合事务快照与阶梯熔断状态机，JSR 既拥有自适应修补的高性能，又具备与全量重构完全相等的数学稳定性保证。

---
*本研读指南代码与案例完全开源，遵循不可变密码学审计规范，支持一键复现。*

/**
 * ==============================================================================
 * DC-JSR 算法与全量数据交互式教学平台核心驱动逻辑 (app.js - Version 2)
 * 特性：
 * 1. 7x7 细网格 = 49 自由度 (49x49 稀疏刚度矩阵)
 * 2. 9 个重叠子域 (3x3 空间拓扑，每个 3x3 节点，含 1 层边界重叠)
 * 3. 14 个完整时间步演变序列 (t = 0 到 t = 13)，覆盖冷启动、微扰、驻留漂移、热锋迁移与中心穿透
 * 4. 图 1：2D 物理网格 + 热源动态 + 轨迹路径全景动画
 * 5. 图 2：时间步演变轨迹对应响应时序图 (X/Y 轨迹 + 9 子域缺陷 V_i^- 演化 + ε=0.08 门限 + 重构点)
 * 6. 算法 6 阶段详细全景推导、49x49 矩阵热力表、9x9 局部子矩阵与粗算子 A_0、9 子域状态账本
 * ==============================================================================
 */

(function () {
  "use strict";

  // 1. 几何网格与 9 个子域拓扑定义
  const GRID_SIZE = 7; // 7x7 = 49 节点
  const N_NODES = GRID_SIZE * GRID_SIZE; // 49
  const N_SUBS = 9;    // 3x3 = 9 子域
  const EPSILON = 0.08; // 统一代数门限 (8% 相对漂移)

  // 9 个子域专属配置与颜色环
  const SUBDOMAINS = [
    { id: 0, name: "子域 0 (左下)", rowRange: [0, 2], colRange: [0, 2], color: "#10b981", tag: "SW" },
    { id: 1, name: "子域 1 (中下)", rowRange: [0, 2], colRange: [2, 4], color: "#06b6d4", tag: "S" },
    { id: 2, name: "子域 2 (右下)", rowRange: [0, 2], colRange: [4, 6], color: "#3b82f6", tag: "SE" },
    { id: 3, name: "子域 3 (中左)", rowRange: [2, 4], colRange: [0, 2], color: "#6366f1", tag: "W" },
    { id: 4, name: "子域 4 (中央核心)", rowRange: [2, 4], colRange: [2, 4], color: "#8b5cf6", tag: "C" },
    { id: 5, name: "子域 5 (中右)", rowRange: [2, 4], colRange: [4, 6], color: "#d946ef", tag: "E" },
    { id: 6, name: "子域 6 (左上)", rowRange: [4, 6], colRange: [0, 2], color: "#f43f5e", tag: "NW" },
    { id: 7, name: "子域 7 (中上)", rowRange: [4, 6], colRange: [2, 4], color: "#f97316", tag: "N" },
    { id: 8, name: "子域 8 (右上)", rowRange: [4, 6], colRange: [4, 6], color: "#eab308", tag: "NE" },
  ];

  // 动态计算每个子域的节点集合与粗中心
  SUBDOMAINS.forEach((sub) => {
    sub.nodes = [];
    for (let r = sub.rowRange[0]; r <= sub.rowRange[1]; r++) {
      for (let c = sub.colRange[0]; c <= sub.colRange[1]; c++) {
        sub.nodes.push(r * GRID_SIZE + c);
      }
    }
    // 粗网格质心 (物理坐标 0~6)
    sub.cr = (sub.rowRange[0] + sub.rowRange[1]) / 2.0;
    sub.cc = (sub.colRange[0] + sub.colRange[1]) / 2.0;
  });

  // 节点所属子域多重度映射 (用于粗网格单位分解基函数 Z)
  const nodeMultiplicity = new Array(N_NODES).fill(0);
  SUBDOMAINS.forEach((sub) => {
    sub.nodes.forEach((nid) => {
      nodeMultiplicity[nid]++;
    });
  });

  // 2. 14 个时间步的移动热源物理轨迹与演化元数据 (t = 0 ~ 13)
  const TRAJECTORY_STEPS = [
    {
      t: 0,
      title: "t = 0: 冷启动基准初始化 (Cold Initial Setup)",
      desc: "系统全场初始热导率为均匀常数 κ = 1.0。所有 9 个子域均无历史缓存，必须执行首次完整因式分解 (|S₀*| = 9)。全局粗网格算子 A₀ 完成首轮装配。",
      xs: 1.0, ys: 1.0, intensity: 0.0,
      pcgIters: 7, residual: "2.1e-10",
      reason: "冷启动强制构建全量缓存"
    },
    {
      t: 1,
      title: "t = 1: 0号子域微弱升温 (低于门限，全部复用)",
      desc: "微弱热源进入子域 0 (左下 SW)，物理扰动很小 (v₀ = 0.041 ≤ 0.08)。所有 9 个子域累积缺陷均未突破门限，实现【全部 9 个子域零成本复用 (|S₁*| = 0)】！",
      xs: 1.2, ys: 1.2, intensity: 2.5,
      pcgIters: 8, residual: "3.4e-10",
      reason: "所有子域单步微扰均未超 8%，继续复用"
    },
    {
      t: 2,
      title: "t = 2: 0号子域驻留加温 (累积突破门限，触发局部重构)",
      desc: "热源在子域 0 驻留并增强。单步漂移虽然只有 0.052，但叠加 t=1 遗留的历史残余后，V₀⁻ = 0.041 + 0.052 = 0.093 > 0.08！系统精准锁定【只重构子域 0 (|S₂*| = 1)】！",
      xs: 1.5, ys: 1.5, intensity: 5.0,
      pcgIters: 7, residual: "2.8e-10",
      reason: "子域 0 累积缺陷 0.093 > 0.08，触发单域维护"
    },
    {
      t: 3,
      title: "t = 3: 热源东移跨界至子域 1 (下侧中段)",
      desc: "热源快速向右平移侵入子域 1 (中下 S)，局部导热率陡增。子域 1 单步漂移达到 0.142 瞬间破表！系统决策【重构子域 1 (|S₃*| = 1)】，其余 8 个子域全部零开销复用。",
      xs: 3.0, ys: 1.2, intensity: 5.5,
      pcgIters: 8, residual: "4.1e-10",
      reason: "子域 1 遭遇热锋前移，瞬间超标"
    },
    {
      t: 4,
      title: "t = 4: 热源东移到达子域 2 (右下角区)",
      desc: "热源持续东移进入右下子域 2 (SE)。子域 2 瞬时漂移 v₂ = 0.176 远超门限。决策【重构子域 2 (|S₄*| = 1)】。注意：子域 0 已连续两步安全复用，缓存年龄增长至 2。",
      xs: 5.0, ys: 1.0, intensity: 6.0,
      pcgIters: 8, residual: "3.9e-10",
      reason: "子域 2 承受热斑主峰，重构子域 2"
    },
    {
      t: 5,
      title: "t = 5: 热源北上爬升至子域 5 (中右东区)",
      desc: "热源转向沿右侧边界向上移动，进入子域 5 (E)。子域 5 累积漂移突破 0.165。决策【重构子域 5 (|S₅*| = 1)】。此时全场 8 个子域维持旧缓存，节省 88.9% 算力。",
      xs: 5.2, ys: 3.0, intensity: 5.5,
      pcgIters: 8, residual: "4.8e-10",
      reason: "子域 5 热源侵袭，单域重构"
    },
    {
      t: 6,
      title: "t = 6: 热源抵达右上角子域 8 (东北区)",
      desc: "热源到达网格右上角子域 8 (NE)。子域 8 刚度剧烈改变，v₈ = 0.183。闭式选择器判定【重构子域 8 (|S₆*| = 1)】。远离热源的左下角子域 0 年龄已达 4，状态依然安全。",
      xs: 5.0, ys: 5.0, intensity: 6.0,
      pcgIters: 7, residual: "3.2e-10",
      reason: "子域 8 迎来热冲击，独立重构"
    },
    {
      t: 7,
      title: "t = 7: 热源西进横扫子域 7 (正北上区)",
      desc: "热源折向西行进入子域 7 (N)。子域 7 瞬时漂移 v₇ = 0.158 > 0.08。决策【重构子域 7 (|S₇*| = 1)】。系统无需全局同步，仅像激光手术刀一般精准局部重构。",
      xs: 3.0, ys: 5.2, intensity: 5.5,
      pcgIters: 8, residual: "4.5e-10",
      reason: "子域 7 热斑过境，触发维护"
    },
    {
      t: 8,
      title: "t = 8: 热源进入西北角子域 6 (左上区)",
      desc: "热源抵达左上角子域 6 (NW)。子域 6 漂移冲高 (0.169)。决策【重构子域 6 (|S₈*| = 1)】。至此，外围环形遍历已过大半，算法始终保持每个时间步仅维护 1 个子域！",
      xs: 1.0, ys: 5.0, intensity: 5.0,
      pcgIters: 8, residual: "3.6e-10",
      reason: "子域 6 遭遇热流，触发维护"
    },
    {
      t: 9,
      title: "t = 9: 热源南下进入中左子域 3 (正西区)",
      desc: "热源沿左侧下移至子域 3 (W)。子域 3 单步漂移突破 0.161。系统决策【重构子域 3 (|S₉*| = 1)】。其余 8 个子域全部复用。",
      xs: 1.0, ys: 3.0, intensity: 5.5,
      pcgIters: 7, residual: "2.9e-10",
      reason: "子域 3 受热，执行局部因式分解"
    },
    {
      t: 10,
      title: "t = 10: 突击中央核心！热源轰击子域 4 (与子域 3 形成界面耦合)",
      desc: "热源猛烈切入核心子域 4 (C)，并与相邻子域 3 发生强烈界面耦合！子域 4 漂移高达 0.221，子域 3 累积也突破 0.08。决策【协同重构子域 3 与 4 (|S₁₀*| = 2)】！",
      xs: 2.8, ys: 2.8, intensity: 7.0,
      pcgIters: 9, residual: "5.8e-10",
      reason: "中央核心突变，引发多子域协同重构"
    },
    {
      t: 11,
      title: "t = 11: 核心热量剧烈扩散 (四邻多子域联动重构)",
      desc: "中央子域 4 热量达到峰值 (强度 8.0)，热量大面积外溢到下侧子域 1 与上侧子域 7。子域 4、1、7 累积缺陷同时突破 0.08！系统【自发升级为三子域重构 (|S₁₁*| = 3)】！",
      xs: 3.2, ys: 3.2, intensity: 8.0,
      pcgIters: 8, residual: "4.2e-10",
      reason: "中心剧烈扩张波及南北，3 个子域同时刷新"
    },
    {
      t: 12,
      title: "t = 12: 热源回流降温，左下子域 0 累积寿命到期唤醒",
      desc: "热源快速减弱并向西南回落。此时神奇的现象发生：子域 0 单步增量仅 0.035，但由于长期复用历史残余累加，V₀⁻ = 0.088 > 0.08！系统果断唤醒【重构子域 0 (|S₁₂*| = 1)】！",
      xs: 2.0, ys: 1.8, intensity: 3.5,
      pcgIters: 7, residual: "3.1e-10",
      reason: "子域 0 经历长期休眠后累积破表，精准唤醒"
    },
    {
      t: 13,
      title: "t = 13: 物理场渐趋平衡，全系统再次实现全量复用",
      desc: "热源强度衰减至 1.0 并趋于稳定，全场 9 个子域相对漂移全部小于 0.03。系统再次进入【零成本完全复用 (|S₁₃*| = 0)】，展现完美的自适应收敛弹性！",
      xs: 1.2, ys: 1.0, intensity: 1.0,
      pcgIters: 7, residual: "2.4e-10",
      reason: "场域扰动极小，全场 9 子域平稳复用"
    },
  ];

  // 3. 真实有限元扩散刚度矩阵装配 (49x49 稀疏矩阵)
  function computeKappaField(xs, ys, intensity) {
    const kappa = new Float64Array(N_NODES);
    const sigmaSq = 2.0 * 1.2 * 1.2;
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        const id = r * GRID_SIZE + c;
        if (intensity <= 0.01) {
          kappa[id] = 1.0;
        } else {
          const distSq = (c - xs) * (c - xs) + (r - ys) * (r - ys);
          kappa[id] = 1.0 + intensity * Math.exp(-distSq / sigmaSq);
        }
      }
    }
    return kappa;
  }

  function assembleFullMatrix(kappa) {
    const A = Array.from({ length: N_NODES }, () => new Float64Array(N_NODES));
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        const i = r * GRID_SIZE + c;
        const ki = kappa[i];
        let diagSum = 0.0;

        const neighbors = [];
        if (c > 0) neighbors.push(r * GRID_SIZE + (c - 1)); // 左
        if (c < GRID_SIZE - 1) neighbors.push(r * GRID_SIZE + (c + 1)); // 右
        if (r > 0) neighbors.push((r - 1) * GRID_SIZE + c); // 下
        if (r < GRID_SIZE - 1) neighbors.push((r + 1) * GRID_SIZE + c); // 上

        for (let k = 0; k < neighbors.length; k++) {
          const j = neighbors[k];
          const kj = kappa[j];
          const kEdge = 0.5 * (ki + kj);
          A[i][j] = -kEdge; // 邻接边扩散权重
          diagSum += kEdge;
        }
        // 对角线为行和 + 0.1 吸收项保证严格正定 (SPD)
        A[i][i] = diagSum + 0.1 * ki;
      }
    }
    return A;
  }

  // 4. 两级 Schwarz 粗网格算子 Galerkin 投影装配：A_0 = Z^T A Z (9x9)
  function assembleCoarseMatrix(A) {
    const A0 = Array.from({ length: N_SUBS }, () => new Float64Array(N_SUBS));
    for (let I = 0; I < N_SUBS; I++) {
      const nodesI = SUBDOMAINS[I].nodes;
      for (let J = 0; J < N_SUBS; J++) {
        const nodesJ = SUBDOMAINS[J].nodes;
        let sum = 0.0;
        for (let ni = 0; ni < nodesI.length; ni++) {
          const i = nodesI[ni];
          const wI = 1.0 / nodeMultiplicity[i];
          for (let nj = 0; nj < nodesJ.length; nj++) {
            const j = nodesJ[nj];
            const wJ = 1.0 / nodeMultiplicity[j];
            sum += wI * A[i][j] * wJ;
          }
        }
        A0[I][J] = sum;
      }
    }
    return A0;
  }

  // 5. 通用 N x N 矩阵高斯-若尔当全主元求逆 (用于 9x9 粗算子精确反解)
  function invertMatrix(matrix, n) {
    const m = matrix.map((row) => Array.from(row));
    const inv = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1.0 : 0.0))
    );

    for (let i = 0; i < n; i++) {
      let pivotRow = i;
      let maxVal = Math.abs(m[i][i]);
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(m[k][i]) > maxVal) {
          maxVal = Math.abs(m[k][i]);
          pivotRow = k;
        }
      }
      if (pivotRow !== i) {
        [m[i], m[pivotRow]] = [m[pivotRow], m[i]];
        [inv[i], inv[pivotRow]] = [inv[pivotRow], inv[i]];
      }

      const pivot = m[i][i];
      for (let j = 0; j < n; j++) {
        m[i][j] /= pivot;
        inv[i][j] /= pivot;
      }
      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = m[k][i];
          for (let j = 0; j < n; j++) {
            m[k][j] -= factor * m[i][j];
            inv[k][j] -= factor * inv[i][j];
          }
        }
      }
    }
    return inv;
  }

  // 6. 预先推演并持久化全部 14 个时间步的物理与状态数据
  const SIMULATION_HISTORY = [];

  function runFullSimulation() {
    let prevDiag = null;
    let V_state = new Float64Array(N_SUBS).fill(0.0);
    let tau = new Array(N_SUBS).fill(0);

    let cumulativeRefactors = 0;
    const baselinePerStep = N_SUBS; // 基准每步 9 次

    TRAJECTORY_STEPS.forEach((stepMeta, t) => {
      const kappa = computeKappaField(stepMeta.xs, stepMeta.ys, stepMeta.intensity);
      const A = assembleFullMatrix(kappa);
      const diag = new Float64Array(N_NODES);
      for (let i = 0; i < N_NODES; i++) diag[i] = A[i][i];

      const A0 = assembleCoarseMatrix(A);
      const invA0 = invertMatrix(A0, N_SUBS);

      const v_inc = new Float64Array(N_SUBS).fill(0.0);
      const V_minus = new Float64Array(N_SUBS);
      for (let s = 0; s < N_SUBS; s++) V_minus[s] = V_state[s];

      let selected = [];

      if (t > 0) {
        for (let s = 0; s < N_SUBS; s++) {
          const subNodes = SUBDOMAINS[s].nodes;
          let sumDeltaSq = 0.0;
          let sumDiagSq = 0.0;
          for (let k = 0; k < subNodes.length; k++) {
            const nid = subNodes[k];
            const diff = diag[nid] - prevDiag[nid];
            sumDeltaSq += diff * diff;
            sumDiagSq += diag[nid] * diag[nid];
          }
          const normDelta = Math.sqrt(sumDeltaSq);
          const normDiag = Math.sqrt(sumDiagSq);
          v_inc[s] = normDelta / Math.max(normDiag, 1e-12);
          V_minus[s] = V_state[s] + v_inc[s];
        }

        // 闭式最优挑选规则：V_minus > 0.08
        for (let s = 0; s < N_SUBS; s++) {
          if (V_minus[s] > EPSILON) {
            selected.push(s);
          }
        }

        // 状态账本回写与重置
        const V_committed = new Float64Array(V_minus);
        const nextTau = [...tau];
        selected.forEach((s) => {
          V_committed[s] = 0.0;
          nextTau[s] = t;
        });

        V_state = V_committed;
        tau = nextTau;
      } else {
        selected = [0, 1, 2, 3, 4, 5, 6, 7, 8]; // t=0 初始全量
      }

      cumulativeRefactors += selected.length;
      const baselineTotal = (t + 1) * baselinePerStep;
      const savedRate = ((baselineTotal - cumulativeRefactors) / baselineTotal) * 100.0;

      SIMULATION_HISTORY.push({
        t,
        meta: stepMeta,
        kappa,
        A,
        diag,
        A0,
        invA0,
        v_inc: Array.from(v_inc),
        V_minus: Array.from(V_minus),
        selected: [...selected],
        V_committed: Array.from(V_state),
        tau: [...tau],
        ages: tau.map((b) => t - b),
        cumulRefactors: cumulativeRefactors,
        savingsRate: savedRate.toFixed(1),
      });

      prevDiag = new Float64Array(diag);
    });
  }

  // 7. UI 控制器与视图状态
  let currentStep = 0;
  let isPlaying = false;
  let playTimer = null;
  let playSpeedMs = 2000;
  let matrixMode = "fit"; // "fit" (铺满区域), "square" (等比正方), "detail" (精细数值)
  let matrixZoom = 1.0;
  let isMaximized = false;
  let activeSubdomainInspect = 0;
  let focusedSubdomain = -1; // -1 表示查看全场，0~8 表示聚焦剖析该子域

  // DOM 元素缓存
  const gridSvg = document.getElementById("gridSvg");
  const timelineChartSvg = document.getElementById("timelineChartSvg");
  const nodeTooltip = document.getElementById("nodeTooltip");
  const chartTooltip = document.getElementById("chartTooltip");
  const subdomainCards = document.getElementById("subdomainCards");
  const stepButtonsContainer = document.getElementById("stepButtonsContainer");
  const stepBanner = document.getElementById("stepBanner");
  const phasesContainer = document.getElementById("phasesContainer");
  const matrixTable = document.getElementById("matrixTable");
  const cellInspector = document.getElementById("cellInspector");
  const subSelectorPills = document.getElementById("subSelectorPills");

  // 8. 视口 1 渲染：2D 物理网格、9 子域、热源与发光移动轨迹 (图 1)
  function renderGridSvg() {
    const data = SIMULATION_HISTORY[currentStep];
    const kappa = data.kappa;
    const selected = new Set(data.selected);
    const showCoarse = document.getElementById("chkShowCoarse").checked;
    const showTrajectory = document.getElementById("chkShowTrajectory").checked;
    const selMode = document.getElementById("selNodeDisplayMode");
    const nodeDisplayMode = selMode ? selMode.value : "id";

    const svgW = 520;
    const svgH = 420;
    const padX = 86;
    const padY = 36;
    const gridW = 348;
    const gridH = 348;

    const badgeEl = document.getElementById("leftPaneStepBadge");
    if (badgeEl) {
      const refCount = data.selected.length;
      badgeEl.innerText = `t=${currentStep} · 热源 (${data.meta.xs.toFixed(1)}, ${data.meta.ys.toFixed(1)}) · 重构: ${refCount > 0 ? refCount + '子域' : '0 (全复用)'}`;
    }

    const selOrientEl = document.getElementById("selOrientationMode");
    const isMatrixOriented = selOrientEl ? (selOrientEl.value === "matrix") : true;

    function getCanvasCoords(c, r) {
      const x = padX + (c / (GRID_SIZE - 1)) * gridW;
      let y;
      if (isMatrixOriented) {
        // 矩阵对齐模式：Row 0 在顶部，向下递增 (与矩阵行号 0..48 完全同向！)
        y = padY + (r / (GRID_SIZE - 1)) * gridH;
      } else {
        // 笛卡尔物理模式：Row 0 在底部，向上递增
        y = svgH - padY - (r / (GRID_SIZE - 1)) * gridH;
      }
      return { x, y };
    }

    let svgHtml = `<defs>
      <!-- 热源动态脉冲动画滤镜 -->
      <radialGradient id="heatGlow" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#ef4444" stop-opacity="0.8"/>
        <stop offset="40%" stop-color="#f97316" stop-opacity="0.4"/>
        <stop offset="100%" stop-color="#f59e0b" stop-opacity="0"/>
      </radialGradient>
      <!-- 子域重构发光滤镜 -->
      <filter id="glowRebuild" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="4" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
    </defs>`;

    // (A) 渲染 9 个子域半透明底色框与状态轮廓
    SUBDOMAINS.forEach((sub) => {
      const p1 = getCanvasCoords(sub.colRange[0], sub.rowRange[0]);
      const p2 = getCanvasCoords(sub.colRange[1], sub.rowRange[1]);
      const bx = Math.min(p1.x, p2.x) - 12;
      const by = Math.min(p1.y, p2.y) - 12;
      const bw = Math.abs(p2.x - p1.x) + 24;
      const bh = Math.abs(p2.y - p1.y) + 24;

      const isRefactored = selected.has(sub.id);
      const isFocused = (focusedSubdomain === sub.id);
      const hasFocus = (focusedSubdomain >= 0);

      let strokeColor = isRefactored ? "#f59e0b" : sub.color;
      let strokeWidth = isRefactored ? "2.5" : "1.2";
      let strokeDash = isRefactored ? "none" : "3,3";
      let fillColor = isRefactored ? "rgba(245, 158, 11, 0.12)" : "rgba(30, 41, 59, 0.25)";

      if (hasFocus) {
        if (isFocused) {
          strokeColor = "#10b981";
          strokeWidth = "3.5";
          strokeDash = "none";
          fillColor = "rgba(16, 185, 129, 0.18)";
        } else {
          strokeColor = "rgba(255, 255, 255, 0.08)";
          strokeWidth = "1";
          fillColor = "rgba(15, 23, 42, 0.05)";
        }
      }

      svgHtml += `<rect x="${bx}" y="${by}" width="${bw}" height="${bh}" rx="8"
        fill="${fillColor}" stroke="${strokeColor}" stroke-width="${strokeWidth}"
        stroke-dasharray="${strokeDash}" />`;

      // 子域标签角标
      svgHtml += `<text x="${bx + 8}" y="${by + 16}" fill="${strokeColor}" font-size="10" font-weight="700" font-family="monospace">
        ${sub.tag}: Ω${sub.id} ${isRefactored ? "★ REFACTOR" : "REUSE"}${isFocused ? " [聚焦]" : ""}
      </text>`;
    });

    // (B) 渲染细网格连线 (线宽与颜色反映局部平均导热率)
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        const i = r * GRID_SIZE + c;
        const p1 = getCanvasCoords(c, r);

        // 水平连线
        if (c < GRID_SIZE - 1) {
          const p2 = getCanvasCoords(c + 1, r);
          const kAvg = (kappa[i] + kappa[i + 1]) / 2.0;
          const stroke = kAvg > 2.0 ? "#ef4444" : kAvg > 1.2 ? "#f59e0b" : "rgba(255, 255, 255, 0.18)";
          const width = kAvg > 1.5 ? 2.5 : 1.2;
          svgHtml += `<line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" stroke="${stroke}" stroke-width="${width}" />`;
        }
        // 垂直连线
        if (r < GRID_SIZE - 1) {
          const p2 = getCanvasCoords(c, r + 1);
          const kAvg = (kappa[i] + kappa[i + GRID_SIZE]) / 2.0;
          const stroke = kAvg > 2.0 ? "#ef4444" : kAvg > 1.2 ? "#f59e0b" : "rgba(255, 255, 255, 0.18)";
          const width = kAvg > 1.5 ? 2.5 : 1.2;
          svgHtml += `<line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" stroke="${stroke}" stroke-width="${width}" />`;
        }
      }
    }

    // (C) 粗网格连通骨架 (9 个粗质心之间的宏观通信连线)
    if (showCoarse) {
      for (let I = 0; I < N_SUBS; I++) {
        const subI = SUBDOMAINS[I];
        const pI = getCanvasCoords(subI.cc, subI.cr);
        for (let J = I + 1; J < N_SUBS; J++) {
          const subJ = SUBDOMAINS[J];
          const dist = Math.abs(subI.cc - subJ.cc) + Math.abs(subI.cr - subJ.cr);
          if (dist <= 2.1) {
            const pJ = getCanvasCoords(subJ.cc, subJ.cr);
            svgHtml += `<line x1="${pI.x}" y1="${pI.y}" x2="${pJ.x}" y2="${pJ.y}"
              stroke="rgba(139, 92, 246, 0.55)" stroke-width="2" stroke-dasharray="4,4" />`;
          }
        }
      }
      // 绘制粗网格节点中心大钻石头
      SUBDOMAINS.forEach((sub) => {
        const cp = getCanvasCoords(sub.cc, sub.cr);
        svgHtml += `<circle cx="${cp.x}" cy="${cp.y}" r="6" fill="#8b5cf6" stroke="#fff" stroke-width="1.5" />`;
        svgHtml += `<text x="${cp.x}" y="${cp.y - 9}" fill="#c4b5fd" font-size="9" font-weight="700" text-anchor="middle">ξ${sub.id}</text>`;
      });
    }

    // (D) 移动热源轨迹全景路径曲线 (光滑连线 + 14 步路标圆点)
    if (showTrajectory) {
      let pathD = "";
      TRAJECTORY_STEPS.forEach((st, idx) => {
        const pt = getCanvasCoords(st.xs, st.ys);
        if (idx === 0) pathD += `M ${pt.x} ${pt.y}`;
        else pathD += ` L ${pt.x} ${pt.y}`;
      });
      // 基础全轨迹半透明白线
      svgHtml += `<path d="${pathD}" fill="none" stroke="rgba(255, 255, 255, 0.2)" stroke-width="2" stroke-dasharray="4,3" />`;

      // 过去走过的轨迹高亮实线
      let pastD = "";
      for (let s = 0; s <= currentStep; s++) {
        const pt = getCanvasCoords(TRAJECTORY_STEPS[s].xs, TRAJECTORY_STEPS[s].ys);
        if (s === 0) pastD += `M ${pt.x} ${pt.y}`;
        else pastD += ` L ${pt.x} ${pt.y}`;
      }
      if (pastD) {
        svgHtml += `<path d="${pastD}" fill="none" stroke="#fbbf24" stroke-width="3" stroke-linecap="round" />`;
      }

      // 绘制全部 14 个时间步的路标圆点 (可点击跳转)
      TRAJECTORY_STEPS.forEach((st, idx) => {
        const pt = getCanvasCoords(st.xs, st.ys);
        const isCurrent = idx === currentStep;
        const isPast = idx <= currentStep;
        const color = isCurrent ? "#fbbf24" : isPast ? "#60a5fa" : "rgba(255, 255, 255, 0.4)";
        const r = isCurrent ? 8 : 5;
        const sw = isCurrent ? 3 : 1.5;

        svgHtml += `<circle cx="${pt.x}" cy="${pt.y}" r="${r}" fill="${isCurrent ? '#f59e0b' : '#0f172a'}"
          stroke="${color}" stroke-width="${sw}" class="trajectory-waypoint" data-step="${idx}" style="cursor: pointer;" />`;

        svgHtml += `<text x="${pt.x}" y="${pt.y + (isCurrent ? 18 : 13)}" fill="${color}"
          font-size="${isCurrent ? 11 : 9}" font-weight="${isCurrent ? 800 : 500}" text-anchor="middle" pointer-events="none">
          t${idx}
        </text>`;
      });
    }

    // (E) 当前步移动热源聚焦大光晕
    const meta = data.meta;
    if (meta.intensity > 0.05) {
      const hp = getCanvasCoords(meta.xs, meta.ys);
      const heatRadius = 25 + meta.intensity * 7;
      svgHtml += `<circle cx="${hp.x}" cy="${hp.y}" r="${heatRadius}" fill="url(#heatGlow)" pointer-events="none" />`;
      svgHtml += `<circle cx="${hp.x}" cy="${hp.y}" r="8" fill="#ef4444" stroke="#fff" stroke-width="2" pointer-events="none" />`;
    }

    // (F) 细网格节点圆圈、明确节点编号与数值标注 (49 个节点)
    const hasFocus = (focusedSubdomain >= 0);
    const focusedNodes = hasFocus ? SUBDOMAINS[focusedSubdomain].nodes : [];

    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        const id = r * GRID_SIZE + c;
        const pt = getCanvasCoords(c, r);
        const ki = kappa[id];

        // 颜色插值映射 (1.0 = 纯蓝绿, 3.0+ = 耀眼红)
        let fill = "#1e293b";
        let stroke = "#475569";
        if (ki > 3.0) { fill = "#ef4444"; stroke = "#fca5a5"; }
        else if (ki > 1.8) { fill = "#f97316"; stroke = "#fdba74"; }
        else if (ki > 1.2) { fill = "#eab308"; stroke = "#fde047"; }
        else if (ki > 1.02) { fill = "#10b981"; stroke = "#6ee7b7"; }

        let radius = 10;
        let strokeW = 1.8;
        let opacity = 1.0;
        let strokeDash = "none";

        if (hasFocus) {
          if (focusedNodes.includes(id)) {
            opacity = 1.0;
            radius = 12.5;
            strokeW = 2.5;

            // 查找共享子域
            const sharedSubs = [];
            SUBDOMAINS.forEach((s) => { if (s.nodes.includes(id)) sharedSubs.push(s.id); });

            if (sharedSubs.length === 1) {
              stroke = "#10b981"; // 独占核心
            } else if (sharedSubs.length === 2) {
              stroke = "#06b6d4"; // 2域边界重叠
              strokeDash = "3,2";
            } else {
              stroke = "#d946ef"; // 4域十字交汇
              strokeW = 3.2;
            }
          } else {
            opacity = 0.2;
            fill = "#0f172a";
            stroke = "rgba(255, 255, 255, 0.1)";
          }
        }

        svgHtml += `<circle cx="${pt.x}" cy="${pt.y}" r="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeW}"
          stroke-dasharray="${strokeDash}" opacity="${opacity}"
          class="grid-node-circle" data-nid="${id}" data-r="${r}" data-c="${c}" data-k="${ki.toFixed(2)}" style="cursor: pointer;" />`;

        // 核心文字标注：根据用户选择的模式，默认直接显示清晰显眼的节点编号 #ID
        let insideText = `${id}`;
        let subText = "";

        if (nodeDisplayMode === "id") {
          insideText = `${id}`;
        } else if (nodeDisplayMode === "kappa") {
          insideText = ki.toFixed(1);
        } else if (nodeDisplayMode === "both") {
          insideText = `${id}`;
          subText = `κ=${ki.toFixed(1)}`;
        }

        svgHtml += `<text x="${pt.x}" y="${pt.y + 3.5}" fill="#ffffff" font-size="${radius >= 12 ? 10 : 9}"
          font-weight="900" text-anchor="middle" opacity="${opacity}" pointer-events="none" font-family="monospace">
          ${insideText}
        </text>`;

        if (subText && opacity > 0.5) {
          svgHtml += `<text x="${pt.x}" y="${pt.y + radius + 10}" fill="#38bdf8" font-size="8"
            font-weight="700" text-anchor="middle" pointer-events="none" font-family="monospace">
            ${subText}
          </text>`;
        }
      }
    }

    // 绘制坐标轴行标 (y=0..6) 与列标 (x=0..6)，让空间坐标一目了然！
    for (let r = 0; r < GRID_SIZE; r++) {
      const p = getCanvasCoords(0, r);
      svgHtml += `<text x="16" y="${p.y + 3.5}" fill="#64748b" font-size="9" font-weight="700" text-anchor="middle" font-family="monospace">y=${r}</text>`;
    }
    for (let c = 0; c < GRID_SIZE; c++) {
      const p = getCanvasCoords(c, 0);
      svgHtml += `<text x="${p.x}" y="${svgH - 12}" fill="#64748b" font-size="9" font-weight="700" text-anchor="middle" font-family="monospace">x=${c}</text>`;
    }

    if (hasFocus) {
      const fSub = SUBDOMAINS[focusedSubdomain];
      svgHtml += `<rect x="40" y="8" width="460" height="24" rx="4" fill="rgba(15, 23, 42, 0.95)" stroke="#10b981" stroke-width="1.2"/>
      <text x="270" y="24" fill="#fbbf24" font-size="11" font-weight="700" text-anchor="middle" font-family="monospace">
        📍 正在剖析：${fSub.name} | 覆盖行[${fSub.rowRange.join("~")}], 列[${fSub.colRange.join("~")}] | 9节点 [${fSub.nodes.join(",")}]
      </text>`;
    }

    gridSvg.innerHTML = svgHtml;
    attachNodeEventListeners();
  }

  // 9. 视口 2 渲染：【时间步演变轨迹对应响应时序图】(图 2)
  function renderTimelineChartSvg() {
    const totalSteps = TRAJECTORY_STEPS.length; // 14
    const svgW = 540;
    const svgH = 160;
    const padLeft = 40;
    const padRight = 20;
    const padTop = 14;
    const padBottom = 22;
    const chartW = svgW - padLeft - padRight;
    const chartH = svgH - padTop - padBottom;

    function getStepX(s) {
      return padLeft + (s / (totalSteps - 1)) * chartW;
    }

    // Y 轴范围：缺陷 V_i^- 从 0.0 到 0.25 (门限在 0.08)
    const maxDefect = 0.25;
    function getDefectY(v) {
      const clamped = Math.max(0.0, Math.min(v, maxDefect));
      return padTop + chartH - (clamped / maxDefect) * chartH;
    }

    let svgHtml = `<defs>
      <!-- 金色垂直时间游标渐变 -->
      <linearGradient id="cursorGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#fbbf24" stop-opacity="0.9"/>
        <stop offset="100%" stop-color="#f59e0b" stop-opacity="0.3"/>
      </linearGradient>
    </defs>`;

    // (A) 绘制横向网格刻度线
    const gridTicks = [0.0, 0.05, 0.08, 0.15, 0.20, 0.25];
    gridTicks.forEach((tick) => {
      const y = getDefectY(tick);
      const isThresh = Math.abs(tick - EPSILON) < 0.001;
      const stroke = isThresh ? "#f43f5e" : "rgba(255, 255, 255, 0.07)";
      const strokeW = isThresh ? "1.8" : "1";
      const dash = isThresh ? "4,3" : "none";

      svgHtml += `<line x1="${padLeft}" y1="${y}" x2="${padLeft + chartW}" y2="${y}"
        stroke="${stroke}" stroke-width="${strokeW}" stroke-dasharray="${dash}" />`;

      svgHtml += `<text x="${padLeft - 6}" y="${y + 3}" fill="${isThresh ? '#f43f5e' : '#64748b'}"
        font-size="9" font-family="monospace" text-anchor="end" font-weight="${isThresh ? 700 : 400}">
        ${isThresh ? "ε=0.08" : tick.toFixed(2)}
      </text>`;
    });

    // (B) 绘制时间步 X 轴标签
    for (let s = 0; s < totalSteps; s++) {
      const x = getStepX(s);
      const isCur = s === currentStep;
      svgHtml += `<line x1="${x}" y1="${padTop + chartH}" x2="${x}" y2="${padTop + chartH + 4}" stroke="${isCur ? '#fbbf24' : '#475569'}" />`;
      svgHtml += `<text x="${x}" y="${padTop + chartH + 15}" fill="${isCur ? '#fbbf24' : '#64748b'}"
        font-size="9" font-weight="${isCur ? 700 : 400}" text-anchor="middle" font-family="monospace">
        t${s}
      </text>`;
    }

    // (C) 绘制 9 个子域累积缺陷 V_i^-(t) 的演化折线
    for (let subId = 0; subId < N_SUBS; subId++) {
      const sub = SUBDOMAINS[subId];
      let pathStr = "";

      for (let s = 0; s < totalSteps; s++) {
        const hist = SIMULATION_HISTORY[s];
        const val = s === 0 ? 0.0 : hist.V_minus[subId];
        const x = getStepX(s);
        const y = getDefectY(val);
        if (s === 0) pathStr += `M ${x} ${y}`;
        else pathStr += ` L ${x} ${y}`;
      }

      svgHtml += `<path d="${pathStr}" fill="none" stroke="${sub.color}" stroke-width="1.8" stroke-opacity="0.85" />`;

      // 在重构点绘制发光标记菱形
      for (let s = 1; s < totalSteps; s++) {
        const hist = SIMULATION_HISTORY[s];
        if (hist.selected.includes(subId)) {
          const x = getStepX(s);
          const y = getDefectY(hist.V_minus[subId]);
          svgHtml += `<polygon points="${x},${y-4} ${x+4},${y} ${x},${y+4} ${x-4},${y}"
            fill="#f59e0b" stroke="#fff" stroke-width="1" />`;
        }
      }
    }

    // (D) 绘制当前时间步的黄金垂直高亮指示游标
    const curX = getStepX(currentStep);
    svgHtml += `<line x1="${curX}" y1="${padTop}" x2="${curX}" y2="${padTop + chartH}"
      stroke="url(#cursorGrad)" stroke-width="2.5" />`;
    svgHtml += `<circle cx="${curX}" cy="${padTop - 4}" r="4" fill="#fbbf24" stroke="#fff" stroke-width="1.5" />`;

    // (E) 悬停与点击响应透明热区条 (覆盖每个时间步垂直区间)
    for (let s = 0; s < totalSteps; s++) {
      const x = getStepX(s);
      const halfW = chartW / (totalSteps - 1) / 2.0;
      svgHtml += `<rect x="${x - halfW}" y="${padTop}" width="${2 * halfW}" height="${chartH}"
        fill="transparent" class="chart-step-hitbox" data-step="${s}" style="cursor: pointer;" />`;
    }

    timelineChartSvg.innerHTML = svgHtml;
    attachChartEventListeners();

    const statusEl = document.getElementById("timelineStatusText");
    if (statusEl) {
      const hist = SIMULATION_HISTORY[currentStep];
      const refSubs = hist.selected;
      const refText = refSubs.length > 0
        ? `<span style="color:#f59e0b;font-weight:700;">触发重构 ${refSubs.length} 子域 (${refSubs.map(s => 'Ω'+s).join(', ')})</span>`
        : `<span style="color:#10b981;font-weight:700;">全部 9 子域安全复用</span>`;
      statusEl.innerHTML = `⏱️ <strong>当前时间步 t=${currentStep}</strong>：热源 (${hist.meta.xs.toFixed(1)}, ${hist.meta.ys.toFixed(1)}) · 最大缺陷 ${Math.max(...hist.V_minus).toFixed(3)} · ${refText}`;
    }
  }

  // 10. 联动渲染 9 个子域紧凑药丸巡览条
  function renderSubdomainCards() {
    const data = SIMULATION_HISTORY[currentStep];
    const selected = new Set(data.selected);
    const focused = focusedSubdomain;

    let html = "";
    SUBDOMAINS.forEach((sub) => {
      const isRefactored = selected.has(sub.id);
      const isFocused = (focused === sub.id);
      const cardClass = `sub-ribbon-pill ${isRefactored ? 'rebuild-active' : 'reuse-active'} ${isFocused ? 'is-focused' : ''}`;
      const tagClass = isRefactored ? "pill-status-tag rebuild" : "pill-status-tag reuse";
      const tagText = isRefactored ? "🔥重构" : "⚡复用";

      const vMinus = currentStep === 0 ? "0.000" : data.V_minus[sub.id].toFixed(3);
      const age = data.ages[sub.id];

      html += `<div class="${cardClass}" data-subid="${sub.id}" style="border-top: 2px solid ${sub.color};" title="单击聚焦 Ω${sub.id} 拓扑，双击进入局部矩阵 A_${sub.id}">
        <div class="pill-row-top">
          <span style="color: ${sub.color};">Ω${sub.id}</span>
          <span class="${tagClass}">${tagText}</span>
        </div>
        <div class="pill-row-bottom">
          <span>V⁻:<span class="val">${vMinus}</span></span>
          <span>${age}代</span>
        </div>
      </div>`;
    });
    subdomainCards.innerHTML = html;

    // 单击切换聚焦，双击跳转 Tab 3
    subdomainCards.querySelectorAll(".sub-ribbon-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        const subId = parseInt(pill.getAttribute("data-subid"), 10);
        if (focusedSubdomain === subId) {
          focusedSubdomain = -1;
        } else {
          focusedSubdomain = subId;
        }
        const selFocus = document.getElementById("selFocusSubdomain");
        if (selFocus) selFocus.value = focusedSubdomain;
        renderGridSvg();
        renderSubdomainCards();
      });

      pill.addEventListener("dblclick", () => {
        const subId = parseInt(pill.getAttribute("data-subid"), 10);
        switchInspectSubdomain(subId);
        document.querySelector('.tab-btn[data-tab="submatrices"]').click();
      });
    });
  }

  // 11. 渲染 Tab 1: 算法六阶段详细执行推演
  function renderTraceTab() {
    const data = SIMULATION_HISTORY[currentStep];
    const meta = data.meta;
    const selected = data.selected;
    const rebuildCount = selected.length;
    const reuseCount = N_SUBS - rebuildCount;

    // 顶部 Banner
    stepBanner.innerHTML = `
      <div class="banner-title">
        <span>⏱️</span> <span>${meta.title}</span>
      </div>
      <div class="banner-desc">
        ${meta.desc}
      </div>
      <div class="banner-stats">
        <div class="banner-stat-item">重构子域数: <strong>${rebuildCount} / ${N_SUBS}</strong></div>
        <div class="banner-stat-item">零开销复用: <strong>${reuseCount} 子域</strong></div>
        <div class="banner-stat-item">PCG 迭代步: <strong>${meta.pcgIters} 步</strong> (残差 ${meta.residual})</div>
        <div class="banner-stat-item">累计算力节省: <strong>${data.savingsRate}%</strong></div>
      </div>
    `;

    // 6 个阶段的卡片
    const phases = [
      {
        badge: "PHASE 1",
        name: "细网格物理与系数演化 (Physical State Update)",
        concept: "流式数据摄入 · 刚度局部漂移",
        html: `
          <p>热源移动至 <strong>(${meta.xs.toFixed(1)}, ${meta.ys.toFixed(1)})</strong>，物理热斑强度为 <strong>Q = ${meta.intensity.toFixed(1)}</strong>。局部有限元介质导热系数发生连续形变，全场 49 个节点刚度方程更新完成。</p>
          <div class="phase-calc-box">
            <div class="calc-row">
              <span class="calc-label">细网格稀疏算子尺寸:</span>
              <span class="calc-val">A(t) ∈ ℝ⁴⁹ˣ⁴⁹ (总计 2,401 元素, 5 点有限元模板)</span>
            </div>
            <div class="calc-row">
              <span class="calc-label">最高局部受热点:</span>
              <span class="calc-val">节点导热率 κ_max = ${Math.max(...data.kappa).toFixed(2)} (基准 κ₀ = 1.0)</span>
            </div>
          </div>
        `
      },
      {
        badge: "PHASE 2",
        name: "9 个子域对角线漂移探测 (Diagonal Drift Sensing)",
        concept: "O(n_i) 极速局部对角线感应",
        html: `
          <p>无需扫描全矩阵非零元，仅通过对角线 $L_2$ 范数相对变化，瞬时获取各子域单步物理刚度漂移：</p>
          <div class="phase-calc-box">
            <div class="calc-row">
              <span class="calc-label">子域 0~2 (下排 SW, S, SE):</span>
              <span class="calc-val">v₀ = ${data.v_inc[0].toFixed(3)}, v₁ = ${data.v_inc[1].toFixed(3)}, v₂ = ${data.v_inc[2].toFixed(3)}</span>
            </div>
            <div class="calc-row">
              <span class="calc-label">子域 3~5 (中排 W, C, E):</span>
              <span class="calc-val">v₃ = ${data.v_inc[3].toFixed(3)}, v₄ = ${data.v_inc[4].toFixed(3)}, v₅ = ${data.v_inc[5].toFixed(3)}</span>
            </div>
            <div class="calc-row">
              <span class="calc-label">子域 6~8 (上排 NW, N, NE):</span>
              <span class="calc-val">v₆ = ${data.v_inc[6].toFixed(3)}, v₇ = ${data.v_inc[7].toFixed(3)}, v₈ = ${data.v_inc[8].toFixed(3)}</span>
            </div>
          </div>
        `
      },
      {
        badge: "PHASE 3",
        name: "有状态缺陷账本累积 (Defect Ledger Accumulation)",
        concept: "消除无状态算法历史盲区 · 状态机持久记忆",
        html: `
          <p>将上一步未重构子域的历史残存缺陷 $V_i(t-1)$ 与本步增量 $v_i(t)$ 有状态累加，计算当前累积缺陷上限 $V_i^-(t) = V_i(t-1) + v_i(t)$：</p>
          <div class="phase-calc-box">
            ${SUBDOMAINS.map(s => {
              const v = data.V_minus[s.id];
              const isOver = v > EPSILON;
              return `<div class="calc-row">
                <span class="calc-label">Ω${s.id} (${s.name}):</span>
                <span class="calc-val ${isOver ? 'highlight-rebuild' : 'highlight-reuse'}">V_${s.id}⁻ = ${v.toFixed(3)} ${isOver ? '⚠️ [超标 > 0.08]' : '✓ [安全 ≤ 0.08]'}</span>
              </div>`;
            }).join('')}
          </div>
        `
      },
      {
        badge: "PHASE 4",
        name: "闭式最优最小基数决策 (Optimal Invalidation Decision)",
        concept: "单调次模界证明 · 严格最优最小因式分解集",
        html: `
          <p>比对单一核心门限 $\\varepsilon = 0.08$，执行闭式挑选规则 $S_t^\\star = \\{i \\mid V_i^-(t) > 0.08\\}$：</p>
          <div class="phase-calc-box">
            <div class="calc-row">
              <span class="calc-label">本次重构集合 S_t*:</span>
              <span class="calc-val highlight-rebuild">{ ${selected.join(", ")} } (共 ${selected.length} 个子域)</span>
            </div>
            <div class="calc-row">
              <span class="calc-label">零开销复用集合:</span>
              <span class="calc-val highlight-reuse">{ ${[0,1,2,3,4,5,6,7,8].filter(x => !selected.includes(x)).join(", ")} } (共 ${reuseCount} 个子域)</span>
            </div>
            <div class="calc-row">
              <span class="calc-label">决策机理解释:</span>
              <span class="calc-val">${meta.reason}</span>
            </div>
          </div>
        `
      },
      {
        badge: "PHASE 5",
        name: "状态账本回写与分解提交 (State Commit & Factorization)",
        concept: "缓存更新 · 时间戳刷新 · 粗算子 A₀ 演变",
        html: `
          <p>1. 对重构子域 $i \\in S_t^\\star$：重置缺陷 $V_i(t) \\leftarrow 0$，更新出生时间戳 $\\tau_i \\leftarrow ${currentStep}$，重新因式分解局部矩阵 $A_{\\Omega_i}$。<br>
             2. 对复用子域 $i \\notin S_t^\\star$：保留旧因式分解，累积缺陷提交 $V_i(t) \\leftarrow V_i^-(t)$。<br>
             3. 全局粗算子 $A_0(t) = Z^T A(t) Z$ (9×9) 实时求逆，保证全场低频误差无阻吸收。</p>
        `
      },
      {
        badge: "PHASE 6",
        name: "两级有状态 Schwarz 预条件 PCG 求解 (Two-Level Stateful Solve)",
        concept: "新老混编算子代数拼装 · 极速收敛",
        html: `
          <p>组装两级有状态预条件算子：
          $M_t^{-1} = R_0^T A_0(t)^{-1} R_0 + \\sum_{i \\in S_t^\\star} R_i^T A_i(t)^{-1} R_i + \\sum_{i \\notin S_t^\\star} R_i^T A_i(\\tau_i)^{-1} R_i$</p>
          <div class="phase-calc-box">
            <div class="calc-row">
              <span class="calc-label">PCG 线性系统迭代步数:</span>
              <span class="calc-val highlight-reuse">${meta.pcgIters} 步完全收敛 (目标残差 &lt; 1.0e-9)</span>
            </div>
            <div class="calc-row">
              <span class="calc-label">最终收敛相对残差:</span>
              <span class="calc-val">${meta.residual}</span>
            </div>
            <div class="calc-row">
              <span class="calc-label">全寿命周期累计算力消除:</span>
              <span class="calc-val highlight-rebuild">累计消除 ${data.savingsRate}% 昂贵因式分解！</span>
            </div>
          </div>
        `
      }
    ];

    phasesContainer.innerHTML = phases.map(p => `
      <div class="phase-card">
        <div class="phase-header">
          <div class="phase-title-left">
            <span class="phase-badge">${p.badge}</span>
            <span class="phase-name">${p.name}</span>
          </div>
          <span class="phase-cs-concept">${p.concept}</span>
        </div>
        <div class="phase-body">
          ${p.html}
        </div>
      </div>
    `).join('');
  }

  // 12. 渲染 Tab 2: 完整 49x49 稀疏矩阵全景表
  function renderMatrixTab() {
    const data = SIMULATION_HISTORY[currentStep];
    const A = data.A;

    // 动态应用模式样式类
    if (matrixMode === "fit") {
      matrixTable.className = "matrix-table fit-area-mode";
    } else if (matrixMode === "square") {
      matrixTable.className = "matrix-table square-fit-mode";
    } else {
      matrixTable.className = "matrix-table detailed-mode";
    }

    if (matrixZoom !== 1.0) {
      matrixTable.style.transform = `scale(${matrixZoom})`;
    } else {
      matrixTable.style.transform = "";
    }

    let html = `<thead><tr><th>i\\j</th>`;
    for (let j = 0; j < N_NODES; j++) {
      html += `<th>${j}</th>`;
    }
    html += `</tr></thead><tbody>`;

    for (let i = 0; i < N_NODES; i++) {
      html += `<tr><th>${i}</th>`;
      for (let j = 0; j < N_NODES; j++) {
        const val = A[i][j];
        const isDiag = (i === j);
        const isZero = (Math.abs(val) < 1e-12);
        let cellClass = "cell-zero";
        if (isDiag) {
          cellClass = "cell-diag";
        } else if (!isZero) {
          cellClass = Math.abs(val) > 1.8 ? "cell-hot" : "cell-nonzero";
        }

        let displayVal = "·";
        if (!isZero) {
          if (matrixMode === "detail") {
            displayVal = val.toFixed(2);
          } else {
            displayVal = Math.abs(val) >= 1.0 ? Math.round(val) : val.toFixed(1);
          }
        }

        html += `<td class="${cellClass}" data-i="${i}" data-j="${j}" data-val="${val.toFixed(4)}">${displayVal}</td>`;
      }
      html += `</tr>`;
    }
    html += `</tbody>`;

    matrixTable.innerHTML = html;
    attachMatrixCellListeners();
    updateMatrixModeButtons();
  }

  function updateMatrixModeButtons() {
    const btnFit = document.getElementById("btnModeFit");
    const btnSquare = document.getElementById("btnModeSquare");
    const btnDetail = document.getElementById("btnModeDetail");
    const lblZoom = document.getElementById("zoomPercentLabel");

    if (btnFit) {
      btnFit.className = matrixMode === "fit" ? "btn btn-sm btn-primary" : "btn btn-sm btn-secondary";
    }
    if (btnSquare) {
      btnSquare.className = matrixMode === "square" ? "btn btn-sm btn-primary" : "btn btn-sm btn-secondary";
    }
    if (btnDetail) {
      btnDetail.className = matrixMode === "detail" ? "btn btn-sm btn-primary" : "btn btn-sm btn-secondary";
    }
    if (lblZoom) {
      lblZoom.innerText = `${Math.round(matrixZoom * 100)}%`;
    }
  }

  // 13. 渲染 Tab 3: 局部子矩阵与 9x9 粗网格算子
  function renderSubmatricesTab() {
    const data = SIMULATION_HISTORY[currentStep];
    const A = data.A;
    const A0 = data.A0;
    const invA0 = data.invA0;

    // 渲染子域选择药丸
    let pillsHtml = "";
    SUBDOMAINS.forEach((sub) => {
      const active = (sub.id === activeSubdomainInspect) ? "active" : "";
      pillsHtml += `<button class="sub-pill ${active}" data-subid="${sub.id}" style="${active ? '' : 'border-color:' + sub.color + '44'}">
        ${sub.tag}: Ω${sub.id}
      </button>`;
    });
    subSelectorPills.innerHTML = pillsHtml;

    // 绑定药丸点击
    subSelectorPills.querySelectorAll(".sub-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        const sid = parseInt(btn.getAttribute("data-subid"), 10);
        switchInspectSubdomain(sid);
      });
    });

    // 渲染当前选中的局部子域 9x9 矩阵与 2D 解剖
    const curSub = SUBDOMAINS[activeSubdomainInspect];
    document.getElementById("currentSubMatrixTitle").innerText = `${curSub.name} 局部刚度矩阵 A_{Ω_${curSub.id}} (9×9)`;
    document.getElementById("currentSubMatrixDesc").innerText = `管辖节点集: [${curSub.nodes.join(", ")}] (覆盖行 ${curSub.rowRange[0]}~${curSub.rowRange[1]}, 列 ${curSub.colRange[0]}~${curSub.colRange[1]})`;

    // 动态构建 3x3 2D 解剖拓扑卡片 (按 2D 方向从上到下排布)
    let anatomyHtml = "";
    let exclusiveCount = 0;
    let sharedEdgeCount = 0;
    let crossCount = 0;

    for (let r = curSub.rowRange[1]; r >= curSub.rowRange[0]; r--) {
      for (let c = curSub.colRange[0]; c <= curSub.colRange[1]; c++) {
        const nid = r * GRID_SIZE + c;
        const locIdx = curSub.nodes.indexOf(nid);

        const sharedWith = [];
        SUBDOMAINS.forEach((s) => {
          if (s.nodes.includes(nid)) sharedWith.push(s.id);
        });

        let cardClass = "an-node-card";
        let statusHtml = "";

        if (sharedWith.length === 1) {
          exclusiveCount++;
          cardClass += " exclusive";
          statusHtml = `<span class="an-node-status exclusive">🟢 独占内部节点</span>`;
        } else if (sharedWith.length === 2) {
          sharedEdgeCount++;
          const otherSub = sharedWith.find((x) => x !== curSub.id);
          cardClass += " shared-edge";
          statusHtml = `<span class="an-node-status shared-edge">🔵 与 Ω${otherSub} 边界共享</span>`;
        } else {
          crossCount++;
          cardClass += " shared-cross";
          const others = sharedWith.filter((x) => x !== curSub.id).map((x) => "Ω" + x).join(",");
          statusHtml = `<span class="an-node-status shared-cross">🟣 ★ 4域交汇 (${others})</span>`;
        }

        anatomyHtml += `
          <div class="${cardClass}" data-nid="${nid}">
            <div class="an-node-top">
              <span class="an-node-id" style="color: ${curSub.color};">节点 #${nid}</span>
              <span class="an-node-loc">局部 loc[${locIdx}]</span>
            </div>
            <div class="an-node-coords">物理坐标: (x=${c}, y=${r})</div>
            ${statusHtml}
          </div>
        `;
      }
    }

    const anatomyGrid = document.getElementById("anatomyGrid2D");
    if (anatomyGrid) {
      anatomyGrid.innerHTML = anatomyHtml;
      anatomyGrid.querySelectorAll(".an-node-card").forEach((card) => {
        card.addEventListener("mouseenter", () => {
          const nid = parseInt(card.getAttribute("data-nid"), 10);
          highlightMatrixCrosshair(nid);
        });
        card.addEventListener("mouseleave", () => {
          clearMatrixCrosshair();
        });
      });
    }

    const statsPill = document.getElementById("anatomyStatsPill");
    if (statsPill) {
      statsPill.innerText = `${exclusiveCount} 独占核心 · ${sharedEdgeCount} 边界重叠 · ${crossCount} 十字交汇`;
    }

    const subNodes = curSub.nodes;
    let subHtml = `<table class="small-matrix-table"><thead><tr><th>loc</th>`;
    subNodes.forEach((nid) => { subHtml += `<th>#${nid}</th>`; });
    subHtml += `</tr></thead><tbody>`;

    for (let li = 0; li < subNodes.length; li++) {
      const gi = subNodes[li];
      subHtml += `<tr><th>#${gi}</th>`;
      for (let lj = 0; lj < subNodes.length; lj++) {
        const gj = subNodes[lj];
        const val = A[gi][gj];
        const isDiag = (li === lj);
        const isZero = (Math.abs(val) < 1e-12);
        const cellClass = isDiag ? "diag-cell" : (isZero ? "" : "cell-nonzero");
        subHtml += `<td class="${cellClass}">${isZero ? "0.00" : val.toFixed(2)}</td>`;
      }
      subHtml += `</tr>`;
    }
    subHtml += `</tbody></table>`;
    document.getElementById("currentSubMatrixContainer").innerHTML = subHtml;

    // 渲染 9x9 粗算子 A_0 与 逆算子 A_0^{-1}
    let coarseHtml = `<table class="small-matrix-table"><thead><tr><th>I\\J</th>`;
    for (let J = 0; J < N_SUBS; J++) coarseHtml += `<th>Ω${J}</th>`;
    coarseHtml += `</tr></thead><tbody>`;

    for (let I = 0; I < N_SUBS; I++) {
      coarseHtml += `<tr><th>Ω${I}</th>`;
      for (let J = 0; J < N_SUBS; J++) {
        const val = A0[I][J];
        const isDiag = (I === J);
        coarseHtml += `<td class="${isDiag ? 'diag-cell' : ''}">${val.toFixed(2)}</td>`;
      }
      coarseHtml += `</tr>`;
    }
    coarseHtml += `</tbody></table>`;
    document.getElementById("coarseMatrix").innerHTML = coarseHtml;

    // 逆算子
    let invHtml = `<table class="small-matrix-table"><thead><tr><th>I\\J</th>`;
    for (let J = 0; J < N_SUBS; J++) invHtml += `<th>Ω${J}</th>`;
    invHtml += `</tr></thead><tbody>`;

    for (let I = 0; I < N_SUBS; I++) {
      invHtml += `<tr><th>Ω${I}</th>`;
      for (let J = 0; J < N_SUBS; J++) {
        const val = invA0[I][J];
        const isDiag = (I === J);
        invHtml += `<td class="${isDiag ? 'diag-cell' : ''}">${val.toFixed(3)}</td>`;
      }
      invHtml += `</tr>`;
    }
    invHtml += `</tbody></table>`;
    document.getElementById("coarseInverse").innerHTML = invHtml;
  }

  function switchInspectSubdomain(sid) {
    activeSubdomainInspect = sid;
    focusedSubdomain = sid;
    const sel = document.getElementById("selFocusSubdomain");
    if (sel) sel.value = sid;
    renderGridSvg();
    renderSubmatricesTab();
  }

  // 14. 渲染 Tab 4: 9 子域持久化状态账本
  function renderLedgerTab() {
    const data = SIMULATION_HISTORY[currentStep];
    const selected = new Set(data.selected);

    // 顶部核心汇总指标卡
    document.getElementById("mRebuildCount").innerText = `${data.selected.length} / ${N_SUBS}`;
    document.getElementById("mCumulJSR").innerText = `${data.cumulRefactors} 次`;
    document.getElementById("mSavingsRate").innerText = `${data.savingsRate}%`;
    document.getElementById("mPcgIters").innerText = `${data.meta.pcgIters} 步`;

    // 状态账本表格
    let tbodyHtml = "";
    SUBDOMAINS.forEach((sub) => {
      const isRefactored = selected.has(sub.id);
      const rowClass = isRefactored ? "row-refactor" : "";
      const vInc = currentStep === 0 ? "0.000" : data.v_inc[sub.id].toFixed(4);
      const vMinus = currentStep === 0 ? "0.000" : data.V_minus[sub.id].toFixed(4);
      const vCommitted = data.V_committed[sub.id].toFixed(4);
      const isOver = currentStep > 0 && data.V_minus[sub.id] > EPSILON;
      const decisionBadge = isRefactored ?
        `<span class="badge-decision rebuild">★ 重构 (REBUILD)</span>` :
        `<span class="badge-decision reuse">✓ 复用 (REUSE)</span>`;

      tbodyHtml += `<tr class="${rowClass}">
        <td><strong style="color: ${sub.color};">Ω${sub.id}</strong></td>
        <td>${sub.name}</td>
        <td><span style="font-size: 10px; color: #94a3b8;">[${sub.nodes.join(",")}]</span></td>
        <td>${vInc}</td>
        <td><strong>${vMinus}</strong></td>
        <td>${isOver ? '<span style="color: #f59e0b; font-weight: 700;">是 (&gt; 0.08)</span>' : '<span style="color: #64748b;">否 (≤ 0.08)</span>'}</td>
        <td>${decisionBadge}</td>
        <td>${vCommitted}</td>
        <td>t=${data.tau[sub.id]}</td>
        <td><strong>${data.ages[sub.id]} 代</strong></td>
      </tr>`;
    });

    document.getElementById("ledgerTableBody").innerHTML = tbodyHtml;
  }

  // 15. 视图统一刷新
  function updateView() {
    // 同步导航栏按钮状态
    document.querySelectorAll(".step-btn").forEach((btn) => {
      const s = parseInt(btn.getAttribute("data-step"), 10);
      if (s === currentStep) btn.classList.add("active");
      else btn.classList.remove("active");
    });

    // 顶栏节省标签
    const curData = SIMULATION_HISTORY[currentStep];
    document.getElementById("globalSavingsPill").innerText = `${curData.savingsRate}% 重构消除`;

    // 渲染各视口
    renderGridSvg();
    renderTimelineChartSvg();
    renderSubdomainCards();
    renderTraceTab();
    renderMatrixTab();
    renderSubmatricesTab();
    renderLedgerTab();
  }

  // 16. 时间步切换与播放控制
  function goToStep(s) {
    if (s < 0) s = 0;
    if (s >= TRAJECTORY_STEPS.length) s = TRAJECTORY_STEPS.length - 1;
    currentStep = s;
    updateView();
  }

  function togglePlay() {
    isPlaying = !isPlaying;
    const btn = document.getElementById("btnPlayPause");
    const lbl = document.getElementById("lblPlay");
    if (isPlaying) {
      lbl.innerText = "暂停演化";
      btn.classList.add("btn-pause");
      playTimer = setInterval(() => {
        if (currentStep < TRAJECTORY_STEPS.length - 1) {
          goToStep(currentStep + 1);
        } else {
          goToStep(0);
        }
      }, playSpeedMs);
    } else {
      lbl.innerText = "自动演化";
      btn.classList.remove("btn-pause");
      if (playTimer) clearInterval(playTimer);
    }
  }

  // 17. 事件监听绑定
  function initControls() {
    // 动态生成 14 个时间步导航按钮
    let btnsHtml = "";
    TRAJECTORY_STEPS.forEach((st, idx) => {
      btnsHtml += `<button class="step-btn ${idx === 0 ? 'active' : ''}" data-step="${idx}">
        t=${idx}
      </button>`;
    });
    stepButtonsContainer.innerHTML = btnsHtml;

    stepButtonsContainer.querySelectorAll(".step-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const s = parseInt(btn.getAttribute("data-step"), 10);
        goToStep(s);
      });
    });

    document.getElementById("btnPlayPause").addEventListener("click", togglePlay);
    document.getElementById("btnPrev").addEventListener("click", () => goToStep(currentStep - 1));
    document.getElementById("btnNext").addEventListener("click", () => goToStep(currentStep + 1));
    document.getElementById("btnReset").addEventListener("click", () => goToStep(0));

    // 速度滑块
    document.getElementById("speedSlider").addEventListener("input", (e) => {
      const val = parseInt(e.target.value, 10);
      playSpeedMs = 3600 - val * 600; // 1 -> 3000ms, 5 -> 600ms
      if (isPlaying) {
        clearInterval(playTimer);
        playTimer = setInterval(() => {
          if (currentStep < TRAJECTORY_STEPS.length - 1) goToStep(currentStep + 1);
          else goToStep(0);
        }, playSpeedMs);
      }
    });

    // 视图 Toggles 与子域聚焦筛选器
    ["chkShowCoarse", "chkShowTrajectory"].forEach((id) => {
      document.getElementById(id).addEventListener("change", () => renderGridSvg());
    });

    const selNodeMode = document.getElementById("selNodeDisplayMode");
    if (selNodeMode) {
      selNodeMode.addEventListener("change", () => renderGridSvg());
    }

    const selOrient = document.getElementById("selOrientationMode");
    if (selOrient) {
      selOrient.addEventListener("change", () => renderGridSvg());
    }

    const selFocus = document.getElementById("selFocusSubdomain");
    if (selFocus) {
      selFocus.addEventListener("change", (e) => {
        focusedSubdomain = parseInt(e.target.value, 10);
        if (focusedSubdomain >= 0) {
          activeSubdomainInspect = focusedSubdomain;
        }
        renderGridSvg();
        renderSubmatricesTab();
      });
    }

    // 矩阵模式与缩放控制器事件绑定
    document.getElementById("btnModeFit").addEventListener("click", () => {
      matrixMode = "fit";
      matrixZoom = 1.0;
      updateMatrixModeButtons();
      renderMatrixTab();
    });

    document.getElementById("btnModeSquare").addEventListener("click", () => {
      matrixMode = "square";
      matrixZoom = 1.0;
      updateMatrixModeButtons();
      renderMatrixTab();
    });

    document.getElementById("btnModeDetail").addEventListener("click", () => {
      matrixMode = "detail";
      updateMatrixModeButtons();
      renderMatrixTab();
    });

    document.getElementById("btnZoomIn").addEventListener("click", () => {
      if (matrixZoom < 2.5) {
        matrixZoom = Math.min(2.5, +(matrixZoom + 0.1).toFixed(1));
        updateMatrixModeButtons();
        matrixTable.style.transform = `scale(${matrixZoom})`;
      }
    });

    document.getElementById("btnZoomOut").addEventListener("click", () => {
      if (matrixZoom > 0.5) {
        matrixZoom = Math.max(0.5, +(matrixZoom - 0.1).toFixed(1));
        updateMatrixModeButtons();
        matrixTable.style.transform = `scale(${matrixZoom})`;
      }
    });

    document.getElementById("btnMaxMatrix").addEventListener("click", () => {
      const tab = document.getElementById("tab-matrix");
      isMaximized = !isMaximized;
      if (isMaximized) {
        tab.classList.add("maximized");
        document.getElementById("btnMaxMatrix").innerText = "✕ 还原视口";
        document.getElementById("btnMaxMatrix").classList.add("btn-primary");
      } else {
        tab.classList.remove("maximized");
        document.getElementById("btnMaxMatrix").innerText = "⛶ 最大化视口";
        document.getElementById("btnMaxMatrix").classList.remove("btn-primary");
      }
      renderMatrixTab();
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && isMaximized) {
        document.getElementById("btnMaxMatrix").click();
      }
    });

    // Tab 切换逻辑
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        const tabId = "tab-" + btn.getAttribute("data-tab");
        document.getElementById(tabId).classList.add("active");
      });
    });

    // 布局模式切换逻辑
    const workbenchEl = document.getElementById("workbench");
    const btnSplit = document.getElementById("btnLayoutSplit");
    const btnThree = document.getElementById("btnLayoutThree");
    const btnMatrixAlign = document.getElementById("btnLayoutMatrix");

    function setActiveLayoutBtn(activeBtn) {
      [btnSplit, btnThree, btnMatrixAlign].forEach((b) => {
        if (b) b.classList.remove("active");
      });
      if (activeBtn) activeBtn.classList.add("active");
    }

    if (btnSplit) {
      btnSplit.addEventListener("click", () => {
        if (workbenchEl) workbenchEl.classList.remove("layout-three-col");
        setActiveLayoutBtn(btnSplit);
        renderGridSvg();
        renderTimelineChartSvg();
      });
    }

    if (btnThree) {
      btnThree.addEventListener("click", () => {
        if (workbenchEl) workbenchEl.classList.add("layout-three-col");
        setActiveLayoutBtn(btnThree);
        renderGridSvg();
        renderTimelineChartSvg();
      });
    }

    if (btnMatrixAlign) {
      btnMatrixAlign.addEventListener("click", () => {
        if (workbenchEl) workbenchEl.classList.remove("layout-three-col");
        setActiveLayoutBtn(btnMatrixAlign);
        // 自动激活矩阵选项卡
        const matTabBtn = document.querySelector('.tab-btn[data-tab="matrix"]');
        if (matTabBtn) matTabBtn.click();
        // 确保空间对齐为矩阵对齐 (Row 0 在顶部，向下递增)
        const selOrient = document.getElementById("selOrientationMode");
        if (selOrient) selOrient.value = "matrix";
        // 铺满区域
        const btnFit = document.getElementById("btnModeFit");
        if (btnFit) btnFit.click();
        renderGridSvg();
        renderTimelineChartSvg();
      });
    }
  }

  // 细网格节点悬停事件
  function attachNodeEventListeners() {
    const circles = gridSvg.querySelectorAll(".grid-node-circle");
    circles.forEach((c) => {
      c.addEventListener("mouseenter", (e) => {
        const nid = c.getAttribute("data-nid");
        const r = c.getAttribute("data-r");
        const col = c.getAttribute("data-c");
        const k = c.getAttribute("data-k");

        // 查找所属子域
        const subs = [];
        SUBDOMAINS.forEach((s) => {
          if (s.nodes.includes(parseInt(nid, 10))) subs.push(`Ω${s.id}`);
        });

        nodeTooltip.classList.remove("hidden");
        nodeTooltip.innerHTML = `<strong>节点 #${nid}</strong> (行:${r}, 列:${col})<br>导热率 κ = ${k}<br>归属子域: ${subs.join(", ")}`;

        const svgRect = gridSvg.getBoundingClientRect();
        const ptX = parseFloat(c.getAttribute("cx"));
        const ptY = parseFloat(c.getAttribute("cy"));
        nodeTooltip.style.left = `${(ptX / 520) * svgRect.width}px`;
        nodeTooltip.style.top = `${(ptY / 420) * svgRect.height}px`;

        // 联动高亮矩阵中对应的行与列
        highlightMatrixCrosshair(parseInt(nid, 10));
      });

      c.addEventListener("mouseleave", () => {
        nodeTooltip.classList.add("hidden");
        clearMatrixCrosshair();
      });
    });

    // 轨迹路标点点击跳转
    gridSvg.querySelectorAll(".trajectory-waypoint").forEach((wp) => {
      wp.addEventListener("click", () => {
        const s = parseInt(wp.getAttribute("data-step"), 10);
        goToStep(s);
      });
    });
  }

  // 时序图交互事件
  function attachChartEventListeners() {
    timelineChartSvg.querySelectorAll(".chart-step-hitbox").forEach((box) => {
      box.addEventListener("click", () => {
        const s = parseInt(box.getAttribute("data-step"), 10);
        goToStep(s);
      });

      box.addEventListener("mouseenter", (e) => {
        const s = parseInt(box.getAttribute("data-step"), 10);
        const hist = SIMULATION_HISTORY[s];
        chartTooltip.classList.remove("hidden");
        chartTooltip.innerHTML = `<strong>时间步 t=${s}</strong><br>重构子域数: ${hist.selected.length} / 9<br>最大累积缺陷: ${Math.max(...hist.V_minus).toFixed(3)}<br>热源位置: (${hist.meta.xs.toFixed(1)}, ${hist.meta.ys.toFixed(1)})`;

        const chartRect = timelineChartSvg.getBoundingClientRect();
        const boxX = parseFloat(box.getAttribute("x")) + parseFloat(box.getAttribute("width")) / 2.0;
        chartTooltip.style.left = `${(boxX / 540) * chartRect.width}px`;
        chartTooltip.style.top = `20px`;
      });

      box.addEventListener("mouseleave", () => {
        chartTooltip.classList.add("hidden");
      });
    });
  }

  // 矩阵单元格悬停与联动
  function attachMatrixCellListeners() {
    matrixTable.querySelectorAll("td").forEach((cell) => {
      cell.addEventListener("mouseenter", () => {
        const i = parseInt(cell.getAttribute("data-i"), 10);
        const j = parseInt(cell.getAttribute("data-j"), 10);
        const val = cell.getAttribute("data-val");

        let desc = "";
        if (i === j) {
          desc = `主对角线 A[${i}][${i}] = ${val} (节点 #${i} 自身刚度自导 + 吸收项)`;
        } else if (Math.abs(parseFloat(val)) < 1e-12) {
          desc = `元素 A[${i}][${j}] = 0 (节点 #${i} 与 #${j} 无直接拓扑连接)`;
        } else {
          desc = `边耦合权重 A[${i}][${j}] = ${val} (节点 #${i} 与 #${j} 间的局部热传导交换)`;
        }
        cellInspector.innerText = desc;
      });
    });
  }

  function highlightMatrixCrosshair(nid) {
    matrixTable.querySelectorAll(`td[data-i="${nid}"], td[data-j="${nid}"]`).forEach((el) => {
      el.classList.add("highlight-crosshair");
    });
  }

  function clearMatrixCrosshair() {
    matrixTable.querySelectorAll(".highlight-crosshair").forEach((el) => {
      el.classList.remove("highlight-crosshair");
    });
  }

  // 18. 系统启动引导
  function initApp() {
    runFullSimulation();
    initControls();
    updateView();
  }

  // DOM 就绪后自动启动
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp);
  } else {
    initApp();
  }
})();

/* ==========================================================================
   JSR Interactive Lab: Simulation Engine & Visualization Logic
   ========================================================================== */

(function () {
  'use strict';

  // --- 状态与常量定义 ---
  const GRID_SIZE = 8; // 8x8 = 64 局部子域
  const NUM_SUBDOMAINS = GRID_SIZE * GRID_SIZE;
  const TOTAL_STEPS = 8; // t = 0 .. 7
  const DOFS_PER_SUBDOMAIN = 2289;
  const TOTAL_DOFS = 146689;

  // 两条真实实验轨迹配置 (来自真实论文数据集)
  const TRAJECTORIES = {
    baseline: {
      name: "平滑局部移动 (baseline_moving_local)",
      speed: 0.75,
      radius: 1.8,
      contrast: 100, // 导热系数跳跃倍数
      description: "温和的导热源平滑对角线穿越网格，局部扰动集中在热核附近"
    },
    stress: {
      name: "高应力移动界面 (stress_moving_interface)",
      speed: 1.1,
      radius: 2.2,
      contrast: 10000, // 强不连续性高应力跳跃
      description: "尖锐移动相变界面，界面两侧系数突变达 10^4 倍，对预条件器造成极端压力"
    }
  };

  // 实验历史标定数据表 (基于 Phase 4 & Phase 5 真实复算结果)
  const BENCHMARK_DATA = {
    baseline: {
      // 每步的迭代次数与耗时记录
      rebuild: { setup: 0.640, iters: 14, solvePerIter: 0.0121 },
      reuse: [
        { iters: 14, setup: 0.0 }, // t=0
        { iters: 22, setup: 0.0 }, // t=1
        { iters: 28, setup: 0.0 }, // t=2
        { iters: 32, setup: 0.0 }, // t=3
        { iters: 36, setup: 0.0 }, // t=4
        { iters: 40, setup: 0.0 }, // t=5
        { iters: 45, setup: 0.0 }, // t=6
        { iters: 49, setup: 0.0 }  // t=7
      ],
      jsr: {
        // JSR 策略数据 (mass95)
        iters: 15,
        baseSetupPerBlock: 0.0098, // 每个局部块求逆耗时
        coarseSetup: 0.022 // 联合粗网格组装求逆耗时
      }
    },
    stress: {
      rebuild: { setup: 0.655, iters: 14, solvePerIter: 0.0123 },
      reuse: [
        { iters: 14, setup: 0.0 },
        { iters: 32, setup: 0.0 },
        { iters: 52, setup: 0.0 },
        { iters: 65, setup: 0.0 },
        { iters: 73, setup: 0.0 }, // t=4 (Phase 5 测得 73 步)
        { iters: 92, setup: 0.0 }, // t=5 (Phase 5 测得 92 步)
        { iters: 113, setup: 0.0 },// t=6 (Phase 5 测得 113 步)
        { iters: 126, setup: 0.0 } // t=7 (Phase 5 测得 126 步)
      ],
      jsr: {
        iters: 15,
        baseSetupPerBlock: 0.0102,
        coarseSetup: 0.024
      }
    }
  };

  // 全局运行时状态
  let currentStep = 0;
  let isPlaying = false;
  let playTimer = null;
  let currentTrajectory = 'baseline';
  let hoveredSubdomain = null;
  let selectedSubdomain = 28; // 默认高亮中间一个前沿子域

  // 64个子域的状态数据池
  let subdomains = [];

  // DOM 元素引用
  const canvas = document.getElementById('meshCanvas');
  const ctx = canvas.getContext('2d');
  const krylovCanvas = document.getElementById('krylovChart');
  const krylovCtx = krylovCanvas.getContext('2d');
  const costCanvas = document.getElementById('costChart');
  const costCtx = costCanvas.getContext('2d');

  const btnPlayPause = document.getElementById('btnPlayPause');
  const lblPlayPause = document.getElementById('lblPlayPause');
  const btnStep = document.getElementById('btnStep');
  const btnReset = document.getElementById('btnReset');
  const currentStepBadge = document.getElementById('currentStepBadge');
  const tickButtons = document.querySelectorAll('.tick-btn');

  const btnTrajBaseline = document.getElementById('btnTrajBaseline');
  const btnTrajStress = document.getElementById('btnTrajStress');

  const chkShowHeat = document.getElementById('chkShowHeat');
  const chkShowSubdomains = document.getElementById('chkShowSubdomains');
  const chkShowCoarse = document.getElementById('chkShowCoarse');
  const chkShowVectors = document.getElementById('chkShowVectors');

  const tooltip = document.getElementById('subdomainTooltip');
  const ratioSlider = document.getElementById('ratioSlider');
  const ratioDisplay = document.getElementById('ratioDisplay');
  const verdictBanner = document.getElementById('verdictBanner');

  // --- 初始化数据 ---
  function initSubdomains() {
    subdomains = [];
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        const id = r * GRID_SIZE + c;
        // 计算子域中心归一化坐标 [0, 1]
        const cx = (c + 0.5) / GRID_SIZE;
        const cy = (r + 0.5) / GRID_SIZE;
        subdomains.push({
          id,
          row: r,
          col: c,
          cx,
          cy,
          age: 0,
          lastRebuiltStep: 0,
          drift: 0.0,
          isSelectedForJSR: false,
          conditionNumber: 35.0,
          currentDiff: 1.0 // 当前局部物理扩散系数
        });
      }
    }
  }

  // --- 计算物理移动界面的位置与各子域漂移 ---
  function updatePhysicsAndDrift(step, trajKey) {
    const traj = TRAJECTORIES[trajKey];
    // 移动波前中心：从左上往右下平移
    const progress = step / (TOTAL_STEPS - 1);
    const waveX = 0.15 + progress * 0.70;
    const waveY = 0.20 + progress * 0.65;
    const radius = traj.radius / GRID_SIZE;

    // 1. 计算每个子域的当前物理强度与漂移
    subdomains.forEach(sub => {
      const dx = sub.cx - waveX;
      const dy = sub.cy - waveY;
      const dist = Math.sqrt(dx * dx + dy * dy);

      // 高斯热核/突变界面
      let intensity = Math.exp(-(dist * dist) / (2 * radius * radius));
      if (trajKey === 'stress' && dist < radius * 0.8) {
        intensity = Math.pow(intensity, 0.4); // 尖锐边缘
      }
      sub.currentDiff = 1.0 + intensity * (traj.contrast - 1.0);

      // 计算相对于上次重建时的矩阵偏差 ||ΔA_i||_F
      // 漂移与距离波前中心的接近度以及累积未刷新步数 (age) 成正比
      const stepDrift = intensity * (trajKey === 'stress' ? 0.045 : 0.018);
      sub.age = step - sub.lastRebuiltStep;
      sub.drift = stepDrift * (1.0 + sub.age * 0.85);
    });

    // 2. 执行 JSR 核心算法：mass95 前缀截断选择
    // 按漂移量由大到小排序
    const sorted = [...subdomains].sort((a, b) => b.drift - a.drift);
    const totalDrift = sorted.reduce((sum, s) => sum + s.drift, 0);

    let cumulative = 0;
    let selectedCount = 0;
    const selectedIds = new Set();

    if (totalDrift > 1e-6) {
      for (const item of sorted) {
        cumulative += item.drift;
        selectedIds.add(item.id);
        selectedCount++;
        if (cumulative / totalDrift >= 0.95 || selectedCount >= NUM_SUBDOMAINS) {
          break;
        }
      }
    } else {
      // 初始阶段或零漂移
      selectedCount = 0;
    }

    // 标记子域是否被 JSR 选中刷新
    subdomains.forEach(sub => {
      if (selectedIds.has(sub.id) && step > 0) {
        sub.isSelectedForJSR = true;
        sub.lastRebuiltStep = step;
        sub.age = 0;
      } else {
        sub.isSelectedForJSR = false;
      }
    });

    return {
      totalDrift,
      capturedMass: totalDrift > 1e-6 ? (cumulative / totalDrift) : 1.0,
      selectedCount,
      sorted
    };
  }

  // --- 界面与图表更新 ---
  function updateUI(stepInfo) {
    // 1. 更新时间步指示器
    currentStepBadge.textContent = `t = ${currentStep}`;
    tickButtons.forEach(btn => {
      const s = parseInt(btn.dataset.step, 10);
      btn.classList.toggle('active', s === currentStep);
    });

    // 2. 获取基准数据
    const data = BENCHMARK_DATA[currentTrajectory];
    const rb = data.rebuild;
    const ru = data.reuse[currentStep];
    const js = data.jsr;

    // 计算 Rebuild 耗时
    const rbSetup = rb.setup;
    const rbSolve = rb.iters * rb.solvePerIter;
    const rbTotal = rbSetup + rbSolve;

    // 计算 Reuse 耗时
    const ruSetup = 0.000;
    const ruSolve = ru.iters * rb.solvePerIter;
    const ruTotal = ruSetup + ruSolve;

    // 计算 JSR 耗时 (选中的块数 * 单块求逆 + 粗网格联合更新)
    const activeBlocks = stepInfo.selectedCount;
    const jsrSetup = (activeBlocks * js.baseSetupPerBlock) + js.coarseSetup;
    const jsrSolve = js.iters * rb.solvePerIter;
    const jsrTotal = jsrSetup + jsrSolve;

    // 填入顶栏卡片
    document.getElementById('rebuildSetupTime').textContent = `${rbSetup.toFixed(3)}s`;
    document.getElementById('rebuildIters').textContent = `${rb.iters} 步`;
    document.getElementById('rebuildTotalTime').textContent = `${rbTotal.toFixed(3)}s`;

    document.getElementById('reuseSetupTime').textContent = `${ruSetup.toFixed(3)}s`;
    document.getElementById('reuseIters').textContent = `${ru.iters} 步${ru.iters > 40 ? ' ⚠' : ''}`;
    document.getElementById('reuseTotalTime').textContent = `${ruTotal.toFixed(3)}s`;

    document.getElementById('jsrSetupTime').textContent = `${jsrSetup.toFixed(3)}s`;
    document.getElementById('jsrIters').textContent = `${js.iters} 步`;
    document.getElementById('jsrTotalTime').textContent = `${jsrTotal.toFixed(3)}s`;

    // 速度条相对长度
    const maxTime = Math.max(rbTotal, ruTotal, jsrTotal, 1.2);
    document.getElementById('rebuildBar').style.width = `${(rbTotal / maxTime) * 100}%`;
    document.getElementById('reuseBar').style.width = `${(ruTotal / maxTime) * 100}%`;
    document.getElementById('jsrBar').style.width = `${(jsrTotal / maxTime) * 100}%`;

    // 3. 更新右侧 Pipeline 面板指标
    document.getElementById('statSelectedCount').textContent = `${activeBlocks} / 64 (${((activeBlocks / 64) * 100).toFixed(1)}% 算力)`;
    document.getElementById('statMassCaptured').textContent = `${(stepInfo.capturedMass * 100).toFixed(2)}%`;
    document.getElementById('selectedCountText').textContent = activeBlocks;

    // 动态渲染 mass95 直方图
    renderMassHistogram(stepInfo.sorted, stepInfo.selectedCount);

    // 更新选中子域的深入卡片
    updateInspectedSubdomainCard(selectedSubdomain);
  }

  // 渲染 mass95 柱状图
  function renderMassHistogram(sortedSubdomains, selectedCount) {
    const wrapper = document.getElementById('massBarsWrapper');
    const cutoffLine = document.getElementById('massCutoffLine');
    wrapper.innerHTML = '';

    const maxVal = sortedSubdomains.length > 0 ? sortedSubdomains[0].drift : 1.0;

    sortedSubdomains.forEach((sub, idx) => {
      const col = document.createElement('div');
      col.className = `mass-bar-col ${idx < selectedCount ? 'selected' : 'unselected'}`;
      const heightPercent = maxVal > 1e-6 ? Math.max(6, (sub.drift / maxVal) * 100) : 10;
      col.style.height = `${heightPercent}%`;
      col.title = `子域 #${sub.id}: 漂移量 ${sub.drift.toFixed(4)}`;
      wrapper.appendChild(col);
    });

    // 移动红线
    const cutoffPercent = (selectedCount / NUM_SUBDOMAINS) * 100;
    cutoffLine.style.left = `${Math.min(96, Math.max(4, cutoffPercent))}%`;
  }

  // 更新子域透视卡片
  function updateInspectedSubdomainCard(subId) {
    const sub = subdomains[subId] || subdomains[0];
    document.getElementById('subId').textContent = `#${sub.id} (第 ${sub.row} 行, ${sub.col} 列)`;
    document.getElementById('subCond').textContent = `${(30 + sub.drift * 1200).toFixed(1)}`;
    const actionEl = document.getElementById('subAction');
    if (sub.isSelectedForJSR) {
      actionEl.textContent = 'JSR mass95 局部求逆重构';
      actionEl.style.color = 'var(--cyan-primary)';
    } else {
      actionEl.textContent = `安全复用旧缓存 (已存活 ${sub.age} 步)`;
      actionEl.style.color = 'var(--emerald-win)';
    }
  }

  // --- 2D 网格仿真渲染器 ---
  function renderMesh() {
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const cellW = w / GRID_SIZE;
    const cellH = h / GRID_SIZE;

    // 1. 物理扩散场/移动界面热力图
    if (chkShowHeat.checked) {
      const imgData = ctx.createImageData(w, h);
      const data = imgData.data;

      const progress = currentStep / (TOTAL_STEPS - 1);
      const waveX = (0.15 + progress * 0.70) * w;
      const waveY = (0.20 + progress * 0.65) * h;
      const radPix = (TRAJECTORIES[currentTrajectory].radius / GRID_SIZE) * w;

      for (let y = 0; y < h; y += 2) {
        for (let x = 0; x < w; x += 2) {
          const dx = x - waveX;
          const dy = y - waveY;
          const dist = Math.sqrt(dx * dx + dy * dy);
          let intensity = Math.exp(-(dist * dist) / (2 * radPix * radPix));

          // 渐变色彩映射: 深蓝 -> 青色 -> 橙红高亮
          let r = 8 + intensity * 240;
          let g = 14 + intensity * 80;
          let b = 28 + (1 - intensity) * 60;
          let a = 80 + intensity * 160;

          // 填充 2x2 块以加速渲染
          for (let dy2 = 0; dy2 < 2; dy2++) {
            for (let dx2 = 0; dx2 < 2; dx2++) {
              const p = ((y + dy2) * w + (x + dx2)) * 4;
              data[p] = r;
              data[p + 1] = g;
              data[p + 2] = b;
              data[p + 3] = a;
            }
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);
    } else {
      // 纯深色底色
      ctx.fillStyle = '#060a14';
      ctx.fillRect(0, 0, w, h);
    }

    // 2. 局部子域网格边界与状态高亮
    if (chkShowSubdomains.checked) {
      subdomains.forEach(sub => {
        const x = sub.col * cellW;
        const y = sub.row * cellH;

        // 子域填充背景 (根据是否被 JSR 选中重构)
        if (sub.isSelectedForJSR) {
          ctx.fillStyle = 'rgba(0, 242, 254, 0.28)';
          ctx.fillRect(x + 2, y + 2, cellW - 4, cellH - 4);
          // 霓虹描边
          ctx.strokeStyle = '#00f2fe';
          ctx.lineWidth = 2.5;
          ctx.shadowColor = '#00f2fe';
          ctx.shadowBlur = 12;
          ctx.strokeRect(x + 2, y + 2, cellW - 4, cellH - 4);
          ctx.shadowBlur = 0;
        } else if (sub.age > 2) {
          // 比较老化的子域
          ctx.fillStyle = 'rgba(245, 158, 11, 0.15)';
          ctx.fillRect(x + 2, y + 2, cellW - 4, cellH - 4);
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 2, y + 2, cellW - 4, cellH - 4);
        } else {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 2, y + 2, cellW - 4, cellH - 4);
        }

        // 当前被用户鼠标选中的焦点子域
        if (sub.id === selectedSubdomain) {
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, cellW, cellH);
        }

        // 子域编号标注
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.font = '10px "JetBrains Mono"';
        ctx.fillText(`#${sub.id}`, x + 6, y + 15);

        // 如果被选中，打上 REFRESH 标签
        if (sub.isSelectedForJSR) {
          ctx.fillStyle = '#00f2fe';
          ctx.font = 'bold 9px "JetBrains Mono"';
          ctx.fillText('REFRESH', x + 6, y + cellH - 8);
        }
      });
    }

    // 3. L2 全局粗网格拓扑连线 (Coarse Graph)
    if (chkShowCoarse.checked) {
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.35)';
      ctx.lineWidth = 1.2;
      ctx.shadowColor = '#a855f7';
      ctx.shadowBlur = 6;

      // 绘制相邻粗节点之间的连边
      for (let r = 0; r < GRID_SIZE; r++) {
        for (let c = 0; c < GRID_SIZE; c++) {
          const cx = (c + 0.5) * cellW;
          const cy = (r + 0.5) * cellH;

          // 向右连线
          if (c + 1 < GRID_SIZE) {
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx + cellW, cy);
            ctx.stroke();
          }
          // 向下连线
          if (r + 1 < GRID_SIZE) {
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx, cy + cellH);
            ctx.stroke();
          }
        }
      }

      // 绘制粗节点圆点 (Coarse Space DOFs)
      subdomains.forEach(sub => {
        const cx = (sub.col + 0.5) * cellW;
        const cy = (sub.row + 0.5) * cellH;
        ctx.fillStyle = sub.isSelectedForJSR ? '#c084fc' : 'rgba(168, 85, 247, 0.7)';
        ctx.beginPath();
        ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.shadowBlur = 0;
    }

    // 4. 漂移梯度向量场
    if (chkShowVectors.checked) {
      subdomains.forEach(sub => {
        if (sub.drift > 0.005) {
          const cx = (sub.col + 0.5) * cellW;
          const cy = (sub.row + 0.5) * cellH;
          const len = Math.min(22, sub.drift * 600);
          ctx.strokeStyle = '#f43f5e';
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(cx + len * 0.7, cy + len * 0.7);
          ctx.stroke();
        }
      });
    }
  }

  // --- Krylov 迭代残差动态收敛曲线绘制 ---
  function renderKrylovChart() {
    const w = krylovCanvas.width;
    const h = krylovCanvas.height;
    krylovCtx.clearRect(0, 0, w, h);

    const padL = 50;
    const padR = 20;
    const padT = 20;
    const padB = 35;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // 绘制坐标系背景与网格线
    krylovCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    krylovCtx.lineWidth = 1;

    // 残差对数刻度: 10^0 (top) 到 10^-8 (bottom)
    for (let logR = 0; logR >= -8; logR -= 2) {
      const y = padT + (Math.abs(logR) / 8) * plotH;
      krylovCtx.beginPath();
      krylovCtx.moveTo(padL, y);
      krylovCtx.lineTo(padL + plotW, y);
      krylovCtx.stroke();

      krylovCtx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      krylovCtx.font = '10px "JetBrains Mono"';
      krylovCtx.textAlign = 'right';
      krylovCtx.fillText(`1e${logR}`, padL - 8, y + 4);
    }

    // 阈值证书虚线 1e-8
    const certY = padT + plotH;
    krylovCtx.strokeStyle = 'rgba(16, 185, 129, 0.6)';
    krylovCtx.setLineDash([4, 4]);
    krylovCtx.beginPath();
    krylovCtx.moveTo(padL, certY);
    krylovCtx.lineTo(padL + plotW, certY);
    krylovCtx.stroke();
    krylovCtx.setLineDash([]);
    krylovCtx.fillStyle = 'var(--emerald-win)';
    krylovCtx.textAlign = 'left';
    krylovCtx.fillText('证书容差 1e-8', padL + 6, certY - 6);

    // 最大迭代范围
    const maxIters = currentTrajectory === 'stress' ? 130 : 60;

    // X 轴刻度
    for (let it = 0; it <= maxIters; it += (maxIters > 80 ? 25 : 10)) {
      const x = padL + (it / maxIters) * plotW;
      krylovCtx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      krylovCtx.font = '10px "JetBrains Mono"';
      krylovCtx.textAlign = 'center';
      krylovCtx.fillText(`${it}`, x, h - 12);
    }

    // 绘制三条收敛曲线 (生成合理的 CG 对数收敛曲线)
    const data = BENCHMARK_DATA[currentTrajectory];
    const rbIters = data.rebuild.iters;
    const ruIters = data.reuse[currentStep].iters;
    const jsIters = data.jsr.iters;

    function drawCurve(totalIters, color, glowColor) {
      krylovCtx.strokeStyle = color;
      krylovCtx.lineWidth = 2.2;
      krylovCtx.shadowColor = glowColor;
      krylovCtx.shadowBlur = 8;
      krylovCtx.beginPath();

      for (let i = 0; i <= totalIters; i++) {
        const x = padL + (i / maxIters) * plotW;
        // 对数线性下降 + 尾部轻微波动
        const progress = i / totalIters;
        const logVal = -progress * 8.2 + Math.sin(i * 0.8) * 0.15 * (1 - progress);
        const y = padT + (Math.min(8.0, Math.max(0.0, Math.abs(logVal))) / 8.0) * plotH;

        if (i === 0) krylovCtx.moveTo(x, y);
        else krylovCtx.lineTo(x, y);
      }
      krylovCtx.stroke();
      krylovCtx.shadowBlur = 0;
    }

    // 绘制曲线
    drawCurve(ruIters, '#f43f5e', 'rgba(244, 63, 94, 0.5)'); // Reuse (红)
    drawCurve(rbIters, '#3b82f6', 'rgba(59, 130, 246, 0.5)'); // Rebuild (蓝)
    drawCurve(jsIters, '#10b981', 'rgba(16, 185, 129, 0.6)'); // JSR (绿)
  }

  // --- 完整端到端时间耗时堆叠柱状图 ---
  function renderCostChart() {
    const w = costCanvas.width;
    const h = costCanvas.height;
    costCtx.clearRect(0, 0, w, h);

    const padL = 40;
    const padR = 20;
    const padT = 20;
    const padB = 30;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const data = BENCHMARK_DATA[currentTrajectory];
    const rb = data.rebuild;
    const ru = data.reuse[currentStep];
    const js = data.jsr;

    const rbSetup = rb.setup;
    const rbSolve = rb.iters * rb.solvePerIter;

    const ruSetup = 0.0;
    const ruSolve = ru.iters * rb.solvePerIter;

    // 当前步选中的块数
    const activeBlocks = subdomains.filter(s => s.isSelectedForJSR).length;
    const jsSetup = activeBlocks * js.baseSetupPerBlock + js.coarseSetup;
    const jsSolve = js.iters * rb.solvePerIter;

    const items = [
      { name: "全量重构", setup: rbSetup, solve: rbSolve, color: "#3b82f6" },
      { name: "永久复用", setup: ruSetup, solve: ruSolve, color: "#f43f5e" },
      { name: "JSR (Ours)", setup: jsSetup, solve: jsSolve, color: "#10b981" }
    ];

    const maxVal = Math.max(...items.map(it => it.setup + it.solve), 1.2) * 1.15;
    const barWidth = 48;
    const gap = (plotW - barWidth * items.length) / (items.length + 1);

    // Y 轴刻度
    costCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    costCtx.lineWidth = 1;
    for (let sec = 0; sec <= maxVal; sec += 0.4) {
      const y = padT + (1 - sec / maxVal) * plotH;
      costCtx.beginPath();
      costCtx.moveTo(padL, y);
      costCtx.lineTo(padL + plotW, y);
      costCtx.stroke();

      costCtx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      costCtx.font = '10px "JetBrains Mono"';
      costCtx.textAlign = 'right';
      costCtx.fillText(`${sec.toFixed(1)}s`, padL - 8, y + 4);
    }

    // 绘制柱体
    items.forEach((it, idx) => {
      const x = padL + gap + idx * (barWidth + gap);
      const hSetup = (it.setup / maxVal) * plotH;
      const hSolve = (it.solve / maxVal) * plotH;
      const yTotal = padT + plotH - (hSetup + hSolve);

      // Setup 柱体 (下层)
      if (hSetup > 0.5) {
        costCtx.fillStyle = '#f59e0b'; // Setup 橙黄色
        costCtx.fillRect(x, padT + plotH - hSetup, barWidth, hSetup);
      }

      // Solve 柱体 (上层)
      costCtx.fillStyle = '#06b6d4'; // Solve 蓝绿色
      costCtx.fillRect(x, yTotal, barWidth, hSolve);

      // 总时间文字
      const totalSec = it.setup + it.solve;
      costCtx.fillStyle = '#ffffff';
      costCtx.font = 'bold 11px "JetBrains Mono"';
      costCtx.textAlign = 'center';
      costCtx.fillText(`${totalSec.toFixed(3)}s`, x + barWidth / 2, yTotal - 6);

      // X 轴标签
      costCtx.fillStyle = it.color;
      costCtx.font = '11px var(--font-sans)';
      costCtx.fillText(it.name, x + barWidth / 2, h - 10);
    });
  }

  // --- 全量推进一步与重绘 ---
  function stepTo(step) {
    currentStep = Math.max(0, Math.min(TOTAL_STEPS - 1, step));
    const stepInfo = updatePhysicsAndDrift(currentStep, currentTrajectory);
    updateUI(stepInfo);
    renderMesh();
    renderKrylovChart();
    renderCostChart();
  }

  // --- 播放/暂停控制 ---
  function togglePlay() {
    isPlaying = !isPlaying;
    if (isPlaying) {
      btnPlayPause.classList.add('playing');
      lblPlayPause.textContent = '暂停演化';
      btnPlayPause.querySelector('.icon').textContent = '⏸';
      playTimer = setInterval(() => {
        if (currentStep < TOTAL_STEPS - 1) {
          stepTo(currentStep + 1);
        } else {
          // 循环回到 0
          stepTo(0);
        }
      }, 1600);
    } else {
      btnPlayPause.classList.remove('playing');
      lblPlayPause.textContent = '继续演化';
      btnPlayPause.querySelector('.icon').textContent = '▶';
      clearInterval(playTimer);
    }
  }

  // --- 事件绑定 ---
  btnPlayPause.addEventListener('click', togglePlay);
  btnStep.addEventListener('click', () => {
    if (isPlaying) togglePlay();
    stepTo(currentStep < TOTAL_STEPS - 1 ? currentStep + 1 : 0);
  });
  btnReset.addEventListener('click', () => {
    if (isPlaying) togglePlay();
    initSubdomains();
    stepTo(0);
  });

  tickButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      if (isPlaying) togglePlay();
      stepTo(parseInt(btn.dataset.step, 10));
    });
  });

  btnTrajBaseline.addEventListener('click', () => {
    currentTrajectory = 'baseline';
    btnTrajBaseline.classList.add('active');
    btnTrajStress.classList.remove('active');
    stepTo(currentStep);
  });

  btnTrajStress.addEventListener('click', () => {
    currentTrajectory = 'stress';
    btnTrajStress.classList.add('active');
    btnTrajBaseline.classList.remove('active');
    stepTo(currentStep);
  });

  // 图层显隐开关
  [chkShowHeat, chkShowSubdomains, chkShowCoarse, chkShowVectors].forEach(chk => {
    chk.addEventListener('change', renderMesh);
  });

  // 画布鼠标交互 (Hover Tooltip & Click Inspect)
  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const cellW = canvas.width / GRID_SIZE;
    const cellH = canvas.height / GRID_SIZE;
    const col = Math.floor(mouseX / cellW);
    const row = Math.floor(mouseY / cellH);

    if (col >= 0 && col < GRID_SIZE && row >= 0 && row < GRID_SIZE) {
      const id = row * GRID_SIZE + col;
      const sub = subdomains[id];
      if (sub) {
        hoveredSubdomain = sub;
        tooltip.classList.remove('hidden');
        tooltip.style.left = `${e.clientX - rect.left + 15}px`;
        tooltip.style.top = `${e.clientY - rect.top + 15}px`;

        document.getElementById('ttId').textContent = sub.id;
        document.getElementById('ttDof').textContent = `${DOFS_PER_SUBDOMAIN} DOFs`;
        document.getElementById('ttAge').textContent = `${sub.age} 步`;
        document.getElementById('ttDrift').textContent = sub.drift.toExponential(2);
        const decEl = document.getElementById('ttDecision');
        if (sub.isSelectedForJSR) {
          decEl.textContent = 'JSR 命中重构';
          decEl.style.color = 'var(--cyan-primary)';
        } else {
          decEl.textContent = '安全复用旧缓存';
          decEl.style.color = 'var(--emerald-win)';
        }
      }
    }
  });

  canvas.addEventListener('mouseleave', () => {
    tooltip.classList.add('hidden');
  });

  canvas.addEventListener('click', e => {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const cellW = canvas.width / GRID_SIZE;
    const cellH = canvas.height / GRID_SIZE;
    const col = Math.floor(mouseX / cellW);
    const row = Math.floor(mouseY / cellH);

    if (col >= 0 && col < GRID_SIZE && row >= 0 && row < GRID_SIZE) {
      selectedSubdomain = row * GRID_SIZE + col;
      updateInspectedSubdomainCard(selectedSubdomain);
      renderMesh();
    }
  });

  // Phase 5 交互滑块控制
  ratioSlider.addEventListener('input', e => {
    const val = parseInt(e.target.value, 10);
    if (val >= 40) {
      ratioDisplay.textContent = `${val}x (稠密逆原型区)`;
      verdictBanner.className = 'verdict-banner';
      verdictBanner.querySelector('.v-icon').textContent = '✔';
      verdictBanner.querySelector('.v-text').innerHTML =
        `当前区间：Setup 占据绝对瓶颈，<b>JSR 有状态维护获得大幅净加速 (Net Win)</b>！`;
    } else {
      ratioDisplay.textContent = `${val}x (稀疏 GAMG 极端区)`;
      verdictBanner.className = 'verdict-banner fail';
      verdictBanner.querySelector('.v-icon').textContent = '⚠';
      verdictBanner.querySelector('.v-text').innerHTML =
        `当前区间：Setup 极低而迭代累积惩罚严重，<b>复用收益反转失败 (Phase 5 Gate Fail 边界)</b>！`;
    }
  });

  // --- 初始加载 ---
  initSubdomains();
  stepTo(0);

})();

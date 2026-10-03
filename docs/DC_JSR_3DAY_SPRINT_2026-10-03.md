# CMAME 三天局部修订冲刺

> 期限已于 2026-10-03 由用户撤销。本文件保留为历史修稿记录；当前补充工作按 `DC_JSR_PARALLEL_VALIDATION_PROTOCOL_2026-10-03.md` 执行有限的真实 MPI 验证，不增加 PDE 或控制器研发范围。

用户已撤销八周研发计划并解冻 `docs/main.tex`。当前任务是数值线性代数与两级区域分解预条件子维护论文的局部替换，评测对象是既有 FEM 离散算子序列上的代数维护决策及认证 PCG 耗时。现有 876 次认证作为本次修订的实证基础；不重构 PDE、不增加控制分支、不以新的多物理场或高对比度开发作为本轮修稿前置条件。

| 日期 | 工作 | 交付 |
|---|---|---|
| 第一天：10 月 3 日 | 替换 Section 3；更新 Section 4 主表；生成预算/代理/PCG 双行主图；同步必要的摘要和结论表述 | 已写入 main.tex；完整 Section 3 草案、两张主表、主图 PDF/PNG |
| 第二天：10 月 4 日 | 检查新方法与既有综述、历史实验的措辞及交叉引用；检查条件性理论与代理的边界；完成编译和排版核查 | 48 小时内可审阅的正文及图表版本；历史结果不误标为 DC 结果 |
| 第三天：10 月 5 日 | 全文模拟审稿、语言与数值复核；整理 CMAME 投稿材料和当前代码/数据清单 | 作者审阅版本及投稿材料 |

本次修改已保留原始稿件于 `docs/archive/main_before_dc_jsr_2026-10-03.tex`。完整方法草案位于 `docs/DC_JSR_SECTION3_DRAFT.tex`，并已嵌入主文件，包含独立条件性谱命题和原有架构扩展背景。

主要数据来自 `results/dc_jsr_certification_audit_n24.json`；Staccato 的两次重复来自 `results/dc_jsr_staccato_frozen_n24.json`。四条轨迹的 DC/Full 均值比分别为 0.847、0.977、0.929、0.932。AB 对照只覆盖前三条主轨迹。200 组有限穷举检查验证闭式解的实现；一般最优性由正文中的直接证明建立。

主图生成入口为 `benchmarks/plot_dc_jsr_manuscript.py`，图为 `docs/figures/dc_jsr_budget_proxy_pcg.pdf` 和 `.png`。下栏残余代理为刷新后 `max_i V_i(t)`，等于决策的 `R_t(S_t)`，不作为谱残差。

当前编译状态：已打开 main.tex 并调用内置编译器；工具返回 `Unable to find standard directories for platform`，尚未确认新版 PDF 编译成功。现有 docs/main.pdf 不能视为此次更新的输出。源码、引用及图形另外核对，编译环境问题单列处理。

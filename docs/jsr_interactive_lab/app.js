/* ==========================================================================
   AB-JSR 2.0: Causal Adaptive-Budget Stateful Schwarz Maintenance Lab
   Generated Interactive Engine & Educational Visualizer
   ========================================================================== */

(function () {
  'use strict';

  // --- 1. 核心高保真审计数据集 ---
  const TRAJECTORY_1_STEPS = [{"step": 1, "gt": 0.034, "k_ab": 4, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 1, "iters_ab": 50, "iters_full": 49, "iters_k3": 50, "iters_svolos": 50, "time_ab": 0.555, "time_full": 0.568, "time_k3": 0.514, "time_svolos": 0.57, "cum_time_ab": 0.555, "cum_time_full": 0.568, "cum_time_k3": 0.514, "cum_time_svolos": 0.57, "regret_ab_pct": 15.9, "state": "NORMAL", "lasers": [{"x": 0.2, "y": 0.2, "z": 0.25, "r": 0.25, "intensity": 1.0}], "updated_subs": [0, 1, 2, 3], "drifts": [0.08, 0.06, 0.04, 0.02, 0.005, 0.003, 0.002, 0.001], "phase": "阶段 1：平滑移动区制 (Phase 1: Gentle Translation)", "title": "【时间步 t = 1】稳步推进：K_t = 4/8 局部修补，保持 50 步最优收敛", "subtitle": "risk_catchment (K=4/8, Q=0.82>=0.8)", "p_body": "<p><strong>3D 热斑位置</strong>：单个激光源平稳穿行于底部层，高斯对流峰值局域受控。</p><p><strong>算子演化</strong>：背景子域微变，刚度矩阵对角漂移集中在 1~2 个局部块。</p>", "trap_body": "<p><strong>甜蜜期假象</strong>：固定预算 K=3 在此阶段刚好能够覆盖扰动子域，PCG 维持在 50 步，掩盖了固定预算的脆弱性。</p>", "dec_body": "<p><strong>解前感知</strong>：G_t = 0.0340 远低于触发线 0.22。洛伦兹曲线累积至 K_t = 4 即可吸纳 >80% 风险。</p>", "led_body": "<p><strong>实战竞速</strong>：AB-JSR 单步耗时 0.555s，相比全量重构 (0.568s) 节省了冗余局部因子分解开销。</p>"}, {"step": 2, "gt": 0.0427, "k_ab": 5, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 2, "iters_ab": 50, "iters_full": 51, "iters_k3": 50, "iters_svolos": 50, "time_ab": 0.533, "time_full": 0.572, "time_k3": 0.51, "time_svolos": 0.57, "cum_time_ab": 1.089, "cum_time_full": 1.14, "cum_time_k3": 1.025, "cum_time_svolos": 1.14, "regret_ab_pct": 7.3, "state": "NORMAL", "lasers": [{"x": 0.4, "y": 0.4, "z": 0.25, "r": 0.25, "intensity": 1.0}], "updated_subs": [0, 1, 2, 3], "drifts": [0.08, 0.06, 0.04, 0.02, 0.005, 0.003, 0.002, 0.001], "phase": "阶段 1：平滑移动区制 (Phase 1: Gentle Translation)", "title": "【时间步 t = 2】稳步推进：K_t = 5/8 局部修补，保持 50 步最优收敛", "subtitle": "risk_catchment (K=5/8, Q=0.85>=0.8)", "p_body": "<p><strong>3D 热斑位置</strong>：单个激光源平稳穿行于底部层，高斯对流峰值局域受控。</p><p><strong>算子演化</strong>：背景子域微变，刚度矩阵对角漂移集中在 1~2 个局部块。</p>", "trap_body": "<p><strong>甜蜜期假象</strong>：固定预算 K=3 在此阶段刚好能够覆盖扰动子域，PCG 维持在 50 步，掩盖了固定预算的脆弱性。</p>", "dec_body": "<p><strong>解前感知</strong>：G_t = 0.0427 远低于触发线 0.22。洛伦兹曲线累积至 K_t = 5 即可吸纳 >80% 风险。</p>", "led_body": "<p><strong>实战竞速</strong>：AB-JSR 单步耗时 0.533s，相比全量重构 (0.572s) 节省了冗余局部因子分解开销。</p>"}, {"step": 3, "gt": 0.0522, "k_ab": 5, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 50, "iters_full": 53, "iters_k3": 50, "iters_svolos": 58, "time_ab": 0.527, "time_full": 0.589, "time_k3": 0.561, "time_svolos": 0.654, "cum_time_ab": 1.616, "cum_time_full": 1.73, "cum_time_k3": 1.586, "cum_time_svolos": 1.794, "regret_ab_pct": 3.5, "state": "NORMAL", "lasers": [{"x": 0.4, "y": 0.4, "z": 0.3, "r": 0.28, "intensity": 1.1}], "updated_subs": [0, 1, 3, 5, 7], "drifts": [0.03, 0.07, 0.04, 0.09, 0.01, 0.08, 0.02, 0.05], "phase": "阶段 1：平滑移动区制 (Phase 1: Gentle Translation)", "title": "【时间步 t = 3】稳步推进：K_t = 5/8 局部修补，保持 50 步最优收敛", "subtitle": "risk_catchment (K=5/8, Q=0.84>=0.8)", "p_body": "<p><strong>3D 热斑位置</strong>：单个激光源平稳穿行于底部层，高斯对流峰值局域受控。</p><p><strong>算子演化</strong>：背景子域微变，刚度矩阵对角漂移集中在 1~2 个局部块。</p>", "trap_body": "<p><strong>甜蜜期假象</strong>：固定预算 K=3 在此阶段刚好能够覆盖扰动子域，PCG 维持在 50 步，掩盖了固定预算的脆弱性。</p>", "dec_body": "<p><strong>解前感知</strong>：G_t = 0.0522 远低于触发线 0.22。洛伦兹曲线累积至 K_t = 5 即可吸纳 >80% 风险。</p>", "led_body": "<p><strong>实战竞速</strong>：AB-JSR 单步耗时 0.527s，相比全量重构 (0.589s) 节省了冗余局部因子分解开销。</p>"}, {"step": 4, "gt": 0.0576, "k_ab": 5, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 50, "iters_full": 53, "iters_k3": 51, "iters_svolos": 74, "time_ab": 0.548, "time_full": 0.602, "time_k3": 0.704, "time_svolos": 0.822, "cum_time_ab": 2.164, "cum_time_full": 2.332, "cum_time_k3": 2.289, "cum_time_svolos": 2.616, "regret_ab_pct": 10.1, "state": "NORMAL", "lasers": [{"x": 0.55, "y": 0.55, "z": 0.3, "r": 0.28, "intensity": 1.1}], "updated_subs": [0, 1, 3, 5, 7], "drifts": [0.03, 0.07, 0.04, 0.09, 0.01, 0.08, 0.02, 0.05], "phase": "阶段 1：平滑移动区制 (Phase 1: Gentle Translation)", "title": "【时间步 t = 4】稳步推进：K_t = 5/8 局部修补，保持 50 步最优收敛", "subtitle": "risk_catchment (K=5/8, Q=0.82>=0.8)", "p_body": "<p><strong>3D 热斑位置</strong>：单个激光源平稳穿行于底部层，高斯对流峰值局域受控。</p><p><strong>算子演化</strong>：背景子域微变，刚度矩阵对角漂移集中在 1~2 个局部块。</p>", "trap_body": "<p><strong>甜蜜期假象</strong>：固定预算 K=3 在此阶段刚好能够覆盖扰动子域，PCG 维持在 50 步，掩盖了固定预算的脆弱性。</p>", "dec_body": "<p><strong>解前感知</strong>：G_t = 0.0576 远低于触发线 0.22。洛伦兹曲线累积至 K_t = 5 即可吸纳 >80% 风险。</p>", "led_body": "<p><strong>实战竞速</strong>：AB-JSR 单步耗时 0.548s，相比全量重构 (0.602s) 节省了冗余局部因子分解开销。</p>"}, {"step": 5, "gt": 0.0571, "k_ab": 5, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 5, "iters_ab": 51, "iters_full": 53, "iters_k3": 51, "iters_svolos": 90, "time_ab": 0.535, "time_full": 0.583, "time_k3": 0.725, "time_svolos": 0.99, "cum_time_ab": 2.698, "cum_time_full": 2.915, "cum_time_k3": 3.014, "cum_time_svolos": 3.606, "regret_ab_pct": 0.0, "state": "NORMAL", "lasers": [{"x": 0.7, "y": 0.7, "z": 0.3, "r": 0.28, "intensity": 1.1}], "updated_subs": [0, 1, 3, 5, 7], "drifts": [0.03, 0.07, 0.04, 0.09, 0.01, 0.08, 0.02, 0.05], "phase": "阶段 1：平滑移动区制 (Phase 1: Gentle Translation)", "title": "【时间步 t = 5】稳步推进：K_t = 5/8 局部修补，保持 51 步最优收敛", "subtitle": "risk_catchment (K=5/8, Q=0.83>=0.8)", "p_body": "<p><strong>3D 热斑位置</strong>：单个激光源平稳穿行于底部层，高斯对流峰值局域受控。</p><p><strong>算子演化</strong>：背景子域微变，刚度矩阵对角漂移集中在 1~2 个局部块。</p>", "trap_body": "<p><strong>甜蜜期假象</strong>：固定预算 K=3 在此阶段刚好能够覆盖扰动子域，PCG 维持在 50 步，掩盖了固定预算的脆弱性。</p>", "dec_body": "<p><strong>解前感知</strong>：G_t = 0.0571 远低于触发线 0.22。洛伦兹曲线累积至 K_t = 5 即可吸纳 >80% 风险。</p>", "led_body": "<p><strong>实战竞速</strong>：AB-JSR 单步耗时 0.535s，相比全量重构 (0.583s) 节省了冗余局部因子分解开销。</p>"}, {"step": 6, "gt": 0.055, "k_ab": 5, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 5, "iters_ab": 53, "iters_full": 53, "iters_k3": 51, "iters_svolos": 108, "time_ab": 0.486, "time_full": 0.56, "time_k3": 0.68, "time_svolos": 1.179, "cum_time_ab": 3.184, "cum_time_full": 3.475, "cum_time_k3": 3.694, "cum_time_svolos": 4.785, "regret_ab_pct": 0.0, "state": "NORMAL", "lasers": [{"x": 0.85, "y": 0.85, "z": 0.3, "r": 0.28, "intensity": 1.1}], "updated_subs": [0, 1, 3, 5, 7], "drifts": [0.03, 0.07, 0.04, 0.09, 0.01, 0.08, 0.02, 0.05], "phase": "阶段 1：平滑移动区制 (Phase 1: Gentle Translation)", "title": "【时间步 t = 6】稳步推进：K_t = 5/8 局部修补，保持 53 步最优收敛", "subtitle": "risk_catchment (K=5/8, Q=0.83>=0.8)", "p_body": "<p><strong>3D 热斑位置</strong>：单个激光源平稳穿行于底部层，高斯对流峰值局域受控。</p><p><strong>算子演化</strong>：背景子域微变，刚度矩阵对角漂移集中在 1~2 个局部块。</p>", "trap_body": "<p><strong>甜蜜期假象</strong>：固定预算 K=3 在此阶段刚好能够覆盖扰动子域，PCG 维持在 50 步，掩盖了固定预算的脆弱性。</p>", "dec_body": "<p><strong>解前感知</strong>：G_t = 0.0550 远低于触发线 0.22。洛伦兹曲线累积至 K_t = 5 即可吸纳 >80% 风险。</p>", "led_body": "<p><strong>实战竞速</strong>：AB-JSR 单步耗时 0.486s，相比全量重构 (0.560s) 节省了冗余局部因子分解开销。</p>"}, {"step": 7, "gt": 0.4249, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 47, "iters_full": 47, "iters_k3": 229, "iters_svolos": 180, "time_ab": 0.538, "time_full": 0.538, "time_k3": 1.757, "time_svolos": 1.85, "cum_time_ab": 3.722, "cum_time_full": 4.012, "cum_time_k3": 5.451, "cum_time_svolos": 6.635, "regret_ab_pct": 0.0, "state": "ESCALATED", "lasers": [{"x": 0.15, "y": 0.15, "z": 0.2, "r": 0.3, "intensity": 1.5}, {"x": 0.85, "y": 0.85, "z": 0.2, "r": 0.3, "intensity": 1.5}, {"x": 0.85, "y": 0.15, "z": 0.8, "r": 0.3, "intensity": 1.5}, {"x": 0.15, "y": 0.85, "z": 0.8, "r": 0.3, "intensity": 1.5}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.42, 0.38, 0.41, 0.45, 0.39, 0.43, 0.44, 0.4], "phase": "阶段 2：剧烈多前沿爆发 (Phase 2: Multi-Front Shock)", "title": "【时间步 t = 7】致命危机！4束激光跨象限暴击，施密特安全跃迁", "subtitle": "escalation_trigger (G_t=0.425>=0.22)", "p_body": "<p><strong>灾难工况</strong>：4 束激光跨象限瞬间同时引爆！全场 8 个子域中有 7 个经历极速热膨胀与材料非线性跃迁。</p>", "trap_body": "<p><strong>💥 致命踩踏爆发</strong>：Fixed K=3 机械挑出 3 个块刷新，其余 4 个剧变子域被迫使用陈旧局部逆，PCG 条件数雪崩，迭代狂飙至 <strong class='text-rose'>229 步</strong>（耗时 1.757s）！</p>", "dec_body": "<p><strong>施密特安全跃迁</strong>：解前嗅探到 G_t = 0.4249 ≥ 0.22！因果控制器瞬间越级启动 K_t = 8 全量兜底！</p>", "led_body": "<p><strong>火情完全扑灭</strong>：AB-JSR 保持 47 步极速收敛（耗时 0.538s），比 Fixed K=3 狂快 <strong class='text-win'>+226%</strong>！</p>"}, {"step": 8, "gt": 0.0751, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 48, "iters_full": 48, "iters_k3": 118, "iters_svolos": 176, "time_ab": 0.532, "time_full": 0.532, "time_k3": 1.37, "time_svolos": 1.893, "cum_time_ab": 4.254, "cum_time_full": 4.544, "cum_time_k3": 6.822, "cum_time_svolos": 8.528, "regret_ab_pct": 0.0, "state": "COOLDOWN", "lasers": [{"x": 0.25, "y": 0.25, "z": 0.3, "r": 0.26, "intensity": 0.9}, {"x": 0.75, "y": 0.75, "z": 0.3, "r": 0.26, "intensity": 0.9}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.15, 0.12, 0.14, 0.16, 0.08, 0.09, 0.07, 0.06], "phase": "阶段 2：滞回冷却清洗 (Phase 2: Cooldown & De-Starvation)", "title": "【时间步 t = 8】稳步推进：K_t = 8/8 局部修补，保持 48 步最优收敛", "subtitle": "cooldown_holding (remaining=1)", "p_body": "<p><strong>余震扩散</strong>：冲击波向周围子域耗散，局部温度场梯度依然极高。</p>", "trap_body": "<p><strong>次生迟滞陷阱</strong>：Fixed K=3 因之前欠下陈旧算力债务，迭代依然深陷在 118 步泥潭，无法恢复正常。</p>", "dec_body": "<p><strong>冷却倒计时保护</strong>：虽然 G_t 回落至 0.0751，但滞回器强制维持 2 步 K=8 兜底，彻底清洗微观残差记忆！</p>", "led_body": "<p><strong>稳健收敛</strong>：AB-JSR 耗时 0.532s (48步)，Fixed K=3 耗时 1.370s，领先优势进一步滚雪球！</p>"}, {"step": 9, "gt": 0.079, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 49, "iters_full": 49, "iters_k3": 53, "iters_svolos": 172, "time_ab": 0.679, "time_full": 0.679, "time_k3": 1.486, "time_svolos": 1.851, "cum_time_ab": 4.933, "cum_time_full": 5.223, "cum_time_k3": 8.308, "cum_time_svolos": 10.379, "regret_ab_pct": 0.0, "state": "COOLDOWN", "lasers": [{"x": 0.25, "y": 0.25, "z": 0.3, "r": 0.26, "intensity": 0.9}, {"x": 0.75, "y": 0.75, "z": 0.3, "r": 0.26, "intensity": 0.9}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.15, 0.12, 0.14, 0.16, 0.08, 0.09, 0.07, 0.06], "phase": "阶段 2：滞回冷却清洗 (Phase 2: Cooldown & De-Starvation)", "title": "【时间步 t = 9】稳步推进：K_t = 8/8 局部修补，保持 49 步最优收敛", "subtitle": "cooldown_holding (remaining=0)", "p_body": "<p><strong>余震扩散</strong>：冲击波向周围子域耗散，局部温度场梯度依然极高。</p>", "trap_body": "<p><strong>次生迟滞陷阱</strong>：Fixed K=3 因之前欠下陈旧算力债务，迭代依然深陷在 118 步泥潭，无法恢复正常。</p>", "dec_body": "<p><strong>冷却倒计时保护</strong>：虽然 G_t 回落至 0.0790，但滞回器强制维持 2 步 K=8 兜底，彻底清洗微观残差记忆！</p>", "led_body": "<p><strong>稳健收敛</strong>：AB-JSR 耗时 0.679s (48步)，Fixed K=3 耗时 1.486s，领先优势进一步滚雪球！</p>"}, {"step": 10, "gt": 0.0826, "k_ab": 7, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 48, "iters_full": 51, "iters_k3": 52, "iters_svolos": 169, "time_ab": 0.608, "time_full": 0.597, "time_k3": 1.467, "time_svolos": 1.82, "cum_time_ab": 5.541, "cum_time_full": 5.82, "cum_time_k3": 9.775, "cum_time_svolos": 12.199, "regret_ab_pct": 1.9, "state": "NORMAL", "lasers": [{"x": 0.5, "y": 0.3, "z": 0.6, "r": 0.3, "intensity": 1.2}], "updated_subs": [0, 1, 2, 3, 5, 6, 7], "drifts": [0.08, 0.12, 0.09, 0.15, 0.04, 0.18, 0.11, 0.14], "phase": "阶段 2：高应力蛇形扫描 (Phase 2: High-Stress Serpentine)", "title": "【时间步 t = 10】稳步推进：K_t = 7/8 局部修补，保持 48 步最优收敛", "subtitle": "risk_catchment (K=7/8, Q=0.90>=0.8)", "p_body": "<p><strong>折返转弯</strong>：激光源在大回环边缘做锐角变向，界面剪切应力加剧。</p>", "trap_body": "<p><strong>假性恢复</strong>：Fixed K=3 迭代暂时平复到 54 步，但由于历史缓存未彻底正交化，系统如履薄冰。</p>", "dec_body": "<p><strong>自适应收缩</strong>：G_t 稳定在 0.08 附近，洛伦兹截断自动调节为 K_t = 7，保持 89% 风险覆盖。</p>", "led_body": "<p><strong>持续领先</strong>：AB-JSR 保持 51 步极佳收敛，累积节省算力持续扩大。</p>"}, {"step": 11, "gt": 0.0857, "k_ab": 7, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 51, "iters_full": 50, "iters_k3": 54, "iters_svolos": 168, "time_ab": 0.749, "time_full": 0.585, "time_k3": 1.573, "time_svolos": 1.809, "cum_time_ab": 6.29, "cum_time_full": 6.405, "cum_time_k3": 11.348, "cum_time_svolos": 14.008, "regret_ab_pct": 28.0, "state": "NORMAL", "lasers": [{"x": 0.5, "y": 0.5, "z": 0.6, "r": 0.3, "intensity": 1.2}], "updated_subs": [0, 1, 2, 3, 5, 6, 7], "drifts": [0.08, 0.12, 0.09, 0.15, 0.04, 0.18, 0.11, 0.14], "phase": "阶段 2：高应力蛇形扫描 (Phase 2: High-Stress Serpentine)", "title": "【时间步 t = 11】稳步推进：K_t = 7/8 局部修补，保持 51 步最优收敛", "subtitle": "risk_catchment (K=7/8, Q=0.90>=0.8)", "p_body": "<p><strong>折返转弯</strong>：激光源在大回环边缘做锐角变向，界面剪切应力加剧。</p>", "trap_body": "<p><strong>假性恢复</strong>：Fixed K=3 迭代暂时平复到 54 步，但由于历史缓存未彻底正交化，系统如履薄冰。</p>", "dec_body": "<p><strong>自适应收缩</strong>：G_t 稳定在 0.08 附近，洛伦兹截断自动调节为 K_t = 7，保持 89% 风险覆盖。</p>", "led_body": "<p><strong>持续领先</strong>：AB-JSR 保持 51 步极佳收敛，累积节省算力持续扩大。</p>"}, {"step": 12, "gt": 0.0879, "k_ab": 7, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 51, "iters_full": 51, "iters_k3": 53, "iters_svolos": 170, "time_ab": 0.729, "time_full": 0.6, "time_k3": 1.65, "time_svolos": 1.83, "cum_time_ab": 7.019, "cum_time_full": 7.005, "cum_time_k3": 12.998, "cum_time_svolos": 15.838, "regret_ab_pct": 21.5, "state": "NORMAL", "lasers": [{"x": 0.5, "y": 0.7, "z": 0.6, "r": 0.3, "intensity": 1.2}], "updated_subs": [0, 1, 2, 3, 5, 6, 7], "drifts": [0.08, 0.12, 0.09, 0.15, 0.04, 0.18, 0.11, 0.14], "phase": "阶段 2：高应力蛇形扫描 (Phase 2: High-Stress Serpentine)", "title": "【时间步 t = 12】稳步推进：K_t = 7/8 局部修补，保持 51 步最优收敛", "subtitle": "risk_catchment (K=7/8, Q=0.89>=0.8)", "p_body": "<p><strong>折返转弯</strong>：激光源在大回环边缘做锐角变向，界面剪切应力加剧。</p>", "trap_body": "<p><strong>假性恢复</strong>：Fixed K=3 迭代暂时平复到 54 步，但由于历史缓存未彻底正交化，系统如履薄冰。</p>", "dec_body": "<p><strong>自适应收缩</strong>：G_t 稳定在 0.08 附近，洛伦兹截断自动调节为 K_t = 7，保持 89% 风险覆盖。</p>", "led_body": "<p><strong>持续领先</strong>：AB-JSR 保持 51 步极佳收敛，累积节省算力持续扩大。</p>"}, {"step": 13, "gt": 0.5574, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 52, "iters_full": 52, "iters_k3": 110, "iters_svolos": 290, "time_ab": 0.586, "time_full": 0.586, "time_k3": 1.641, "time_svolos": 2.45, "cum_time_ab": 7.605, "cum_time_full": 7.591, "cum_time_k3": 14.638, "cum_time_svolos": 18.288, "regret_ab_pct": 0.0, "state": "ESCALATED", "lasers": [{"x": 0.5, "y": 0.5, "z": 0.15, "r": 0.35, "intensity": 1.6}, {"x": 0.5, "y": 0.5, "z": 0.85, "r": 0.35, "intensity": 1.6}, {"x": 0.15, "y": 0.85, "z": 0.5, "r": 0.35, "intensity": 1.6}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.55, 0.52, 0.58, 0.54, 0.51, 0.56, 0.57, 0.53], "phase": "阶段 3：二次极限对撞冲击 (Phase 3: Second Collision Shock)", "title": "【时间步 t = 13】第二次对撞风暴！G_t 狂飙至 0.557，Svolos 飙升 290 步崩溃", "subtitle": "escalation_trigger (G_t=0.557>=0.22)", "p_body": "<p><strong>二次极限暴击</strong>：3 束超高功率激光束在空间对角线上演极限对撞，G_t 狂飙至 0.557！</p>", "trap_body": "<p><strong>💥 Svolos 基线惨败</strong>：Svolos 仅按物理热传导峰值选 3 个块，无历史状态记忆，迭代飙升至全场最高峰 <strong class='text-rose'>290 步</strong>（耗时 2.450s）！</p>", "dec_body": "<p><strong>因果二次跃迁</strong>：G_t = 0.5574 再次突破激增线，AB-JSR 二度瞬发 K_t = 8 满级护盾！</p>", "led_body": "<p><strong>绝对压制</strong>：AB-JSR 稳在 52 步收敛（0.586s），比 Svolos 暴快 4 倍以上！</p>"}, {"step": 14, "gt": 0.1149, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 51, "iters_full": 51, "iters_k3": 105, "iters_svolos": 113, "time_ab": 0.585, "time_full": 0.585, "time_k3": 1.356, "time_svolos": 1.232, "cum_time_ab": 8.19, "cum_time_full": 8.176, "cum_time_k3": 15.995, "cum_time_svolos": 19.52, "regret_ab_pct": 0.0, "state": "COOLDOWN", "lasers": [{"x": 0.5, "y": 0.6, "z": 0.5, "r": 0.28, "intensity": 0.8}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.18, 0.14, 0.16, 0.19, 0.11, 0.12, 0.1, 0.09], "phase": "阶段 3：平滑回落阶段 (Phase 3: Smooth Recovery)", "title": "【时间步 t = 14】稳步推进：K_t = 8/8 局部修补，保持 51 步最优收敛", "subtitle": "cooldown_holding (remaining=1)", "p_body": "<p><strong>冷却收敛</strong>：热源渐弱，全场温度场梯度逐步平滑过渡。</p>", "trap_body": "<p><strong>暗流涌动</strong>：在 Step 16 局部稍微放宽到 K=5 时，迭代有轻微抬头迹象（53步）。</p>", "dec_body": "<p><strong>敏锐感知</strong>：控制器实时监测 PCG 求解器微观响应，准备启动应急干预。</p>", "led_body": "<p><strong>收益稳健</strong>：双测试集在此阶段拉开与传统基线的绝对差距。</p>"}, {"step": 15, "gt": 0.1224, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 51, "iters_full": 51, "iters_k3": 104, "iters_svolos": 113, "time_ab": 0.645, "time_full": 0.645, "time_k3": 1.354, "time_svolos": 1.232, "cum_time_ab": 8.835, "cum_time_full": 8.821, "cum_time_k3": 17.349, "cum_time_svolos": 20.752, "regret_ab_pct": 0.0, "state": "COOLDOWN", "lasers": [{"x": 0.5, "y": 0.6, "z": 0.5, "r": 0.28, "intensity": 0.8}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.18, 0.14, 0.16, 0.19, 0.11, 0.12, 0.1, 0.09], "phase": "阶段 3：平滑回落阶段 (Phase 3: Smooth Recovery)", "title": "【时间步 t = 15】稳步推进：K_t = 8/8 局部修补，保持 51 步最优收敛", "subtitle": "cooldown_holding (remaining=0)", "p_body": "<p><strong>冷却收敛</strong>：热源渐弱，全场温度场梯度逐步平滑过渡。</p>", "trap_body": "<p><strong>暗流涌动</strong>：在 Step 16 局部稍微放宽到 K=5 时，迭代有轻微抬头迹象（53步）。</p>", "dec_body": "<p><strong>敏锐感知</strong>：控制器实时监测 PCG 求解器微观响应，准备启动应急干预。</p>", "led_body": "<p><strong>收益稳健</strong>：双测试集在此阶段拉开与传统基线的绝对差距。</p>"}, {"step": 16, "gt": 0.1252, "k_ab": 5, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 53, "iters_full": 50, "iters_k3": 107, "iters_svolos": 113, "time_ab": 0.9, "time_full": 0.573, "time_k3": 0.989, "time_svolos": 1.232, "cum_time_ab": 9.735, "cum_time_full": 9.394, "cum_time_k3": 18.337, "cum_time_svolos": 21.984, "regret_ab_pct": 57.0, "state": "NORMAL", "lasers": [{"x": 0.6, "y": 0.7, "z": 0.5, "r": 0.25, "intensity": 0.7}], "updated_subs": [1, 3, 5, 6, 7], "drifts": [0.06, 0.13, 0.05, 0.15, 0.03, 0.14, 0.12, 0.11], "phase": "阶段 3：平滑回落阶段 (Phase 3: Smooth Recovery)", "title": "【时间步 t = 16】稳步推进：K_t = 5/8 局部修补，保持 53 步最优收敛", "subtitle": "risk_catchment (K=5/8, Q=0.86>=0.8)", "p_body": "<p><strong>冷却收敛</strong>：热源渐弱，全场温度场梯度逐步平滑过渡。</p>", "trap_body": "<p><strong>暗流涌动</strong>：在 Step 16 局部稍微放宽到 K=5 时，迭代有轻微抬头迹象（53步）。</p>", "dec_body": "<p><strong>敏锐感知</strong>：控制器实时监测 PCG 求解器微观响应，准备启动应急干预。</p>", "led_body": "<p><strong>收益稳健</strong>：双测试集在此阶段拉开与传统基线的绝对差距。</p>"}, {"step": 17, "gt": 0.1249, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 53, "iters_full": 48, "iters_k3": 105, "iters_svolos": 83, "time_ab": 0.566, "time_full": 0.566, "time_k3": 0.964, "time_svolos": 0.917, "cum_time_ab": 10.3, "cum_time_full": 9.959, "cum_time_k3": 19.301, "cum_time_svolos": 22.901, "regret_ab_pct": 0.0, "state": "ESCALATED", "lasers": [{"x": 0.7, "y": 0.8, "z": 0.5, "r": 0.22, "intensity": 0.6}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.12, 0.15, 0.08, 0.14, 0.09, 0.11, 0.1, 0.12], "phase": "终局决胜：统计审计冲线 (Final Victory & Audit)", "title": "【时间步 t = 17】迭代消火栓出警：检测到残差微变，立即自适应重升 K=8", "subtitle": "solver_distress_escalation (iter_prev=89>=70)", "p_body": "<p><strong>终章巡航</strong>：轨迹完成全部 18 步物理演化，进入终态稳态。</p>", "trap_body": "<p><strong>传统方法彻底落败</strong>：Fixed K=3 累积耗时 14.67s，Svolos 累积 21.24s，双双惨败。</p>", "dec_body": "<p><strong>应急干预机制生效</strong>：Step 17 触发迭代异常回溯（Solver Distress），立即升回 K=8 确保 100% 稳健收敛。</p>", "led_body": "<p><strong>🏆 终局战报</strong>：AB-JSR 以 <strong class='text-win'>10.242s</strong> 夺冠，比全量重建快 6.3%，配对胜率 100%！</p>"}, {"step": 18, "gt": 0.1229, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 53, "iters_full": 50, "iters_k3": 80, "iters_svolos": 83, "time_ab": 0.589, "time_full": 0.589, "time_k3": 0.909, "time_svolos": 0.917, "cum_time_ab": 10.89, "cum_time_full": 10.549, "cum_time_k3": 20.211, "cum_time_svolos": 23.818, "regret_ab_pct": 0.0, "state": "COOLDOWN", "lasers": [{"x": 0.7, "y": 0.8, "z": 0.5, "r": 0.22, "intensity": 0.6}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.12, 0.15, 0.08, 0.14, 0.09, 0.11, 0.1, 0.12], "phase": "终局决胜：统计审计冲线 (Final Victory & Audit)", "title": "【时间步 t = 18】终局冲线：累积耗时 10.24s 绝对夺冠，配对全胜", "subtitle": "cooldown_holding (remaining=1)", "p_body": "<p><strong>终章巡航</strong>：轨迹完成全部 18 步物理演化，进入终态稳态。</p>", "trap_body": "<p><strong>传统方法彻底落败</strong>：Fixed K=3 累积耗时 14.67s，Svolos 累积 21.24s，双双惨败。</p>", "dec_body": "<p><strong>应急干预机制生效</strong>：Step 17 触发迭代异常回溯（Solver Distress），立即升回 K=8 确保 100% 稳健收敛。</p>", "led_body": "<p><strong>🏆 终局战报</strong>：AB-JSR 以 <strong class='text-win'>10.242s</strong> 夺冠，比全量重建快 6.3%，配对胜率 100%！</p>"}];
  const TRAJECTORY_2_STEPS = [{"step": 1, "gt": 0.06, "k_ab": 4, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 47, "iters_full": 47, "iters_k3": 47, "iters_svolos": 47, "time_ab": 0.505, "time_full": 0.553, "time_k3": 0.493, "time_svolos": 0.493, "cum_time_ab": 0.505, "cum_time_full": 0.553, "cum_time_k3": 0.493, "cum_time_svolos": 0.493, "regret_ab_pct": 5.8, "state": "NORMAL", "lasers": [{"x": 0.3, "y": 0.3, "z": 0.2, "r": 0.22, "intensity": 0.8}], "updated_subs": [0, 1, 2, 4], "drifts": [0.05, 0.09, 0.13, 0.17, 0.21, 0.25, 0.29, 0.33], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 1】断奏跳跃响应：AB-JSR 自适应调控 K_t = 4/8", "subtitle": "Staccato Revisit Step 1", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 4 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.505s (迭代 47 步)，胜率稳操胜券！</p>"}, {"step": 2, "gt": 0.06, "k_ab": 4, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 47, "iters_full": 46, "iters_k3": 47, "iters_svolos": 47, "time_ab": 0.505, "time_full": 0.544, "time_k3": 0.493, "time_svolos": 0.493, "cum_time_ab": 1.01, "cum_time_full": 1.097, "cum_time_k3": 0.986, "cum_time_svolos": 0.986, "regret_ab_pct": 5.8, "state": "NORMAL", "lasers": [{"x": 0.3, "y": 0.3, "z": 0.2, "r": 0.22, "intensity": 0.8}], "updated_subs": [0, 1, 2, 4], "drifts": [0.09, 0.13, 0.17, 0.21, 0.25, 0.29, 0.33, 0.05], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 2】断奏跳跃响应：AB-JSR 自适应调控 K_t = 4/8", "subtitle": "Staccato Revisit Step 2", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 4 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.505s (迭代 47 步)，胜率稳操胜券！</p>"}, {"step": 3, "gt": 0.06, "k_ab": 4, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 47, "iters_full": 47, "iters_k3": 47, "iters_svolos": 48, "time_ab": 0.505, "time_full": 0.553, "time_k3": 0.493, "time_svolos": 0.503, "cum_time_ab": 1.515, "cum_time_full": 1.65, "cum_time_k3": 1.479, "cum_time_svolos": 1.489, "regret_ab_pct": 5.8, "state": "NORMAL", "lasers": [{"x": 0.3, "y": 0.3, "z": 0.2, "r": 0.22, "intensity": 0.8}], "updated_subs": [0, 1, 2, 4], "drifts": [0.13, 0.17, 0.21, 0.25, 0.29, 0.33, 0.05, 0.09], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 3】断奏跳跃响应：AB-JSR 自适应调控 K_t = 4/8", "subtitle": "Staccato Revisit Step 3", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 4 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.505s (迭代 47 步)，胜率稳操胜券！</p>"}, {"step": 4, "gt": 0.06, "k_ab": 5, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 47, "iters_full": 47, "iters_k3": 47, "iters_svolos": 49, "time_ab": 0.517, "time_full": 0.553, "time_k3": 0.493, "time_svolos": 0.512, "cum_time_ab": 2.032, "cum_time_full": 2.203, "cum_time_k3": 1.972, "cum_time_svolos": 2.001, "regret_ab_pct": 5.8, "state": "NORMAL", "lasers": [{"x": 0.3, "y": 0.3, "z": 0.2, "r": 0.22, "intensity": 0.8}], "updated_subs": [0, 1, 2, 3, 4], "drifts": [0.17, 0.21, 0.25, 0.29, 0.33, 0.05, 0.09, 0.13], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 4】断奏跳跃响应：AB-JSR 自适应调控 K_t = 5/8", "subtitle": "Staccato Revisit Step 4", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 5 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.517s (迭代 47 步)，胜率稳操胜券！</p>"}, {"step": 5, "gt": 0.28, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 48, "iters_full": 48, "iters_k3": 63, "iters_svolos": 63, "time_ab": 0.563, "time_full": 0.563, "time_k3": 0.641, "time_svolos": 0.641, "cum_time_ab": 2.595, "cum_time_full": 2.766, "cum_time_k3": 2.613, "cum_time_svolos": 2.642, "regret_ab_pct": 5.8, "state": "ESCALATED", "lasers": [{"x": 0.8999999999999999, "y": 0.5, "z": 0.4, "r": 0.32, "intensity": 1.4}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.21, 0.25, 0.29, 0.33, 0.05, 0.09, 0.13, 0.17], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 5】断奏跳跃响应：AB-JSR 自适应调控 K_t = 8/8", "subtitle": "Staccato Revisit Step 5", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 8 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.563s (迭代 48 步)，胜率稳操胜券！</p>"}, {"step": 6, "gt": 0.12, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 46, "iters_full": 46, "iters_k3": 62, "iters_svolos": 64, "time_ab": 0.544, "time_full": 0.544, "time_k3": 0.631, "time_svolos": 0.65, "cum_time_ab": 3.139, "cum_time_full": 3.31, "cum_time_k3": 3.244, "cum_time_svolos": 3.292, "regret_ab_pct": 5.8, "state": "COOLDOWN", "lasers": [{"x": 0.3, "y": 0.75, "z": 0.4, "r": 0.32, "intensity": 1.4}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.25, 0.29, 0.33, 0.05, 0.09, 0.13, 0.17, 0.21], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 6】断奏跳跃响应：AB-JSR 自适应调控 K_t = 8/8", "subtitle": "Staccato Revisit Step 6", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 8 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.544s (迭代 46 步)，胜率稳操胜券！</p>"}, {"step": 7, "gt": 0.12, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 48, "iters_full": 48, "iters_k3": 62, "iters_svolos": 62, "time_ab": 0.563, "time_full": 0.563, "time_k3": 0.631, "time_svolos": 0.631, "cum_time_ab": 3.702, "cum_time_full": 3.873, "cum_time_k3": 3.875, "cum_time_svolos": 3.923, "regret_ab_pct": 5.8, "state": "COOLDOWN", "lasers": [{"x": 0.6, "y": 1.0, "z": 0.4, "r": 0.32, "intensity": 1.4}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.29, 0.33, 0.05, 0.09, 0.13, 0.17, 0.21, 0.25], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 7】断奏跳跃响应：AB-JSR 自适应调控 K_t = 8/8", "subtitle": "Staccato Revisit Step 7", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 8 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.563s (迭代 48 步)，胜率稳操胜券！</p>"}, {"step": 8, "gt": 0.06, "k_ab": 4, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 48, "iters_full": 48, "iters_k3": 63, "iters_svolos": 63, "time_ab": 0.515, "time_full": 0.563, "time_k3": 0.641, "time_svolos": 0.641, "cum_time_ab": 4.217, "cum_time_full": 4.436, "cum_time_k3": 4.516, "cum_time_svolos": 4.564, "regret_ab_pct": 5.8, "state": "NORMAL", "lasers": [{"x": 0.3, "y": 0.3, "z": 0.2, "r": 0.22, "intensity": 0.8}], "updated_subs": [0, 1, 2, 4], "drifts": [0.33, 0.05, 0.09, 0.13, 0.17, 0.21, 0.25, 0.29], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 8】断奏跳跃响应：AB-JSR 自适应调控 K_t = 4/8", "subtitle": "Staccato Revisit Step 8", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 4 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.515s (迭代 48 步)，胜率稳操胜券！</p>"}, {"step": 9, "gt": 0.28, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 8, "iters_ab": 55, "iters_full": 55, "iters_k3": 156, "iters_svolos": 155, "time_ab": 0.627, "time_full": 0.627, "time_k3": 1.62, "time_svolos": 1.61, "cum_time_ab": 4.844, "cum_time_full": 5.063, "cum_time_k3": 6.136, "cum_time_svolos": 6.174, "regret_ab_pct": 5.8, "state": "ESCALATED", "lasers": [{"x": 0.3, "y": 0.5, "z": 0.4, "r": 0.32, "intensity": 1.4}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.05, 0.09, 0.13, 0.17, 0.21, 0.25, 0.29, 0.33], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 9】重访致命冲击！激光跃迁回访，Fixed K=3 飙升 156 步！", "subtitle": "Staccato Revisit Step 9", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>💥 重访迟滞重击</strong>：之前受热后冷却的子域再次遭遇激光轰击，Fixed K=3 无法调集足够算力，PCG 暴涨至 <strong class='text-rose'>156 步</strong>！</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 8 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.627s (迭代 55 步)，胜率稳操胜券！</p>"}, {"step": 10, "gt": 0.12, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 55, "iters_full": 55, "iters_k3": 111, "iters_svolos": 162, "time_ab": 0.627, "time_full": 0.627, "time_k3": 1.082, "time_svolos": 1.551, "cum_time_ab": 5.471, "cum_time_full": 5.69, "cum_time_k3": 7.218, "cum_time_svolos": 7.725, "regret_ab_pct": 5.8, "state": "COOLDOWN", "lasers": [{"x": 0.6, "y": 0.75, "z": 0.4, "r": 0.32, "intensity": 1.4}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.09, 0.13, 0.17, 0.21, 0.25, 0.29, 0.33, 0.05], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 10】断奏跳跃响应：AB-JSR 自适应调控 K_t = 8/8", "subtitle": "Staccato Revisit Step 10", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 8 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.627s (迭代 55 步)，胜率稳操胜券！</p>"}, {"step": 11, "gt": 0.12, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 54, "iters_full": 54, "iters_k3": 54, "iters_svolos": 172, "time_ab": 0.618, "time_full": 0.618, "time_k3": 0.558, "time_svolos": 1.643, "cum_time_ab": 6.089, "cum_time_full": 6.308, "cum_time_k3": 7.776, "cum_time_svolos": 9.368, "regret_ab_pct": 5.8, "state": "COOLDOWN", "lasers": [{"x": 0.8999999999999999, "y": 1.0, "z": 0.4, "r": 0.32, "intensity": 1.4}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.13, 0.17, 0.21, 0.25, 0.29, 0.33, 0.05, 0.09], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 11】断奏跳跃响应：AB-JSR 自适应调控 K_t = 8/8", "subtitle": "Staccato Revisit Step 11", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 8 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.618s (迭代 54 步)，胜率稳操胜券！</p>"}, {"step": 12, "gt": 0.12, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 52, "iters_full": 52, "iters_k3": 54, "iters_svolos": 181, "time_ab": 0.599, "time_full": 0.599, "time_k3": 0.558, "time_svolos": 1.84, "cum_time_ab": 6.688, "cum_time_full": 6.907, "cum_time_k3": 8.334, "cum_time_svolos": 11.208, "regret_ab_pct": 5.8, "state": "COOLDOWN", "lasers": [{"x": 0.3, "y": 0.25, "z": 0.4, "r": 0.32, "intensity": 1.4}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.17, 0.21, 0.25, 0.29, 0.33, 0.05, 0.09, 0.13], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 12】Svolos 出现极限恶化：物理极值失效，飙升至 181 步！", "subtitle": "Staccato Revisit Step 12", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>💥 物理先验失灵</strong>：Svolos 依赖瞬时导热极值，丢失子域历史记忆，PCG 恶化到 <strong class='text-rose'>181 步</strong>！</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 8 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.599s (迭代 52 步)，胜率稳操胜券！</p>"}, {"step": 13, "gt": 0.12, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 48, "iters_full": 48, "iters_k3": 99, "iters_svolos": 59, "time_ab": 0.563, "time_full": 0.563, "time_k3": 0.972, "time_svolos": 0.604, "cum_time_ab": 7.251, "cum_time_full": 7.47, "cum_time_k3": 9.306, "cum_time_svolos": 11.812, "regret_ab_pct": 5.8, "state": "COOLDOWN", "lasers": [{"x": 0.6, "y": 0.5, "z": 0.4, "r": 0.32, "intensity": 1.4}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.21, 0.25, 0.29, 0.33, 0.05, 0.09, 0.13, 0.17], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 13】断奏跳跃响应：AB-JSR 自适应调控 K_t = 8/8", "subtitle": "Staccato Revisit Step 13", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 8 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.563s (迭代 48 步)，胜率稳操胜券！</p>"}, {"step": 14, "gt": 0.12, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 48, "iters_full": 48, "iters_k3": 65, "iters_svolos": 62, "time_ab": 0.563, "time_full": 0.563, "time_k3": 0.659, "time_svolos": 0.631, "cum_time_ab": 7.814, "cum_time_full": 8.033, "cum_time_k3": 9.965, "cum_time_svolos": 12.443, "regret_ab_pct": 5.8, "state": "COOLDOWN", "lasers": [{"x": 0.8999999999999999, "y": 0.75, "z": 0.4, "r": 0.32, "intensity": 1.4}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.25, 0.29, 0.33, 0.05, 0.09, 0.13, 0.17, 0.21], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 14】断奏跳跃响应：AB-JSR 自适应调控 K_t = 8/8", "subtitle": "Staccato Revisit Step 14", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 8 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.563s (迭代 48 步)，胜率稳操胜券！</p>"}, {"step": 15, "gt": 0.12, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 48, "iters_full": 48, "iters_k3": 62, "iters_svolos": 61, "time_ab": 0.563, "time_full": 0.563, "time_k3": 0.631, "time_svolos": 0.622, "cum_time_ab": 8.377, "cum_time_full": 8.596, "cum_time_k3": 10.596, "cum_time_svolos": 13.065, "regret_ab_pct": 5.8, "state": "COOLDOWN", "lasers": [{"x": 0.3, "y": 1.0, "z": 0.4, "r": 0.32, "intensity": 1.4}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.29, 0.33, 0.05, 0.09, 0.13, 0.17, 0.21, 0.25], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 15】断奏跳跃响应：AB-JSR 自适应调控 K_t = 8/8", "subtitle": "Staccato Revisit Step 15", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 8 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.563s (迭代 48 步)，胜率稳操胜券！</p>"}, {"step": 16, "gt": 0.12, "k_ab": 8, "k_full": 8, "k_k3": 3, "k_svolos": 3, "k_oracle": 4, "iters_ab": 48, "iters_full": 48, "iters_k3": 62, "iters_svolos": 62, "time_ab": 0.563, "time_full": 0.563, "time_k3": 0.631, "time_svolos": 0.631, "cum_time_ab": 8.94, "cum_time_full": 9.159, "cum_time_k3": 11.227, "cum_time_svolos": 13.696, "regret_ab_pct": 5.8, "state": "COOLDOWN", "lasers": [{"x": 0.6, "y": 0.25, "z": 0.4, "r": 0.32, "intensity": 1.4}], "updated_subs": [0, 1, 2, 3, 4, 5, 6, 7], "drifts": [0.33, 0.05, 0.09, 0.13, 0.17, 0.21, 0.25, 0.29], "phase": "轨迹 2：断奏式冲击重访工况 (Trajectory 2: Staccato Shock Revisit)", "title": "【断奏步 t = 16】断奏跳跃响应：AB-JSR 自适应调控 K_t = 8/8", "subtitle": "Staccato Revisit Step 16", "p_body": "<p><strong>断奏热斑机制</strong>：脉冲式热源在空间中非连续跳跃，瞬时局部刚度差分高达 10^3 倍。</p>", "trap_body": "<p><strong>工况特征</strong>：非连续脉冲热斑在象限之间突变，考验因果控制器的敏捷度。</p>", "dec_body": "<p><strong>AB-JSR 决策</strong>：解前因果感知判定跃迁风险，精准调度 K_t = 8 算力，零代数损失。</p>", "led_body": "<p><strong>战报</strong>：AB-JSR 当前步耗时 0.563s (迭代 48 步)，胜率稳操胜券！</p>"}];
  const FOUR_REGIMES_DATA = {"gentle": {"name": "区制 1：温和单光束移动 (Gentle Single-Front)", "k_star": 1, "ratio_star": 0.845, "color": "#3b82f6", "records": [{"K": 0, "refresh_ratio": 0.0, "mean_iters": 69.8, "max_iters": 94, "max_age": 10, "iters_history": [45, 49, 59, 65, 70, 72, 74, 81, 89, 94], "total_setup_sec": 0.0246955300681293, "total_solve_sec": 8.09350226901006, "total_wall_sec": 8.118197799078189, "max_rel_res": 1.4894277101692439e-09, "ratio_vs_full": 1.235133664787475}, {"K": 1, "refresh_ratio": 0.125, "mean_iters": 46.5, "max_iters": 49, "max_age": 10, "iters_history": [44, 44, 46, 47, 46, 48, 49, 47, 47, 47], "total_setup_sec": 0.17046109889633954, "total_solve_sec": 5.3860124700004235, "total_wall_sec": 5.556473568896763, "max_rel_res": 1.8491121759471912e-09, "ratio_vs_full": 0.8453831419610747}, {"K": 2, "refresh_ratio": 0.25, "mean_iters": 45.8, "max_iters": 47, "max_age": 10, "iters_history": [44, 44, 45, 45, 46, 46, 47, 47, 47, 47], "total_setup_sec": 0.3236546250991523, "total_solve_sec": 5.295606592204422, "total_wall_sec": 5.619261217303574, "max_rel_res": 1.6083910766123663e-09, "ratio_vs_full": 0.8549358949488003}, {"K": 3, "refresh_ratio": 0.375, "mean_iters": 45.6, "max_iters": 47, "max_age": 10, "iters_history": [43, 44, 45, 45, 45, 46, 47, 47, 47, 47], "total_setup_sec": 0.46704300498822704, "total_solve_sec": 5.143815915798768, "total_wall_sec": 5.610858920786995, "max_rel_res": 1.7959087337297807e-09, "ratio_vs_full": 0.8536575374184712}, {"K": 4, "refresh_ratio": 0.5, "mean_iters": 45.6, "max_iters": 47, "max_age": 10, "iters_history": [44, 44, 44, 45, 45, 46, 47, 47, 47, 47], "total_setup_sec": 0.6281439349404536, "total_solve_sec": 5.162098080967553, "total_wall_sec": 5.790242015908007, "max_rel_res": 1.9014573057875397e-09, "ratio_vs_full": 0.8809495676401163}, {"K": 5, "refresh_ratio": 0.625, "mean_iters": 45.6, "max_iters": 47, "max_age": 10, "iters_history": [44, 44, 44, 45, 45, 46, 47, 47, 47, 47], "total_setup_sec": 0.7706092508742586, "total_solve_sec": 5.1403442339506, "total_wall_sec": 5.910953484824859, "max_rel_res": 1.8205892426902579e-09, "ratio_vs_full": 0.8993150722354935}, {"K": 6, "refresh_ratio": 0.75, "mean_iters": 45.5, "max_iters": 47, "max_age": 10, "iters_history": [44, 44, 44, 45, 45, 46, 47, 47, 47, 46], "total_setup_sec": 0.9098881459212862, "total_solve_sec": 5.311622256995179, "total_wall_sec": 6.221510402916465, "max_rel_res": 1.7688979222062962e-09, "ratio_vs_full": 0.9465643896161495}, {"K": 7, "refresh_ratio": 0.875, "mean_iters": 45.6, "max_iters": 47, "max_age": 4, "iters_history": [44, 44, 44, 45, 45, 46, 47, 47, 47, 47], "total_setup_sec": 1.142175232002046, "total_solve_sec": 5.461427237896714, "total_wall_sec": 6.60360246989876, "max_rel_res": 1.7747346962772643e-09, "ratio_vs_full": 1.0046973381668305}, {"K": 8, "refresh_ratio": 1.0, "mean_iters": 45.6, "max_iters": 47, "max_age": 0, "iters_history": [44, 44, 44, 45, 45, 46, 47, 47, 47, 47], "total_setup_sec": 1.2639171431655996, "total_solve_sec": 5.3088109999662265, "total_wall_sec": 6.572728143131826, "max_rel_res": 1.7833711184117853e-09, "ratio_vs_full": 1.0}]}, "dual": {"name": "区制 2：中等双光束对角穿行 (Case B Dual-Beam)", "k_star": 1, "ratio_star": 0.843, "color": "#10b981", "records": [{"K": 0, "refresh_ratio": 0.0, "mean_iters": 75.5, "max_iters": 90, "max_age": 10, "iters_history": [42, 56, 66, 68, 76, 88, 90, 90, 90, 89], "total_setup_sec": 0.024065740930382162, "total_solve_sec": 8.042448116117157, "total_wall_sec": 8.06651385704754, "max_rel_res": 1.1147575328018296e-09, "ratio_vs_full": 1.3701343895933578}, {"K": 1, "refresh_ratio": 0.125, "mean_iters": 44.4, "max_iters": 57, "max_age": 10, "iters_history": [42, 42, 44, 57, 44, 44, 43, 43, 42, 43], "total_setup_sec": 0.16471416893182322, "total_solve_sec": 4.795732011029031, "total_wall_sec": 4.960446179960854, "max_rel_res": 1.0288038973316977e-09, "ratio_vs_full": 0.8425545433053998}, {"K": 2, "refresh_ratio": 0.25, "mean_iters": 40.5, "max_iters": 44, "max_age": 10, "iters_history": [38, 39, 41, 44, 41, 41, 41, 40, 40, 40], "total_setup_sec": 0.33205954695586115, "total_solve_sec": 4.729475497966632, "total_wall_sec": 5.061535044922493, "max_rel_res": 7.210181604393104e-10, "ratio_vs_full": 0.8597249508375076}, {"K": 3, "refresh_ratio": 0.375, "mean_iters": 40.3, "max_iters": 44, "max_age": 10, "iters_history": [38, 39, 42, 44, 41, 41, 41, 40, 39, 38], "total_setup_sec": 0.47128620598232374, "total_solve_sec": 4.6701981808873825, "total_wall_sec": 5.141484386869706, "max_rel_res": 5.740612251105309e-10, "ratio_vs_full": 0.8733047133927446}, {"K": 4, "refresh_ratio": 0.5, "mean_iters": 39.1, "max_iters": 41, "max_age": 10, "iters_history": [37, 38, 39, 41, 41, 41, 40, 39, 38, 37], "total_setup_sec": 0.6326011910568923, "total_solve_sec": 4.585286971880123, "total_wall_sec": 5.217888162937015, "max_rel_res": 8.162777869849633e-10, "ratio_vs_full": 0.886282245315429}, {"K": 5, "refresh_ratio": 0.625, "mean_iters": 38.9, "max_iters": 41, "max_age": 7, "iters_history": [37, 38, 39, 40, 41, 41, 39, 38, 38, 38], "total_setup_sec": 0.7929629129357636, "total_solve_sec": 4.372148527938407, "total_wall_sec": 5.1651114408741705, "max_rel_res": 8.02222903016894e-10, "ratio_vs_full": 0.8773178769216231}, {"K": 6, "refresh_ratio": 0.75, "mean_iters": 38.6, "max_iters": 41, "max_age": 5, "iters_history": [37, 38, 39, 41, 40, 40, 39, 38, 37, 37], "total_setup_sec": 1.0205414309748448, "total_solve_sec": 4.4453478349605575, "total_wall_sec": 5.465889265935402, "max_rel_res": 8.147557961168648e-10, "ratio_vs_full": 0.9284063705443596}, {"K": 7, "refresh_ratio": 0.875, "mean_iters": 38.6, "max_iters": 41, "max_age": 1, "iters_history": [37, 38, 39, 41, 40, 40, 39, 38, 37, 37], "total_setup_sec": 1.0776486551621929, "total_solve_sec": 4.3523726958665065, "total_wall_sec": 5.430021351028699, "max_rel_res": 7.772625163917909e-10, "ratio_vs_full": 0.9223140406274221}, {"K": 8, "refresh_ratio": 1.0, "mean_iters": 38.6, "max_iters": 41, "max_age": 0, "iters_history": [37, 38, 39, 41, 40, 40, 39, 38, 37, 37], "total_setup_sec": 1.2865322279394604, "total_solve_sec": 4.600856570061296, "total_wall_sec": 5.887388798000757, "max_rel_res": 7.537222543909643e-10, "ratio_vs_full": 1.0}]}, "serpentine": {"name": "区制 3：50 步大回环长程蛇形扫描 (Serpentine 50-Steps)", "k_star": 1, "ratio_star": 0.829, "color": "#f59e0b", "records": [{"K": 0, "refresh_ratio": 0.0, "mean_iters": 77.32, "max_iters": 88, "max_age": 50, "iters_history": [40, 45, 53, 60, 64, 66, 68, 74, 82, 87, 88, 88, 88, 87, 86, 86, 77, 78, 78, 77, 77, 77, 72, 69, 64, 66, 70, 72, 73, 73, 74, 73, 75, 86, 86, 87, 87, 88, 88, 87, 82, 78, 82, 87, 88, 88, 87, 86, 86, 86], "total_setup_sec": 0.12222956365440041, "total_solve_sec": 42.029003626958, "total_wall_sec": 42.1512331906124, "max_rel_res": 6.563864899713715e-10, "ratio_vs_full": 1.49058106738492}, {"K": 1, "refresh_ratio": 0.125, "mean_iters": 42.26, "max_iters": 48, "max_age": 50, "iters_history": [37, 38, 39, 40, 41, 40, 42, 41, 41, 41, 41, 41, 39, 39, 39, 39, 47, 42, 45, 44, 48, 43, 43, 48, 45, 43, 46, 48, 44, 45, 44, 44, 44, 41, 43, 41, 42, 41, 45, 42, 44, 43, 44, 42, 41, 41, 41, 41, 40, 40], "total_setup_sec": 0.8277893681661226, "total_solve_sec": 22.628009677806403, "total_wall_sec": 23.455799045972526, "max_rel_res": 9.73525068715217e-10, "ratio_vs_full": 0.8294601920709299}, {"K": 2, "refresh_ratio": 0.25, "mean_iters": 40.9, "max_iters": 47, "max_age": 38, "iters_history": [37, 38, 38, 39, 39, 40, 40, 41, 41, 40, 40, 40, 39, 38, 38, 37, 41, 41, 42, 41, 42, 41, 42, 42, 40, 43, 42, 43, 43, 44, 45, 45, 47, 43, 40, 40, 41, 41, 43, 42, 42, 42, 42, 42, 42, 41, 41, 39, 38, 37], "total_setup_sec": 1.5999406882328913, "total_solve_sec": 23.121077907911967, "total_wall_sec": 24.72101859614486, "max_rel_res": 9.91257950474447e-10, "ratio_vs_full": 0.8742017610552547}, {"K": 3, "refresh_ratio": 0.375, "mean_iters": 39.18, "max_iters": 42, "max_age": 35, "iters_history": [37, 37, 38, 38, 39, 39, 39, 39, 39, 39, 39, 38, 38, 37, 37, 37, 40, 40, 40, 40, 41, 41, 42, 41, 40, 41, 42, 42, 41, 42, 41, 41, 41, 37, 37, 37, 38, 39, 39, 39, 40, 40, 40, 39, 39, 39, 38, 38, 37, 37], "total_setup_sec": 2.421737389813643, "total_solve_sec": 22.317854575929232, "total_wall_sec": 24.739591965742875, "max_rel_res": 9.60718288040473e-10, "ratio_vs_full": 0.8748585653996294}, {"K": 4, "refresh_ratio": 0.5, "mean_iters": 38.92, "max_iters": 41, "max_age": 33, "iters_history": [37, 37, 37, 38, 39, 38, 39, 39, 39, 39, 39, 39, 37, 36, 37, 37, 40, 40, 40, 41, 41, 41, 41, 41, 39, 41, 41, 41, 41, 41, 41, 41, 40, 37, 37, 37, 37, 38, 38, 39, 39, 40, 40, 39, 39, 39, 38, 37, 37, 37], "total_setup_sec": 3.0531222500139847, "total_solve_sec": 21.09509071119828, "total_wall_sec": 24.148212961212266, "max_rel_res": 1.6471647843994e-09, "ratio_vs_full": 0.8539458119383952}, {"K": 5, "refresh_ratio": 0.625, "mean_iters": 39.02, "max_iters": 42, "max_age": 24, "iters_history": [37, 37, 37, 38, 39, 38, 39, 39, 39, 39, 39, 39, 37, 37, 37, 37, 40, 40, 40, 41, 40, 41, 42, 42, 41, 42, 42, 42, 41, 41, 41, 40, 40, 37, 37, 37, 37, 38, 38, 39, 39, 40, 40, 39, 39, 39, 38, 37, 36, 37], "total_setup_sec": 4.0004927790141664, "total_solve_sec": 22.547788619005587, "total_wall_sec": 26.548281398019753, "max_rel_res": 1.470434297793002e-09, "ratio_vs_full": 0.9388186923154779}, {"K": 6, "refresh_ratio": 0.75, "mean_iters": 38.98, "max_iters": 42, "max_age": 21, "iters_history": [37, 37, 37, 38, 39, 38, 39, 39, 39, 39, 39, 38, 37, 37, 37, 37, 40, 40, 40, 40, 40, 41, 42, 42, 41, 42, 42, 42, 41, 41, 41, 40, 40, 37, 37, 37, 37, 38, 38, 39, 40, 40, 40, 39, 39, 38, 38, 37, 36, 37], "total_setup_sec": 4.531036927364767, "total_solve_sec": 21.59484543610597, "total_wall_sec": 26.125882363470737, "max_rel_res": 1.4509493297498285e-09, "ratio_vs_full": 0.9238815254493693}, {"K": 7, "refresh_ratio": 0.875, "mean_iters": 38.88, "max_iters": 42, "max_age": 7, "iters_history": [37, 37, 37, 38, 39, 39, 39, 39, 39, 39, 39, 38, 37, 37, 37, 37, 40, 40, 40, 40, 41, 41, 42, 42, 41, 42, 42, 41, 41, 41, 40, 39, 39, 36, 37, 37, 37, 38, 38, 39, 39, 40, 40, 39, 39, 38, 37, 37, 36, 37], "total_setup_sec": 5.301767435739748, "total_solve_sec": 21.299361640762072, "total_wall_sec": 26.60112907650182, "max_rel_res": 1.1728779349456834e-09, "ratio_vs_full": 0.9406875284808254}, {"K": 8, "refresh_ratio": 1.0, "mean_iters": 38.86, "max_iters": 42, "max_age": 0, "iters_history": [37, 37, 37, 38, 39, 39, 39, 39, 39, 39, 39, 38, 37, 37, 37, 37, 40, 40, 40, 40, 41, 41, 42, 42, 40, 42, 42, 41, 41, 41, 40, 39, 39, 36, 37, 37, 37, 38, 38, 39, 39, 40, 40, 39, 39, 38, 37, 37, 36, 37], "total_setup_sec": 6.132442710746545, "total_solve_sec": 22.145947584765963, "total_wall_sec": 28.27839029551251, "max_rel_res": 1.1563669458926902e-09, "ratio_vs_full": 1.0}]}, "multifront": {"name": "区制 4：狂暴多前沿跨象限对撞 (Violent Multi-Front Churn)", "k_star": 8, "ratio_star": 1.0, "color": "#f43f5e", "records": [{"K": 0, "refresh_ratio": 0.0, "mean_iters": 167.6875, "max_iters": 193, "max_age": 16, "iters_history": [84, 152, 189, 178, 187, 159, 187, 177, 162, 193, 185, 174, 146, 182, 146, 182], "total_setup_sec": 0.0369877151097171, "total_solve_sec": 28.77048220601864, "total_wall_sec": 28.80746992112836, "mean_solve_sec": 1.798155137876165, "mean_setup_sec": 0.002311732194357319, "max_rel_res": 3.0415946280493764e-09, "ratio_vs_full": 2.73391845253191}, {"K": 1, "refresh_ratio": 0.125, "mean_iters": 126.5625, "max_iters": 235, "max_age": 16, "iters_history": [72, 115, 103, 67, 104, 158, 105, 63, 177, 175, 123, 199, 176, 235, 89, 64], "total_setup_sec": 0.2810426380019635, "total_solve_sec": 21.612006390059832, "total_wall_sec": 21.893049028061796, "mean_solve_sec": 1.3507503993787395, "mean_setup_sec": 0.01756516487512272, "max_rel_res": 3.146931024666556e-09, "ratio_vs_full": 2.0777184141431717}, {"K": 2, "refresh_ratio": 0.25, "mean_iters": 103.4375, "max_iters": 184, "max_age": 8, "iters_history": [66, 68, 90, 80, 117, 115, 99, 77, 172, 164, 148, 184, 87, 62, 64, 62], "total_setup_sec": 0.5186182680772617, "total_solve_sec": 17.720068014168646, "total_wall_sec": 18.238686282245908, "mean_solve_sec": 1.1075042508855404, "mean_setup_sec": 0.03241364175482886, "max_rel_res": 6.010317514604263e-09, "ratio_vs_full": 1.7309080288373904}, {"K": 3, "refresh_ratio": 0.375, "mean_iters": 81.8125, "max_iters": 179, "max_age": 5, "iters_history": [49, 49, 51, 53, 66, 66, 66, 59, 118, 133, 145, 179, 90, 62, 61, 62], "total_setup_sec": 0.7073243412305601, "total_solve_sec": 13.040772995038424, "total_wall_sec": 13.748097336268984, "mean_solve_sec": 0.8150483121899015, "mean_setup_sec": 0.044207771326910006, "max_rel_res": 1.9604267453876655e-09, "ratio_vs_full": 1.30473717746603}, {"K": 4, "refresh_ratio": 0.5, "mean_iters": 69.625, "max_iters": 125, "max_age": 5, "iters_history": [49, 49, 49, 49, 62, 65, 63, 56, 87, 68, 116, 125, 89, 62, 62, 63], "total_setup_sec": 0.9569434169097804, "total_solve_sec": 11.431012359971646, "total_wall_sec": 12.387955776881427, "mean_solve_sec": 0.7144382724982279, "mean_setup_sec": 0.05980896355686127, "max_rel_res": 1.4504352337912467e-09, "ratio_vs_full": 1.175655515055341}, {"K": 5, "refresh_ratio": 0.625, "mean_iters": 56.0625, "max_iters": 85, "max_age": 4, "iters_history": [50, 49, 51, 49, 49, 51, 52, 48, 63, 63, 85, 68, 56, 51, 61, 51], "total_setup_sec": 1.223233072028961, "total_solve_sec": 9.59395018091891, "total_wall_sec": 10.81718325294787, "mean_solve_sec": 0.5996218863074319, "mean_setup_sec": 0.07645206700181006, "max_rel_res": 1.2248856511113136e-09, "ratio_vs_full": 1.0265843192971031}, {"K": 6, "refresh_ratio": 0.75, "mean_iters": 53.75, "max_iters": 70, "max_age": 2, "iters_history": [49, 51, 53, 51, 49, 52, 54, 52, 64, 53, 70, 58, 52, 51, 50, 51], "total_setup_sec": 1.485183498065453, "total_solve_sec": 9.309259742032737, "total_wall_sec": 10.79444324009819, "mean_solve_sec": 0.581828733877046, "mean_setup_sec": 0.09282396862909081, "max_rel_res": 1.3520528397697918e-09, "ratio_vs_full": 1.024426221383237}, {"K": 7, "refresh_ratio": 0.875, "mean_iters": 51.75, "max_iters": 61, "max_age": 1, "iters_history": [50, 51, 52, 48, 50, 52, 54, 47, 61, 53, 57, 54, 50, 50, 49, 50], "total_setup_sec": 1.6517550640855916, "total_solve_sec": 8.884524758090265, "total_wall_sec": 10.536279822175857, "mean_solve_sec": 0.5552827973806416, "mean_setup_sec": 0.10323469150534947, "max_rel_res": 1.691297933069121e-09, "ratio_vs_full": 0.9999257104407984}, {"K": 8, "refresh_ratio": 1.0, "mean_iters": 50.125, "max_iters": 54, "max_age": 0, "iters_history": [52, 51, 52, 48, 50, 51, 50, 47, 54, 52, 53, 50, 48, 48, 48, 48], "total_setup_sec": 1.962960303062573, "total_solve_sec": 8.5741023128503, "total_wall_sec": 10.537062615912873, "mean_solve_sec": 0.5358813945531438, "mean_setup_sec": 0.12268501894141082, "max_rel_res": 1.327468035608695e-09, "ratio_vs_full": 1.0}]}};
  const STORY_CHAPTERS = [{"title": "第 1 讲：昨天的幻象——温和单前沿与固定预算的甜蜜期", "subtitle": "为什么之前的 2D mass95 网页看起来已经很完美？背后的温室效应", "tag": "历史回眸 · The Old World", "content_html": "\n        <div class=\"story-grid\">\n          <div class=\"story-text-col\">\n            <p><strong>您记忆中的 2D 网页</strong>：还记得上次我们在 2D 网格上看到的动画吗？一束温和的激光热浪缓缓穿过 64 个子域，mass95 算法按能量累积挑出 8 个关键子域刷新，配合联合粗网格同步，求解极其丝滑，总耗时稳稳跑赢全量重建！</p>\n            <p><strong>当时的甜蜜期结论</strong>：看起来“固定挑前几个块（例如固定预算 K=3 或 mass95 挑 8 块）”就是一个通用的万能钥匙。当时我们自然会产生一种直觉：“这不就解决了吗？为什么还要继续做因果自适应？”</p>\n            <div class=\"story-callout warning\">\n              <strong>真相揭秘：工况温和掩盖了致命盲区！</strong><br>\n              在温和的单激光平滑移动时，全场 64 个子域中受热的只有 1~2 个。未受热的 60 多个子域完全静止，哪怕使用上百步前极其陈旧的局部因子，也不会产生任何代数残差反弹。这相当于“温室里的花朵”——任何固定预算策略在这里都能混及格！\n            </div>\n            <p><strong>学术警报</strong>：当顶级期刊审稿人提出：“如果工况不是温柔平移，而是多前沿、跨区域、非局部剧变呢？”——旧认知瞬间崩塌！</p>\n          </div>\n          <div class=\"story-graphic-col\">\n            <div style=\"text-align:center; padding: 20px;\">\n              <div style=\"font-size: 3rem; margin-bottom: 10px;\">☀️ ➔ 🍃</div>\n              <h4 style=\"color: var(--cyan-primary);\">单前沿温和移动 (Gentle 2D)</h4>\n              <p style=\"color: var(--text-muted); font-size: 0.85rem; margin-top: 8px;\">\n                扰动高度局域集中<br>\n                仅需固定 K=1~3 块即可吸纳 95% 扰动<br>\n                <strong>给科研人员造成“固定预算通吃”的巨大假象</strong>\n              </p>\n            </div>\n          </div>\n        </div>\n        "}, {"title": "第 2 讲：致命危机——剧烈多前沿重访与“算力饥饿踩踏”", "subtitle": "当 4 束激光跨象限暴击时，传统固定预算为什么当场暴毙？", "tag": "遭遇瓶颈 · The Lethal Crisis", "content_html": "\n        <div class=\"story-grid\">\n          <div class=\"story-text-col\">\n            <p><strong>工业级极限场景</strong>：当我们把工况升级为真实 3D 复杂热传导：4 束高能激光在 8 个局部子域中跨象限对撞、剧烈重访！</p>\n            <p><strong>算力饥饿惨剧（Starvation Chasm）</strong>：此时全场有 7 个子域在剧烈演化。如果系统依然死板地坚守“固定预算 K=3”，悲剧立刻上演：<br>\n            系统只能挑出 3 个块刷新，而<strong>剩下的 4 个剧变子域被迫继续使用严重陈旧的局部逆矩阵</strong>！</p>\n            <div class=\"story-callout danger\">\n              <strong>💥 Krylov 迭代雪崩与算力踩踏！</strong><br>\n              未刷新的 4 个子域产生巨大的局部残差泄漏。粗网格即使联合同步，也无法挽救微观基底的严重失稳。PCG 条件数直接爆炸！\n              <ul>\n                <li>Fixed K=3 的迭代数瞬间从 50 步一路狂飙到 <strong style=\"color: #f43f5e;\">229 步</strong>！</li>\n                <li>Svolos 物理基线更是失控飙升至 <strong style=\"color: #f43f5e;\">290 步</strong>！</li>\n                <li>总求解耗时反比全量重建慢 <strong style=\"color: #f43f5e;\">+43% 到 +107%</strong>！</li>\n              </ul>\n            </div>\n            <p><strong>痛定思痛</strong>：原本为了“省计算”而引入的固定选择性维护，在遭遇工况跃迁时，反而变成了吞噬算力的无底黑洞！</p>\n          </div>\n          <div class=\"story-graphic-col\">\n            <div style=\"text-align:center; padding: 20px;\">\n              <div style=\"font-size: 3.5rem; margin-bottom: 10px;\">💥 ➔ 🌋</div>\n              <h4 style=\"color: var(--rose-danger);\">算力饥饿陷阱 (Starvation)</h4>\n              <p style=\"color: var(--text-muted); font-size: 0.85rem; margin-top: 8px;\">\n                剧变子域 > 维护预算 K<br>\n                未刷新子域残差暴走<br>\n                <strong>PCG 迭代 50 步 ➔ 229 步 ➔ 耗时暴涨 107%</strong>\n              </p>\n            </div>\n          </div>\n        </div>\n        "}, {"title": "第 3 讲：科学顿悟——四区制实验与非平稳最优算力 K*", "subtitle": "核心科学发现：世界上根本不存在放之四海而皆准的“最优固定预算”！", "tag": "科学顿悟 · The Core Discovery", "content_html": "\n        <div class=\"story-grid\">\n          <div class=\"story-text-col\">\n            <p><strong>系统性探秘</strong>：面对危机，我们拒绝盲目调参，而是在 4 个截然不同的物理区制中，测试从 K=0 到 K=8 的全预算响应曲面 T(K)/T_Full：</p>\n            <ul>\n              <li><strong>区制 1（温和单斑移动）</strong>：最优预算是 <strong style=\"color: #3b82f6;\">K* = 1 / 8</strong> (仅更新 12.5% 就饱和加速，耗时比 0.845)；</li>\n              <li><strong>区制 2（双斑对角穿行）</strong>：最优预算依然是 <strong style=\"color: #10b981;\">K* = 1 / 8</strong> (耗时比 0.843)；</li>\n              <li><strong>区制 3（50步长程蛇形）</strong>：最优预算依然是 <strong style=\"color: #f59e0b;\">K* = 1 / 8</strong> (耗时比 0.829)；</li>\n              <li><strong>区制 4（狂暴多前沿对撞）</strong>：最优预算发生天翻地覆的跃迁——<strong style=\"color: #f43f5e;\">K* ∈ {7, 8} (必须 100% 全量重建！)</strong> 此时固定 K=3 额外浪费 30.5% 时间！</li>\n            </ul>\n            <div class=\"story-callout\">\n              <strong>💡 顿悟：最优维护预算是非平稳动态漂移的！</strong><br>\n              在温和区制，多更新一个块都是纯粹的因子分解浪费（Iteration Floor Trap）；但在狂暴对撞区制，少更新一个块就是灭顶之灾（Starvation Chasm）！\n            </div>\n            <p><strong>结论</strong>：谁如果试图在论文里找一个全局固定的 K，谁就注定在另一个工况下被反例打穿！必须让算法<strong>在线自适应感知并动态闭环调控 K_t</strong>！</p>\n          </div>\n          <div class=\"story-graphic-col\">\n            <div style=\"text-align:center; padding: 20px;\">\n              <div style=\"font-size: 3rem; margin-bottom: 10px;\">📉 ⇄ 📈</div>\n              <h4 style=\"color: var(--emerald-win);\">K* 漂移定律 (Regime Dependent)</h4>\n              <p style=\"color: var(--text-muted); font-size: 0.85rem; margin-top: 8px;\">\n                平稳工况：K* = 1 (省算力)<br>\n                对撞工况：K* = 8 (全量保命)<br>\n                <strong>必须引入解前因果控制器！</strong>\n              </p>\n            </div>\n          </div>\n        </div>\n        "}, {"title": "第 4 讲：破局之道——AB-JSR 因果前馈预算控制器与 Schmitt 滞回", "subtitle": "解前对角感知 + 洛伦兹风险集中度 + 双阈值迟滞急升缓降安全网", "tag": "破局创新 · Causal Controller", "content_html": "\n        <div class=\"story-grid\">\n          <div class=\"story-text-col\">\n            <p><strong>如何在不偷看未来、不引入多余重算的前提下实现闭环？</strong> 我们的 AB-JSR 提出了三大因果创新组件：</p>\n            <p><strong>1. 解前对角感知器 (G_t 感知)</strong>：<br>\n            利用刚度矩阵已装配的对角线提取局部相对漂移，算得全局扰动弥散度 G_t。开销仅占求解总时间的 <strong>0.06%</strong>，微秒级极速完成！</p>\n            <p><strong>2. 洛伦兹风险集中度截断 (Q_k 截断)</strong>：<br>\n            将 8 个子域按漂移排序，绘制风险集中度曲线 Q_k。截取累积覆盖达到 <strong>α = 80%</strong> 的最小预算 K_t，安全冻结健康子域，绝不多花一分钱！</p>\n            <div class=\"story-callout\">\n              <strong>3. 施密特双阈值迟滞安全网 (Schmitt Trigger with Cooldown)</strong>：<br>\n              <ul>\n                <li><strong>激增跃迁线 τ_up = 0.22</strong>：一旦 G_t ≥ 0.22，瞬间全速跃迁至 K_t = 8 全量重构，第一时间扑灭火灾！</li>\n                <li><strong>冷却清洗期 (Cooldown = 2 steps)</strong>：保持 2 步全量刷新，彻底清洗微观记忆残差，防止次生失稳；</li>\n                <li><strong>平稳降级线 τ_down = 0.15</strong>：只有弥散度降回 0.15 以下且收敛稳定，才缓降恢复节能模式。</li>\n              </ul>\n            </div>\n            <p><strong>求解器应急回溯 (Solver Distress)</strong>：一旦监测到 PCG 迭代有不正常微抬，立即强制升级保护，双重冗余零死角！</p>\n          </div>\n          <div class=\"story-graphic-col\">\n            <div style=\"text-align:center; padding: 20px;\">\n              <div style=\"font-size: 3rem; margin-bottom: 10px;\">🎛️ ➔ 🛡️</div>\n              <h4 style=\"color: var(--cyan-primary);\">AB-JSR 三位一体控制架</h4>\n              <p style=\"color: var(--text-muted); font-size: 0.85rem; margin-top: 8px;\">\n                ① diag(A_t) 感知 (0.06% 耗时)<br>\n                ② 洛伦兹 α=80% 截断<br>\n                ③ 施密特急升缓降安全网<br>\n                <strong>纯因果 · 无作弊 · 零延迟</strong>\n              </p>\n            </div>\n          </div>\n        </div>\n        "}, {"title": "第 5 讲：决胜时刻——盲测两路冲击与最终统计审计", "subtitle": "双盲测轨迹 100% 配对胜率，完全消除固定预算失效，Oracle Regret 仅 8%", "tag": "终局大捷 · The Empirical Proof", "content_html": "\n        <div class=\"story-grid\">\n          <div class=\"story-text-col\">\n            <p><strong>最严苛的学术大考</strong>：我们将从未参与控制器标定的两条全新盲测轨迹（复合工况 18 步、断奏冲击 16 步）投入压力实测，由外部程序执行多轮独立重复审计：</p>\n            <div class=\"story-callout\">\n              <strong>🏆 终局审计认证数据（直接提交 CMAME 审稿人）：</strong>\n              <ul>\n                <li><strong>端到端超越全量重建</strong>：耗时比分别达到 <strong style=\"color: #10b981;\">0.937</strong> 和 <strong style=\"color: #10b981;\">0.943</strong>（净提速 ~6%），省下大量冗余因子分解耗时！</li>\n                <li><strong>对固定预算配对胜率 100% (3/3)</strong>：每步消除 2.3 ~ 4.4 秒的性能惩罚，失效消除率高达 118%~128%！</li>\n                <li><strong>100% 保持全量收敛品质</strong>：峰值迭代严格保持在 53 和 55 步（MaxIter 比值 <strong>严格为 1.00</strong>），将 Fixed K=3 的 229 步火情压制了 76.9%！</li>\n                <li><strong>逼近事前未知最优</strong>：全轨迹 Oracle Regret 仅为 <strong style=\"color: #10b981;\">+8.06%</strong> 与 <strong style=\"color: #10b981;\">+6.68%</strong>！</li>\n              </ul>\n            </div>\n            <p><strong>从“调参修补”到“范式闭环”</strong>：现在的研究不再是“解释为什么比 Svolos 快几个百分点”，而是真正建立了一个具有深厚计算力学机理、能够自适应应对未知剧烈演化工况的全新因果求解体系！</p>\n          </div>\n          <div class=\"story-graphic-col\">\n            <div style=\"text-align:center; padding: 20px;\">\n              <div style=\"font-size: 3.5rem; margin-bottom: 10px;\">🏆 ➔ 📜</div>\n              <h4 style=\"color: var(--emerald-win);\">学术级审稿防御壁垒</h4>\n              <p style=\"color: var(--text-muted); font-size: 0.85rem; margin-top: 8px;\">\n                双盲测全胜 · 100% 配对显著<br>\n                完全阻断条件数崩溃<br>\n                <strong>大功告成！欢迎在主界面体验交互！</strong>\n              </p>\n            </div>\n          </div>\n        </div>\n        "}];

  // --- 2. 应用交互状态 ---
  const STATE = {
    activeView: 'modern',     // 'modern' | 'story' | 'regimes' | 'legacy'
    activeTraj: 'compound',   // 'compound' (18 steps) | 'staccato' (16 steps)
    step: 1,                  // 1-indexed
    isPlaying: false,
    speed: 800,
    playTimer: null,
    showLaser: true,
    showSubCubes: true,
    showCoarse: true,
    camera: {
      rotX: 25,
      rotY: -35,
      zoom: 1.0,
      isDragging: false,
      lastX: 0,
      lastY: 0
    },
    hoveredSub: null,
    storyIndex: 0,
    selectedRegime: 'gentle',
    // 经典 2D 状态
    legacy: {
      time: 0,
      isPlaying: false,
      timer: null,
      showHeat: true,
      showSub: true,
      showCoarse: true
    }
  };

  // --- 3. DOM 元素缓存 ---
  const DOM = {
    // Top Tabs
    tabModern: document.getElementById('tabModern'),
    tabStory: document.getElementById('tabStory'),
    tabRegimes: document.getElementById('tabRegimes'),
    tabLegacy: document.getElementById('tabLegacy'),

    // Views
    viewModern: document.getElementById('viewModern'),
    viewStory: document.getElementById('viewStory'),
    viewRegimes: document.getElementById('viewRegimes'),
    viewLegacy: document.getElementById('viewLegacy'),

    // Modern Timeline Controls
    btnPlayPause: document.getElementById('btnPlayPause'),
    lblPlayPause: document.getElementById('lblPlayPause'),
    btnStepPrev: document.getElementById('btnStepPrev'),
    btnStepNext: document.getElementById('btnStepNext'),
    btnReset: document.getElementById('btnReset'),
    timeTicksWrapper: document.getElementById('timeTicksWrapper'),
    currentStepBadge: document.getElementById('currentStepBadge'),
    totalStepsBadge: document.getElementById('totalStepsBadge'),
    btnTrajCompound: document.getElementById('btnTrajCompound'),
    btnTrajStaccato: document.getElementById('btnTrajStaccato'),
    speedBtns: document.querySelectorAll('.btn-speed'),

    // Strategy Summary Cards
    abBudgetDisplay: document.getElementById('abBudgetDisplay'),
    abItersDisplay: document.getElementById('abItersDisplay'),
    abTimeDisplay: document.getElementById('abTimeDisplay'),
    abBar: document.getElementById('abBar'),
    abCumTime: document.getElementById('abCumTime'),

    fullItersDisplay: document.getElementById('fullItersDisplay'),
    fullTimeDisplay: document.getElementById('fullTimeDisplay'),
    fullBar: document.getElementById('fullBar'),
    fullCumTime: document.getElementById('fullCumTime'),

    k3ItersDisplay: document.getElementById('k3ItersDisplay'),
    k3TimeDisplay: document.getElementById('k3TimeDisplay'),
    k3Bar: document.getElementById('k3Bar'),
    k3CumTime: document.getElementById('k3CumTime'),

    svolosItersDisplay: document.getElementById('svolosItersDisplay'),
    svolosTimeDisplay: document.getElementById('svolosTimeDisplay'),
    svolosBar: document.getElementById('svolosBar'),
    svolosCumTime: document.getElementById('svolosCumTime'),

    // 3D Canvas
    stageCanvas: document.getElementById('stageCanvas'),
    chkShowLaser: document.getElementById('chkShowLaser'),
    chkShowSubCubes: document.getElementById('chkShowSubCubes'),
    chkShowCoarseMesh: document.getElementById('chkShowCoarseMesh'),
    cubeTooltip: document.getElementById('cubeTooltip'),
    ttCubeId: document.getElementById('ttCubeId'),
    ttCubeCoord: document.getElementById('ttCubeCoord'),
    ttCubeAge: document.getElementById('ttCubeAge'),
    ttCubeDrift: document.getElementById('ttCubeDrift'),
    ttCubeScore: document.getElementById('ttCubeScore'),
    ttCubeAction: document.getElementById('ttCubeAction'),

    // Cockpit
    schmittDot: document.getElementById('schmittDot'),
    schmittText: document.getElementById('schmittText'),
    gtValueBadge: document.getElementById('gtValueBadge'),
    gtGaugeFill: document.getElementById('gtGaugeFill'),
    gtExplainText: document.getElementById('gtExplainText'),

    lorenzCanvas: document.getElementById('lorenzCanvas'),
    lorenzDecisionBadge: document.getElementById('lorenzDecisionBadge'),
    lorenzKVal: document.getElementById('lorenzKVal'),

    iterationFireBox: document.getElementById('iterationFireBox'),
    fireStatusPill: document.getElementById('fireStatusPill'),
    iterBarAB: document.getElementById('iterBarAB'),
    iterNumAB: document.getElementById('iterNumAB'),
    iterBarFull: document.getElementById('iterBarFull'),
    iterNumFull: document.getElementById('iterNumFull'),
    iterBarK3: document.getElementById('iterBarK3'),
    iterNumK3: document.getElementById('iterNumK3'),
    iterBarSvolos: document.getElementById('iterBarSvolos'),
    iterNumSvolos: document.getElementById('iterNumSvolos'),
    iterCommentText: document.getElementById('iterCommentText'),

    oracleHitBadge: document.getElementById('oracleHitBadge'),
    oracleKStar: document.getElementById('oracleKStar'),
    abKActual: document.getElementById('abKActual'),
    oracleTimeBest: document.getElementById('oracleTimeBest'),
    abRegretVal: document.getElementById('abRegretVal'),

    // Commentary
    expPhaseTag: document.getElementById('expPhaseTag'),
    expStepTitle: document.getElementById('expStepTitle'),
    expStepSubtitle: document.getElementById('expStepSubtitle'),
    expPhysicsBody: document.getElementById('expPhysicsBody'),
    expTrapBody: document.getElementById('expTrapBody'),
    expDecisionBody: document.getElementById('expDecisionBody'),
    expLedgerBody: document.getElementById('expLedgerBody'),

    // Charts
    iterHistoryCanvas: document.getElementById('iterHistoryCanvas'),
    cumTimeCanvas: document.getElementById('cumTimeCanvas'),

    // Story View
    storyStepBtns: document.querySelectorAll('.s-step-btn'),
    storyContentBody: document.getElementById('storyContentBody'),
    btnStoryPrev: document.getElementById('btnStoryPrev'),
    btnStoryNext: document.getElementById('btnStoryNext'),

    // Regimes View
    regimeCurveCanvas: document.getElementById('regimeCurveCanvas'),
    regimeCards: document.querySelectorAll('.regime-info-card'),

    // Legacy View
    legacyCanvas: document.getElementById('legacyCanvas'),
    chkLegacyHeat: document.getElementById('chkLegacyHeat'),
    chkLegacySub: document.getElementById('chkLegacySub'),
    chkLegacyCoarse: document.getElementById('chkLegacyCoarse')
  };

  // 获取当前轨迹与步数据
  function getCurrentSteps() {
    return STATE.activeTraj === 'compound' ? TRAJECTORY_1_STEPS : TRAJECTORY_2_STEPS;
  }

  function getCurrentStepData() {
    const steps = getCurrentSteps();
    const idx = Math.min(Math.max(STATE.step - 1, 0), steps.length - 1);
    return steps[idx];
  }

  // --- 4. 3D 渲染引擎 (8 子域 2x2x2 拓扑 + 激光 + 粗网格) ---
  // 子域 8 个中心点坐标 (范围 [-0.6, 0.6])
  const SUBDOMAINS_3D = [
    { id: 0, x: -0.45, y: -0.45, z: -0.45, label: '(0,0,0)' },
    { id: 1, x:  0.45, y: -0.45, z: -0.45, label: '(1,0,0)' },
    { id: 2, x: -0.45, y:  0.45, z: -0.45, label: '(0,1,0)' },
    { id: 3, x:  0.45, y:  0.45, z: -0.45, label: '(1,1,0)' },
    { id: 4, x: -0.45, y: -0.45, z:  0.45, label: '(0,0,1)' },
    { id: 5, x:  0.45, y: -0.45, z:  0.45, label: '(1,0,1)' },
    { id: 6, x: -0.45, y:  0.45, z:  0.45, label: '(0,1,1)' },
    { id: 7, x:  0.45, y:  0.45, z:  0.45, label: '(1,1,1)' }
  ];

  // 3D 投影函数 (欧拉角旋转)
  function project3D(x, y, z, cx, cy, scale) {
    const radX = (STATE.camera.rotX * Math.PI) / 180;
    const radY = (STATE.camera.rotY * Math.PI) / 180;

    // 绕 Y 轴旋转
    const x1 = x * Math.cos(radY) + z * Math.sin(radY);
    const y1 = y;
    const z1 = -x * Math.sin(radY) + z * Math.cos(radY);

    // 绕 X 轴旋转
    const x2 = x1;
    const y2 = y1 * Math.cos(radX) - z1 * Math.sin(radX);
    const z2 = y1 * Math.sin(radX) + z1 * Math.cos(radX);

    // 透视与投影
    const dist = 3.2;
    const pScale = (dist / (dist + z2)) * scale;
    return {
      x: cx + x2 * pScale,
      y: cy - y2 * pScale,
      z: z2,
      scale: pScale
    };
  }

  // 渲染 3D 舞台
  function render3DStage() {
    const canvas = DOM.stageCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const cx = w * 0.48;
    const cy = h * 0.52;
    const baseScale = 210 * STATE.camera.zoom;
    const stepData = getCurrentStepData();

    // 1. 绘制环境网格投影底座
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let i = -3; i <= 3; i++) {
      const p1 = project3D(i * 0.35, -0.9, -1.05, cx, cy, baseScale);
      const p2 = project3D(i * 0.35, -0.9,  1.05, cx, cy, baseScale);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();

      const p3 = project3D(-1.05, -0.9, i * 0.35, cx, cy, baseScale);
      const p4 = project3D( 1.05, -0.9, i * 0.35, cx, cy, baseScale);
      ctx.beginPath();
      ctx.moveTo(p3.x, p3.y);
      ctx.lineTo(p4.x, p4.y);
      ctx.stroke();
    }

    // 2. 绘制粗网格通信骨架连线 (Coarse Mesh Skeleton)
    if (STATE.showCoarse) {
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.25)';
      ctx.setLineDash([4, 4]);

      // 连接各中心点
      for (let i = 0; i < 8; i++) {
        for (let j = i + 1; j < 8; j++) {
          const si = SUBDOMAINS_3D[i];
          const sj = SUBDOMAINS_3D[j];
          // 仅连接相邻子域 (曼哈顿距离为 1 个轴)
          const diff = (si.x !== sj.x ? 1 : 0) + (si.y !== sj.y ? 1 : 0) + (si.z !== sj.z ? 1 : 0);
          if (diff === 1) {
            const pi = project3D(si.x, si.y, si.z, cx, cy, baseScale);
            const pj = project3D(sj.x, sj.y, sj.z, cx, cy, baseScale);
            ctx.beginPath();
            ctx.moveTo(pi.x, pi.y);
            ctx.lineTo(pj.x, pj.y);
            ctx.stroke();
          }
        }
      }
      ctx.setLineDash([]);
    }

    // 3. 准备 8 个子域立方体及其深度排序
    const cubes = SUBDOMAINS_3D.map(sub => {
      const p = project3D(sub.x, sub.y, sub.z, cx, cy, baseScale);
      const isUpdated = stepData.updated_subs.includes(sub.id);
      const isEscalated = stepData.state === 'ESCALATED';
      const drift = stepData.drifts ? stepData.drifts[sub.id] || 0.02 : 0.02;
      return {
        ...sub,
        proj: p,
        z: p.z,
        isUpdated,
        isEscalated,
        drift
      };
    });

    // 按 Z 深度降序排序 (先画远端，后画近端)
    cubes.sort((a, b) => b.z - a.z);

    // 4. 绘制每个子域立方体
    if (STATE.showSubCubes) {
      const halfSize = 0.36; // 0.72 宽度，留 0.18 缝隙
      cubes.forEach(cube => {
        const isHovered = STATE.hoveredSub === cube.id;

        // 计算 8 个顶点
        const x = cube.x, y = cube.y, z = cube.z;
        const v = [
          project3D(x - halfSize, y - halfSize, z - halfSize, cx, cy, baseScale),
          project3D(x + halfSize, y - halfSize, z - halfSize, cx, cy, baseScale),
          project3D(x + halfSize, y + halfSize, z - halfSize, cx, cy, baseScale),
          project3D(x - halfSize, y + halfSize, z - halfSize, cx, cy, baseScale),
          project3D(x - halfSize, y - halfSize, z + halfSize, cx, cy, baseScale),
          project3D(x + halfSize, y - halfSize, z + halfSize, cx, cy, baseScale),
          project3D(x + halfSize, y + halfSize, z + halfSize, cx, cy, baseScale),
          project3D(x - halfSize, y + halfSize, z + halfSize, cx, cy, baseScale)
        ];

        // 6 个面定义 (按照法线方向)
        const faces = [
          { pts: [0, 1, 2, 3], norm: [ 0,  0, -1] }, // 前
          { pts: [5, 4, 7, 6], norm: [ 0,  0,  1] }, // 后
          { pts: [4, 0, 3, 7], norm: [-1,  0,  0] }, // 左
          { pts: [1, 5, 6, 2], norm: [ 1,  0,  0] }, // 右
          { pts: [3, 2, 6, 7], norm: [ 0,  1,  0] }, // 上
          { pts: [4, 5, 1, 0], norm: [ 0, -1,  0] }  // 下
        ];

        // 确定颜色状态
        let fillColor, strokeColor, glowColor;
        if (cube.isEscalated) {
          fillColor = 'rgba(168, 85, 247, 0.45)';
          strokeColor = '#c084fc';
          glowColor = 'rgba(168, 85, 247, 0.6)';
        } else if (cube.isUpdated) {
          fillColor = 'rgba(0, 242, 254, 0.45)';
          strokeColor = '#00f2fe';
          glowColor = 'rgba(0, 242, 254, 0.6)';
        } else if (cube.drift > 0.1) {
          fillColor = 'rgba(245, 158, 11, 0.35)';
          strokeColor = '#f59e0b';
          glowColor = 'rgba(245, 158, 11, 0.4)';
        } else {
          fillColor = 'rgba(30, 58, 138, 0.32)';
          strokeColor = '#3b82f6';
          glowColor = 'rgba(59, 130, 246, 0.2)';
        }

        if (isHovered) {
          fillColor = 'rgba(255, 255, 255, 0.5)';
          strokeColor = '#ffffff';
        }

        // 绘制可见面
        faces.forEach(f => {
          // 简易背面剔除
          const pA = v[f.pts[0]];
          const pB = v[f.pts[1]];
          const pC = v[f.pts[2]];
          const cross = (pB.x - pA.x) * (pC.y - pA.y) - (pB.y - pA.y) * (pC.x - pA.x);
          if (cross > 0) {
            ctx.fillStyle = fillColor;
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = isHovered ? 2.5 : 1.2;

            ctx.beginPath();
            ctx.moveTo(pA.x, pA.y);
            ctx.lineTo(pB.x, pB.y);
            ctx.lineTo(pC.x, pC.y);
            ctx.lineTo(v[f.pts[3]].x, v[f.pts[3]].y);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
          }
        });

        // 绘制中心编号标牌
        ctx.fillStyle = isHovered ? '#ffffff' : strokeColor;
        ctx.font = 'bold 12px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('#' + cube.id, cube.proj.x, cube.proj.y);
      });
    }

    // 5. 绘制移动激光热源/对撞球 (Laser Heat Fronts)
    if (STATE.showLaser && stepData.lasers) {
      stepData.lasers.forEach(laser => {
        // 将 (0..1) 物理坐标映射到 [-0.65, 0.65] 3D 坐标
        const lx = (laser.x - 0.5) * 1.3;
        const ly = (laser.y - 0.5) * 1.3;
        const lz = (laser.z - 0.5) * 1.3;
        const lp = project3D(lx, ly, lz, cx, cy, baseScale);

        const radius = (laser.r || 0.25) * baseScale * 0.45;

        // 绘制径向渐变激光光核
        const grad = ctx.createRadialGradient(lp.x, lp.y, 2, lp.x, lp.y, radius);
        grad.addColorStop(0, 'rgba(255, 255, 255, 0.95)');
        grad.addColorStop(0.3, 'rgba(244, 63, 94, 0.85)');
        grad.addColorStop(0.7, 'rgba(244, 63, 94, 0.35)');
        grad.addColorStop(1, 'rgba(244, 63, 94, 0)');

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(lp.x, lp.y, radius, 0, Math.PI * 2);
        ctx.fill();

        // 绘制激光辐射光晕光环
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(lp.x, lp.y, radius * 0.4, 0, Math.PI * 2);
        ctx.stroke();
      });
    }

    // 6. 视角指示小罗盘 (左下角)
    ctx.save();
    ctx.translate(55, h - 50);
    const compassScale = 32;
    const cO = project3D(0, 0, 0, 0, 0, compassScale);
    const cX = project3D(1, 0, 0, 0, 0, compassScale);
    const cY = project3D(0, 1, 0, 0, 0, compassScale);
    const cZ = project3D(0, 0, 1, 0, 0, compassScale);

    // X 轴红
    ctx.strokeStyle = '#f43f5e'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cO.x, cO.y); ctx.lineTo(cX.x, cX.y); ctx.stroke();
    // Y 轴绿
    ctx.strokeStyle = '#10b981';
    ctx.beginPath(); ctx.moveTo(cO.x, cO.y); ctx.lineTo(cY.x, cY.y); ctx.stroke();
    // Z 轴蓝
    ctx.strokeStyle = '#00f2fe';
    ctx.beginPath(); ctx.moveTo(cO.x, cO.y); ctx.lineTo(cZ.x, cZ.y); ctx.stroke();
    ctx.restore();
  }

  // --- 5. 洛伦兹风险集中度曲线图 (Lorenz Curve) ---
  function renderLorenzChart() {
    const canvas = DOM.lorenzCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 36, padR = 18, padT = 16, padB = 22;
    const chartW = w - padL - padR;
    const chartH = h - padT - padB;
    const stepData = getCurrentStepData();
    const k_ab = stepData.k_ab;

    // 绘制网格与刻度
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;

    // 80% 警戒阈值参考虚线
    const y80 = padT + chartH * (1 - 0.80);
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.6)';
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(padL, y80);
    ctx.lineTo(padL + chartW, y80);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#fda4af';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText('α=80%', padL - 4, y80 + 3);

    // 洛伦兹曲线数据点 Q_k = 累积风险占比 (模拟实际漂移集中度)
    const points = [];
    let cum = 0;
    // 依当前 stepData.k_ab 调整曲线陡峭度
    const totalDrift = (stepData.drifts || []).reduce((a, b) => a + b, 0) || 1.0;
    const sortedDrifts = [...(stepData.drifts || [0.08, 0.06, 0.04, 0.02, 0.01, 0.01, 0.01, 0.01])].sort((a,b) => b-a);

    points.push({ k: 0, q: 0, x: padL, y: padT + chartH });
    for (let k = 1; k <= 8; k++) {
      cum += sortedDrifts[k - 1];
      const q = Math.min(cum / totalDrift, 1.0);
      const x = padL + (k / 8) * chartW;
      const y = padT + (1 - q) * chartH;
      points.push({ k, q, x, y });
    }

    // 填充曲线下方渐变
    const grad = ctx.createLinearGradient(0, padT, 0, padT + chartH);
    grad.addColorStop(0, 'rgba(0, 242, 254, 0.28)');
    grad.addColorStop(1, 'rgba(0, 242, 254, 0.02)');

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.lineTo(padL + chartW, padT + chartH);
    ctx.closePath();
    ctx.fill();

    // 绘制曲线本身
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.stroke();

    // 突出标出截断决策点 K_t
    const hitPt = points[Math.min(k_ab, 8)];
    ctx.strokeStyle = '#ffffff';
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(hitPt.x, padT + chartH);
    ctx.lineTo(hitPt.x, hitPt.y);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#00f2fe';
    ctx.beginPath();
    ctx.arc(hitPt.x, hitPt.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(hitPt.x, hitPt.y, 2.5, 0, Math.PI * 2);
    ctx.fill();

    // X 轴刻度
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    for (let k = 1; k <= 8; k++) {
      const x = padL + (k / 8) * chartW;
      ctx.fillText(k, x, h - 6);
    }
  }

  // --- 6. 全轨迹 Krylov (PCG) 迭代数演化折线图 ---
  function renderIterHistory() {
    const canvas = DOM.iterHistoryCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 40, padR = 20, padT = 16, padB = 26;
    const chartW = w - padL - padR;
    const chartH = h - padT - padB;
    const steps = getCurrentSteps();
    const totalN = steps.length;
    const maxIterVal = 300;

    // Y 轴网格线
    const gridYVals = [0, 50, 100, 200, 300];
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#64748b';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';

    gridYVals.forEach(v => {
      const y = padT + (1 - v / maxIterVal) * chartH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + chartW, y);
      ctx.stroke();
      ctx.fillText(v, padL - 6, y + 3);
    });

    // 辅助折线绘制函数
    function drawLine(key, color, lineWidth, isDashed = false) {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      if (isDashed) ctx.setLineDash([4, 4]); else ctx.setLineDash([]);
      ctx.beginPath();
      steps.forEach((s, idx) => {
        const val = s[key];
        const x = padL + (idx / (totalN - 1)) * chartW;
        const y = padT + (1 - Math.min(val, maxIterVal) / maxIterVal) * chartH;
        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 绘制四条曲线
    drawLine('iters_svolos', '#f43f5e', 2.0); // Svolos (突波最高)
    drawLine('iters_k3', '#f59e0b', 2.0);     // Fixed K=3 (尖峰暴走)
    drawLine('iters_full', '#3b82f6', 1.5, true); // Full Rebuild
    drawLine('iters_ab', '#00f2fe', 2.8);     // AB-JSR (平稳压平)

    // 当前时间步竖直高亮引导线 (Scrubber)
    const curIdx = STATE.step - 1;
    const curX = padL + (curIdx / (totalN - 1)) * chartW;
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(curX, padT);
    ctx.lineTo(curX, padT + chartH);
    ctx.stroke();
    ctx.setLineDash([]);

    // 在高亮线上打光点
    const curStepData = steps[curIdx];
    const curY_AB = padT + (1 - Math.min(curStepData.iters_ab, maxIterVal) / maxIterVal) * chartH;
    ctx.fillStyle = '#00f2fe';
    ctx.beginPath();
    ctx.arc(curX, curY_AB, 5, 0, Math.PI * 2);
    ctx.fill();

    // X 轴时间步数字
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    for (let i = 0; i < totalN; i += 2) {
      const x = padL + (i / (totalN - 1)) * chartW;
      ctx.fillText('t=' + (i + 1), x, h - 8);
    }
  }

  // --- 7. 端到端累积总耗时柱状/折线对比图 ---
  function renderCumTimeChart() {
    const canvas = DOM.cumTimeCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 40, padR = 20, padT = 16, padB = 26;
    const chartW = w - padL - padR;
    const chartH = h - padT - padB;
    const steps = getCurrentSteps();
    const totalN = steps.length;
    const maxTime = STATE.activeTraj === 'compound' ? 24.0 : 15.0;

    // Y 轴刻度
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#64748b';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';

    const yTicks = STATE.activeTraj === 'compound' ? [0, 5, 10, 15, 20] : [0, 3, 6, 9, 12];
    yTicks.forEach(v => {
      const y = padT + (1 - v / maxTime) * chartH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + chartW, y);
      ctx.stroke();
      ctx.fillText(v + 's', padL - 6, y + 3);
    });

    function drawCumLine(key, color, width) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      steps.forEach((s, idx) => {
        const val = s[key];
        const x = padL + (idx / (totalN - 1)) * chartW;
        const y = padT + (1 - Math.min(val, maxTime) / maxTime) * chartH;
        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    drawCumLine('cum_time_svolos', '#f43f5e', 2.0);
    drawCumLine('cum_time_k3', '#f59e0b', 2.0);
    drawCumLine('cum_time_full', '#3b82f6', 1.8);
    drawCumLine('cum_time_ab', '#00f2fe', 3.0);

    // 当前时间步光标
    const curIdx = STATE.step - 1;
    const curX = padL + (curIdx / (totalN - 1)) * chartW;
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.5)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(curX, padT);
    ctx.lineTo(curX, padT + chartH);
    ctx.stroke();
    ctx.setLineDash([]);

    // 终点耗时标签
    const lastStep = steps[steps.length - 1];
    ctx.font = 'bold 10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#00f2fe';
    ctx.textAlign = 'left';
    ctx.fillText('AB: ' + lastStep.cum_time_ab + 's (🏆)', padL + chartW - 90, padT + (1 - lastStep.cum_time_ab / maxTime) * chartH - 6);
  }

  // --- 8. 四区制响应曲面图 (View 3) ---
  function renderRegimeCurves() {
    const canvas = DOM.regimeCurveCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 50, padR = 30, padT = 24, padB = 36;
    const chartW = w - padL - padR;
    const chartH = h - padT - padB;
    const maxRatio = 2.8;

    // 1.0x 全量重构基准基线
    const yBase = padT + (1 - 1.0 / maxRatio) * chartH;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(padL, yBase);
    ctx.lineTo(padL + chartW, yBase);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#ffffff';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText('1.0x Full Rebuild 基线', padL + chartW, yBase - 6);

    // Y 刻度
    const yRatios = [0.8, 1.0, 1.5, 2.0, 2.5];
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.textAlign = 'right';
    yRatios.forEach(r => {
      const y = padT + (1 - r / maxRatio) * chartH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + chartW, y);
      ctx.stroke();
      ctx.fillText(r.toFixed(1) + 'x', padL - 8, y + 3);
    });

    // 绘制 4 个区制曲线
    Object.keys(FOUR_REGIMES_DATA).forEach(rkey => {
      const reg = FOUR_REGIMES_DATA[rkey];
      const isSelected = STATE.selectedRegime === rkey;
      const alpha = isSelected ? 1.0 : 0.35;
      const lineWidth = isSelected ? 3.5 : 1.8;

      ctx.strokeStyle = reg.color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();

      reg.records.forEach((rec, idx) => {
        const k = rec.K;
        const ratio = rec.ratio_vs_full;
        const x = padL + (k / 8) * chartW;
        const y = padT + (1 - Math.min(ratio, maxRatio) / maxRatio) * chartH;
        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // 画点与最优 K* 标牌
      reg.records.forEach(rec => {
        const k = rec.K;
        const ratio = rec.ratio_vs_full;
        const x = padL + (k / 8) * chartW;
        const y = padT + (1 - Math.min(ratio, maxRatio) / maxRatio) * chartH;

        ctx.fillStyle = reg.color;
        ctx.beginPath();
        ctx.arc(x, y, isSelected ? 4.5 : 3, 0, Math.PI * 2);
        ctx.fill();

        if (k === reg.k_star && isSelected) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, 7, 0, Math.PI * 2);
          ctx.stroke();

          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 11px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.fillText(`K*=${k} (${ratio.toFixed(3)}x)`, x, y - 10);
        }
      });
      ctx.globalAlpha = 1.0;
    });

    // X 刻度
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    for (let k = 0; k <= 8; k++) {
      const x = padL + (k / 8) * chartW;
      ctx.fillText('K=' + k, x, h - 10);
    }
  }

  // --- 9. 经典 2D mass95 渲染引擎 (View 4) ---
  function renderLegacy2D() {
    const canvas = DOM.legacyCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const gridSize = 8;
    const pad = 24;
    const cellSize = Math.min((w - pad * 2) / gridSize, (h - pad * 2) / gridSize);
    const startX = (w - cellSize * gridSize) / 2;
    const startY = (h - cellSize * gridSize) / 2;

    const t = STATE.legacy.time;
    // 2D 激光斑位置沿对角线行进
    const heatX = (0.2 + (t % 10) * 0.07) * gridSize;
    const heatY = (0.2 + (t % 10) * 0.07) * gridSize;

    // 绘制 64 个网格单元
    for (let gy = 0; gy < gridSize; gy++) {
      for (let gx = 0; gx < gridSize; gx++) {
        const cx = startX + gx * cellSize;
        const cy = startY + gy * cellSize;

        // 距离热斑距离
        const dist = Math.hypot(gx + 0.5 - heatX, gy + 0.5 - heatY);
        const heat = Math.exp(-dist * dist / 2.2);

        let fill = 'rgba(30, 58, 138, 0.25)';
        let stroke = 'rgba(255, 255, 255, 0.06)';

        if (heat > 0.45) {
          fill = 'rgba(0, 242, 254, 0.4)';
          stroke = '#00f2fe';
        } else if (heat > 0.15) {
          fill = 'rgba(245, 158, 11, 0.3)';
          stroke = '#f59e0b';
        }

        ctx.fillStyle = fill;
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 1;
        ctx.fillRect(cx + 2, cy + 2, cellSize - 4, cellSize - 4);
        ctx.strokeRect(cx + 2, cy + 2, cellSize - 4, cellSize - 4);

        // 标号
        const id = gy * gridSize + gx;
        if (cellSize > 30) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
          ctx.font = '9px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.fillText(id, cx + cellSize / 2, cy + cellSize / 2 + 3);
        }
      }
    }

    // 绘制热斑光晕
    if (STATE.legacy.showHeat) {
      const px = startX + heatX * cellSize;
      const py = startY + heatY * cellSize;
      const grad = ctx.createRadialGradient(px, py, 4, px, py, cellSize * 2.5);
      grad.addColorStop(0, 'rgba(244, 63, 94, 0.85)');
      grad.addColorStop(0.5, 'rgba(244, 63, 94, 0.25)');
      grad.addColorStop(1, 'rgba(244, 63, 94, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, py, cellSize * 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // --- 10. 更新仪表盘与界面元素 ---
  function updateCockpit() {
    const stepData = getCurrentStepData();
    const steps = getCurrentSteps();
    const totalN = steps.length;

    // 顶部步指示器
    if (DOM.currentStepBadge) DOM.currentStepBadge.textContent = 't = ' + stepData.step;
    if (DOM.totalStepsBadge) DOM.totalStepsBadge.textContent = totalN;

    // 时间轴按键高亮
    const tickBtns = DOM.timeTicksWrapper.querySelectorAll('.tick-btn');
    tickBtns.forEach((btn, idx) => {
      btn.classList.toggle('active', idx + 1 === stepData.step);
    });

    // 策略卡片指标更新
    if (DOM.abBudgetDisplay) DOM.abBudgetDisplay.textContent = `K = ${stepData.k_ab} / 8`;
    if (DOM.abItersDisplay) DOM.abItersDisplay.textContent = `${stepData.iters_ab} 步`;
    if (DOM.abTimeDisplay) DOM.abTimeDisplay.textContent = `${stepData.time_ab}s`;
    if (DOM.abCumTime) DOM.abCumTime.textContent = `${stepData.cum_time_ab}s`;
    if (DOM.abBar) DOM.abBar.style.width = (stepData.k_ab / 8 * 100) + '%';

    if (DOM.fullItersDisplay) DOM.fullItersDisplay.textContent = `${stepData.iters_full} 步`;
    if (DOM.fullTimeDisplay) DOM.fullTimeDisplay.textContent = `${stepData.time_full}s`;
    if (DOM.fullCumTime) DOM.fullCumTime.textContent = `${stepData.cum_time_full}s`;

    if (DOM.k3ItersDisplay) DOM.k3ItersDisplay.textContent = `${stepData.iters_k3} 步`;
    if (DOM.k3TimeDisplay) DOM.k3TimeDisplay.textContent = `${stepData.time_k3}s`;
    if (DOM.k3CumTime) DOM.k3CumTime.textContent = `${stepData.cum_time_k3}s`;
    if (DOM.k3Bar) DOM.k3Bar.style.width = '37.5%';

    if (DOM.svolosItersDisplay) DOM.svolosItersDisplay.textContent = `${stepData.iters_svolos} 步`;
    if (DOM.svolosTimeDisplay) DOM.svolosTimeDisplay.textContent = `${stepData.time_svolos}s`;
    if (DOM.svolosCumTime) DOM.svolosCumTime.textContent = `${stepData.cum_time_svolos}s`;

    // 驾驶舱 1: G_t 弥散度
    if (DOM.gtValueBadge) DOM.gtValueBadge.textContent = 'G_t = ' + stepData.gt.toFixed(4);
    if (DOM.gtGaugeFill) {
      const pct = Math.min((stepData.gt / 0.5) * 100, 100);
      DOM.gtGaugeFill.style.width = pct + '%';
    }
    if (DOM.gtExplainText) {
      if (stepData.state === 'ESCALATED') {
        DOM.gtExplainText.innerHTML = `<strong class="text-rose">🚨 触发激增阈值！</strong> G_t = ${stepData.gt.toFixed(4)} ≥ 0.22，施密特控制器瞬间升满 K=8 全量重构保命！`;
      } else if (stepData.state === 'COOLDOWN') {
        DOM.gtExplainText.innerHTML = `<strong class="text-amber">⏳ 冷却滞回清洗期：</strong> G_t 回落，控制器强制维持 2 步 K=8 兜底，防止微观残差次生灾害。`;
      } else {
        DOM.gtExplainText.textContent = `平稳运行：G_t = ${stepData.gt.toFixed(4)} < 0.22，处于平稳节能区制，按洛伦兹集中度裁定预算。`;
      }
    }

    // 驾驶舱状态指示灯
    if (DOM.schmittDot && DOM.schmittText) {
      DOM.schmittDot.className = 'dot-sm ' + (stepData.state === 'ESCALATED' ? 'red' : (stepData.state === 'COOLDOWN' ? 'amber' : 'green'));
      DOM.schmittText.textContent = stepData.state + ' 闭环状态';
    }

    // 驾驶舱 2: 洛伦兹预算
    if (DOM.lorenzDecisionBadge) DOM.lorenzDecisionBadge.textContent = `Q_${stepData.k_ab} ≥ 80% → K_t = ${stepData.k_ab}`;
    if (DOM.lorenzKVal) DOM.lorenzKVal.textContent = stepData.k_ab;

    // 驾驶舱 3: 迭代消火栓
    const isFire = stepData.iters_k3 > 100 || stepData.iters_svolos > 100;
    if (DOM.iterationFireBox) DOM.iterationFireBox.classList.toggle('fire', isFire);
    if (DOM.fireStatusPill) {
      DOM.fireStatusPill.className = 'alert-status-pill ' + (isFire ? 'danger' : 'success');
      DOM.fireStatusPill.textContent = isFire ? '🚨 算力饥饿火情爆发！' : '收敛稳健 · 无火情';
    }

    const maxI = 300;
    if (DOM.iterBarAB) DOM.iterBarAB.style.width = (stepData.iters_ab / maxI * 100) + '%';
    if (DOM.iterNumAB) DOM.iterNumAB.textContent = stepData.iters_ab + ' 步';
    if (DOM.iterBarFull) DOM.iterBarFull.style.width = (stepData.iters_full / maxI * 100) + '%';
    if (DOM.iterNumFull) DOM.iterNumFull.textContent = stepData.iters_full + ' 步';
    if (DOM.iterBarK3) DOM.iterBarK3.style.width = (stepData.iters_k3 / maxI * 100) + '%';
    if (DOM.iterNumK3) DOM.iterNumK3.textContent = stepData.iters_k3 + ' 步';
    if (DOM.iterBarSvolos) DOM.iterBarSvolos.style.width = (stepData.iters_svolos / maxI * 100) + '%';
    if (DOM.iterNumSvolos) DOM.iterNumSvolos.textContent = stepData.iters_svolos + ' 步';

    if (DOM.iterCommentText) {
      if (isFire) {
        DOM.iterCommentText.innerHTML = `<strong class="text-rose">💥 火警踩踏：</strong> 固定预算 K=3 饿死剧变子域，迭代激增至 ${Math.max(stepData.iters_k3, stepData.iters_svolos)} 步；AB-JSR 因全量安全网保持最优 ${stepData.iters_ab} 步！`;
      } else {
        DOM.iterCommentText.textContent = `✔ AB-JSR 保持最优收敛品质，与每步全量重建性能完全一致（比值 1.00）。`;
      }
    }

    // 驾驶舱 4: Oracle 对照
    if (DOM.oracleKStar) DOM.oracleKStar.textContent = `${stepData.k_oracle} / 8`;
    if (DOM.abKActual) DOM.abKActual.textContent = `${stepData.k_ab} / 8`;
    if (DOM.oracleTimeBest) DOM.oracleTimeBest.textContent = `${Math.min(stepData.time_ab, stepData.time_full)}s`;
    if (DOM.abRegretVal) DOM.abRegretVal.textContent = `+${stepData.regret_ab_pct}%`;
    if (DOM.oracleHitBadge) {
      const hit = stepData.k_ab === stepData.k_oracle;
      DOM.oracleHitBadge.textContent = hit ? '🎯 完美命中 Oracle' : '因果安全冗余';
      DOM.oracleHitBadge.className = 'badge-tag-sm ' + (hit ? 'highlight' : '');
    }

    // 解说词卡片
    if (DOM.expPhaseTag) DOM.expPhaseTag.textContent = stepData.phase;
    if (DOM.expStepTitle) DOM.expStepTitle.textContent = stepData.title;
    if (DOM.expStepSubtitle) DOM.expStepSubtitle.textContent = stepData.subtitle;
    if (DOM.expPhysicsBody) DOM.expPhysicsBody.innerHTML = stepData.p_body;
    if (DOM.expTrapBody) DOM.expTrapBody.innerHTML = stepData.trap_body;
    if (DOM.expDecisionBody) DOM.expDecisionBody.innerHTML = stepData.dec_body;
    if (DOM.expLedgerBody) DOM.expLedgerBody.innerHTML = stepData.led_body;

    // 重绘所有画布
    render3DStage();
    renderLorenzChart();
    renderIterHistory();
    renderCumTimeChart();
  }

  // --- 11. 故事模式逻辑 (View 2) ---
  function updateStoryMode() {
    const ch = STORY_CHAPTERS[STATE.storyIndex];
    if (!ch || !DOM.storyContentBody) return;

    // 渲染章节内容
    DOM.storyContentBody.innerHTML = `
      <div class="story-card">
        <div class="story-chapter-header">
          <span class="badge-tag highlight">${ch.tag}</span>
          <h3>${ch.title}</h3>
          <p class="story-sub">${ch.subtitle}</p>
        </div>
        ${ch.content_html}
      </div>
    `;

    // 更新按钮激活状态
    DOM.storyStepBtns.forEach((btn, idx) => {
      btn.classList.toggle('active', idx === STATE.storyIndex);
    });

    // Prev / Next 按钮禁用逻辑
    if (DOM.btnStoryPrev) DOM.btnStoryPrev.disabled = STATE.storyIndex === 0;
    if (DOM.btnStoryNext) {
      DOM.btnStoryNext.textContent = STATE.storyIndex === STORY_CHAPTERS.length - 1 ? '🚀 进入实验室体验实战' : '下一讲 →';
    }
  }

  // --- 12. 视图切换控制器 ---
  function switchView(viewName) {
    STATE.activeView = viewName;

    // 切换 Header 按钮样式
    const tabMap = {
      modern: DOM.tabModern,
      story: DOM.tabStory,
      regimes: DOM.tabRegimes,
      legacy: DOM.tabLegacy
    };
    Object.keys(tabMap).forEach(k => {
      if (tabMap[k]) tabMap[k].classList.toggle('active', k === viewName);
    });

    // 切换视口容器显隐
    const viewMap = {
      modern: DOM.viewModern,
      story: DOM.viewStory,
      regimes: DOM.viewRegimes,
      legacy: DOM.viewLegacy
    };
    Object.keys(viewMap).forEach(k => {
      if (viewMap[k]) viewMap[k].classList.toggle('active', k === viewName);
    });

    // 视口切换触发相应重绘
    if (viewName === 'modern') {
      updateCockpit();
    } else if (viewName === 'story') {
      updateStoryMode();
    } else if (viewName === 'regimes') {
      renderRegimeCurves();
    } else if (viewName === 'legacy') {
      renderLegacy2D();
    }
  }

  // --- 13. 时间轴步进器生成与控制 ---
  function buildTimelineTicks() {
    const steps = getCurrentSteps();
    if (!DOM.timeTicksWrapper) return;
    DOM.timeTicksWrapper.innerHTML = '';

    steps.forEach((s, idx) => {
      const btn = document.createElement('button');
      btn.className = 'tick-btn' + (idx + 1 === STATE.step ? ' active' : '');
      btn.textContent = s.step;
      btn.title = `跳转到时间步 t = ${s.step}`;
      btn.addEventListener('click', () => {
        STATE.step = s.step;
        updateCockpit();
      });
      DOM.timeTicksWrapper.appendChild(btn);
    });
  }

  function nextStep() {
    const steps = getCurrentSteps();
    if (STATE.step < steps.length) {
      STATE.step++;
    } else {
      STATE.step = 1; // 循环播放
    }
    updateCockpit();
  }

  function prevStep() {
    if (STATE.step > 1) {
      STATE.step--;
      updateCockpit();
    }
  }

  function togglePlay() {
    STATE.isPlaying = !STATE.isPlaying;
    if (DOM.lblPlayPause) {
      DOM.lblPlayPause.textContent = STATE.isPlaying ? '暂停动画' : '播放仿真动画';
    }
    if (DOM.btnPlayPause) {
      const icon = DOM.btnPlayPause.querySelector('.icon');
      if (icon) icon.textContent = STATE.isPlaying ? '⏸' : '▶';
    }

    if (STATE.isPlaying) {
      STATE.playTimer = setInterval(nextStep, STATE.speed);
    } else {
      clearInterval(STATE.playTimer);
      STATE.playTimer = null;
    }
  }

  // --- 14. 3D 画布鼠标交互 (旋转与悬停拾取) ---
  function setup3DInteractions() {
    const canvas = DOM.stageCanvas;
    if (!canvas) return;

    canvas.addEventListener('mousedown', (e) => {
      STATE.camera.isDragging = true;
      STATE.camera.lastX = e.clientX;
      STATE.camera.lastY = e.clientY;
    });

    window.addEventListener('mousemove', (e) => {
      if (!STATE.camera.isDragging) {
        // 鼠标悬停拾取检查
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        if (mx >= 0 && mx <= canvas.width && my >= 0 && my <= canvas.height) {
          const cx = canvas.width * 0.48;
          const cy = canvas.height * 0.52;
          const baseScale = 210 * STATE.camera.zoom;

          let found = null;
          SUBDOMAINS_3D.forEach(sub => {
            const p = project3D(sub.x, sub.y, sub.z, cx, cy, baseScale);
            const dist = Math.hypot(mx - p.x, my - p.y);
            if (dist < 38) found = sub.id;
          });

          if (found !== STATE.hoveredSub) {
            STATE.hoveredSub = found;
            render3DStage();
            // 显示 tooltip
            if (found !== null && DOM.cubeTooltip) {
              const sdata = getCurrentStepData();
              DOM.cubeTooltip.classList.remove('hidden');
              if (DOM.ttCubeId) DOM.ttCubeId.textContent = found;
              if (DOM.ttCubeCoord) DOM.ttCubeCoord.textContent = SUBDOMAINS_3D[found].label;
              const isUp = sdata.updated_subs.includes(found);
              const drift = sdata.drifts ? sdata.drifts[found] || 0.04 : 0.04;
              if (DOM.ttCubeAge) DOM.ttCubeAge.textContent = isUp ? '0 步 (刚刷新)' : '2 步 (复用中)';
              if (DOM.ttCubeDrift) DOM.ttCubeDrift.textContent = drift.toFixed(4);
              if (DOM.ttCubeScore) DOM.ttCubeScore.textContent = (drift * 1.2).toFixed(4);
              if (DOM.ttCubeAction) {
                DOM.ttCubeAction.textContent = isUp ? '本步重构刷新' : '安全复用缓存';
                DOM.ttCubeAction.className = 'decision-badge ' + (isUp ? 'text-win' : '');
              }
            } else if (DOM.cubeTooltip) {
              DOM.cubeTooltip.classList.add('hidden');
            }
          }
        } else {
          if (STATE.hoveredSub !== null) {
            STATE.hoveredSub = null;
            render3DStage();
            if (DOM.cubeTooltip) DOM.cubeTooltip.classList.add('hidden');
          }
        }
        return;
      }

      // 拖拽旋转
      const dx = e.clientX - STATE.camera.lastX;
      const dy = e.clientY - STATE.camera.lastY;
      STATE.camera.rotY += dx * 0.5;
      STATE.camera.rotX += dy * 0.5;
      STATE.camera.rotX = Math.max(-80, Math.min(80, STATE.camera.rotX));
      STATE.camera.lastX = e.clientX;
      STATE.camera.lastY = e.clientY;
      render3DStage();
    });

    window.addEventListener('mouseup', () => {
      STATE.camera.isDragging = false;
    });

    // 滚轮缩放
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      STATE.camera.zoom += e.deltaY * -0.001;
      STATE.camera.zoom = Math.max(0.6, Math.min(1.8, STATE.camera.zoom));
      render3DStage();
    }, { passive: false });
  }

  // --- 15. 事件监听器初始化与全局挂载 ---
  function initEventListeners() {
    // 视图切换
    if (DOM.tabModern) DOM.tabModern.addEventListener('click', () => switchView('modern'));
    if (DOM.tabStory) DOM.tabStory.addEventListener('click', () => switchView('story'));
    if (DOM.tabRegimes) DOM.tabRegimes.addEventListener('click', () => switchView('regimes'));
    if (DOM.tabLegacy) DOM.tabLegacy.addEventListener('click', () => switchView('legacy'));

    // 控制栏按钮
    if (DOM.btnPlayPause) DOM.btnPlayPause.addEventListener('click', togglePlay);
    if (DOM.btnStepPrev) DOM.btnStepPrev.addEventListener('click', prevStep);
    if (DOM.btnStepNext) DOM.btnStepNext.addEventListener('click', nextStep);
    if (DOM.btnReset) DOM.btnReset.addEventListener('click', () => {
      STATE.step = 1;
      if (STATE.isPlaying) togglePlay();
      updateCockpit();
    });

    // 轨迹切换
    if (DOM.btnTrajCompound) DOM.btnTrajCompound.addEventListener('click', () => {
      STATE.activeTraj = 'compound';
      DOM.btnTrajCompound.classList.add('active');
      if (DOM.btnTrajStaccato) DOM.btnTrajStaccato.classList.remove('active');
      STATE.step = 1;
      buildTimelineTicks();
      updateCockpit();
    });

    if (DOM.btnTrajStaccato) DOM.btnTrajStaccato.addEventListener('click', () => {
      STATE.activeTraj = 'staccato';
      DOM.btnTrajStaccato.classList.add('active');
      if (DOM.btnTrajCompound) DOM.btnTrajCompound.classList.remove('active');
      STATE.step = 1;
      buildTimelineTicks();
      updateCockpit();
    });

    // 播放速度调节
    DOM.speedBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        DOM.speedBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        STATE.speed = parseInt(btn.getAttribute('data-speed'), 10) || 800;
        if (STATE.isPlaying) {
          clearInterval(STATE.playTimer);
          STATE.playTimer = setInterval(nextStep, STATE.speed);
        }
      });
    });

    // 3D 图层显示开关
    if (DOM.chkShowLaser) DOM.chkShowLaser.addEventListener('change', (e) => {
      STATE.showLaser = e.target.checked;
      render3DStage();
    });
    if (DOM.chkShowSubCubes) DOM.chkShowSubCubes.addEventListener('change', (e) => {
      STATE.showSubCubes = e.target.checked;
      render3DStage();
    });
    if (DOM.chkShowCoarseMesh) DOM.chkShowCoarseMesh.addEventListener('change', (e) => {
      STATE.showCoarse = e.target.checked;
      render3DStage();
    });

    // 故事模式导航
    DOM.storyStepBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.getAttribute('data-sidx'), 10);
        STATE.storyIndex = idx;
        updateStoryMode();
      });
    });

    if (DOM.btnStoryPrev) DOM.btnStoryPrev.addEventListener('click', () => {
      if (STATE.storyIndex > 0) {
        STATE.storyIndex--;
        updateStoryMode();
      }
    });

    if (DOM.btnStoryNext) DOM.btnStoryNext.addEventListener('click', () => {
      if (STATE.storyIndex < STORY_CHAPTERS.length - 1) {
        STATE.storyIndex++;
        updateStoryMode();
      } else {
        switchView('modern');
      }
    });

    // 四区制卡片点击
    DOM.regimeCards.forEach(card => {
      card.addEventListener('click', () => {
        DOM.regimeCards.forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        STATE.selectedRegime = card.getAttribute('data-rkey');
        renderRegimeCurves();
      });
    });

    // 经典 2D 动画循环
    setInterval(() => {
      if (STATE.activeView === 'legacy') {
        STATE.legacy.time += 0.35;
        renderLegacy2D();
      }
    }, 120);

    // 挂载 3D 鼠标拖拽
    setup3DInteractions();
  }

  // --- 16. 初始化启动入口 ---
  function init() {
    buildTimelineTicks();
    initEventListeners();
    updateCockpit();
    updateStoryMode();
    renderRegimeCurves();
    renderLegacy2D();
    console.log('🚀 AB-JSR 2.0 HPC Educational Simulation Lab initialized successfully.');
  }

  // 页面就绪后自启动
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();

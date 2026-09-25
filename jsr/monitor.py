"""
================================================================================
模块名称: corrected_phase2.monitor
功能定位: JSR (Joint Stateful Repair) 有状态漂移与陈旧风险感知器
================================================================================

【导读：为什么需要这个模块？】
在时间推进物理仿真中，矩阵 A_t 随物理界面移动而演化。传统的无状态做法（如只看 A_t - A_{t-1}）
存在严重盲区——如果物理场每步只产生微弱变化，单步残差看似安全，但多步累积下来旧预条件器早
已严重失效（即经典的“温水煮青蛙”现象）。

本模块充当 JSR 架构的“前沿雷达”：
1. 【有状态感知 (Stateful)】: 它记录每个子域上次构建的时间戳 (build_state) 和存活代数 (age)，
   计算相对于“该子域上次求逆时的基准状态”的累积相对 Frobenius 漂移量。
2. 【因果律保证 (Causal)】: 严格禁止前瞻！只能读取已提交的历史矩阵字典 (state_matrices)，
   杜绝一切面向未来的数据泄露。
3. 【粗空间风险联动】: 不仅监控局部子域，还综合评估低频宏观粗空间的陈旧度，为两级联合维护
   提供量化决策依据。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MonitorConfig:
    """
    监控器超参数配置项 (冻结参数，不可变数据类)
    
    属性说明:
    - age_weight: 存活年龄惩罚系数 (默认 0.20)。
      公式: risk = drift * (1.0 + age_weight * age)
      含义: 某个子域即使当前漂移较小，但如果存活了 5 代 (age=5)，其名义风险也会翻倍 (1 + 0.2*5 = 2.0)，
      强制打破慢性累积，促使算法定期淘汰老化缓存。
    - coarse_mean_weight: 粗空间风险计算中，全部局部子域平均风险的权重 (默认 0.70)。
    - coarse_max_weight: 粗空间风险计算中，局部子域最大单点风险的权重 (默认 0.30)。
    """
    age_weight: float = 0.20
    coarse_mean_weight: float = 0.70
    coarse_max_weight: float = 0.30


@dataclass(frozen=True)
class RiskSnapshot:
    """
    状态风险快照 (不可变对象，用于在时间步转换决策前固化当前所有的风险指标)
    
    属性说明:
    - previous_state: 上一个物理时间步 (如 t=3)
    - current_state:  当前即将求解的时间步 (如 t=4)
    - local_drift:    元组，记录每个子域自上次重构以来的纯物理相对 Frobenius 漂移量 ||ΔA_i||_F / ||A_i||_F
    - local_risk:     元组，记录每个子域叠加存活年龄惩罚后的综合风险值
    - local_age:      元组，记录每个局部子域缓存已存活的时间步数 (Age)
    - coarse_matrix_risk: 全局粗网格矩阵的综合陈旧风险评估值
    - coarse_age:     全局粗网格缓存已存活的时间步数
    - trajectory:     当前运行的演化轨迹名称 (如 baseline_moving_local 或 stress_moving_interface)
    - basis_fixed:    粗空间投影基是否固定 (Phase 4 默认为 True)
    - future_information_used: 因果标志位，恒为 False，用于形式化审计是否泄露未来数据
    """
    previous_state: int
    current_state: int
    local_drift: Tuple[float, ...]
    local_risk: Tuple[float, ...]
    local_age: Tuple[int, ...]
    coarse_matrix_risk: float
    coarse_age: int
    trajectory: str = ""
    basis_fixed: bool = True
    future_information_used: bool = False

    @property
    def max_local_risk(self) -> float:
        """获取所有局部子域中的最高单点风险值 (用于触发局部硬预算熔断)"""
        return max(self.local_risk) if self.local_risk else 0.0

    @property
    def mean_local_risk(self) -> float:
        """获取全场所有局部子域的平均风险值"""
        return float(sum(self.local_risk) / max(len(self.local_risk), 1))

    def to_dict(self) -> Dict[str, object]:
        """将不可变快照转换为标准字典，用于序列化记录到实验 JSONL 日志中"""
        return {
            "previous_state": int(self.previous_state),
            "current_state": int(self.current_state),
            "local_drift": list(self.local_drift),
            "local_risk": list(self.local_risk),
            "local_age": list(self.local_age),
            "coarse_matrix_risk": float(self.coarse_matrix_risk),
            "coarse_age": int(self.coarse_age),
            "trajectory": self.trajectory,
            "basis_fixed": bool(self.basis_fixed),
            "future_information_used": bool(self.future_information_used),
        }


def compute_snapshot(
    current: Dict[str, np.ndarray],
    domains: Sequence[np.ndarray],
    local_build_states: Sequence[int],
    local_ages: Iterable[int],
    state_matrices: Dict[int, Dict[str, np.ndarray]],
    coarse_build_state: int,
    previous_state: int,
    current_state: int,
    config: MonitorConfig = MonitorConfig(),
    trajectory: str = "",
) -> RiskSnapshot:
    """
    【核心感知函数】：计算当前状态与历史基准之间的多层风险快照
    
    设计准则:
    1. 严格遵守因果律: 仅使用当前矩阵 ``current`` 以及先前已经产生并缓存在 ``state_matrices`` 中的历史状态。
       绝不允许传入或查询任何大于 ``current_state`` 的未来算子。
    2. 计算开销极低: 仅针对各子域的非零元执行切片差分，不执行任何矩阵求逆或求解操作，毫秒级完成。
    
    参数说明:
    - current: 当前即将求解的时间步矩阵 CSR 字典 (包含 'indptr', 'indices', 'data')
    - domains: 子域行索引切分列表，domains[i] 记录子域 i 拥有的全局节点行号
    - local_build_states: 数组，记录每个子域上一次求逆重构时对应的时间步 (如 [0, 0, 2, ...])
    - local_ages: 可迭代整数，记录每个子域目前的存活年龄
    - state_matrices: 状态缓存字典 {state_id: matrix_csr}，只存储历史已观测过的矩阵
    - coarse_build_state: 粗网格当前缓存是基于哪一个历史步组装求逆的
    - previous_state: 前一时间步编号
    - current_state: 当前时间步编号
    - config: 监控配置项 (权重因子)
    - trajectory: 轨迹名称字符串
    """
    ages = tuple(int(value) for value in local_ages)
    
    # 维度一致性基础校验
    if len(ages) != len(domains) or len(local_build_states) != len(domains):
        raise ValueError("local metadata and domains have different lengths")
    if int(coarse_build_state) not in state_matrices:
        raise ValueError("coarse reference state is not available")
        
    current_pattern = np.asarray(current["indptr"])
    drifts = []
    risks = []

    # =========================================================================
    # 第一步：遍历每一个局部子域，计算相对物理漂移量与年龄加权风险
    # =========================================================================
    for block, (build_state, age, indices) in enumerate(
        zip(local_build_states, ages, domains)
    ):
        # 1. 提取该子域构建时的基准历史矩阵 (有状态记忆溯源)
        if int(build_state) not in state_matrices:
            raise ValueError("local reference state is not available")
        reference = state_matrices[int(build_state)]
        
        # 检查稀疏结构是否改变 (演化算子要求数值变化但稀疏拓扑不变)
        if not np.array_equal(reference["indptr"], current_pattern):
            raise ValueError("local reference has a different sparse pattern")
            
        # 2. 计算当前矩阵与基准历史矩阵的数值差分向量 ΔA = A_current - A_reference
        delta = np.asarray(current["data"], dtype=np.float64) - np.asarray(
            reference["data"], dtype=np.float64
        )
        
        # 3. 计算属于该局部子域所有行的差分平方和与当前矩阵范数平方和
        #    利用稀疏 CSR 的 indptr 范围定位对应非零元，避免稠密化带来的内存与计算浪费
        delta_sq = 0.0
        current_sq = 0.0
        for row in np.asarray(indices, dtype=np.int64):
            lo = int(current["indptr"][int(row)])
            hi = int(current["indptr"][int(row) + 1])
            delta_sq += float(np.dot(delta[lo:hi], delta[lo:hi]))
            current_sq += float(np.dot(current["data"][lo:hi], current["data"][lo:hi]))
            
        # 4. 计算子域相对 Frobenius 漂移: drift = ||ΔA_i||_F / ||A_{current, i}||_F
        drift = math.sqrt(delta_sq) / max(math.sqrt(current_sq), 1.0e-12)
        drifts.append(float(drift))
        
        # 5. 叠加存活年龄惩罚: risk = drift * (1 + 0.20 * age)
        #    存活越久，抗漂移能力越弱，名义风险放大
        risks.append(float(drift * (1.0 + config.age_weight * age)))

    # =========================================================================
    # 第二步：评估全局粗空间的陈旧与低频泄漏风险
    # =========================================================================
    # 1. 提取粗网格构建时的全局参考矩阵
    coarse_reference = state_matrices[int(coarse_build_state)]
    if not np.array_equal(coarse_reference["indptr"], current_pattern):
        raise ValueError("coarse reference has a different sparse pattern")
        
    # 2. 计算全局算子的宏观漂移率
    coarse_delta = np.asarray(current["data"], dtype=np.float64) - np.asarray(
        coarse_reference["data"], dtype=np.float64
    )
    coarse_norm = max(float(np.linalg.norm(np.asarray(current["data"], dtype=np.float64))), 1.0e-12)
    coarse_global_drift = float(np.linalg.norm(coarse_delta) / coarse_norm)
    
    # 3. 粗空间风险加权合成:
    #    粗空间吸收的是跨子域宏观误差，既受整体平均漂移影响 (70%)，也容易被单点最剧烈的突变撕裂 (30%)
    coarse_risk = (
        config.coarse_mean_weight * float(np.mean(risks))
        + config.coarse_max_weight * float(np.max(risks) if risks else 0.0)
    )
    # 取宏观漂移与局部合成风险的包络上界，确保保守安全
    coarse_risk = max(coarse_risk, coarse_global_drift)

    # =========================================================================
    # 第三步：打包生成不可变快照并返回
    # =========================================================================
    return RiskSnapshot(
        previous_state=int(previous_state),
        current_state=int(current_state),
        local_drift=tuple(drifts),
        local_risk=tuple(risks),
        local_age=ages,
        coarse_matrix_risk=float(coarse_risk),
        coarse_age=int(max(0, current_state - int(coarse_build_state))),
        trajectory=str(trajectory),
    )

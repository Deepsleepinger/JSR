"""
================================================================================
模块名称: corrected_phase2.selector
功能定位: JSR (Joint Stateful Repair) 算力成本驱动的自适应动作选择器与熔断调度器
================================================================================

【导读：为什么需要这个模块？】
在感知器 (monitor.py) 提供了各个子域的漂移量与粗空间风险后，系统面临最关键的抉择：
“面对当前的物理演化状态，我们究竟该复用、只更新高危子域、只更新粗网格，还是联合刷新？”

本模块充当 JSR 的“高级决策大脑”：
1. 【算力账本驱动 (Cost-Driven)】:
   不凭感觉决策，而是量化评估每个候选动作的“预估 Setup 开销”与“预估 Krylov 迭代开销”，
   在所有满足安全预算的动作中，挑选能让总耗时最短的方案 (贪心成本优化)。
2. 【安全预算硬约束 (Safety Budgets)】:
   设立局部软/硬风险预算与寿命上限 (max_local_age / max_coarse_age)。一旦某个子域老化超标
   或漂移剧烈，强制剥夺“复用”资格，确保数值稳定性。
3. 【确定性熔断升级梯队 (Deterministic Escalation)】:
   若某步因为扰动突变导致求解失败，定义清晰的阶梯式升级路径:
   reuse -> local_partial -> joint_partial -> full_rebuild，绝不陷入死循环。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import statistics
from typing import Dict, Iterable, List, Optional, Tuple

from .monitor import RiskSnapshot


# 预注册的 6 类合法动作执行优先级排序
ACTION_ORDER = (
    "reuse",            # 1. 完全复用旧预条件器 (Setup 开销为 0)
    "local_partial",    # 2. 仅局部刷新选中的高危子块 (粗网格陈旧)
    "coarse_refresh",   # 3. 仅重新组装求逆全局粗网格 (局部块全部复用)
    "joint_partial",    # 4. ★ 联合维护: 局部高危块求逆 + 全局粗网格同步刷新 (JSR 主打动作)
    "full_local_only",  # 5. 重构全部局部子域，粗网格保持陈旧
    "full_rebuild",     # 6. 全量推倒重构: 所有局部块 + 粗网格全部从头重算 (保底兜底)
)


@dataclass(frozen=True)
class SelectorConfig:
    """
    【自适应选择器决策参数与成本先验】
    
    属性说明:
    - local_soft_budget: 局部子域软风险阈值 (默认 0.50)。子域风险超过此值即被标记为必须重构。
    - local_hard_budget: 局部子域硬风险阈值 (默认 2.50)。一旦最大单点风险破表，禁止任何局部修补，强制全量重构。
    - coarse_soft_budget: 粗空间软风险阈值 (默认 0.80)。超过此值粗空间必须重构。
    - coarse_hard_budget: 粗空间硬风险阈值 (默认 1.80)。
    - max_local_age: 局部子域最大允许存活代数 (默认 2 代)。超过 2 步未刷新的子域强制换血，防止慢性累积。
    - max_coarse_age: 粗空间最大允许存活代数 (默认 2 代)。
    - max_partial_fraction: 局部修补的最大比例上限 (默认 0.75，即 75%)。如果需要修补的块超过 75%，直接全量重构更划算。
    - local_seconds_per_block_prior: 单个局部块求逆耗时的硬件先验估计 (默认 0.095s)。
    - coarse_seconds_prior: 粗网格组装求逆耗时的硬件先验估计 (默认 0.75s)。
    - solve_seconds_prior: Krylov 求解耗时的先验估计 (默认 5.0s)。
    """
    local_soft_budget: float = 0.50
    local_hard_budget: float = 2.50
    coarse_soft_budget: float = 0.80
    coarse_hard_budget: float = 1.80
    max_local_age: int = 2
    max_coarse_age: int = 2
    max_partial_fraction: float = 0.75
    local_seconds_per_block_prior: float = 0.095
    coarse_seconds_prior: float = 0.75
    solve_seconds_prior: float = 5.0


@dataclass(frozen=True)
class RepairAction:
    """
    【候选修复动作对象】
    
    封装了某个具体动作的所有决策元数据与预期耗时:
    - action_id: 动作唯一标识符 (来自 ACTION_ORDER)
    - selected_local_blocks: 本动作决定重新求逆的局部子域编号元组
    - refresh_coarse_matrix: 是否同步刷新粗网格矩阵
    - refresh_coarse_basis: 是否重新计算粗空间投影基函数 (固定基实验中恒为 False)
    - safety_admissible: 安全准入标志位 (是否满足所有的软硬安全预算)
    - rejection_reason: 如果被拒绝准入，记录具体原因 (如 "local risk exceeds hard budget")
    - predicted_total_seconds: 预估端到端总耗时 (Setup 预测 + Solve 预测)
    - predicted_repair_seconds: 预估维护耗时 (Setup 预测)
    - predicted_solve_seconds: 预估求解耗时 (Solve 预测)
    - used_future_information: 因果标志位，恒为 False
    """
    action_id: str
    selected_local_blocks: Tuple[int, ...]
    refresh_coarse_matrix: bool
    refresh_coarse_basis: bool
    safety_admissible: bool
    rejection_reason: str
    predicted_total_seconds: float
    predicted_repair_seconds: float
    predicted_solve_seconds: float
    used_future_information: bool = False

    def to_dict(self) -> Dict[str, object]:
        """序列化输出到日志记录中"""
        return {
            "action_id": self.action_id,
            "selected_local_blocks": list(self.selected_local_blocks),
            "refresh_coarse_matrix": bool(self.refresh_coarse_matrix),
            "refresh_coarse_basis": bool(self.refresh_coarse_basis),
            "safety_admissible": bool(self.safety_admissible),
            "rejection_reason": self.rejection_reason,
            "predicted_total_seconds": float(self.predicted_total_seconds),
            "predicted_repair_seconds": float(self.predicted_repair_seconds),
            "predicted_solve_seconds": float(self.predicted_solve_seconds),
            "used_future_information": bool(self.used_future_information),
        }


@dataclass(frozen=True)
class ActionObservation:
    """单个动作的历史执行实测记录样本"""
    total_seconds: float
    solve_seconds: float
    iterations: int
    success: bool


@dataclass
class OnlineHistory:
    """
    【在线历史成本追踪器 (Online Cost Profiler)】
    在仿真推进过程中，动态收集各动作的历史耗时中位数，用于校准成本预测。
    """
    observations: Dict[str, List[ActionObservation]] = field(default_factory=dict)
    max_observations_per_action: int = 32

    def update(self, action_id: str, total_seconds: float, solve_seconds: float,
               iterations: int, success: bool) -> None:
        """记录一次动作的实测开销"""
        values = self.observations.setdefault(action_id, [])
        values.append(ActionObservation(float(total_seconds), float(solve_seconds),
                                        int(iterations), bool(success)))
        # 保留最近 32 个滑动窗口样本，适应物理阶段变化
        if len(values) > self.max_observations_per_action:
            del values[:-self.max_observations_per_action]

    def estimate(self, action_id: str, config: SelectorConfig,
                 selected_count: int = 0) -> Tuple[float, float]:
        """
        【成本预估核心函数】: 输出 (预估维护耗时或总耗时, 预估求解耗时)
        
        策略:
        1. 优先使用滑动窗口内该动作的历史实测中位数 (Median)，抗偶然抖动；
        2. 如果尚无历史样本，退回到配置项中声明的线性硬件成本先验公式计算。
        """
        values = self.observations.get(action_id, [])
        if values:
            return (
                float(statistics.median(value.total_seconds for value in values)),
                float(statistics.median(value.solve_seconds for value in values)),
            )
        # 无样本时的先验线性模型:
        if action_id == "reuse":
            return 0.0, config.solve_seconds_prior
        if action_id == "coarse_refresh":
            return config.coarse_seconds_prior, config.solve_seconds_prior
        if action_id == "full_local_only":
            return config.local_seconds_per_block_prior * 144.0, config.solve_seconds_prior
        if action_id == "full_rebuild":
            return config.coarse_seconds_prior + config.local_seconds_per_block_prior * 144.0, config.solve_seconds_prior
        # 局部修补: 耗时与选中的块数成正比
        return config.local_seconds_per_block_prior * max(1, selected_count), config.solve_seconds_prior


def ranked_blocks(snapshot: RiskSnapshot, count: int) -> Tuple[int, ...]:
    """将局部子域按风险从大到小排序，返回前 count 个子域的索引"""
    ranking = sorted(range(len(snapshot.local_risk)),
                     key=lambda index: (-snapshot.local_risk[index], index))
    return tuple(ranking[:max(0, int(count))])


def required_local_blocks(snapshot: RiskSnapshot, config: SelectorConfig) -> Tuple[int, ...]:
    """
    【筛选必须修补的高危子域集合】
    标准: 只要满足以下任一条件，必须强制重构:
    1. risk > local_soft_budget (单点物理风险超标);
    2. age > max_local_age (缓存存活超过 2 代，强制换血防长尾累积)。
    """
    required = [
        index for index, (risk, age) in enumerate(
            zip(snapshot.local_risk, snapshot.local_age)
        )
        if risk > config.local_soft_budget or age > config.max_local_age
    ]
    # 按风险降序排列返回
    return tuple(sorted(required,
                        key=lambda index: (-snapshot.local_risk[index], index)))


def make_action(action_id: str, selected: Iterable[int], refresh_coarse: bool,
                admissible: bool, reason: str, history: OnlineHistory,
                config: SelectorConfig) -> RepairAction:
    """包装生成单个具有成本预测与准入判断的 RepairAction 对象"""
    blocks = tuple(sorted(set(int(index) for index in selected)))
    repair_or_total, solve = history.estimate(action_id, config, len(blocks))
    observations = history.observations.get(action_id, [])
    if observations:
        predicted_total = repair_or_total
        predicted_repair = max(repair_or_total - solve, 0.0)
    else:
        predicted_repair = repair_or_total
        predicted_total = repair_or_total + solve
    return RepairAction(
        action_id=action_id,
        selected_local_blocks=blocks,
        refresh_coarse_matrix=bool(refresh_coarse),
        refresh_coarse_basis=False,
        safety_admissible=bool(admissible),
        rejection_reason=str(reason),
        predicted_total_seconds=float(predicted_total),
        predicted_repair_seconds=float(predicted_repair),
        predicted_solve_seconds=float(solve),
    )


def enumerate_actions(snapshot: RiskSnapshot, factor_count: int,
                      history: Optional[OnlineHistory] = None,
                      config: SelectorConfig = SelectorConfig()) -> List[RepairAction]:
    """
    【枚举并审查所有 6 类候选动作的安全准入资格】
    
    审查机制:
    - 检查是否超过局部硬预算 (local_hard_budget)；
    - 检查是否超过修补比例上限 (max_partial_fraction, 如 75%)；
    - 检查粗空间是否需要维护 (coarse_required)。
    """
    history = history or OnlineHistory()
    required = required_local_blocks(snapshot, config)
    local_hard = snapshot.max_local_risk > config.local_hard_budget
    coarse_required = (
        snapshot.coarse_matrix_risk > config.coarse_soft_budget
        or snapshot.coarse_age > config.max_coarse_age
    )
    coarse_hard = snapshot.coarse_matrix_risk > config.coarse_hard_budget
    partial_limit = max(1, int(config.max_partial_fraction * factor_count))
    partial_safe = len(required) <= partial_limit and not local_hard
    
    local_reason = ""
    if local_hard:
        local_reason = "local risk exceeds hard budget"
    elif len(required) > partial_limit:
        local_reason = "required local repair exceeds partial fraction"
    coarse_reason = "coarse risk exceeds hard budget" if coarse_hard else ""
    full_local_only_safe = not coarse_required
    
    # 分别评估 6 种动作的准入资格:
    return [
        # 1. 彻底复用: 只有在没有任何局部子域超标且粗空间安全时才被允许
        make_action(
            "reuse", (), False,
            not required and not coarse_required,
            "local or coarse safety budget requires repair" if (required or coarse_required) else "",
            history, config,
        ),
        # 2. 仅局部刷新: 局部安全可修且粗空间不需要刷新时才被允许
        make_action(
            "local_partial", required, False,
            partial_safe and not coarse_required,
            local_reason or ("coarse repair required" if coarse_required else ""),
            history, config,
        ),
        # 3. 仅刷新粗空间: 局部全部健康但全局粗空间老旧时被允许
        make_action(
            "coarse_refresh", (), True,
            not required and coarse_required and not coarse_hard,
            "local repair required" if required else coarse_reason,
            history, config,
        ),
        # 4. ★ 联合局部与粗空间维护 (JSR 主打): 局部有高危块且粗空间同步刷新时被允许
        make_action(
            "joint_partial", required, True,
            bool(required) and partial_safe and coarse_required and not coarse_hard,
            local_reason or coarse_reason or "coarse repair not required",
            history, config,
        ),
        # 5. 全局部重构但不刷新粗网格
        make_action(
            "full_local_only", tuple(range(factor_count)), False,
            full_local_only_safe,
            coarse_reason,
            history, config,
        ),
        # 6. 全量推倒重构: 永远合法，保底兜底
        make_action(
            "full_rebuild", tuple(range(factor_count)), True,
            True,
            "",
            history, config,
        ),
    ]


def select_action(snapshot: RiskSnapshot, factor_count: int,
                  history: Optional[OnlineHistory] = None,
                  config: SelectorConfig = SelectorConfig()
                  ) -> Tuple[RepairAction, List[RepairAction]]:
    """
    【最终决策函数】: 贪心选择预期总耗时最短的安全动作
    
    决策原则:
    1. 过滤出所有 safety_admissible == True 的合格动作；
    2. 在合格动作中，以 predicted_total_seconds (预估总耗时) 为第一排序键，
       以 ACTION_ORDER (动作复杂度升序) 为平局决胜键；
    3. 如果所有自适应动作均不安全，果断回退选择 full_rebuild。
    """
    candidates = enumerate_actions(snapshot, factor_count, history, config)
    admissible = [candidate for candidate in candidates if candidate.safety_admissible]
    selected = min(
        admissible,
        key=lambda action: (
            action.predicted_total_seconds,
            ACTION_ORDER.index(action.action_id),
        ),
    ) if admissible else next(
        candidate for candidate in candidates if candidate.action_id == "full_rebuild"
    )
    return selected, candidates


def widened_local_blocks(snapshot: RiskSnapshot, current: Iterable[int],
                         factor_count: int) -> Tuple[int, ...]:
    """故障升级辅助函数: 发生失败时，将重构子域的范围翻倍扩大 (2x 扩展保护带)"""
    current_set = set(int(index) for index in current)
    target = min(factor_count, max(2 * len(current_set), 1))
    return tuple(sorted(set(ranked_blocks(snapshot, target))))


def next_escalation(failed_action: RepairAction, snapshot: RiskSnapshot,
                    factor_count: int,
                    config: SelectorConfig = SelectorConfig()) -> Optional[RepairAction]:
    """
    【确定性熔断升级状态机 (Escalation State Machine)】
    
    当某次求解未通过外部真实残差证书时，系统调用该函数获取下一个升级动作:
    - reuse 失败 -> 升级为 local_partial (扩大局部重构范围);
    - local_partial 失败 -> 升级为 joint_partial (拉上粗网格一起刷新);
    - coarse_refresh 失败 -> 升级为 joint_partial;
    - joint_partial 失败 -> 最终升级为 full_rebuild (彻底推倒重来);
    - full_rebuild 失败 -> 返回 None (宣告该时间步不可恢复).
    """
    if failed_action.action_id == "full_rebuild":
        return None
    if failed_action.action_id in ("reuse", "full_local_only"):
        return make_action(
            "local_partial", widened_local_blocks(snapshot, (), factor_count), False,
            True, "deterministic escalation after failed solve", OnlineHistory(), config,
        ) if failed_action.action_id == "reuse" else make_action(
            "full_rebuild", range(factor_count), True, True,
            "deterministic escalation after failed solve", OnlineHistory(), config,
        )
    if failed_action.action_id == "local_partial":
        return make_action(
            "joint_partial", widened_local_blocks(snapshot, failed_action.selected_local_blocks, factor_count), True,
            True, "deterministic escalation after failed solve", OnlineHistory(), config,
        )
    if failed_action.action_id == "coarse_refresh":
        return make_action(
            "joint_partial", widened_local_blocks(snapshot, (), factor_count), True,
            True, "deterministic escalation after failed solve", OnlineHistory(), config,
        )
    return make_action(
        "full_rebuild", range(factor_count), True, True,
        "deterministic escalation after failed solve", OnlineHistory(), config,
    )

#!/usr/bin/env python3
"""
=============================================================================================
【JSR 科研消融实验驱动器：Phase 4 固定基底粗矩阵消融分析 (Fixed-Basis Coarse-Matrix Ablation)】
=============================================================================================

一、本驱动脚本的科学研究定位与核心物理问题
---------------------------------------------------------------------------------------------
在顶级期刊（如 SISC / JCP）的同行评议中，审稿人常提出一个关键尖锐质疑：
“JSR 算法在演化偏微分方程中表现出色，究竟是因为它修补了局部子域的 Cholesky 因子？
 还是因为更新了粗空间网格算子 A_0 = R_0 * A * R_0^T？
 如果粗基底 (Coarse Basis) 彻底固定不变，仅仅重新组装粗网格稀疏矩阵，能带来多大的收敛稳定性贡献？”

为了在完全不引入额外自由度混淆的情况下给出定量回答，Phase 4 设定了极度严谨的“正交消融协议”：
1. 绝对冻结粗网格几何投影基 R_0（基底版本号恒为 0，永不重算）。
2. 在该受限空间下，正交对比 4 个对照臂（Ablation Arms）：
   - Arm 1 (reuse_all_fixed_basis): 全复用基线（完全不修补局部，也不重算粗矩阵），即纯惰性策略。
   - Arm 2 (mass95_local_stale_coarse): 局部修复但粗矩阵陈旧（仅用 mass95 选出高漂移子域重新分解，
                                        但强行保留旧的粗网格矩阵 A_0）。
   - Arm 3 (mass95_local_refresh_coarse): JSR 联合维护（mass95 局部子域修复 + 重新组装粗矩阵 A_0）。
   - Arm 4 (full_local_refresh_coarse): 全量重构但基底固定（所有子域重新分解 + 粗矩阵重新组装）。

二、实验的可重复性与密码学防篡改保证 (Cryptographic Provenance)
---------------------------------------------------------------------------------------------
为确保实验数据的不可伪造性（Peer Review Auditability）：
1. 实验参数在执行前由独立 JSON 规格文件 (PHASE4_FREEZE_SPEC.json) 冻结，通过 SHA-256 签名锁定。
2. 物理执行前必须获得带签名的授权文件 (PHASE4_RUN_AUTHORIZATION.json)。
3. 数据输出采用 Append-only Ledger（仅追加账本）格式，单步运行日志记录严格包含时间戳、残差、
   内存与浮点耗时。
"""
from __future__ import annotations

from hashlib import sha256
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_phase2_2_comparative as base

SPEC = Path(os.environ.get("PHASE4_FREEZE_SPEC", str(ROOT / "PHASE4_FREEZE_SPEC.json")))
AUTHORIZATION = Path(os.environ.get(
    "PHASE4_AUTHORIZATION", str(ROOT / "PHASE4_RUN_AUTHORIZATION.json")))
RESULTS = Path(os.environ.get(
    "PHASE4_RESULTS", str(ROOT / "results" / "phase4_fixed_basis_v1")))
RAW = Path(os.environ.get(
    "PHASE4_RAW", str(RESULTS / "phase4_fixed_basis_raw_v1.jsonl")))
PROTOCOL = ROOT / "PHASE4_FIXED_BASIS_PROTOCOL.md"

# 实验元数据模式与实验对照臂常量定义
SCHEMA = "phase4-fixed-basis-ablation-v1"
PROTOCOL_ID = "phase4-fixed-basis-ablation-v1-2026-08-29"
RUN_ID = "phase4_fixed_basis_ablation_v1_seed20260829"

# 4 个消融实验臂
ARMS = (
    "reuse_all_fixed_basis",        # 臂1：全复用，基底固定
    "mass95_local_stale_coarse",    # 臂2：局部mass95修补，但粗算子不重算（故意保留陈旧矩阵）
    "mass95_local_refresh_coarse",  # 臂3：JSR核心——局部mass95修补 + 粗算子重新装配
    "full_local_refresh_coarse",    # 臂4：全量局部修补 + 粗算子重新装配（基底依然固定）
)
TRAJECTORIES = ("baseline_moving_local", "stress_moving_interface")
WARMUP = ((0, 1), (1, 2), (2, 3))               # 前3个步长作为系统热身（Warmup）
SCORED = ((3, 4), (4, 5), (5, 6), (6, 7))       # 真实计入得分与统计检验的步长（Scored）
ALL = WARMUP + SCORED
REPEATS = 3                                     # 独立重复测试次数（消除底层硬件抖动偏差）
ORDER_SEED = 20260829                           # 随机扰动种子，用于实验运行顺序的伪随机置换（防止缓存倾斜）


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True).encode("utf-8")).hexdigest()


def portable_path(path: Path) -> str:
    text = str(path).replace("\\", "/")
    if text.lower().startswith("/mnt/"):
        return text
    if len(text) >= 3 and text[1:3] == ":/":
        return "/mnt/%s/%s" % (text[0].lower(), text[3:])
    return text


def expected_ledger() -> dict:
    return {
        "trajectories": 2,
        "arms": 4,
        "repeats": 3,
        "transitions_per_rollout": 7,
        "expected_records": 168,
        "expected_warmup_records": 72,
        "expected_scored_records": 96,
    }


def load_spec() -> dict:
    if not SPEC.exists() or not PROTOCOL.exists():
        raise FileNotFoundError("Phase 4 spec or protocol is missing")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    content = dict(spec)
    recorded = content.pop("content_sha256", None)
    if (spec.get("schema") != "phase4-fixed-basis-freeze-v1"
            or spec.get("status") != "FROZEN_DESIGN"
            or recorded != json_sha(content)):
        raise ValueError("Phase 4 freeze is invalid")
    if file_sha(PROTOCOL) != str(spec.get("protocol", {}).get("sha256", "")):
        raise ValueError("Phase 4 protocol hash mismatch")
    if tuple(spec.get("trajectory_order", [])) != TRAJECTORIES:
        raise ValueError("Phase 4 trajectory order mismatch")
    if tuple(tuple(x) for x in spec.get("warmup_transitions", [])) != WARMUP:
        raise ValueError("Phase 4 warm-up split mismatch")
    if tuple(tuple(x) for x in spec.get("scored_transitions", [])) != SCORED:
        raise ValueError("Phase 4 scored split mismatch")
    if tuple(str(item.get("arm_id")) for item in spec.get("arms", [])) != ARMS:
        raise ValueError("Phase 4 arm order mismatch")
    if spec.get("ledger") != expected_ledger():
        raise ValueError("Phase 4 ledger dimensions mismatch")
    if spec.get("physical_run_authorized") is not False:
        raise ValueError("freeze record must not self-authorize physical execution")
    if not AUTHORIZATION.exists():
        raise PermissionError("Phase 4 authorization artifact is missing")
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if (authorization.get("status") != "AUTHORIZED_FOR_PHASE4_PHYSICAL_RUN"
            or authorization.get("freeze_spec_sha256") != file_sha(SPEC)
            or authorization.get("runner_sha256") != file_sha(Path(__file__))):
        raise PermissionError("Phase 4 authorization does not bind this runner/freeze")
    return spec


def phase4_order_rank(trajectory: str, previous: int, current: int,
                      repeat: int, arm: str) -> int:
    seed = ORDER_SEED
    for value in (trajectory, previous, current, repeat):
        if isinstance(value, str):
            seed = (seed * 131 + sum(ord(char) for char in value)) % (2 ** 32)
        else:
            seed = (seed * 131 + int(value)) % (2 ** 32)
    permutation = np.random.default_rng(seed).permutation(len(ARMS)).tolist()
    return int(permutation.index(ARMS.index(arm)))


def choose_phase4_action(policy: str, scores: np.ndarray,
                         target_info: Mapping[str, Any], snapshot: Any = None
                         ) -> Tuple[dict, float, float, list]:
    """
    【消融臂动作派发器：严密映射 4 类受限控制策略】
    -----------------------------------------------------------------------------------------
    本函数是 Phase 4 消融实验的核心动作路由器。它接收当前步的子域漂移能量打分，
    并严格根据指定的 policy（即 4 个消融臂之一）返回对应的预条件子维护指令：

    关键约束：
      - 所有返回的 action 字典中，refresh_coarse_basis 必须为 False（保证基底永不更新）。
      - 仅在局部子域分解范围 (selected factors) 和粗空间网格算子更新 (refresh_coarse) 上施加控制。

    参数解析:
      :param policy: 当前实验臂标识 ("reuse_all_fixed_basis", "mass95_local_stale_coarse", 等)
      :param scores: 各子域的局部刚度偏导漂移打分向量 (长度为因子总数)
      :param target_info: 目标校准信息，必须绑定已冻结的 mass95 策略 (捕获 >= 95% 扰动能量)
      :param snapshot: 当前风险快照（本消融实验不依赖动态成本优化器，故置空）
    :return: (action_dict, captured_mass, total_mass, ranking_list)
    """
    del snapshot
    # 步骤 1：利用 Phase 2.2 生产环境标准的帕累托能量截断算法计算局部修补子域集合
    selected, captured, total, ranking = base.select_mass_prefix(
        scores, base.MASS_TARGET)

    # 策略臂 1：纯惰性复用 (全量复用旧因子与旧粗矩阵，完全不花时间维护)
    if policy == "reuse_all_fixed_basis":
        return (base.action_dict("reuse", (), False, None,
                                 "fixed-basis reuse arm"),
                0.0, total, ranking)

    # 完整性校验：其余策略臂必须依赖冻结的 mass95 能量目标
    if target_info.get("kind") != "mass" or target_info.get("fallback") is not False:
        raise ValueError("Phase 4 requires the frozen mass95 calibration target")

    # 策略臂 2：局部修复，但强行保留陈旧粗网格矩阵 (Ablation: Stale Coarse Operator)
    # 意义：若求解器迭代次数暴增，直接证明“仅靠局部因子修补无法弥补全局低频长波误差的恶化”。
    if policy == "mass95_local_stale_coarse":
        return (base.action_dict("mass95_stale_coarse", selected, False,
                                 base.MASS_TARGET,
                                 "frozen mass95 local repair; retain coarse matrix"),
                captured, total, ranking)

    # 策略臂 3：JSR 核心联合维护 (局部 mass95 修复 + 粗网格矩阵同步更新 Galerkin 投影)
    # 意义：验证即使几何投影基底 R_0 不变，仅仅同步局部高频与粗网格低频算子，能否维持极低迭代次数。
    if policy == "mass95_local_refresh_coarse":
        return (base.action_dict("mass95", selected, True, base.MASS_TARGET,
                                 "frozen mass95 local and coarse-matrix repair"),
                captured, total, ranking)

    # 策略臂 4：全量局部因子重构 + 粗网格矩阵重新装配 (基底固定上限基准)
    # 意义：衡量 JSR 策略（臂3）在耗费仅极少数子域分解时间的前提下，能否逼近全量重构的数学收敛质量。
    if policy == "full_local_refresh_coarse":
        return (base.action_dict("full_rebuild", range(base.FACTOR_COUNT), True,
                                 None, "full local and coarse-matrix refresh; basis fixed"),
                1.0, total, ranking)

    raise ValueError("unknown Phase 4 arm: %s" % policy)


def rollout(trajectory: str, arm: str, repeat: int, manifest: Mapping[str, Any],
            catalog: Mapping[Tuple[str, int], Tuple[dict, np.ndarray, np.ndarray]],
            domains: Sequence[np.ndarray], owner: np.ndarray,
            target: Mapping[str, Any], manifest_path: Path, manifest_sha: str,
            backend_module: Any) -> list:
    """
    【完整时间演化轨迹步进执行器 (Rollout Execution)】
    -----------------------------------------------------------------------------------------
    本函数在一条给定的动力学物理轨迹上，对特定的消融策略臂运行全套离散时间步转移。

    执行核心原则：
      1. 状态生命周期单例：在轨迹开始时实例化 CorrectedStatefulTwoLevel 预条件器，
         在整条轨迹演化过程中原地更新，严禁无状态地随意销毁重置。
      2. 预热步隔离 (Warm-up Separation)：
         前 3 个转移步 ((0,1), (1,2), (2,3)) 统一采用全量重构以消除不同初始条件对预条件器缓存的干扰；
         后续步 ((3,4), (4,5), (5,6), (6,7)) 严格运行当前消融策略臂并计入学术指标。
      3. 侧车文件独立存储 (Sidecar Isolation)：
         为了严防不同重复测试或不同臂之间覆盖数值解与残差记录，所有侧车文件路径按
         `trajectory/arm/rXX/` 深度隔离，确保密码学审计轨迹可追溯。
      4. 严格断言不变量：
         验证整个执行过程没有任何一处尝试刷新 coarse_basis，且基底版本号恒为 0。
    """
    entries = manifest["trajectories"][trajectory]
    # 提取时间步 t=0 的初始刚度矩阵、载荷向量与自由度
    matrix0, rhs0, free0 = catalog[(trajectory, 0)]
    initial = dict(matrix0)
    initial["_free_dofs"] = free0

    # 步骤 1：构造底层的两层预条件器有状态生命周期对象
    lifecycle = backend_module.CorrectedStatefulTwoLevel(
        None, initial, rhs0, domains, owner, 0, base.KSP_RTOL, base.MAX_IT)
    initial_setup = float(lifecycle.initial_setup_seconds)
    state_matrices: Dict[int, dict] = {0: initial}
    records = []
    previous_results = base.RESULTS

    # 步骤 2：建立隔离的 Sidecar 磁盘存储目录
    sidecar_root = RESULTS / "sidecars" / trajectory / arm / ("r%02d" % repeat)
    sidecar_root.mkdir(parents=True, exist_ok=True)
    base.RESULTS = sidecar_root
    try:
        # 步骤 3：遍历预热与正式计分步
        for previous, current_state in ALL:
            matrix, rhs, free = catalog[(trajectory, current_state)]
            current = dict(matrix)
            current["_free_dofs"] = free
            split = "warmup" if (previous, current_state) in WARMUP else "held_out"
            # 预热步采用公共动作以消除启动偏置，正式计分步运行实际待评估臂
            effective_arm = "full_local_refresh_coarse" if split == "warmup" else arm
            rank = 0 if split == "warmup" else phase4_order_rank(
                trajectory, previous, current_state, repeat, arm)

            # 步骤 4：调用 Phase 2.2 的生产级事务转移函数执行物理求解
            record = base.execute_transition(
                trajectory, effective_arm, repeat, previous, current_state, split,
                state_matrices[previous], rhs, current, entries, state_matrices,
                lifecycle, domains, owner, target, manifest_path, manifest_sha,
                rank, None)

            # 步骤 5：打上严苛的 Phase 4 元数据防篡改标签
            record["type"] = "phase4_fixed_basis_transition"
            record["schema"] = SCHEMA
            record["protocol"] = PROTOCOL_ID
            record["run_id"] = RUN_ID
            record["arm_id"] = arm
            record["policy"] = arm
            record["controller"] = arm
            record["rollout_id"] = "%s/%s/r%02d" % (trajectory, arm, repeat)
            record["freeze_spec_sha256"] = file_sha(SPEC)
            record["fixed_basis"] = {
                "build_state": 0,
                "refresh_permitted": False,
                "basis_build_state_before": int(record["state_before"]["coarse_basis_build_state"]),
                "basis_build_state_after": int(record["state_after"]["coarse_basis_build_state"]),
            }
            record["target_binding"] = dict(target)
            record["warmup_common_action"] = bool(split == "warmup")
            record["scored"] = bool(split == "held_out")
            record["initial_setup_seconds"] = initial_setup
            record["production_cost"]["initial_setup_seconds"] = initial_setup
            record["source_binding"]["phase4_base_trajectory"] = trajectory

            # 规范化侧车文件路径，确保跨平台相对路径一致
            for sidecar_key in ("solution_sidecar", "residual_sidecar"):
                sidecar = record.get(sidecar_key)
                if isinstance(sidecar, dict) and sidecar.get("path"):
                    sidecar["path"] = (Path("sidecars") / trajectory / arm /
                                        ("r%02d" % repeat) /
                                        str(sidecar["path"])).as_posix()

            # 步骤 6：科学断言不变量审计（Invariant Assertion）
            # 若任何逻辑偷偷刷新了粗基底，立即抛出致命错误，坚决捍卫固定基底消融的科学纯洁性
            if any(bool(attempt["action"].get("refresh_coarse_basis"))
                   for attempt in record.get("attempts", [])):
                raise AssertionError("Phase 4 attempted a basis refresh")
            if int(record["state_after"]["coarse_basis_build_state"]) != 0:
                raise AssertionError("Phase 4 basis build state changed")

            records.append(record)
            if not bool(record["state_committed"]):
                raise RuntimeError("uncommitted Phase 4 transition %s/%s r%d %d->%d" %
                                   (trajectory, arm, repeat, previous, current_state))
            state_matrices[current_state] = current
    finally:
        # 步骤 7：释放 PETSc C 语言原生矩阵与求解器内存
        lifecycle.close()
        base.RESULTS = previous_results
    return records


def run_physical() -> dict:
    """
    【物理消融实验主执行入口：带密码学与防篡改验证的运行流程】
    -----------------------------------------------------------------------------------------
    本函数串联 Phase 4 消融实验的全套物理执行步骤，遵循顶级期刊严格的科学复现规范：
      1. 规格审查：检查 PHASE4_FREEZE_SPEC.json 与 PHASE4_RUN_AUTHORIZATION.json 签名匹配。
      2. 资源装载：读取动力学网格轨迹文件与初边值边界条件。
      3. 状态初始化：构建 PETSc 运行环境并输出包含软硬件哈希的 Header 记录。
      4. 笛卡尔积演化：遍历 (轨迹) × (消融臂) × (重复次数) 执行实际数值步进。
      5. 严格对账：校验生成的总记录数（期望 168 条，其中预热 72 条，计分 96 条）。
    """
    # 步骤 1：加载已冻结的实验设计规格
    spec = load_spec()
    if RAW.exists():
        raise FileExistsError("refusing to overwrite Phase 4 raw ledger: %s" % RAW)

    # 步骤 2：校验并加载离散网格轨迹清单
    manifest_path = base.resolve_path(spec["source"]["manifest_path"])
    if (not manifest_path.exists()
            or file_sha(manifest_path) != spec["source"]["manifest_sha256"]):
        raise ValueError("Phase 4 source manifest binding mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog, free = base.validate_manifest(manifest)
    domains, owner, partition = base.physical_partition(free)

    # 步骤 3：验证 mass95 能量标定参数
    target = base.load_frozen_target(base.CALIBRATION_SUMMARY)
    if target.get("kind") != "mass" or target.get("target") != base.MASS_TARGET:
        raise ValueError("Phase 4 mass95 calibration binding mismatch")

    # 动态导入 Phase 2 生产后端
    from corrected_phase2 import backend as backend_module
    RESULTS.mkdir(parents=True, exist_ok=True)
    environment = base.environment_record()

    # 步骤 4：构建物理不可篡改的账本头（Header Record）
    header = {
        "type": "header", "schema": SCHEMA, "protocol": PROTOCOL_ID,
        "run_id": RUN_ID, "freeze_spec_path": portable_path(SPEC),
        "freeze_spec_sha256": file_sha(SPEC),
        "authorization_path": portable_path(AUTHORIZATION),
        "authorization_sha256": file_sha(AUTHORIZATION),
        "runner_sha256": file_sha(Path(__file__)),
        "base_production_tree_sha256": base.production_tree_sha(),
        "trajectories": list(TRAJECTORIES), "arms": list(ARMS),
        "repeats": REPEATS, "warmup_transitions": [list(x) for x in WARMUP],
        "scored_transitions": [list(x) for x in SCORED],
        "expected_records": 168, "expected_warmup_records": 72,
        "expected_scored_records": 96,
        "warmup_action": "full_local_refresh_coarse",
        "fixed_basis": {"build_state": 0, "refresh_permitted": False,
                        "all_arms": True},
        "arm_specs": spec["arms"], "target_binding": target,
        "manifest_path": portable_path(manifest_path),
        "manifest_sha256": file_sha(manifest_path),
        "protocol_sha256": file_sha(PROTOCOL), "partition": partition,
        "environment": environment, "environment_sha256": json_sha(environment),
        "action_order_seed": ORDER_SEED, "future_information_used": False,
    }
    original_choose = base.choose_action
    count = warmup = scored = 0
    try:
        # 步骤 5：将基础运行器的动作决策函数替换为 Phase 4 的消融专用路由器
        base.choose_action = choose_phase4_action
        with RAW.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(header, sort_keys=True) + "\n")
            stream.flush()

            # 三重循环：轨迹 -> 消融臂 -> 重复试验
            for trajectory in TRAJECTORIES:
                for arm in ARMS:
                    for repeat in range(REPEATS):
                        records = rollout(trajectory, arm, repeat, manifest, catalog,
                                          domains, owner, target, manifest_path,
                                          header["manifest_sha256"], backend_module)
                        for record in records:
                            stream.write(json.dumps(record, sort_keys=True) + "\n")
                            stream.flush()
                            count += 1
                            warmup += int(not record["scored"])
                            scored += int(record["scored"])
                            print(json.dumps({"records": count, "expected": 168,
                                              "trajectory": trajectory, "arm": arm,
                                              "repeat": repeat,
                                              "transition": [record["previous_state"],
                                                             record["current_state"]]},
                                             sort_keys=True), flush=True)

            # 步骤 6：写入对账页脚（Footer Record）
            footer = {
                "type": "footer", "schema": SCHEMA, "protocol": PROTOCOL_ID,
                "run_id": RUN_ID, "records": count, "warmup_records": warmup,
                "scored_records": scored, "expected_records": 168,
                "expected_warmup_records": 72, "expected_scored_records": 96,
            }
            stream.write(json.dumps(footer, sort_keys=True) + "\n")
    finally:
        # 恢复初始环境的全局决策函数
        base.choose_action = original_choose

    # 步骤 7：最终一致性审计
    if count != 168 or warmup != 72 or scored != 96:
        raise RuntimeError("Phase 4 ledger count mismatch")
    return {"status": "PASS_PHASE4_FIXED_BASIS_RAW", "path": str(RAW),
            "records": count, "warmup_records": warmup, "scored_records": scored}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    【命令行入口解析器】
    -----------------------------------------------------------------------------------------
    本入口强制要求显式传入 `--authorize-physical` 开关，以确保任何人在运行实际耗时
    的物理消融实验前，都已经阅读并签署了对应的实验冻结授权协议，杜绝误操作。
    """
    parser = argparse.ArgumentParser(
        description="Phase 4 固定基底粗矩阵消融物理实验驱动脚本")
    parser.add_argument("--authorize-physical", action="store_true",
                        help="物理执行必须显式传递此参数，证明已完成 Phase 4 冻结协议审查")
    args = parser.parse_args(argv)
    if not args.authorize_physical:
        raise PermissionError("pass --authorize-physical only after independent review")
    print(json.dumps(run_physical(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


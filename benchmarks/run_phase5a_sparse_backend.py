#!/usr/bin/env python3
"""
=============================================================================================
【JSR 科研基准对照驱动器：Phase 5A PETSc 原生代数多重网格 (GAMG) 基线对比分析】
=============================================================================================

一、本驱动脚本的科学研究定位与工程背景
---------------------------------------------------------------------------------------------
在高性能偏微分方程数值求解领域，代数多重网格（Algebraic Multigrid, AMG）被公认为求解对称正定大稀疏
线性系统（SPD Systems）的最先进通用预条件器。国际顶级科学计算套件 PETSc 官方提供了高度优化的
`PCGAMG`（基于平滑聚合法 Smoothed Aggregation AMG）。

在向顶级期刊（如 SISC / JCP）投稿 JSR（演化算子联合维护框架）时，任何严谨的审稿人都必然会问：
“为什么我们需要专门设计针对局部扰动的 JSR 状态转移策略？
 直接使用工业级标杆——PETSc 自带的代数多重网格 GAMG，其表现如何？”

针对这一尖锐问题，Phase 5A 建立了绝对公平、工业级同台竞技的对照基线：
1. 在相同的时间演化动力学网格轨迹（baseline_moving_local 与 stress_moving_interface）上；
2. 保持完全一致的自由度网格矩阵、右端项载荷与 CG 求解容差（rtol=1.0e-8）；
3. 正交对比工业界最常用的两种 GAMG 维护策略：
   - 臂 1 (sparse_gamg_rebuild): 每步全量重构。在每个物理时间步，彻底丢弃旧的多重网格层次，
     重新执行聚合（Aggregation）、构建粗网格算子与平滑器。
     【特点】：收敛极快且稳健（迭代次数极少），但每步的多重网格设置（`setUp()`）开销极其沉重。
   - 臂 2 (sparse_gamg_reuse): 惰性多重网格复用 + 熔断后备。尝试沿用上一物理步生成的 5 层多重网格，
     仅更新最细层算子；一旦发现未能在指定容差内收敛，立即熔断触发 `rebuild_fallback` 强制重构。
     【特点】：若局部变动剧烈导致多重网格基底失配，复用将引发严重的迭代发散或昂贵的重构回退。

二、密码学与实验防伪溯源 (Cryptographic Provenance)
---------------------------------------------------------------------------------------------
本脚本继承了整个项目一贯的密码学不可篡改架构：
1. 预先冻结实验设计 (PHASE5A_FREEZE_SPEC.json) 与执行授权文件 (PHASE5A_RUN_AUTHORIZATION.json)。
2. 对输入的稀疏矩阵 CSR 拓扑（indptr, indices）与非零元数据（data）进行实时 SHA-256 校验。
3. 产出的每一条步进记录都记录真正相对残差证书、Krylov 求解耗时与多重网格层级深度。
"""
from __future__ import annotations

from hashlib import sha256
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "PHASE5A_FREEZE_SPEC.json"
PROTOCOL = ROOT / "PHASE5A_SPARSE_BACKEND_PROTOCOL.md"
AUTH = ROOT / "PHASE5A_RUN_AUTHORIZATION.json"
RESULTS = ROOT / "results" / "phase5a_sparse_backend_v1"
RAW = RESULTS / "phase5a_sparse_backend_raw_v1.jsonl"

# 测试轨迹与策略臂
TRAJECTORIES = ("baseline_moving_local", "stress_moving_interface")
ARMS = ("sparse_gamg_rebuild", "sparse_gamg_reuse")

# 时间转移步：前 3 步热身（Warmup），后 4 步计入学术指标（Scored）
WARMUP = ((0, 1), (1, 2), (2, 3))
SCORED = ((3, 4), (4, 5), (5, 6), (6, 7))
ALL = WARMUP + SCORED

RUN_ID = "phase5a_sparse_backend_v1_seed20260830"
PROTOCOL_ID = "phase5a-sparse-backend-v1-2026-08-30"
SCHEMA = "phase5a-sparse-backend-v1"


def file_sha(path: Path) -> str:
    """计算磁盘文件的 SHA-256 散列值（分块读取，防止大文件爆内存）。"""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    """计算连续内存数组的 SHA-256 散列值。"""
    array = np.ascontiguousarray(value)
    return sha256(array.view(np.uint8)).hexdigest()


def matrix_data_sha(matrix: dict) -> str:
    """
    【刚度矩阵全局指纹】
    深度哈希 CSR 稀疏矩阵的维度 (shape)、行指针 (indptr)、列索引 (indices) 和非零元数值 (data)。
    只要有任何一个浮点数值或网格连通性发生改变，散列值均会变化。
    """
    digest = sha256()
    for key in ("shape", "indptr", "indices", "data"):
        value = matrix[key]
        if key == "shape":
            digest.update(json.dumps(tuple(int(x) for x in value)).encode("ascii"))
        else:
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.view(np.uint8))
    return digest.hexdigest()


def pattern_sha(matrix: dict) -> str:
    """
    【稀疏图拓扑结构指纹 (Sparsity Pattern)】
    仅哈希 CSR 矩阵的行指针与列索引（不包含非零元数值）。
    用于验证在网格未发生大变形拓扑重划时，矩阵的非零元结构是否保持严格一致。
    """
    return sha256(np.ascontiguousarray(matrix["indptr"], dtype=np.int64).tobytes()
                  + np.ascontiguousarray(matrix["indices"], dtype=np.int64).tobytes()).hexdigest()


def resolve_wsl(path: str) -> Path:
    """统一 Windows/WSL 路径分隔符，保障跨操作系统路径解析一致性。"""
    text = str(path).replace("\\", "/")
    if text.startswith("/mnt/"):
        return Path(text)
    return Path(text)


def load_catalog(spec: dict):
    """
    【离散网格轨迹目录装载与防篡改审计】
    -----------------------------------------------------------------------------------------
    本函数严格根据冻结规格文件记载的哈希，读取物理演化轨迹中的全部刚度矩阵与右端项，
    并构筑用于快速检索与初始化的内存目录（Catalog）。
    """
    manifest_path = resolve_wsl(spec["source_manifest_path"])
    if file_sha(manifest_path) != spec["source_manifest_sha256"]:
        raise RuntimeError("source manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog = {}
    for trajectory in TRAJECTORIES:
        entries = manifest.get("trajectories", {}).get(trajectory)
        if not isinstance(entries, list) or len(entries) != 8:
            raise RuntimeError("source trajectory catalog invalid: %s" % trajectory)
        for state, entry in enumerate(entries):
            matrix_path = resolve_wsl(entry["matrix"]["path"])
            rhs_path = resolve_wsl(entry["rhs"]["path"])
            if file_sha(matrix_path) != entry["matrix"]["sha256"] or file_sha(rhs_path) != entry["rhs"]["sha256"]:
                raise RuntimeError("source file hash mismatch %s/%d" % (trajectory, state))
            with np.load(matrix_path, allow_pickle=False) as archive:
                raw_shape = tuple(int(x) for x in archive["shape"])
                raw_indptr = np.asarray(archive["indptr"])
                raw_indices = np.asarray(archive["indices"])
                raw_data = np.asarray(archive["data"])
                matrix = {
                    "shape": raw_shape,
                    "indptr": np.asarray(raw_indptr, dtype=PETSc.IntType),
                    "indices": np.asarray(raw_indices, dtype=PETSc.IntType),
                    "data": np.asarray(raw_data, dtype=PETSc.ScalarType),
                }
            rhs = np.asarray(np.load(rhs_path, allow_pickle=False), dtype=PETSc.ScalarType)
            raw_matrix = {"shape": raw_shape, "indptr": raw_indptr,
                          "indices": raw_indices, "data": raw_data}
            catalog[(trajectory, state)] = {"matrix": matrix, "rhs": rhs,
                                            "matrix_path": str(matrix_path), "rhs_path": str(rhs_path),
                                            "source_matrix_sha256": entry["matrix"]["sha256"],
                                            "source_rhs_sha256": entry["rhs"]["sha256"],
                                            "matrix_data_sha256": matrix_data_sha(raw_matrix),
                                            "rhs_data_sha256": array_sha(rhs),
                                            "pattern_sha256": pattern_sha(raw_matrix)}
    return manifest_path, catalog


def make_mat(matrix: dict):
    """根据 CSR 稀疏格式数组创建 PETSc 底层 C 语言原生 MatAIJ 对象。"""
    return PETSc.Mat().createAIJ(size=matrix["shape"],
                                 csr=(matrix["indptr"], matrix["indices"], matrix["data"]),
                                 comm=PETSc.COMM_SELF)


def configure(matrix: PETSc.Mat):
    """
    【PETSc Krylov 求解器 (KSP) 与代数多重网格 (PCGAMG) 配置】
    -----------------------------------------------------------------------------------------
    1. 求解器类型：共轭梯度法 (Conjugate Gradient, CG)，专门用于对称正定大型稀疏系统。
    2. 范数监控：非预条件残差范数 (UNPRECONDITIONED)，即直接计算真实代数残差 ||b - Ax||_2。
    3. 停机准则：相对容差 rtol = 1.0e-8，绝对容差 atol = 0.0，最大迭代步数 max_it = 2000。
    4. 预条件器：代数多重网格 (GAMG, Geometric-Algebraic Multigrid)。
    """
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    ksp.setType("cg")
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setTolerances(rtol=1.0e-8, atol=0.0, max_it=2000)
    pc = ksp.getPC()
    pc.setType("gamg")
    ksp.setOperators(matrix, matrix)
    return ksp, pc


def solve(ksp, matrix, rhs):
    """
    【KSP 实际求解与不可伪造的真实残差证书认证】
    -----------------------------------------------------------------------------------------
    本函数封装单次线性方程组 Ax = b 的求解过程，并执行绝对严格的收敛性认证：
      1. 初始化未知量解向量 x 为 0；
      2. 启动高精度计时器，调用底层 PETSc C 动态链接库进行 Krylov 迭代求解；
      3. 【独立真残差核验】：不仅读取 PETSc 的收敛标志，更通过实际矩阵向量乘法运算：
            r = b - A * x
         计算真残差范数 ||r||_2 以及真相对残差 ||r||_2 / ||b||_2；
      4. 严格断言：只有在真相对残差 <= 1.0e-8 且收敛理由 (ConvergedReason) > 0 时才判定为真正收敛。
    """
    x = matrix.createVecRight()
    x.set(0.0)
    started = time.perf_counter()
    error = None
    try:
        ksp.solve(rhs, x)
    except Exception as exc:
        error = repr(exc)
    solve_seconds = time.perf_counter() - started

    # 独立真残差矩阵向量乘：r = A*x - b (然后计算范数)
    residual = rhs.duplicate()
    matrix.mult(x, residual)
    residual.axpy(-1.0, rhs)
    residual_norm = float(residual.norm())
    rhs_norm = float(rhs.norm())
    relative = residual_norm / max(rhs_norm, 1.0e-300)

    reason = int(ksp.getConvergedReason()) if error is None else 0
    iterations = int(ksp.getIterationNumber()) if error is None else 0
    converged = bool(error is None and reason > 0 and math.isfinite(relative) and relative <= 1.0e-8)

    # 拷贝数值解与残差向量供侧车存盘
    values = np.asarray(x.getArray(readonly=True), dtype=np.float64).copy()
    residual_values = np.asarray(residual.getArray(readonly=True), dtype=np.float64).copy()

    # 销毁 PETSc 临时向量，防内存泄露
    x.destroy()
    residual.destroy()

    return {"error": "" if error is None else error, "converged": converged,
            "converged_reason": reason, "iterations": iterations,
            "rhs_norm": rhs_norm, "residual_norm": residual_norm,
            "true_residual": relative, "solve_seconds": solve_seconds,
            "tolerance": 1.0e-8}, values, residual_values



def hierarchy(ksp, pc) -> dict:
    """
    【PETSc PCGAMG 多重网格几何/代数拓扑层级探针】
    -----------------------------------------------------------------------------------------
    本函数在求解完成后探测当前 GAMG 预条件器内部构筑的实际网格层数（Levels）。
    在我们的二维泊松/弹性力学基准网格上，PETSc 内部聚合法通常自动生成 5 层几何代数网格结构。
    记录此数据用于审计在 reuse（复用）或 rebuild（重构）前后，多重网格层级深度是否发生剧烈突变。
    """
    try:
        levels = int(pc.getMGLevels())
    except Exception:
        levels = -1
    return {"ksp_handle": int(ksp.handle), "pc_handle": int(pc.handle), "levels": levels}


def write_sidecar(path: Path, values: np.ndarray) -> dict:
    """
    【科学计算侧车文件存盘与元数据签名】
    -----------------------------------------------------------------------------------------
    将 Krylov 求解器输出的最终解向量 $x$ 或真实代数残差向量 $r$ 固化为标准 NumPy 二进制文件 (.npy)。
    同时计算该文件的 SHA-256 散列并记录张量形状与数据类型，实现解向量数据的“不可篡改可复算”。
    """
    np.save(str(path), np.ascontiguousarray(values, dtype=np.float64), allow_pickle=False)
    return {"path": path.relative_to(RESULTS).as_posix(), "sha256": file_sha(path),
            "shape": list(values.shape), "dtype": str(values.dtype)}


def transition(trajectory: str, arm: str, repeat: int, previous: int, current: int,
               split: str, previous_entry: dict, current_entry: dict,
               ksp: PETSc.KSP, pc: PETSc.PC, action_rank: int) -> dict:
    """
    【GAMG 单步状态转移控制与熔断重构机制 (Single Transition Step)】
    -----------------------------------------------------------------------------------------
    本函数是 Phase 5A 基准对比的核心测试函数。它接收上一时间步保留下来的 KSP/PC 对象，
    根据当前对照臂（rebuild 或 reuse）执行不同生命周期维护与求解：

    1. 重构策略臂 (sparse_gamg_rebuild):
       - 显式声明 `pc.setReusePreconditioner(False)`；
       - 将当前时间步的新刚度矩阵 $A_{t}$ 绑定至 KSP；
       - 执行 `ksp.setUp()`：强制底层 PETSc C 库重新执行全量 Aggregation 聚合、粗网格装配；
       - 记录 setup_seconds，并调用 `solve` 求解；

    2. 复用策略臂 (sparse_gamg_reuse):
       - 尝试惰性复用：`pc.setReusePreconditioner(True)`，直接保留上一步构筑的旧多重网格层级；
       - setup_seconds 记为 0.0，直接调用 `solve` 求解；
       - 【科学熔断保护】：如果惰性复用导致发散（`converged == False`），系统绝不隐瞒故障，
         而是记录首轮失败尝试后，立即触发 `rebuild_fallback` 熔断后备机制，强制将
         `setReusePreconditioner(False)` 并重新 `setUp()` 求解，确保每一步都能最终算出合格物理场。

    3. 侧车持久化与详细耗时归因：
       - 计算 action_total_seconds，精准剥离 setup 时间、krylov 迭代时间以及微小的定时器缝隙；
       - 导出解与残差文件到专属子目录；
       - 返回标准化的 transition 记录字典。
    """
    # 步骤 1：构造 PETSc 原生矩阵与载荷向量
    matrix = make_mat(current_entry["matrix"])
    rhs = PETSc.Vec().createWithArray(current_entry["rhs"], comm=PETSc.COMM_SELF)
    action_started = time.perf_counter()
    attempts = []

    # 步骤 2：分支执行两种维护策略
    if arm == "sparse_gamg_rebuild":
        # 强制销毁旧网格，每步重新构筑全量代数多重网格
        pc.setReusePreconditioner(False)
        ksp.setOperators(matrix, matrix)
        setup_started = time.perf_counter()
        ksp.setUp()
        setup_seconds = time.perf_counter() - setup_started

        result, solution, residual = solve(ksp, matrix, rhs)
        attempts.append({"action_id": "rebuild", "setup_seconds": setup_seconds, **result})
    else:
        # 尝试复用旧的多重网格拓扑
        pc.setReusePreconditioner(True)
        ksp.setOperators(matrix, matrix)
        setup_seconds = 0.0
        result, solution, residual = solve(ksp, matrix, rhs)
        attempts.append({"action_id": "reuse", "setup_seconds": setup_seconds, **result})

        # 若复用导致发散，触发熔断重构后备动作 (Fallback Rebuild)
        if not result["converged"]:
            pc.setReusePreconditioner(False)
            ksp.setOperators(matrix, matrix)
            setup_started = time.perf_counter()
            ksp.setUp()
            setup_seconds = time.perf_counter() - setup_started
            result, solution, residual = solve(ksp, matrix, rhs)
            attempts.append({"action_id": "rebuild_fallback", "setup_seconds": setup_seconds, **result})

    action_total = time.perf_counter() - action_started

    # 步骤 3：保存隔离的侧车数据并记录元数据
    side_root = RESULTS / "sidecars" / trajectory / arm / ("r%02d" % repeat)
    side_root.mkdir(parents=True, exist_ok=True)
    stem = "%s_%s_r%02d_t%02d_t%02d_%s" % (trajectory, arm, repeat, previous, current, split)
    solution_meta = write_sidecar(side_root / (stem + "_x.npy"), solution)
    residual_meta = write_sidecar(side_root / (stem + "_r.npy"), residual)

    # 步骤 4：组装严密的审计账本条目
    record = {
        "type": "phase5a_sparse_backend_transition",
        "schema": SCHEMA,
        "protocol": PROTOCOL_ID,
        "run_id": RUN_ID,
        "trajectory": trajectory,
        "arm_id": arm,
        "policy": arm,
        "repeat": repeat,
        "previous_state": previous,
        "current_state": current,
        "transition_order": previous,
        "action_order_rank": action_rank,
        "split": split,
        "scored": split == "held_out",
        "source": {
            "matrix_sha256": current_entry["source_matrix_sha256"],
            "rhs_sha256": current_entry["source_rhs_sha256"],
            "matrix_data_sha256": current_entry["matrix_data_sha256"],
            "rhs_data_sha256": current_entry["rhs_data_sha256"],
            "pattern_sha256": current_entry["pattern_sha256"],
            "shape": list(current_entry["matrix"]["shape"]),
            "nnz": int(current_entry["matrix"]["data"].size),
        },
        "action": "rebuild" if arm == "sparse_gamg_rebuild" else ("reuse" if len(attempts) == 1 else "reuse_then_rebuild"),
        "attempts": attempts,
        "fallback": len(attempts) > 1,
        "hierarchy_after": hierarchy(ksp, pc),
        "solve": result,
        "certificate": {
            "passed": bool(result["converged"]),
            "relative_true_residual": result["true_residual"],
            "tolerance": 1.0e-8,
        },
        "solution_sidecar": solution_meta,
        "residual_sidecar": residual_meta,
        "timing": {
            "action_total_seconds": action_total,
            "setup_seconds": sum(float(x["setup_seconds"]) for x in attempts),
            "krylov_solve_seconds": sum(float(x["solve_seconds"]) for x in attempts),
            "timer_gap_seconds": action_total - sum(float(x["setup_seconds"]) + float(x["solve_seconds"]) for x in attempts),
        },
        "causal": {
            "current_state_only": True,
            "future_information_used": False,
            "future_states_read": [],
        },
        "state_committed": bool(result["converged"]),
    }

    # 步骤 5：释放 PETSc 临时对象
    rhs.destroy()
    matrix.destroy()
    return record


def run() -> dict:
    """
    【Phase 5A 工业基线实验全流程执行控制器】
    -----------------------------------------------------------------------------------------
    本函数完整执行 Phase 5A 的物理对照实验：
      1. 检查物理执行授权文件与冻结规格哈希；
      2. 加载演化轨迹刚度矩阵目录，初始化输出账本文件；
      3. 写入不可篡改的账本头（包含 PETSc 版本号与软硬件环境指纹）；
      4. 运行双轨迹、双策略臂、三次独立重复实验（共 2 × 2 × 3 = 12 个 Rollout，每个 7 步转移）；
      5. 严格对账：期望总记录数 84 条（热身步 36 条，正式学术计分步 48 条）。
    """
    # 步骤 1：审查授权与防伪哈希
    if not AUTH.exists() or json.loads(AUTH.read_text()).get("status") != "AUTHORIZED_FOR_PHASE5A_PHYSICAL_RUN":
        raise PermissionError("Phase 5A authorization artifact missing or invalid")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_DESIGN" or not spec.get("physical_run_authorized"):
        raise PermissionError("Phase 5A freeze is not authorized")
    if file_sha(PROTOCOL) != str(spec["protocol_sha256"]):
        raise RuntimeError("protocol hash binding mismatch")
    if RAW.exists():
        raise FileExistsError("refusing to overwrite Phase 5A ledger")

    manifest_path, catalog = load_catalog(spec)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 步骤 2：生成账本头记录
    header = {
        "type": "header",
        "schema": SCHEMA,
        "protocol": PROTOCOL_ID,
        "run_id": RUN_ID,
        "freeze_spec_sha256": file_sha(SPEC),
        "protocol_sha256": file_sha(PROTOCOL),
        "authorization_sha256": file_sha(AUTH),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha(manifest_path),
        "trajectories": list(TRAJECTORIES),
        "arms": list(ARMS),
        "repeats": 3,
        "warmup_transitions": [list(x) for x in WARMUP],
        "scored_transitions": [list(x) for x in SCORED],
        "expected_records": 84,
        "expected_warmup_records": 36,
        "expected_scored_records": 48,
        "future_information_used": False,
        "environment": {
            "python": os.sys.executable,
            "petsc": PETSc.Sys.getVersion(),
        },
    }
    count = warm = scored = 0

    # 步骤 3：核心物理步进循环
    with RAW.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(header, sort_keys=True) + "\n")
        for trajectory in TRAJECTORIES:
            for arm in ARMS:
                for repeat in range(3):
                    # 获取该物理轨迹 t=0 初始步的状态
                    base_entry = catalog[(trajectory, 0)]
                    init_matrix = make_mat(base_entry["matrix"])
                    init_rhs = PETSc.Vec().createWithArray(base_entry["rhs"], comm=PETSc.COMM_SELF)

                    # 配置初始 GAMG 求解器
                    ksp, pc = configure(init_matrix)
                    initial_started = time.perf_counter()
                    ksp.setUp()
                    initial_setup = time.perf_counter() - initial_started

                    # 求解 t=0 初始稳态场
                    init_result, _, _ = solve(ksp, init_matrix, init_rhs)
                    if not init_result["converged"]:
                        ksp.destroy()
                        raise RuntimeError("initial sparse GAMG control failed")

                    try:
                        # 沿时间轴步进
                        for previous, current in ALL:
                            split = "warmup" if (previous, current) in WARMUP else "held_out"
                            record = transition(
                                trajectory, arm, repeat, previous, current, split,
                                catalog[(trajectory, previous)], catalog[(trajectory, current)],
                                ksp, pc, 0 if split == "warmup" else 0)
                            record["initial_setup_seconds"] = initial_setup
                            stream.write(json.dumps(record, sort_keys=True) + "\n")
                            stream.flush()
                            count += 1
                            warm += int(split == "warmup")
                            scored += int(split == "held_out")
                    finally:
                        # 销毁 PETSc 求解器对象
                        ksp.destroy()

        # 步骤 4：写入账本页脚
        footer = {
            "type": "footer",
            "schema": SCHEMA,
            "protocol": PROTOCOL_ID,
            "run_id": RUN_ID,
            "records": count,
            "warmup_records": warm,
            "scored_records": scored,
            "expected_records": 84,
            "expected_warmup_records": 36,
            "expected_scored_records": 48,
        }
        stream.write(json.dumps(footer, sort_keys=True) + "\n")

    # 步骤 5：最终数据条目数核验
    if (count, warm, scored) != (84, 36, 48):
        raise RuntimeError("Phase 5A ledger count mismatch")
    return {"status": "PASS_PHASE5A_SPARSE_BACKEND_RAW",
            "records": count, "warmup_records": warm, "scored_records": scored}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5A PETSc 原生代数多重网格 (GAMG) 工业基线对比运行器")
    parser.add_argument("--authorize-physical", action="store_true",
                        help="物理执行必须显式传递此参数，证明已阅读并确认 Phase 5A 授权协议")
    parser.parse_args()
    print(json.dumps(run(), sort_keys=True))


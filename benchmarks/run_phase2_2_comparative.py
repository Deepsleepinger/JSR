#!/usr/bin/env python3
"""Run the frozen Phase-2.2 repeated comparative held-out evaluation.

The calibration ledger and the corrected Phase-2 smoke ledger are immutable
inputs.  This runner writes a new ledger containing an explicit state-0 to
state-3 warm-up followed by the four held-out tail transitions.  The warm-up
records make the cache entering state 3 auditable while only tail records are
used for the primary repeated evaluation.

The numerical work is intentionally delegated to ``corrected_phase2.backend``
but action selection is implemented here so that the frozen mass target is
visible and causally checkable.  Physical execution requires the frozen WSL
environment and ``ALLOW_STATEFUL_MAINTENANCE_V1=1``.
"""
from __future__ import annotations

from hashlib import sha256
import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = (
    "/mnt/h/Codex/2026-07-27/new-chat/experiments/repair_gnn_prototype/"
    "results/refresh_event_trigger/"
    "spd_nonlinear_diffusion_amg_stress_v5_ic_res1e8_20260819/"
    "source_manifest_v5.json"
)
MANIFEST = Path(os.environ.get("MANIFEST_PATH", DEFAULT_MANIFEST))
CALIBRATION_SUMMARY = Path(os.environ.get(
    "PHASE2_1_SUMMARY_JSON",
    str(ROOT / "results" / "phase2_1_calibration_v1" /
        "phase2_1_calibration_summary.json"),
))
RESULTS = Path(os.environ.get(
    "PHASE2_2_RESULTS",
    str(ROOT / "results" / "phase2_2_comparative_v1"),
))
RAW = Path(os.environ.get(
    "PHASE2_2_RAW",
    str(RESULTS / "phase2_2_comparative_sequential_raw_v1.jsonl"),
))
PROTOCOL = ROOT / "PHASE2_2_COMPARATIVE_PROTOCOL.md"

TRAJECTORIES = ("baseline_moving_local", "stress_moving_interface")
POLICIES = (
    "always_reuse", "fixed_partial_50", "raw_drift_ranking",
    "local_only_adaptive", "mass95_frozen", "full_rebuild",
)
WARMUP_TRANSITIONS = ((0, 1), (1, 2), (2, 3))
TAIL_TRANSITIONS = ((3, 4), (4, 5), (5, 6), (6, 7))
ALL_TRANSITIONS = WARMUP_TRANSITIONS + TAIL_TRANSITIONS

GRID = 383
MESH_CELLS = 384
BLOCK_SIZE = 32
SIDE_COUNT = 12
FACTOR_COUNT = 144
CERT_TOL = 1.0e-8
KSP_RTOL = 1.0e-10
MAX_IT = 2000
MASS_TARGET = 0.95
ORDER_SEED = 20260829
RUN_ID = "phase2_2_comparative_sequential_v1_seed20260829"
SCHEMA = "phase2.2-comparative-sequential-v1"
PROTOCOL_ID = "phase2.2-comparative-sequential-v1-2026-08-29"
PARTIAL_COUNT = 72


def resolve_path(value: Any, base: Optional[Path] = None) -> Path:
    """Resolve native, WSL and relative paths without rewriting metadata."""
    text = str(value)
    candidates: List[Path] = [Path(text)]
    match = re.match(r"^/mnt/([A-Za-z])/(.*)$", text)
    if match:
        candidates.append(Path(match.group(1).upper() + ":/" + match.group(2)))
    native = Path(text)
    if base is not None and not native.is_absolute():
        candidates.append(base / native)
    candidates.append(ROOT / native)
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    if match:
        return Path(match.group(1).upper() + ":/" + match.group(2))
    return native


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value, dtype=np.float64)
    return sha256(value.view(np.uint8)).hexdigest()


def raw_array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return sha256(value.view(np.uint8)).hexdigest()


def json_sha(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def matrix_data_sha(matrix: Mapping[str, Any]) -> str:
    """Hash the complete CSR payload in the backend's canonical order."""
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


def csr_pattern_sha(matrix: Mapping[str, Any]) -> str:
    payload = (np.ascontiguousarray(matrix["indptr"], dtype=np.int64).tobytes()
               + np.ascontiguousarray(matrix["indices"], dtype=np.int64).tobytes())
    return sha256(payload).hexdigest()


def production_tree_sha() -> str:
    """Bind only files that can affect production numerical behavior."""
    paths = [
        ROOT / "run_phase2_2_comparative.py",
        ROOT / "corrected_phase2" / "backend.py",
        ROOT / "corrected_phase2" / "monitor.py",
        ROOT / "corrected_phase2" / "selector.py",
    ]
    digest = sha256()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(str(path))
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_record() -> dict:
    record: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "backend": "corrected_phase2.backend",
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "MKL_DYNAMIC")
        },
    }
    try:
        import dolfin as df  # type: ignore
        record["dolfin"] = str(getattr(df, "__version__", "unknown"))
        record["dolfin_executable"] = str(getattr(df, "__file__", ""))
    except Exception as exc:  # pragma: no cover - physical environment only
        record["dolfin"] = "unavailable:%s" % type(exc).__name__
        record["dolfin_executable"] = ""
    try:
        import petsc4py  # type: ignore
        record["petsc4py"] = str(getattr(petsc4py, "__version__", "unknown"))
        from petsc4py import PETSc  # type: ignore
        record["petsc"] = ".".join(str(x) for x in PETSc.Sys.getVersion())
    except Exception as exc:  # pragma: no cover - physical environment only
        record["petsc4py"] = "unavailable:%s" % type(exc).__name__
        record["petsc"] = "unavailable:%s" % type(exc).__name__
    return record


def load_state(entry: Mapping[str, Any]) -> Tuple[dict, np.ndarray, np.ndarray]:
    matrix_meta = entry["matrix"]
    matrix_path = resolve_path(matrix_meta["path"])
    rhs_path = resolve_path(entry["rhs"]["path"])
    with np.load(matrix_path, allow_pickle=False) as archive:
        required = ("shape", "data", "indices", "indptr", "free_dofs")
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError("matrix archive lacks %s" % ",".join(missing))
        matrix = {
            "shape": tuple(int(x) for x in archive["shape"]),
            "data": np.asarray(archive["data"], dtype=np.float64),
            "indices": np.asarray(archive["indices"], dtype=np.int64),
            "indptr": np.asarray(archive["indptr"], dtype=np.int64),
        }
        free = np.asarray(archive["free_dofs"], dtype=np.int64)
    rhs = np.asarray(np.load(rhs_path, allow_pickle=False), dtype=np.float64)
    return matrix, rhs, free


def validate_manifest(manifest: Mapping[str, Any]) -> Tuple[Dict[Tuple[str, int], Tuple[dict, np.ndarray, np.ndarray]], np.ndarray]:
    trajectories = manifest.get("trajectories")
    if not isinstance(trajectories, Mapping):
        raise ValueError("manifest has no trajectories")
    cache: Dict[Tuple[str, int], Tuple[dict, np.ndarray, np.ndarray]] = {}
    reference_pattern: Optional[Tuple[np.ndarray, np.ndarray]] = None
    reference_free: Optional[np.ndarray] = None
    for trajectory in TRAJECTORIES:
        entries = trajectories.get(trajectory)
        if not isinstance(entries, list) or len(entries) < 8:
            raise ValueError("manifest lacks eight states for %s" % trajectory)
        for state in range(8):
            entry = entries[state]
            matrix_meta = entry["matrix"]
            rhs_meta = entry["rhs"]
            matrix_path = resolve_path(matrix_meta["path"])
            rhs_path = resolve_path(rhs_meta["path"])
            if not matrix_path.exists() or not rhs_path.exists():
                raise FileNotFoundError("missing source state %s/%d" % (trajectory, state))
            if file_sha(matrix_path) != str(matrix_meta["sha256"]):
                raise ValueError("source matrix hash mismatch %s/%d" % (trajectory, state))
            if file_sha(rhs_path) != str(rhs_meta["sha256"]):
                raise ValueError("source RHS hash mismatch %s/%d" % (trajectory, state))
            matrix, rhs, free = load_state(entry)
            n = int(matrix["shape"][0])
            if tuple(matrix["shape"]) != (GRID * GRID, GRID * GRID):
                raise ValueError("unexpected matrix shape %s/%d" % (trajectory, state))
            if matrix["indptr"].shape != (n + 1,):
                raise ValueError("invalid CSR row pointer shape %s/%d" % (trajectory, state))
            if matrix["indices"].shape != matrix["data"].shape:
                raise ValueError("CSR index/data shape mismatch %s/%d" % (trajectory, state))
            if int(matrix["indptr"][0]) != 0 or int(matrix["indptr"][-1]) != int(matrix["data"].size):
                raise ValueError("CSR pointer endpoints mismatch %s/%d" % (trajectory, state))
            if np.any(np.diff(matrix["indptr"]) < 0):
                raise ValueError("CSR pointers are not monotone %s/%d" % (trajectory, state))
            if np.any(matrix["indices"] < 0) or np.any(matrix["indices"] >= n):
                raise ValueError("CSR column index out of range %s/%d" % (trajectory, state))
            if rhs.shape != (n,) or free.shape != (n,):
                raise ValueError("source dimensions mismatch %s/%d" % (trajectory, state))
            if (not np.all(np.isfinite(matrix["data"]))
                    or not np.all(np.isfinite(rhs))):
                raise ValueError("non-finite source values %s/%d" % (trajectory, state))
            if np.any(free < 0) or np.any(free >= (MESH_CELLS + 1) ** 2):
                raise ValueError("free_dofs out of range %s/%d" % (trajectory, state))
            if np.unique(free).size != free.size:
                raise ValueError("duplicate free_dofs %s/%d" % (trajectory, state))
            if "data_sha256" in matrix_meta and raw_array_sha(matrix["data"]) != str(matrix_meta["data_sha256"]):
                raise ValueError("source matrix data hash mismatch %s/%d" % (trajectory, state))
            if "pattern_sha256" in matrix_meta and csr_pattern_sha(matrix) != str(matrix_meta["pattern_sha256"]):
                raise ValueError("source pattern hash mismatch %s/%d" % (trajectory, state))
            if "data_sha256" in rhs_meta and raw_array_sha(rhs) != str(rhs_meta["data_sha256"]):
                raise ValueError("source RHS data hash mismatch %s/%d" % (trajectory, state))
            if reference_pattern is None:
                reference_pattern = (matrix["indptr"].copy(), matrix["indices"].copy())
                reference_free = free.copy()
            elif (not np.array_equal(matrix["indptr"], reference_pattern[0])
                  or not np.array_equal(matrix["indices"], reference_pattern[1])
                  or not np.array_equal(free, reference_free)):
                raise ValueError("source pattern/free map changed at %s/%d" % (trajectory, state))
            # Keep the source map attached to the immutable matrix object so
            # record-level partition provenance can be reconstructed without
            # falling back to a placeholder hash.
            matrix["_free_dofs"] = free.copy()
            cache[(trajectory, state)] = (matrix, rhs, free)
    if reference_free is None:
        raise ValueError("manifest has no states")
    return cache, reference_free


def physical_partition(free: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, dict]:
    try:
        import dolfin as df  # type: ignore
    except Exception as exc:  # pragma: no cover - physical environment only
        raise RuntimeError("DOLFIN is required for the physical partition") from exc
    if free.shape != (GRID * GRID,) or np.unique(free).size != free.size:
        raise ValueError("invalid free_dofs")
    mesh = df.UnitSquareMesh(MESH_CELLS, MESH_CELLS)
    space = df.FunctionSpace(mesh, "CG", 1)
    coords = space.tabulate_dof_coordinates().reshape((-1, 2))[free]
    scaled = coords * MESH_CELLS
    rounded = np.rint(scaled)
    if np.any(np.abs(scaled - rounded) > 1.0e-10):
        raise ValueError("source coordinates are not integer grid points")
    x = rounded[:, 0].astype(np.int64)
    y = rounded[:, 1].astype(np.int64)
    if np.any(x < 1) or np.any(x > GRID) or np.any(y < 1) or np.any(y > GRID):
        raise ValueError("restricted boundary dof in partition")
    if np.unique(np.column_stack((x, y)), axis=0).shape[0] != GRID * GRID:
        raise ValueError("duplicate physical coordinates")
    bx = np.minimum((x - 1) // BLOCK_SIZE, SIDE_COUNT - 1)
    by = np.minimum((y - 1) // BLOCK_SIZE, SIDE_COUNT - 1)
    owner = (SIDE_COUNT * by + bx).astype(np.int64)
    domains = [np.flatnonzero(owner == block).astype(np.int64)
               for block in range(FACTOR_COUNT)]
    if any(item.size == 0 for item in domains) or sum(item.size for item in domains) != GRID * GRID:
        raise ValueError("partition coverage mismatch")
    return domains, owner, {
        "source": "dolfin.UnitSquareMesh(384,384)+CG1.tabulate_dof_coordinates[free_dofs]",
        "factor_count": FACTOR_COUNT,
        "block_size": BLOCK_SIZE,
        "grid": GRID,
        "domain_sizes": [int(item.size) for item in domains],
        "free_dofs_sha256": array_sha(free),
        "owner_sha256": array_sha(owner.astype(np.float64)),
    }


def matrix_pattern_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (tuple(left["shape"]) == tuple(right["shape"])
            and np.array_equal(left["indptr"], right["indptr"])
            and np.array_equal(left["indices"], right["indices"]))


def local_drift_scores(current: Mapping[str, Any],
                       reference_by_block: Sequence[Mapping[str, Any]],
                       domains: Sequence[np.ndarray]) -> np.ndarray:
    """
    【计算所有局部子域的累积扰动能量评分】
    
    算法机理:
    针对每个局部子域 i:
    1. 提取该子域构建时对应的历史参考矩阵 A_{ref, i}；
    2. 计算当前矩阵与参考矩阵的数据差值 ΔA = A_current - A_ref；
    3. 只累加该子域行范围内的非零元差分平方和 (即局部 Frobenius 范数的平方 ||ΔA_i||_F^2)；
    4. 返回长度为 FACTOR_COUNT 的一维扰动能量向量 scores。
    """
    scores = np.zeros(FACTOR_COUNT, dtype=np.float64)
    for block, (reference, rows) in enumerate(zip(reference_by_block, domains)):
        if not matrix_pattern_equal(reference, current):
            raise ValueError("reference/current CSR pattern changed")
        delta = np.asarray(current["data"], dtype=np.float64) - np.asarray(
            reference["data"], dtype=np.float64)
        total = 0.0
        for row in rows:
            lo = int(current["indptr"][int(row)])
            hi = int(current["indptr"][int(row) + 1])
            total += float(np.dot(delta[lo:hi], delta[lo:hi]))
        scores[block] = total
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValueError("invalid local drift scores")
    return scores


def select_mass_prefix(scores: Sequence[float], target: float = MASS_TARGET) -> Tuple[List[int], float, float, List[int]]:
    """
    【JSR 核心算法：mass95 动态扰动能量前缀截断选择器】
    
    设计哲学 (帕累托法则 / 80-20 原则):
    在物理界面移动过程中，往往只有少数几个前沿子域贡献了全场绝大部分的物理扰动。
    因此我们绝不采取粗暴的固定比例刷新 (如 50%)，而是:
    1. 将全场所有子域按其累积扰动能量从大到小降序排列；
    2. 像倒水一样将子域逐个装入能量桶，累加能量和；
    3. 【核心截断】: 一旦累加能量占总能量比例达到 target (预注册为 0.95，即 95%)，
       立即在此处一刀切断 (Prefix Truncation)！
    4. 只有桶内的少数高危子域被选中重新求逆，桶外剩余 5% 的微弱长尾子域免检放行，继续复用旧缓存。
    
    返回:
    - selected_order: 选中的高危子域编号升序列表
    - captured: 实际捕获的总能量占比 (>= 0.95)
    - total: 全场总扰动能量和
    - ranking: 全量子域按能量降序排列的原始名次列表
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (FACTOR_COUNT,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("invalid score vector")
    # 按扰动能量降序排序
    ranking = sorted(range(FACTOR_COUNT), key=lambda i: (-float(values[i]), i))
    total = float(np.sum(values))
    if total <= 0.0:
        selected_order: List[int] = []
    else:
        selected_order = []
        running = 0.0
        # 降序累加，达到 95% 立即截断跳出
        for block in ranking:
            selected_order.append(block)
            running += float(values[block])
            if running / total >= float(target):
                break
    captured = float(sum(float(values[i]) for i in selected_order)
                     / max(total, 1.0e-300))
    return sorted(selected_order), captured, total, ranking


def ordered_policy_rank(trajectory: str, previous_state: int,
                        current_state: int, repeat: int,
                        policy: str) -> int:
    seed = ORDER_SEED
    for value in (trajectory, previous_state, current_state, repeat):
        if isinstance(value, str):
            seed = (seed * 131 + sum(ord(c) for c in value)) % (2 ** 32)
        else:
            seed = (seed * 131 + int(value)) % (2 ** 32)
    rng = np.random.default_rng(seed)
    names = list(POLICIES)
    permutation = rng.permutation(len(names)).tolist()
    return int(permutation.index(names.index(policy)))


def load_frozen_target(path: Path) -> dict:
    """Load and validate the immutable calibration decision and provenance.

    The held-out runner must not accept a hand-written or partially copied
    summary.  Validate the summary, the independent audit, and the raw ledger
    header/footer as one chain before exposing the selected target to a
    rollout.  This function deliberately does not import the calibration
    runner or validator.
    """
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError("calibration summary does not exist: %s" % resolved)
    report = json.loads(resolved.read_text(encoding="utf-8"))
    expected_schema = "phase2.1-repair-size-calibration-summary-v1"
    expected_run_id = "phase2_1_calibration_v1_seed20260828"
    expected_records = 66
    expected_transitions = [list(pair) for pair in ((0, 1), (1, 2), (2, 3))]
    if report.get("schema") != expected_schema:
        raise ValueError("calibration summary schema mismatch")
    if str(report.get("run_id", "")) != expected_run_id:
        raise ValueError("calibration summary run id mismatch")
    try:
        summary_records = int(report.get("records", -1))
        footer_records = int(report.get("footer_records", -1))
    except (TypeError, ValueError):
        raise ValueError("calibration summary record counts are invalid")
    if summary_records != expected_records or footer_records != expected_records:
        raise ValueError("calibration summary record count mismatch")
    if report.get("transitions") != expected_transitions:
        raise ValueError("calibration summary calibration split mismatch")
    if report.get("trajectories") != list(TRAJECTORIES):
        raise ValueError("calibration summary trajectory catalog mismatch")
    decision = report.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("calibration summary has no decision")
    action = str(decision.get("selected_action", ""))
    fallback = bool(decision.get("fallback", False))
    if action == "mass95":
        target = MASS_TARGET
        kind = "mass"
        if fallback:
            raise ValueError("summary marks mass95 as fallback")
        try:
            selected_target = float(decision.get("selected_mass_target"))
        except (TypeError, ValueError):
            raise ValueError("mass95 decision has no numeric target")
        if not math.isclose(selected_target, MASS_TARGET,
                            rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError("mass95 decision target is not the frozen target")
    elif action == "local_full" and fallback:
        target = None
        kind = "fallback_local_full"
    else:
        raise ValueError("held-out requires selected mass95 or explicit local_full fallback")
    held_out = report.get("held_out_transitions")
    if held_out != [list(pair) for pair in TAIL_TRANSITIONS]:
        raise ValueError("calibration summary held-out split mismatch")

    # Bind the summary to the independently generated audit and its exact raw
    # ledger.  A missing/partial audit is not sufficient evidence for a
    # frozen target.
    audit_path = resolved.parent / "phase2_1_calibration_independent_audit.json"
    if not audit_path.exists():
        raise ValueError("independent calibration audit is missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("verdict") != "PASS_PHASE2_1_CALIBRATION_INDEPENDENT_AUDIT":
        raise ValueError("calibration independent audit did not pass")
    for key in ("records", "expected_records", "certified_passes"):
        try:
            value = int(audit.get(key, -1))
        except (TypeError, ValueError):
            raise ValueError("calibration audit %s is invalid" % key)
        if value != expected_records:
            raise ValueError("calibration audit %s mismatch" % key)
    try:
        certified_failures = int(audit.get("certified_failures", -1))
        residual_rechecks = int(audit.get("residual_rechecks", -1))
        future_violations = int(audit.get("future_information_violations", -1))
    except (TypeError, ValueError):
        raise ValueError("calibration audit counters are invalid")
    if certified_failures != 0 or residual_rechecks != expected_records:
        raise ValueError("calibration audit does not certify all records")
    if future_violations != 0:
        raise ValueError("calibration audit contains future-information violations")

    raw_ledger_value = report.get("raw_ledger")
    if not raw_ledger_value:
        raise ValueError("calibration raw ledger path is missing")
    raw_ledger_path = resolve_path(raw_ledger_value, resolved.parent)
    if not raw_ledger_path.exists():
        raise ValueError("calibration raw ledger is missing")
    audit_ledger_value = audit.get("ledger")
    if not audit_ledger_value:
        raise ValueError("calibration audit ledger path is missing")
    audit_ledger_path = resolve_path(audit_ledger_value, audit_path.parent)
    if audit_ledger_path.resolve() != raw_ledger_path.resolve():
        raise ValueError("calibration audit/raw ledger path mismatch")
    raw_lines = [line for line in raw_ledger_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if len(raw_lines) != expected_records + 2:
        raise ValueError("calibration raw ledger line count mismatch")
    try:
        raw_header = json.loads(raw_lines[0])
        raw_footer = json.loads(raw_lines[-1])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("calibration raw ledger is not valid JSONL") from exc
    if (raw_header.get("type") != "header"
            or raw_header.get("schema") != "phase2.1-repair-size-calibration-ledger-v1"
            or str(raw_header.get("run_id", "")) != expected_run_id
            or int(raw_header.get("expected_records", -1)) != expected_records):
        raise ValueError("calibration raw ledger header binding mismatch")
    if (raw_header.get("calibration_transitions") != expected_transitions
            or raw_header.get("trajectories") != list(TRAJECTORIES)):
        raise ValueError("calibration raw ledger split/catalog mismatch")
    if (raw_footer.get("type") != "footer"
            or raw_footer.get("schema") != raw_header.get("schema")
            or int(raw_footer.get("records", -1)) != expected_records
            or int(raw_footer.get("expected_records", -1)) != expected_records):
        raise ValueError("calibration raw ledger footer binding mismatch")
    # The frozen calibration runner predates the explicit footer run_id field;
    # if a later ledger includes it, it must still agree with the header.
    if "run_id" in raw_footer and str(raw_footer.get("run_id")) != expected_run_id:
        raise ValueError("calibration raw ledger footer run id mismatch")
    if str(audit.get("manifest_sha256", "")) != str(raw_header.get("manifest_sha256", "")):
        raise ValueError("calibration audit manifest binding mismatch")
    if str(audit.get("protocol_sha256", "")) != str(raw_header.get("protocol_sha256", "")):
        raise ValueError("calibration audit protocol binding mismatch")
    if str(audit.get("controller_tree_sha256", "")) != str(raw_header.get("controller_tree_sha256", "")):
        raise ValueError("calibration audit controller binding mismatch")
    return {
        "summary_path": str(path),
        "summary_resolved_path": str(resolved),
        "summary_sha256": file_sha(resolved),
        "summary_decision_sha256": json_sha(decision),
        "calibration_audit_path": str(audit_path),
        "calibration_audit_sha256": file_sha(audit_path),
        "calibration_raw_ledger_path": str(raw_ledger_path),
        "calibration_raw_ledger_sha256": file_sha(raw_ledger_path),
        "selected_action": action,
        "kind": kind,
        "target": target,
        "fallback": fallback,
        "summary_run_id": report.get("run_id"),
    }


def save_sidecar(path: Path, value: np.ndarray) -> dict:
    value = np.ascontiguousarray(value, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    return {
        "path": path.relative_to(RESULTS).as_posix(),
        "sha256": file_sha(path),
        "byte_sha256": array_sha(value),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "order": "C",
    }


def action_dict(action_id: str, selected: Iterable[int], refresh_coarse: bool,
                target: Optional[float], reason: str = "") -> dict:
    return {
        "action_id": str(action_id),
        "selected_local_blocks": sorted(set(int(x) for x in selected)),
        "selected_count": len(set(int(x) for x in selected)),
        "refresh_coarse_matrix": bool(refresh_coarse),
        "refresh_coarse_basis": False,
        "target": None if target is None else float(target),
        "reason": str(reason),
        "used_future_information": False,
    }


def failure_result(rhs: np.ndarray, error: str) -> dict:
    rhs_norm = float(np.linalg.norm(rhs))
    return {
        "iterations": 0,
        "converged_reason": -1,
        "rhs_norm": rhs_norm,
        "residual_norm": rhs_norm,
        "true_residual": 1.0,
        "tolerance": CERT_TOL,
        "converged": False,
        "solve_seconds": 0.0,
        "error": str(error),
    }


def expected_state(before: Mapping[str, Any], selected: Sequence[int],
                   refresh_coarse: bool, current_state: int) -> dict:
    selected_set = set(int(x) for x in selected)
    return {
        "local_build_state": [int(current_state) if block in selected_set else int(before["local_build_state"][block])
                              for block in range(FACTOR_COUNT)],
        "local_age": [0 if block in selected_set else int(before["local_age"][block]) + 1
                       for block in range(FACTOR_COUNT)],
        "coarse_matrix_build_state": int(current_state) if refresh_coarse else int(before["coarse_matrix_build_state"]),
        "coarse_matrix_age": 0 if refresh_coarse else int(before["coarse_matrix_age"]) + 1,
        "coarse_basis_build_state": 0,
        "coarse_basis_age": int(current_state),
    }


def choose_action(policy: str, scores: np.ndarray,
                  target_info: Mapping[str, Any], snapshot: Any = None
                  ) -> Tuple[dict, float, float, List[int]]:
    selected_mass, captured, total, ranking = select_mass_prefix(scores, MASS_TARGET)
    if policy == "always_reuse":
        return action_dict("reuse", (), False, None,
                           "fixed reuse comparator"), 0.0, total, ranking
    if snapshot is None and policy in ("fixed_partial_50", "raw_drift_ranking", "local_only_adaptive"):
        raise ValueError("adaptive comparator requires normalized causal snapshot")
    if policy in ("fixed_partial_50", "raw_drift_ranking"):
        order = sorted(range(FACTOR_COUNT),
                       key=lambda i: (-float(snapshot.local_drift[i]), i))
        selected = order[:PARTIAL_COUNT]
        action_id = "joint_partial" if policy == "fixed_partial_50" else "local_partial"
        coarse = policy == "fixed_partial_50"
        reason = ("fixed top-50-percent diagnostic policy" if coarse
                  else "fixed top-50-percent local-only policy")
        selected_capture = float(sum(float(scores[i]) for i in selected) / max(total, 1.0e-300))
        return action_dict(action_id, selected, coarse, None, reason), selected_capture, total, ranking
    if policy == "local_only_adaptive":
        local_soft = 0.50
        local_hard = 2.50
        max_age = 2
        max_partial = int(0.75 * FACTOR_COUNT)
        required = [i for i, (risk, age) in enumerate(
            zip(snapshot.local_risk, snapshot.local_age))
                    if float(risk) > local_soft or int(age) > max_age]
        required = sorted(required, key=lambda i: (-float(snapshot.local_risk[i]), i))
        if float(snapshot.max_local_risk) > local_hard or len(required) > max_partial:
            return action_dict("full_local_only", range(FACTOR_COUNT), False, None,
                               "local safety rule requires full local repair"), 1.0, total, ranking
        if required:
            selected_capture = float(sum(float(scores[i]) for i in required) / max(total, 1.0e-300))
            return action_dict("local_partial", required, False, None,
                               "causal local safety rule"), selected_capture, total, ranking
        return action_dict("reuse", (), False, None,
                           "local risk below soft budget"), 0.0, total, ranking
    if policy == "mass95_frozen":
        if target_info.get("kind") == "fallback_local_full":
            selected = list(range(FACTOR_COUNT))
            target = None
            action_id = "local_full_fallback"
            reason = "explicit calibration safety fallback"
        else:
            selected = selected_mass
            target = MASS_TARGET
            action_id = "mass95"
            reason = "frozen calibration mass target"
        selected_capture = 1.0 if target is None else captured
        return action_dict(action_id, selected, True, target, reason), selected_capture, total, ranking
    if policy == "full_rebuild":
        return action_dict("full_rebuild", range(FACTOR_COUNT), True, None,
                           "full local and coarse rebuild"), 1.0, total, ranking
    raise ValueError("unknown policy: %s" % policy)


def _source_binding(manifest_path: Path, manifest_sha: str,
                    entries: Sequence[Mapping[str, Any]], trajectory: str,
                    previous_state: int, current_state: int) -> dict:
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "trajectory": trajectory,
        "previous_state": int(previous_state),
        "current_state": int(current_state),
        "previous_matrix": entries[previous_state]["matrix"],
        "previous_rhs": entries[previous_state]["rhs"],
        "matrix": entries[current_state]["matrix"],
        "rhs": entries[current_state]["rhs"],
    }


def execute_transition(trajectory: str, policy: str, repeat: int,
                       previous_state: int, current_state: int,
                       split: str, previous: dict, rhs: np.ndarray,
                       current: dict, entries: Sequence[Mapping[str, Any]],
                       state_matrices: Mapping[int, dict], lifecycle: Any,
                       domains: Sequence[np.ndarray], owner: np.ndarray,
                       target_info: Mapping[str, Any], manifest_path: Path,
                       manifest_sha: str, action_rank: int,
                       inject: Optional[Mapping[str, Any]] = None) -> dict:
    """
    【单步物理状态转换执行器 (带事务快照与确定性熔断重构循环)】
    
    核心生命周期:
    1. 保存快照: lifecycle.snapshot() 记录当前干净缓存，作为容错回滚底板；
    2. 因果感知: local_drift_scores + compute_snapshot 计算扰动，严禁读取未来算子；
    3. 策略决策: choose_action 依据当前策略挑选动作 (如 mass95 选中的高危子域与联合更新标志)；
    4. 原地更新: lifecycle.update_operator(current) 装载新物理算子 A_t；
    5. 事务尝试循环:
       - 恢复快照 -> 执行选中的局部块求逆 + 粗网格联合更新 -> 调用 PETSc CG 迭代求解；
       - 外部核验物理真实残差 ||b - Ax|| / ||b|| <= 1e-8；
       - 若认证通过 -> 成功跳出，提交状态 (Commit)；
       - 若认证失败 -> 触发熔断升级！自动回滚并升级为 full_rebuild (全量重构) 重新求解，
         确保 100% 物理真实收敛，杜绝任何发散崩溃！
    """
    # ``_free_dofs`` is attached by ``run_rollout`` from the validated source
    # archive.  Refuse to construct a partition provenance record if a caller
    # forgets it (or supplies a placeholder), rather than silently hashing a
    # zero vector.
    free_dofs_value = current.get("_free_dofs")
    if free_dofs_value is None:
        raise ValueError("current state is missing validated free_dofs")
    free_dofs = np.asarray(free_dofs_value, dtype=np.int64)
    if (free_dofs.shape != (GRID * GRID,)
            or np.unique(free_dofs).size != free_dofs.size):
        raise ValueError("current state has invalid free_dofs")
        
    # 1. 导出当前状态与不可变快照 (作为容错事务回滚底板)
    state_before = lifecycle.state_dict(previous_state)
    cache_before = lifecycle.cache_proof()
    snapshot = lifecycle.snapshot()
    
    # 生产动作计时开始 (包含感知、决策、装载、维护与求解全流程)
    action_started = time.perf_counter()
    
    # 2. 状态记忆溯源: 提取各个局部子域上次构建时的历史矩阵 (严禁前瞻未来)
    references = [state_matrices[int(value)] for value in state_before["local_build_state"]]
    scores_started = time.perf_counter()
    # 计算全场子域累积扰动能量向量 ||ΔA_i||_F^2
    scores = local_drift_scores(current, references, domains)
    try:
        from corrected_phase2.monitor import compute_snapshot
        adaptive_snapshot = compute_snapshot(
            current=current, domains=domains,
            local_build_states=state_before["local_build_state"],
            local_ages=state_before["local_age"], state_matrices=state_matrices,
            coarse_build_state=int(state_before["coarse_matrix_build_state"]),
            previous_state=int(previous_state), current_state=int(current_state),
            trajectory=trajectory,
        )
    except Exception:
        adaptive_snapshot = None
        
    # 3. 策略决策: 执行 mass95 前缀截断选择
    action, captured, total_mass, ranking = choose_action(
        policy, scores, target_info, adaptive_snapshot)
    selection_seconds = time.perf_counter() - scores_started
    selected = list(action["selected_local_blocks"])

    # 4. 原地安装当前时间步算子 A_t
    operator_started = time.perf_counter()
    operator_seconds = float(lifecycle.update_operator(current))
    # Keep a separate operator timer field even if the backend returns the
    # measured interval; the timer starts after causal selection.
    del operator_started
    attempts: List[dict] = []
    current_action = action
    final_result: Optional[dict] = None
    final_x: Optional[np.ndarray] = None
    final_residual: Optional[np.ndarray] = None
    restore_seconds_total = 0.0
    terminal_index = -1

    while True:
        restore_started = time.perf_counter()
        lifecycle.restore(snapshot)
        restore_seconds = float(time.perf_counter() - restore_started)
        restore_seconds_total += restore_seconds
        restored_digest = lifecycle.cache_digest()
        attempt_started = time.perf_counter()
        coarse_meta: Optional[dict] = None
        local_meta: Optional[dict] = None
        solve_result: Optional[dict] = None
        x_value: Optional[np.ndarray] = None
        residual_value: Optional[np.ndarray] = None
        error_text = ""
        injected = False
        try:
            if bool(current_action["refresh_coarse_matrix"]):
                coarse_meta = lifecycle.refresh_coarse(current, current_state)
            else:
                lifecycle.retain_coarse()
                coarse_meta = {"seconds": 0.0, "assembly_seconds": 0.0,
                                "factorization_seconds": 0.0}
            local_meta = lifecycle.refresh_local(
                current, current_state, current_action["selected_local_blocks"]
            )
            if (inject is not None and not attempts
                    and str(inject.get("trajectory")) == trajectory
                    and str(inject.get("policy")) == policy
                    and int(inject.get("repeat", -1)) == int(repeat)
                    and int(inject.get("previous_state", -1)) == int(previous_state)
                    and int(inject.get("current_state", -1)) == int(current_state)):
                # Deliberately fail after mutating the cache.  The next loop
                # must restore the exact snapshot before escalating.
                injected = True
                raise RuntimeError("registered_phase2_1_heldout_injection")
            calls_before = int(lifecycle.context.calls)
            apply_before = float(lifecycle.context.seconds)
            solve_result, x_value, residual_value = lifecycle.solve(rhs, CERT_TOL)
            apply_calls = int(lifecycle.context.calls) - calls_before
            apply_seconds = float(lifecycle.context.seconds) - apply_before
            if not bool(solve_result["converged"]):
                error_text = "solver did not produce a passing certificate"
        except Exception as exc:
            error_text = "%s: %s" % (type(exc).__name__, str(exc))
            apply_calls = 0
            apply_seconds = 0.0
            solve_result = failure_result(rhs, error_text)
            x_value = None
            residual_value = None
        attempt_total = float(time.perf_counter() - attempt_started)
        post_attempt = lifecycle.cache_proof()
        attempt = {
            "attempt_index": len(attempts),
            "action": current_action,
            "pre_attempt_cache_digest": cache_before["cache_digest"],
            "restored_cache_digest": restored_digest,
            "post_attempt_cache_digest": post_attempt["cache_digest"],
            "restore_seconds": restore_seconds,
            "coarse_refresh_seconds": float((coarse_meta or {}).get("seconds", 0.0)),
            "coarse_assembly_seconds": float((coarse_meta or {}).get("assembly_seconds", 0.0)),
            "coarse_factorization_seconds": float((coarse_meta or {}).get("factorization_seconds", 0.0)),
            "factor_refresh_seconds": float((local_meta or {}).get("seconds", 0.0)),
            "preconditioner_apply_calls": int(apply_calls),
            "preconditioner_apply_seconds": float(apply_seconds),
            "solve_seconds": float((solve_result or {}).get("solve_seconds", 0.0)),
            "attempt_total_seconds": attempt_total,
            "iterations": int((solve_result or {}).get("iterations", 0)),
            "converged_reason": int((solve_result or {}).get("converged_reason", -1)),
            "true_residual": float((solve_result or {}).get("true_residual", 1.0)),
            "converged": bool((solve_result or {}).get("converged", False)),
            "error": error_text,
            "injected_failure": bool(injected),
            "future_states_read": [],
            "future_information_used": False,
        }
        attempts.append(attempt)
        final_result, final_x, final_residual = solve_result, x_value, residual_value
        terminal_index = len(attempts) - 1
        if bool(solve_result and solve_result.get("converged")):
            break
        # There is one deterministic escalation path.  It is deliberately
        # independent of the failed outcome beyond the pass/fail certificate.
        if current_action["action_id"] == "full_rebuild":
            break
        current_action = action_dict(
            "full_rebuild", range(FACTOR_COUNT), True, None,
            "deterministic escalation after failed certificate",
        )

    action_total = float(time.perf_counter() - action_started)
    if final_result is None:
        final_result = failure_result(rhs, "no solve result")
    state_committed = bool(final_result.get("converged", False))
    if state_committed:
        state_after = lifecycle.state_dict(current_state)
        cache_after = lifecycle.cache_proof()
        wanted_after = expected_state(
            state_before, current_action["selected_local_blocks"],
            bool(current_action["refresh_coarse_matrix"]), current_state,
        )
        if state_after != wanted_after:
            raise AssertionError("backend state/action mismatch")
    else:
        lifecycle.restore(snapshot)
        state_after = dict(state_before)
        cache_after = lifecycle.cache_proof()

    # Sidecars are written only after stopping the production timer.
    tag = "%s_%s_r%02d_t%02d_t%02d_%s" % (
        trajectory, policy, int(repeat), int(previous_state), int(current_state), split
    )
    solution_meta = None
    residual_meta = None
    if state_committed and final_x is not None and final_residual is not None:
        solution_meta = save_sidecar(RESULTS / (tag + "_x.npy"), final_x)
        residual_meta = save_sidecar(RESULTS / (tag + "_r.npy"), final_residual)

    coarse_total = sum(float(x["coarse_refresh_seconds"]) for x in attempts)
    local_total = sum(float(x["factor_refresh_seconds"]) for x in attempts)
    solve_total = sum(float(x["solve_seconds"]) for x in attempts)
    components_sum = (selection_seconds + operator_seconds + restore_seconds_total
                      + coarse_total + local_total + solve_total)
    timer_gap = action_total - components_sum
    if timer_gap < -1.0e-6:
        raise RuntimeError("negative production timer gap %.9g" % timer_gap)
    if timer_gap < 0.0:
        # Account for sub-microsecond clock-ordering noise, but never hide a
        # material accounting error from the independent audit.
        timer_gap = 0.0

    reference_states = sorted(set(int(x) for x in state_before["local_build_state"])
                              | {int(state_before["coarse_matrix_build_state"])})
    record = {
        "type": "phase2_2_comparative_transition",
        "schema": SCHEMA,
        "protocol": PROTOCOL_ID,
        "run_id": RUN_ID,
        "rollout_id": "%s/%s/r%02d" % (trajectory, policy, int(repeat)),
        "trajectory": trajectory,
        "policy": policy,
        "controller": policy,
        "repeat": int(repeat),
        "split": split,
        "previous_state": int(previous_state),
        "current_state": int(current_state),
        "transition_order": int(previous_state),
        "action_order_rank": int(action_rank),
        "action": current_action,
        "initial_action": action,
        "final_action": current_action,
        "monitor": {
            "local_drift_squared_scores": scores.tolist(),
            "total_drift_squared": float(total_mass),
            "captured_drift_mass": float(captured),
            "ranking": ranking,
            "ranking_sha256": sha256(json.dumps(ranking, separators=(",", ":")).encode("ascii")).hexdigest(),
            "target": MASS_TARGET,
            "local_drift": [] if adaptive_snapshot is None else list(adaptive_snapshot.local_drift),
            "local_risk": [] if adaptive_snapshot is None else list(adaptive_snapshot.local_risk),
            "coarse_matrix_risk": 0.0 if adaptive_snapshot is None else float(adaptive_snapshot.coarse_matrix_risk),
            "coarse_age": 0 if adaptive_snapshot is None else int(adaptive_snapshot.coarse_age),
            "reference_states_read": reference_states,
            "future_states_read": [],
            "future_information_used": False,
        },
        "causal_inputs": {
            "previous_state_id": int(previous_state),
            "current_state_id": int(current_state),
            "local_build_states": list(state_before["local_build_state"]),
            "coarse_build_state": int(state_before["coarse_matrix_build_state"]),
            "reference_states_read": reference_states,
            "future_states_read": [],
            "future_information_used": False,
        },
        "state_before": state_before,
        "state_after": state_after,
        "state_committed": state_committed,
        "cache_before": cache_before,
        "cache_after": cache_after,
        "attempts": attempts,
        "fallback_escalation": {
            "events": [
                {"from": attempts[i]["action"]["action_id"],
                 "to": attempts[i + 1]["action"]["action_id"],
                 "reason": "failed independent true-residual certificate",
                 "future_information_used": False}
                for i in range(max(0, len(attempts) - 1))
            ],
            "restart_count": max(0, len(attempts) - 1),
            "terminal_attempt_index": int(terminal_index),
            "deterministic": True,
            "future_information_used": False,
        },
        "certificate": {
            "relative_true_residual": float(final_result["true_residual"]),
            "true_residual_norm": float(final_result["residual_norm"]),
            "rhs_norm": float(final_result["rhs_norm"]),
            "tolerance": CERT_TOL,
            "passed": bool(final_result["converged"]),
            "solution_sidecar_sha256": None if solution_meta is None else solution_meta["sha256"],
        },
        "solve": final_result,
        "solution_sidecar": solution_meta,
        "residual_sidecar": residual_meta,
        "timing": {
            "selection_seconds": float(selection_seconds),
            "operator_update_seconds": float(operator_seconds),
            "coarse_refresh_seconds": float(coarse_total),
            "coarse_assembly_seconds": float(sum(float(x["coarse_assembly_seconds"]) for x in attempts)),
            "coarse_factorization_seconds": float(sum(float(x["coarse_factorization_seconds"]) for x in attempts)),
            "factor_refresh_seconds": float(local_total),
            "cache_restore_seconds": float(restore_seconds_total),
            "preconditioner_apply_calls": int(sum(int(x["preconditioner_apply_calls"]) for x in attempts)),
            "preconditioner_apply_seconds": float(sum(float(x["preconditioner_apply_seconds"]) for x in attempts)),
            "preconditioner_apply_included_in_krylov": True,
            "krylov_solve_seconds": float(solve_total),
            "attempt_total_seconds": float(sum(float(x["attempt_total_seconds"]) for x in attempts)),
            "action_total_seconds": float(action_total),
            "components_sum_seconds": float(components_sum),
            "timer_gap_seconds": float(timer_gap),
            "sidecar_serialization_excluded": True,
        },
        "production_cost": {
            "warm_action_seconds": float(action_total),
            "common_initialization_seconds": 0.0,
            "initial_setup_seconds": 0.0,
            "initial_setup_included_in_primary": False,
            "accounting_mode": "heldout_sequential_warm_cache_action",
        },
        "source_binding": _source_binding(
            manifest_path, manifest_sha, entries, trajectory, previous_state, current_state
        ),
        "matrix_data_sha256": matrix_data_sha(current),
        "rhs_data_sha256": array_sha(rhs),
        "csr": {
            "shape": list(current["shape"]),
            "nnz": int(current["data"].size),
            "pattern_sha256": csr_pattern_sha(current),
            "data_sha256": raw_array_sha(current["data"]),
        },
        "partition": {
            "source": "dolfin.UnitSquareMesh(384,384)+CG1.tabulate_dof_coordinates[free_dofs]",
            "factor_count": FACTOR_COUNT,
            "block_size": BLOCK_SIZE,
            "grid": GRID,
            "domain_sizes": [int(x.size) for x in domains],
            "free_dofs_sha256": array_sha(free_dofs),
            "owner_sha256": array_sha(owner.astype(np.float64)),
        },
        "target_binding": dict(target_info),
        "future_information_used": False,
    }
    return record


def run_rollout(trajectory: str, policy: str, repeat: int,
                manifest: Mapping[str, Any], state_cache: Mapping[Tuple[str, int], Tuple[dict, np.ndarray, np.ndarray]],
                domains: Sequence[np.ndarray], owner: np.ndarray,
                target_info: Mapping[str, Any], manifest_path: Path,
                manifest_sha: str, backend_module: Any,
                inject: Optional[Mapping[str, Any]] = None) -> List[dict]:
    entries = manifest["trajectories"][trajectory]
    matrix0, rhs0, free0 = state_cache[(trajectory, 0)]
    matrix0["_free_dofs"] = free0
    lifecycle = backend_module.CorrectedStatefulTwoLevel(
        None, matrix0, rhs0, domains, owner, 0, KSP_RTOL, MAX_IT
    )
    initial_setup = float(lifecycle.initial_setup_seconds)
    records: List[dict] = []
    # Only states already traversed are exposed to the monitor.  We do not
    # preload the tail into this map.
    state_matrices: Dict[int, dict] = {0: matrix0}
    try:
        for previous_state, current_state in ALL_TRANSITIONS:
            current, rhs, free_current = state_cache[(trajectory, current_state)]
            current["_free_dofs"] = free_current
            previous = state_matrices[previous_state]
            if not np.array_equal(free0, free_current):
                raise ValueError("free_dofs changed along rollout")
            if not matrix_pattern_equal(previous, current):
                raise ValueError("CSR pattern changed along rollout")
            split = "warmup" if (previous_state, current_state) in WARMUP_TRANSITIONS else "held_out"
            if split == "warmup":
                # Prefix actions are fixed and common across policies.  The
                # action is represented by the same transition function, but
                # target selection is not used to score the tail.
                effective_policy = "full_rebuild"
                rank = 0
            else:
                effective_policy = policy
                rank = ordered_policy_rank(trajectory, previous_state, current_state,
                                           repeat, policy)
            record = execute_transition(
                trajectory, effective_policy, repeat, previous_state, current_state,
                split, previous, rhs, current, entries, state_matrices, lifecycle,
                domains, owner, target_info, manifest_path, manifest_sha, rank,
                inject if split == "held_out" else None,
            )
            # Preserve the requested controller name on warm-up records and
            # annotate the common-prefix semantics explicitly.
            record["controller"] = policy
            record["policy"] = policy
            record["warmup_common_action"] = bool(split == "warmup")
            record["scored"] = bool(split == "held_out")
            record["initial_setup_seconds"] = initial_setup
            record["production_cost"]["initial_setup_seconds"] = initial_setup
            records.append(record)
            if not bool(record["state_committed"]):
                # A failed transition must not expose an uncommitted current
                # matrix as the causal predecessor of the next transition.
                # The failure record is retained in the in-memory rollout only
                # long enough to make the error explicit; normal ledgers are
                # written only after a complete rollout succeeds.
                raise RuntimeError(
                    "transition failed without a committed state: %s/%s r%d %d->%d"
                    % (trajectory, policy, int(repeat),
                       int(previous_state), int(current_state))
                )
            state_matrices[current_state] = current
    finally:
        lifecycle.close()
    return records


def parse_injection(value: Optional[str]) -> Optional[dict]:
    if not value:
        return None
    # Explicit key=value syntax keeps a fault run auditable and avoids a
    # hidden random failure.  Example: trajectory=stress_moving_interface,
    # policy=mass95_frozen,repeat=0,previous=6,current=7
    fields: Dict[str, str] = {}
    for part in value.split(","):
        if "=" not in part:
            raise ValueError("injection must use key=value fields")
        key, val = part.split("=", 1)
        fields[key.strip()] = val.strip()
    required = ("trajectory", "policy", "repeat", "previous", "current")
    if any(key not in fields for key in required):
        raise ValueError("injection lacks one of %s" % ",".join(required))
    unknown = sorted(set(fields) - set(required))
    if unknown:
        raise ValueError("injection has unknown fields: %s" % ",".join(unknown))
    repeat = int(fields["repeat"])
    previous_state = int(fields["previous"])
    current_state = int(fields["current"])
    if repeat < 0:
        raise ValueError("injection repeat must be non-negative")
    if (previous_state, current_state) not in TAIL_TRANSITIONS:
        raise ValueError("injection must target one held-out tail transition")
    return {
        "trajectory": fields["trajectory"],
        "policy": fields["policy"],
        "repeat": repeat,
        "previous_state": previous_state,
        "current_state": current_state,
    }


def run(repeats: int, injection: Optional[dict] = None) -> None:
    if os.environ.get("ALLOW_STATEFUL_MAINTENANCE_V1") != "1":
        raise SystemExit("set ALLOW_STATEFUL_MAINTENANCE_V1=1")
    if int(repeats) < 3:
        raise ValueError("held-out evaluation requires at least three repeats")
    if injection is not None and int(injection.get("repeat", -1)) >= int(repeats):
        raise ValueError("injection repeat is outside the requested rollout set")
    if RAW.exists():
        raise FileExistsError("refusing to overwrite existing held-out ledger: %s" % RAW)
    manifest_path = resolve_path(MANIFEST)
    protocol_path = resolve_path(PROTOCOL)
    if not manifest_path.exists() or not protocol_path.exists():
        raise FileNotFoundError("manifest or held-out protocol is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_cache, free = validate_manifest(manifest)
    domains, owner, partition = physical_partition(free)
    target_info = load_frozen_target(CALIBRATION_SUMMARY)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if injection is not None:
        # A fault ledger must be physically separate from normal performance
        # output.  The caller should pass PHASE2_2_RESULTS/RAW to a new
        # directory; refuse an explicitly shared normal path (including a
        # custom RAW path that points back into the normal output tree).
        normal_root = (ROOT / "results" / "phase2_2_comparative_v1").resolve()
        raw_resolved = RAW.resolve()
        if RESULTS.resolve() == normal_root or normal_root in raw_resolved.parents:
            raise ValueError("injected runs require a separate output directory")
    from corrected_phase2 import backend as backend_module

    manifest_sha = file_sha(manifest_path)
    environment = environment_record()
    expected_tail = len(TRAJECTORIES) * len(POLICIES) * int(repeats) * len(TAIL_TRANSITIONS)
    expected_warmup = len(TRAJECTORIES) * len(POLICIES) * int(repeats) * len(WARMUP_TRANSITIONS)
    header = {
        "type": "header",
        "schema": SCHEMA,
        "protocol": PROTOCOL_ID,
        "run_id": RUN_ID,
        "trajectories": list(TRAJECTORIES),
        "policies": list(POLICIES),
        "repeats": int(repeats),
        "warmup_transitions": [list(x) for x in WARMUP_TRANSITIONS],
        "held_out_transitions": [list(x) for x in TAIL_TRANSITIONS],
        "expected_tail_records": expected_tail,
        "expected_warmup_records": expected_warmup,
        "expected_records": expected_tail + expected_warmup,
        "records_include_warmup": True,
        "warmup_action": "full_rebuild",
        "controller_specs": {
            "always_reuse": {"local": "none", "coarse": "retain"},
            "fixed_partial_50": {"local": "top_72_normalized_drift", "coarse": "refresh"},
            "raw_drift_ranking": {"local": "top_72_normalized_drift", "coarse": "retain"},
            "local_only_adaptive": {"local": "causal_risk_rule", "coarse": "retain"},
            "mass95_frozen": {"local": "frozen_mass95_prefix", "coarse": "refresh"},
            "full_rebuild": {"local": "all", "coarse": "refresh"},
        },
        "target_binding": target_info,
        "selector_config": {
            "mass_target": MASS_TARGET,
            "ranking": "descending raw squared causal drift, ties by block id",
            "reference": "each block last-built matrix",
        },
        "manifest_path": str(MANIFEST),
        "manifest_sha256": manifest_sha,
        "protocol_sha256": file_sha(protocol_path),
        "production_tree_sha256": production_tree_sha(),
        "action_order_seed": ORDER_SEED,
        "partition": partition,
        "environment": environment,
        "environment_sha256": json_sha(environment),
        "failure_injection": injection,
        "future_information_used": False,
    }
    count = 0
    tail_count = 0
    warmup_count = 0
    with RAW.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(header, sort_keys=True) + "\n")
        stream.flush()
        for trajectory in TRAJECTORIES:
            for policy in POLICIES:
                for repeat in range(int(repeats)):
                    rollout = run_rollout(
                        trajectory, policy, repeat, manifest, state_cache,
                        domains, owner, target_info, manifest_path, manifest_sha,
                        backend_module, injection,
                    )
                    for record in rollout:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                        stream.flush()
                        count += 1
                        if record["scored"]:
                            tail_count += 1
                        else:
                            warmup_count += 1
                        print(json.dumps({
                            "records": count,
                            "expected": expected_tail + expected_warmup,
                            "trajectory": trajectory,
                            "policy": policy,
                            "repeat": repeat,
                            "split": record["split"],
                            "transition": "%d->%d" % (record["previous_state"], record["current_state"]),
                            "action": record["final_action"]["action_id"],
                            "attempts": len(record["attempts"]),
                            "converged": record["solve"]["converged"],
                        }, sort_keys=True), flush=True)
        footer = {
            "type": "footer",
            "schema": SCHEMA,
            "run_id": RUN_ID,
            "records": count,
            "tail_records": tail_count,
            "warmup_records": warmup_count,
            "expected_records": expected_tail + expected_warmup,
            "expected_tail_records": expected_tail,
            "expected_warmup_records": expected_warmup,
        }
        stream.write(json.dumps(footer, sort_keys=True) + "\n")
        stream.flush()
    print(json.dumps({
        "status": "PASS_PHASE2_2_COMPARATIVE_RAW",
        "records": count,
        "tail_records": tail_count,
        "warmup_records": warmup_count,
        "path": str(RAW),
    }, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int,
                        default=int(os.environ.get("PHASE2_2_REPEATS", "3")))
    parser.add_argument("--inject", default=os.environ.get("PHASE2_2_INJECT"),
                        help="separate-run key=value fault specification")
    args = parser.parse_args(argv)
    run(args.repeats, parse_injection(args.inject))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

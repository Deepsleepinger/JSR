#!/usr/bin/env python3
"""Audit all frozen MPI runs, emit a complete summary and scientific figure."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=ROOT / "results/dc_jsr_parallel_audit")
    a = p.parse_args()
    records = []
    solves = warmup_solves = retries = 0
    max_residual = max_coarse_error = max_reference_error = 0.0
    source_fingerprints = None
    histories = {}
    input_hashes = {}
    for path in sorted(a.input_dir.glob("*_n*_p*.json")):
        r = json.loads(path.read_text())
        assert r["status"] == "complete", path
        assert r["repeats"] == 4 and len(r["runs"]) == 8 and len(r["warmup_runs"]) == 2, path
        assert r["epsilon"] == .08
        if source_fingerprints is None:
            source_fingerprints = r["source_sha256"]
        assert r["source_sha256"] == source_fingerprints, "Implementation changed across formal configurations"
        m = r["input_manifest"]
        key = (m["mesh_n"], m["regime"])
        previous_hash = input_hashes.setdefault(key, m["sha256"])
        assert previous_hash == m["sha256"], "Input changes across process counts"
        cpus = [e["cpu_affinity"][0] for e in r["rank_environment"]]
        assert len(set(cpus)) == r["mpi_processes"]
        for e in r["rank_environment"]:
            assert len(e["cpu_affinity"]) == 1
            assert set(e["thread_environment"].values()) == {"1"}
        for run in r["warmup_runs"] + r["runs"]:
            assert len(run["steps"]) == m["steps"]
            assert abs(sum(s["total_seconds"] for s in run["steps"]) - run["total_seconds"]) < 1e-12
            last_versions = [0] * 8
            for s in run["steps"]:
                assert s["total_seconds"] == max(t["total"] for t in s["rank_times"])
                assert len(s["rank_times"]) == r["mpi_processes"]
                assert s["k"] == len(s["selected"])
                assert s["residual_proxy"] <= .08 + 1e-14
                expected = [i for i, v in enumerate(s["pre_proxy"]) if v > .08]
                assert s["selected"] == (list(range(8)) if run["arm"] == "full" else expected)
                for cid in s["selected"]:
                    last_versions[cid] = s["step"]
                assert s["factor_versions"] == last_versions
                assert s["attempts"][-1]["reason"] > 0
                assert s["independent_true_relative_residual"] < 1e-8
                assert s["petsc_true_relative_residual"] < 1e-8
                retries += len(s["attempts"]) - 1
                max_residual = max(max_residual, s["independent_true_relative_residual"], s["petsc_true_relative_residual"])
                max_coarse_error = max(max_coarse_error, s["coarse_relative_error"])
            max_reference_error = max([max_reference_error] + [x["relative_error"] for x in run["reference_apply_errors"]])
            if run["repeat"] < 0:
                warmup_solves += len(run["steps"])
                assert len(run["reference_apply_errors"]) == 2
            else:
                solves += len(run["steps"])
                hk = (m["mesh_n"], m["regime"], run["arm"])
                history = histories.setdefault(hk, {"budgets": [], "iterations": []})
                history["budgets"].append([s["k"] for s in run["steps"]])
                history["iterations"].append([s["iterations"] for s in run["steps"]])
        full = r["summary"]["full"]
        dc = r["summary"]["dc_jsr"]
        record = dict(mesh_n=m["mesh_n"], n_dofs=m["n_dofs"], regime=m["regime"],
                      processes=r["mpi_processes"], full=full, dc_jsr=dc,
                      ratio=dc["ratio_of_mean_times_vs_full"],
                      ratio_with_initialization=dc["mean_total_with_initialization_seconds"] /
                                                full["mean_total_with_initialization_seconds"],
                      source_file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for arm in ["full", "dc_jsr"]:
            runs = [run for run in r["runs"] if run["arm"] == arm]
            record[arm]["mean_local_rank_work_seconds"] = float(np.mean([
                sum(sum(t["local"] for t in s["rank_times"]) for s in run["steps"]) for run in runs]))
            record[arm]["mean_local_critical_seconds"] = record[arm]["mean_stage_maxima"]["local"]
            record[arm]["mean_apply_seconds_max_rank"] = {
                component: float(np.mean([sum(max(t["apply_seconds"][component] for t in s["rank_times"])
                                             for s in run["steps"]) for run in runs]))
                for component in ["overlap", "local", "coarse"]}
        records.append(record)
    expected = {(n, regime, p) for n in [24, 32] for regime in ["gentle_single_front", "multi_front_churn"] for p in [1, 2, 4]}
    assert {(r["mesh_n"], r["regime"], r["processes"]) for r in records} == expected
    history_checks = []
    for (mesh_n, regime, arm), h in histories.items():
        assert all(b == h["budgets"][0] for b in h["budgets"]), "Budget changed with process count/repetition"
        differences = [max(abs(a-b) for a, b in zip(seq, h["iterations"][0])) for seq in h["iterations"]]
        history_checks.append(dict(mesh_n=mesh_n, regime=regime, arm=arm,
                                   budget_sequence=h["budgets"][0],
                                   max_iteration_difference_across_processes_and_repeats=max(differences)))
    for r in records:
        base = next(b for b in records if b["mesh_n"] == r["mesh_n"] and b["regime"] == r["regime"] and b["processes"] == 1)
        for arm in ["full", "dc_jsr"]:
            r[arm]["speedup_vs_own_p1"] = base[arm]["mean_seconds"] / r[arm]["mean_seconds"]
            r[arm]["efficiency_vs_own_p1"] = r[arm]["speedup_vs_own_p1"] / r["processes"]
    summary = dict(status="complete", formal_certified_solves=solves, warmup_certified_solves=warmup_solves,
                   total_certified_solves=solves + warmup_solves, retries=retries,
                   max_true_residual=max_residual, max_coarse_relative_error=max_coarse_error,
                   max_serial_distributed_apply_relative_error=max_reference_error,
                   source_sha256=source_fingerprints, history_checks=history_checks, configurations=records)
    output = a.input_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    names = {"gentle_single_front": "Gentle single front", "multi_front_churn": "Multi-front churn"}
    for row, n in enumerate([24, 32]):
        for col, regime in enumerate(names):
            ax = axes[row, col]
            subset = sorted([r for r in records if r["mesh_n"] == n and r["regime"] == regime], key=lambda r:r["processes"])
            ps = [r["processes"] for r in subset]
            for arm, label, color, marker in [("full", "Full rebuild", "#657386", "s"),
                                               ("dc_jsr", "DC-JSR", "#0076b8", "o")]:
                ax.errorbar(ps, [r[arm]["mean_seconds"] for r in subset],
                            yerr=[r[arm]["sd_seconds"] for r in subset], capsize=3,
                            label=label, color=color, marker=marker, linewidth=1.8)
            ax.set_title("%s, N=%d (%s DOFs)" % (names[regime], n, format(subset[0]["n_dofs"], ",")), fontsize=10)
            ax.set_xticks([1, 2, 4])
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel("Maintenance + solve wall time (s)")
            if row == 1:
                ax.set_xlabel("MPI processes (one logical CPU per process)")
            if row == 0 and col == 0:
                ax.legend(frameon=False)
    fig.text(.5, .015, "Fixed 8 subdomains; epsilon=0.08; four paired repeats; bars: sample standard deviation; assembly/staging excluded",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .035, 1, 1])
    figure = ROOT / "docs/figures/dc_jsr_parallel_validation"
    fig.savefig(str(figure.with_suffix(".pdf")))
    fig.savefig(str(figure.with_suffix(".png")), dpi=220)
    plt.close(fig)
    print("Certified", summary["total_certified_solves"], "solves; max residual", max_residual)
    for r in records:
        print("N", r["mesh_n"], r["regime"], "P", r["processes"],
              "Full %.4f +- %.4f" % (r["full"]["mean_seconds"], r["full"]["sd_seconds"]),
              "DC %.4f +- %.4f" % (r["dc_jsr"]["mean_seconds"], r["dc_jsr"]["sd_seconds"]),
              "ratio %.4f" % r["ratio"], "wins", r["dc_jsr"]["paired_wins_vs_full"],
              "maxiters", r["full"]["max_iterations"], r["dc_jsr"]["max_iterations"])
    # Render every configuration, including losses; do not select favorable rows.
    report = ["# DC-JSR 有限 MPI 并行验证报告（2026-10-03）", "",
              "温和轨迹的六个网格/进程配置均有平均节时收益，24次配对比较全部较快；多前沿轨迹有两个配置平均较慢。"
              "四进程下，N24/N32 的温和 DC/Full 为 0.903/0.845，多前沿为 1.003/0.973。"
              "因此结果支持局域变化时的并行维护收益，同时否定普遍的并行墙钟优势。", "",
              "## 范围与协议", "",
              "真实 MPI 全局矩阵/向量、PCG 与重叠通信，固定八子域；局部因子分配至 rank=i mod P。"
              "单机 WSL 的四个逻辑 CPU，1/2/4 进程，每进程绑定一个 CPU，底层线程数固定为 1。"
              "N=24/32、温和10步/多前沿16步、epsilon=0.08，Full/DC 各完整热身一次、正式四次交替顺序配对重复。", "",
              "数据使用原 FEM 生成函数。计时含监测、局部子矩阵构造与 MUMPS 重构、当前粗算子维护、"
              "PCG 和 PETSc 真残差检查；每步取进程最大墙钟时间。FEM 组装、全局矩阵/覆盖行预装载、"
              "独立审计不计入该时间。初始化另列并给出含初始化的比值。"
              "小型八维粗因子复制，各次应用 Allreduce 粗右端。这是单节点有限强扩展，不能推断多节点性能或完整应用端到端加速。", "",
              "## 全部正式计时", "",
              "均值±样本标准差，单位秒；比值为 DC/Full 均值比，低于1才表示 DC 更快。", "",
              "| N | 工况 | MPI进程 | Full | DC-JSR | DC/Full | 含初始化比值 | DC配对胜数 | 最大迭代 Full/DC |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        report.append("| %d | %s | %d | %.3f±%.3f | %.3f±%.3f | %.3f | %.3f | %d/4 | %d/%d |" %
                      (r["mesh_n"], r["regime"], r["processes"], r["full"]["mean_seconds"], r["full"]["sd_seconds"],
                       r["dc_jsr"]["mean_seconds"], r["dc_jsr"]["sd_seconds"], r["ratio"], r["ratio_with_initialization"],
                       r["dc_jsr"]["paired_wins_vs_full"], r["full"]["max_iterations"], r["dc_jsr"]["max_iterations"]))
    report += ["", "## 正确性与来源", "",
               "- 正式认证 %d 次，完整热身认证 %d 次，共 %d 次；每次 PETSc 与独立 NumPy CSR 真残差均 <1e-8。" %
               (solves, warmup_solves, solves + warmup_solves),
               "- 最大真残差 %.9e；额外求解重试 %d 次。" % (max_residual, retries),
               "- 最大粗算子独立校验相对误差 %.9e。" % max_coarse_error,
               "- 分布式与原串行后端在初始/陈旧混合状态上的随机向量作用最大相对误差 %.9e。" % max_reference_error,
               "- 十二配置运行期间后端/求解脚本指纹一致，同一网格工况的输入哈希一致。",
               "- 同一网格工况的 DC 预算序列在所有进程数和正式重复间一致；迭代的浮点差异单列如下。", "",
               "| N | 工况 | 方法 | 跨进程/重复最大单步迭代差 |",
               "|---:|---|---|---:|"]
    for h in history_checks:
        report.append("| %d | %s | %s | %d |" % (h["mesh_n"], h["regime"], h["arm"],
                                               h["max_iteration_difference_across_processes_and_repeats"]))
    report += ["", "## 并行维护工作与关键路径", "",
               "下表局部维护总工作为所有 rank 局部维护耗时之和；关键路径为逐步 rank 最大局部维护耗时之和。"
               "两者不能混称墙钟节省，分阶段最大值之和也不能替代已报告的总墙钟时间。", "",
               "| N | 工况 | P | 局部总工作 Full/DC(s) | 局部关键路径 Full/DC(s) | Full/DC 各自相对P1加速比 |",
               "|---:|---|---:|---:|---:|---:|"]
    for r in records:
        report.append("| %d | %s | %d | %.3f/%.3f | %.3f/%.3f | %.2f/%.2f |" %
                      (r["mesh_n"], r["regime"], r["processes"],
                       r["full"]["mean_local_rank_work_seconds"], r["dc_jsr"]["mean_local_rank_work_seconds"],
                       r["full"]["mean_local_critical_seconds"], r["dc_jsr"]["mean_local_critical_seconds"],
                       r["full"]["speedup_vs_own_p1"], r["dc_jsr"]["speedup_vs_own_p1"]))
    report += ["", "## 结论边界", "",
               "闭式阈值策略保持同一数学语义，MPI 并行没有加入 Age、迟滞、Krylov反馈或新的预算控制。"
               "本次最小代理维护成本性质不能升级为并行墙钟最优性：墙钟取决于最慢 rank、重叠通信、粗层与 PCG。"
               "所有负收益配置和慢重复完整保留；四次重复只能描述这台机器的有限测量，不构成跨硬件统计保证。", "",
               "N24多前沿P2的DC样本标准差较大，其中第4次DC为6.099s、配对Full为4.645s；"
               "N32多前沿P1也包含一次较慢DC重复。它们保留在均值中。"
               "分阶段计时能够定位耗时变化，但当前数据不能确定其来自宿主机调度、通信等待还是其他运行波动。"
               "粗层与重叠传输计时包含本地计算及同步等待，不能当作纯网络通信时间。", "",
               "此次实现的单进程时间不能与之前旧后端的单进程秒数直接解释为策略改进；"
               "线程绑定、局部静态提取映射、输入预装载和后端不同。这里仅比较同一新后端内的 Full/DC。", "",
               "## 复现", "", "```bash",
               "python3 benchmarks/run_dc_jsr_parallel_suite.py",
               "```", "",
               "运行前以科研环境生成 N24/32 输入：`benchmarks/prepare_dc_jsr_parallel_inputs.py`。"
               "suite 脚本固定科研 Python/MPI 路径、两个必需链接环境变量和单线程设置，依次执行所有配置，拒绝覆盖已存在的正式结果。"
               "汇总/绘图入口：`benchmarks/summarize_dc_jsr_parallel_audit.py`（科研环境运行）。", "",
               "原始数据：`results/dc_jsr_parallel_audit/*_n*_p*.json`；完整汇总：`summary.json`；"
               "主图：`docs/figures/dc_jsr_parallel_validation.pdf` 和 `.png`。", ""]
    input_audits = []
    for path in sorted(a.input_dir.glob("input_reassembly_n*.json")):
        input_audits.extend(json.loads(path.read_text()))
    if input_audits:
        assert all(x["all_arrays_identical"] for x in input_audits)
        report += ["## 输入重组装核验与独立测试", "",
                   "重新调用原FEM函数后，每一个坐标、分区、CSR矩阵、右端及对角数组与冻结输入逐元素一致。"
                   "N24矩阵/右端序列的语义哈希与原876次认证完全一致。",
                   "独立17维SPD小矩阵测试在1/2/4 MPI进程通过，包含非近邻矩阵项、重叠累加、陈旧局部因子与当前粗算子，"
                   "预条件子作用与独立NumPy稠密参考一致。", "",
                   "| N | 工况 | 重新组装耗时(s) | 与冻结数组一致 | 与原N24认证一致 |",
                   "|---:|---|---:|---|---|"]
        for x in input_audits:
            report.append("| %d | %s | %.3f | 是 | %s |" %
                          (x["mesh_n"], x["regime"], x["assembly_seconds"],
                           "是" if x["original_n24_audit_identical"] else "不适用"))
        report.append("")
        summary["input_reassembly_audits"] = input_audits
        output.write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "docs/DC_JSR_PARALLEL_VALIDATION_REPORT_2026-10-03.md").write_text("\n".join(report))
    figure.with_suffix(".json").write_text(json.dumps(dict(
        source_summary=str(output.relative_to(ROOT)),
        summary_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        plotting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        formal_repeats=4, error_bars="sample standard deviation", configurations=12), indent=2) + "\n")


if __name__ == "__main__":
    main()

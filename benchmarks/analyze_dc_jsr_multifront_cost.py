#!/usr/bin/env python3
"""Explain the frozen MPI audit using existing timings; run no new solves."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/dc_jsr_parallel_audit"
STAGES = ("monitor", "local", "coarse", "solve_and_certification")


def mean(values):
    return statistics.mean(values)


def analyze(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    p = data["mpi_processes"]
    arms = {}
    for arm in ("full", "dc_jsr"):
        runs = [r for r in data["runs"] if r["arm"] == arm]
        assert len(runs) == 4
        records = []
        for run in runs:
            # Attribute every phase to the SAME rank that finishes this step last.
            # Independently maximized phase durations must never be added.
            critical = {k: 0.0 for k in STAGES}
            local_work = local_makespan = 0.0
            for step in run["steps"]:
                rank = max(step["rank_times"], key=lambda r: r["total"])
                assert abs(sum(rank[k] for k in STAGES) - rank["total"]) < 1e-10
                for key in STAGES:
                    critical[key] += rank[key]
                local_work += sum(r["local"] for r in step["rank_times"])
                local_makespan += max(r["local"] for r in step["rank_times"])
            assert abs(sum(critical.values()) - run["total_seconds"]) < 1e-9
            records.append(dict(repeat=run["repeat"], total=run["total_seconds"],
                                iterations=sum(s["iterations"] for s in run["steps"]),
                                critical_rank_stages=critical, local_rank_work=local_work,
                                local_makespan=local_makespan))
        steps = runs[0]["steps"]
        assert all([s["selected"] for s in r["steps"]] ==
                   [s["selected"] for s in steps] for r in runs)
        counts = [[sum(cid % p == rank for cid in s["selected"])
                   for rank in range(p)] for s in steps]
        arms[arm] = dict(mean_total=mean(r["total"] for r in records),
                         sd_total=statistics.stdev(r["total"] for r in records),
                         mean_iterations=mean(r["iterations"] for r in records),
                         mean_critical_rank_stages={k: mean(r["critical_rank_stages"][k]
                                                          for r in records) for k in STAGES},
                         mean_local_rank_work=mean(r["local_rank_work"] for r in records),
                         mean_local_makespan=mean(r["local_makespan"] for r in records),
                         refresh_counts=[s["k"] for s in steps], rank_refresh_counts=counts,
                         full_makespan_count_steps=sum(max(c) == 8 // p for c in counts),
                         equal_cost_count_makespan_ratio=sum(max(c) for c in counts) /
                                                        (len(steps) * (8 / p)),
                         refresh_fraction=sum(s["k"] for s in steps) / (8 * len(steps)),
                         repetitions=records)
    full, dc = arms["full"], arms["dc_jsr"]
    delta = {k: dc["mean_critical_rank_stages"][k] - full["mean_critical_rank_stages"][k]
             for k in STAGES}
    ratio = dc["mean_total"] / full["mean_total"]
    return dict(input_file=str(path.relative_to(ROOT)), input_sha256=hashlib.sha256(raw).hexdigest(),
                mesh_n=data["input_manifest"]["mesh_n"], mpi_processes=p, arms=arms,
                dc_over_full=ratio, walltime_saving_fraction=1 - ratio,
                additive_phase_delta_dc_minus_full=delta,
                maintenance_saving_seconds=-sum(delta[k] for k in ("monitor", "local", "coarse")),
                solve_extra_seconds=delta["solve_and_certification"],
                local_work_saving_fraction=1 - dc["mean_local_rank_work"] / full["mean_local_rank_work"],
                local_makespan_saving_fraction=1 - dc["mean_local_makespan"] / full["mean_local_makespan"])


def main():
    configs = [analyze(AUDIT / ("multi_front_churn_n%d_p%d.json" % (n, p)))
               for n in (24, 32) for p in (1, 2, 4)]
    gentle = [analyze(AUDIT / ("gentle_single_front_n%d_p4.json" % n)) for n in (24, 32)]
    output = dict(configurations=configs, gentle_p4_comparison=gentle,
                  timing_note="Critical-rank phase attribution sums to total. Coarse timing includes "
                              "collective waiting after unequal local work. Local makespan is a separate "
                              "diagnostic and must not be added to independently maximized coarse times.",
                  count_model_note="Equal-cost task counts explain scheduling exposure, not predict "
                                   "real timings; factor costs, contention and synchronization differ.")
    (AUDIT / "multifront_cost_analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = ["# DC-JSR 多前沿 MPI 耗时归因（2026-10-03）", "",
             "仅分析既有四轮冻结测试；不运行新求解，不调整 ε=0.08，不修改 PDE 或维护规则。",
             "统计口径为同进程数 Full 对照、逐步墙钟时间四轮平均；不含输入组装、分发、初始化和额外审计。", "",
             "## 总耗时与维护暴露", "",
             "| N | MPI 进程 | Full / s | DC / s | DC/Full | 刷新比例 | 最忙进程任务数比例* |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for c in configs:
        f, d = c["arms"]["full"], c["arms"]["dc_jsr"]
        lines.append("| %d | %d | %.4f | %.4f | %.4f | %.2f%% | %.2f%% |" %
                     (c["mesh_n"], c["mpi_processes"], f["mean_total"], d["mean_total"],
                      c["dc_over_full"], 100*d["refresh_fraction"],
                      100*d["equal_cost_count_makespan_ratio"]))
    lines += ["", "*假设因子成本相等，仅作为任务分布解释；不作为真实时间预测。", "",
              "## 可复核原因", "",
              "1. 多前沿确实需要刷新多数块：N24 为 98/128，N32 为 90/128；完全刷新分别有 4、3 步。",
              "2. 重构工作量与并行等待时间不同。因子 i 固定归属 i % P；四进程各拥有两块。",
              "   N24 第 1 步刷新 [0,1,2,3,7]，进程负载为 [1,1,1,2]，Full 为 [2,2,2,2]。",
              "   N24 全部 16 步、N32 的 14/16 步仍有进程刷新两块。温和轨迹两种网格每步最多只需一块。",
              "3. 复用只省局部因子重构；每次 PCG 预条件应用仍使用全部八个局部因子，并执行重叠通信和粗层修正。",
              "4. 非零代理容差不保证 Full 相同迭代数。四进程 N24 总迭代 802→812，N32 为 850→886。", "",
              "## 四进程实测分解", "",
              "逐步选择最终完成最晚的同一进程，再将其各阶段时间相加；这样可以严格还原总墙钟时间。",
              "粗层阶段包含 Allreduce 等待，不能把它全部解释成粗矩阵计算变贵。", "",
              "| N | Full 求解阶段占总耗时 | 局部重构总进程工时减少 | 局部重构最长进程时间减少 | 维护阶段净省 / s | 求解阶段增加 / s | 总净省 / s |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for c in configs:
        if c["mpi_processes"] != 4:
            continue
        f = c["arms"]["full"]
        lines.append("| %d | %.2f%% | %.2f%% | %.2f%% | %.6f | %.6f | %.6f |" %
                     (c["mesh_n"], 100*f["mean_critical_rank_stages"]["solve_and_certification"] / f["mean_total"],
                      100*c["local_work_saving_fraction"], 100*c["local_makespan_saving_fraction"],
                      c["maintenance_saving_seconds"], c["solve_extra_seconds"],
                      c["maintenance_saving_seconds"]-c["solve_extra_seconds"]))
    lines += ["", "这些是测量归因，不是谱理论或因果干预。阶段变化还包含系统波动。", "",
              "## 波动与结论边界", "",
              "N24 两进程四轮 DC/Full 为 0.9233、0.9829、1.0123、1.3130；均值 1.0555，不能宣称加速。",
              "最后一轮 DC 为 6.0993 s，Full 为 4.6455 s。迭代数及选择集合跨重复一致，说明该轮额外时间",
              "并非维护决策或迭代次数变化所致。现有计时无法进一步区分宿主调度、资源竞争与 MPI 等待；保留全部四轮。",
              "N24 四进程均值 1.0026、2/4 获胜，应称持平附近，不能将 0.26% 当成确定退化。",
              "N32 四进程均值 0.9728、2/4 获胜，同样不足以声称稳定逐轮加速。", "",
              "可支持的结论：当前固定八子域映射中，多前沿刷新密集、最忙进程工作量下降小、求解阶段占比高，",
              "共同压缩了选择性重构的墙钟收益。最小刷新基数并不等于最小并行墙钟时间。",
              "优化执行调度有空间，但块成本、数据驻留、迁移通信都必须计入；不能保证仅换映射即可加速。", "",
              "复现：运行 benchmarks/analyze_dc_jsr_multifront_cost.py；原始输入文件及 SHA256 见",
              "results/dc_jsr_parallel_audit/multifront_cost_analysis.json。"]
    (ROOT / "docs/DC_JSR_MULTIFRONT_PARALLEL_ANALYSIS_2026-10-03.md").write_text("\n".join(lines) + "\n")
    print("Analyzed six multi-front configurations and two gentle comparisons; no new solves.")


if __name__ == "__main__":
    main()

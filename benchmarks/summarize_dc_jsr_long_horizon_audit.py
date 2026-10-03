#!/usr/bin/env python3
"""Certify, summarize and plot the fixed larger/longer audit and its animation data."""
import hashlib
import json
import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUDIT = ROOT / "results/dc_jsr_long_horizon_audit"
INPUTS = ROOT / "results/dc_jsr_long_horizon_inputs"
FIGURES = ROOT / "docs/figures"
INLINE = Path("/mnt/c/Users/Administrator/.codex/visualizations/2026/10/03/01a10068-e73c-7681-af06-1587fa095a18/dc-jsr-long-horizon.html")
CLIENT_INLINE = r"C:\Users\Administrator\.codex\visualizations\2026\10\03\01a10068-e73c-7681-af06-1587fa095a18\dc-jsr-long-horizon.html"


def average_runs(runs, key, cumulative=False):
    values = np.array([[s[key] for s in r["steps"]] for r in runs], dtype=float)
    if cumulative:
        values = values.cumsum(axis=1)
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-sizes", nargs="+", type=int, choices=[40,48], default=[40,48])
    args = parser.parse_args()
    configurations, full_results = [], []
    certificates, warmup_certificates, retries = 0, 0, 0
    max_residual, max_coarse_error = 0., 0.
    common_source = None
    for n in args.mesh_sizes:
        path = AUDIT / ("multi_front_churn_n%d_t64_p4.json" % n)
        d = json.loads(path.read_text())
        assert d["status"] == "complete" and d["mpi_processes"] == 4
        assert d["epsilon"] == .08 and d["repeats"] == 4
        assert d["input_manifest"]["steps"] == 64
        assert not d["input_manifest"]["state_reset_between_cycles"]
        if common_source is None:
            common_source = d["source_sha256"]
        assert common_source == d["source_sha256"]
        arms = {a: sorted([r for r in d["runs"] if r["arm"] == a], key=lambda r: r["repeat"])
                for a in ("full", "dc_jsr")}
        assert all(len(runs) == 4 for runs in arms.values()) and len(d["warmup_runs"]) == 2
        for run in d["runs"] + d["warmup_runs"]:
            assert len(run["steps"]) == 64
            previous = [0] * 8
            for s in run["steps"]:
                assert s["independent_true_relative_residual"] < 1e-8
                assert s["petsc_true_relative_residual"] < 1e-8
                assert s["coarse_relative_error"] < 1e-10
                assert s["residual_proxy"] <= .08 + 1e-14
                expected = [s["step"] if i in s["selected"] else previous[i] for i in range(8)]
                assert expected == s["factor_versions"]
                previous = s["factor_versions"]
                if run["arm"] == "dc_jsr":
                    assert s["selected"] == [i for i, value in enumerate(s["pre_proxy"]) if value > .08]
                else:
                    assert s["selected"] == list(range(8))
                max_residual = max(max_residual, s["independent_true_relative_residual"], s["petsc_true_relative_residual"])
                max_coarse_error = max(max_coarse_error, s["coarse_relative_error"])
                retries += len(s["attempts"]) - 1
                if run["repeat"] < 0:
                    warmup_certificates += 1
                else:
                    certificates += 1
        dc, full = arms["dc_jsr"], arms["full"]
        reference_steps = dc[0]["steps"]
        assert all([s["selected"] for s in r["steps"]] == [s["selected"] for s in reference_steps] for r in dc)
        di, _ = average_runs(dc, "iterations")
        fi, _ = average_runs(full, "iterations")
        dt, ds = average_runs(dc, "total_seconds", True)
        ft, fs = average_runs(full, "total_seconds", True)
        geometry = json.loads((INPUTS / ("multi_front_churn_n%d_t64.geometry.json" % n)).read_text())
        assert all(abs(s["area"] - 1) < 1e-10 and len(s["triangles"]) == 2*n*n for s in geometry["slices"])
        config = dict(mesh_n=n, n_dofs=d["input_manifest"]["n_dofs"], geometry=geometry,
                      sources=geometry["sources"], coefficient=geometry["coefficient"],
                      full_mean=float(ft[-1]), dc_mean=float(dt[-1]), ratio=float(dt[-1]/ft[-1]),
                      paired_wins=sum(a["total_seconds"] < b["total_seconds"] for a,b in zip(dc, full)),
                      steps=[dict(step=s["step"], k=s["k"], selected=s["selected"],
                                  pre_proxy=s["pre_proxy"], residual_proxy=s["residual_proxy"],
                                  rank_counts=[sum(cid % 4 == r for cid in s["selected"]) for r in range(4)],
                                  dc_iterations=float(di[j]), full_iterations=float(fi[j]),
                                  dc_cumulative=float(dt[j]), dc_cumulative_sd=float(ds[j]),
                                  full_cumulative=float(ft[j]), full_cumulative_sd=float(fs[j]))
                             for j,s in enumerate(reference_steps)])
        configurations.append(config)
        summary = dict(mesh_n=n, n_dofs=config["n_dofs"], mpi_processes=4, steps=64,
                       full_mean=config["full_mean"], dc_mean=config["dc_mean"], ratio=config["ratio"],
                       full_sd=float(fs[-1]), dc_sd=float(ds[-1]), paired_wins=config["paired_wins"],
                       paired_ratios=[a["total_seconds"] / b["total_seconds"] for a,b in zip(dc,full)],
                       full_max_iterations=max(r["max_iterations"] for r in full),
                       dc_max_iterations=max(r["max_iterations"] for r in dc),
                       full_mean_total_iterations=float(fi.sum()), dc_mean_total_iterations=float(di.sum()),
                       refresh_fraction=sum(s["k"] for s in reference_steps)/(64*8),
                       k_by_cycle=[[s["k"] for s in reference_steps[i*16:(i+1)*16]] for i in range(4)],
                       input_sha256=d["input_manifest"]["sha256"],
                       result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        full_results.append(summary)
    assert certificates == 512*len(args.mesh_sizes) and warmup_certificates == 128*len(args.mesh_sizes)
    summary = dict(configurations=full_results, formal_certificates=certificates,
                   warmup_certificates=warmup_certificates, max_true_residual=max_residual,
                   max_coarse_error=max_coarse_error, retries=retries, source_sha256=common_source,
                   scope="Periodic 16-step prescribed source cycle repeated four times; no state reset; "
                         "single-node 4-rank replay, fixed 8 subdomains; not 64 distinct source states.")
    (AUDIT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Actual topology and trajectory from frozen inputs; all curves use measured four-run means.
    visual = dict(configurations=configurations, summary=summary)
    (AUDIT / "animation_data.json").write_text(json.dumps(visual, separators=(",", ":")) + "\n")
    template = (ROOT / "docs/visualizations/dc-jsr-long-horizon.html.in").read_text()
    library = (ROOT / "docs/visualizations/d3.v7.9.0.min.js").read_text()
    fragment = template.replace("__DC_DATA__", json.dumps(visual, separators=(",", ":"))).replace("__D3_LIBRARY__", library)
    assert "__DC_DATA__" not in fragment and "__D3_LIBRARY__" not in fragment and len(fragment.encode()) < 1_000_000
    INLINE.parent.mkdir(parents=True, exist_ok=True)
    INLINE.write_text(fragment)
    (ROOT / "docs/visualizations/dc-jsr-long-horizon-fragment.html").write_text(fragment)
    # A fragment requires host styles; this HTML must also work when opened
    # directly via file:///H:/... in a regular browser.
    subprocess.run(["python3", "/mnt/c/Users/Administrator/.codex/plugins/cache/openai-bundled/visualize/1.0.46/skills/visualize/scripts/render.py",
                    str(INLINE), str(ROOT / "docs/visualizations/dc-jsr-long-horizon.html"), "--force"], check=True)
    # The desktop reader runs on Windows. /mnt/c paths exist only in WSL;
    # its path authorization must receive the native absolute Windows path.
    (ROOT / "docs/visualizations/dc-jsr-long-horizon.reference.json").write_text(
        json.dumps(dict(path=CLIENT_INLINE,mode="wide"),indent=2)+"\n")
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype":42, "ps.fonttype":42, "font.size":9})
    fig, axes = plt.subplots(4,len(configurations),figsize=(6*len(configurations),9), sharex=True, squeeze=False)
    for col,c in enumerate(configurations):
        x = np.arange(1,65)
        axes[0,col].set_title("N=%d | %s DOFs | 4 MPI ranks" % (c["mesh_n"], format(c["n_dofs"],",")))
        steps = c["steps"]
        axes[0,col].step(x,[s["k"] for s in steps],where="mid",color="#087f8c",label="DC-JSR")
        axes[0,col].axhline(8,color="#9d5722",linestyle="--",label="Full")
        axes[0,col].set_ylabel("Refreshed factors K")
        axes[0,col].set_ylim(0,8.7)
        axes[1,col].plot(x,[s["residual_proxy"] for s in steps],color="#087f8c")
        axes[1,col].axhline(.08,color="#9d5722",linestyle="--",label="epsilon = 0.08")
        axes[1,col].set_ylabel("Post-refresh proxy max V")
        for arm,color,style in [("dc","#087f8c","-"),("full","#9d5722","--")]:
            axes[2,col].plot(x,[s[arm+"_iterations"] for s in steps],color=color,linestyle=style,label="DC-JSR" if arm=="dc" else "Full")
            mu=np.array([s[arm+"_cumulative"] for s in steps]);sd=np.array([s[arm+"_cumulative_sd"] for s in steps])
            axes[3,col].plot(x,mu,color=color,linestyle=style,label="DC-JSR" if arm=="dc" else "Full")
            axes[3,col].fill_between(x,mu-sd,mu+sd,color=color,alpha=.14)
        axes[2,col].set_ylabel("PCG iterations")
        axes[3,col].set_ylabel("Cumulative wall time (s)")
        axes[3,col].set_xlabel("Time step (dt = 0.05)")
        for row in range(4):
            axes[row,col].set_xlim(.5,64.5)
            axes[row,col].grid(alpha=.18)
            for boundary in (16.5,32.5,48.5):axes[row,col].axvline(boundary,color=".6",linewidth=.8)
            for start in (8.5,24.5,40.5,56.5):axes[row,col].axvspan(start,start+4,color=".6",alpha=.08)
        for row in (0,1,2):axes[row,col].legend(loc="best",frameon=False)
        axes[3,col].legend(loc="upper left",frameon=False)
        axes[3,col].text(.98,.05,"DC/Full = %.3f; wins %d/4" % (c["ratio"],c["paired_wins"]),ha="right",transform=axes[3,col].transAxes)
    fig.suptitle("Larger-mesh, 64-step periodic multi-front audit: four-run means",y=.995)
    fig.tight_layout(rect=(0,0,1,.975))
    for extension in ("pdf","png"):fig.savefig(FIGURES / ("dc_jsr_long_horizon_trajectory."+extension),dpi=180)
    plt.close(fig)
    lines=["# DC-JSR 大网格长轨迹验证（2026-10-03）", "",
           "固定 ε=0.08、8 子域、4 MPI 进程、一进程一线程。64 步为四次完整 16 步周期，周期之间不重置状态。",
           "主耗时不含输入装配、加载、分发、初始化和额外审计；全部四轮结果保留。", "",
           "| 网格 | 自由度 | Full / s | DC-JSR / s | DC/Full | 胜率 | Full/DC 最大迭代 | 刷新比例 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in full_results:
        lines.append("| N=%d | %s | %.3f ± %.3f | %.3f ± %.3f | %.4f | %d/4 | %d / %d | %.2f%% |" %
                     (c["mesh_n"],format(c["n_dofs"],","),c["full_mean"],c["full_sd"],c["dc_mean"],c["dc_sd"],
                      c["ratio"],c["paired_wins"],c["full_max_iterations"],c["dc_max_iterations"],100*c["refresh_fraction"]))
    lines.append("")
    for c in full_results:
        lines.append("N=%d 四轮配对比值：%s。" % (c["mesh_n"],", ".join("%.4f" % x for x in c["paired_ratios"])))
    lines += ["", "%d 次正式求解、%d 次预热求解全部通过双重真残差认证；最大真残差 %.6e，粗算子相对误差 %.6e，重试 %d 次。" % (certificates,warmup_certificates,max_residual,max_coarse_error,retries),
              "", "更细网格上的固定容差并不保证 Full 相同迭代表现。预算、省去重构的数量与并行墙钟收益必须分开报告。",
              "四次周期载荷可检查持续复用与重访，不能当成四个独立物理工况，也不能据此宣称任意长轨迹稳定。", "",
              "静态图：docs/figures/dc_jsr_long_horizon_trajectory.pdf 与 .png。",
              "动画：docs/visualizations/dc-jsr-long-horizon.html；真实网格来自输入 geometry.json，日志及数据哈希在 results/dc_jsr_long_horizon_audit。",
              "动画颜色表示 κ 节点值可视化，非求解温度；显示的是实际 tetrahedral FEM 截面。刷新标记为核心区，局部因子包含额外一层重叠。", "",
              "复现：准备两个网格输入 → run_dc_jsr_long_horizon_suite.py → summarize_dc_jsr_long_horizon_audit.py。"]
    (ROOT / "docs/DC_JSR_LONG_HORIZON_VALIDATION_REPORT_2026-10-03.md").write_text("\n".join(lines)+"\n")
    print("Certified %d formal + %d warmup solves; max true residual %.3e" % (certificates,warmup_certificates,max_residual))
    for c in full_results:print("N=%d DC/Full=%.4f, wins=%d/4" % (c["mesh_n"],c["ratio"],c["paired_wins"]))


if __name__ == "__main__":
    main()

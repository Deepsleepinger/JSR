#!/usr/bin/env python3
"""Launch frozen MPI configurations sequentially without resource contention."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCI = Path("/mnt/h/CodexLinux/stokes-r3/envs/stokes-fenics-2020-abi6-r3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-sizes", nargs="+", type=int, default=[24, 32])
    parser.add_argument("--processes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--regimes", nargs="+", default=["gentle_single_front", "multi_front_churn"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/dc_jsr_parallel_audit")
    a = parser.parse_args()
    available = len(os.sched_getaffinity(0))
    if max(a.processes) > available:
        raise RuntimeError("Requested more MPI ranks than available logical CPUs")
    env = os.environ.copy()
    env.update(PKG_CONFIG_PATH=str(SCI / "lib/pkgconfig"),
               LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    configuration = 0
    for mesh_n in a.mesh_sizes:
        for regime in a.regimes:
            # Rotate process-count order across workload/size configurations.
            offset = configuration % len(a.processes)
            order = a.processes[offset:] + a.processes[:offset]
            for ranks in order:
                output = a.output_dir / ("%s_n%d_p%d.json" % (regime, mesh_n, ranks))
                if output.exists():
                    raise RuntimeError("Refusing to overwrite a formal run: %s" % output)
                command = [str(SCI / "bin/mpiexec"), "-n", str(ranks), str(SCI / "bin/python"),
                           str(ROOT / "benchmarks/run_dc_jsr_parallel_audit.py"),
                           "--input", str(ROOT / "results/dc_jsr_parallel_inputs" / ("%s_n%d.npz" % (regime, mesh_n))),
                           "--output", str(output), "--repeats", str(a.repeats), "--epsilon", "0.08"]
                print("Launching N=%d %s P=%d" % (mesh_n, regime, ranks), flush=True)
                subprocess.run(command, cwd=str(ROOT), env=env, check=True)
                result = json.loads(output.read_text())
                if result["status"] != "complete":
                    raise RuntimeError("Incomplete configuration")
                print("Completed", output.name, "DC/Full=%.4f" %
                      result["summary"]["dc_jsr"]["ratio_of_mean_times_vs_full"], flush=True)
            configuration += 1


if __name__ == "__main__":
    main()

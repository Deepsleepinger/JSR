#!/usr/bin/env python3
"""Launch the two preregistered larger/longer MPI configurations sequentially."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCI = Path("/mnt/h/CodexLinux/stokes-r3/envs/stokes-fenics-2020-abi6-r3")


def main():
    if len(os.sched_getaffinity(0)) < 4:
        raise RuntimeError("Four available CPUs required")
    env = os.environ.copy()
    env.update(PKG_CONFIG_PATH=str(SCI / "lib/pkgconfig"),
               LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    output_dir = ROOT / "results/dc_jsr_long_horizon_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    for n in (40, 48):
        inputs = ROOT / "results/dc_jsr_long_horizon_inputs" / ("multi_front_churn_n%d_t64.npz" % n)
        output = output_dir / ("multi_front_churn_n%d_t64_p4.json" % n)
        if output.exists():
            raise RuntimeError("Refusing to overwrite formal results: %s" % output)
        print("Launching N=%d, T=64, P=4, repeats=4, epsilon=0.08" % n, flush=True)
        command = [str(SCI / "bin/mpiexec"), "-n", "4", str(SCI / "bin/python"),
                   str(ROOT / "benchmarks/run_dc_jsr_parallel_audit.py"), "--input", str(inputs),
                   "--output", str(output), "--repeats", "4", "--epsilon", "0.08"]
        try:
            subprocess.run(command, cwd=str(ROOT), env=env, check=True)
        except Exception as error:
            (output_dir / ("failure_n%d.json" % n)).write_text(json.dumps(dict(
                mesh_n=n, error=str(error), command=command), indent=2) + "\n")
            raise
        data = json.loads(output.read_text())
        assert data["status"] == "complete"
        print("Completed N=%d DC/Full=%.4f" %
              (n, data["summary"]["dc_jsr"]["ratio_of_mean_times_vs_full"]), flush=True)


if __name__ == "__main__":
    main()

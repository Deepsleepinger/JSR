#!/usr/bin/env python3
"""Finish artifacts when the already-running finite audit finishes."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCI=Path('/mnt/h/CodexLinux/stokes-r3/envs/stokes-fenics-2020-abi6-r3')
AUDIT=ROOT/'results/dc_jsr_long_horizon_audit'


def main():
    deadline=time.monotonic()+7200
    status=AUDIT/'artifact_generation.json'
    status.write_text(json.dumps(dict(status='waiting_for_existing_audit'))+'\n')
    while time.monotonic()<deadline:
        if list(AUDIT.glob('failure_n*.json')):
            status.write_text(json.dumps(dict(status='blocked_by_audit_failure'))+'\n')
            return
        paths=[AUDIT/('multi_front_churn_n%d_t64_p4.json'%n) for n in (40,48)]
        if all(p.exists() and json.loads(p.read_text()).get('status')=='complete' for p in paths):
            break
        time.sleep(10)
    else:
        status.write_text(json.dumps(dict(status='audit_wait_timeout'))+'\n')
        return
    env=os.environ.copy()
    env.update(PKG_CONFIG_PATH=str(SCI/'lib/pkgconfig'),LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    try:
        subprocess.run([str(SCI/'bin/python'),str(ROOT/'benchmarks/summarize_dc_jsr_long_horizon_audit.py')],cwd=str(ROOT),env=env,check=True)
        subprocess.run([str(SCI/'bin/python'),str(ROOT/'benchmarks/render_dc_jsr_long_horizon_animation.py'),'--mesh-n','48'],cwd=str(ROOT),env=env,check=True)
        status.write_text(json.dumps(dict(status='complete',meshes=[40,48],steps=64,animation='docs/figures/dc_jsr_long_horizon_mesh_n48.gif'),indent=2)+'\n')
        print('Both larger-mesh audits and artifacts complete.',flush=True)
    except Exception as error:
        status.write_text(json.dumps(dict(status='artifact_generation_failed',error=str(error)),indent=2)+'\n')
        raise


if __name__=='__main__':
    main()

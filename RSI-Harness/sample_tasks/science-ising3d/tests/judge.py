"""The science-ising3d Judge: this task's routes and deadlines.

tests/test.sh runs this file in every Judge; ``python3 /tests/judge.py
--recompute /workspace/candidate`` previews the routes in Work.
science_judge.py, the generic public science Judge library (copied unchanged),
does everything else: it freezes the candidate, recomputes each route
isolated, audits the round's trajectory with the run's own Agent and reviews
every rubric with the Judge model, both through rsi-agent, then writes the
reward.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # -I drops the script directory
import science_judge as sj  # noqa: E402

# Finite pressure and derivative probes of model.py:pressure(K); no verdict.
PRESSURE_PROBE = """import importlib.util,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).parent))
spec=importlib.util.spec_from_file_location('candidate_model',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=[]
for K in [0,0.01,0.1,0.2,0.3,1,8]:
 try:
  p=float(m.pressure(K))
  row={'K':K,'p':p,'p_minus_3K':p-3*K}
  if K>0:
   h=1e-4;lo=float(m.pressure(K-h));hi=float(m.pressure(K+h))
   row.update(dp=(hi-lo)/(2*h),d2p=(hi-2*p+lo)/(h*h))
  rows.append(row)
 except Exception as e: rows.append({'K':K,'error':type(e).__name__+': '+str(e)})
print(json.dumps({'rows':rows,'p0_reference':math.log(2),'scope':'finite public probes only; no rubric verdict or asymptotic proof'},allow_nan=False))
"""  # noqa: E501

CONFIG = sj.JudgeConfig(
    workdir="/workspace",
    routes=(
        sj.Route(
            name="candidate_reproduction",
            # Without -I, so probe.py can import the candidate's own modules.
            argv=("python3", "-B", "{root}/probe.py"),
            timeout_sec=60,
            requires=("probe.py",),
            missing_reason="the candidate has no optional probe.py",
        ),
        sj.Route(
            name="public_pressure_probes",
            argv=("python3", "-I", "-B", "-c", PRESSURE_PROBE, "{root}/model.py"),
            timeout_sec=60,
            requires=("model.py",),
            missing_reason="the candidate has no optional model.py",
        ),
    ),
    audit_seconds=20 * 60,
    review_seconds=30 * 60,
    review_workers=4,
    thread_env=("OMP_NUM_THREADS",),
    candidate_order=("theory.md",),
)

if __name__ == "__main__":
    sys.exit(sj.main(CONFIG))

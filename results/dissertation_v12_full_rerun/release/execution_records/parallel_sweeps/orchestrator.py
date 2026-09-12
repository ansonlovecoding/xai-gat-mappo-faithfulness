from pathlib import Path
import concurrent.futures
from datetime import datetime, timezone
import json, os, shlex, signal, subprocess, sys, time
ROOT=Path('/Users/hongweilin/Documents/workspace/documents/Personal/DMU/Final Dissertation/dissertation-v12-full-rerun')
PYTHON=ROOT/'.venv/bin/python'
PARENT=55342
os.chdir(ROOT)
env=os.environ.copy()
env.update(SUMO_HOME=str(ROOT/'runs/runtime/sumo-1.20.0'),
           PATH=str(ROOT/'runs/runtime/sumo-1.20.0/bin')+os.pathsep+str(PYTHON.parent)+os.pathsep+env['PATH'],
           DISPATCH_MARL_DEVICE='cpu',DISPATCH_MARL_FORCE_TRACI='1',
           OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
raw=subprocess.check_output([str(PYTHON),'scripts/run_dissertation_experiments.py','--config','configs/experiments/dissertation_v12_full_rerun.toml','--stage','sweep','--dry-run'],text=True,env=env)
commands=[shlex.split(line[2:]) for line in raw.splitlines() if line.startswith('$ ') and 'scripts/sweep_severity.py' in line]
commands=[cmd for cmd in commands if '/B2_gat/seed_42/' not in ' '.join(cmd)]
assert len(commands)==9,len(commands)
proc=subprocess.check_output(['ps','-p',str(PARENT),'-o','command='],text=True)
assert 'run_dissertation_experiments.py' in proc and '--stage sweep' in proc
out=ROOT/'runs/dissertation_v12_sumo120/execution_records/parallel_sweeps'
out.mkdir(exist_ok=True)
record=dict(started_utc=datetime.now(timezone.utc).isoformat(),status='running',workers=3,
            additional_original_checkpoint='B2_gat/seed_42',paused_parent=PARENT,
            note='Original checkpoint finishes normally. Three additional workers own disjoint checkpoint roots. Parent resumes and verifies/skips completed cells.',commands=commands,results=[])
(out/'status.json').write_text(json.dumps(record,indent=2)+'\n')
os.kill(PARENT,signal.SIGSTOP)
def run(cmd):
    checkpoint=Path(cmd[2]);label=checkpoint.parent.parent.name+'_'+checkpoint.parent.name
    start=time.monotonic()
    with (out/f'{label}.log').open('w') as log:
        code=subprocess.call(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    return dict(label=label,returncode=code,wall_seconds=time.monotonic()-start,
                finished_utc=datetime.now(timezone.utc).isoformat())
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,c) for c in commands]):
            result=future.result();record['results'].append(result)
            (out/'status.json').write_text(json.dumps(record,indent=2)+'\n')
            print(result,flush=True)
    record['status']='completed' if all(r['returncode']==0 for r in record['results']) else 'failed'
finally:
    os.kill(PARENT,signal.SIGCONT)
    record['finished_utc']=datetime.now(timezone.utc).isoformat()
    (out/'status.json').write_text(json.dumps(record,indent=2)+'\n')

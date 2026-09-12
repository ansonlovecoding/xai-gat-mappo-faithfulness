from pathlib import Path
import concurrent.futures
from datetime import datetime, timezone
import json,os,shlex,signal,subprocess,time
ROOT=Path('/Users/hongweilin/Documents/workspace/documents/Personal/DMU/Final Dissertation/dissertation-v12-full-rerun')
PYTHON=ROOT/'.venv/bin/python';PARENT=48997
os.chdir(ROOT);env=os.environ.copy()
env.update(SUMO_HOME=str(ROOT/'runs/runtime/sumo-1.20.0'),PATH=str(ROOT/'runs/runtime/sumo-1.20.0/bin')+os.pathsep+str(PYTHON.parent)+os.pathsep+env['PATH'],DISPATCH_MARL_DEVICE='cpu',DISPATCH_MARL_FORCE_TRACI='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
tasks=[]
for stage,script in [('robustness','scripts/sweep_severity.py'),('controls','scripts/run_faithfulness_controls.py')]:
 raw=subprocess.check_output([str(PYTHON),'scripts/run_dissertation_experiments.py','--config','configs/experiments/dissertation_v12_full_rerun.toml','--stage',stage,'--dry-run'],text=True,env=env)
 for line in raw.splitlines():
  if line.startswith('$ ') and script in line:
   tasks.append((stage,shlex.split(line[2:])))
assert len(tasks)==14,len(tasks)
proc=subprocess.check_output(['ps','-p',str(PARENT),'-o','command='],text=True)
assert 'scripts/run_full_rerun.py' in proc
out=ROOT/'runs/dissertation_v12_sumo120/execution_records/parallel_support';out.mkdir(exist_ok=True)
record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),workers=3,paused_parent=PARENT,
 note='Independent robustness and ranking/query-row control cells use the frozen protocol. Main wrapper is paused until all tasks finish. Its sweep-stage elapsed time spans this overlapping explanation-evaluation batch; later robustness/control stage times cover reuse/aggregation, not original simulation cost.',tasks=tasks,results=[])
(out/'status.json').write_text(json.dumps(record,indent=2)+'\n');os.kill(PARENT,signal.SIGSTOP)
def run(task):
 stage,cmd=task;out_path=Path(cmd[cmd.index('--out')+1]);label=stage+'_'+out_path.parent.name+'_'+out_path.name
 start=time.monotonic()
 with (out/f'{label}.log').open('w') as f:code=subprocess.call(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 return dict(stage=stage,label=label,returncode=code,wall_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat())
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for future in concurrent.futures.as_completed([pool.submit(run,t) for t in tasks]):
   result=future.result();record['results'].append(result);(out/'status.json').write_text(json.dumps(record,indent=2)+'\n');print(result,flush=True)
 record['status']='completed' if all(x['returncode']==0 for x in record['results']) else 'failed'
finally:
 os.kill(PARENT,signal.SIGCONT);record['finished_utc']=datetime.now(timezone.utc).isoformat();(out/'status.json').write_text(json.dumps(record,indent=2)+'\n')

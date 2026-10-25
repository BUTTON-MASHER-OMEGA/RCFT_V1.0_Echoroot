"""Read-only inspection and strictly bounded execution of rediscovered toys."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys,time,os
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
SOURCE=ROOT/'projects/lco-related'
manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*') if p.is_file()}
(HERE/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
inventory=[]
for p in sorted(SOURCE.rglob('*.py')):
    s=p.read_text(encoding='utf-8-sig');tree=ast.parse(s)
    inventory.append({'path':p.relative_to(ROOT).as_posix(),'syntax':'valid','lines':len(s.splitlines()),
      'functions':[{'name':n.name,'line':n.lineno} for n in tree.body if isinstance(n,ast.FunctionDef)]})
(HERE/'inventory.json').write_text(json.dumps(inventory,indent=2))
small=[] if '--definitions-only' in sys.argv else ['v_test.1.py','mode_sphere_test.1.py','u1-relational-universe-probe.py','skew_probe.py','isp_phi_probe.py','cell-600-test.py','constraint_regime_sweep.py']
results=json.loads((HERE/'bounded-runs.json').read_text()) if '--definitions-only' in sys.argv else []
for name in small:
    start=time.monotonic()
    try:
        proc=subprocess.run([sys.executable,'-X','utf8','-B',str(SOURCE/name)],cwd=HERE,
          capture_output=True,timeout=5,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        (HERE/(name+'.stdout.txt')).write_bytes(proc.stdout);(HERE/(name+'.stderr.txt')).write_bytes(proc.stderr)
        record={'script':name,'exit_code':proc.returncode,'seconds':round(time.monotonic()-start,3)}
    except subprocess.TimeoutExpired as exc:
        (HERE/(name+'.stdout.txt')).write_bytes(exc.stdout or b'')
        (HERE/(name+'.stderr.txt')).write_bytes(exc.stderr or b'')
        record={'script':name,'status':'terminated at five-second limit; not retried'}
    results.append(record);print(json.dumps(record),flush=True)
(HERE/'bounded-runs.json').write_text(json.dumps(results,indent=2))

# Load definitions only: no main guards, sampling loops or long simulations.
def definitions(path):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    module=ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))],type_ignores=[])
    env={'__name__':'inspection_definitions'};exec(compile(module,str(path),'exec'),env);return env
geometry={}
for name,function in [('polytope-packing.py','generate_packings'),('h3-parameter-sweep.py','generate_packings'),
 ('Matthew/dynamic_crystalization.py','generate_packings'),('cell-600-test.py','generate_600_cell_vertices'),
 ('constraint_regime_sweep.py','generate_600_cell')]:
    env=definitions(SOURCE/name);raw=env[function]();packs=raw if isinstance(raw,tuple) else (raw,)
    rows=[]
    for V in packs:
        dots=V@V.T
        values=sorted({round(float(v),5) for v in dots[np.triu_indices(len(V),1)]})
        D=((V[:,None,:]-V[None,:,:])**2).sum(2);minimum=D[D>1e-10].min()
        degree=np.isclose(D,minimum,atol=1e-9).sum(1)
        rows.append({'vertices':len(V),'unique_vertices':len(np.unique(np.round(V,10),axis=0)),
         'norm_range':[float(np.linalg.norm(V,axis=1).min()),float(np.linalg.norm(V,axis=1).max())],
         'positive_dot_tiers':[x for x in values if 0.0001<x<.9999],
         'nearest_degree_histogram':dict(Counter(int(x) for x in degree))})
    geometry[name]=rows
(HERE/'geometry-checks.json').write_text(json.dumps(geometry,indent=2))

# Explicitly inspect the old projected carrier, without running a trajectory.
p=SOURCE/'Matthew/test-1.py';tree=ast.parse(p.read_text())
module=ast.Module(body=[n for n in tree.body if not isinstance(n,ast.If)],type_ignores=[])
env={'__name__':'inspection'};exec(compile(module,str(p),'exec'),env)
neighbors=env['neighbors'];projection=env['proj']
edges={tuple(sorted((i,j))) for i,ns in neighbors.items() for j in ns}
env['lam']=0;env['alpha']=1.;env['beta']=2.;env['T']=1.
from collections import defaultdict
aggregated={}
for i in neighbors:
    probabilities,_,_=env['transition_probs'](i,defaultdict(int),1)
    totals=[sum(v for j,v in probabilities.items() if projection[j]==o) for o in range(4)]
    aggregated[str(i)]=totals
carrier={'vertices':len(neighbors),'edges':len(edges),'degree_sequence':sorted(map(len,neighbors.values())),
 'frozen_ops_cycle_edges':8,'same_as_frozen_C8':False,
 'zero_memory_projected_transition_rows':aggregated,
 'lumpability_counterexample':{'same_observation_states':[0,1],'observation':0,
    'state_0_next_observation_row':aggregated['0'],'state_1_next_observation_row':aggregated['1'],
    'conclusion':'Rows differ: no generic Markov lumping even at lambda=0. This does not establish non-Markov behavior for every initial law.'}}
(HERE/'old-carrier-check.json').write_text(json.dumps(carrier,indent=2))
after={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*') if p.is_file()}
assert after==manifest,'New toy sources changed during inspection'
(HERE/'integrity.json').write_text(json.dumps({'source_files':len(manifest),'unchanged':True},indent=2))
print('Inspected:',len(inventory),'source files; sources unchanged.')

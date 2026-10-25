"""Run bounded existing computations on explicitly extracted OPS code copies."""
from pathlib import Path
import ast, json, re, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
COPIES = OUT/'extracted-ops'
COPIES.mkdir(exist_ok=True)
records = []
for p in sorted((ROOT/'projects/Operational_Phase_Space').glob('SIM*.py')):
    lines = p.read_text(encoding='utf-8-sig').splitlines(keepends=True)
    cut = next((i for i,l in enumerate(lines) if re.match(r'^\s*(~{5,}|RESULTS:)',l)),len(lines))
    if p.name == 'SIM14.5.py':
        cut = 208  # Source line 209 starts the pasted-results separator.
    code = ''.join(lines[:cut])
    prefix = ''
    if p.name in {'SIM14.4.py','SIM14.7.py','SIM14.8.py','SIM15.1.py','SIM15.2.py'}:
        prefix = '"""\n'
    code = prefix+code
    row = {'source':p.relative_to(ROOT).as_posix(), 'kept_source_lines':[1,cut],
           'prepended_opening_docstring':bool(prefix)}
    try:
        ast.parse(code)
        row['extracted_syntax']='valid'
        (COPIES/p.name).write_text(code,encoding='utf-8')
    except SyntaxError as e:
        row.update(extracted_syntax='invalid',error_line=e.lineno,error=e.msg)
    records.append(row)
(OUT/'extraction-manifest.json').write_text(json.dumps(records,indent=2))

selected = ['SIM14.3.py','SIM14.5.py','SIM14.8.py','SIM15.6.py']+[f'SIM16.{i}.py' for i in range(5)]
if len(sys.argv)>1:
    selected=sys.argv[1:]
results=json.loads((OUT/'reproduction-results.json').read_text()) if len(sys.argv)>1 else []
results=[r for r in results if r['script'] not in selected]
for name in selected:
    if not (COPIES/name).exists():
        results.append({'script':name,'status':'extraction failed'});continue
    start=time.monotonic()
    try:
        proc=subprocess.run([sys.executable,'-B',str(COPIES/name)],cwd=OUT,
                            capture_output=True,timeout=60,env={**__import__('os').environ,'PYTHONIOENCODING':'utf-8'})
        (OUT/(name+'.stdout.txt')).write_bytes(proc.stdout)
        (OUT/(name+'.stderr.txt')).write_bytes(proc.stderr)
        row={'script':name,'exit_code':proc.returncode,'seconds':round(time.monotonic()-start,2)}
    except subprocess.TimeoutExpired as exc:
        (OUT/(name+'.stdout.txt')).write_bytes(exc.stdout or b'')
        (OUT/(name+'.stderr.txt')).write_bytes(exc.stderr or b'')
        row={'script':name,'status':'timeout at 60 seconds'}
    results.append(row)
    (OUT/'reproduction-results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(row),flush=True)

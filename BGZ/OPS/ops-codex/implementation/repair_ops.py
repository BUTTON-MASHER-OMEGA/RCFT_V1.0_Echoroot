"""Separate archived OPS results from executable code, retaining full originals."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys,time,os,re

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
ops=ROOT/'projects/Operational_Phase_Space'
extractions=json.loads((ROOT/'projects/ops-codex/research/extraction-manifest.json').read_text())
archive=[]
for row in extractions:
    rel=row['source'];original=HERE/'before'/rel
    assert original.is_file()
    raw=original.read_text(encoding='utf-8-sig');lines=raw.splitlines(keepends=True)
    keep=row['kept_source_lines'][1]
    code=('"""\n' if row['prepended_opening_docstring'] else '')+''.join(lines[:keep])
    ast.parse(code)
    target=ROOT/rel;target.write_text(code,encoding='utf-8')
    historical=ops/'recorded_output'/(target.stem+'.txt');historical.parent.mkdir(exist_ok=True)
    historical.write_text(''.join(lines[keep:]),encoding='utf-8')
    archive.append({'source':rel,'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
      'full_original_archive':original.relative_to(ROOT).as_posix(),
      'kept_source_lines':[1,keep],'prepended_opening_docstring':row['prepended_opening_docstring'],
      'clean_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
      'historical_result':historical.relative_to(ROOT).as_posix(),
      'historical_result_sha256':hashlib.sha256(historical.read_bytes()).hexdigest()})
(ops/'ARCHIVE_MANIFEST.json').write_text(json.dumps(archive,indent=2)+'\n')
# Only previously measured small jobs. No new/unknown census is dispatched.
fast=['SIM14.3.py','SIM14.5.py','SIM14.8.py','SIM15.6.py','SIM16.0.py','SIM16.1.py','SIM16.3.py','SIM16.4.py']
results=[]
def packets(text):
    found=[]
    for line in text.splitlines():
        if line.startswith('TRUTH_PACKET_JSON='):
            try:found.append(json.loads(line.split('=',1)[1]))
            except json.JSONDecodeError:pass
    return found
def normalized(text):
    return [' '.join(l.split()) for l in text.splitlines() if l.strip() and not re.fullmatch(r'[\s=~\-]+',l) and not re.fullmatch(r'\s*\d+\s*\|\s*',l)]
for name in fast:
    started=time.monotonic()
    try:
        p=subprocess.run([sys.executable,'-X','utf8','-B',str(ops/name)],cwd=HERE,capture_output=True,timeout=10,
                         env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        (HERE/(name+'.stdout.txt')).write_bytes(p.stdout);(HERE/(name+'.stderr.txt')).write_bytes(p.stderr)
        fresh=p.stdout.decode('utf-8');old=(ops/'recorded_output'/(Path(name).stem+'.txt')).read_text(encoding='utf-8')
        a,b=packets(fresh),packets(old)
        freshlines=normalized(fresh);oldlines=normalized(old)
        row={'script':name,'exit_code':p.returncode,'seconds':round(time.monotonic()-started,2),
          'historical_packet_present':bool(b),'reproduced_packet_present':bool(a),
          'packet_equal':a==b if a and b else None,
          'normalized_historical_lines':len(oldlines),'normalized_reproduced_lines':len(freshlines),
          'historical_lines_present_in_reproduction':sum(l in set(freshlines) for l in oldlines),
          'historical_lines_not_repeated':[l for l in oldlines if l not in set(freshlines)]}
    except subprocess.TimeoutExpired:
        row={'script':name,'status':'stopped at 10-second limit; not retried'}
    results.append(row);print(json.dumps({k:v for k,v in row.items() if not isinstance(v,list)}),flush=True)
for row in archive:
    name=Path(row['source']).name
    if name not in fast:results.append({'script':name,'status':'not run: runtime restriction; static parse only',
       'note':'Prior onboarding evidence, if any, is separate and not represented as a new run.'})
(HERE/'ops-repair-verification.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
print('Clean scripts:',len(archive),'bounded reruns:',len(fast))

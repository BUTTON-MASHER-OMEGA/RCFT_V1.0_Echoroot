"""Read-only documentation census, source baseline and bounded Git-history capture."""
from pathlib import Path
import hashlib,json,re,subprocess,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
AREAS=['rcf','rcmt','projects/Operational_Phase_Space','projects/lisi-smg-test']

def git(repo,*args):
    folder=ROOT/repo
    p=subprocess.run(['git','-c',f'safe.directory={folder.as_posix()}','-C',str(folder),*args],capture_output=True)
    if p.returncode: raise RuntimeError(p.stderr.decode('utf-8',errors='replace'))
    return p.stdout.decode('utf-8-sig')

def manifest():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for area in AREAS for p in sorted((ROOT/area).rglob('*'))
            if p.is_file() and '.git' not in p.parts}

if '--check' in sys.argv:
    before=json.loads((HERE/'source-baseline.json').read_text())
    after=manifest()
    delta={k:'changed/added/deleted' for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
    (HERE/'source-integrity.json').write_text(json.dumps(delta,indent=2))
    print('Source changes:',delta)
    sys.exit(bool(delta))

(HERE/'source-baseline.json').write_text(json.dumps(manifest(),indent=2))
census=[]
fields=['owner','type','scope','authority','upstream dependencies','status','purpose','anti-backflow']
for area in AREAS:
    for p in sorted((ROOT/area).rglob('*.md')):
        s=p.read_text(encoding='utf-8-sig')
        # Ignore fenced examples for the mechanical header-presence screen.
        plain=re.sub(r'(?ms)^```.*?^```[^\n]*','',s)
        present={f:bool(re.search(r'^- '+re.escape(f)+r':\s*\S',plain,re.M)) for f in fields}
        census.append({'path':p.relative_to(ROOT).as_posix(),'lines':len(s.splitlines()),
                       'canonical_fields_present':present,'headings':[{'line':i,'text':l}
                       for i,l in enumerate(s.splitlines(),1) if re.match(r'^#{1,3} ',l)]})
(HERE/'document-census.json').write_text(json.dumps(census,indent=2,ensure_ascii=False),encoding='utf-8')
captures={
 'rcf-earliest-geometry.txt':('rcf',['show','b7a3406:core/relational-constraints/geometry.md']),
 'rcf-two-lane-optionalization.diff':('rcf',['show','8c7bd51','--','core/relational-constraints/geometry.md']),
 'rcf-geometry-history.txt':('rcf',['log','--all','--reverse','--format=%H %aI %s','--','core/relational-constraints/geometry.md']),
 'rcmt-earliest-invariants.txt':('rcmt',['show','e9c51a3:local/invariants.md']),
 'rcmt-earliest-symmetry.txt':('rcmt',['show','e9c51a3:local/constraints/symmetry_chain.md']),
 'rcmt-candidate-revision.diff':('rcmt',['show','7a17282','--','local/invariants.md','local/constraints/symmetry_chain.md']),
 'rcmt-compactness-revision.diff':('rcmt',['show','b7eecd0','--','local/invariants.md']),
 'rcmt-history.txt':('rcmt',['log','--all','--reverse','--format=%H %aI %s']),
 'rcf-history.txt':('rcf',['log','--all','--reverse','--format=%H %aI %s']),
}
for name,(repo,args) in captures.items():
    (HERE/name).write_text(git(repo,*args),encoding='utf-8')
meta={r:{'head':git(r,'rev-parse','HEAD').strip(),'status':git(r,'status','--porcelain')} for r in ['rcf','rcmt']}
(HERE/'git-provenance.json').write_text(json.dumps(meta,indent=2))
summary={a:{'markdown_files':sum(x['path'].startswith(a+'/') for x in census),
    'complete_field_presence':sum(x['path'].startswith(a+'/') and all(x['canonical_fields_present'].values()) for x in census)} for a in AREAS}
(HERE/'census-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

"""Snapshot approved maintenance inputs before any source mutation."""
from pathlib import Path
import hashlib,json,shutil,subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
areas=['rcf','rcmt','projects/Operational_Phase_Space','projects/lisi-smg-test']
manifest={}
if (HERE/'baseline.json').exists():raise SystemExit('Baseline already exists; refusing to replace it.')
for area in areas:
    for p in sorted((ROOT/area).rglob('*')):
        if not p.is_file() or '.git' in p.parts:continue
        rel=p.relative_to(ROOT)
        manifest[rel.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.suffix in {'.md','.py','.txt','.json'} and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:
            q=HERE/'before'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
(HERE/'baseline.json').write_text(json.dumps(manifest,indent=2))
attachment=Path('C:/Users/blaur/.codex/attachments/954f9999-2397-4b10-a419-f3563eecd63e/Pasted text.txt')
shutil.copyfile(attachment,HERE/'approved-owner-register.txt')
headers=[]
for p in (ROOT/'rcmt').rglob('*.md'):
    s=p.read_text(encoding='utf-8-sig')
    headers.append({'path':p.relative_to(ROOT).as_posix(),'opening':s.splitlines()[:16],
                    'declarations':[l for l in s.splitlines() if any(k in l.lower() for k in ['owner:','authority:','depends_on:','upstream dependencies:','status:'])]})
(HERE/'legacy-header-inventory.json').write_text(json.dumps(headers,indent=2,ensure_ascii=False),encoding='utf-8')
print('Source files baselined:',len(manifest),'RCMT Markdown:',len(headers))

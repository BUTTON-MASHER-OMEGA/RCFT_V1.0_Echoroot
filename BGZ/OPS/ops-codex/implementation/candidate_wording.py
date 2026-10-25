from pathlib import Path
import json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
p=ROOT/'rcmt/local/constraints/mode_space.md'
s=p.read_text(encoding='utf-8');old=s
s=s.replace('The current provisional implementation uses:','The active research candidate under examination is:')
s=s.replace('is a survivor implementation and may be replaced by any','is a research candidate, not an established realization, and may be replaced by any')
assert s!=old
p.write_text(s,encoding='utf-8')
events=json.loads((HERE/'final-text-edits.json').read_text())
events.append({'path':p.relative_to(ROOT).as_posix(),'decisions':['D2'],'reason':'Replace residual implementation-selected wording with active research candidate status'})
(HERE/'final-text-edits.json').write_text(json.dumps(events,indent=2),encoding='utf-8')

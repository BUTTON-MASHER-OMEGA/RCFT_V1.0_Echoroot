"""RCP structural/dependency validation plus explicit semantic review gates.

Does not claim to prove arbitrary prose true. Unresolved metadata is a failure
of compression readiness, not a reason to invent a replacement source.
"""
from pathlib import Path
import re,json
import networkx as nx

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
fields=['owner','type','scope','authority','upstream dependencies','status','purpose','anti-backflow']
types={'Invariant','Contract','Observation','Dependency','Permission','Prohibition','Interpretation','Status'}
rows=[];graph=nx.DiGraph()
def plain(s):
    output=[];inside=False
    for l in s.splitlines():
        if re.match(r'^\s*(```|~~~)',l):inside=not inside;output.append('');continue
        output.append('' if inside else l)
    return '\n'.join(output)
for p in sorted((ROOT/'rcmt').rglob('*.md')):
    s=p.read_text(encoding='utf-8-sig');s0=plain(s);errors=[]
    if not re.match(r'^# [^\n]+\n\n## Document Contract\n\n',s):errors.append('opening structure')
    if len(re.findall(r'^# ',s0,re.M))!=1:errors.append('single H1')
    if len(re.findall(r'^## Document Contract\s*$',s0,re.M))!=1:errors.append('single contract')
    m=re.search(r'^## Document Contract\n\n((?:- [^\n]+\n){8})',s,re.M)
    values={}
    if not m:errors.append('eight contiguous canonical fields')
    else:
        items=[l[2:].split(':',1) for l in m[1].splitlines()]
        if [k for k,v in items]!=fields:errors.append('field order/names')
        values={k:v.strip() for k,v in items}
        if any(not v for v in values.values()):errors.append('blank field')
        if values.get('type') not in types:errors.append('statement type')
        if values.get('anti-backflow')=='none':errors.append('non-root anti-backflow missing')
        if values.get('authority') in {'active','provisional','experimental'}:errors.append('status substituted for authority')
    key=p.relative_to(ROOT).as_posix();graph.add_node(key)
    dependencies=[]
    for token in re.findall(r'`([^`]+)`',values.get('upstream dependencies','')):
        q=(p.parent/token).resolve()
        if not q.is_relative_to(ROOT):errors.append('dependency outside authorized theory boundary: '+token);continue
        if not q.is_file():errors.append('unresolved dependency: '+token)
        else:
            dest=q.relative_to(ROOT).as_posix();graph.add_edge(key,dest);dependencies.append(dest)
    if 'not compression-ready' in values.get('status',''):errors.append('declared unresolved source metadata')
    rows.append({'path':key,'structural_pass':not [e for e in errors if e!='declared unresolved source metadata'],
                 'compression_ready':not errors,'errors':errors,'dependencies':dependencies})
cycles=[sorted(s) for s in nx.strongly_connected_components(graph) if len(s)>1]
result={'documents':len(rows),'structural_pass':sum(r['structural_pass'] for r in rows),
        'compression_ready_by_header_dependency_screen':sum(r['compression_ready'] for r in rows),
        'unresolved':[r for r in rows if r['errors']],'dependency_cycles':cycles,
        'scope':'Full canonical opening, field order/nonempty/type, unique headings outside fences, resolved header paths, anti-backflow and status/authority checks; graph cycles; owner/boundary semantic review separately recorded. A mechanical pass alone is not full semantic RCP certification.'}
(HERE/'rcp-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in {'scope'}},indent=2))

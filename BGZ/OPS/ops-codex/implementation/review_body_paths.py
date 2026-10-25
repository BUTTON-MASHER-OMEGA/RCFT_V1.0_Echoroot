from pathlib import Path
import re,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
bad=[];amb=[]
for p in (ROOT/'rcmt').rglob('*.md'):
    if p.name=='rcmt_compressed.md':continue
    s=p.read_text(encoding='utf-8').split('## Migration provenance and unresolved metadata')[0]
    output=[];inside=False
    for l in s.splitlines():
        if re.match(r'^\s*(```|~~~)',l):inside=not inside;continue
        if not inside:output.append(l)
    s='\n'.join(output)
    for t in set(re.findall(r'`([^`\n]+\.md)`',s)):
        if any(x in t for x in ['*','<','>']):continue
        # An existing document-relative path is unambiguous. Fallback roots
        # detect old conceptual aliases only when relative lookup fails.
        bases=[p.parent] if (p.parent/t).is_file() else [ROOT/'rcmt',ROOT,ROOT/'rcmt/physics_interface',ROOT/'rcmt/smg_v2.1']
        hits={(b/t).resolve().relative_to(ROOT).as_posix() for b in bases if (b/t).is_file() and (b/t).resolve().is_relative_to(ROOT)}
        if not hits:bad.append({'path':p.relative_to(ROOT).as_posix(),'reference':t})
        elif len(hits)>1:amb.append({'path':p.relative_to(ROOT).as_posix(),'reference':t,'candidates':sorted(hits)})
(HERE/'body-path-review.json').write_text(json.dumps({'missing':bad,'ambiguous_aliases':amb},indent=2),encoding='utf-8')
print(json.dumps({'missing':bad,'ambiguous_alias_count':len(amb)},indent=2))

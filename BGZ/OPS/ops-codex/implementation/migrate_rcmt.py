"""Classify existing RCMT documents into canonical RCP headers.

Legacy fields are preserved below the canonical header. Missing sources and
ambiguous dependency directions are reported, never silently filled in.
"""
from pathlib import Path
import re,os,json,hashlib

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];REPO=ROOT/'rcmt'
FIELDS=['owner','type','scope','authority','upstream dependencies','status','purpose','anti-backflow']
rows=[]
def rel(p,q):return os.path.relpath(q,p.parent).replace('\\','/')
def mask_fences(s):
    lines=s.splitlines();inside=False;out=[]
    for l in lines:
        if re.match(r'^\s*(```|~~~)',l):inside=not inside;out.append('');continue
        out.append('' if inside else l)
    return '\n'.join(out)
def first_purpose(s,title):
    # Existing sentences under a Purpose heading, not a new scientific claim.
    m=re.search(r'(?im)^#{1,5}[^\n]*purpose[^\n]*\n+([^#\n][^\n]*(?:\n[^#\n][^\n]*)*)',s)
    if m:
        candidate=re.sub(r'\s+',' ',m[1]).strip(' *>`')
        candidate=re.split(r'(?<=[.!?])\s+',candidate)[0]
        if len(candidate)<400 and candidate:return candidate
    return 'Classify and preserve the existing '+title.strip('`*')+' specification and its stated scope.'
def known_upstream(p,text):
    sources=[]
    # Resolve only named owner interfaces already consumed by legacy declarations.
    names=[('upstream LCO',ROOT/'rcf/modules/lco/interface.md'),
      ('upstream LCO outputs',ROOT/'rcf/modules/lco/interface.md'),
      ('upstream LCO contract',ROOT/'rcf/modules/lco/interface.md'),
      ('RCF core constraints',ROOT/'rcf/core/S∞.RCF.Invariants.v1.1.md'),
      ('upstream projection outputs',ROOT/'rcf/modules/geometry/projection/interface.md'),
      ('upstream geometry stack',ROOT/'rcf/modules/geometry/discrete/interface.md')]
    for name,q in names:
        if name.lower() in text.lower():sources.append(q)
    return sources

for p in sorted(REPO.rglob('*.md')):
    s=p.read_text(encoding='utf-8-sig');original=s;legacy={};legacy_text=''
    m=re.search(r'(?m)^## Document Contract\s*\n',s)
    if m:
        # Contract ends at the first nonblank, non-field line.
        start=m.start();pos=m.end();end=pos
        for line in s[pos:].splitlines(keepends=True):
            if not line.strip():end+=len(line);continue
            fm=re.match(r'^- ([^:]+):\s*(.*)',line)
            if not fm:break
            legacy[fm[1]]=fm[2].strip();end+=len(line)
        legacy_text=s[start:end];s=s[:start]+s[end:]
    # An initial path-only heading was historical navigation, not a second title.
    stripped=s.lstrip('\ufeff\r\n')
    leading=[]
    while stripped.startswith('---\n'):stripped=stripped[4:].lstrip()
    lines=stripped.splitlines()
    first_h1=next((i for i,l in enumerate(lines) if re.match(r'^# ',l)),None)
    if first_h1 is not None:
        title=lines[first_h1][2:].strip()
        leading=lines[:first_h1]
        body='\n'.join(lines[first_h1+1:]).lstrip('\n')
    else:
        # Global files use a path heading followed by their actual named subject.
        heads=[(i,l) for i,l in enumerate(lines) if re.match(r'^#{2,6} ',l)]
        if len(heads)>=2 and '/' in heads[0][1]:
            i,l=heads[1];title=re.sub(r'^#+\s*','',l).strip('`');leading=lines[:i];body='\n'.join(lines[i+1:]).lstrip()
        elif heads:
            i,l=heads[0];title=re.sub(r'^#+\s*','',l).strip('`');leading=lines[:i];body='\n'.join(lines[i+1:]).lstrip()
        else:raise ValueError(f'No existing title for {p}')
    # Preserve all former section content; only normalize additional H1s.
    out=[];fenced=False
    for line in body.splitlines():
        if re.match(r'^\s*(```|~~~)',line):fenced=not fenced
        if not fenced and line.startswith('# '):line='#'+line
        out.append(line)
    body='\n'.join(out).strip()
    path=p.relative_to(REPO).as_posix();lower=path.lower();plain=mask_fences(body)
    existing_type=legacy.get('type')
    if existing_type and existing_type not in {'Invariant','Contract','Observation','Dependency','Permission','Prohibition','Interpretation','Status'}:
        raise ValueError((p,existing_type))
    if lower.startswith('smg_v2.1/'):
        owner='SMGv2.1'
        typ=existing_type or ('Status' if p.name in {'README.md','REFACTOR_STATUS.md'} else 'Interpretation' if '/claims/' in lower or '/translations/' in lower or '/dictionary/' in lower else 'Contract')
        authority='SMGv2.1-local model authority; subordinate to imported RCF and RCMT contracts; adopter obligations only'
        status='proposed; explicit extension adoption required' if p.name.startswith('24-cell-') else legacy.get('status','existing downstream model specification')
        scope=legacy.get('scope','SMGv2.1 '+title.strip('`*'))
    elif lower.startswith('physics_interface/'):
        inline=re.search(r'(?m)^\*\*Owner:\*\*\s*([^\n]+)',plain)
        owner=legacy.get('owner',inline[1].strip(' *') if inline else 'RCMT physics interface')
        typ=existing_type or ('Interpretation' if p.name in {'examples.md','interpretations.md','overview.md'} else 'Contract')
        authority=legacy.get('authority','existing module-local '+('interpretation' if typ=='Interpretation' else 'interface contract')+'; subordinate to RCMT local requirements and all consumed RCF contracts')
        status=legacy.get('status','scaffold; no local primitives declared' if 'No project‑specific local primitives are currently declared' in body else 'existing specification; consumer permissions remain explicitly scoped')
        scope=legacy.get('scope',title.strip('`*')+'; construction/effective-consumer permissions as explicitly declared, not inferred from location')
    elif lower.startswith('global/'):
        owner=legacy.get('owner','RCMT')
        typ=existing_type or 'Contract'
        authority=legacy.get('authority','RCMT global representation contract; subordinate to local invariants and RCF')
        status=legacy.get('status','existing model-local specification')
        scope=legacy.get('scope',title.strip('`*')+'; downstream global-invariant representations')
    elif lower.startswith('local/'):
        owner=legacy.get('owner','RCMT');typ=existing_type or 'Contract'
        authority=legacy.get('authority','RCMT-local '+('invariant authority, subordinate to RCF' if path=='local/invariants.md' else 'contract, subordinate to local/invariants.md and consumed RCF interfaces'))
        status=legacy.get('status','existing local specification')
        scope=legacy.get('scope',title.strip('`*')+'; RCMT-local')
    elif lower.startswith('modules/'):
        assert 'owner' in legacy,p
        owner=legacy['owner'];typ=existing_type or 'Contract';scope=legacy.get('scope',title)
        authority=legacy.get('authority','existing '+owner+' module-local contract; subordinate to consumed upstream records; provisional interpretations do not become upstream facts')
        status=legacy.get('status','existing scoped specification')
    elif path=='README.md':
        owner='RCMT';typ='Status';scope='RCMT identity, architectural posture and navigation'
        authority='project entrypoint subordinate to RCMT local invariants and RCF';status='active research'
    else:raise ValueError(f'Unclassified document {path}')
    deptext=legacy.get('upstream dependencies',legacy.get('depends_on',''))
    deps=[];unresolved=[];resolution=[]
    for token in re.findall(r'`([^`]+)`',deptext):
        candidates=[p.parent/token]
        # Legacy absolute concepts are not OS absolute paths.
        if token.startswith('theory/rcf_projects/'):
            unresolved.append({'legacy':token,'reason':'legacy compressed-tree reference not a current source contract'});continue
        if token.startswith('/'):
            candidates=[REPO/token.lstrip('/')]
        hits=[]
        for q in candidates:
            if '*' in str(q):hits.extend(q.parent.glob(q.name))
            elif q.is_file():hits.append(q)
        if not hits:
            unresolved.append({'legacy':token,'reason':'declared dependency does not resolve; source not invented'})
        for q in hits:
            q=q.resolve()
            if q==p.resolve():unresolved.append({'legacy':token,'reason':'self dependency'});continue
            if q.suffix=='.md':deps.append(q);resolution.append({'legacy':token,'resolved':q.relative_to(ROOT).as_posix()})
    deps+=known_upstream(p,deptext)
    # Existing explicit owner contracts define root relationships.
    if path=='local/invariants.md':deps.append(ROOT/'rcf/core/S∞.RCF.Invariants.v1.1.md')
    if path=='README.md':deps.append(REPO/'local/invariants.md')
    if path=='local/README.md':deps.append(ROOT/'rcf/core/S∞.RCF.Invariants.v1.1.md')
    if lower.startswith('global/') and not deptext:deps.append(REPO/'local/invariants.md')
    if lower.startswith('smg_v2.1/') and not deptext:
        if path=='smg_v2.1/smg_contract.md':
            deps += [ROOT/'rcf/modules/gcm_interface/representation_contract.md',REPO/'physics_interface/README.md']
        elif path=='smg_v2.1/bindings/gcm_bindings.md':deps.append(ROOT/'rcf/modules/gcm_interface/survivor_indexing.md')
        elif path=='smg_v2.1/bindings/physics_bindings.md':deps.append(REPO/'physics_interface/local/conversion_laws/formal_structure.md')
        else:deps.append(REPO/'smg_v2.1/smg_contract.md')
    if lower.startswith('physics_interface/') and not deptext:
        if path=='physics_interface/README.md':deps.append(REPO/'local/invariants.md')
        elif '/pipeline/' in lower:
            # Inline contracts explicitly compile against RCF/RCMT requirements.
            deps.append(REPO/'local/invariants.md')
        else:
            # Public local-layer requirements are explicitly loaded before modules.
            if path=='physics_interface/local/invariants.md':deps.append(REPO/'local/invariants.md')
            elif path=='physics_interface/local/interfaces.md':deps.append(REPO/'physics_interface/local/invariants.md')
            else:deps.append(REPO/'physics_interface/README.md')
    if path.startswith('local/symmetry_groups/patterns/') and not deps:
        deps.append(REPO/'local/symmetry_groups/README.md')
    if 'upstream direction-bearing unit contract' in deptext:
        unresolved.append({'legacy':deptext,'reason':'exact upstream direction-bearing contract not identified by original header'})
    # Preserve legacy notes without granting them canonical header semantics.
    extras={k:v for k,v in legacy.items() if k not in FIELDS and k!='depends_on'}
    if legacy.get('depends_on') or extras or leading or unresolved:
        appendix='\n\n## Migration provenance and unresolved metadata\n\n'
        appendix+='The canonical header classifies existing ownership and scope; it does not promote maturity, grant access or supply a missing implementation.\n'
        if legacy.get('depends_on'):appendix+='\n- Legacy dependency declaration: '+legacy['depends_on']+'\n'
        for k,v in extras.items():appendix+='- Legacy '+k+': '+v+'\n'
        if leading:appendix+='- Former leading navigation/title material: '+ ' / '.join(x for x in leading if x.strip())+'\n'
        for u in unresolved:appendix+='- Unresolved: '+u['legacy']+' — '+u['reason']+'.\n'
        body+=appendix
    if unresolved:status+='; dependency metadata unresolved; not compression-ready'
    deps=list(dict.fromkeys(q.resolve() for q in deps if q.resolve()!=p.resolve()))
    purpose=legacy.get('purpose',first_purpose(body,title))
    anti=legacy.get('anti-backflow',f'This {owner} document may classify its stated scope but may not redefine consumed RCF or RCMT invariants, temporal parameters, projection ownership or upstream meanings; downstream labels and hypotheses do not become upstream facts.')
    values=[owner,typ,scope,authority,', '.join('`'+rel(p,q)+'`' for q in deps) or 'none',status,purpose,anti]
    header='# '+title+'\n\n## Document Contract\n\n'+''.join('- '+k+': '+v+'\n' for k,v in zip(FIELDS,values))+'\n'
    output=header+body.strip()+'\n'
    p.write_text(output,encoding='utf-8')
    rows.append({'path':'rcmt/'+path,'canonical':dict(zip(FIELDS,values)),
      'legacy':legacy,'dependency_resolutions':resolution,'unresolved':unresolved,
      'before_sha256':hashlib.sha256(original.encode('utf-8')).hexdigest(),'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
      'review_basis':'Existing header fields first; explicit body module/owner statements and parent interface requirements for formerly headerless documents. Consumer stage not assigned from directory.'})
(HERE/'rcmt-migration.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
print('Migrated',len(rows),'documents; unresolved metadata:',sum(bool(r['unresolved']) for r in rows))

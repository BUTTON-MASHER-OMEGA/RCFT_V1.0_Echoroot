"""Rebuild subordinate contract registries from final source bytes.

Frozen Lisi concatenations are historical experiment inputs and are excluded.
"""
from pathlib import Path
import re,json,hashlib,subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(b):return hashlib.sha256(b).hexdigest()
def head(repo):
    p=ROOT/repo
    return subprocess.check_output(['git','-c',f'safe.directory={p.as_posix()}','-C',str(p),'rev-parse','HEAD'],text=True).strip()
def contract(p):
    s=p.read_text(encoding='utf-8-sig');title=s.splitlines()[0].lstrip('# ')
    m=re.search(r'(?m)^## Document Contract\s*\n((?:\n|- [^\n]+\n)+)',s)
    assert m,p
    return {'title':title,**{k:v.strip() for k,v in re.findall(r'^- ([^:]+):\s*([^\n]*)',m[1],re.M)}}
rcf_summary='''## Conservative rehydration summary

RCF is a constraint framework, not a physical validation claim. Kernel sections 1–9 state substantive invariants and section 10 states invariant closure; no module-local lane, group or geometry requirement is added to that kernel. The memory/motion two-lane geometry profile remains optional.

Time owns event order; Temporal Interface exposes sealed parameters. Density κ and bounded memory weight γ are distinct. LCO evaluates supplied admissible candidates deterministically; candidate-relative ranking does not guarantee improvement over a prior neighborhood, convergence, zero mismatch or basin stabilization. Selection, acceptance and mutation belong to separately declared policies. GCO aggregates traceable records, including failures, without acting as a controller. Optional Lyapunov/descent profiles remain opt-in.

The geometry stack consumes evaluation records under declared profiles. Triangle-specific restrictions belong to explicit downstream geometric profiles, not the neutral LCO. Historical density tables, tier-drag and crossover narratives remain repository-reported exploratory findings with unresolved run provenance and an unresolved relationship to the displayed fixed-coefficient friction formula. Their numbers are not binding current-interface constraints.

Π↑ constructs a representation of authorized discrete survivors. Π↓ diagnoses downstream compatibility, consistency or loss; it is not an inverse and cannot control upstream survivorship. Continuous, metric, field and physical interpretations retain their separate owners/stages.

GCM exposes conditional structural descriptors. Historical 12 antipodal-pair indices and three target-driven offset/regime labels yield 36 indexed configurations; precise historical offsets remain unverified. This does not independently derive generations or impose universal GCM counts. Downstream SMG adopter-specific policies do not backflow into RCF.
'''
rcmt_summary='''## Conservative rehydration summary

RCMT adopts mandatory symplectic mode-space and geometric lanes as local architectural requirements; RCF's optional profile remains optional. No necessity theorem from projection alone is asserted. Sp(8,R) is an active research candidate. SO(8)/SO(9), product notation, carrier, representation, compatibility and Lorentz realization remain provisional or unresolved; historical E8/SO(10) cascades are not active dependencies.

Compactness is profile-specific, not a universal RCMT requirement. Locality, finite-dimensional adjacency-compatible representation and the separate global-curvature exclusion remain in force. Density κ, bounded memory γ and historical sweep gamma have no established conversion. LCO evaluates candidates without a universal improvement or convergence guarantee; policy decisions and mutation remain separate.

Construction-facing interfaces consume explicitly authorized local/upstream records. Effective-output consumers receive exported projected artifacts only. These are distinct permissions, not an unrestricted union, and directory placement does not assign ambiguous individual modules. Π↓ is a downstream diagnostic, not an inverse or feedback controller. Global-invariant implementations may require declared orientation/transport data in addition to adjacency; missing algorithms are not supplied.

SMG is a downstream semantic model. Historical 12×3 indexing was target-driven toward three-generation semantics, not an independently derived prediction. Existing 36-mode/three-generation obligations remain local to adopters. Proposed 24-cell routing identities remain unadopted unless an explicit model/version selects the extension. Matter and geometry-behavior interpretations retain their provisional scopes.

This registry preserves unresolved source metadata as unresolved; it is not a certificate that every source is compression-ready or that every claim is mathematically/physically established. Exact source contracts and their scopes outrank this derived artifact.
'''
for repo,summary in [('rcf',rcf_summary),('rcmt',rcmt_summary)]:
    folder=ROOT/repo;output=folder/(repo+'_compressed.md')
    if repo=='rcf':
        files=[folder/'README.md']+[p for base in ['core','core-adjacent','modules'] for p in (folder/base).rglob('*.md')]
    else:files=[p for p in folder.rglob('*.md') if p!=output]
    files=sorted(set(files))
    manifest={p.relative_to(folder).as_posix():sha(p.read_bytes()) for p in files}
    digest=sha(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode())
    source_file=folder/(repo+'_compressed.sources.json')
    source_file.write_text(json.dumps({'base_commit':head(repo),'working_tree_source_manifest':manifest,'manifest_sha256':digest,
      'digest_method':'SHA-256 of UTF-8 JSON mapping relative path to file SHA-256; sorted keys; compact separators',
      'generator':'projects/ops-codex/implementation/build_derived.py','build_date':'2026-10-24'},indent=2)+'\n',encoding='utf-8')
    deps='`README.md`, `core/S∞.RCF.Invariants.v1.1.md`, `rcp-1.2.md`' if repo=='rcf' else '`README.md`, `local/invariants.md`, `../rcf/rcp-1.2.md`'
    status='generated; subordinate to all indexed sources' if repo=='rcf' else 'generated; unresolved sources included as records only; not compression-ready'
    title='RCF Compressed Contract Registry' if repo=='rcf' else 'RCMT Compressed Contract Registry'
    body=f'''# {title}

## Document Contract

- owner: {repo.upper()} project
- type: Status
- scope: source-versioned rehydration summary and exact document-contract registry
- authority: derived artifact subordinate to every indexed source contract
- upstream dependencies: {deps}
- status: {status}
- purpose: index source ownership, boundaries, maturity and evidence without widening their authority
- anti-backflow: this registry cannot repair, reinterpret, promote or outrank any source; unresolved metadata remains unresolved

## Build record

- Build date: 2026-10-24 (America/Phoenix).
- Base commit: `{head(repo)}`. This commit alone does not identify the changed working tree.
- Source documents: {len(files)}.
- Source manifest: `{source_file.name}`.
- Manifest SHA-256: `{digest}`.
- Rebuild method: run `projects/ops-codex/implementation/build_derived.py` with the existing Python runtime from the theory workspace.
- The manifest records final source bytes, including uncommitted changes. It excludes this derived file and its own manifest to avoid self-reference.
- Frozen Lisi comparison references and historical records outside this repository are not regenerated.
- This is a conservative summary plus lossless header registry, not a replacement for all source body mathematics. Load the exact source before relying on a substantive claim.

'''+summary+'\n## Exact source contract registry\n\nPaths and dependency strings below are interpreted in each original source document context, not relative to this generated file. Header fields are retained in full; these embedded records are not additional document headers.\n\n'
    for p in files:
        body+='### '+p.relative_to(folder).as_posix()+'\n\n```json\n'+json.dumps(contract(p),ensure_ascii=False,indent=2)+'\n```\n\n'
    output.write_text(body,encoding='utf-8')
    print(repo,'sources:',len(files),'manifest:',digest)

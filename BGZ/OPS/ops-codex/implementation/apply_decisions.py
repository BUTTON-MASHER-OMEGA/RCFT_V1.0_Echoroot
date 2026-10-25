"""Apply approved owner decisions to active text, with an explicit per-file ledger."""
from pathlib import Path
import hashlib,json,re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
events=[]
def read(p):return (ROOT/p).read_text(encoding='utf-8-sig')
def put(p,s,d,why):
    f=ROOT/p;old=f.read_bytes() if f.exists() else b''
    if old.decode('utf-8-sig')==s:return
    f.parent.mkdir(parents=True,exist_ok=True);f.write_text(s,encoding='utf-8')
    events.append(dict(path=p,decisions=d,reason=why,before=hashlib.sha256(old).hexdigest(),after=hashlib.sha256(f.read_bytes()).hexdigest()))
def repl(p,a,b,d,why):
    s=read(p)
    if a not in s:raise ValueError(f'Missing exact text in {p}: {a[:80]}')
    put(p,s.replace(a,b),d,why)
def section(p,start,end,new,d,why):
    s=read(p);a=s.index(start);b=s.index(end,a)
    put(p,s[:a]+new.rstrip()+'\n\n'+s[b:],d,why)
def note(p,title,body,d):
    s=read(p)
    if title in s:raise ValueError(f'Duplicate note {p}')
    # Insert after the existing header fields, before substantive prose.
    m=re.search(r'(?m)^## Document Contract\s*\n(?:\s*\n)*((?:- [^\n]+\n|\n)+)',s)
    pos=m.end() if m else (s.find('\n')+1)
    addition=f'\n## {title}\n\n{body.strip()}\n\n'
    put(p,s[:pos]+addition+s[pos:],d,title)

# Protocol-only correction stays a separate ledger item.
p='rcf/rcp-1.2.md'
repl(p,'- the leading header sentinel is absent;','- the document does not begin with its single H1 title followed immediately by the canonical Document Contract;', ['D10'],'Clarify opening structure; no extra sentinel or delimiter')
repl(p,'RCP-1.2 uses no leading delimiter or YAML front matter.','RCP-1.2 uses no leading delimiter, additional sentinel marker, or YAML front matter. The H1 and immediately following canonical Document Contract are the required opening structure.', ['D10'],'Clarify canonical opening')

# Current RCMT requirements; historical arrows stay historical.
p='rcmt/local/invariants.md'
note(p,'Owner clarification — 2026-10-24',
'''RCMT independently adopts its mandatory symplectic mode-space lane and geometric lane as architectural constraints. RCF's memory/motion two-lane profile remains optional. No necessity theorem deriving the RCMT requirement from projection alone is asserted.

The approved owner decisions distinguish the required symplectic structure from the active research candidate Sp(8,R). Product-group notation below names candidate realizations, not proved decompositions. Carrier, representation and compatibility remain under investigation.''',['D1','D2'])
repl(p,'Sp(8) \\times SO(9)','Sp(8,\\mathbb{R}) \\times SO(9)',['D2'],'Identify real symplectic research candidate without adopting it')
repl(p,'Sp(8) \\times SO(8)','Sp(8,\\mathbb{R}) \\times SO(8)',['D2'],'Identify real symplectic research candidate without adopting it')
section(p,'The **geometric lane** is subject to compactness constraints,','No infinite towers,',
'''Compactness is not a universal RCMT requirement. It applies only where an explicitly adopted mathematical or representation profile requires it. Noncompact candidates remain subject to finite-dimensional representation, locality, adjacency compatibility and all other RCMT constraints. The symplectic lane is not automatically required to be compact.
''',['D4'],'Remove universal compactness; preserve other constraints')
repl(p,'No infinite towers, KK modes, string excitations, or non-compact global geometric groups are admissible.','No infinite towers, KK modes or string excitations are admissible. The Local Curvature Invariant still excludes global curvature groups; removal of blanket compactness does not remove that restriction.',['D4'],'Keep finite/local curvature exclusions distinct from compactness')
repl(p,'modulated by local constraint density (\\(\\gamma\\)).','modulated by local constraint density (\\(\\kappa\\)), through an explicitly declared representation. The bounded memory weight \\(\\gamma\\) and historical sweep gamma are distinct; no mapping between them is established here.',['D5'],'Separate density and bounded memory weight')

for p in ['rcmt/local/constraints/mode_space.md','rcmt/local/constraints/symmetry_chain.md']:
    s=read(p)
    # Historical chain remains an exact historical statement.
    if p.endswith('symmetry_chain.md'):
        a=s.index('### 2.3 Implementation status');b=s.index('## 3.',a)
        active=s[:a].replace('Sp(8)','Sp(8,\\mathbb{R})')
        s=active+s[a:b]+s[b:]
    else:s=s.replace('Sp(8)','Sp(8,\\mathbb{R})')
    s=s.replace('non‑compact global groups','global curvature groups')
    s=s.replace('with compact internal symmetry at low energy','with internal symmetry constrained by its explicitly adopted profile at low energy')
    s=s.replace('together with the required compact internal symmetry, potentially','together with an explicitly declared internal-symmetry profile, potentially')
    s=s.replace('Permit the required compact internal symmetry at the appropriate downstream interface.','Declare the internal-symmetry profile at the appropriate downstream interface; compactness is required only if that profile adopts it.')
    s=s.replace('with compact internal symmetry,','with an explicitly declared internal-symmetry profile,')
    s=s.replace('and the group must support a **non‑degenerate bilinear form** suitable for projection.','and the declared action must preserve a **non-degenerate alternating bilinear form** suitable for the representation.')
    put(p,s,['D1','D2','D4'],'Propagate candidate and profile-specific compactness language')
    note(p,'Architectural status and realization limits',
'''The two-lane requirement is adopted RCMT architecture, not a theorem of the RCF kernel or of projection alone. Sp(8,R) is an active candidate, not a finalized realization. Each candidate must declare its field, state carrier, representation and alternating form. Transformations claimed symplectic satisfy S^T Ω S = Ω; this does not require every stochastic update kernel to be a symplectic matrix. Distinct lane roles do not establish a direct-product group. The existing representational-separation requirement is retained; an operational independence criterion and the realization of Lorentz-compatible projection remain open. Compactness is required only by a separately adopted profile.''',['D1','D2','D4'])

p='rcmt/local/constraints/projection.md'
s=read(p).replace('compact internal symmetry','an explicitly declared internal-symmetry profile').replace('compact internal\nstructure','an explicitly declared internal\nsymmetry profile').replace('- non‑compact global symmetries,','- non-local information channels,')
s=s.replace('\\Pi : Sp(2n) \\;\\longrightarrow\\; \\mathcal{G}','\\Pi_{\\mathrm{RCMT}} : \\mathcal{M} \\;\\longrightarrow\\; \\mathcal{G}')
s=s.replace('where \\(\\mathcal{G}\\) is the geometric lane','where \\(\\mathcal{M}\\) is the declared mode-state carrier (not the symmetry group itself), and \\(\\mathcal{G}\\) is the geometric lane')
put(p,s,['D2','D4','D7'],'Distinguish schematic state projection from group homomorphism')
note(p,'Projection scope',
'''This contract describes RCMT mode-to-geometric representation, distinct from the upstream RCF discrete-to-continuous handoff. The displayed map is a schema, not a constructed projection. No inverse, conventional compact-group breaking chain or feedback into upstream survivor selection is supplied. Compactness is profile-specific; locality, finite representation and the global-curvature exclusion remain in force.''',['D2','D4','D7'])

p='rcmt/local/constraints/geometry.md'
s=read(p).replace('- compact internal symmetry,','- an explicitly declared internal-symmetry profile,')
a=s.index('## 3. Compact internal symmetry');b=s.index('## 4.',a)
s=s[:a]+'''## 3. Internal-symmetry profiles

Internal symmetry must admit a finite-dimensional, adjacency-compatible local representation. Compactness is not a universal constraint: it applies only when an explicitly adopted profile requires it. Noncompact candidates must still satisfy locality, finite representation, the global-curvature exclusion and every other applicable RCMT invariant. This does not validate any particular candidate group.

---

'''+s[b:]
put(p,s,['D4'],'Replace blanket internal compactness with declared profile')

p='rcmt/global/constraints.md'
s=read(p).replace('An implementation is inadmissible if it cannot be evaluated from adjacency structure alone.','An implementation must declare adjacency together with any orientation or transport records needed to evaluate its invariant. No missing transport algorithm or topology-only encoding is assumed.')
s=s.replace('Local operators such as LCO or GCO may simplify, prune, or regularize a neighborhood, but they must not be able to erase a genuine global invariant through purely local steps.','Declared local update policies may simplify, prune or regularize a neighborhood only under their stated invariant-preserving move class. The LCO evaluates supplied candidates without mutating the neighborhood; GCO aggregates records and does not prune or control local updates.')
a=s.index('##### 6. Finite and Compact Representation Constraint');b=s.index('##### 7.',a)
s=s[:a]+'''##### 6. Finite Representation and Profile-Specific Compactness

Representations must be finite-dimensional, adjacency-compatible and consistent with locality and the RCMT finite-structure constraints. Compact transport images are optional representation profiles unless separately adopted; compactness is not a universal restriction on either lane. No infinite towers, unbounded mode proliferation or string-like excitation ladders are introduced. The separate prohibition on global curvature groups remains in force.

---

'''+s[b:]
put(p,s,['D4','D7','D8'],'Scope compactness, transport inputs and local evaluation actors')

# Explicit stages, without assigning unspecified individual modules to a new API.
p='rcmt/physics_interface/README.md'
note(p,'Consumer responsibilities',
'''Construction-facing interfaces and adapters may consume explicitly authorized local/upstream records through their declared boundaries to construct projected representations. Effective-output physics consumers may consume exported projected artifacts only; they may not independently inspect raw substrate adjacency, modes or coherence internals. The pipeline Physics Interface Contract governs this latter consumer responsibility. These permissions are not a union available to every module. A module must explicitly declare the responsibility under which it consumes a record; directory placement alone grants no permission. Modules without a documented stage assignment remain subject to this unresolved assignment and receive no new access through this clarification.''',['D3'])
repl(p,'Physics modules may access only:','Construction-facing interfaces may access only explicitly authorized inputs:',['D3'],'Limit local input permissions to construction-facing responsibility')
p='rcmt/physics_interface/pipeline/physics_interface_contract.md'
s=read(p).replace('Physics modules','Effective-output physics consumers').replace('physics modules','effective-output physics consumers')
put(p,s,['D3'],'Scope effective-artifact-only firewall to effective consumers')
note(p,'Consumer-stage scope',
'''This contract governs effective-output consumers, not the construction-facing adapters that produce exported representations from explicitly authorized local records. It grants no raw-substrate access. The two consumer responsibilities remain separate; no individual module stage is inferred from its location.''',['D3'])
for p in ['rcmt/physics_interface/local/interfaces.md','rcmt/physics_interface/local/adapters/local_to_physics_adapter.md','rcmt/physics_interface/local/adapters/local_to_geometry_adapter.md']:
    note(p,'Construction-facing import scope',
'''The local-record permissions in this document apply to construction-facing interface/adapter responsibilities. They do not authorize an effective-output physics consumer to bypass exported artifacts or access raw substrate records. Existing declared import pathways remain unchanged; ambiguous module responsibilities require explicit assignment.''',['D3'])

p='rcmt/physics_interface/pipeline/overview.md'
put(p,'''# RCMT Pipeline Overview

## Scope

This active overview distinguishes construction-facing interfaces from effective-output physics consumers. The previous exploratory overview is preserved in the maintenance baseline and Git history; its non-local projection, guaranteed-coherence and historical symmetry-chain statements do not define the active architecture.

## Construction and evaluation

Authorized local records feed construction-facing interfaces through declared adapters. The LCO deterministically scores supplied admissible candidates; ranking identifies the best among those supplied, not necessarily an improvement over the previous neighborhood. Selection, acceptance, stochastic realization and state mutation belong to separately declared update policies. LCO evaluation guarantees neither monotonic neighborhood descent, convergence, zero mismatch nor basin stabilization. GCO aggregates traceable outcomes, including failures, without mutating or selecting updates.

Upstream event order already exists. Physical time records enter through time_bridge; metric records enter through emergent_metric_geometry. A projection summary cannot bypass these named boundaries.

## Projection and diagnostic boundary

RCF Π↑ exposes representations of authorized discrete survivors. Π↓ is a downstream compatibility, consistency or loss diagnostic, not an inverse reconstruction or controller of upstream survivorship. RCMT mode-to-geometric projection is a separate model-local representation schema. It must not introduce non-local information transfer.

RCMT adopts a mandatory symplectic mode-space lane and geometric lane. Sp(8,R), with SO(8)/SO(9) geometric candidates where documented, remains under investigation. A group product, carrier, representation, compatibility map or Lorentz realization is not established by naming these candidates. No historical E8/SO(10) cascade or conventional compact-to-Lorentz breaking chain is an active dependency. Compactness is profile-specific, not universal.

## Effective-output consumption

Effective-output physics consumers see explicitly exported artifacts only and may not independently inspect raw substrate adjacency, modes or coherence records. Construction-facing permissions and effective-consumer permissions are distinct; no module receives their unrestricted union. Module responsibilities not explicitly assigned remain unresolved.

## Evidence status

Historical sweep gamma is distinct from current density κ and bounded memory weight γ; no conversion is established. Historical 12 antipodal-pair indices across three offset/regime labels produce 36 indexed configurations. The three-regime choice is historically target-driven toward three-generation semantics; the precise offset interpretation remains to be verified. Neither that count nor successful finite OPS/Lisi calculations independently derives physical generations, selects a dynamics or validates the theory.
''',['D1','D2','D3','D4','D5','D7','D8','D9'],'Replace stale active overview; preserve full prior source in baseline')
p='rcmt/physics_interface/pipeline/projection_contract.md'
note(p,'Construction-facing representation scope',
'''This is a construction-facing representation contract, not the RCF projection module's ownership boundary. Effective fields, metric and physical-time artifacts are exposed only after their separately authorized construction stages; this contract does not create them at the upstream handoff. Raw candidate records do not permit rehabilitating rejected upstream survivors. Π↓ diagnoses downstream compatibility/loss and never controls the upstream sieve. No non-local information channel or inverse reconstruction is authorized.''',['D3','D7'])
repl(p,'It is the **only** module allowed to produce geometric‑lane artifacts, field‑like behavior, or continuous‑looking structure.','It exposes declared representations through the construction-facing boundary; ownership of their time, metric and field records remains with the explicitly declared producer stages.',['D3','D7'],'Remove overbroad sole-producer claim')

# RCF geometry profile remains optional; no independent-necessity claim.
p='rcf/core/relational-constraints/geometry.md'
repl(p,'This dual‑lane architecture is the minimal structure capable of supporting:','This optional profile adopts these two functional roles to support the following desiderata; no universal minimality or necessity proof is supplied here:',['D1'],'Qualify minimality without demoting RCMT requirement')
repl(p,'- two independent geometric lanes (memory and motion)','- functionally distinct memory and motion roles, permitting shared representational structure within this optional profile',['D1'],'Align summary with shared-representation permission')

p='rcf/modules/geometry/projection/dual_projection.md'
section(p,'### **Π↓ — Back Projection**','## **3. structural consequences',
'''### **Π↓ — Downstream Compatibility Diagnostic**

Π↓ evaluates declared downstream compatibility, consistency and representational loss using an authorized survivor record and its Π↑ representation. Its output is a diagnostic record or compatibility predicate, not a reconstructed substrate graph.

It does not select upstream survivors, alter their state, generate a missing carrier, implement an inverse, or feed rejection decisions into the upstream sieve. A failed diagnostic identifies incompatibility in the declared downstream representation/profile.

## **2. Dual compatibility (diagnostic form)**

A representation is compatible when its explicitly declared downstream compatibility predicate holds for the supplied survivor/representation pair. The former schematic equality G = Π↓(Π↑(G)) is retained only as historical notation in the maintenance baseline; it is not an inverse or fixed-point theorem. No diagnostic implementation or reconstruction algorithm is supplied here.

---

''',['D7'],'Replace untyped fixed-point/reconstruction with downstream diagnostic')
s=read(p).replace('- Π↑ and Π↓ act as complementary filters','- Π↑ constructs the representation and Π↓ exposes downstream diagnostics').replace('- only configurations that remain stable under both mappings are retained','- diagnostic failures do not change upstream survivor status')
put(p,s,['D7'],'Remove survivor-selection backflow from consequences')
section('rcf/modules/geometry/discrete/survival-sieve.md','### **6. triangle‑compatibility**','## **invariants**',
'''### **6. Triangle compatibility in declared geometry profiles**

A downstream geometric evaluation profile may explicitly require triangle motifs or a particular face-incidence structure. It must name the carrier and relation it tests. Such a restriction belongs to that adopted geometry profile, not to the geometry-neutral core LCO scorer. No candidate is excluded merely by claiming that LCO universally requires graph triangles.

---

''',['D7'],'Keep triangle-specific restrictions downstream of neutral LCO')
repl('rcf/modules/geometry/discrete/candidate-geometries.md','A candidate geometry must support local triangle packing sufficient to realize\nthe minimum viable relational neighborhood.\n\nIn the current sieve profile, viable local closure requires at least three\nmutually supporting relations. Two-relation neighborhoods do not satisfy the\nclosure requirement under local evaluation.\n\nThis criterion is structural and does not imply downstream physical meaning.',
'''A declared geometric sieve profile may require local triangle motifs or three mutually supporting relations. Its carrier, tested relation and closure criterion must be explicit. This is a profile restriction, not a universal LCO or RCF invariant. The core LCO remains geometry-neutral; no new triangle-packing rule is introduced here.''',['D7'],'Scope geometric triangle profile without changing mathematical results')

# D8 supersedes the proposal's general descent recommendation.
p='rcf/modules/lco/lco-invariants.md'
section(p,'## **6. Monotonic Local Descent Capability**','## **7.',
'''## **6. Candidate-relative evaluation without a descent guarantee**

The LCO deterministically evaluates supplied admissible candidates under its scoring contract. A best-ranked or lowest-mismatch candidate is best only among those evaluated. It need not have lower mismatch than the previous neighborhood state.

The LCO does not independently select a realized update or mutate the neighborhood, and guarantees neither monotonic descent, convergence, zero mismatch nor basin stabilization. A separately declared policy may restrict acceptance relative to a current-state baseline; that restriction is policy-specific, not a universal LCO commitment.

---

''',['D8'],'Replace universal descent-capability requirement with evaluator contract')
p='rcf/modules/lco/lco-operator.md'
repl(p,'- supports local coherence descent under admissible update rules','- supplies candidate-relative coherence diagnostics without guaranteeing improvement over the previous state',['D8'],'Remove scorer descent obligation')
section(p,'### **Compatibility with Local Coherence Descent**','## **6. Separation from Update Policy**',
'''### **Candidate-relative evaluation**

Identical scoring inputs produce the same score. Comparing scores can identify the best supplied candidate, but supplies no guarantee that neighborhood mismatch decreases from its previous value. No convergence, zero mismatch or basin stabilization follows. Selection, stochastic realization, acceptance restrictions and state mutation are separate policy responsibilities.

---

''',['D8'],'Separate deterministic scoring from realization and mutation')
p='rcf/modules/lco/README.md'
repl(p,'The LCO is a design choice within this framework. It is the only operator we have identified that satisfies the invariants implied by Relational Primacy under the current constraint set. Different invariants would yield different operators. Within this system, it represents the minimal operator that survives the applied constraints.','LCO names the current family of geometry-neutral local scoring contracts. No uniqueness theorem for its scoring formula is asserted. It evaluates supplied candidates; a lowest-mismatch result is candidate-relative and does not guarantee improvement over the prior neighborhood. Selection and mutation remain separate policy responsibilities.',['D8'],'Remove uniqueness and optimization overclaim')
for p in ['rcf/modules/lco/README.md','rcf/modules/lco/lco-core.md','rcf/modules/lco/lco-dispersion.md']:
    s=read(p)
    s=s.replace('supports local coherence descent under admissible update conditions','provides candidate-relative scores without a general neighborhood-descent guarantee')
    s=s.replace('is designed to contribute to local coherence descent under admissible update conditions','evaluates supplied candidates without guaranteeing lower mismatch than the previous neighborhood')
    s=s.replace('be compatible with local coherence descent under admissible update policies','return deterministic candidate scores without guaranteeing neighborhood descent')
    s=s.replace('support local coherence descent under admissible update policies','support local candidate comparison without guaranteeing neighborhood descent')
    s=s.replace('supporting smooth local coherence descent','supporting comparison of local candidate dispersion')
    s=s.replace('It ensures that candidate updates maintain coherence across the entire relational neighborhood and supports the operator’s ability to guide local coherence descent.','It supplies a local dispersion diagnostic. It does not ensure that an accepted update improves neighborhood coherence; acceptance and mutation belong to the declared policy.')
    put(p,s,['D8'],'Propagate evaluator-only semantics without changing score equations')
p='rcf/modules/lco/lco-update-policy.md'
s=read(p).replace('### **3.3 Enforcement of Local Coherence Descent**','### **3.3 Declared acceptance profile**')
s=s.replace('Ensuring that accepted updates do not increase local incoherence.\n\nThe policy may enforce strict descent or allow controlled stochastic exploration, as long as invariants are preserved.','A policy explicitly declares its acceptance rules. It may impose a current-state non-increase or strict-descent restriction, or use another admissible stochastic profile. Such restrictions are policy-specific; LCO scoring imposes no universal descent or convergence guarantee.')
s=s.replace('- supports local coherence descent under the chosen acceptance policy','- applies its explicitly declared acceptance profile without attributing a universal descent guarantee to LCO')
s=s.replace('Identical inputs must yield identical decisions.','Identical non-random scoring inputs give identical scores. A stochastic policy specifies the same conditional distribution for identical policy inputs; deterministic replay additionally requires an explicit random trace or seed contract.')
s=s.replace('### **Strict Descent**\nAccept the candidate with the lowest coherence score.','### **Candidate-relative minimum**\nReturn the lowest-scored supplied candidate. This need not improve the previous neighborhood. A separately declared strict-descent acceptance profile additionally compares against C_current and rejects proposals that fail its stated inequality.')
s=s.replace('### **Stochastic Descent**','### **Stochastic candidate selection**')
s=s.replace('for some externally supplied \\(\\beta\\).','for some externally supplied \\(\\beta\\), normalized over the declared finite admissible candidate set. This distribution can select a candidate with higher mismatch than the prior state; it is not a descent guarantee. A policy-specific acceptance filter, when adopted, is stated separately.')
s=s.replace('### **Admissibility‑Filtered Descent**','### **Admissibility-filtered candidate evaluation**')
put(p,s,['D8'],'Correct stochastic/deterministic examples without imposing general descent')

# Historical numerical records remain intact, with explicit evidential boundary.
history_note='''This document retains historical repository-reported exploratory findings, including its original numerical values, formulas, tier-drag narrative and crossover discussion. Original run provenance may be unavailable; reproduction is not established here. Historical sweep gamma is distinct from current local density κ and bounded Temporal/LCO memory weight γ. No conversion, normalization or fitted mapping is supplied. The tabulated values are not verified output of the currently documented fixed-coefficient affine formula; in particular, the 600-cell values at historical gamma 0/1/2 have successive increments 0.078 and 0.200. The discrepancy is unresolved. These findings do not impose binding numerical constraints on current interfaces.'''
for p in ['rcf/modules/geometry/discrete/mismatch-function.md','rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md',
 'rcf/modules/geometry/discrete/600-cell/survival-stats.md','rcf/modules/geometry/discrete/dual-5-demicube/survival-stats.md','rcf/modules/geometry/discrete/e8/survival-stats.md']:
    note(p,'Historical exploratory evidence — owner clarification',history_note,['D5','D6'])
repl('rcf/modules/geometry/discrete/600-cell/survival-stats.md','- φ‑structured behavior is recorded as an empirical invariant, not an interpretation','- φ-structured survivorship is a historical repository-reported exploratory pattern, not an RCF invariant or empirically validated physical law',['D6'],'Correct empirical-invariant overstatement')
p='rcf/modules/geometry/adjacent/gamma-plateau-behavior.md'
s=read(p).replace('Survivors must remain stable across all plateaus.','Survivor status is scoped to the declared density regime. Across-plateau robustness, if separately evaluated, is an additional descriptor rather than a universal survival requirement.')
s=s.replace('Survivors must satisfy the following conditions across all γ‑plateaus:','The following historical descriptors apply only within the evaluated candidate/profile/regime; they do not redefine upstream admissibility:')
s=s.replace('Mismatch must remain within the admissibility bound \\( F \\leq 1.0 \\) at all γ‑levels.','The historical F ≤ 1.0 threshold is a reported criterion of that exploratory profile, not a universal all-gamma constraint on every survivor or current interface.')
put(p,s,['D5','D6'],'Scope plateau observations without changing historical data')
note(p,'Parameter and evidence boundary',history_note+' The legacy three-plateau indexing language is not an independently derived three-generation prediction.',['D5','D6','D9'])

# Historical indexing versus SMG's adopter-specific semantic policy.
gcm_note='''The owner records the historical construction as 12 antipodal-pair mode indices across three offset/regime labels, giving 36 indexed configurations. The three-regime choice was historically target-driven toward representing three Standard Model generations; it is not an independently derived prediction. The precise historical offset interpretation remains subject to source verification. Independently established antipodal-pair geometry is distinct from choosing or physically interpreting regime labels. The legacy “gamma plateau” names do not establish identity with current κ, bounded γ, or historical sweep gamma. These are conditional historical indexing records, not universal GCM requirements. Existing SMG adopter-specific policies remain downstream model commitments.'''
for p in ['rcf/modules/gcm_interface/survivor_indexing.md','rcf/modules/gcm_interface/README.md','rcf/modules/geometry/adjacent/mode-structure.md','rcf/modules/geometry/adjacent/symmetry-alignment.md']:
    note(p,'Historical indexing and target-driven regime choice',gcm_note,['D5','D9'])
p='rcf/modules/geometry/adjacent/mode-structure.md'
s=read(p).replace('The mode structure is a consequence of survivor geometry, not an imposed assumption.','The historical 12×3 indexing construction is a chosen organization of reported geometric records; it is not an independently derived generation prediction.')
s=s.replace('This count is an emergent property of survivor geometries.','This count follows from the historical 12×3 indexing choice; no independent necessity derivation is supplied.')
put(p,s,['D9'],'Remove independent-emergence overclaim while retaining construction')
p='rcf/modules/gcm_interface/sectorization.md'
note(p,'Carrier, geometry-profile and historical-indexing scope',gcm_note+''' Triangle-specific evaluation is restricted to a declared downstream geometry profile, not the neutral core LCO. A route must name its source/target carriers and preserved adjacency or incidence relation. No triangles in the 120-cell edge graph or singleton state-to-vertex dual map are implied; no missing route or sector payload is invented here.''',['D7','D9'])
s=read(p).replace('The LCO requires that the projected structure support a consistent triangle‑packing configuration.','A declared downstream geometric evaluation profile may require specified triangle motifs or face incidence. This does not impose triangle packing on the core LCO or assert graph triangles in the 120-cell.')
s=s.replace('It is used because its local combinatorial structure supports the triangle‑packing constraints required by the LCO.','Its triangle-compatible combinatorics can be evaluated under explicitly adopted geometry profiles; that choice is not a core LCO constraint.')
put(p,s,['D7'],'Remove universal triangle requirement from GCM body')
p='rcmt/smg_v2.1/bindings/gcm_bindings.md'
repl(p,'`gcm_interface/mode_structure.md`','`../../../rcf/modules/gcm_interface/survivor_indexing.md` (Mode Indexing section)',['D9'],'Correct authoritative public GCM index reference')
note(p,'Historical binding status',gcm_note+' `Gamma_Plateaus = [A,B,C]` remains a legacy binding identifier, not a conversion between temporal parameters or a universal plateau law.',['D5','D9'])
p='rcmt/smg_v2.1/bindings/physics_bindings.md'
repl(p,'`physics_interface/conversion_laws/formal_structure.md`','`../../physics_interface/local/conversion_laws/formal_structure.md`',['D3'],'Correct support/boundary schema path')
note(p,'Schema and consumer-stage scope','Support and boundary entries are schema imports from `../../physics_interface/local/conversion_laws/formal_structure.md`, including its Final Boundary Record section. This file supplies no instantiated numerical payload. SMG consumes explicitly exported projected records as an effective-output consumer; construction-facing raw-record permissions are not transferred to SMG.',['D3'])
for p in ['rcmt/smg_v2.1/smg_contract.md','rcmt/smg_v2.1/claims/dual_closure.md','rcmt/smg_v2.1/dictionary/gamma_generations.md','rcmt/smg_v2.1/exports/standard_model_kinematics.md']:
    note(p,'Historical indices and downstream adoption',gcm_note+' The existing 36-mode and three-generation policies remain binding only for models adopting the current SMG mapping; this clarification does not remove them.',['D5','D9'])
for p in ['rcmt/smg_v2.1/translations/24-cell-routing.md','rcmt/smg_v2.1/translations/24-cell-tranlsation.md']:
    note(p,'Provisional routing adoption','The routing identities in this document are proposed SMG-local translation requirements. They become obligations only for a model/version that explicitly adopts this extension; they are not silently activated for all SMG adopters. No routing architecture, substrate reconstruction or upstream API is implemented by this document.',['D9'])
put('rcmt/smg_v2.1/REFACTOR_STATUS.md', '''# SMG Documentation and Routing Status

This status record was added during the owner-authorized 2026-10-24 maintenance pass. It records current scope; it is not evidence of an earlier completed mathematical refactor.

- The split SMG bindings, claims, dictionary, translations, policies and exports retain their downstream ownership.
- Historical indexing uses 12 antipodal-pair indices and three target-driven regime labels; no independent generation derivation is claimed. The precise historical offset interpretation remains to be verified.
- Sp(8,R) is a research candidate in RCMT, not an SMG-imported universal group requirement.
- Proposed 24-cell/Hopf routing identities remain provisional unless an adopter explicitly selects that extension. Existing SMG 36-mode/three-generation adopter policies are preserved.
- GCM's public indexing reference is `../../../rcf/modules/gcm_interface/survivor_indexing.md`; concrete absent adjacency/action payloads remain unresolved.
- The frozen Lisi comparison input is preserved; Phase 6 conclusions are scoped to that hashed reference, not silently re-run against current prose.
''',['D9'],'Record factual current status without fabricating historical completion')

p='rcmt/README.md'
section(p,'- A dual‑lane architecture using:','- A model of physical behavior',
'''- A mandatory two-lane architecture adopted by RCMT: a symplectic mode-space lane and a Lorentz-compatible geometric lane. Sp(8,R) is an active candidate realization; SO(8)/SO(9) and product expressions remain provisional, with carrier, representation and projection questions unresolved.
''',['D1','D2','D4'],'State adopted roles versus provisional implementation')
s=read(p).replace('- A theory that explains:','- A research program exploring candidate representations of:').replace('randomness as unresolved Sp(8) structure','randomness as unresolved projected mode structure').replace('RCMT provides **mechanisms**, not metaphysics.','These are model-local representational hypotheses and research targets, not established physical mechanisms or empirical validation.')
a=s.index('rcmt/\n');b=s.index('Each file defines',a)
s=s[:a]+'''```text
rcmt/
├── local/                  # RCMT invariants, constraints and provisional symmetry patterns
├── global/                 # Scoped global-invariant representations
├── physics_interface/      # Construction-facing interfaces and effective-consumer firewall
├── smg_v2.1/               # Downstream semantic model and adopter policies
└── modules/                # Candidate matter, geometry-behavior and metric profiles
```

'''+s[b:]
s=s.replace('1. The RCF upstream core (`rcf_compressed.md`) is present and unchanged.','1. Read the upstream RCF source contracts. `../rcf/rcf_compressed.md` is a subordinate, source-versioned derived artifact, not immutable authority over its sources.')
s=s.replace('2. All RCMT invariants are defined in `invariants.md`.','2. RCMT-wide model requirements are defined in `local/invariants.md`; narrower interface/profile requirements retain their own scopes.')
s=s.replace('4. No global curvature groups or non‑compact symmetries are introduced.','4. Global curvature exclusions and other locality/finite-representation constraints remain; compactness applies only to explicitly adopted profiles.')
s=s.replace('5. All mode‑selection behavior is expressed through the LCO.','5. LCO evaluates supplied candidates; selection/acceptance, stochastic realization and mutation belong to declared policies, without a universal neighborhood-descent guarantee.')
put(p,s,['D1','D2','D4','D8'],'Correct architectural summary and repository navigation')

# Navigation and scoped theorem/comparison status; historical bodies stay archived.
p='rcf/modules/geometry/projection/README.md'
s=read(p)
for a,b in [('projection_core.md','projection-core.md'),('projection_invariants.md','projection-invariants.md'),('projection_outputs.md','projection-outputs.md'),('adjacency_preservation.md','adjacency-preservation.md'),('slush_projection.md','slush-projection.md')]:s=s.replace(a,b)
put(p,s,['D10'],'Correct existing-file navigation')
p='rcmt/local/README.md'
s=read(p).replace('`theory/rcf_projects/rcf_compressed.md`','`../../rcf/core/S∞.RCF.Invariants.v1.1.md` and explicitly consumed RCF interfaces').replace('`theory/rcf_projects/rcp-1.1.md`','`../../rcf/rcp-1.2.md`')
put(p,s,['D10'],'Point authority and protocol references to actual sources')
note('projects/lisi-smg-test/PLAN.md','Historical planning snapshot','The following plan retains its planning-time inventory and assumptions. Its empty src/tests and unfinished-phase statements are historical, not the current state. Completed phase reports and tests are indexed separately. The frozen SMG reference remains byte-identical; current split-module prose is not silently substituted into the recorded comparison.',['D9','D10'])
note('projects/lisi-smg-test/summary.md','Verification and configuration scope',
'''The explicit subset definition makes the raw four-carrier configuration lattice Boolean B4; the recovered V4 action yields a seven-class non-distributive orbit lattice. The global 24-fiber orbit poset is a different object and is reportedly not a lattice. Phase 6 established no full structure-preserving correspondence to its frozen SMG reference. A local 3+1 singleton is not invariant under the retained transitive four-pair action; the heterogeneous 36-record projection requires additional choices and discards relations. Physical interpretations remain downstream. No expensive global enumeration was rerun in this maintenance pass.''',['D9'])
note('projects/Operational_Phase_Space/Vocabulary_Packet.md','Later verified status — 2026-10-24',
'''The orbit-bridge criterion is proved in V4_Orbit-Bridge_Regularity_Theorem.md under its free-V4/two-orbit/commuting-independent-involution hypotheses. Selected finite-carrier and SIM16 algebraic results were reproduced during onboarding from documented extracts. This establishes conditional mathematics/computation, not physical M2 selection, operational observability, dynamics or a generation mechanism. Historical discussions retain their original chronological uncertainty.''',['D1','D9'])

(HERE/'decision-edits.json').write_text(json.dumps(events,indent=2,ensure_ascii=False),encoding='utf-8')
print('Decision edit operations:',len(events),'files:',len({x['path'] for x in events}))

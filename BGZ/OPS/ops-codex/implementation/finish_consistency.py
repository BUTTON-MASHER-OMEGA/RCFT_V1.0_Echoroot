"""Targeted post-migration consistency corrections from approved D1–D10."""
from pathlib import Path
import re,json,os,hashlib

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];events=[]
def edit(p,func,decisions,why):
    f=ROOT/p;old=f.read_text(encoding='utf-8');new=func(old)
    if new!=old:
        f.write_text(new,encoding='utf-8');events.append({'path':p,'decisions':decisions,'reason':why})
def replacements(p,pairs,decisions,why):
    def apply(s):
        for a,b in pairs:
            if a not in s:raise ValueError((p,a))
            s=s.replace(a,b)
        return s
    edit(p,apply,decisions,why)
def field(p,name,value,why):
    edit(p,lambda s:re.sub(r'(?m)^- '+re.escape(name)+r':[^\n]*',lambda m:'- '+name+': '+value,s,count=1),['D10'],why)
def note(p,title,body,d):
    def apply(s):
        marker=re.search(r'(?m)^- anti-backflow:[^\n]*\n',s)
        assert marker
        return s[:marker.end()]+'\n## '+title+'\n\n'+body+'\n\n'+s[marker.end():].lstrip('\n')
    edit(p,apply,d,title)

replacements('rcf/modules/lco/interface.md',[
 ('updated local record','evaluation diagnostic record (without neighborhood mutation)'),
 ('LCO may not perform global selection,','LCO may not independently select a realized update, mutate neighborhood state, guarantee mismatch descent/convergence, perform global selection,')],['D8'],'Make non-mutation and lack of descent guarantee explicit at public interface')
replacements('rcf/modules/physics/physics-contract.md',[
 ('* Lyapunov descent or equivalent monotonicity constraints','* explicitly adopted stability/acceptance criteria; Lyapunov descent or monotonicity is optional and profile-specific, never a universal LCO guarantee')],['D8'],'Remove universal descent requirement from downstream physics summary')
replacements('rcf/modules/gcm_interface/sectorization.md',[('- stable under LCO pruning','- reported stable under declared update policies using LCO scores, within the historical evaluation profile')],['D8'],'Replace scorer-as-pruning actor')
replacements('rcf/modules/geometry/discrete/README.md',[('LCO defines the local update rule; GCO defines long‑horizon coherence.','LCO evaluates supplied local candidates; a separately declared policy selects and applies updates. GCO aggregates long-horizon coherence records without modifying updates.')],['D8'],'Propagate scoring/policy/aggregation ownership')
edit('rcmt/global/README.md',lambda s:s.replace('Local operators (like LCO) may change representations, smooth mismatch, or prune noise,','Declared local update policies may change representations, smooth mismatch or prune noise under their stated move class; the LCO evaluates supplied candidates without mutation,').replace('no matter how LCO prunes locally','under the declared invariant-preserving local moves').replace('LCO can smooth, align, and prune—but cannot remove global invariants.','Declared admissible policies preserve the represented global invariant; LCO scoring itself does not smooth, prune or mutate.').replace('persists no matter how hard LCO prunes locally','persists under the declared invariant-preserving local update moves'),['D7','D8'],'Remove LCO mutation from global examples')
for p in ['rcmt/local/constraints/lco.md','rcmt/local/constraints/projection.md']:
    edit(p,lambda s:s.replace('and enforce coherence thresholds.','and expose scores to a separately declared threshold/acceptance policy.'),['D8'],'Threshold enforcement belongs to policy')
note('rcmt/local/constraints/lco.md','Candidate-relative scoring boundary',
 'The LCO evaluates supplied admissible candidates under its declared scoring contract. A lowest-mismatch ranking is relative to that supplied set, not a guarantee of improvement over the previous neighborhood. It does not select a realized update, mutate the neighborhood, enforce convergence or guarantee zero mismatch/basin stabilization. Selection, acceptance restrictions and stochastic realization belong to separately declared policies. The precise adaptation of mode/geometric records to scorer inputs remains a representation issue, not permission to redefine upstream scoring.',['D8'])
for p in (ROOT/'rcmt/local/symmetry_groups/patterns').glob('*.md'):
    edit(p.relative_to(ROOT).as_posix(),lambda s:s.replace('LCO evaluation accepts the','The declared policy accepts the LCO-scored'),['D8'],'Separate evaluation from policy acceptance in examples')
edit('rcmt/local/symmetry_groups/actions.md',lambda s:s.replace('- LCO acceptance','- acceptance by the declared policy using LCO scores'),['D8'],'Correct acceptance actor')
edit('rcmt/local/interpretations.md',lambda s:s.replace('local LCO update rates','local policy update rates evaluated using LCO scores'),['D8'],'LCO evaluator is not a scheduler')
for p in ['rcmt/physics_interface/pipeline/effective_field_contract.md','rcmt/physics_interface/pipeline/projection_interpretation_rule.md']:
    note(p,'Construction/export boundary',
      'Projection-layer language here describes construction-facing representation/export responsibilities, not an expansion of the upstream RCF handoff. Metric and physics-time data must already be produced at their authorized stages. LCO supplies evaluation records, not an automatically coherent mutated state; candidate selection and state mutation belong to declared policies. Effective-output consumers receive exports only, without raw-substrate permission.',['D3','D7','D8'])
    edit(p,lambda s:s.replace('coherent relational state from LCO','authorized relational state plus LCO evaluation records').replace('**coherent relational state** from LCO','**authorized relational state plus LCO evaluation records**'),['D8'],'Remove guaranteed coherent-state output from LCO')

# Normalize historical labels without selecting a new real form for unspecified models.
edit('rcmt/local/constraints/symmetry_chain.md',lambda s:s.replace('the specific factorization into \\(Sp(8) \\times SO(9)\\),','the historical factorization notation \\(Sp(8) \\times SO(9)\\), not an active proved decomposition,'),['D2'],'Classify residual bare Sp notation as historical')
for p in (ROOT/'rcmt').rglob('*.md'):
    text=p.read_text(encoding='utf-8')
    purpose=re.search(r'(?m)^- purpose: (.*)',text)
    if purpose and purpose[1].count('`')%2:
        field(p.relative_to(ROOT).as_posix(),'purpose',purpose[1].replace('`',''),'Remove unbalanced inline-code delimiter from migrated purpose')
field('rcmt/smg_v2.1/translations/24-cell-routing.md','type','Contract','Proposed normative extension is a conditional contract, not an activated rule')
field('rcmt/smg_v2.1/translations/24-cell-routing.md','authority','proposed SMGv2.1-local translation contract; normative only for explicit adopters; no upstream authority','Preserve proposed/adopter-only scope')

# Record explicit upstream relationships stated in each module's pre-migration README.
base=ROOT/'rcmt/physics_interface/local'
upstreams={
 'group_structures':[ROOT/'rcf/modules/group_structures/interface.md'],
 'relational_kinematics':[ROOT/'rcf/modules/geometry/discrete/interface.md',ROOT/'rcf/modules/geometry/continuous/interface.md',ROOT/'rcf/core/rcd/README.md',base/'invariants.md'],
 'relational_dynamics':[base/'relational_kinematics/interface.md',ROOT/'rcf/core/rcd/README.md',base/'invariants.md'],
 'relational_field_theory_discrete':[base/'relational_kinematics/interface.md',base/'relational_dynamics/interface.md',base/'invariants.md'],
 'time_bridge':[base/'relational_kinematics/interface.md',base/'relational_dynamics/interface.md',base/'relational_field_theory_discrete/interface.md',ROOT/'rcf/modules/time/interface.md',ROOT/'rcf/modules/temporal-interface/interface.md',base/'constraints/time_constraints.md'],
 'continuous_field_theory':[base/'relational_field_theory_discrete/interface.md',base/'time_bridge/interface.md'],
 'emergent_metric_geometry':[base/'time_bridge/interface.md',base/'continuous_field_theory/interface.md',base/'constraints/geometry_constraints.md'],
 'relational_quantum_mechanics':[base/'time_bridge/interface.md',base/'invariants.md',base/'constraints/adjacency_refinements.md'],
 'conversion_laws':[base/'relational_field_theory_discrete/interface.md',base/'continuous_field_theory/interface.md',base/'emergent_metric_geometry/interface.md',base/'relational_quantum_mechanics/interface.md',base/'time_bridge/interface.md']}
for name,sources in upstreams.items():
    assert all(q.is_file() for q in sources)
    for p in (base/name).glob('*.md'):
        s=p.read_text(encoding='utf-8');m=re.search(r'(?m)^- upstream dependencies: (.*)',s)
        old=re.findall(r'`([^`]+)`',m[1]);paths=[(p.parent/x).resolve() for x in old]
        paths=list(dict.fromkeys(paths+[q.resolve() for q in sources]));paths=[q for q in paths if q!=p.resolve()]
        value=', '.join('`'+os.path.relpath(q,p.parent).replace('\\','/')+'`' for q in paths)
        field(p.relative_to(ROOT).as_posix(),'upstream dependencies',value,'Resolve named upstream inputs from existing module Position in Stack and dependency declarations; no new APIs')
field('rcmt/physics_interface/local/constraints/group_structure_constraints.md','upstream dependencies','`../../README.md`, `../../../../rcf/modules/group_structures/interface.md`','Resolve explicitly named group-structure owner')
edit('rcmt/physics_interface/local/constraints/group_structure_constraints.md',lambda s:s.replace('`foundation/abstract/group_structures`','`../../../../rcf/modules/group_structures/interface.md`'),['D3','D10'],'Correct conceptual legacy group path to existing public interface')

# Clarify old compatibility footer language; preserve its historical assertion.
for p in (ROOT/'rcmt').rglob('*.md'):
    edit(p.relative_to(ROOT).as_posix(),lambda s:s.replace('## RCP‑1.1 Compliance','## Legacy RCP-1.1 statements (retained history)').replace('# RCP‑1.1 Compliance','# Legacy RCP-1.1 statements (retained history)').replace('## RCP-1.1 Compatibility','## Legacy RCP-1.1 compatibility notes'),['D10'],'Distinguish historical compliance prose from current header validation')
(HERE/'followup-edits.json').write_text(json.dumps(events,indent=2,ensure_ascii=False),encoding='utf-8')
print('Targeted operations:',len(events))

from pathlib import Path
import json,os,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];changes=[]
def fix(p,pairs):
    f=ROOT/p;s=f.read_text(encoding='utf-8');old=s
    for a,b in pairs:
        assert a in s,(p,a);s=s.replace(a,b)
    if old!=s:f.write_text(s,encoding='utf-8');changes.append({'path':p,'decisions':['D10'],'reason':'Resolve existing document references without changing scientific claims'})
fix('rcmt/smg_v2.1/REFACTOR_STATUS.md',[('`../../../rcf/modules/gcm_interface/survivor_indexing.md`','`../../rcf/modules/gcm_interface/survivor_indexing.md`')])
fix('rcmt/local/symmetry_groups/README.md',[('`symmetry_chain.md`','`../constraints/symmetry_chain.md`')])
fix('rcmt/local/symmetry_groups/patterns/README.md',[(f'`{x}.md`',f'`../{x}.md`') for x in ['generators','actions','representations']])
for n in ['README.md','charged_lepton_class.md','neutral_lepton_class.md']:
    fix('rcmt/modules/survivor_matter_structure/leptons/'+n,[('`spinor_projection_contract.md`','`../local/spinor_projection_contract.md`')])
for n in ['modules/README_MODULES.md','stdlib/README_STDLIB.md']:
    p='rcmt/physics_interface/'+n
    target=os.path.relpath(ROOT/'rcf/rcf_compressed.md',(ROOT/p).parent).replace('\\','/')
    fix(p,[('`rcf_compressed.md`','`'+target+'`'),('`local/invariants.md`','`../local/invariants.md`')])
for n in ['local_to_geometry_adapter.md','local_to_physics_adapter.md']:
    p='rcmt/physics_interface/local/adapters/'+n
    s=(ROOT/p).read_text(encoding='utf-8')
    tokens=set(re.findall(r'`(constraints/[^`]+\.md)`',s))
    fix(p,[('`'+t+'`','`../'+t+'`') for t in tokens])
fix('rcmt/smg_v2.1/bindings/gcm_bindings.md',[
 ('`gcm_interface/survivor_indexing.md`','`../../../rcf/modules/gcm_interface/survivor_indexing.md`'),
 ('`gcm_interface/sectorization.md`','`../../../rcf/modules/gcm_interface/sectorization.md`')])
for n in ['relational_dynamics','relational_field_theory_discrete','relational_kinematics','relational_quantum_mechanics']:
    p='rcmt/physics_interface/local/'+n+'/interface.md'
    s=(ROOT/p).read_text(encoding='utf-8')
    if '`local/invariants.md`' in s:fix(p,[('`local/invariants.md`','`../invariants.md`')])
(HERE/'path-edits.json').write_text(json.dumps(changes,indent=2),encoding='utf-8')
print('Path repairs:',len(changes))

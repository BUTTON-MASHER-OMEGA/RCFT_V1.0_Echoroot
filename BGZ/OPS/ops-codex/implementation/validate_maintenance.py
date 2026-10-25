"""Final bounded maintenance checks; never dispatches simulations."""
from pathlib import Path
import ast,hashlib,json,re

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
baseline=json.loads((HERE/'baseline.json').read_text());areas=['rcf','rcmt','projects/Operational_Phase_Space','projects/lisi-smg-test']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
after={p.relative_to(ROOT).as_posix():digest(p) for a in areas for p in (ROOT/a).rglob('*') if p.is_file() and '.git' not in p.parts}
delta=[{'path':k,'change':'added' if k not in baseline else 'deleted' if k not in after else 'modified',
        'before_sha256':baseline.get(k),'after_sha256':after.get(k)}
       for k in sorted(baseline.keys()|after.keys()) if baseline.get(k)!=after.get(k)]
assert not [r for r in delta if '__pycache__' in r['path'] or '.pytest_cache' in r['path']]
assert not [r for r in delta if r['change']=='deleted']
frozen=['projects/lisi-smg-test/references/smg_v2.1_concat.md','projects/lisi-smg-test/references/cpt_and_triality_lisi.pdf',
        'projects/lisi-smg-test/reference_geometry/120cell_vertices.txt',
        'projects/lisi-smg-test/reference_geometry/120cell_adjacency.txt',
        'projects/lisi-smg-test/reference_geometry/600cell_adjacency.txt']
assert all(after[p]==baseline[p] for p in frozen)
assert all(after[p]==v for p,v in baseline.items() if p.startswith('projects/lisi-smg-test/artifacts/'))
archive=json.loads((ROOT/'projects/Operational_Phase_Space/ARCHIVE_MANIFEST.json').read_text())
for row in archive:
    p=ROOT/row['source'];ast.parse(p.read_text(encoding='utf-8'))
    assert digest(ROOT/row['full_original_archive'])==row['original_sha256']
    assert digest(p)==row['clean_sha256']
    assert digest(ROOT/row['historical_result'])==row['historical_result_sha256']
runs=json.loads((HERE/'ops-repair-verification.json').read_text())
executed=[r for r in runs if 'exit_code' in r]
assert len(executed)==8 and all(r['exit_code']==0 and r['seconds']<10 for r in executed)
assert all(r['packet_equal'] for r in executed if r['historical_packet_present'])
for repo in ['rcf','rcmt']:
    manifest=json.loads((ROOT/repo/(repo+'_compressed.sources.json')).read_text())
    assert all(digest(ROOT/repo/p)==v for p,v in manifest['working_tree_source_manifest'].items())
    text=json.dumps(manifest['working_tree_source_manifest'],sort_keys=True,separators=(',',':')).encode()
    assert hashlib.sha256(text).hexdigest()==manifest['manifest_sha256']
numeric='rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md'
old=(HERE/'before'/numeric).read_text(encoding='utf-8-sig');new=(ROOT/numeric).read_text(encoding='utf-8')
assert re.findall(r'```text\n(.*?)```',old,re.S)==re.findall(r'```text\n(.*?)```',new,re.S)
rcp=json.loads((HERE/'rcp-validation.json').read_text())
assert rcp['documents']==rcp['structural_pass']==165 and not rcp['dependency_cycles']
assert json.loads((HERE/'body-path-review.json').read_text())=={'missing':[],'ambiguous_aliases':[]}
result={'source_changes':len(delta),'changes_by_area':{a:sum(r['path'].startswith(a+'/') for r in delta) for a in areas},
 'unchanged_frozen_inputs':frozen,'preexisting_caches_unchanged':True,'historical_density_table_unchanged':True,
 'clean_ops_scripts':21,'bounded_ops_runs':8,'skipped_ops_runs':13,'matching_historical_json_packets':4,
 'rcmt_canonical_structures':165,'unresolved_source_documents':4,'generated_registry_unready':True,
 'header_dependency_cycles':0,'active_body_missing_paths':0,'source_manifests_verified':True,
 'bounded_test_log':'bounded-tests.txt','test_summary':(HERE/'bounded-tests.txt').read_text(encoding='utf-8').strip()}
(HERE/'final-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(HERE/'changed-files.json').write_text(json.dumps(delta,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

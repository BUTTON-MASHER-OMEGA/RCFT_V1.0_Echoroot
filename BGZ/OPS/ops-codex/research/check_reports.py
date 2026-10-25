"""Check deliverables, citation targets, and recorded verification outcomes."""
from pathlib import Path
import json, re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
reports=sorted((HERE.parent/'onboarding').glob('*.md'))
issues=[]
citations=0
for p in reports:
    body=p.read_text(encoding='utf-8')
    if '@{' in body: issues.append(f'Unrendered reference in {p.name}')
    for target, line in re.findall(r'\]\((C:/[^)]+):(\d+)\)',body):
        citations+=1
        source=Path(target)
        if not source.is_file(): issues.append(f'Missing citation target: {target}')
        elif int(line)>len(source.read_text(encoding='utf-8-sig').splitlines()):
            issues.append(f'Citation out of bounds: {target}:{line}')
assert len(reports)==8
assert not issues,issues
extractions=json.loads((HERE/'extraction-manifest.json').read_text())
assert len(extractions)==21 and all(x['extracted_syntax']=='valid' for x in extractions)
runs=json.loads((HERE/'reproduction-results.json').read_text())
assert len(runs)==9 and all(x.get('exit_code')==0 for x in runs)
assert json.loads((HERE/'source-integrity-check.json').read_text())=={}
result={'reports':len(reports),'valid_citations':citations,'extracted_scripts':21,
        'successful_selected_ops_runs':9,'source_changes_since_baseline':0,
        'lisi_test_logs':['lisi-geometry-tests.txt','lisi-selected-tests.txt']}
(HERE/'report-check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

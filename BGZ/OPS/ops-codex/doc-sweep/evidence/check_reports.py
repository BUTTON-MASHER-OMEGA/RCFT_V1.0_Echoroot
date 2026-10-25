"""Validate proposal deliverables and source citation targets without source edits."""
from pathlib import Path
import json,re

HERE=Path(__file__).resolve().parent
reports=sorted(HERE.parent.glob('*.md'))
expected=['00-executive-summary.md','01-authority-and-evidence-audit.md',
 '02-two-lane-origin-investigation.md','03-contract-and-interface-conflicts.md',
 '04-mathematical-and-numerical-claims.md','05-proposed-change-ledger.md',
 '06-dependency-and-propagation-plan.md','07-human-decisions-required.md']
assert [p.name for p in reports]==expected
issues=[];count=0
for p in reports:
    text=p.read_text(encoding='utf-8')
    if '@{' in text:issues.append(f'Unrendered reference: {p.name}')
    for target,line in re.findall(r'\]\((C:/[^)]+):(\d+)\)',text):
        count+=1;source=Path(target)
        if not source.is_file():issues.append(f'Missing target: {target}')
        elif int(line)>len(source.read_text(encoding='utf-8-sig').splitlines()):
            issues.append(f'Out of bounds: {target}:{line}')
ledger=(HERE.parent/'05-proposed-change-ledger.md').read_text(encoding='utf-8')
ids=re.findall(r'^## (C\d+) —',ledger,re.M)
assert ids==[f'C{i:02d}' for i in range(1,31)]
for field in ['File and section','Current wording or precise claim','Identified inconsistency',
 'Evidence and source references','Proposed replacement wording','Change rationale',
 'Affected downstream documents','Risk and approval']:
    assert ledger.count(f'**{field}:**')==30,(field,ledger.count(f'**{field}:**'))
assert not issues,issues
result={'reports':len(reports),'proposal_entries':len(ids),'valid_source_links':count,
        'all_eight_proposal_fields_present':True}
(HERE/'report-check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

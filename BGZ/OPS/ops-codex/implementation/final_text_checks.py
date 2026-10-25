from pathlib import Path
import re,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];events=[]
def change(p,pairs,d,why):
    f=ROOT/p;s=f.read_text(encoding='utf-8');old=s
    for a,b in pairs:
        assert a in s,(p,a);s=s.replace(a,b)
    if old!=s:f.write_text(s,encoding='utf-8');events.append({'path':p,'decisions':d,'reason':why})
change('rcf/modules/gcm_interface/survivor_indexing.md',[
 ('Upstream evaluations provide a **12‑element chromatic indexing scheme** that partitions survivor structure into twelve repeating components.','The historical construction supplies a **12-element antipodal-pair indexing scheme**, also called Chromatic-12 in legacy descriptors; no universal 12-element requirement for every GCM input is asserted.'),
 ('Survivor geometries exhibit **three broad γ‑plateau regions** in upstream evaluations.','The historical indexing construction adopts **three offset/regime labels**. The legacy gamma-plateau terminology is retained for compatibility; the precise original offset meaning remains unverified and the three-label choice was target-driven.'),
 ('Upstream evaluations currently provide a **36‑mode structure** arising from the repetition of Chromatic‑12 across the γ‑plateaus.','The historical index product contains **36 configurations** (12 pair indices × 3 regime labels). This is indexing arithmetic under the adopted construction, not an independently derived physical mode count.')],['D5','D9'],'Align active public indexing prose with approved target-driven history')
change('rcf/modules/geometry/adjacent/mode-structure.md',[
 ('All survivors contain a recurring 12‑element combinatorial motif known as the **Chromatic 12**.','The historical 12/36 indexing profile uses a recurring 12-element base called **Chromatic 12**; it is not a universal requirement on all survivor geometries.'),
 ('Geometries lacking a stable 12‑mode base are eliminated during plateau transitions.','Within the historical target-driven indexing profile, a stable 12-index base was an adopted filter. This does not independently derive that base or impose it on other profiles.')],['D9'],'Remove universal and independent-emergence implications')
change('rcf/modules/geometry/adjacent/symmetry-alignment.md',[
 ('The Chromatic 12 motif must remain stable across all γ‑plateaus.','The historical 12/36 indexing profile sought a stable base across its three chosen regime labels; this is not an additional universal sieve condition.'),
 ('The 36‑mode structure arises from symmetry alignment:','The historical 36-index organization was described through symmetry alignment; the following is that profile narrative, not an independent generation derivation:')],['D6','D9'],'Keep alignment narrative explicitly historical/profile-scoped')
change('rcmt/physics_interface/local/constraints/time_constraints.md',[
 ('Time must remain explicit, emergent, and downstream‑owned.','Time ownership remains upstream with the RCF Time contract; downstream physics-time records must remain explicit and pass through time_bridge.'),
 ('The Upstream Time Index is **not temporal**; it is a symbolic ordering key.','The Upstream Time Index represents RCF event order; it is **not physical or metric time**. Calling it a symbolic key does not remove its upstream temporal-order meaning.'),
 ('no reinterpretation of the Upstream Time Index as temporal','no reinterpretation of upstream event order as an already available physical/metric time coordinate')],['D3','D5'],'Resolve event-order versus physics-time ambiguity against explicit upstream owner')
change('rcmt/physics_interface/local/adapters/local_to_geometry_adapter.md',[
 ('Geometry modules may import:','Construction-facing geometry consumers within RCMT may import:'),
 ('Geometry modules may:','Construction-facing geometry consumers within RCMT may:')],['D3'],'Avoid suggesting downstream refinements grant access or new requirements to upstream RCF modules')
change('rcmt/physics_interface/pipeline/coherence_contract.md',[
 ('- the definition of coherence scoring','- RCMT-local classification of imported coherence-scoring records; this does not redefine or retune the upstream RCF LCO score')],['D3','D8'],'Constrain local scoring terminology to imported-record interpretation')
for p in ['rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md','rcf/modules/geometry/discrete/600-cell/survival-stats.md','rcf/modules/geometry/discrete/dual-5-demicube/survival-stats.md','rcf/modules/geometry/discrete/e8/survival-stats.md']:
    change(p,[('- status: active','- status: historical repository-reported exploratory findings; original execution provenance and reproduction unresolved')],['D6'],'Put historical evidence status in the document contract')
change('rcf/modules/geometry/discrete/mismatch-function.md',[
 ('- status: active','- status: current formula documentation with historical exploratory records; numerical relationship and provenance unresolved')],['D6'],'Distinguish formula specification from unverified historical numerical records')
(HERE/'final-text-edits.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
print('Final consistency edits:',len(events))

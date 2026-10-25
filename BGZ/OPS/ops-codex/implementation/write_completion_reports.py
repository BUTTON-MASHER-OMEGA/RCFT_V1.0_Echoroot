"""Produce the four requested implementation reports from recorded evidence."""
from pathlib import Path
import json,re,hashlib

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
final=json.loads((HERE/'final-verification.json').read_text())
delta=json.loads((HERE/'changed-files.json').read_text())
rcp=json.loads((HERE/'rcp-validation.json').read_text())
runs=json.loads((HERE/'ops-repair-verification.json').read_text())
migration=json.loads((HERE/'rcmt-migration.json').read_text())
coord=json.loads((HERE/'coordinate-repair-result.json').read_text())
operations=[]
for name in ['decision-edits.json','followup-edits.json','final-text-edits.json','path-edits.json']:
    operations+=json.loads((HERE/name).read_text(encoding='utf-8'))
byfile={}
for row in operations:
    r=byfile.setdefault(row['path'],{'decisions':set(),'reasons':[]})
    r['decisions'].update(row['decisions'])
    reason=row.get('reason','approved document correction')
    if reason not in r['reasons']:r['reasons'].append(reason)
for row in delta:
    p=row['path'];r=byfile.setdefault(p,{'decisions':set(),'reasons':[]})
    if p.startswith('rcmt/') and p.endswith('.md'):
        r['decisions'].add('D10');r['reasons'].append('Canonical RCP-1.2 migration/classification; legacy metadata retained where applicable')
    if 'compressed' in p:
        r['decisions'].add('D10');r['reasons'].append('Regenerated subordinate contract registry/source manifest after authoritative edits')
    if p.startswith('projects/Operational_Phase_Space/') and p.endswith(('.py','.txt','.json')):
        r['decisions'].add('D11');r['reasons'].append('Separate executable code, original mixed record archive, and historical output; preserve calculations')
    if p.startswith('projects/lisi-smg-test/') and any(x in p for x in ['coordinate_repair','repaired.txt','COORDINATE_REPAIR']):
        r['decisions'].add('D11');r['reasons'].append('Graph-certified numerical coordinate reconstruction, explicit provenance and bounded regression checks')
    row.update(decisions=sorted(r['decisions']),reasons=r['reasons'])
(HERE/'change-ledger.json').write_text(json.dumps(delta,indent=2,ensure_ascii=False),encoding='utf-8')

reports={}
ledger='''## Execution scope

Implemented the approved owner D1–D11 register, with the later runtime restriction: no long-running simulation or expensive enumeration was launched. D4 removes blanket compactness; D8 removes a general LCO descent requirement; D9 classifies the historical regime choice as target-driven. These decisions supersede the corresponding earlier proposal recommendations.

Original source snapshots and SHA-256 baseline are retained under `before/` and `baseline.json`. Existing RCF Python-cache modifications were present before work and were not changed. No commits were created, dependencies installed, remote messages sent or other-agent workspace inspected.

## Decision batches

| Batch | Implemented scope |
|---|---|
| Protocol | RCF RCP-1.2 opening structure clarified separately from RCMT migration; no additional sentinel |
| D1–D4 | Mandatory RCMT lanes/optional RCF profile; Sp(8,R) research candidate; staged consumer permissions; compactness only in adopted profiles |
| D5–D7 | κ/bounded γ/historical sweep separation; preserved numbers and provenance gap; Π↓ diagnostic; triangle and transport carrier scope |
| D8 | Candidate-relative evaluation, no universal descent/convergence/mutation guarantee; deterministic scoring versus stochastic selection |
| D9 | Historical 12 antipodal-pair × three target-driven offset/regime labels; SMG adopter-specific policies retained; routing still provisional |
| D11 OPS | 21 clean executable scripts, archived originals and recorded outputs; eight bounded reruns; thirteen skipped |
| D11 coordinates | Original input preserved; new repaired realization, exact discrete graph witness, certificate and five regression checks |
| D10 | 163 existing RCMT Markdown sources migrated; new SMG status record and generated RCMT registry also have canonical headers; dependencies and unresolved metadata recorded |
| Derived | RCF and RCMT contract registries/source manifests regenerated last; frozen Lisi comparison inputs unchanged |

## Files changed

244 source/artifact paths changed or were added: RCF 28, RCMT 166, OPS 44, Lisi study 6. No source file was deleted. Each row below names its applied decision and reason; exact before/after hashes are in `change-ledger.json`. The separate operation logs preserve reviewable protocol, content, header and path-repair batches.

| File | Change | Decisions | Reason |
|---|---|---|---|
'''
for row in delta:
    path=row['path'];reasons='; '.join(dict.fromkeys(row['reasons']))
    ledger+=f'| @{{{path}#1}} | {row["change"]} | {", ".join(row["decisions"])} | {reasons.replace("|","/")} |\n'
reports['00-change-ledger.md']=('Approved cleanup change ledger',ledger)

text='''## Coverage and protocol interpretation

All **163 pre-existing RCMT Markdown files** received canonical RCP-1.2 headers. The new `smg_v2.1/REFACTOR_STATUS.md` and `rcmt_compressed.md` also comply with the opening/field structure, bringing the final corpus to **165** Markdown files. No RCMT Markdown file was silently excluded. There was no byte-frozen Markdown artifact inside this migrated corpus; the Lisi frozen concatenation lies outside RCMT and remains unchanged.

The RCF protocol correction is separate: one document H1 followed immediately by the canonical Document Contract, with no undefined extra sentinel. Fields are owner, type, scope, authority, upstream dependencies, status, purpose and anti-backflow in canonical order. Fenced examples are excluded when counting headings/contracts.

## Classification method and preservation

Existing explicit owner/type/scope/status/authority fields were retained. Legacy `depends_on` entries were checked against the actual source and named public owner interfaces before conversion; unresolved declarations are preserved in a migration-provenance section and do not become invented dependencies. Formerly headerless files were classified using their stated module/owner/purpose and existing parent interface requirements. Existing provisional, proposed and exploratory meanings remain scoped; directory placement does not confer new consumer permissions.

Additional document H1s were changed to section headings without removing their text. Pre-title path material and noncanonical header fields were preserved as migration provenance. RCP-1.1 compatibility/compliance prose is retained as legacy notes, not a current validation certificate. Header migration did not implement new projection, scoring or transport algorithms.

The final canonical records and legacy declarations are traceable through `rcmt-migration.json`, the source baseline, current files, and the exact derived registry. Later path/source-input refinements are in `followup-edits.json` and `path-edits.json`.

## Validation against RCP requirements

| Requirement | Check / result |
|---|---|
| Leading H1 and immediate contract | 165/165 pass |
| Single H1 / contract outside fenced examples | 165/165 pass |
| Eight ordered, nonempty canonical fields / allowed type | 165/165 pass |
| Anti-backflow and authority distinct from status | All have explicit boundaries; no active status substituted for authority |
| Canonical header path resolution | All declared resolved paths exist within theory; missing original inputs remain explicitly unresolved metadata |
| Dependency direction | No cycle in the current canonical header dependency graph; no downstream evidence promoted to kernel authority |
| Active inline Markdown file references | No missing or ambiguous references under document-relative lookup and reviewed legacy-root conventions, excluding intentionally preserved legacy-provenance text/fenced examples |
| Scope/body semantic review | Approved ownership, compactness, parameter, scoring, projection and indexing decisions propagated; proposed SMG policies remain provisional; numerical/physical evidence not promoted |
| Full readiness | **Not certified for the entire corpus:** four source documents retain unresolved upstream metadata, and the aggregate registry propagates that limitation |

This is more than field-presence checking: it includes placement, multiplicity, field/type rules, dependency resolution/cycles and an explicit owner/scope/evidence review. It is not an automated proof that every body assertion is correct. In particular, a canonical header does not resolve missing sources, establish a mathematical theorem or assign an ambiguous deployment-stage permission.

## Unresolved source metadata

'''
for r in rcp['unresolved']:
    if r['path'].endswith('rcmt_compressed.md'):continue
    match=next(x for x in migration if x['path']==r['path'])
    text+='- @{'+r['path']+'#1}: '+ '; '.join(x['legacy']+' — '+x['reason'] for x in match['unresolved'])+'.\n'
text+='''
The exact direction-bearing upstream contract has not been recovered. The declared Echoon Minimal Closure sources do not resolve in the authorized input layout. No replacement owner, API or implementation was invented. These four files say metadata is unresolved/not compression-ready; the generated RCMT registry also remains not compression-ready. Of 165 final documents, 160 pass the header/dependency readiness screen, four source documents are withheld, and one generated aggregate carries their limitation. This is a readiness screen, not a universal semantic-compliance badge.

## Derived artifacts

`rcf/rcf_compressed.md` indexes 194 final RCF source documents. `rcmt/rcmt_compressed.md` indexes 164 final RCMT source documents, excluding itself. Both carry exact header registries, conservative summaries and reproducible source manifests containing base commit plus working-tree file hashes. They are subordinate indexes, not replacements for source-body mathematics. Source hashes were checked after the final edits. No claim that the base commit alone describes uncommitted changes is made.

Evidence: @{projects/ops-codex/implementation/rcp-validation.json#1}, @{projects/ops-codex/implementation/body-path-review.json#1}, @{projects/ops-codex/implementation/final-verification.json#1}.
'''
reports['01-rcp-migration-report.md']=('RCP-1.2 migration coverage and exceptions',text)

text='''## Runtime boundary

No long-running simulation, global 2^24 enumeration, full SIM15 kernel/normalizer census or full historical suite was run. Only eight scripts with prior measured short runtimes were dispatched, each with a hard ten-second timeout; all finished in under four seconds. Static parsing covers all 21 repaired OPS scripts. The later user runtime restriction limits the broader D11 execute instruction; skipped runs are not represented as successes.

## OPS defects and actual repair

All 21 raw `.py` files originally mixed code with unquoted recorded output; five also lacked an opening docstring delimiter. Clean executable versions now occupy the original script paths. Their calculations were preserved exactly from the documented code portions; no equation, group definition, tolerance or gate was changed to match history. Full original mixed files are archived byte-for-byte under `implementation/before/projects/Operational_Phase_Space/`. Historical output is now separate in `Operational_Phase_Space/recorded_output/`.

`ARCHIVE_MANIFEST.json` records each original SHA-256, archive location, retained line interval, delimiter repair, clean-script hash and recorded-output hash. All 21 parse, and their archive/current hashes were verified. No source artifact was deleted.

### Reproduction and historical comparison

| Repaired script | Seconds | Exit | Historical comparison |
|---|---:|---:|---|
'''
for row in runs:
    if 'exit_code' not in row:continue
    if row['historical_packet_present']:
        desc='Machine JSON truth packet exactly matches; display residuals/layout may differ'
    elif row['script']=='SIM14.8.py':
        desc='Historical display has concatenated headings/lines; 104 of 116 normalized lines recur exactly; remaining layout/text differences retained for review; no full byte-equality claim'
    else:desc='All substantive normalized historical lines recur; only result/process wrappers differ'
    text+=f'| {row["script"]} | {row["seconds"]:.2f} | {row["exit_code"]} | {desc} |\n'
text+='''
Four available historical JSON packets match exactly: SIM16.0, 16.1, 16.3 and 16.4. Their printed floating-point residuals vary at approximately machine precision in some witness lines, without changing assertion outcomes or truth packets. Historical wrapper markers and some merged headings differ. Raw old/new output and every unmatched normalized historical line are preserved in `ops-repair-verification.json` and per-run stdout/stderr files. No byte-identical stdout claim is made, particularly for SIM14.8.

Thirteen scripts were not run: SIM14.0/14.1/14.2/14.4/14.6/14.7, SIM15.0–15.5, and SIM16.2. Some have prior onboarding evidence, but this report does not count it as a new run. All thirteen have static executability checks only in this pass.

## Lisi coordinates: correction and provenance

The original `120cell_vertices.txt` remains byte-identical. It had duplicate coordinate keys and graph-ID alignment defects, including severe corruptions at **327, 552 and 578**. A separate active numerical realization is supplied as @{projects/lisi-smg-test/reference_geometry/120cell_vertices.repaired.txt#1}; usage and limitations are in @{projects/lisi-smg-test/reference_geometry/COORDINATE_REPAIR.md#1}.

The repair uses an existing native construction rather than inventing rows:

1. Extract the unchanged `parity` and `roots_h4` functions from the preserved SIM15.6 code portion. Construct 120 native H4 roots and their degree-12 nearest-neighbor graph.
2. Enumerate its 600 tetrahedra and 1,200 shared triangular faces; construct the tetrahedral dual with degree four.
3. Obtain a discrete graph-isomorphism witness to the supplied 600-vertex/1,200-edge 120-cell adjacency. Compare the complete mapped edge sets exactly.
4. Use tetrahedral centroids as the dual coordinates. Recover a common orthogonal frame and scale from the intact supplied rows. An initial untrimmed registration is biased by corruptions; a documented residual-based rigid refit excludes the three severe rows without deforming the polytope.
5. Regenerate all 600 rows coherently in that frame, rather than silently retain small inconsistencies in the approximate original table. Validate the full nearest-neighbor edge set under original source IDs, uniqueness, uniform radius and antipodes.

The maximum discrepancy on the 597 intact old rows is '''+f'{coord["intact_source_max_frame_error"]:.12g}'+'''. The output has 600 unique four-dimensional coordinates, uniform radius, all 1,200 target edges and 300 antipodal pairs. The discrete isomorphism is exact; the coordinate frame/values are numerical and rounded to ten decimals, not symbolic exact arithmetic. Original orientation/scale is recovered within the supplied table's numerical accuracy, not asserted as a uniquely recorded exact original frame.

Every output ID has a source tetrahedron in @{projects/lisi-smg-test/artifacts/coordinate_repair.json#1}, together with the native roots, hashes, rotation/scale, severe-row before/after values and method. That certificate specifies the generation method for every regenerated row. It is independently checked by the new tests, including reconstruction from centroid/transform data and full edge-set comparison.

### Newly identified legacy-generator limitation

The existing `parse_120cell_adjacency_v2.py` generator returns 600 points but its nearest-neighbor degrees range from two to four in the bounded diagnostic. Its phi-weighted adjacency is a separate historical proposal. It was rejected as correction authority and was not modified to force a regular graph. This generator defect remains unresolved and is documented; use the new graph-certified realization explicitly. No physical implication follows from the repair.

## Tests executed

The five new coordinate-certificate tests plus the existing Phase 6 comparison, Phase 6B refinement and Phase 6D local suites passed: **21 passed in 0.94 seconds**. The new checks cover original hash preservation, 600 unique uniform-radius rows, complete nearest-neighbor edge equality, 300 antipodal pairs, independent tetrahedral-dual certificate reconstruction, and source-frame/outlier agreement.

The earlier expensive geometry/configuration suites were not rerun. Frozen SMG and original adjacency/coordinate/PDF inputs, all pre-existing Lisi artifact JSON and pre-existing caches were hash-verified unchanged. Changed/new files are exhaustively listed in the change ledger.

Evidence: @{projects/ops-codex/implementation/ops-repair-verification.json#1}, @{projects/Operational_Phase_Space/ARCHIVE_MANIFEST.json#1}, @{projects/ops-codex/implementation/coordinate-repair-result.json#1}, @{projects/ops-codex/implementation/bounded-tests.txt#1}, @{projects/lisi-smg-test/tests/test_coordinate_repair.py#1}.
'''
reports['02-research-repair-report.md']=('Research artifact repairs and bounded verification',text)

reports['03-final-consistency-report.md']=('Final consistency status and remaining limits',r'''
## Outcome

The approved cleanup has been applied to active documentation and supporting artifacts. The RCMT corpus is structurally migrated, OPS scripts are executable with original history preserved, and a new graph-verified coordinate realization is available. The unresolved items below remain explicit; this is not a claim that all research or every compression-readiness condition is resolved.

## Resolved and preserved boundaries

| Decision | Final state |
|---|---|
| D1 | RCMT's symplectic/geometric lanes remain mandatory adopted architecture; RCF profile remains optional; no necessity theorem invented |
| D2 | Sp(8,R) is an active candidate, not a finalized realization or physical validation; product/carrier/Lorentz questions remain open |
| D3 | Construction-facing versus effective-output permissions distinguished; no union of permissions or directory-based grant |
| D4 | Universal compactness removed; adopted-profile restrictions and independent locality/finite/global-curvature constraints preserved |
| D5 | κ density, bounded γ memory and historical gamma separated; no conversion supplied |
| D6 | Historical values and formulas preserved, original numerical table unchanged; provenance gap and formula discrepancy explicit |
| D7 | Π↓ is a compatibility/loss diagnostic, not an inverse/feedback controller; triangle requirements scoped to explicit downstream profiles; transport data declared rather than invented |
| D8 | LCO evaluates supplied candidates; lowest among those candidates does not mean lower than prior mismatch; selection/mutation/policy restrictions remain separate; no general convergence/descent guarantee |
| D9 | Historical 12 pair indices × three target-driven regime labels retained without an independent generation derivation; precise offset meaning unresolved; SMG adopter policies retained; routing extension provisional |
| D10 | RCF protocol corrected; 163 original RCMT sources plus new status/derived files have canonical headers; exact registries/manifests regenerated after source changes |
| D11 | 21 repaired OPS scripts with archives/results split; eight short reruns; corrected coordinate artifact and certificate/tests; long runs skipped |

## Validation evidence

- 165/165 RCMT Markdown files pass canonical structure, heading/contract multiplicity, field ordering/value/type and resolved-header-path checks.
- Current header dependency graph has no cycles. Reviewed active inline Markdown paths have no missing or ambiguous targets; preserved migration-provenance strings are historical records, not repaired or fabricated inputs.
- Four source documents remain explicitly not compression-ready because of unresolved metadata; the generated RCMT aggregate propagates their limitation. 160 pass the header/dependency readiness screen, which is not blanket semantic certification.
- All 21 OPS files parse. Eight measured short runs exited zero, each below four seconds and a ten-second hard limit. Four historical machine truth packets match exactly. Thirteen runs remain skipped.
- Five coordinate repair checks plus sixteen selected comparison/refinement/local tests passed, 21 total in 0.94 seconds. The coordinate certificate preserves all 1,200 source-ID graph edges and 300 antipodal pairs.
- Original Lisi coordinate/adjacency/PDF/SMG-reference inputs and every pre-existing artifact JSON are unchanged; pre-existing caches are unchanged. Historical numerical table text is unchanged.
- Derived source manifests were checked against final working-tree bytes. Commit SHAs describe the base revisions; working-tree manifests describe the changed sources. No commit was created.

Machine evidence: @{projects/ops-codex/implementation/final-verification.json#1}. Full file hashes/decisions: @{projects/ops-codex/implementation/change-ledger.json#1}.

## Outstanding metadata and research limits

1. **Direction-bearing upstream contract:** `rcmt/local/constraints/direction_semantics.md` still names an upstream direction invariant without an identified exact source. Its declaration is preserved and marked unresolved; no upstream invariant was invented.
2. **Echoon source references:** `rcmt/modules/geometry_behavior/README.md`, `echoon_field.md` and `closure_reinjection_cycles.md` refer to declared Echoon Minimal Closure sources that do not resolve in the authorized layout. Their unresolved status is preserved, including in the generated registry. Supply or identify the intended source before claiming full compression readiness for them.
3. **Individual consumer-stage assignments:** the interface responsibilities are approved and documented, but deployment responsibility for ambiguous module uses is not inferred from folders. In particular, the group/kinematics/dynamics/discrete-field/time/continuous-field/metric/amplitude/conversion families must use their explicitly declared producer/consumer boundary; no new module API/ownership assignment was made here. Future deployments must state which responsibility they exercise rather than combine permissions.
4. **Historical density provenance and offset meaning:** original runs, coefficient/parameter relationships and precise offset interpretation remain unavailable or unverified. No reproduction study, rescaling or new generation mechanism was initiated.
5. **Legacy coordinate generator:** the old generator's degree-two-to-four nearest-neighbor output remains a documented defect. The corrected realization uses the independently graph-certified native-root/dual construction; original legacy code is preserved.
6. **Skipped reproductions:** thirteen OPS scripts, the full SIM15 family census and global configuration enumeration were not run. Their static repairs are complete, but their new reproduction is not claimed. Coordinate/selected test success does not validate physical interpretations or the original Lisi paper's full spinorial phases.
7. **Architectural realization questions:** group representation, lane-independence criterion, Lorentz-compatible map, operational observability and dynamics remain research questions. The cleanup does not solve them or choose new commitments.

Items 1–3 need source identification or explicit deployment metadata, not a rerun of expensive simulations. No already approved D1–D11 decision is reopened by these limitations.

## History, review and next boundary

The previous onboarding and documentation-sweep reports remain dated audit/proposal records. Their initial source-line references describe the pre-maintenance state; matching snapshots are in `implementation/before/`. The approved owner register supersedes conflicting prior recommendations, especially the earlier compactness and general-descent proposals. Current source files and this implementation report describe the resulting state.

All changes remain reviewable as working-tree modifications and explicit unversioned artifact manifests. The change ledger records 244 paths, their decisions, reasons and before/after hashes. No source deletion, theory redesign, long-running simulation or external publication was performed. Stop point: approved maintenance executed and verified within the stated limits; missing metadata, legacy-generator defects and skipped reproductions remain openly reported.
''')

def linkrefs(s):
    return re.sub(r'@\{([^#}]+)#(\d+)\}',lambda m:f'[{m[1]}:{m[2]}]({(ROOT/m[1]).as_posix()}:{m[2]})',s)
for filename,(title,body) in reports.items():
    header=f'''# {title}

## Document Contract

- owner: Codex approved maintenance record
- type: Observation
- scope: owner-approved D1–D11 cleanup and bounded verification, 2026-10-24
- authority: implementation report; no scientific authority over the cited source contracts
- upstream dependencies: approved-owner-register.txt, baseline.json and cited execution/source records
- status: executed with explicit unresolved metadata and skipped reproductions
- purpose: record actual changes, validation and remaining limitations
- anti-backflow: maintenance evidence does not invent upstream invariants, missing sources, mathematical necessity or physical validation

'''
    (HERE/filename).write_text(header+linkrefs(body.strip())+'\n',encoding='utf-8')
# Verify each rendered source citation points to an existing, in-range target.
count=0
for filename in reports:
    body=(HERE/filename).read_text(encoding='utf-8')
    assert '@{' not in body
    for target,line in re.findall(r'\]\((C:/[^)]+):(\d+)\)',body):
        p=Path(target);assert p.is_file(),target
        assert int(line)<=len(p.read_text(encoding='utf-8-sig').splitlines()),(target,line)
        count+=1
(HERE/'report-check.json').write_text(json.dumps({'reports':4,'valid_citations':count,'ledger_paths':len(delta)},indent=2))
print('Reports:',len(reports),'valid citations:',count,'ledger entries:',len(delta))

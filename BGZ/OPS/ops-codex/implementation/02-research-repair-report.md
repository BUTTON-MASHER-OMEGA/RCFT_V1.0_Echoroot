# Research artifact repairs and bounded verification

## Document Contract

- owner: Codex approved maintenance record
- type: Observation
- scope: owner-approved D1–D11 cleanup and bounded verification, 2026-10-24
- authority: implementation report; no scientific authority over the cited source contracts
- upstream dependencies: approved-owner-register.txt, baseline.json and cited execution/source records
- status: executed with explicit unresolved metadata and skipped reproductions
- purpose: record actual changes, validation and remaining limitations
- anti-backflow: maintenance evidence does not invent upstream invariants, missing sources, mathematical necessity or physical validation

## Runtime boundary

No long-running simulation, global 2^24 enumeration, full SIM15 kernel/normalizer census or full historical suite was run. Only eight scripts with prior measured short runtimes were dispatched, each with a hard ten-second timeout; all finished in under four seconds. Static parsing covers all 21 repaired OPS scripts. The later user runtime restriction limits the broader D11 execute instruction; skipped runs are not represented as successes.

## OPS defects and actual repair

All 21 raw `.py` files originally mixed code with unquoted recorded output; five also lacked an opening docstring delimiter. Clean executable versions now occupy the original script paths. Their calculations were preserved exactly from the documented code portions; no equation, group definition, tolerance or gate was changed to match history. Full original mixed files are archived byte-for-byte under `implementation/before/projects/Operational_Phase_Space/`. Historical output is now separate in `Operational_Phase_Space/recorded_output/`.

`ARCHIVE_MANIFEST.json` records each original SHA-256, archive location, retained line interval, delimiter repair, clean-script hash and recorded-output hash. All 21 parse, and their archive/current hashes were verified. No source artifact was deleted.

### Reproduction and historical comparison

| Repaired script | Seconds | Exit | Historical comparison |
|---|---:|---:|---|
| SIM14.3.py | 0.08 | 0 | All substantive normalized historical lines recur; only result/process wrappers differ |
| SIM14.5.py | 2.19 | 0 | All substantive normalized historical lines recur; only result/process wrappers differ |
| SIM14.8.py | 1.00 | 0 | Historical display has concatenated headings/lines; 104 of 116 normalized lines recur exactly; remaining layout/text differences retained for review; no full byte-equality claim |
| SIM15.6.py | 1.16 | 0 | All substantive normalized historical lines recur; only result/process wrappers differ |
| SIM16.0.py | 1.53 | 0 | Machine JSON truth packet exactly matches; display residuals/layout may differ |
| SIM16.1.py | 3.28 | 0 | Machine JSON truth packet exactly matches; display residuals/layout may differ |
| SIM16.3.py | 2.14 | 0 | Machine JSON truth packet exactly matches; display residuals/layout may differ |
| SIM16.4.py | 3.12 | 0 | Machine JSON truth packet exactly matches; display residuals/layout may differ |

Four available historical JSON packets match exactly: SIM16.0, 16.1, 16.3 and 16.4. Their printed floating-point residuals vary at approximately machine precision in some witness lines, without changing assertion outcomes or truth packets. Historical wrapper markers and some merged headings differ. Raw old/new output and every unmatched normalized historical line are preserved in `ops-repair-verification.json` and per-run stdout/stderr files. No byte-identical stdout claim is made, particularly for SIM14.8.

Thirteen scripts were not run: SIM14.0/14.1/14.2/14.4/14.6/14.7, SIM15.0–15.5, and SIM16.2. Some have prior onboarding evidence, but this report does not count it as a new run. All thirteen have static executability checks only in this pass.

## Lisi coordinates: correction and provenance

The original `120cell_vertices.txt` remains byte-identical. It had duplicate coordinate keys and graph-ID alignment defects, including severe corruptions at **327, 552 and 578**. A separate active numerical realization is supplied as [projects/lisi-smg-test/reference_geometry/120cell_vertices.repaired.txt:1](C:/Users/blaur/projects/theory/projects/lisi-smg-test/reference_geometry/120cell_vertices.repaired.txt:1); usage and limitations are in [projects/lisi-smg-test/reference_geometry/COORDINATE_REPAIR.md:1](C:/Users/blaur/projects/theory/projects/lisi-smg-test/reference_geometry/COORDINATE_REPAIR.md:1).

The repair uses an existing native construction rather than inventing rows:

1. Extract the unchanged `parity` and `roots_h4` functions from the preserved SIM15.6 code portion. Construct 120 native H4 roots and their degree-12 nearest-neighbor graph.
2. Enumerate its 600 tetrahedra and 1,200 shared triangular faces; construct the tetrahedral dual with degree four.
3. Obtain a discrete graph-isomorphism witness to the supplied 600-vertex/1,200-edge 120-cell adjacency. Compare the complete mapped edge sets exactly.
4. Use tetrahedral centroids as the dual coordinates. Recover a common orthogonal frame and scale from the intact supplied rows. An initial untrimmed registration is biased by corruptions; a documented residual-based rigid refit excludes the three severe rows without deforming the polytope.
5. Regenerate all 600 rows coherently in that frame, rather than silently retain small inconsistencies in the approximate original table. Validate the full nearest-neighbor edge set under original source IDs, uniqueness, uniform radius and antipodes.

The maximum discrepancy on the 597 intact old rows is 9.9458411586e-05. The output has 600 unique four-dimensional coordinates, uniform radius, all 1,200 target edges and 300 antipodal pairs. The discrete isomorphism is exact; the coordinate frame/values are numerical and rounded to ten decimals, not symbolic exact arithmetic. Original orientation/scale is recovered within the supplied table's numerical accuracy, not asserted as a uniquely recorded exact original frame.

Every output ID has a source tetrahedron in [projects/lisi-smg-test/artifacts/coordinate_repair.json:1](C:/Users/blaur/projects/theory/projects/lisi-smg-test/artifacts/coordinate_repair.json:1), together with the native roots, hashes, rotation/scale, severe-row before/after values and method. That certificate specifies the generation method for every regenerated row. It is independently checked by the new tests, including reconstruction from centroid/transform data and full edge-set comparison.

### Newly identified legacy-generator limitation

The existing `parse_120cell_adjacency_v2.py` generator returns 600 points but its nearest-neighbor degrees range from two to four in the bounded diagnostic. Its phi-weighted adjacency is a separate historical proposal. It was rejected as correction authority and was not modified to force a regular graph. This generator defect remains unresolved and is documented; use the new graph-certified realization explicitly. No physical implication follows from the repair.

## Tests executed

The five new coordinate-certificate tests plus the existing Phase 6 comparison, Phase 6B refinement and Phase 6D local suites passed: **21 passed in 0.94 seconds**. The new checks cover original hash preservation, 600 unique uniform-radius rows, complete nearest-neighbor edge equality, 300 antipodal pairs, independent tetrahedral-dual certificate reconstruction, and source-frame/outlier agreement.

The earlier expensive geometry/configuration suites were not rerun. Frozen SMG and original adjacency/coordinate/PDF inputs, all pre-existing Lisi artifact JSON and pre-existing caches were hash-verified unchanged. Changed/new files are exhaustively listed in the change ledger.

Evidence: [projects/ops-codex/implementation/ops-repair-verification.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/implementation/ops-repair-verification.json:1), [projects/Operational_Phase_Space/ARCHIVE_MANIFEST.json:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/ARCHIVE_MANIFEST.json:1), [projects/ops-codex/implementation/coordinate-repair-result.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/implementation/coordinate-repair-result.json:1), [projects/ops-codex/implementation/bounded-tests.txt:1](C:/Users/blaur/projects/theory/projects/ops-codex/implementation/bounded-tests.txt:1), [projects/lisi-smg-test/tests/test_coordinate_repair.py:1](C:/Users/blaur/projects/theory/projects/lisi-smg-test/tests/test_coordinate_repair.py:1).

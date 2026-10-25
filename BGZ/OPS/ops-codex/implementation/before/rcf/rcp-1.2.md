# RCF Compression Protocol (RCP-1.2)

## Document Contract

- owner: RCF Compression Protocol
- type: Contract
- scope: compression-ready RCF and downstream project documents
- authority: normative protocol revision
- upstream dependencies: `rcp-1.1.md`
- status: active
- purpose: require a canonical document header that preserves ownership, authority, dependency direction, and anti-backflow constraints during compression and rehydration
- anti-backflow: this protocol classifies document authority but does not grant downstream documents authority over upstream contracts

### Typed English for Canon-Safe Compression, Validation, and Fast Model Rehydration

---

## Purpose

RCP-1.2 extends RCP-1.1 by requiring a canonical **Document Contract** at the top of every Markdown document participating in RCF or downstream project compression.

The header makes each document self-identifying before its body is interpreted. A model must be able to determine the document's owner, statement class, scope, authority, dependencies, maturity, purpose, and backflow boundary without inferring them from narrative prose.

All RCP-1.1 statement types, ownership rules, directionality rules, compression passes, normal forms, and validator directives remain in force unless this revision explicitly narrows them.

---

## Canonical Document Header

Every Markdown file must begin with the following structure and field order:

```markdown
# Title

## Document Contract

- owner:
- type:
- scope:
- authority:
- upstream dependencies:
- status:
- purpose:
- anti-backflow:
```

The labels and order are canonical.

Values must be explicit and non-empty. Use `none` when a field has no applicable value; do not leave the value blank in a compression-ready document.

---

## Placement Rules

The Document Contract must:

- appear before all substantive document content;
- follow the document's single H1 title;
- occur exactly once;
- use the canonical field names and order;
- remain visible in human-readable and compression-ready sources;
- survive compression in full or in an equivalent lossless canonical record.

The H1 title is the first line of the document. RCP-1.2 uses no leading delimiter or YAML front matter.

Subtitles, summaries, introductions, and navigation content must appear after the Document Contract.

Validators must ignore header-like text inside fenced examples when counting titles, contracts, or fields.

---

## Field Semantics

### `owner`

Names the module, layer, project, or profile allowed to make the document's claims.

The owner must not be inferred from the directory name alone.

### `type`

Names the document's primary RCP statement type.

Allowed values:

- `Invariant`
- `Contract`
- `Observation`
- `Dependency`
- `Permission`
- `Prohibition`
- `Interpretation`
- `Status`

A mixed document must choose the strongest load-bearing type and explicitly classify weaker sections in its body.

### `scope`

States where the document's claims apply.

Preferred forms include:

- `core-canonical`
- `current formalism`
- `current evaluations`
- `module-local contract`
- `downstream interpretation`
- a narrower explicit project or profile scope

### `authority`

States the source and rank of the document's authority.

Examples:

- `RCF invariant kernel`
- `module-local contract`
- `downstream profile`
- `current evaluation record`
- `noncanonical interpretation`

The authority field records existing authority. It cannot promote a document beyond the authority granted by its owner and upstream contracts.

### `upstream dependencies`

Lists every load-bearing upstream document, module, interface, or compressed shard consumed by the file.

Dependencies must:

- be directional;
- be resolvable when paths are available;
- exclude downstream consumers;
- use `none` for a root document with no upstream dependency.

### `status`

States maturity and revision state.

Examples:

- `active`
- `draft`
- `provisional`
- `experimental`
- `deprecated`
- `superseded`
- `compression-ready`

Status does not alter authority.

### `purpose`

Provides one concise sentence describing why the document exists and what it contributes.

Purpose must not introduce claims absent from the document body.

### `anti-backflow`

States the document's one-way ownership boundary.

It must identify what the document cannot redefine, widen, repair, reinterpret, or promote upstream.

`none` is allowed only for a root authority document that has no upstream layer.

---

## Authority and Dependency Distinction

`authority` and `upstream dependencies` are not interchangeable.

- `authority` answers: **why may this document make these claims?**
- `upstream dependencies` answers: **what prior contracts or records does this document consume?**

A document may depend on an upstream source without inheriting all of that source's authority.

A downstream interpretation may cite a canonical upstream dependency while retaining only downstream interpretive authority.

---

## Header Precedence

The Document Contract constrains interpretation of the entire file.

If body prose conflicts with the header:

1. upstream contracts remain authoritative;
2. the Document Contract bounds the document;
3. conflicting body prose fails validation and must be rewritten or removed.

The header cannot legalize body content forbidden upstream.

---

## Compression Behavior

Compression must preserve the Document Contract as a canonical document record.

At minimum, the compressed form must retain:

```text
document {
  title
  owner
  type
  scope
  authority
  upstream_dependencies
  status
  purpose
  anti_backflow
}
```

Field values may be normalized for compactness, but no field may be omitted or inferred during rehydration.

Repeated narrative statements may be removed when the Document Contract preserves their complete ownership and directionality content.

---

## Rehydration Rule

During model rehydration, load the Document Contract before the document body.

The model must use the contract to determine:

- whether the body is canonical or interpretive;
- which claims the owner may make;
- which upstream documents must already be loaded;
- whether the document may constrain downstream modules;
- and which inferences are prohibited from traveling upstream.

No body statement may widen its own header during rehydration.

---

## Validator Requirements

A document fails RCP-1.2 validation when:

- the leading header sentinel is absent;
- the H1 title is absent or duplicated;
- the Document Contract is absent or duplicated;
- a canonical field is missing, reordered, renamed, or blank;
- `type` is not an allowed RCP statement type;
- authority exceeds owner or upstream authority;
- an upstream dependency points downstream;
- a declared path does not resolve in the validation environment;
- status is used to imply authority;
- purpose introduces unsupported content;
- anti-backflow is absent, vague, or contradicted by the body;
- body prose exceeds the scope or authority declared in the header.

Validators should report header failures before evaluating body-level compression passes.

---

## Migration Rule

For documents created before RCP-1.2:

1. preserve the existing H1 title;
2. insert the canonical Document Contract immediately after it;
3. classify the existing document without strengthening its claims;
4. identify actual upstream dependencies;
5. state authority separately from ownership;
6. add an explicit anti-backflow boundary;
7. validate the body against the new header.

Adding a Document Contract is a classification operation. It must not silently revise the document's theory content.

---

## RCP-1.1 Compatibility

RCP-1.2 retains the RCP-1.1:

- statement type system;
- ownership rule;
- directionality rule;
- scope rule;
- modal verb discipline;
- twelve compression passes;
- module compression normal form;
- canonical sentence patterns;
- output standard;
- validator directive;
- and meta principle.

Where RCP-1.1 says a document *should* make ownership, type, or scope classifiable, RCP-1.2 requires those properties to be declared in the Document Contract.

---

## Status

- protocol: RCF Compression Protocol
- version: RCP-1.2
- predecessor: RCP-1.1
- revision type: normative document-header extension
- state: active

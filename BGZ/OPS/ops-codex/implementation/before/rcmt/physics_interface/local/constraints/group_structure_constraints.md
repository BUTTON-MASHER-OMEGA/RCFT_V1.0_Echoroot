# Group Structures Constraints

This file defines how physics modules may consume the structural labels provided
by `foundation/abstract/group_structures`.

Physics modules may use:

- `group-state-label`
- `projection-operator-label`
- `continuous-limit-map`
- `phase-orientation-band`

Physics modules must not:

- reinterpret these labels as physical operators,
- assign intrinsic domain semantics to the labels themselves,
- introduce new invariants upstream,
- redefine admissibility rules,
- or modify the structural meaning of any label.

These labels act as structural hooks only. Physical interpretation may be introduced in downstream modules, which must remain compatible with the abstract layer.

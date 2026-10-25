# group_structures

This module provides the **physics-facing interpretation layer** for the upstream `rcf/modules/group_structures` module.

It does **not** define physics, operators, fields, or dynamics.

Instead, it:

- interprets abstract group structure labels in a way that is usable by physics modules,
- specifies what physics modules **may** assume about these labels,
- specifies what physics modules **may not** assume,
- preserves the upstream invariants and constraints defined in `rcf/modules/group_structures`,
- acts as a firewall to prevent physics semantics from leaking upstream into RCF core.

This module is part of the `physics_interface/local` layer and is intended to be:

- **downstream** of `rcf/modules/group_structures/`,
- **upstream** of physics modules that consume group structure labels,
- **replaceable** by custom physics projects, as long as upstream contracts are respected.

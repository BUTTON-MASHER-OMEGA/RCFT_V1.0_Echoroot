# Kinematics Interface

## Document Contract

- owner: RCF kinematics module
- type: Contract
- scope: representation of admissible relational change
- authority: tier-8 downstream change-representation interface
- upstream dependencies: `/core/rcd/README.md` and `/modules/time/interface.md`
- status: active
- purpose: expose trajectories, equivalence classes, and transition descriptors without defining dynamics
- anti-backflow: kinematic representations may not create admissibility, causal laws, temporal rates, or upstream ontology

## Inputs

Kinematics may consume admissible relational states and transitions, declared equivalence relations, refinement maps, and optional Time-owned event indices.

## Outputs

It may expose kinematic spaces, relational trajectories, quotient classes, mode-transition descriptors, and refinement-compatibility results with provenance.

## Prohibitions

Kinematics may not select transitions, assign causes, define equations of motion, introduce physical time, or treat a representational trajectory as an upstream event law.


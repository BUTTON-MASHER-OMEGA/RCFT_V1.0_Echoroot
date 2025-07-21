
---  

**File: docs/chapter_10_qft_coherence.md**  
```markdown
# Chapter 10 – Quantum Field Theoretic Coherence

## Description
Introduces path-integral formalism for coherence fields, derives two-point correlation functions and propagator structure.

## Key Equations
```math
Z[J] = \int Dφ\,e^{iS[φ] + i\int Jφ}  
G₂(x,y) = \langle φ(x)\,φ(y)\rangle

## Mathematical Findings
Gaussian integral evaluation for Z[J]

Propagator poles and shard quasiparticles

## Topics
Functional integral techniques

Shard propagators in momentum space

## Research
Compute 2-point functions for common shard actions

## Visualizations
Feynman-style diagrams of shard exchange

## Indexes
Equation Index: Z[J], G₂

Figure Index: 10.1

number: 10
    code_snippets:
      - name: compute_two_point_function
        file: rcft_lib/chapter10.py
        function: compute_two_point(phi_grid, action)
        description: Metropolis sampling to approximate G₂(x,y)
      - name: metropolis_sampler
        file: rcft_lib/chapter10.py
        function: metropolis_update(phi_grid, beta)
        description: Update function for Metropolis algorithm in coherence path integral
    extra_equations:
      - lattice_corrections: "G₂^L(x) = G₂(x) + O(a²)"
    field_tests:
      - name: FPGA Propagator Benchmark
        description: Hardware-accelerated shard propagator evaluation compared to Python baseline
    visualizations:
      - name: G₂ vs Distance Plot
        notebook: notebooks/chapter10/two_point_plot.ipynb

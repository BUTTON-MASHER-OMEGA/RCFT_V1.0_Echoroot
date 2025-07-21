
---  

**File: docs/chapter_7_thermo_fusion.md**  
```markdown
# Chapter 7 – Shard Fusion & Thermodynamics

## Description
Frames shard coalescence as a thermodynamic process, computes partition functions and free-energy landscapes.

## Key Equations
```math
Z = \sum e^{-βE}  
F = -β^{-1} \log Z

## Mathematical Findings
Thermodynamic potentials for shard ensembles

Fusion-rate estimates via Boltzmann weights

## Topics
Partition functions

Free energy in coherence systems

## Research
Statistical distribution of shard energy levels

## Visualizations
Free energy F vs. temperature T

## Indexes
Equation Index: Z, F

Figure Index: 7.1

code_snippets:
      - name: partition_function_mc
        file: rcft_lib/chapter7.py
        function: partition_function(energies, beta_values)
        description: Monte Carlo estimation of Z(β) = ∑ e^{-β E}
      - name: free_energy_sweep
        file: rcft_lib/chapter7.py
        function: free_energy(energies, beta_values)
        description: Computes F(β) = -β^{-1} log Z
    extra_equations:
      - heat_capacity_relation: "C(β) = ∂²F/∂β²"
    field_tests:
      - name: Cellular Automaton Assembly
        description: Automaton-based simulation of shard coalescence measuring empirical fusion rates
    visualizations:
      - name: Z(β) & F(β) Plot
        notebook: notebooks/chapter7/partition_free_energy.ipynb


---  

**File: docs/chapter_8_hyperbolic_tessellations.md**  
```markdown
# Chapter 8 – Hyperbolic Geometry & Tessellations

## Description
Maps coherence cells onto hyperbolic tessellations, computes cell areas and geodesic decay rates in the Poincaré disk.

## Key Equations
```math
A = \pi\bigl(1 - \tfrac{2}{p} - \tfrac{2}{q}\bigr)

## Mathematical Findings
{7,3} tessellation area formula

Radial decay rate of geodesic flows

Computed geodesic decay exponent λ for {p,q} beyond (7,3), extended to (8,3), (9,4)

Linked hyperbolic area growth to shard-field curvature via Gauss–Bonnet

## Topics
Poincaré disk model

Coherence cell structures

## Research
Derivation of hyperbolic cell areas for shard networks

Visualizations
{7,3} tessellation diagram

## Indexes
Figure Index: 8.1, 8.2

code_snippets:
      - name: generate_hyperbolic_tessellation
        file: rcft_lib/chapter8.py
        function: generate_tessellation(p, q, depth)
        description: Generates node and edge lists for {p,q} tessellations
      - name: export_tessellation_json
        file: rcft_lib/chapter8.py
        function: export_to_json(tessellation, path)
        description: Exports tessellation data for d3.js live visualization
    numeric_tables:
      - title: Hyperbolic Cell Areas & Decay Exponents
        headers: ["{p,q}", "Area A", "λ_decay"]
        rows:
          - ["{7,3}", 0.415, 0.18]
          - ["{8,3}", 0.588, 0.22]
    field_tests:
      - name: Laser-Etched Tiling
        description: Fabricated hyperbolic tiling on acrylic, measured light-guide decay rates
    visualizations:
      - name: Tessellation Diagram
        notebook: notebooks/chapter8/tessellation_plot.ipynb

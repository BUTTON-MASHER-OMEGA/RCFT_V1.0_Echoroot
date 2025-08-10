
---  

**File: docs/chapter_37_spherical_harmonics.md**  
```markdown
Chapter 37 – Spherical Harmonics

Description
Expands shard fields on S² via spherical harmonics, proving orthogonality relations and mode decompositions.

Key Equations
```math
Y_{l,m}(θ,φ)  
\int Y^*_{l,m} Y_{l',m'}\,dΩ = δ_{ll'}\,δ_{mm'}
Mathematical Findings
Eigenfunction expansion of shard fields

Orthogonality and completeness proofs

Topics
Angular mode decomposition

Field expansions on sphere
Research
Construct basis for shard-field angular spectra

Visualizations
Spherical harmonic surface plots

Indexes
Equation Index: Spherical harmonics

Figure Index: 37.1

number: 37
    code_snippets:
      - name: compute_spherical_harmonics
        file: rcft_lib/chapter37.py
        function: spherical_harmonics_grid(l, m, grid)
        description: Generates Y_{l,m}(θ,φ) values on a meshgrid
      - name: verify_orthonormality
        file: rcft_lib/chapter37.py
        function: check_orthonormality(Y_grid, Omega)
        description: Numerically integrates Y*Y' over sphere to test orthonormality
    field_tests:
      - name: 3D-Printed Harmonic Shells
        description: Printed spherical harmonic shells to count nodal lines for validation
    visualizations:
      - name: Spherical Harmonics Surface Plot
        notebook: notebooks/chapter37/spherical_surface.ipynb

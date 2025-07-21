**File: docs/chapter_2_geometry.md**  
```markdown
# Chapter 2 – Geometric Foundations

## Description
Develops warp-product metrics to sculpt coherence fields in d₃, computes curvature profiles, and frames lens-like focusing effects.

Key Equations
```math
a(u) = e^{-u²}  
R(u) = 12 − 48 u²

code_snippets:
      - name: warp_metric_computation
        file: rcft_lib/chapter2.py
        function: warp_metric(a, u_range)
        description: Computes warp metric scale factors a(u) over a range of u values
      - name: plot_curvature_slider
        file: rcft_lib/chapter2.py
        function: plot_curvature(u_range, slider=True)
        description: Interactive Jupyter slider for curvature profile R(u)
    numeric_tables:
      - title: Warp Metric & Curvature
        headers: [u, a(u)=e^{-u^2}, R(u)]
        rows:
          - [0, 1.000, 12]
          - [0.5, 0.778, 0]
          - [1, 0.368, -36]
    test_scripts:
      - name: test_curvature_sign_change
        file: tests/test_chapter2.py
        description: Unit test verifying R(u) crosses zero at u ≈ 0.5
    field_tests:
      - name: Warp Bump Propagation
        description: Measure focal intensity of Gaussian pulse through warp bump via finite-difference solver
    visualizations:
 

Mathematical Findings
Warp-product metric with scale factor a(u) = e^{-u²}

Ricci curvature scalar R(u) = 12 − 48u² (positive at u=0, negative tails)

Coherence-lensing via localized warp “bumps”

Topics
Warp metrics in fibered spaces

Ricci curvature & focusing

Field-lensing analogy

Research
Reinforce warp curvature derivation with Penrose’s Road to Reality insights

Compare coherence-lensing to GR gravitational lensing

Visualizations
Plot of R(u) vs. u showing curvature sign-change

Gaussian pulse propagation through warp bump

     - name: Curvature vs u Plot
        notebook: notebooks/chapter2/curvature_plot.ipynb

Indexes
Equation Index: (2.1)–(2.3)

Figure Index: 2.1, 2.2

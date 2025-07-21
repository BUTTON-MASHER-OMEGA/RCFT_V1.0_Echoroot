
**File: docs/chapter_3_duality.md**  
```markdown
# Chapter 3 – Resonant Dualities

## Description
Derives Koide’s lepton-mass relation as a resonance condition in flavor space, interprets the 2/3 ratio via SU(3) invariance, and studies perturbative drift.

## Key Equations
```math
Q = \frac{m_e + m_μ + m_τ}{(\sqrt{m_e} + \sqrt{m_μ} + \sqrt{m_τ})^2} = \tfrac{2}{3}  
\cos^2 θ = \frac{1}{3Q}

## Mathematical Findings
45° vector alignment explanation of Q = 2/3

Perturbed ratio Q_ε = 2/3 + ε; angle shift θ(ε) = arccos(1/√(3Q_ε))

## Topics
Koide triad & flavor symmetry

SU(3)-invariant quadratic forms

Perturbation analysis

## Research
Link twistor-like interpretation of (√mᵢ)ᵢ to flavor spinors

Explore ε deviations as hidden-sector undulations

## Visualizations
Q vs. ε curve

Angle drift diagram: θ(ε) around 45°

## Indexes
Code Snippet: Python simulation of Q(ε)

Figure Index: 3.1, 3.2

code_snippets:
      - name: simulate_koide_distribution
        file: rcft_lib/chapter3.py
        function: simulate_koide(mu0, sigma0, trials)
        description: Samples random lepton masses and computes Q distribution under perturbations
      - name: koide_sensitivity
        file: rcft_lib/chapter3.py
        function: sensitivity_dQ_dm(m_e, m_mu, m_tau)
        description: Analytic computation of ∂Q/∂m_i for each lepton mass
    extra_equations:
      - sensitivity_expression: "∂Q/∂m_i = analytic expression in terms of (m_e, m_μ, m_τ)"
    field_tests:
      - name: Optical Fringe Ratio
        description: Physical interference experiment to measure 2/3 ratio in fringe spacing
    visualizations:
      - name: Q Distribution vs ε
        notebook: notebooks/chapter3/q_distribution.ipynb

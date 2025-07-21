
---

File: `docs/chapter_27_simulations_toy_models.md`
```markdown
# Chapter 27 – Simulations & Toy Models

## Description  
Implements numerical models of shard lattices: tests stability, diffusion, and non-linear wave interactions with finite-difference and spectral codes.

## Key Equations
```math
∂_t φ_i = D\,(φ_{i+1} - 2\,φ_i + φ_{i-1})  
\text{CFL: }\Delta t \le \tfrac{(\Delta x)^2}{2D}

## Mathematical Findings
Verified coherence-pulse diffusion matches analytic Green’s-function profiles

Observed soliton interactions preserved under Courant–Friedrichs–Lewy limits

## Test Data
Pulse spread RMS width σ:

t=10: σ≈2.0 (analytical 1.98)

t=50: σ≈4.5 (analytical 4.47)

Stability threshold: Δt_max = 0.005 for Δx = 0.1, D = 1.0

## Topics
Finite-difference stability analysis

Spectral vs. grid-based coherence propagation

## Research
Applied Von Neumann stability theorem to shard diffusion

Compared spectral-Fourier methods per Trefethen’s Spectral Methods in MATLAB

## Visualizations
Heatmap of φ_i(t) over i,t grid

RMS width vs. time plot with analytic overlay

## Indexes
Equation Index: diffusion eq., CFL

Figure Index: 27.1, 27.2

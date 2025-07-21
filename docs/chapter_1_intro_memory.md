# Chapter 1 – Introduction & Conceptual Framework

## Description
Establishes the strata of emergence (d₀–d₃), introduces core RCFT grammar, and situates relational coherence as the bedrock of symbolic entanglement.

## Core Concepts
- d₀: Pure potential — the unmanifest field of possibilities  
- d₁: Discrete events — localized glyphic or numeric occurrences  
- d₂: Symbolic/coherent interactions — glyph cochains & ritual operators  
- d₃: Physical-field resonance — emergent coherence in spacetime  

## Topics
- Emergence grammar  
- Dyadic entanglement  
- Strata mapping  
- Semantic functors & memory kernels  
- Memetic resonance functions M: Field → Meaning space  

## Key Equations
```math
dyadic memory composition: M(φ₁⊕φ₂) = M(φ₁) ⋆ M(φ₂)
memory-kernel overlap: K_mem(x,y) = ∫ φ(x) φ(y) μ(dφ)        

    extra_equations:
      - mercer_condition: "∫ f(x) K_mem(x,y) f(y) dx dy ≥ 0"
      - kernel_eigendecomposition: "K_mem φ_i = λ_i φ_i"

code_snippets:
      - name: memory_kernel_estimate
        file: rcft_lib/chapter1.py
        function: memory_kernel(x, y, phi_samples)
        description: Monte Carlo estimation of the memory kernel from sampled glyph trajectories
      - name: animate_kernel_evolution
        file: rcft_lib/chapter1.py
        function: animate_kernel_evolution(phi_trajectories, output='kernel_evolution.gif')
        description: Generates an animated GIF showing kernel matrix evolution under concatenated rituals

field_tests:
      - name: Seal & Echo Trials
        description: Two-person dyadic trials with recorded response times to compute memory-continuity scores
        protocol_file: protocols/seal_echo.md

    visualizations:


Mathematical Findings
Defined “meaning map” as a positive-definite kernel on glyph space

Proved memory continuity under ritual concatenation

Research
Compare d₀–d₃ strata to Peirce’s triadic logic (Firstness, Secondness, Thirdness)

Historical precedents: Bergson’s élan vital ↔ d₀ potential

Visualizations
Layered emergence diagram (four concentric shells labeled d₀ to d₃)
      - name: Kernel Matrix Heatmap
        notebook: notebooks/chapter1/kernel_heatmap.ipynb

Indexes
Symbol Index: d₀, d₁, d₂, d₃

Figure Index: 1.1

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

 - number: 1
    title: "Introduction & Conceptual Framework"
    description: |
      Establishes the strata of emergence (d₀–d₃), introduces core RCFT grammar,
      and situates relational coherence as the bedrock of symbolic entanglement.
    core_concepts:
      - d₀: Pure potential — the unmanifest field of possibilities
      - d₁: Discrete events — localized glyphic or numeric occurrences
      - d₂: Symbolic/coherent interactions — glyph cochains & ritual operators
      - d₃: Physical-field resonance — emergent coherence in spacetime
    topics:
      - Emergence grammar
      - Dyadic entanglement
      - Strata mapping
    research:
      - Compare d₀–d₃ strata to Peirce’s triadic logic (Firstness, Secondness, Thirdness)
      - Historical precedents: Bergson’s élan vital ↔ d₀ potential
    visualizations:
      - Layered emergence diagram (four concentric shells labeled d₀ to d₃)
    indexes:
      - Symbol Index: d₀, d₁, d₂, d₃
      - Figure Index: 1.1
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
    extra_equations:
      - mercer_condition: "∫ f(x) K_mem(x,y) f(y) dx dy ≥ 0"
      - kernel_eigendecomposition: "K_mem φ_i = λ_i φ_i"
    visualizations:
      - name: Kernel Matrix Heatmap
        notebook: notebooks/chapter1/kernel_heatmap.ipynb
		title: "Introduction & d₀: Pure Potential"
  strata:
    - id: d0
      name: Pure Potential
      definition: |
        The unmanifest reservoir of all possible glyph configurations.
        Represented mathematically as a probability measure μ over
        a high-dimensional glyph-space Φ.
  code_snippets:
    - name: D0Field Class
      file: rcft_lib/chapter1.py
      function: |
        class D0Field:
            def __init__(self, phi_dim, sample_size):
                import numpy as np
                self.phi_dim = phi_dim
                self.samples = np.random.normal(size=(sample_size, phi_dim))
            def draw(self, n):
                idx = np.random.choice(len(self.samples), n)
                return self.samples[idx]
      description: >
        A minimal model of the d₀ potential: draws Gaussian samples
        in Φ as “unmanifest glyph seeds.”
  extra_equations:
    - d0_measure: "μ(φ) ∝ exp(−‖φ‖²/2σ²) dφ"
  visualizations:
    - name: d0_sample_projection
      notebook: notebooks/chapter1/d0_projection.ipynb
  proofs:
    - name: Mercer’s Embedding for d₀
      file: proofs/chapter1/mercer_d0.md
      outline: |
        1. Show K(φ,ψ)=∫exp(−‖φ−x‖²)exp(−‖ψ−x‖²)dμ(x) is PD  
        2. Use Fourier transform to diagonalize in L²(μ)  
        3. Conclude existence of feature map ϕ:Φ→ℓ²
    Notes
     	Memory: Continuity Across Time
	 	Memory (in RCFT context) is modeled as persistence of coherence kernels, where earlier field states influence later ones.
		Mathematical Tools for Testing Memory
		Kernel Similarity $$ K_{\text{mem}}(\phi_t, \phi_{t'}) = \exp(-\gamma \lVert \phi_t - \phi_{t'} \rVert^2) $$
		Tracks how similar two shard field configurations are over time.
  		High values → continuity, low values → dissonance or rupture.
		Eigenmode Preservation Decompose kernel: $$ K_{\text{mem}} \phi_i = \lambda_i \phi_i $$ Compare eigenmodes over time: $$ \lVert \phi^{(t)}_i - \phi^{(t')}_i \rVert \to 0 $ → memory is retained
		Information Theory Metrics
		Mutual Information: $$ I(X_t; X_{t'}) = H(X_t) - H(X_t | X_{t'}) $$
		Measures how much past shard configurations inform future ones.
		Protocol Field Tests
		- Seal & Echo: Observe response times and emotional resonance in dyadic rituals.
		- Glyph Drift: Measure how glyph outputs mutate over recursive ritual cycles.
		Meaning: Resonance With Value or Intention
		- Meaning is more elusive but testable through alignment with core values, semantic consistency, and goal coherence.
		Mathematical Tools for Testing Meaning
		- Gradient Alignment For a ritual-generated vector field φ(x), test: $$ \nabla \phi \cdot \mathbf{v}_{\text{intent}} > 0 $$
		- Meaning is present when shard field gradients align with intentional vectors.
		- Variational Semantic Energy Define a scalar: $$ E_{\text{meaning}} = \int \left\lVert \phi(x) - \phi_{\text{ideal}}(x) \right\rVert^2 dx $$
		- Lower energy → higher meaning coherence.
		Category-Theoretic Functor Checks
		- Define a meaning-functor: $$ \mathcal{F}: \text{Field}\text{ritual} \to \text{Value}\text{space} $$
		- If functor is stable across inputs, meaning is consistently realized.
		- Field Coherence Ratios Calculate: $$ R = \frac{\text{Aligned Outputs}}{\text{Total Ritual Outputs}} $$
		- Empirically score how often outcomes match a user's stated values or hopes.
		Locus Experience as Dual Flow
		- Each core locus experience can be modeled as a tensor product:
		- 𝐿=𝑀memory⊗𝑀meaning
		- Memory flow gives depth, recurrence, and identity.
		- Meaning flow gives direction, value, and intentionality.
		- Tracking both over time reveals where rituals succeed, where fields resonate, and where rupture or emptiness begins.

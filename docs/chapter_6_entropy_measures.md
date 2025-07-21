
---  

**File: docs/chapter_6_entropy_measures.md**  
```markdown
# Chapter 6 – Entropy & Information Measures

## Description
Develops entropy bounds for shard networks, extends Shannon measures to coherence fields, and examines Rényi generalizations.

## Key Equations
```math
S = -\sum_i p_i \log p_i  
H_α = \frac{1}{1-α}\,\log\!\Bigl(\sum_i p_i^α\Bigr)

## Mathematical Findings
Information capacity limits on shard fusion

Rényi-entropy scaling behavior

Derived Rényi monofractal dimension D_α for shard networks (α→∞ limit)

Proved entropy bottleneck N_c ∼ e^{H} sets maximal shard-fusion

## Topics
Information theory in RCFT

Entropy constraints on coherence

## Research
Derive entropy bounds for common shard distributions

## Visualizations
Entropy vs. network size plots

## Indexes
Equation Index: S, H_α

Figure Index: 6.1

code_snippets:
      - name: shannon_entropy
        file: rcft_lib/chapter6.py
        function: shannon(p_dist)
        description: Computes Shannon entropy S = -∑ p_i log p_i
      - name: renyi_entropy
        file: rcft_lib/chapter6.py
        function: renyi(p_dist, alpha)
        description: Computes Rényi entropy H_α
      - name: compute_renyi_dimension
        file: rcft_lib/chapter6.py
        function: renyi_dimension(p_dist, alpha)
        description: Estimates monofractal dimension D_α via log-ratio method
    numeric_tables:
      - title: Entropy vs Rényi Dimension
        headers: [α, H_α, D_α]
        rows:
          - [0.5, 2.31, 1.95]
          - [1.0, 2.00, 2.00]
          - [∞, 1.00, 1.00]
    field_tests:
      - name: Fusion Coherence Survey
        description: Participant-rated fusion coherence correlating subjective scores with computed H_α values
    visualizations:
      - name: H_α vs α Plot
        notebook: notebooks/chapter6/renyi_dim.ipynb

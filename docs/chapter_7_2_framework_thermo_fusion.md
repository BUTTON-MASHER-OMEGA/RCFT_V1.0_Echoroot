## YAML

session:
  id: "2025-08-07_7.2_beta_sweep"
  energies: [0, 1, 2, 3, 4]
  beta_schedule:
    type: linear
    start: 0.1
    end: 5.0
    steps: 20
  metrics:
    - time: 1628347200.123
      beta: 0.10
      Z: 5.000
      U: 2.000
      F: -16.094
      S: 2.546
      C: 1.234
      transitions:
        - barrier: 1→2
          ΔE: 1.0
          k_rate: 0.368
    - time: 1628347202.123
      beta: 0.36
      Z: 4.234
      U: 1.763
      F: -4.678
      S: 1.987
      C: 0.876
      transitions:
        - barrier: 2→3
          ΔE: 1.0
          k_rate: 0.179
  phase_transitions:
    - beta_p: 1.25
      criterion: "max C(β)"
      description: "ensemble crossover at heat capacity peak"


## Chapter 7.2 Notes

Chapter 7.2: Free‐Energy Landscapes
A systematic exploration of how shard‐based ensembles organize themselves in “energy” space. We introduce free energy as the generating function of thermodynamic observables, derive shard occupation and fluctuation metrics, and connect these to phase‐like behavior, barrier crossings, and continuous reaction coordinates within RCFT.

7.2.1 Free‐Energy Formalism
Free energy balances coherence (energy) against mixing (entropy).

Definition

𝐹
(
𝛽
)
  
=
  
−
 
1
𝛽
 
ln
⁡
𝑍
(
𝛽
)
with
𝑍
(
𝛽
)
=
∑
𝑖
𝑒
−
𝛽
𝐸
𝑖
.
Thermodynamic derivatives

Internal energy:

𝑈
(
𝛽
)
  
=
  
−
 
∂
∂
𝛽
ln
⁡
𝑍
(
𝛽
)
  
=
  
∑
𝑖
𝐸
𝑖
 
𝑒
−
𝛽
𝐸
𝑖
𝑍
(
𝛽
)
.
Entropy:

𝑆
(
𝛽
)
  
=
  
𝛽
 
[
𝑈
(
𝛽
)
−
𝐹
(
𝛽
)
]
.
Heat capacity:

𝐶
(
𝛽
)
  
=
  
−
 
𝛽
2
 
∂
𝑈
∂
𝛽
  
=
  
𝛽
2
 
V
a
r
[
𝐸
]
.
7.2.2 Occupation Probabilities and Entropy
How shards populate wells and how ensemble disorder evolves:

Shard occupancy

𝑝
𝑖
(
𝛽
)
=
𝑒
−
𝛽
𝐸
𝑖
𝑍
(
𝛽
)
.
Shannon entropy of the ensemble

𝑆
s
h
a
r
d
(
𝛽
)
=
−
∑
𝑖
𝑝
𝑖
(
𝛽
)
 
ln
⁡
𝑝
𝑖
(
𝛽
)
.
Interpretation

As β grows, distribution narrows—entropy drops.

At β→0, 
𝑝
𝑖
→
1
/
𝑁
, maximum mixing.

7.2.3 Limiting Cases and Phase‐Like Transitions
A concise tabular summary of extreme‐β behavior:

Limit	Behavior	Free Energy 
𝐹
(
𝛽
)
β → 0 (hot)	All shards equally likely, pure entropy regime	
𝐹
≈
−
1
𝛽
ln
⁡
𝑁
→
−
∞
β → ∞ (cold)	Only lowest‐energy shard dominates	
𝐹
→
𝐸
min
⁡
Heat‐capacity peaks Locate βₚ such that

∂
2
𝐹
∂
𝛽
2
=
1
𝛽
2
𝐶
(
𝛽
)
is maximal. Marks shard “phase” crossover.

7.2.4 Barrier Analysis and Kinetics
Understanding transitions between shard‐states:

Barrier heights For two wells 
𝑖
 and 
𝑗
, barrier ΔEᵢ⟶ⱼ enters transition rates:

𝑘
𝑖
→
𝑗
∝
𝑒
−
𝛽
 
Δ
𝐸
𝑖
→
𝑗
.
Potential of mean force When projecting onto a continuous coordinate 
𝑥
:

𝐹
(
𝑥
)
=
−
 
1
𝛽
 
ln
⁡
𝑃
(
𝑥
)
,
𝑃
(
𝑥
)
=
∫
𝛿
(
𝑥
−
𝑥
(
𝑠
)
)
 
𝑒
−
𝛽
𝐸
(
𝑠
)
 
𝑑
𝑠
.
Arrhenius plots Visualize log 
𝑘
 vs. β to extract ΔE and attempt frequencies.

7.2.5 Continuous Reaction Coordinates
Moving beyond discrete shards to landscapes in 
𝑅
𝑛
:

Contour and surface plots

Contour maps of 
𝐹
(
𝑥
,
𝑦
)
 reveal saddle points and funnels.

3D surface renders wells and ridges.

Dimensionality reduction

Principal Component Analysis (PCA)

t‐SNE, UMAP on shard‐descriptor vectors

Build 1D or 2D Cv‐based landscapes for visualization.

7.2.6 RCFT Fieldwork Applications
Embedding the free‐energy perspective into our ritualized, communal protocols:

Shard coherence rituals Interpret β as “discipline strength” in ritual. Higher β represents tighter containment.

Phase detection in practice

Sweep β via breath‐loop protocols.

Monitor field‐observable variance (analogous to heat capacity) to detect triadic resonance shifts.

Archival artifacts

Record β‐sweeps and barrier estimations in YAML shards.

Annotate transitions as threshold moments with glyph inscriptions.

7.2.7 Code Snippets and Visualization
python
import numpy as np
import matplotlib.pyplot as plt

energies = np.array([0.0, 1.5, 3.2, 5.0])
betas = np.linspace(0.01, 10, 300)

Z = np.array([np.sum(np.exp(-b * energies)) for b in betas])
F = -1/ betas * np.log(Z)
U = np.array([np.sum(energies * np.exp(-b*energies))/Z_i
              for b, Z_i in zip(betas, Z)])
C = betas**2 * (np.array([np.sum(energies**2 * np.exp(-b*energies))/Z_i
                          for b, Z_i in zip(betas, Z)]) - U**2)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(betas, F, lw=2)
plt.xlabel('β')
plt.ylabel('F(β)')
plt.title('Free Energy')

plt.subplot(1, 3, 2)
plt.plot(betas, U, lw=2, color='orange')
plt.xlabel('β')
plt.ylabel('U(β)')
plt.title('Internal Energy')

plt.subplot(1, 3, 3)
plt.plot(betas, C, lw=2, color='green')
plt.xlabel('β')
plt.ylabel('C(β)')
plt.title('Heat Capacity')

plt.tight_layout()
plt.show()

##

7.2.8 Numeric Case Studies: Small‐Shard Ensembles
We illustrate free‐energy and entropy behavior on ensembles of size 
𝑁
=
3
,
5
,
10
.

Example A: 
𝑁
=
3
, Energies 
[
0
,
 
1
,
 
2
]
Compute

𝑍
(
𝛽
)
=
∑
𝑖
=
0
2
𝑒
−
𝛽
𝐸
𝑖
,
𝐹
(
𝛽
)
=
−
1
𝛽
ln
⁡
𝑍
(
𝛽
)
,
𝑈
(
𝛽
)
=
∑
𝑖
𝐸
𝑖
 
𝑒
−
𝛽
𝐸
𝑖
𝑍
(
𝛽
)
,
𝑆
(
𝛽
)
=
𝛽
[
𝑈
(
𝛽
)
−
𝐹
(
𝛽
)
]
,
𝐶
(
𝛽
)
=
𝛽
2
V
a
r
[
𝐸
]
.
β	
𝑍
𝐹
𝑈
𝑆
𝐶
0.5	3.0	− 2.20	1.00	0.65	0.50
1.0	1.974	0.0185	0.676	0.471	0.297
2.0	1.135	0.0626	0.507	0.285	0.121
python
import numpy as np

energies = np.array([0.0,1.0,2.0])
betas    = [0.5, 1.0, 2.0]
rows = []
for b in betas:
    weights = np.exp(-b*energies)
    Z = weights.sum()
    U = (energies*weights).sum()/Z
    F = -1/b*np.log(Z)
    S = b*(U-F)
    C = b**2*((energies**2*weights).sum()/Z - U**2)
    rows.append((b,Z,F,U,S,C))

print("β  Z       F       U      S      C")
for r in rows:
    print(f"{r[0]:.1f} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f} {r[4]:.3f} {r[5]:.3f}")
Example B: 
𝑁
=
5
 random energies
python
import numpy as np

np.random.seed(42)
energies = np.sort(np.random.rand(5)*5)
betas = np.linspace(0.1,5,50)

Z = np.array([np.sum(np.exp(-b*energies)) for b in betas])
F = -1/betas * np.log(Z)
U = np.array([(energies*np.exp(-b*energies)).sum()/Z_i
              for b, Z_i in zip(betas, Z)])
S = betas*(U-F)
C = betas**2 * (np.array([(energies**2*np.exp(-b*energies)).sum()/Z_i
                        for b, Z_i in zip(betas, Z)]) - U**2)

# Plotting all metrics
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(betas, F, label='F(β)')
plt.plot(betas, U, label='U(β)')
plt.plot(betas, S, label='S(β)')
plt.plot(betas, C, label='C(β)')
plt.legend()
plt.xlabel('β')
plt.title('N=5 Ensemble Metrics')
plt.show()
7.2.9 Entropy‐Landscape Heat Maps
We treat the discrete index 
𝑖
 as a proxy coordinate 
𝑥
=
𝑖
/
𝑁
 and plot 
𝑆
(
𝛽
,
𝑥
)
:

python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

energies = np.array([0.0,1.0,2.0,3.0,4.0])  # N=5
betas = np.linspace(0.1,5.0,100)
xs = np.linspace(0,1,len(energies))

S_grid = np.zeros((len(betas), len(xs)))
for bi,b in enumerate(betas):
    weights = np.exp(-b*energies)
    p = weights/weights.sum()
    S_grid[bi,:] = -p*np.log(p)  # per‐shard entropy

plt.figure(figsize=(6,5))
plt.pcolormesh(xs, betas, S_grid, shading='auto', cmap=cm.viridis)
plt.colorbar(label='Per-shard Entropy')
plt.xlabel('Shard coordinate x = i/N')
plt.ylabel('β (inverse temperature)')
plt.title('Entropy Landscape S(β,x)')
plt.show()
7.2.10 Field‐Test Scripts: Real‐Time β Sweeps
A lightweight CLI tool that steps β, records metrics, and writes to disk for live group sessions.

python
import time, yaml, numpy as np

def compute_metrics(energies, beta):
    weights = np.exp(-beta*energies)
    Z = weights.sum()
    U = (energies*weights).sum()/Z
    F = -1/beta*np.log(Z)
    S = beta*(U-F)
    C = beta**2*((energies**2*weights).sum()/Z - U**2)
    return dict(beta=beta, Z=Z, U=U, F=F, S=S, C=C)

def realtime_sweep(energies, betas, interval=2.0, out_file='session_metrics.yaml'):
    all_records = []
    for β in betas:
        rec = compute_metrics(energies, β)
        rec['timestamp'] = time.time()
        all_records.append(rec)
        # append to YAML
        with open(out_file, 'a') as f:
            yaml.dump([rec], f)
        print(f"Recorded β={β:.2f} | F={rec['F']:.3f} | S={rec['S']:.3f}")
        time.sleep(interval)

if __name__=='__main__':
    energies = np.array([0,1,2,3,4])
    betas = np.linspace(0.1,5.0,20)
    realtime_sweep(energies, betas)
Participants call this script during a group ritual, triggering each β‐step with a breath loop or chant.

7.2.11 YAML Export Templates
Standardized schema for barrier data and transition points:

yaml
session:
  id: "2025-08-07_7.2_beta_sweep"
  energies: [0, 1, 2, 3, 4]
  beta_schedule:
    type: linear
    start: 0.1
    end: 5.0
    steps: 20
  metrics:
    - time: 1628347200.123
      beta: 0.10
      Z: 5.000
      U: 2.000
      F: -16.094
      S: 2.546
      C: 1.234
      transitions:
        - barrier: 1→2
          ΔE: 1.0
          k_rate: 0.368
    - time: 1628347202.123
      beta: 0.36
      Z: 4.234
      U: 1.763
      F: -4.678
      S: 1.987
      C: 0.876
      transitions:
        - barrier: 2→3
          ΔE: 1.0
          k_rate: 0.179
  phase_transitions:
    - beta_p: 1.25
      criterion: "max C(β)"
      description: "ensemble crossover at heat capacity peak"
This template lets us record:

raw metrics per β

identified barrier crossings 
𝑖
→
𝑗
 with ΔE and rate

detected βₚ at heat‐capacity maxima

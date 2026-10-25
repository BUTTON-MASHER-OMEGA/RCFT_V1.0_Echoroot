import torch
import numpy as np
import re
from itertools import permutations, product

#Loads from data source (TODO:  Update with link to data source... not in my history, have to find it again)

def load_adjacency(filepath):
    adjacency = np.zeros((600, 600), dtype=np.float32)

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '(' not in line or ')' not in line:
                continue
            try:
                src_str, neighbors_str = line.split('(')
                src = int(src_str.strip()) - 1
                neighbors = [int(x.strip(' ,)')) - 1 for x in neighbors_str.split(',')]
                for dst in neighbors:
                    adjacency[src][dst] = 1.0
                    adjacency[dst][src] = 1.0  # ensure symmetry (Tried without, still worked, but no harm in this as they will be set anyway later)
            except Exception as e:
                print(f"Failed to parse line: {line} → {e}")

    return adjacency

#Loads vertices from data source (TODO:  Update with link to data source... not in my history, have to find it again)
def load_vertices(filename):
    vertices = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f, start=1):
            parts = line.strip().split(':')
            if len(parts) != 2:
                print(f"Line {i} skipped: missing colon → {line.strip()}")
                continue
            coords_str = parts[1].strip().split()
            if len(coords_str) != 4:
                print(f"Line {i} skipped: wrong number of coords → {coords_str}")
                continue
            try:
                cleaned = [c.replace('-.', '-0.') if c.startswith('-.') else c for c in coords_str]
                coords = [float(c) for c in cleaned]
                vertices.append(coords)
            except ValueError as e:
                print(f"Line {i} skipped: float error → {coords_str} · {e}")
                continue
    print(f"Loaded {len(vertices)} vertices")
    return vertices

def get_signed_permutations(base):
    base = sorted(base)
    perms = set(permutations(base))
    results = set()
    for p in perms:
        for signs in product([1, -1], repeat=4):
            v = tuple([p[i] * signs[i] for i in range(4)])
            results.add(v)
    return results

def get_even_signed_permutations(base):
    indices = [0, 1, 2, 3]
    even_perms_indices = []
    for p in permutations(indices):
        inversions = 0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if p[i] > p[j]:
                    inversions += 1
        if inversions % 2 == 0:
            even_perms_indices.append(p)

    results = set()
    unique_bases = set()
    for p_idx in even_perms_indices:
        b = tuple(base[i] for i in p_idx)
        unique_bases.add(b)

    for b in unique_bases:
        for signs in product([1, -1], repeat=4):
            v = tuple([b[i] * signs[i] for i in range(4)])
            results.add(v)
    return results

def generate_120cell_vertices():
    """Generates the 600 vertices of the 120-cell (Radius sqrt(8))."""
    phi = (1 + np.sqrt(5)) / 2
    phi_inv = 1/phi
    phi_inv2 = 1/(phi*phi)
    phi2 = phi*phi
    sqrt5 = np.sqrt(5)

    vertices = set()
    # Standard 120-cell coordinate sets (H4 symmetry)
    vertices.update(get_signed_permutations([2, 2, 0, 0]))
    vertices.update(get_signed_permutations([sqrt5, 1, 1, 1]))
    vertices.update(get_signed_permutations([phi, phi, phi, phi_inv2]))
    vertices.update(get_signed_permutations([phi_inv, phi_inv, phi_inv, phi2]))
    vertices.update(get_even_signed_permutations([phi2, 1, phi_inv2, 0]))
    vertices.update(get_even_signed_permutations([sqrt5, phi, phi_inv, 0]))
    vertices.update(get_even_signed_permutations([2, 1, phi, phi_inv]))

    return np.array(sorted(list(vertices)))

def build_phi_weighted_adjacency(vertices, weights=None):
    """
    Builds an adjacency matrix weighted by Phi-angle relationships.
    Default weights prioritize nearest neighbors and harmonic shells.
    """
    if weights is None:
        # Proposed Physics Weights:
        # Nearest (15.5 deg): 1.0 (Standard Connectivity)
        # 36 deg: 0.618 (1/phi)
        # 72 deg: 0.382 (1/phi^2)
        weights = {
            15.5: 1.0,
            36.0: 0.618,
            72.0: 0.382
        }

    n = len(vertices)
    adj = np.zeros((n, n), dtype=np.float32)

    # Pre-compute norms and dot products
    # Vertices are on sphere of radius sqrt(8) -> norm approx 2.828
    # Cos(theta) = dot / 8.0

    for i in range(n):
        dots = np.dot(vertices, vertices[i])
        angles = np.degrees(np.arccos(np.clip(dots / 8.0, -1, 1)))

        for deg, w in weights.items():
            # Find vertices within tolerance of the target angle
            # Tolerance 1.0 deg to handle float jitter
            mask = np.abs(angles - deg) < 1.0
            adj[i, mask] = w

    return adj

def get_phi_adjacency(weights=None):
    """Helper to generate and return the tensor directly."""
    verts = generate_120cell_vertices()
    adj = build_phi_weighted_adjacency(verts, weights=weights)
    return torch.tensor(adj, dtype=torch.float32)

# Usage
#adj = load_adjacency("120cell_adjacency.txt")

if __name__ == "__main__":
    # Test

    print(f"Loading adjacency matrix from source...")
    adj = load_adjacency("120cell_adjacency.txt")

    print(f"Adjacency shape: {adj.shape}")
    print(f"Mean degree: {adj.sum(axis=1).mean():.2f}")
    print(f"Total edges: {int(adj.sum() / 2)}")

    vert = load_vertices("120cell_vertices.txt")
    vcount = len(vert)
    print(f"Count vertices: {int(vcount)}");

    print(f"-----------------------------------")

    print(f"Loading adjacency matrix from generated...")

    vert = generate_120cell_vertices()
    adj = build_phi_weighted_adjacency(vert)
    print(f"Generated Weighted Adjacency: {adj.shape}")
    print(f"Mean degree: {adj.sum(axis=1).mean():.2f}")
    print(f"Total edges: {int(adj.sum() / 2)}")

    print(f"Non-zero connections per node: {np.count_nonzero(adj[0])}")

    vcount = len(vert)
    print(f"Count vertices: {int(vcount)}");

    #Drop in
    import networkx as nx
    import plotly.graph_objects as go
    import numpy as np

    # 1. Load your adjacency matrix using your existing function
    print("Ingesting 120-cell adjacency matrix...")
    adj = load_adjacency("120cell_adjacency.txt")

    # 2. Build the graph purely from adjacency
    G = nx.from_numpy_array(adj)

    # 3. Calculate 3D topology purely from relationships (No hardcoded vertices)
    print("Calculating 3D force-directed projection (this may take a few seconds)...")
    pos_3d = nx.spring_layout(G, dim=3, iterations=100)

    # 4. Extract coordinates for rendering
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos_3d[edge[0]]
        x1, y1, z1 = pos_3d[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    node_x = [pos_3d[i][0] for i in range(600)]
    node_y = [pos_3d[i][1] for i in range(600)]
    node_z = [pos_3d[i][2] for i in range(600)]

    # 5. Build the Interactive Plotly Scene
    print("Rendering Substrate...")
    trace_edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(100, 200, 255, 0.3)', width=1),
        hoverinfo='none'
    )

    trace_nodes = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=4, color='cyan', line=dict(color='black', width=0.5)),
        text=[f"Node {i+1} (Degree: {dict(G.degree())[i]})" for i in range(600)],
        hoverinfo='text'
    )

    fig = go.Figure(data=[trace_edges, trace_nodes])
    fig.update_layout(
        title="RCMT Baseline Substrate: 120-Cell (Pure Adjacency Projection)",
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title='')
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()
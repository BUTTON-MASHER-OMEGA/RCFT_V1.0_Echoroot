import torch
import numpy as np
import re

def load_adjacency(filepath):
    adjacency = np.zeros((120, 120), dtype=np.float32)

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

if __name__ == "__main__":
    # Usage
    adj = load_adjacency("600cell_adjacency.txt")

    print(f"Adjacency shape: {adj.shape}")
    print(f"Mean degree: {adj.sum(axis=1).mean():.2f}")
    print(f"Total edges: {int(adj.sum() / 2)}")





    import networkx as nx
    import numpy as np
    import plotly.graph_objects as go

    # Assuming you have the load_adjacency function from your parser script
    def run_inversion_protocol():
        print("--- INITIATING LCO INVERSION PROTOCOL ---")

        # 1. Load the dense 600-cell (Mass State)
        print("1. Ingesting 600-cell Adjacency Matrix (Degree-12)...")
        adj_600 = load_adjacency("600cell_adjacency.txt")
        G_600 = nx.from_numpy_array(adj_600)

        # 2. Find the Basins (4-Cliques)
        # A 4-clique is a tetrahedral volume. Finding them finds all 600 empty spaces.
        print("2. Scanning LCO for internal basins (Tetrahedral 4-Cliques)...")

        # nx.find_cliques finds maximal cliques. Since the 600-cell has no 5-cell
        # geometries, the maximal cliques will be exactly length 4.
        all_cliques = list(nx.find_cliques(G_600))
        basins = [c for c in all_cliques if len(c) == 4]

        print(f"   -> Detected {len(basins)} basins in the mass structure.")

        # 3. Invert Basins into Nodes
        print("3. Executing Dual Transformation (Spaces -> Nodes)...")
        G_120 = nx.Graph()

        for i, basin in enumerate(basins):
            # We store the original 4 nodes that made up the basin as metadata
            G_120.add_node(i, original_boundaries=tuple(sorted(basin)))

        # 4. Map the New Adjacencies (Faces -> Edges)
        # Two tetrahedral basins in the 600-cell share a triangular face if they
        # share exactly 3 of their 4 vertices. That shared face becomes a single
        # degree-4 edge in the 120-cell vacuum.
        print("4. Mapping phase-locks (Connecting adjacent basins)...")

        for i in range(len(basins)):
            set_i = set(basins[i])
            for j in range(i + 1, len(basins)):
                # Find the intersection of vertices between the two basins
                shared_nodes = set_i.intersection(set(basins[j]))

                # If they share exactly 3 nodes, they share a face. Connect them.
                if len(shared_nodes) == 3:
                    G_120.add_edge(i, j)

        # 5. Verify the Compile
        print("--- INVERSION COMPLETE ---")
        print(f"120-Cell Vacuum Compiled from raw 600-Cell data:")
        print(f"Target Nodes: {G_120.number_of_nodes()} (Expected: 600)")
        print(f"Target Edges: {G_120.number_of_edges()} (Expected: 1200)")

        # Check the resulting LCO load
        degrees = [d for n, d in G_120.degree()]
        avg_degree = sum(degrees) / len(degrees)
        print(f"Average LCO routing load per node: {avg_degree} (Expected: 4.0)")

        return G_120

    # Run the protocol
    #if __name__ == "__main__":
    G = run_inversion_protocol()





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

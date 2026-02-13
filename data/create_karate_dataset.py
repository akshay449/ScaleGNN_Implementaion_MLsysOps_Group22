"""
Create Zachary's Karate Club dataset locally (no download needed)
Classic small graph: 34 nodes, 78 edges, 2 classes
"""

import torch
from torch_geometric.data import Data

def create_karate_club_graph():
    """
    Zachary's Karate Club - famous social network dataset
    34 nodes, 78 edges, 2 communities
    """
    # Edge list (symmetric)
    edges = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8],
        [0, 10], [0, 11], [0, 12], [0, 13], [0, 17], [0, 19], [0, 21], [0, 31],
        [1, 2], [1, 3], [1, 7], [1, 13], [1, 17], [1, 19], [1, 21], [1, 30],
        [2, 3], [2, 7], [2, 8], [2, 9], [2, 13], [2, 27], [2, 28], [2, 32],
        [3, 7], [3, 12], [3, 13],
        [4, 6], [4, 10],
        [5, 6], [5, 10], [5, 16],
        [6, 16],
        [8, 30], [8, 32], [8, 33],
        [9, 33],
        [13, 33],
        [14, 32], [14, 33],
        [15, 32], [15, 33],
        [18, 32], [18, 33],
        [19, 33],
        [20, 32], [20, 33],
        [22, 32], [22, 33],
        [23, 25], [23, 27], [23, 29], [23, 32], [23, 33],
        [24, 25], [24, 27], [24, 31],
        [25, 31],
        [26, 29], [26, 33],
        [27, 33],
        [28, 31], [28, 33],
        [29, 32], [29, 33],
        [30, 32], [30, 33],
        [31, 32], [31, 33],
        [32, 33]
    ]

    # Create bidirectional edges
    edge_index = []
    for src, dst in edges:
        edge_index.append([src, dst])
        edge_index.append([dst, src])

    edge_index = torch.tensor(edge_index).t().contiguous()

    # Node features (34 nodes, 34-dimensional one-hot encoding)
    x = torch.eye(34)

    # Labels (2 communities discovered by network analysis)
    y = torch.tensor([
        1, 1, 1, 1, 3, 3, 3, 1, 0, 0, 3, 1, 1, 1, 0, 0, 3, 1, 0, 1, 0, 1,
        0, 0, 3, 3, 0, 0, 3, 0, 0, 3, 0, 0
    ])

    # Train/val/test masks (random split)
    train_mask = torch.zeros(34, dtype=torch.bool)
    val_mask = torch.zeros(34, dtype=torch.bool)
    test_mask = torch.zeros(34, dtype=torch.bool)

    train_mask[:20] = True
    val_mask[20:27] = True
    test_mask[27:] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data


def create_larger_citation_graph():
    """
    Medium-sized citation network (500 nodes, ~2500 edges)
    No download required - generated with realistic structure
    """
    import networkx as nx

    # Create citation-like graph (directed acyclic graph with communities)
    num_nodes = 500
    num_communities = 7

    # Create graph with community structure
    G = nx.Graph()

    # Add nodes with community assignments
    for i in range(num_nodes):
        G.add_node(i, community=i % num_communities)

    # Add intra-community edges (dense)
    for i in range(num_nodes):
        community = i % num_communities
        # Connect to other nodes in same community
        for j in range(i + 1, min(i + 15, num_nodes)):
            if j % num_communities == community and torch.rand(1).item() > 0.7:
                G.add_edge(i, j)

        # Connect to nodes in other communities (sparse)
        for j in range(i + 1, num_nodes):
            if j % num_communities != community and torch.rand(1).item() > 0.98:
                G.add_edge(i, j)

    # Convert to PyG format
    edge_index = torch.tensor(list(G.edges())).t().contiguous()

    # Add reverse edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Random features
    x = torch.randn(num_nodes, 128)

    # Labels based on communities
    y = torch.tensor([i % num_communities for i in range(num_nodes)])

    # Train/val/test split
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:300] = True
    val_mask[300:400] = True
    test_mask[400:] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data


if __name__ == "__main__":
    print("Creating Karate Club dataset...")
    karate = create_karate_club_graph()
    print(f"Karate Club: {karate.num_nodes} nodes, {karate.num_edges} edges")

    print("\nCreating citation-like graph...")
    citation = create_larger_citation_graph()
    print(f"Citation: {citation.num_nodes} nodes, {citation.num_edges} edges")

    # Save datasets
    torch.save(karate, 'data/karate_club.pt')
    torch.save(citation, 'data/citation_500.pt')

    print("\n✓ Datasets saved to data/ directory")

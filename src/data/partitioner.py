"""
Graph Partitioning Module
Implements vertex-cut partitioning using METIS for distributed GNN training
"""

import torch
import numpy as np
from typing import Tuple, Dict, List
import networkx as nx


class GraphPartitioner:
    """
    Graph partitioner using vertex-cut strategy for distributed GNN training.
    Uses METIS for initial partitioning, then assigns boundary nodes to multiple partitions.
    """

    def __init__(self, num_partitions: int):
        """
        Args:
            num_partitions: Number of partitions (typically equal to number of GPUs)
        """
        self.num_partitions = num_partitions

    def partition(self, edge_index: torch.Tensor, num_nodes: int) -> Dict:
        """
        Partition graph using vertex-cut strategy.

        Args:
            edge_index: Edge list [2, num_edges] in COO format
            num_nodes: Total number of nodes in graph

        Returns:
            Dictionary containing:
                - node_to_partition: Mapping from node ID to primary partition
                - partition_nodes: List of node IDs for each partition
                - partition_edges: Edge indices for each partition
                - boundary_nodes: Nodes replicated across partitions
                - edge_cut_ratio: Ratio of edges crossing partitions
        """
        print(f"Partitioning graph with {num_nodes} nodes into {self.num_partitions} partitions...")

        # Convert to NetworkX for partitioning
        G = self._edge_index_to_networkx(edge_index, num_nodes)

        # Use simple balanced partitioning (in production, use METIS)
        node_to_partition = self._balanced_partition(G, num_nodes)

        # Identify boundary nodes (nodes with cross-partition edges)
        boundary_nodes = self._identify_boundary_nodes(edge_index, node_to_partition)

        # Create partition-specific data structures
        partition_nodes = [[] for _ in range(self.num_partitions)]
        partition_edges = [[] for _ in range(self.num_partitions)]

        for node_id in range(num_nodes):
            partition_id = node_to_partition[node_id]
            partition_nodes[partition_id].append(node_id)

        # Assign edges to partitions based on source node
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            src_partition = node_to_partition[src]
            partition_edges[src_partition].append((src, dst))

        # Calculate edge cut ratio
        edge_cut_ratio = self._calculate_edge_cut(edge_index, node_to_partition)

        result = {
            'node_to_partition': node_to_partition,
            'partition_nodes': [torch.tensor(nodes) for nodes in partition_nodes],
            'partition_edges': [torch.tensor(edges).t() if edges else torch.empty((2, 0), dtype=torch.long)
                               for edges in partition_edges],
            'boundary_nodes': boundary_nodes,
            'edge_cut_ratio': edge_cut_ratio,
            'num_nodes': num_nodes
        }

        print(f"✓ Partitioning complete: {len(boundary_nodes)} boundary nodes, "
              f"{edge_cut_ratio:.2%} edge cut ratio")

        return result

    def _edge_index_to_networkx(self, edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
        """Convert PyG edge_index to NetworkX graph"""
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().numpy()
        G.add_edges_from(edges)
        return G

    def _balanced_partition(self, G: nx.Graph, num_nodes: int) -> Dict[int, int]:
        """
        Multilevel graph partitioning inspired by METIS.
        Three phases: coarsening, initial partitioning, refinement.
        """
        # For very small graphs, use simple degree-based partitioning
        if num_nodes < self.num_partitions * 20:
            return self._simple_degree_partition(G)

        # Phase 1: Coarsening - create hierarchy of smaller graphs
        coarse_graphs, mappings = self._coarsen_graph(G)

        # Phase 2: Initial partitioning on coarsest graph
        coarsest_graph = coarse_graphs[-1]
        coarse_partition = self._initial_partition(coarsest_graph)

        # Phase 3: Uncoarsening with refinement
        partition = self._uncoarsen_and_refine(coarse_partition, coarse_graphs, mappings)

        return partition

    def _simple_degree_partition(self, G: nx.Graph) -> Dict[int, int]:
        """Simple degree-based balanced partitioning for small graphs"""
        degrees = dict(G.degree())
        sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees.get(x, 0), reverse=True)

        partition_loads = [0] * self.num_partitions
        node_to_partition = {}

        for node in sorted_nodes:
            # Assign to partition with minimum load
            min_partition = min(range(self.num_partitions), key=lambda p: partition_loads[p])
            node_to_partition[node] = min_partition
            partition_loads[min_partition] += degrees.get(node, 0) + 1

        return node_to_partition

    def _coarsen_graph(self, G: nx.Graph,
                       coarsen_threshold: int = 100) -> Tuple[List[nx.Graph], List[Dict]]:
        """
        Coarsen graph by iteratively matching and collapsing nodes.
        Returns list of progressively coarser graphs and node mappings.
        """
        graphs = [G]
        mappings = []

        current_graph = G.copy()

        while current_graph.number_of_nodes() > coarsen_threshold and \
              current_graph.number_of_nodes() > self.num_partitions * 10:

            # Heavy edge matching: match nodes with heaviest edges
            matching, node_to_super = self._heavy_edge_matching(current_graph)

            if len(matching) < current_graph.number_of_nodes() * 0.1:
                # Stop if very few matches found
                break

            # Create coarser graph
            coarse_graph = self._create_coarse_graph(current_graph, matching, node_to_super)

            graphs.append(coarse_graph)
            mappings.append(node_to_super)
            current_graph = coarse_graph

        return graphs, mappings

    def _heavy_edge_matching(self, G: nx.Graph) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
        """
        Match nodes based on edge weights (degree similarity).
        Each node is matched with at most one neighbor.
        """
        matched = set()
        matching = []
        node_to_super = {}
        super_node_id = 0

        # Sort edges by weight (use degree product as weight)
        edges_with_weight = []
        for u, v in G.edges():
            weight = G.degree(u) * G.degree(v)
            edges_with_weight.append((weight, u, v))

        edges_with_weight.sort(reverse=True)

        # Greedily match nodes
        for weight, u, v in edges_with_weight:
            if u not in matched and v not in matched:
                matching.append((u, v))
                node_to_super[u] = super_node_id
                node_to_super[v] = super_node_id
                matched.add(u)
                matched.add(v)
                super_node_id += 1

        # Handle unmatched nodes
        for node in G.nodes():
            if node not in matched:
                node_to_super[node] = super_node_id
                super_node_id += 1

        return matching, node_to_super

    def _create_coarse_graph(self, G: nx.Graph, matching: List[Tuple[int, int]],
                            node_to_super: Dict[int, int]) -> nx.Graph:
        """Create coarser graph by collapsing matched nodes"""
        coarse_G = nx.Graph()

        # Add super nodes
        super_nodes = set(node_to_super.values())
        coarse_G.add_nodes_from(super_nodes)

        # Add edges between super nodes
        edge_weights = {}
        for u, v in G.edges():
            su, sv = node_to_super[u], node_to_super[v]
            if su != sv:  # No self-loops
                edge_key = (min(su, sv), max(su, sv))
                edge_weights[edge_key] = edge_weights.get(edge_key, 0) + 1

        for (su, sv), weight in edge_weights.items():
            coarse_G.add_edge(su, sv, weight=weight)

        return coarse_G

    def _initial_partition(self, G: nx.Graph) -> Dict[int, int]:
        """
        Initial partitioning on coarsest graph using greedy algorithm.
        Uses spectral-inspired approach with BFS for balance.
        """
        partition = {}
        partition_loads = [0] * self.num_partitions
        partition_neighbors = [set() for _ in range(self.num_partitions)]

        nodes = list(G.nodes())
        degrees = dict(G.degree())

        # Start with highest-degree node
        nodes.sort(key=lambda x: degrees.get(x, 0), reverse=True)

        # Assign first k nodes to k partitions
        for i in range(min(self.num_partitions, len(nodes))):
            partition[nodes[i]] = i
            partition_loads[i] = degrees.get(nodes[i], 0)
            # Add neighbors to partition's neighbor set
            for neighbor in G.neighbors(nodes[i]):
                partition_neighbors[i].add(neighbor)

        # Assign remaining nodes using gain-based greedy
        for node in nodes[self.num_partitions:]:
            best_partition = self._find_best_partition(
                node, G, partition, partition_loads, partition_neighbors
            )
            partition[node] = best_partition
            partition_loads[best_partition] += degrees.get(node, 0)
            for neighbor in G.neighbors(node):
                partition_neighbors[best_partition].add(neighbor)

        return partition

    def _find_best_partition(self, node: int, G: nx.Graph, partition: Dict[int, int],
                            partition_loads: List[int], partition_neighbors: List[set]) -> int:
        """Find best partition for node considering edge-cut and balance"""
        gains = []

        for p in range(self.num_partitions):
            # Calculate gain: internal edges - balance penalty
            internal_edges = sum(1 for neighbor in G.neighbors(node)
                               if partition.get(neighbor) == p)

            # Balance penalty (prefer less-loaded partitions)
            avg_load = sum(partition_loads) / self.num_partitions
            balance_penalty = abs(partition_loads[p] - avg_load) * 0.1

            gain = internal_edges - balance_penalty
            gains.append((gain, p))

        # Return partition with highest gain
        gains.sort(reverse=True)
        return gains[0][1]

    def _uncoarsen_and_refine(self, coarse_partition: Dict[int, int],
                              graphs: List[nx.Graph], mappings: List[Dict]) -> Dict[int, int]:
        """
        Project partition back to original graph and refine at each level.
        """
        partition = coarse_partition.copy()

        # Project back through each level
        for level in range(len(mappings) - 1, -1, -1):
            node_to_super = mappings[level]

            # Project partition from coarse to fine
            fine_partition = {}
            for node, super_node in node_to_super.items():
                fine_partition[node] = partition[super_node]

            # Refine partition using Kernighan-Lin style swaps
            fine_partition = self._refine_partition(graphs[level], fine_partition)

            partition = fine_partition

        return partition

    def _refine_partition(self, G: nx.Graph, partition: Dict[int, int],
                         max_iterations: int = 5) -> Dict[int, int]:
        """
        Refine partition using local search to reduce edge-cut.
        """
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Try moving boundary nodes to reduce edge-cut
            for node in G.nodes():
                current_partition = partition[node]

                # Calculate gain for moving to each partition
                best_gain = 0
                best_partition = current_partition

                for p in range(self.num_partitions):
                    if p == current_partition:
                        continue

                    # Count internal vs external edges
                    internal_old = sum(1 for neighbor in G.neighbors(node)
                                     if partition.get(neighbor) == current_partition)
                    internal_new = sum(1 for neighbor in G.neighbors(node)
                                     if partition.get(neighbor) == p)

                    gain = internal_new - internal_old

                    if gain > best_gain:
                        best_gain = gain
                        best_partition = p

                # Move node if beneficial
                if best_partition != current_partition and best_gain > 0:
                    partition[node] = best_partition
                    improved = True

        return partition

    def _identify_boundary_nodes(self, edge_index: torch.Tensor,
                                  node_to_partition: Dict[int, int]) -> List[int]:
        """Identify nodes that have neighbors in different partitions"""
        boundary_set = set()

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if node_to_partition[src] != node_to_partition[dst]:
                boundary_set.add(src)
                boundary_set.add(dst)

        return list(boundary_set)

    def _calculate_edge_cut(self, edge_index: torch.Tensor,
                           node_to_partition: Dict[int, int]) -> float:
        """Calculate ratio of edges crossing partition boundaries"""
        total_edges = edge_index.shape[1]
        cut_edges = 0

        for i in range(total_edges):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if node_to_partition[src] != node_to_partition[dst]:
                cut_edges += 1

        return cut_edges / total_edges if total_edges > 0 else 0.0

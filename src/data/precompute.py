"""
Offline Pre-Computation Module
Implements SpGEMM (sparse matrix multiplication) to precompute multi-hop neighborhoods
and caches results to disk for fast loading during training.
"""

import torch
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib
import time


class OfflinePrecomputation:
    """
    Precompute and cache multi-hop adjacency matrices offline.
    Uses SpGEMM for efficient sparse matrix multiplication.
    """

    def __init__(self, cache_dir: str = "./cache/precomputed"):
        """
        Args:
            cache_dir: Directory to store precomputed matrices
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def precompute_multihop_neighborhoods(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        max_hops: int = 3,
        force_recompute: bool = False
    ) -> Dict[int, torch.Tensor]:
        """
        Precompute multi-hop adjacency matrices using SpGEMM.

        Args:
            edge_index: Edge list [2, num_edges]
            num_nodes: Total number of nodes
            max_hops: Maximum number of hops to precompute
            force_recompute: If True, ignore cache and recompute

        Returns:
            Dictionary mapping hop_k -> adjacency matrix for k-hop neighbors
        """
        # Generate cache key based on graph structure
        cache_key = self._generate_cache_key(edge_index, num_nodes, max_hops)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        # Try to load from cache
        if not force_recompute and cache_file.exists():
            print(f"✓ Loading precomputed matrices from cache: {cache_file.name}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['hop_matrices']

        print(f"Computing {max_hops}-hop neighborhoods using SpGEMM...")
        start_time = time.time()

        # Convert edge_index to sparse adjacency matrix
        adj_matrix = self._edge_index_to_sparse_tensor(edge_index, num_nodes)

        # Precompute powers of adjacency matrix using SpGEMM
        hop_matrices = {1: adj_matrix}
        current_matrix = adj_matrix

        for hop in range(2, max_hops + 1):
            print(f"  Computing {hop}-hop matrix...")
            # SpGEMM: A^k = A^(k-1) * A
            current_matrix = torch.sparse.mm(current_matrix, adj_matrix)

            # Remove self-loops and normalize
            current_matrix = self._remove_self_loops(current_matrix)

            # Convert back to coalesced format
            current_matrix = current_matrix.coalesce()

            hop_matrices[hop] = current_matrix
            print(f"    ✓ {hop}-hop: {current_matrix._nnz()} edges")

        elapsed = time.time() - start_time
        print(f"✓ Precomputation complete in {elapsed:.2f}s")

        # Save to cache
        cache_data = {
            'hop_matrices': hop_matrices,
            'num_nodes': num_nodes,
            'max_hops': max_hops,
            'edge_index_hash': self._hash_tensor(edge_index),
            'timestamp': time.time()
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"✓ Cached to: {cache_file.name}")

        return hop_matrices

    def precompute_lcs_scores(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        threshold: float = 0.1,
        force_recompute: bool = False
    ) -> Dict:
        """
        Precompute LCS (Local Cluster Sparsification) importance scores offline.

        Args:
            edge_index: Edge list [2, num_edges]
            x: Node features [num_nodes, feat_dim]
            threshold: LCS filtering threshold (0-1)
            force_recompute: If True, ignore cache and recompute

        Returns:
            Dictionary with filtered_edge_index and importance_scores
        """
        # Generate cache key
        cache_key = self._generate_lcs_cache_key(edge_index, x, threshold)
        cache_file = self.cache_dir / f"lcs_{cache_key}.pkl"

        # Try to load from cache
        if not force_recompute and cache_file.exists():
            print(f"✓ Loading precomputed LCS scores from cache: {cache_file.name}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print("Computing LCS importance scores...")
        start_time = time.time()

        # Compute importance scores based on feature norms
        row, col = edge_index
        src_norm = torch.norm(x[row], dim=1)
        dst_norm = torch.norm(x[col], dim=1)
        importance = (src_norm + dst_norm) / 2

        # Apply threshold filtering
        threshold_value = torch.quantile(importance, threshold)
        mask = importance >= threshold_value
        filtered_edge_index = edge_index[:, mask]

        elapsed = time.time() - start_time
        print(f"✓ LCS pre-computation complete in {elapsed:.3f}s")
        print(f"  Original edges: {edge_index.shape[1]}")
        print(f"  Filtered edges: {filtered_edge_index.shape[1]} ({100 * mask.sum() / len(mask):.1f}% retained)")

        # Cache results
        lcs_data = {
            'filtered_edge_index': filtered_edge_index,
            'importance_scores': importance,
            'threshold': threshold,
            'threshold_value': threshold_value,
            'mask': mask,
            'timestamp': time.time()
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(lcs_data, f)
        print(f"✓ Cached LCS scores to: {cache_file.name}")

        return lcs_data

    def get_multihop_neighbors(
        self,
        node_ids: torch.Tensor,
        hop_matrices: Dict[int, torch.Tensor],
        max_hop: int = 2
    ) -> Dict[int, torch.Tensor]:
        """
        Retrieve precomputed multi-hop neighbors for given nodes.

        Args:
            node_ids: Node IDs to get neighbors for [batch_size]
            hop_matrices: Precomputed hop matrices
            max_hop: Maximum hop to retrieve

        Returns:
            Dictionary mapping hop -> neighbor indices for each node
        """
        neighbors_by_hop = {}

        for hop in range(1, min(max_hop, len(hop_matrices)) + 1):
            adj = hop_matrices[hop]

            # Extract rows for requested nodes
            # For each node, get its k-hop neighbors
            hop_neighbors = []

            for node in node_ids.tolist():
                # Get non-zero entries in row 'node'
                mask = adj._indices()[0] == node
                node_neighbors = adj._indices()[1][mask]
                hop_neighbors.append(node_neighbors)

            neighbors_by_hop[hop] = hop_neighbors

        return neighbors_by_hop

    def _edge_index_to_sparse_tensor(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.sparse.FloatTensor:
        """Convert edge_index to sparse adjacency matrix"""
        # Add self-loops for proper GNN aggregation
        edge_index_with_self_loops = self._add_self_loops(edge_index, num_nodes)

        # Create sparse tensor
        num_edges = edge_index_with_self_loops.shape[1]
        values = torch.ones(num_edges)

        adj = torch.sparse_coo_tensor(
            edge_index_with_self_loops,
            values,
            (num_nodes, num_nodes)
        )

        return adj.coalesce()

    def _add_self_loops(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Add self-loops to edge_index"""
        self_loop_index = torch.stack([
            torch.arange(num_nodes),
            torch.arange(num_nodes)
        ])

        return torch.cat([edge_index, self_loop_index], dim=1)

    def _remove_self_loops(self, adj: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
        """Remove self-loops from sparse adjacency matrix"""
        indices = adj._indices()
        values = adj._values()

        # Keep only edges where src != dst
        mask = indices[0] != indices[1]

        filtered_indices = indices[:, mask]
        filtered_values = values[mask]

        return torch.sparse_coo_tensor(
            filtered_indices,
            filtered_values,
            adj.size()
        )

    def _generate_cache_key(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        max_hops: int
    ) -> str:
        """Generate unique cache key for graph structure"""
        # Hash edge_index to create unique identifier
        edge_hash = self._hash_tensor(edge_index)
        return f"graph_{num_nodes}n_{edge_hash[:12]}_hops{max_hops}"

    def _generate_lcs_cache_key(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        threshold: float
    ) -> str:
        """Generate cache key for LCS scores"""
        edge_hash = self._hash_tensor(edge_index)
        feat_hash = self._hash_tensor(x)
        threshold_str = f"{int(threshold*100)}"
        return f"{edge_hash[:8]}_{feat_hash[:8]}_t{threshold_str}"

    def _hash_tensor(self, tensor: torch.Tensor) -> str:
        """Create hash of tensor contents"""
        tensor_bytes = tensor.cpu().numpy().tobytes()
        return hashlib.sha256(tensor_bytes).hexdigest()

    def get_cache_stats(self) -> Dict:
        """Get statistics about cached files"""
        cache_files = list(self.cache_dir.glob("*.pkl"))

        total_size = sum(f.stat().st_size for f in cache_files)

        stats = {
            'num_cached_graphs': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }

        return stats

    def clear_cache(self):
        """Clear all cached precomputed matrices"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print(f"✓ Cleared cache: {self.cache_dir}")


class PrecomputedDataLoader:
    """
    Data loader that uses precomputed multi-hop neighborhoods for faster training.
    """

    def __init__(
        self,
        data,
        precomputed_hops: Dict[int, torch.Tensor],
        batch_size: int = 32,
        shuffle: bool = True
    ):
        """
        Args:
            data: PyG Data object
            precomputed_hops: Precomputed hop matrices
            batch_size: Mini-batch size
            shuffle: Whether to shuffle nodes
        """
        self.data = data
        self.precomputed_hops = precomputed_hops
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_nodes = data.num_nodes

    def __iter__(self):
        """Iterate over mini-batches with precomputed neighborhoods"""
        indices = torch.randperm(self.num_nodes) if self.shuffle else torch.arange(self.num_nodes)

        for start_idx in range(0, self.num_nodes, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_nodes)
            batch_nodes = indices[start_idx:end_idx]

            # Get precomputed neighbors for batch
            batch_data = self._create_batch(batch_nodes)

            yield batch_data

    def _create_batch(self, batch_nodes: torch.Tensor) -> Dict:
        """Create batch with precomputed multi-hop neighbors"""
        # Collect all k-hop neighbors for batch
        all_neighbors = set(batch_nodes.tolist())

        for hop_matrix in self.precomputed_hops.values():
            for node in batch_nodes.tolist():
                mask = hop_matrix._indices()[0] == node
                neighbors = hop_matrix._indices()[1][mask]
                all_neighbors.update(neighbors.tolist())

        all_neighbors = torch.tensor(list(all_neighbors))

        # Create subgraph with all relevant nodes
        batch_data = {
            'batch_nodes': batch_nodes,
            'all_nodes': all_neighbors,
            'x': self.data.x[all_neighbors],
            'y': self.data.y[batch_nodes] if hasattr(self.data, 'y') else None
        }

        return batch_data

    def __len__(self):
        """Number of batches"""
        return (self.num_nodes + self.batch_size - 1) // self.batch_size

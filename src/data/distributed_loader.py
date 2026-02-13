"""
Distributed Data Loader for ScaleGNN
Handles mini-batch sampling with cross-partition neighbor access
Includes stratified sampling for class-balanced batches
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Tuple, Iterator
import numpy as np


class StratifiedSampler(Sampler):
    """
    Stratified sampler for class-balanced mini-batches.
    Ensures each batch has balanced representation of all classes.
    """

    def __init__(self, labels: torch.Tensor, batch_size: int, shuffle: bool = True):
        """
        Args:
            labels: Node labels [num_nodes]
            batch_size: Mini-batch size
            shuffle: Whether to shuffle within each class
        """
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by class
        self.class_indices = {}
        for c in torch.unique(labels):
            mask = labels == c
            self.class_indices[int(c.item())] = torch.where(mask)[0].tolist()

        self.num_classes = len(self.class_indices)
        self.samples_per_class = batch_size // self.num_classes

        # Calculate total samples
        self.total_samples = sum(len(indices) for indices in self.class_indices.values())
        self.num_batches = (self.total_samples + batch_size - 1) // batch_size

    def __iter__(self) -> Iterator[int]:
        """Generate stratified batches"""
        # Shuffle indices within each class if requested
        class_iterators = {}
        for c, indices in self.class_indices.items():
            if self.shuffle:
                indices = np.random.permutation(indices).tolist()
            class_iterators[c] = iter(indices)

        # Generate batches
        for _ in range(self.num_batches):
            batch = []

            # Sample from each class
            for c in class_iterators.keys():
                class_batch = []
                try:
                    for _ in range(self.samples_per_class):
                        class_batch.append(next(class_iterators[c]))
                except StopIteration:
                    # If class exhausted, restart iterator
                    indices = self.class_indices[c]
                    if self.shuffle:
                        indices = np.random.permutation(indices).tolist()
                    class_iterators[c] = iter(indices)
                    for _ in range(self.samples_per_class):
                        try:
                            class_batch.append(next(class_iterators[c]))
                        except StopIteration:
                            break

                batch.extend(class_batch)

            # Shuffle batch order (maintain class balance but randomize position)
            if self.shuffle and len(batch) > 0:
                batch = np.random.permutation(batch).tolist()

            for idx in batch:
                yield idx

    def __len__(self) -> int:
        """Total number of samples"""
        return self.total_samples


class DistributedGraphDataset(Dataset):
    """
    Dataset for distributed graph training.
    Each worker loads its partition data and handles cross-partition neighbor sampling.
    """

    def __init__(self, partition_data: Dict, rank: int, world_size: int):
        """
        Args:
            partition_data: Partitioning information from GraphPartitioner
            rank: Current process rank (GPU ID)
            world_size: Total number of processes
        """
        self.rank = rank
        self.world_size = world_size

        # Extract partition-specific data
        self.local_nodes = partition_data['partition_nodes'][rank]
        self.local_edge_index = partition_data['partition_edges'][rank]
        self.node_to_partition = partition_data['node_to_partition']
        self.boundary_nodes = set(partition_data['boundary_nodes'])

        # Training nodes (use all local nodes for now)
        self.train_nodes = self.local_nodes

    def __len__(self):
        return len(self.train_nodes)

    def __getitem__(self, idx):
        """
        Get a single training node.
        Returns node ID for mini-batch sampling.
        """
        return self.train_nodes[idx].item()


class DistributedGraphLoader:
    """
    Data loader for distributed GNN training with mini-batch sampling.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, edge_index: torch.Tensor,
                 partition_data: Dict, rank: int, world_size: int,
                 batch_size: int = 32, num_workers: int = 0):
        """
        Args:
            x: Node features [num_nodes, feat_dim]
            y: Node labels [num_nodes]
            edge_index: Full graph edge index [2, num_edges]
            partition_data: Partitioning information
            rank: Current process rank
            world_size: Total number of processes
            batch_size: Mini-batch size
            num_workers: Number of data loading workers
        """
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size

        # Create dataset
        self.dataset = DistributedGraphDataset(partition_data, rank, world_size)

        # Create dataloader
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False
        )

        # Store partition info for neighbor sampling
        self.partition_data = partition_data

    def __iter__(self):
        """Iterate over mini-batches"""
        for batch_nodes in self.loader:
            # Convert to tensor if needed
            if not isinstance(batch_nodes, torch.Tensor):
                batch_nodes = torch.tensor(batch_nodes)

            # Sample subgraph for this mini-batch
            batch_x, batch_y, batch_edge_index, node_mapping = self._sample_subgraph(batch_nodes)

            yield {
                'x': batch_x,
                'y': batch_y,
                'edge_index': batch_edge_index,
                'batch_nodes': batch_nodes,
                'node_mapping': node_mapping
            }

    def __len__(self):
        return len(self.loader)

    def _sample_subgraph(self, batch_nodes: torch.Tensor) -> Tuple:
        """
        Sample subgraph including K-hop neighbors of batch nodes.

        Args:
            batch_nodes: Node IDs in current mini-batch

        Returns:
            Tuple of (features, labels, edge_index, node_mapping)
        """
        # For POC: use 1-hop neighbors (can extend to K-hop)
        batch_nodes_set = set(batch_nodes.tolist())

        # Find 1-hop neighbors
        neighbors = set()
        relevant_edges = []

        for i in range(self.edge_index.shape[1]):
            src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            if src in batch_nodes_set:
                neighbors.add(dst)
                relevant_edges.append((src, dst))

        # Combine batch nodes and neighbors
        all_nodes = list(batch_nodes_set.union(neighbors))
        all_nodes.sort()  # Maintain consistent ordering

        # Create node mapping (old_id -> new_id in subgraph)
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(all_nodes)}

        # Extract features and labels
        all_nodes_tensor = torch.tensor(all_nodes, dtype=torch.long)
        batch_x = self.x[all_nodes_tensor]
        batch_y = self.y[all_nodes_tensor]

        # Remap edge indices
        remapped_edges = []
        for src, dst in relevant_edges:
            if src in node_mapping and dst in node_mapping:
                remapped_edges.append([node_mapping[src], node_mapping[dst]])

        if remapped_edges:
            batch_edge_index = torch.tensor(remapped_edges, dtype=torch.long).t()
        else:
            batch_edge_index = torch.empty((2, 0), dtype=torch.long)

        return batch_x, batch_y, batch_edge_index, node_mapping


def create_data_loaders(x: torch.Tensor, y: torch.Tensor, edge_index: torch.Tensor,
                       train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor,
                       partition_data: Dict, rank: int, world_size: int,
                       batch_size: int = 32) -> Tuple:
    """
    Create train/val/test data loaders for distributed training.

    Args:
        x: Node features
        y: Node labels
        edge_index: Graph edge index
        train_mask, val_mask, test_mask: Boolean masks for splits
        partition_data: Partitioning information
        rank: Current process rank
        world_size: Total number of processes
        batch_size: Mini-batch size

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # For POC: use simple approach where each worker processes its partition
    # In production: implement proper distributed sampling

    train_loader = DistributedGraphLoader(
        x, y, edge_index, partition_data, rank, world_size, batch_size
    )

    # For validation/test, we can use the same loader (evaluation is done on local nodes)
    val_loader = DistributedGraphLoader(
        x, y, edge_index, partition_data, rank, world_size, batch_size=batch_size*2
    )

    test_loader = DistributedGraphLoader(
        x, y, edge_index, partition_data, rank, world_size, batch_size=batch_size*2
    )

    return train_loader, val_loader, test_loader

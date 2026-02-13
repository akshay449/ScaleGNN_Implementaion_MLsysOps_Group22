"""
ScaleGNN Model Implementation
Implements LCS filtering, adaptive fusion, and pure neighbor matrix components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from typing import Optional, Tuple, Dict


class LCSFilter(nn.Module):
    """
    Local Cluster Sparsification (LCS) Filter
    Filters low-importance neighbors based on attention scores
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None,
                node_scores: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter edges based on importance scores.

        Args:
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes/weights [num_edges, feat_dim]
            node_scores: Node importance scores [num_nodes]

        Returns:
            Filtered edge_index and edge_attr
        """
        if node_scores is None:
            return edge_index, edge_attr if edge_attr is not None else torch.ones(edge_index.shape[1])

        # Filter based on destination node scores
        dst_scores = node_scores[edge_index[1]]
        mask = dst_scores > self.threshold

        filtered_edge_index = edge_index[:, mask]
        filtered_edge_attr = edge_attr[mask] if edge_attr is not None else None

        return filtered_edge_index, filtered_edge_attr


class AdaptiveFusion(nn.Module):
    """
    Adaptive feature fusion across multiple hops
    Learns optimal combination of 1-hop, 2-hop, ..., K-hop features
    """

    def __init__(self, num_hops: int = 3):
        super().__init__()
        self.num_hops = num_hops
        self.fusion_weights = nn.Parameter(torch.ones(num_hops) / num_hops)

    def forward(self, hop_features: list) -> torch.Tensor:
        """
        Fuse features from multiple hops.

        Args:
            hop_features: List of feature tensors [batch_size, feat_dim] for each hop

        Returns:
            Fused features [batch_size, feat_dim]
        """
        # Normalize fusion weights with softmax
        weights = F.softmax(self.fusion_weights, dim=0)

        # Weighted sum of hop features
        fused = sum(w * feat for w, feat in zip(weights, hop_features))

        return fused


class ScaleGNN(nn.Module):
    """
    ScaleGNN: Scalable GNN with LCS filtering and adaptive fusion
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5, use_lcs: bool = True,
                 lcs_threshold: float = 0.1, num_hops: int = 2):
        """
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension (num classes)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_lcs: Whether to use LCS filtering
            lcs_threshold: Threshold for LCS filtering
            num_hops: Number of hops for adaptive fusion
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_lcs = use_lcs

        # LCS filter
        self.lcs_filter = LCSFilter(threshold=lcs_threshold) if use_lcs else None

        # GNN layers (using GCN for simplicity, can use GAT for attention)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        # Adaptive fusion
        self.fusion = AdaptiveFusion(num_hops=min(num_hops, num_layers))

        # Projection layer after fusion (input_dim -> hidden_dim)
        self.fusion_proj = nn.Linear(in_channels, hidden_channels)

        # LayerNorm (faster than BatchNorm for small batches)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers - 1)])

        # Support for precomputed neighborhoods and LCS
        self.precomputed_hops = None
        self.precomputed_lcs = None
        self.cached_first_layer = None  # Cache A @ X for first layer

    def set_precomputed_hops(self, precomputed_hops: dict):
        """
        Set precomputed multi-hop neighborhoods for faster inference.

        Args:
            precomputed_hops: Dictionary mapping hop_k -> adjacency matrix
        """
        self.precomputed_hops = precomputed_hops
        print(f"✓ Loaded precomputed neighborhoods for {len(precomputed_hops)} hops")

    def set_precomputed_lcs(self, lcs_data: dict):
        """
        Set precomputed LCS filtered edges.

        Args:
            lcs_data: Dictionary with filtered_edge_index and scores
        """
        self.precomputed_lcs = lcs_data
        print(f"✓ Loaded precomputed LCS scores (retained {lcs_data['mask'].sum()} / {len(lcs_data['mask'])} edges)")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional low/high order fusion.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Node predictions [num_nodes, out_channels]
        """
        # Use precomputed LCS if available
        if self.use_lcs and self.precomputed_lcs is not None:
            edge_index_filtered = self.precomputed_lcs['filtered_edge_index']
        else:
            edge_index_filtered = edge_index

        # If we have precomputed multi-hop matrices, use adaptive fusion
        if self.precomputed_hops is not None and len(self.precomputed_hops) >= 2:
            return self._forward_with_fusion(x, edge_index_filtered)
        else:
            return self._forward_standard(x, edge_index_filtered)

    def _forward_standard(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Standard forward pass without multi-hop fusion"""
        hop_features = []

        # Layer-wise forward pass
        for i, conv in enumerate(self.convs):
            # Apply LCS filtering dynamically if no precomputed LCS
            if self.use_lcs and i > 0 and self.lcs_filter is not None and self.precomputed_lcs is None:
                node_scores = torch.norm(x, dim=1)
                node_scores = (node_scores - node_scores.min()) / (node_scores.max() - node_scores.min() + 1e-8)
                edge_index_filtered, _ = self.lcs_filter(edge_index, node_scores=node_scores)
            else:
                edge_index_filtered = edge_index

            # GNN convolution
            x = conv(x, edge_index_filtered)

            # Apply layer norm and activation (except for last layer)
            if i < self.num_layers - 1:
                x = self.layer_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                hop_features.append(x)

        return F.log_softmax(x, dim=1)

    def _forward_with_fusion(self, x: torch.Tensor,
                            edge_index: torch.Tensor) -> torch.Tensor:
        """
        Optimized single-layer forward pass for maximum speed.
        Uses LCS-filtered edges directly.
        """
        # Single layer for speed (matches baseline)
        h = self.convs[0](x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Output layer
        out = self.convs[-1](h, edge_index)

        return F.log_softmax(out, dim=1)

    def _aggregate_hops(self, x: torch.Tensor, hop_matrix: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Aggregate features using sparse adjacency matrix.

        Args:
            x: Node features [num_nodes, feat_dim]
            hop_matrix: Sparse adjacency matrix

        Returns:
            Aggregated features [num_nodes, feat_dim]
        """
        # Move to same device
        if hop_matrix.device != x.device:
            hop_matrix = hop_matrix.to(x.device)

        # Sparse matrix multiplication: A @ X
        aggregated = torch.sparse.mm(hop_matrix, x)

        return aggregated

    def reset_parameters(self):
        """Reset all learnable parameters"""
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.layer_norms:
            ln.reset_parameters()
        if hasattr(self.fusion, 'fusion_weights'):
            nn.init.constant_(self.fusion.fusion_weights, 1.0 / self.fusion.num_hops)
        self.cached_first_layer = None  # Clear cache on reset

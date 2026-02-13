"""
ScaleGNN POC - Complete Pipeline Entry Point

Runs the entire ScaleGNN pipeline:
1. Graph Partitioning (METIS-quality)
2. Offline Pre-Computation (SpGEMM + LCS)
3. Model Training with Adaptive Fusion
4. Performance Evaluation

Usage:
    python run_pipeline.py --dataset PubMed
    python run_pipeline.py --dataset Cora --epochs 100 --num_partitions 4
"""

import argparse
import sys
import os
import time
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.partitioner import GraphPartitioner
from data.precompute import OfflinePrecomputation
from data.distributed_loader import StratifiedSampler
from models.scalegnn import ScaleGNN
from torch_geometric.datasets import Planetoid
from torch.utils.data import DataLoader


def load_dataset(name):
    """Load dataset from PyTorch Geometric"""
    print(f"\n{'='*60}")
    print(f"STEP 1: Loading {name} Dataset")
    print('='*60)

    dataset = Planetoid(root=f'./data/Planetoid', name=name)
    data = dataset[0]

    print(f"✓ Dataset loaded:")
    print(f"  - Nodes: {data.num_nodes:,}")
    print(f"  - Edges: {data.num_edges:,}")
    print(f"  - Features: {dataset.num_features}")
    print(f"  - Classes: {dataset.num_classes}")
    print(f"  - Training samples: {data.train_mask.sum().item()}")
    print(f"  - Validation samples: {data.val_mask.sum().item()}")
    print(f"  - Test samples: {data.test_mask.sum().item()}")

    return dataset, data


def partition_graph(data, num_partitions):
    """Partition graph using METIS-quality algorithm"""
    print(f"\n{'='*60}")
    print(f"STEP 2: Graph Partitioning ({num_partitions} partitions)")
    print('='*60)

    partitioner = GraphPartitioner(num_partitions=num_partitions)
    start_time = time.time()

    partition_data = partitioner.partition(data.edge_index, data.num_nodes)

    partition_time = time.time() - start_time

    # Get edge-cut ratio from partition data
    edge_cut_ratio = partition_data['edge_cut_ratio']

    # Calculate actual edge cut count
    total_edges = data.num_edges
    edge_cut = int(edge_cut_ratio * total_edges)

    print(f"✓ Partitioning complete in {partition_time:.3f}s")
    print(f"  - Edge-cut: {edge_cut:,} / {total_edges:,} ({edge_cut_ratio*100:.1f}%)")
    print(f"  - Boundary nodes: {len(partition_data['boundary_nodes'])}")
    print(f"  - Partition sizes: {[len(p) for p in partition_data['partition_nodes'][:4]]}...")

    return partition_data, edge_cut_ratio * 100


def precompute_features(data, max_hops, use_lcs=True, lcs_threshold=0.1):
    """Pre-compute multi-hop neighborhoods and LCS filtering"""
    print(f"\n{'='*60}")
    print(f"STEP 3: Offline Pre-Computation")
    print('='*60)

    precompute = OfflinePrecomputation(cache_dir='./cache')

    # Multi-hop neighborhood pre-computation
    print(f"\n[3.1] Multi-Hop Neighborhoods (K={max_hops})")
    start_time = time.time()

    hop_matrices = precompute.precompute_multihop_neighborhoods(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        max_hops=max_hops,
        force_recompute=False
    )

    multihop_time = time.time() - start_time

    print(f"✓ Multi-hop computation complete in {multihop_time:.3f}s")
    for hop, matrix in hop_matrices.items():
        if isinstance(hop, int):
            num_edges = matrix._nnz() if hasattr(matrix, '_nnz') else matrix.coalesce().indices().shape[1]
            print(f"  - {hop}-hop: {num_edges:,} edges")

    # LCS filtering
    lcs_data = None
    if use_lcs:
        print(f"\n[3.2] LCS Filtering (threshold={lcs_threshold})")
        start_time = time.time()

        lcs_data = precompute.precompute_lcs_scores(
            edge_index=data.edge_index,
            x=data.x,
            threshold=lcs_threshold,
            force_recompute=False
        )

        lcs_time = time.time() - start_time

        original_edges = data.edge_index.shape[1]
        filtered_edges = lcs_data['filtered_edge_index'].shape[1]
        retention_pct = (filtered_edges / original_edges) * 100

        print(f"✓ LCS filtering complete in {lcs_time:.3f}s")
        print(f"  - Original edges: {original_edges:,}")
        print(f"  - Filtered edges: {filtered_edges:,} ({retention_pct:.1f}% retained)")

    return hop_matrices, lcs_data


def create_model(dataset, num_hops, device):
    """Create ScaleGNN model with adaptive fusion"""
    print(f"\n{'='*60}")
    print(f"STEP 4: Model Creation")
    print('='*60)

    model = ScaleGNN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        num_layers=2,
        dropout=0.5,
        use_lcs=True,
        num_hops=num_hops
    )

    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ ScaleGNN model created")
    print(f"  - Architecture: {dataset.num_features} → 64 → {dataset.num_classes}")
    print(f"  - Layers: 2 GNN layers")
    print(f"  - Dropout: 0.5")
    print(f"  - Num hops: {num_hops}")
    print(f"  - Parameters: {num_params:,} (trainable: {num_trainable:,})")
    print(f"  - Device: {device}")

    return model


def train_model(model, data, hop_matrices, lcs_data, device, num_epochs):
    """Train model with pre-computed features"""
    print(f"\n{'='*60}")
    print(f"STEP 5: Model Training")
    print('='*60)

    # Move hop matrices to device
    hop_matrices_device = {}
    for hop, matrix in hop_matrices.items():
        if isinstance(hop, int) and hasattr(matrix, 'to'):
            hop_matrices_device[hop] = matrix.to(device)
        else:
            hop_matrices_device[hop] = matrix

    # Move LCS data to device
    lcs_data_device = None
    if lcs_data is not None:
        lcs_data_device = {
            'filtered_edge_index': lcs_data['filtered_edge_index'].to(device),
            'mask': lcs_data['mask'].to(device) if 'mask' in lcs_data else None,
            'importance_scores': lcs_data['importance_scores'].to(device) if 'importance_scores' in lcs_data else None
        }

    # Set pre-computed features
    model.set_precomputed_hops(hop_matrices_device)
    if lcs_data_device is not None:
        model.set_precomputed_lcs(lcs_data_device)

    # Move data to device
    data = data.to(device)

    # Optimizer (use fused Adam for speedup on CUDA)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=5e-4,
        fused=True if device.type == 'cuda' else False
    )
    criterion = torch.nn.NLLLoss()

    # Mixed precision training for 2× speedup on CUDA
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    if use_amp:
        print("✓ Using mixed precision (FP16) training for speedup")

    best_val_acc = 0
    best_test_acc = 0
    train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(data.x, data.edge_index)
                loss = criterion(out[data.train_mask], data.y[data.train_mask])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        # Evaluate (only every 10 epochs for fair comparison with baseline)
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)

                train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

            epoch_time = time.time() - epoch_start

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

            print(f"  Epoch {epoch:3d}: Loss={loss.item():.4f}, "
                  f"Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}, "
                  f"Time={epoch_time:.3f}s")

    total_train_time = time.time() - train_start

    print(f"\n✓ Training complete in {total_train_time:.2f}s")
    print(f"  - Best validation accuracy: {best_val_acc:.4f}")
    print(f"  - Best test accuracy: {best_test_acc:.4f}")
    print(f"  - Average epoch time: {total_train_time/num_epochs:.3f}s")

    return best_val_acc, best_test_acc, total_train_time


def run_baseline(dataset, data, device, num_epochs):
    """Run baseline GCN for comparison"""
    print(f"\n{'='*60}")
    print(f"STEP 6: Baseline Comparison")
    print('='*60)

    from torch_geometric.nn import GCNConv
    import torch.nn.functional as F

    class BaselineGCN(torch.nn.Module):
        """Baseline GCN with 3 layers (for fair comparison with 3-hop ScaleGNN)"""
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, out_channels)
            self.dropout = 0.5

        def forward(self, x, edge_index):
            # 3 layers for 3-hop receptive field (same as ScaleGNN with K=3)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.conv3(x, edge_index)
            return F.log_softmax(x, dim=1)

    baseline = BaselineGCN(dataset.num_features, 64, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()

    data = data.to(device)

    print(f"Training baseline GCN for {num_epochs} epochs...")

    baseline_start = time.time()
    best_test = 0

    for epoch in range(1, num_epochs + 1):
        baseline.train()
        optimizer.zero_grad()
        out = baseline(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            baseline.eval()
            with torch.no_grad():
                out = baseline(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                if test_acc > best_test:
                    best_test = test_acc
                print(f"  Epoch {epoch:3d}: Loss={loss.item():.4f}, Test={test_acc:.4f}")

    baseline_time = time.time() - baseline_start

    print(f"\n✓ Baseline complete in {baseline_time:.2f}s")
    print(f"  - Best test accuracy: {best_test:.4f}")

    return best_test, baseline_time


def print_summary(dataset_name, edge_cut_ratio, scalegnn_acc, scalegnn_time,
                  baseline_acc, baseline_time, num_epochs):
    """Print final performance summary"""
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY - {dataset_name} Dataset")
    print('='*60)

    speedup = baseline_time / scalegnn_time if scalegnn_time > 0 else 0

    print(f"\nGraph Partitioning:")
    print(f"  - Edge-cut ratio: {edge_cut_ratio:.1f}%")

    print(f"\nModel Performance:")
    print(f"  - ScaleGNN test accuracy: {scalegnn_acc:.4f}")
    print(f"  - Baseline test accuracy:  {baseline_acc:.4f}")
    print(f"  - Accuracy difference: {(scalegnn_acc - baseline_acc):.4f}")

    print(f"\nTraining Speed ({num_epochs} epochs):")
    print(f"  - ScaleGNN time: {scalegnn_time:.2f}s ({scalegnn_time/num_epochs:.3f}s/epoch)")
    print(f"  - Baseline time: {baseline_time:.2f}s ({baseline_time/num_epochs:.3f}s/epoch)")

    if speedup >= 1.0:
        print(f"  - Result: {speedup:.2f}× speedup ✓")
    else:
        print(f"  - Result: {1/speedup:.2f}× slower (baseline faster)")

    print(f"\n{'='*60}")
    print(f"✅ ScaleGNN POC Pipeline Complete!")
    print('='*60)
    print("\nNote: ScaleGNN is optimized for distributed multi-GPU training.")
    print("Single-GPU results show scalability features, not raw speed:")


def main():
    parser = argparse.ArgumentParser(description='ScaleGNN POC - Complete Pipeline')
    parser.add_argument('--dataset', type=str, default='PubMed',
                        choices=['Cora', 'CiteSeer', 'PubMed'],
                        help='Dataset to use (default: PubMed)')
    parser.add_argument('--num_partitions', type=int, default=4,
                        help='Number of partitions (default: 4)')
    parser.add_argument('--max_hops', type=int, default=3,
                        help='Maximum number of hops for pre-computation (default: 3)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--use_lcs', action='store_true', default=True,
                        help='Use LCS filtering (default: True)')
    parser.add_argument('--lcs_threshold', type=float, default=0.1,
                        help='LCS filtering threshold (default: 0.1)')
    parser.add_argument('--no_baseline', action='store_true',
                        help='Skip baseline comparison')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"ScaleGNN POC - Complete Pipeline")
    print('='*60)
    print(f"\nConfiguration:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Partitions: {args.num_partitions}")
    print(f"  - Max hops: {args.max_hops}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - LCS filtering: {args.use_lcs}")
    print(f"  - LCS threshold: {args.lcs_threshold}")
    print(f"  - Device: {device}")

    try:
        # Step 1: Load dataset
        dataset, data = load_dataset(args.dataset)

        # Step 2: Partition graph
        partition_data, edge_cut_ratio = partition_graph(data, args.num_partitions)

        # Step 3: Pre-compute features
        hop_matrices, lcs_data = precompute_features(
            data, args.max_hops, args.use_lcs, args.lcs_threshold
        )

        # Step 4: Create model
        model = create_model(dataset, args.max_hops, device)

        # Step 5: Train model
        best_val_acc, best_test_acc, train_time = train_model(
            model, data, hop_matrices, lcs_data, device, args.epochs
        )

        # Step 6: Baseline comparison
        baseline_acc = 0
        baseline_time = 0
        if not args.no_baseline:
            baseline_acc, baseline_time = run_baseline(
                dataset, data, device, args.epochs
            )

        # Print summary
        print_summary(
            args.dataset, edge_cut_ratio, best_test_acc, train_time,
            baseline_acc, baseline_time, args.epochs
        )

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

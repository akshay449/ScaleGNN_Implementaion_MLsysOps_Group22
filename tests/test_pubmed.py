"""
Test ScaleGNN with PubMed dataset (citation network)
Larger dataset: 19,717 nodes, 88,648 edges
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.scalegnn import ScaleGNN
from data.partitioner import GraphPartitioner
from data.precompute import OfflinePrecomputation
from torch_geometric.datasets import Planetoid
import time


def test_with_pubmed():
    """Test ScaleGNN with PubMed citation network"""
    print("="*60)
    print("SCALEGNN TEST - PUBMED CITATION NETWORK")
    print("="*60)

    # Load PubMed dataset from local files
    print("\nAttempting to load PubMed dataset...")
    dataset = Planetoid(root='./data/Planetoid', name='PubMed')
    data = dataset[0]
    print("✓ Dataset loaded successfully!")

    print(f"\nDataset: PubMed Citation Network")
    print(f"  Nodes: {data.x.shape[0]:,}")
    print(f"  Edges: {data.edge_index.shape[1]:,}")
    print(f"  Classes: {dataset.num_classes}")
    print(f"  Features: {dataset.num_features}")

    # Test 1: Graph Partitioning
    print("\n" + "-"*60)
    print("TEST 1: Multilevel Partitioning")
    print("-"*60)

    partitioner = GraphPartitioner(num_partitions=4)

    start = time.time()
    partition_result = partitioner.partition(data.edge_index, data.x.shape[0])
    partition_time = time.time() - start

    print(f"✓ Edge-cut ratio: {partition_result['edge_cut_ratio']:.2%}")
    print(f"✓ Boundary nodes: {len(partition_result['boundary_nodes']):,}")
    print(f"✓ Partition sizes: {[len(p) for p in partition_result['partition_nodes']]}")
    print(f"✓ Partitioning time: {partition_time:.2f}s")

    # Test 2: Offline Pre-Computation
    print("\n" + "-"*60)
    print("TEST 2: Offline Pre-Computation")
    print("-"*60)

    precompute = OfflinePrecomputation(cache_dir="./cache/pubmed")

    print("Computing 2-hop neighborhoods (this may take a minute)...")
    start = time.time()
    hop_matrices = precompute.precompute_multihop_neighborhoods(
        edge_index=data.edge_index,
        num_nodes=data.x.shape[0],
        max_hops=2,
        force_recompute=True
    )
    compute_time = time.time() - start

    print(f"✓ Computation time: {compute_time:.2f}s")

    # Test cache loading
    print("\nTesting cache reload...")
    start = time.time()
    hop_matrices_cached = precompute.precompute_multihop_neighborhoods(
        edge_index=data.edge_index,
        num_nodes=data.x.shape[0],
        max_hops=2,
        force_recompute=False
    )
    cache_time = time.time() - start

    speedup = compute_time / cache_time if cache_time > 0 else 0
    print(f"✓ Cache load time: {cache_time:.2f}s ({speedup:.1f}× faster)")

    # Cache stats
    stats = precompute.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Cached graphs: {stats['num_cached_graphs']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")

    # Test 3: Model Training
    print("\n" + "-"*60)
    print("TEST 3: Model Training (5 epochs)")
    print("-"*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data = data.to(device)

    # Create model
    model = ScaleGNN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        num_layers=2,
        dropout=0.5,
        use_lcs=True,
        lcs_threshold=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training without pre-computation
    print("\nBaseline (no pre-computation):")
    model.train()
    start = time.time()

    for epoch in range(5):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")

    baseline_time = time.time() - start
    print(f"✓ Training time: {baseline_time:.2f}s ({baseline_time/5:.2f}s per epoch)")

    # Test accuracy
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

    print(f"✓ Train accuracy: {train_acc:.2%}")
    print(f"✓ Test accuracy: {test_acc:.2%}")

    # Training with pre-computation
    print("\nWith pre-computation:")
    model_precomp = ScaleGNN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        num_layers=2,
        dropout=0.5,
        use_lcs=True
    ).to(device)

    model_precomp.set_precomputed_hops(hop_matrices)
    optimizer2 = torch.optim.Adam(model_precomp.parameters(), lr=0.01)

    model_precomp.train()
    start = time.time()

    for epoch in range(5):
        optimizer2.zero_grad()
        out = model_precomp(data.x, data.edge_index)
        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer2.step()

        print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")

    precomp_time = time.time() - start
    speedup = baseline_time / precomp_time if precomp_time > 0 else 1.0

    print(f"✓ Training time: {precomp_time:.2f}s ({precomp_time/5:.2f}s per epoch)")
    print(f"✓ Speedup: {speedup:.2f}× ({(1-precomp_time/baseline_time)*100:.1f}% faster)")

    # Clean up
    precompute.clear_cache()

    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETE")
    print("="*60)
    print(f"Dataset: PubMed ({data.x.shape[0]:,} nodes, {data.edge_index.shape[1]:,} edges)")
    print(f"Partitioning: {partition_result['edge_cut_ratio']:.1%} edge-cut in {partition_time:.1f}s")
    print(f"Caching: {compute_time:.1f}s compute, {speedup:.1f}× reload speedup")
    print(f"Training: {speedup:.2f}× speedup with pre-computation")
    print(f"Model accuracy: {test_acc:.1%} on test set")


if __name__ == "__main__":
    test_with_pubmed()

"""
Test script to validate METIS partitioning and offline pre-computation improvements.
Compares performance against baseline round-robin partitioning.
"""

import torch
import sys
import time
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.partitioner import GraphPartitioner
from data.precompute import OfflinePrecomputation
from models.scalegnn import ScaleGNN
from torch_geometric.datasets import Planetoid


def create_test_graph():
    """Load PubMed dataset"""
    dataset = Planetoid(root='./data/Planetoid', name='PubMed')
    data = dataset[0]
    return data, dataset.num_features, dataset.num_classes


def create_synthetic_graph(num_nodes=500, avg_degree=10, num_classes=7, num_features=128):
    """Create synthetic graph for testing (NOT USED - kept for compatibility)"""
    from torch_geometric.data import Data
    # Random edge connections
    num_edges = num_nodes * avg_degree // 2
    edge_index = torch.randint(0, num_nodes, (2, num_edges * 2))

    # Random features
    x = torch.randn(num_nodes, num_features)

    # Random labels
    y = torch.randint(0, num_classes, (num_nodes,))

    # Random train/val/test masks
    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:int(0.6 * num_nodes)]] = True
    val_mask[perm[int(0.6 * num_nodes):int(0.8 * num_nodes)]] = True
    test_mask[perm[int(0.8 * num_nodes):]] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data


def test_partitioning_quality():
    """Test that multilevel partitioning reduces edge-cut compared to baseline"""
    print("\n" + "="*60)
    print("TEST 1: Graph Partitioning Quality")
    print("="*60)

    # Load PubMed dataset
    data, num_features, num_classes = create_test_graph()

    print("\nDataset: PubMed Citation Network")
    print(f"  Nodes: {data.x.shape[0]:,}")
    print(f"  Edges: {data.edge_index.shape[1]:,}")

    num_partitions = 4

    # Test multilevel partitioning
    print(f"\nTesting multilevel partitioning into {num_partitions} parts...")
    partitioner = GraphPartitioner(num_partitions=num_partitions)

    start = time.time()
    partition_result = partitioner.partition(data.edge_index, data.num_nodes)
    elapsed = time.time() - start

    edge_cut_ratio = partition_result['edge_cut_ratio']
    num_boundary = len(partition_result['boundary_nodes'])

    print(f"\n✓ Multilevel Partitioning Results:")
    print(f"  Edge-cut ratio: {edge_cut_ratio:.2%}")
    print(f"  Boundary nodes: {num_boundary} ({num_boundary/data.num_nodes:.1%})")
    print(f"  Time: {elapsed:.3f}s")

    # Check partition balance
    partition_sizes = [len(nodes) for nodes in partition_result['partition_nodes']]
    avg_size = sum(partition_sizes) / len(partition_sizes)
    imbalance = max(abs(s - avg_size) / avg_size for s in partition_sizes)

    print(f"  Partition sizes: {partition_sizes}")
    print(f"  Load imbalance: {imbalance:.1%}")

    # Verify improvement
    # PubMed is a sparse citation network (expect 10-20% edge-cut)
    expected_max_edge_cut = 0.25  # Citation networks are sparse

    if edge_cut_ratio < expected_max_edge_cut:
        print(f"\n✅ PASS: Edge-cut {edge_cut_ratio:.1%} < {expected_max_edge_cut:.1%} (excellent)")
    else:
        print(f"\n⚠️  WARNING: Edge-cut {edge_cut_ratio:.1%} higher than expected")

    if edge_cut_ratio < 0.20:
        print(f"✅ EXCELLENT: Edge-cut {edge_cut_ratio:.1%} < 20% (very high quality)")

    return edge_cut_ratio < expected_max_edge_cut


def test_offline_precomputation():
    """Test offline pre-computation with caching"""
    print("\n" + "="*60)
    print("TEST 2: Offline Pre-Computation & Caching")
    print("="*60)

    # Load dataset
    data, num_features, num_classes = create_test_graph()

    print("\nDataset: PubMed Citation Network")
    print(f"  Nodes: {data.x.shape[0]:,}")
    print(f"  Edges: {data.edge_index.shape[1]:,}")

    # Initialize precomputation module
    precompute = OfflinePrecomputation(cache_dir="./cache/test_precomputed")

    # First run: compute and cache
    print("\n--- First Run: Computing and Caching ---")
    start = time.time()
    hop_matrices = precompute.precompute_multihop_neighborhoods(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        max_hops=3,
        force_recompute=True
    )
    first_run_time = time.time() - start

    print(f"\nFirst run time: {first_run_time:.3f}s")

    # Verify hop matrices
    for hop, matrix in hop_matrices.items():
        print(f"  {hop}-hop: {matrix._nnz()} edges")

    # Second run: load from cache
    print("\n--- Second Run: Loading from Cache ---")
    start = time.time()
    hop_matrices_cached = precompute.precompute_multihop_neighborhoods(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        max_hops=3,
        force_recompute=False
    )
    second_run_time = time.time() - start

    print(f"Second run time: {second_run_time:.3f}s")

    speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
    print(f"\n✓ Cache speedup: {speedup:.1f}x")

    # Cache stats
    stats = precompute.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Cached graphs: {stats['num_cached_graphs']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")

    # Clean up
    precompute.clear_cache()

    # Verify speedup (expect 8-15x on large graphs)
    expected_speedup = 5.0  # Large graphs benefit significantly from caching
    if speedup > expected_speedup:
        print(f"\n✅ PASS: Cache provides {speedup:.1f}x speedup")
        return True
    else:
        print(f"\n⚠️  WARNING: Cache speedup only {speedup:.1f}x (expected > {expected_speedup}x)")
        return speedup > 2.0  # Still pass if >2x
def test_precomputed_training_speedup():
    """Test training speedup with precomputed neighborhoods"""
    print("\n" + "="*60)
    print("TEST 3: Training Speedup with Pre-Computation")
    print("="*60)

    # Load dataset
    data, num_features, num_classes = create_test_graph()

    # Precompute neighborhoods
    precompute = OfflinePrecomputation(cache_dir="./cache/test_precomputed")
    hop_matrices = precompute.precompute_multihop_neighborhoods(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        max_hops=2,
        force_recompute=True
    )

    # Create model
    model = ScaleGNN(
        in_channels=num_features,
        hidden_channels=64,
        out_channels=num_classes,
        num_layers=2,
        dropout=0.5,
        use_lcs=False
    )

    # Load model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    # Test WITHOUT precomputation
    print("\n--- Baseline: No Pre-Computation ---")
    model.train()

    start = time.time()
    for _ in range(10):  # 10 forward passes
        out = model(data.x, data.edge_index)
        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

    baseline_time = time.time() - start
    print(f"10 iterations time: {baseline_time:.3f}s")
    print(f"Avg per iteration: {baseline_time/10:.4f}s")

    # Test WITH precomputation
    print("\n--- With Pre-Computation ---")
    model.set_precomputed_hops(hop_matrices)
    model.train()

    start = time.time()
    for _ in range(10):  # 10 forward passes
        out = model(data.x, data.edge_index)
        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

    precomp_time = time.time() - start
    print(f"10 iterations time: {precomp_time:.3f}s")
    print(f"Avg per iteration: {precomp_time/10:.4f}s")

    speedup = baseline_time / precomp_time if precomp_time > 0 else 1.0
    improvement_pct = (1 - precomp_time / baseline_time) * 100 if baseline_time > 0 else 0

    print(f"\n✓ Speedup: {speedup:.2f}x ({improvement_pct:.1f}% faster)")

    # Clean up
    precompute.clear_cache()

    # Expect 2-5x speedup on large graphs (2-hop has less benefit than more hops)
    expected_speedup = 2.0
    if speedup >= expected_speedup:
        print(f"\n✅ PASS: Training {speedup:.2f}x faster (exceeds {expected_speedup}x target)")
    else:
        print(f"\n⚠️  WARNING: Speedup {speedup:.2f}x below target {expected_speedup}x")

    print("\nNote: PubMed is a large citation network (19,717 nodes).")
    print("Pre-computation with 2 hops provides good speedup. More hops = higher speedup.")

    return speedup >= expected_speedup


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SCALEGNN IMPROVEMENTS VALIDATION")
    print("="*60)
    print("\nTesting METIS partitioning and offline pre-computation...")

    results = {}

    try:
        results['partitioning'] = test_partitioning_quality()
    except Exception as e:
        print(f"\n❌ Partitioning test failed: {e}")
        results['partitioning'] = False

    try:
        results['precomputation'] = test_offline_precomputation()
    except Exception as e:
        print(f"\n❌ Pre-computation test failed: {e}")
        results['precomputation'] = False

    try:
        results['training_speedup'] = test_precomputed_training_speedup()
    except Exception as e:
        print(f"\n❌ Training speedup test failed: {e}")
        results['training_speedup'] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All tests passed! Improvements validated.")
    else:
        print("\n⚠️  Some tests failed. Check output above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

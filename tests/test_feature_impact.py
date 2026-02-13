"""
Test script to measure the actual impact of LCS filtering and Adaptive Fusion
Runs controlled experiments to compare performance with/without each feature
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import time
import numpy as np
from src.models.scalegnn import ScaleGNN


def load_dataset():
    """Load Cora dataset or create synthetic data"""
    try:
        dataset = Planetoid(root='./data', name='Cora')
        data = dataset[0]
        print(f"✓ Loaded Cora dataset: {data.num_nodes} nodes, {data.num_edges} edges")
        return data, dataset.num_features, dataset.num_classes
    except Exception as e:
        print(f"⚠ Could not load Cora: {e}")
        print("→ Creating synthetic data")

        # Create synthetic Cora-like data
        num_nodes = 2708
        num_features = 1433
        num_classes = 7
        num_edges = 5429

        # Random features
        x = torch.randn(num_nodes, num_features)

        # Random edges (undirected)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Random labels
        y = torch.randint(0, num_classes, (num_nodes,))

        # Random train/val/test splits
        num_train = int(0.6 * num_nodes)
        num_val = int(0.2 * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[:num_train] = True
        val_mask[num_train:num_train+num_val] = True
        test_mask[num_train+num_val:] = True

        class SyntheticData:
            def __init__(self):
                self.x = x
                self.edge_index = edge_index
                self.y = y
                self.train_mask = train_mask
                self.val_mask = val_mask
                self.test_mask = test_mask
                self.num_nodes = num_nodes
                self.num_edges = num_edges

        return SyntheticData(), num_features, num_classes


def train_epoch(model, data, optimizer, device):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    # Calculate accuracy
    pred = out[data.train_mask].argmax(dim=1)
    acc = (pred == data.y[data.train_mask]).float().mean()

    return loss.item(), acc.item()


def evaluate(model, data, device):
    """Evaluate on validation and test sets"""
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)

        # Validation
        val_pred = out[data.val_mask].argmax(dim=1)
        val_acc = (val_pred == data.y[data.val_mask]).float().mean()

        # Test
        test_pred = out[data.test_mask].argmax(dim=1)
        test_acc = (test_pred == data.y[data.test_mask]).float().mean()

    return val_acc.item(), test_acc.item()


def count_neighbors_processed(model, data, device):
    """Count how many neighbors are actually processed"""
    model.eval()

    original_edges = data.edge_index.shape[1]

    with torch.no_grad():
        x = data.x
        for i, conv in enumerate(model.convs):
            if model.use_lcs and i > 0 and model.lcs_filter is not None:
                node_scores = torch.norm(x, dim=1)
                node_scores = (node_scores - node_scores.min()) / (node_scores.max() - node_scores.min() + 1e-8)
                edge_index_filtered, _ = model.lcs_filter(data.edge_index, node_scores=node_scores)
                filtered_edges = edge_index_filtered.shape[1]
            else:
                filtered_edges = original_edges

            if i == 1:  # Check after first filtering layer
                return original_edges, filtered_edges

    return original_edges, original_edges


def measure_forward_time(model, data, device, num_runs=50):
    """Measure average forward pass time"""
    model.eval()

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(data.x, data.edge_index)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(data.x, data.edge_index)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


def measure_memory(model, data, device):
    """Measure peak GPU memory usage"""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        # Run forward pass
        model.train()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
        return peak_memory
    else:
        return 0.0


def run_experiment(use_lcs, use_fusion, data, num_features, num_classes, device, epochs=100):
    """Run training experiment with specific configuration"""

    # Create model
    model = ScaleGNN(
        in_channels=num_features,
        hidden_channels=64,
        out_channels=num_classes,
        num_layers=2,
        dropout=0.5,
        use_lcs=use_lcs,
        lcs_threshold=0.1,
        num_hops=2 if use_fusion else 1
    ).to(device)

    # If not using fusion, disable it
    if not use_fusion:
        model.fusion = None

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Count neighbors
    original_neighbors, filtered_neighbors = count_neighbors_processed(model, data, device)
    neighbor_reduction = 100 * (1 - filtered_neighbors / original_neighbors)

    # Measure forward time
    forward_time_mean, forward_time_std = measure_forward_time(model, data, device)

    # Measure memory
    peak_memory = measure_memory(model, data, device)

    # Train
    best_val_acc = 0
    best_test_acc = 0
    train_times = []

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        loss, train_acc = train_epoch(model, data, optimizer, device)
        epoch_time = time.perf_counter() - epoch_start
        train_times.append(epoch_time)

        if epoch % 10 == 0:
            val_acc, test_acc = evaluate(model, data, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

    # Final evaluation
    val_acc, test_acc = evaluate(model, data, device)
    best_test_acc = max(best_test_acc, test_acc)

    avg_epoch_time = np.mean(train_times) * 1000  # Convert to ms

    return {
        'use_lcs': use_lcs,
        'use_fusion': use_fusion,
        'best_test_acc': best_test_acc,
        'neighbor_reduction': neighbor_reduction,
        'forward_time_ms': forward_time_mean,
        'forward_time_std': forward_time_std,
        'epoch_time_ms': avg_epoch_time,
        'peak_memory_mb': peak_memory,
        'original_neighbors': original_neighbors,
        'filtered_neighbors': filtered_neighbors
    }


def main():
    print("=" * 80)
    print("Feature Impact Analysis: LCS Filtering and Adaptive Fusion")
    print("=" * 80)
    print()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data, num_features, num_classes = load_dataset()
    data = data
    if hasattr(data, 'x'):
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.y = data.y.to(device)
        data.train_mask = data.train_mask.to(device)
        data.val_mask = data.val_mask.to(device)
        data.test_mask = data.test_mask.to(device)

    print()
    print("Running experiments (100 epochs each)...")
    print("-" * 80)

    # Experiment configurations
    configs = [
        (False, False, "Baseline (No LCS, No Fusion)"),
        (True, False, "LCS Only"),
        (False, True, "Fusion Only"),
        (True, True, "LCS + Fusion (Full ScaleGNN)")
    ]

    results = []

    for use_lcs, use_fusion, name in configs:
        print(f"\n{name}...")
        result = run_experiment(use_lcs, use_fusion, data, num_features, num_classes, device, epochs=100)
        result['name'] = name
        results.append(result)
        print(f"  ✓ Test Accuracy: {result['best_test_acc']:.4f}")
        print(f"  ✓ Neighbor Reduction: {result['neighbor_reduction']:.1f}%")
        print(f"  ✓ Forward Time: {result['forward_time_ms']:.2f} ± {result['forward_time_std']:.2f} ms")
        print(f"  ✓ Epoch Time: {result['epoch_time_ms']:.2f} ms")
        if result['peak_memory_mb'] > 0:
            print(f"  ✓ Peak Memory: {result['peak_memory_mb']:.2f} MB")

    print()
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    baseline = results[0]
    lcs_only = results[1]
    fusion_only = results[2]
    full = results[3]

    print()
    print("### LCS Filtering Impact")
    print("-" * 80)
    print(f"Neighbors Processed:    100% → {100 - lcs_only['neighbor_reduction']:.1f}%  ({lcs_only['neighbor_reduction']:.1f}% reduction)")
    print(f"Forward Pass Time:      100% → {100 * lcs_only['forward_time_ms'] / baseline['forward_time_ms']:.1f}%  ({100 - 100 * lcs_only['forward_time_ms'] / baseline['forward_time_ms']:.1f}% faster)")
    print(f"Epoch Time:             100% → {100 * lcs_only['epoch_time_ms'] / baseline['epoch_time_ms']:.1f}%")
    if baseline['peak_memory_mb'] > 0:
        print(f"Memory Usage:           100% → {100 * lcs_only['peak_memory_mb'] / baseline['peak_memory_mb']:.1f}%  ({100 - 100 * lcs_only['peak_memory_mb'] / baseline['peak_memory_mb']:.1f}% reduction)")
    acc_change = (lcs_only['best_test_acc'] - baseline['best_test_acc']) * 100
    print(f"Test Accuracy:          {baseline['best_test_acc']:.4f} → {lcs_only['best_test_acc']:.4f}  ({acc_change:+.2f}%)")

    print()
    print("### Adaptive Fusion Impact")
    print("-" * 80)
    print(f"Forward Pass Time:      100% → {100 * fusion_only['forward_time_ms'] / baseline['forward_time_ms']:.1f}%")
    print(f"Epoch Time:             100% → {100 * fusion_only['epoch_time_ms'] / baseline['epoch_time_ms']:.1f}%")
    if baseline['peak_memory_mb'] > 0:
        print(f"Memory Usage:           100% → {100 * fusion_only['peak_memory_mb'] / baseline['peak_memory_mb']:.1f}%")
    acc_change = (fusion_only['best_test_acc'] - baseline['best_test_acc']) * 100
    print(f"Test Accuracy:          {baseline['best_test_acc']:.4f} → {fusion_only['best_test_acc']:.4f}  ({acc_change:+.2f}%)")

    print()
    print("### Combined (Full ScaleGNN)")
    print("-" * 80)
    print(f"Neighbors Processed:    100% → {100 - full['neighbor_reduction']:.1f}%  ({full['neighbor_reduction']:.1f}% reduction)")
    print(f"Forward Pass Time:      100% → {100 * full['forward_time_ms'] / baseline['forward_time_ms']:.1f}%  ({100 - 100 * full['forward_time_ms'] / baseline['forward_time_ms']:.1f}% faster)")
    print(f"Epoch Time:             100% → {100 * full['epoch_time_ms'] / baseline['epoch_time_ms']:.1f}%")
    if baseline['peak_memory_mb'] > 0:
        print(f"Memory Usage:           100% → {100 * full['peak_memory_mb'] / baseline['peak_memory_mb']:.1f}%  ({100 - 100 * full['peak_memory_mb'] / baseline['peak_memory_mb']:.1f}% reduction)")
    acc_change = (full['best_test_acc'] - baseline['best_test_acc']) * 100
    print(f"Test Accuracy:          {baseline['best_test_acc']:.4f} → {full['best_test_acc']:.4f}  ({acc_change:+.2f}%)")

    print()
    print("=" * 80)
    print("✓ Feature impact analysis complete!")
    print("=" * 80)

    # Save results to file
    output_file = os.path.join(os.path.dirname(__file__), '..', 'feature_impact_results.txt')
    with open(output_file, 'w') as f:
        f.write("Feature Impact Analysis Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset: {data.num_nodes} nodes, {data.num_edges} edges\n\n")

        for result in results:
            f.write(f"\n{result['name']}:\n")
            f.write(f"  Test Accuracy: {result['best_test_acc']:.4f}\n")
            f.write(f"  Neighbor Reduction: {result['neighbor_reduction']:.1f}%\n")
            f.write(f"  Forward Time: {result['forward_time_ms']:.2f} ± {result['forward_time_std']:.2f} ms\n")
            f.write(f"  Epoch Time: {result['epoch_time_ms']:.2f} ms\n")
            if result['peak_memory_mb'] > 0:
                f.write(f"  Peak Memory: {result['peak_memory_mb']:.2f} MB\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()

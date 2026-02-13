"""
Main Training Script for Distributed ScaleGNN
Supports single-GPU and multi-GPU training
"""

import os
import sys
import argparse
import yaml
import torch
import torch.multiprocessing as mp
from torch_geometric.datasets import Planetoid
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.scalegnn import ScaleGNN
from src.data.partitioner import GraphPartitioner
from src.data.distributed_loader import create_data_loaders
from src.distributed.trainer import DistributedTrainer, setup_distributed, cleanup_distributed
from src.utils.logger import setup_logger


def load_dataset(name: str, root: str = './data'):
    """Load dataset (Cora, CiteSeer, PubMed, or OGB)"""
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        try:
            dataset = Planetoid(root=root, name=name)
            data = dataset[0]
            return data, dataset.num_features, dataset.num_classes
        except Exception as e:
            print(f"\n⚠ Could not download {name} dataset: {e}")
            print("→ Using synthetic data for demonstration")
            # Create synthetic data similar to the dataset
            num_nodes = 2708
            num_features = 1433
            num_classes = 7
            num_edges = 5429

            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            x = torch.randn(num_nodes, num_features)
            y = torch.randint(0, num_classes, (num_nodes,))
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[:140] = True
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask[140:640] = True
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask[1708:] = True

            class SyntheticData:
                def __init__(self):
                    self.edge_index = edge_index
                    self.x = x
                    self.y = y
                    self.train_mask = train_mask
                    self.val_mask = val_mask
                    self.test_mask = test_mask
                    self.num_nodes = num_nodes
                    self.num_edges = num_edges

            return SyntheticData(), num_features, num_classes
    else:
        raise ValueError(f"Dataset {name} not supported yet. Use: Cora, CiteSeer, PubMed")


def train_worker(rank: int, world_size: int, config: dict):
    """
    Training worker for each process.

    Args:
        rank: Process rank (GPU ID)
        world_size: Total number of processes
        config: Configuration dictionary
    """
    # Setup logger
    logger = setup_logger(rank=rank if world_size > 1 else None)

    if rank == 0:
        logger.info(f"Starting distributed training with {world_size} GPUs")

    # Setup distributed training (if multi-GPU)
    if world_size > 1:
        backend = 'gloo' if os.name == 'nt' else 'nccl'  # Use gloo on Windows
        setup_distributed(rank, world_size, backend=backend)

    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() and world_size > 1
                         else 'cuda:0' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        logger.info(f"Using device: {device}")

    # Load dataset
    if rank == 0:
        logger.info(f"Loading dataset: {config['dataset']}")

    data, num_features, num_classes = load_dataset(config['dataset'])

    if rank == 0:
        logger.info(f"Dataset: {data.num_nodes} nodes, {data.num_edges} edges, "
                   f"{num_features} features, {num_classes} classes")

    # Partition graph
    if rank == 0:
        logger.info(f"Partitioning graph into {world_size} partitions...")

    partitioner = GraphPartitioner(num_partitions=world_size)
    partition_data = partitioner.partition(data.edge_index, data.num_nodes)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        x=data.x,
        y=data.y,
        edge_index=data.edge_index,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        partition_data=partition_data,
        rank=rank,
        world_size=world_size,
        batch_size=config.get('batch_size', 32)
    )

    if rank == 0:
        logger.info(f"Created data loaders: {len(train_loader)} train batches")

    # Create model
    model = ScaleGNN(
        in_channels=num_features,
        hidden_channels=config.get('hidden_channels', 64),
        out_channels=num_classes,
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.5),
        use_lcs=config.get('use_lcs', True),
        lcs_threshold=config.get('lcs_threshold', 0.1)
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created: {total_params:,} parameters")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('lr', 0.01),
        weight_decay=config.get('weight_decay', 5e-4)
    )

    # Create trainer
    trainer = DistributedTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        rank=rank,
        world_size=world_size
    )

    # Training loop
    num_epochs = config.get('num_epochs', 100)
    best_val_acc = 0.0

    if rank == 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        if rank == 0:
            logger.info(f"Epoch {epoch}/{num_epochs}")

        # Train
        train_metrics = trainer.train_epoch(train_loader, data.x, data.edge_index)

        # Validate (every 10 epochs)
        if epoch % 10 == 0 or epoch == num_epochs:
            val_metrics = trainer.evaluate(val_loader, data.x, data.edge_index)

            if rank == 0:
                logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                          f"Acc: {train_metrics['accuracy']:.4f}, "
                          f"Time: {train_metrics['epoch_time']:.2f}s")
                logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                          f"Acc: {val_metrics['accuracy']:.4f}")

                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    logger.info(f"  ✓ New best validation accuracy: {best_val_acc:.4f}")
        else:
            if rank == 0:
                logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                          f"Acc: {train_metrics['accuracy']:.4f}, "
                          f"Time: {train_metrics['epoch_time']:.2f}s")

    # Final test evaluation
    if rank == 0:
        logger.info(f"\n{'='*60}")
        logger.info("Final Test Evaluation")
        logger.info(f"{'='*60}\n")

    test_metrics = trainer.evaluate(test_loader, data.x, data.edge_index)

    if rank == 0:
        logger.info(f"Test - Loss: {test_metrics['loss']:.4f}, "
                   f"Acc: {test_metrics['accuracy']:.4f}")
        logger.info(f"\nBest Val Acc: {best_val_acc:.4f}")
        logger.info(f"Final Test Acc: {test_metrics['accuracy']:.4f}")

    # Cleanup
    if world_size > 1:
        cleanup_distributed()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Distributed ScaleGNN Training')
    parser.add_argument('--config', type=str, default='config/cora.yaml',
                       help='Path to config file')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs to use (1 for single-GPU)')
    parser.add_argument('--dataset', type=str, default='Cora',
                       help='Dataset name (Cora, CiteSeer, PubMed)')
    args = parser.parse_args()

    # Load config if exists, otherwise use defaults
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Ensure numeric types are correct
        config['lr'] = float(config.get('lr', 0.01))
        config['weight_decay'] = float(config.get('weight_decay', 5e-4))
        config['dropout'] = float(config.get('dropout', 0.5))
        config['lcs_threshold'] = float(config.get('lcs_threshold', 0.1))
    else:
        config = {
            'dataset': args.dataset,
            'num_epochs': 100,
            'batch_size': 32,
            'hidden_channels': 64,
            'num_layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'weight_decay': 5e-4,
            'use_lcs': True,
            'lcs_threshold': 0.1
        }
        print(f"Config file not found, using defaults: {config}")

    # Determine number of GPUs
    if torch.cuda.is_available():
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
        print(f"Found {torch.cuda.device_count()} GPUs, using {num_gpus}")
    else:
        num_gpus = 1
        print("CUDA not available, using CPU")

    # Launch training
    if num_gpus > 1:
        # Multi-GPU training
        mp.spawn(train_worker, args=(num_gpus, config), nprocs=num_gpus, join=True)
    else:
        # Single-GPU or CPU training
        train_worker(0, 1, config)


if __name__ == '__main__':
    main()

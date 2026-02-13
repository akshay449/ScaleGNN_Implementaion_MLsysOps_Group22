"""
Distributed Trainer for ScaleGNN
Implements PyTorch DDP with AllReduce gradient synchronization
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Optional
import time


class DistributedTrainer:
    """
    Distributed trainer using PyTorch DDP for multi-GPU training.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: torch.device, rank: int, world_size: int):
        """
        Args:
            model: ScaleGNN model
            optimizer: Optimizer (e.g., Adam, SGD)
            device: Device for this process
            rank: Process rank (GPU ID)
            world_size: Total number of processes
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.rank = rank
        self.world_size = world_size

        # Wrap model with DDP
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank)

        self.criterion = nn.NLLLoss()

    def train_epoch(self, train_loader, x_full: torch.Tensor,
                   edge_index_full: torch.Tensor) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            x_full: Full node features (for accessing neighbors)
            edge_index_full: Full edge index

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()

            # Move batch to device
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(x, edge_index)

            # Compute loss on batch nodes only (first len(batch_nodes) nodes in subgraph)
            batch_size = len(batch['batch_nodes'])
            loss = self.criterion(out[:batch_size], y[:batch_size])

            # Backward pass (DDP handles gradient synchronization automatically)
            loss.backward()
            self.optimizer.step()

            # Metrics
            pred = out[:batch_size].argmax(dim=1)
            correct = (pred == y[:batch_size]).sum().item()

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

            batch_time = time.time() - batch_start

            if self.rank == 0 and batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}, Acc={correct/batch_size:.4f}, "
                      f"Time={batch_time*1000:.1f}ms")

        epoch_time = time.time() - epoch_start

        # Aggregate metrics across all workers
        if self.world_size > 1:
            total_loss_tensor = torch.tensor(total_loss).to(self.device)
            total_correct_tensor = torch.tensor(total_correct).to(self.device)
            total_samples_tensor = torch.tensor(total_samples).to(self.device)

            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

            total_loss = total_loss_tensor.item()
            total_correct = total_correct_tensor.item()
            total_samples = total_samples_tensor.item()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time,
            'samples': total_samples
        }

    @torch.no_grad()
    def evaluate(self, loader, x_full: torch.Tensor,
                edge_index_full: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.

        Args:
            loader: Data loader
            x_full: Full node features
            edge_index_full: Full edge index

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in loader:
            # Move batch to device
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)

            # Forward pass
            out = self.model(x, edge_index)

            # Compute metrics on batch nodes only
            batch_size = len(batch['batch_nodes'])
            loss = self.criterion(out[:batch_size], y[:batch_size])

            pred = out[:batch_size].argmax(dim=1)
            correct = (pred == y[:batch_size]).sum().item()

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

        # Aggregate metrics across all workers
        if self.world_size > 1:
            total_loss_tensor = torch.tensor(total_loss).to(self.device)
            total_correct_tensor = torch.tensor(total_correct).to(self.device)
            total_samples_tensor = torch.tensor(total_samples).to(self.device)

            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

            total_loss = total_loss_tensor.item()
            total_correct = total_correct_tensor.item()
            total_samples = total_samples_tensor.item()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }


def setup_distributed(rank: int, world_size: int, backend: str = 'gloo'):
    """
    Initialize distributed training environment.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: DDP backend ('nccl' for GPU, 'gloo' for CPU/Windows)
    """
    # On Windows, use gloo backend
    # On Linux with CUDA, use nccl for better performance
    import os

    if backend == 'nccl' and not torch.cuda.is_available():
        backend = 'gloo'
        print(f"Warning: NCCL not available, using {backend} backend")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if rank == 0:
        print(f"✓ Distributed training initialized: {world_size} workers, backend={backend}")


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

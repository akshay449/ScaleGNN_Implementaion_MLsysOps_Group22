"""
Utility functions for metrics computation
"""

import torch
from sklearn.metrics import f1_score
import numpy as np


def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        pred: Predictions [num_samples, num_classes] or [num_samples]
        target: Ground truth labels [num_samples]

    Returns:
        Accuracy as float
    """
    if pred.dim() > 1:
        pred = pred.argmax(dim=1)

    correct = (pred == target).sum().item()
    accuracy = correct / target.size(0)

    return accuracy


def compute_f1(pred: torch.Tensor, target: torch.Tensor, average: str = 'weighted') -> float:
    """
    Compute F1 score.

    Args:
        pred: Predictions [num_samples, num_classes] or [num_samples]
        target: Ground truth labels [num_samples]
        average: Averaging method ('micro', 'macro', 'weighted')

    Returns:
        F1 score as float
    """
    if pred.dim() > 1:
        pred = pred.argmax(dim=1)

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    return f1_score(target_np, pred_np, average=average, zero_division=0)

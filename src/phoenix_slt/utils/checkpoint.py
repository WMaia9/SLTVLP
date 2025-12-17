#!/usr/bin/env python3
"""
Utility for model checkpointing and resuming.
"""
import os
import torch
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path):
    """Save training checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    metrics = checkpoint.get("metrics", {})
    print(f"✓ Checkpoint loaded from epoch {epoch}: {checkpoint_path}")
    return epoch, metrics

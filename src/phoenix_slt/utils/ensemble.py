#!/usr/bin/env python
"""Save checkpoints at regular intervals for ensemble decoding.

Modify Phase 2 training to save every N epochs instead of only best BLEU.
This creates multiple checkpoints for ensemble averaging.

Usage: Just import and call save_checkpoint_every_n_epochs() during training.
"""

import os
import torch
from pathlib import Path


def save_checkpoint_every_n_epochs(
    model,
    epoch,
    save_every=5,
    checkpoint_dir="checkpoints",
    prefix="slt_epoch",
):
    """Save checkpoint every N epochs for ensemble."""
    if (epoch + 1) % save_every == 0:
        ckpt_path = Path(checkpoint_dir) / f"{prefix}_{epoch+1}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Unwrap DDP if needed
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), ckpt_path)
        return str(ckpt_path)
    return None

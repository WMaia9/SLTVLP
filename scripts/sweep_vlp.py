#!/usr/bin/env python
"""Hyperparameter sweep runner for Phase 1 VLP.

Usage:
  # Initialize sweep
  wandb sweep sweep_config.yaml
  
  # This outputs a sweep ID, then run agents:
  wandb agent <SWEEP_ID>
  
  # Or run multiple agents in parallel:
  for i in {1..4}; do wandb agent <SWEEP_ID> & done
"""

import os
import sys
import wandb
from phoenix_slt.train.phase1_vlp import main
from phoenix_slt.config import (
    BATCH_SIZE_PHASE1,
    DROPOUT,
    VLP_LR,
    ACCUMULATE_STEPS,
)

def run_sweep():
    """Run a single training job with W&B sweep parameters."""
    with wandb.init() as run:
        # Get hyperparams from sweep
        config = run.config
        
        # Override config with sweep values
        batch_size = config.get("batch_size", BATCH_SIZE_PHASE1)
        lr = config.get("learning_rate", VLP_LR)
        dropout = config.get("dropout", DROPOUT)
        accumulate = config.get("accumulate_steps", ACCUMULATE_STEPS)
        
        # Patch config (simple approach; for production, use environment variables or config files)
        import phoenix_slt.config as cfg
        cfg.BATCH_SIZE_PHASE1 = batch_size
        cfg.VLP_LR = lr
        cfg.DROPOUT = dropout
        cfg.ACCUMULATE_STEPS = accumulate
        
        print(f"Running with batch={batch_size}, lr={lr}, dropout={dropout}, accum={accumulate}")
        
        # Run training
        main()

if __name__ == "__main__":
    run_sweep()

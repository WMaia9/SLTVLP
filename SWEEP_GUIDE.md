# Professional Hyperparameter Sweep for Phase 1 VLP

## Overview
Systematic grid search over batch size, learning rate, dropout, and gradient accumulation to find the optimal configuration.

## Quick Start

### 1. Initialize the sweep
```bash
cd /home/wesleyferreiramaia/data/PHOENIX14T
wandb sweep sweep_config.yaml
```

This outputs a **SWEEP_ID** like `usp/phoenix-slt/abc123xyz`

### 2. Run sweep agents in parallel (4 parallel workers)
```bash
export SWEEP_ID="usp/phoenix-slt/abc123xyz"
for i in {1..4}; do
  wandb agent $SWEEP_ID &
done
```

Or run a single agent:
```bash
wandb agent usp/phoenix-slt/abc123xyz
```

### 3. Monitor in W&B dashboard
- Go to your project: https://wandb.ai/usp/phoenix-slt
- Click "Sweeps" tab
- Watch real-time progress; best hyperparams highlighted

## Config Details

**Parameters being tested:**
- **batch_size**: [200, 400, 600, 800] — contrastive negative pool size
- **learning_rate**: [5e-5, 1e-4, 2e-4] — optimizer step size
- **dropout**: [0.1, 0.2, 0.3] — regularization
- **accumulate_steps**: [1, 2, 4] — gradient accumulation for effective batch scaling

**Total combinations**: 4 × 3 × 3 × 3 = 108 runs

**Optimization metric**: `best_val_loss` (minimize)

## Expected Timeline
- ~15-20 min per run (depending on early stopping)
- 4 parallel workers: ~6-8 hours to complete all 108 runs
- 1 worker: ~30-35 hours

## Post-Sweep Analysis

After sweep completes, view results:
```bash
# Best config
wandb artifact get usp/phoenix-slt/sweep-best-config

# Download all metrics as CSV
# Use W&B UI: Sweeps tab → Download table as CSV
```

Then retrain final model with best hyperparams:
```bash
# Update config.py with best values
# python scripts/train_vlp.py
```

## Pro Tips

1. **Early stopping**: Runs stop automatically after patience=15 epochs with no improvement
2. **Parallel efficiency**: Run 4 agents on different GPUs/nodes for fast completion
3. **Cost control**: Limit sweep to fewer params (e.g., just batch_size + lr)
4. **Reproducibility**: All runs logged to W&B with full config snapshots

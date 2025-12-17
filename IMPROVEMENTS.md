# Phase 1 VLP Performance Improvement Plan

## Current Performance (Epoch 243)
- **R@1**: 19.65% | **R@5**: 47.78% | **R@10**: 65.13% | **MedR**: 6.0
- Train Loss: 3.10 | Val Loss: 3.56
- **Configuration**: Keypoints only (USE_SIGLIP=False)

## Changes Applied ✅

### 1. Enable Multimodal Fusion (BIGGEST IMPACT)
- **Before**: `USE_SIGLIP = False` (keypoints only)
- **After**: `USE_SIGLIP = True` (keypoints + SigLIP visual features)
- **Expected Impact**: +5-10% R@1 improvement
- **Rationale**: Multimodal fusion provides complementary information - keypoints capture hand/body motion, SigLIP captures appearance and context

### 2. Reduce Dropout
- **Before**: `DROPOUT = 0.4`
- **After**: `DROPOUT = 0.2`
- **Expected Impact**: +2-3% R@1 improvement
- **Rationale**: Lower dropout helps learn stronger representations for contrastive loss; 0.4 was too aggressive

### 3. Increase Gradient Accumulation
- **Before**: `ACCUMULATE_STEPS = 2` (effective batch = 3000)
- **After**: `ACCUMULATE_STEPS = 4` (effective batch = 6000)
- **Expected Impact**: +1-2% R@1 improvement
- **Rationale**: Larger effective batch provides more negative samples, improving contrastive learning signal

## Expected Results
**Target**: R@1 ≥ 30-35% | R@5 ≥ 60% | R@10 ≥ 75% | MedR ≤ 3

## Next Steps

### Immediate Actions
1. **Retrain Phase 1 with new settings**:
   ```bash
   # Clear old checkpoints or back them up
   mv checkpoints/vlp_best_encoder.pt checkpoints/vlp_best_encoder_kptsonly.pt.bak
   mv checkpoints/vlp_best_full.pt checkpoints/vlp_best_full_kptsonly.pt.bak
   
   # Start new training run
   unset WORLD_SIZE
   export CUDA_VISIBLE_DEVICES=0,1
   export PYTORCH_ALLOC_CONF=expandable_segments:True
   python scripts/train_vlp.py
   ```

2. **Monitor W&B**: Watch for:
   - Faster convergence with multimodal features
   - Lower validation loss (target: <3.0)
   - Check that both kpts and siglip are being used (logged in config)

3. **Evaluate periodically**:
   ```bash
   python scripts/eval_vlp.py --split dev --ckpt checkpoints/vlp_best_full.pt --examples 10 --top-k 5
   ```

### If Results Are Still Not Satisfactory

#### Option A: Adjust MAX_FRAMES
- Current: 180 frames (long sequences)
- Try: 120 or 150 frames
- **Rationale**: Shorter sequences = more samples per batch, less noise

```python
MAX_FRAMES = 120  # in config.py
```

#### Option B: Add Warmup Scheduler
Add this to `phase1_vlp.py`:
```python
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    return LambdaLR(optimizer, lr_lambda)

# In train_vlp() after optimizer creation:
warmup_steps = 500  # first 500 steps
scheduler = get_warmup_scheduler(optimizer, warmup_steps)
# Call scheduler.step() after each optimizer.step()
```

#### Option C: Tune Temperature Parameter
- Modify initial `logit_scale` in VLP_PretrainingModel
- Try initializing to `log(1/0.07)` instead of `log(100)` for sharper gradients

#### Option D: Hard Negative Mining
- In current setup, all other samples in batch are negatives
- Could implement hard negative mining to focus on difficult pairs

#### Option E: Data Augmentation
- Add temporal jittering (random frame offsets)
- Add noise to keypoints
- Random masking of keypoints

## Monitoring Checklist

During training, ensure:
- [ ] Both `USE_KPTS=True` and `USE_SIGLIP=True` logged in W&B config
- [ ] Train loss decreasing steadily (should start ~4.5-5.0 with multimodal)
- [ ] Val loss < 3.0 by epoch 100
- [ ] No OOM errors (if OOM occurs, reduce BATCH_SIZE_PHASE1 to 1200)
- [ ] Both GPUs utilized (~90%+ GPU util on both)

## Performance Benchmarks

### Good Performance (Target)
- R@1: 30-40%
- R@5: 60-70%
- R@10: 75-85%
- MedR: 2-3

### Excellent Performance (Ambitious)
- R@1: >40%
- R@5: >70%
- R@10: >85%
- MedR: 1-2

## Troubleshooting

### If Training is Slower
- Multimodal model has more parameters → slightly slower per epoch
- Expected: ~1.2-1.5x slower than keypoints-only
- Still worth it for better performance!

### If OOM Occurs
```python
# Reduce batch size
BATCH_SIZE_PHASE1 = 1200  # or 1000

# Or keep batch size but reduce accumulation
ACCUMULATE_STEPS = 2
```

### If Val Loss Plateaus Early
- Increase learning rate: `VLP_LR = 2e-4`
- Or add warmup (see Option B above)

## Notes
- Original run (keypoints-only) achieved 19.65% R@1 - solid baseline
- Multimodal fusion is standard practice in sign language understanding
- These changes align with best practices in contrastive learning research

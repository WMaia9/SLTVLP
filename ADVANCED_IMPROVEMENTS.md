# Advanced SLT Improvements - Implementation Guide

## ✅ Already Implemented

### 1. **CTC Auxiliary Loss** (Expected: +2-3 BLEU)
- Added CTC head to model for better alignment
- Curriculum learning: CTC weight ramps up over first 20 epochs
- Config: `USE_CTC_LOSS = True`, `CTC_WEIGHT = 0.3`

### 2. **Model Capacity Increase** (Expected: +1-2 BLEU)
- Encoder layers: 4 → 6
- Attention heads: 4 → 8  
- Model dim: 384 → 512

### 3. **Unfrozen mBART Encoder** (Expected: +2-3 BLEU)
- Top 4 layers (8-11) of mBART encoder are trainable
- Better learning rates tuned for unfrozen encoder

### 4. **Better Beam Search** (Expected: +1-2 BLEU)
- Beams: 5 → 8
- Length penalty: 0.9 → 1.0 (less biased to short)
- Max tokens: 50 → 60
- Added `early_stopping=False`

### 5. **Improved Hyperparameters**
- Higher LRs: encoder 5e-5, decoder 1e-5
- Lower weight decay: 0.01
- More label smoothing: 0.15
- Longer warmup: 10 epochs

## 📋 Usage Instructions

### Training from Scratch
```bash
# Phase 1: Pretrain encoder (larger architecture)
torchrun --nproc_per_node=2 scripts/train_vlp.py

# Phase 2: Fine-tune with CTC + curriculum
torchrun --nproc_per_node=2 scripts/train_slt.py
```

### Ensemble Decoding (Post-training)
After training completes, you'll have multiple checkpoints. Use ensemble:

```bash
# Save checkpoints every 10 epochs manually or modify training loop
# Then ensemble the best 3-5:
python scripts/ensemble_decode.py \
    --ckpts checkpoints/best_slt_model.pt \
            checkpoints/slt_epoch_25.pt \
            checkpoints/slt_epoch_30.pt \
    --split dev
```

**To automatically save ensemble checkpoints during training:**
Add this to Phase 2 training loop (after line 377 in phase2_slt.py):
```python
from phoenix_slt.utils.ensemble import save_checkpoint_every_n_epochs

# Inside training loop, after logging:
if rank == 0 and (epoch + 1) % 10 == 0:
    ckpt_path = save_checkpoint_every_n_epochs(
        model, epoch, save_every=10, 
        checkpoint_dir=str(CHECKPOINTS_DIR)
    )
    if ckpt_path:
        print(f"  -> Saved ensemble checkpoint: {ckpt_path}")
```

## 🎯 Expected BLEU Gains

Starting from BLEU 12:
- CTC loss: +2-3 → **~14-15**
- Larger model: +1-2 → **~15-17**
- Unfrozen encoder: +2-3 → **~17-20**
- Better decoding: +1-2 → **~18-22**
- Ensemble (3-5 ckpts): +1-2 → **~19-24**

**Total expected: BLEU 19-24** (target was 20)

## 🚀 Additional SOTA Techniques (Not Yet Implemented)

### Back-Translation Augmentation
Requires:
1. Train German→Sign model
2. Translate German monolingual data
3. Add synthetic sign-German pairs to training

### Advanced Post-Processing
- Language model reranking
- Minimum Bayes Risk (MBR) decoding
- Remove repeated n-grams post-generation

### Architecture Enhancements
- Add cross-attention between keypoint and text features
- Multi-task learning with gloss prediction
- Adapter modules for domain adaptation

## 📊 Monitoring Training

Key metrics to watch:
- CTC loss should decrease alongside CE loss
- BLEU should improve steadily (save checkpoints every 5-10 epochs)
- Dev loss plateau → try ensemble of last few checkpoints
- If overfitting: increase dropout, more label smoothing

## 🔧 Troubleshooting

**OOM with larger model?**
- Reduce `BATCH_SIZE_PHASE2` to 32 or 24
- Increase `ACCUMULATE_STEPS` to 4

**CTC causing NaN loss?**
- Check that `blank=0` is correct (pad_token_id)
- Reduce `CTC_WEIGHT` to 0.2
- Ensure input_lengths > target_lengths

**BLEU not improving?**
- Train longer (500 epochs with patience 300)
- Try ensemble of checkpoints from epochs 20-50
- Check learning rate isn't too high/low

## 📝 Files Modified

1. `src/phoenix_slt/config.py` - Added CTC and curriculum flags
2. `src/phoenix_slt/models/modeling.py` - Added CTC head
3. `src/phoenix_slt/train/phase2_slt.py` - CTC loss + curriculum
4. `scripts/ensemble_decode.py` - New ensemble script
5. `src/phoenix_slt/utils/ensemble.py` - Checkpoint helper

# Architecture & Design Overview

## Project Philosophy

This project follows **professional software engineering** standards:
- **Modularity**: Clear separation of concerns (data → models → training)
- **Reusability**: Import-based, no code duplication across phases
- **Configurability**: Single source of truth for hyperparameters
- **Observability**: W&B logging at every critical step
- **Reproducibility**: Fixed seeds, versioned configs, checkpoint saving

## Module Breakdown

### `src/phoenix_slt/config.py`
**Responsibility**: Centralized configuration management

```python
# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
KPTS_DIR = DATA_DIR / "kpts"
SIGLIP_DIR = DATA_DIR / "siglip_vitb16"
META_DIR = DATA_DIR / "annotations"

# Hyperparameters (Phase 1 & 2)
VLP_LR = 1e-4
ENCODER_LR = 1e-4
DECODER_LR = 5e-6
# ... etc
```

**Usage**: 
```python
from phoenix_slt.config import VLP_LR, BATCH_SIZE_PHASE1, KPTS_DIR
```

### `src/phoenix_slt/data/datasets.py`
**Responsibility**: Data loading and preprocessing

**Key Functions**:
- `load_tokenizer()` → mBART tokenizer
- `load_splits()` → train/dev/test DataFrames from CSV
- `PhoenixDataset` → Custom `torch.Dataset` (handles .npy loading, augmentation)
- `phoenix_collate_fn()` → Batching with padding, tokenization
- `build_loaders()` → Creates DataLoaders for all splits

**Features**:
- Lazy loading (numpy files loaded on-the-fly)
- Temporal augmentation (time-warp, scale, noise)
- Masking for variable-length sequences
- German text normalization

### `src/phoenix_slt/models/modeling.py`
**Responsibility**: All model architectures

**Encoder Stack** (Squeezeformer):
```
Input (Kpts + SigLIP)
  ↓
Project to D_MODEL
  ↓
Temporal Transformer (on visual)
  ↓
Gated Fusion (kpts + visual)
  ↓
4× Squeezeformer Blocks
  (FF + MHA + Conv + Stochastic Depth)
  ↓
Output: (B, T, D_MODEL)
```

**Phase 1**: `VLP_PretrainingModel`
- Visual encoder → projection head (D_MODEL → 256)
- Text encoder (frozen mBART) → projection head
- Contrastive loss (NT-Xent with learned temperature)

**Phase 2**: `SignTranslationModel`
- Encoder + Adapter (D_MODEL → MBART_DIM)
- mBART decoder (frozen encoder, trainable decoder)
- Cross-entropy loss with label smoothing

### `src/phoenix_slt/train/phase1_vlp.py`
**Responsibility**: Visual-Language Pre-training workflow

```
main()
  ├─ Load tokenizer, splits, loaders
  ├─ Initialize encoder + VLP model
  ├─ wandb.init() → log config
  ├─ For each epoch:
  │   ├─ Train loop
  │   │   ├─ Forward: vis_vec, txt_vec = model(batch)
  │   │   ├─ Loss: contrastive(vis_vec @ txt_vec.T)
  │   │   ├─ Backward, step
  │   │   └─ wandb.log(train_loss)
  │   │
  │   ├─ Eval loop
  │   │   ├─ Loss on dev set
  │   │   └─ wandb.log(val_loss)
  │   │
  │   ├─ If best: save encoder, reset patience
  │   └─ Else: patience++
  │       If patience >= 15: early stop
  │
  └─ wandb.finish()
```

**I/O**:
- Input: CSV metadata + kpts/siglip features
- Output: `vlp_best_encoder.pt` (encoder state dict only)

### `src/phoenix_slt/train/phase2_slt.py`
**Responsibility**: Sign Language Translation fine-tuning

```
main()
  ├─ Load encoder from Phase 1
  ├─ Initialize SignTranslationModel
  ├─ Build differential param groups (enc_lr ≠ dec_lr)
  ├─ wandb.init() → log config
  ├─ For each epoch:
  │   ├─ Train loop
  │   │   ├─ Forward: loss, logits = model(batch)
  │   │   ├─ AMP (mixed precision)
  │   │   ├─ Gradient accumulation (2 steps)
  │   │   ├─ Grad norm clipping
  │   │   └─ wandb.log(train_loss)
  │   │
  │   ├─ Eval loop (loss)
  │   ├─ BLEU computation (beam search)
  │   │
  │   ├─ If BLEU improves: save model, reset patience
  │   └─ Else: patience++
  │       If patience >= 8: early stop
  │
  └─ wandb.finish()
```

**I/O**:
- Input: `vlp_best_encoder.pt` + kpts/siglip/text
- Output: `best_slt_model.pt` (full model)

### `scripts/train_vlp.py` & `scripts/train_slt.py`
**Responsibility**: Thin CLI wrappers

```python
from phoenix_slt.train.phase1_vlp import main
if __name__ == "__main__":
    main()
```

Allows users to run training without touching internal modules.

### `src/phoenix_slt/utils/`
**Responsibility**: Shared utilities (expandable)

Currently:
- `checkpoint.py` → Save/load training checkpoints (with epoch, metrics)

Future additions:
- Metrics (BLEU, WER, CER)
- Logging utilities
- Evaluation scripts

## Data Flow

```
CSV (train/dev/test)
  ↓
PhoenixDataset.__getitem__()
  ├─ Load .npy (kpts, siglip)
  ├─ Augment (if training)
  └─ Return dict
  ↓
DataLoader (batch)
  ↓
phoenix_collate_fn()
  ├─ Pad sequences
  ├─ Tokenize text
  └─ Return batched dict
  ↓
Model(batch)
  ├─ Phase 1: contrastive loss
  └─ Phase 2: cross-entropy loss
  ↓
Optimizer.step() → wandb.log()
```

## Configuration Hierarchy

```
src/phoenix_slt/config.py (single source of truth)
          ↓
    Imported by:
  ├─ data/datasets.py
  ├─ train/phase1_vlp.py
  ├─ train/phase2_slt.py
  └─ models/modeling.py
```

**Never hardcode values** — always reference `config.py`.

## Training Patterns

### Phase 1: Contrastive Learning
```python
vis_vec = encode_visual(batch)      # (B, 256)
txt_vec = encode_text(batch)        # (B, 256)
logits = scale * (vis_vec @ txt_vec.T)  # (B, B)

loss = (CE(logits, targets) + CE(logits.T, targets)) / 2
```

### Phase 2: Seq2Seq with Teacher Forcing
```python
enc_out, _ = encoder(kpts, kpts_mask, siglip)  # (B, T, D)
enc_out = adapter(enc_out)                      # (B, T, MBART_DIM)

logits = mbart(
    inputs_embeds=enc_out,
    attention_mask=kpts_mask,
    labels=target_ids  # Teacher forcing
)
```

## Scalability Considerations

- **Mixed Precision (AMP)**: Reduces memory by ~30%
- **Gradient Accumulation**: Simulates larger batches
- **Differential LR**: Encoder (1e-4) vs Decoder (5e-6)
- **Early Stopping**: Prevents overfitting + saves compute
- **Checkpointing**: Resume from any epoch

## Testing & Validation

TODO:
- [ ] Unit tests for `PhoenixDataset` (mock .npy files)
- [ ] Integration tests for training loops
- [ ] Evaluation script on test set
- [ ] Inference pipeline (video → predictions)

## Performance Targets

| Phase | Task | Metric | Target |
|-------|------|--------|--------|
| 1 | Contrastive alignment | Loss (lower better) | < 0.5 |
| 2 | Translation | BLEU-4 | > 20 |
| 2 | Translation | Loss | < 2.0 |

---

**Next**: See README.md for usage, SETUP.md for installation.

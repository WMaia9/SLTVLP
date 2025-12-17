# Sign Language Translation (SLT) - Phoenix 2014-T

A novel two-phase training pipeline for sign language translation combining **Visual-Language Pre-training (VLP)** and **Sign Language Translation (SLT)** fine-tuning.

## 🎯 Model Overview

This project implements a **new neural architecture** for sign language translation that leverages:

1. **Phase 1: Visual-Language Pre-training (VLP)**
   - Contrastive learning to align visual encodings (keypoints + SigLIP embeddings) with text representations
   - Trains a Squeezeformer encoder to understand multimodal sign language semantics
   - Builds a strong foundation for downstream translation tasks

2. **Phase 2: Sign Language Translation (SLT)**
   - Fine-tunes the VLP encoder with mBART decoder for end-to-end translation
   - Uses differential learning rates for encoder vs. decoder
   - Optimized for translation quality (BLEU metric)

**Key Innovation**: Pre-training on contrastive vision-language alignment significantly improves translation performance compared to training from scratch.

## Directory Structure

Training pipeline with modular architecture, W&B logging, and best practices.

```
PHOENIX14T/
├── data/                          # All datasets (keep version-controlled separately if large)
│   ├── annotations/               # CSV metadata (train/dev/test splits)
│   ├── kpts/                      # Keypoint features (75 joints × 3 coords per frame)
│   ├── siglip_vitb16/             # Visual embeddings from SigLIP (768-dim per frame)
│   ├── videos/                    # Original video files (optional)
│   └── timesformer_sliding_window/ # Timesformer features (optional)
├── checkpoints/                   # Model weights & checkpoints
│   ├── vlp_best_encoder.pt        # Phase 1 VLP encoder
│   └── best_slt_model.pt          # Phase 2 SLT model
├── src/
│   └── phoenix_slt/               # Main package
│       ├── config.py              # Shared hyperparameters & paths
│       ├── data/
│       │   └── datasets.py        # Data loading, collate, tokenizer
│       ├── models/
│       │   └── modeling.py        # Encoder, VLP, SLT models
│       ├── train/
│       │   ├── phase1_vlp.py      # Visual-Language Pre-training
│       │   └── phase2_slt.py      # Sign Language Translation fine-tuning
│       └── utils/                 # (Future: metrics, logging, etc.)
├── scripts/
│   ├── train_vlp.py               # CLI wrapper for Phase 1
│   ├── train_slt.py               # CLI wrapper for Phase 2
│   ├── gpu_check.py               # Verify CUDA/GPU
│   └── extract_*.py               # Feature extraction (legacy)
├── notebooks/                      # Jupyter notebooks for exploration
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── SETUP.md                       # Setup and installation guide
├── ARCHITECTURE.md                # Architecture and design details
├── CHECKLIST.md                   # Feature completion checklist
└── .gitignore                     # Git ignore patterns
```

## Why Two Phases?

The VLP→SLT approach provides several advantages:

- **Stronger Representations**: Phase 1 learns to align visual and textual modalities before translation, creating better encodings
- **Transfer Learning**: Pre-trained encoder weights transfer knowledge to the translation task
- **Improved Convergence**: Starting from a well-aligned encoder makes Phase 2 training more stable
- **Modularity**: Each phase can be evaluated independently for quality

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Phase 1: Visual-Language Pre-training (VLP)

Trains the Squeezeformer encoder using **contrastive learning** to align sign language visual features (MediaPipe keypoints + SigLIP embeddings) with text encoder representations from mBART.

```bash
python scripts/train_vlp.py
```

**What happens**:
- Encoder learns to project visual features into the same space as text embeddings
- Contrastive loss: minimizes distance between matching pairs, maximizes distance between non-matching pairs
- Best model is saved to `checkpoints/vlp_best_encoder.pt`

**Configuration** (in `src/phoenix_slt/config.py`):
- Epochs: 100
- Batch size: 4
- Learning rate: 1e-4
- Early stopping patience: 15 epochs

### 3. Phase 2: Sign Language Translation (SLT)

Fine-tunes the **pre-trained encoder** from Phase 1 along with an mBART decoder for end-to-end sign→German translation.

```bash
python scripts/train_slt.py
```

**What happens**:
- Encoder: Fine-tuned from Phase 1 checkpoint (initialized with learned visual-language alignment)
- Decoder: Trains from scratch on translation task (lower learning rate)
- Differential learning rates: Encoder (1e-4) learns slowly to preserve pre-training, Decoder (5e-6) learns carefully from limited SLT data
- Best model is saved to `checkpoints/best_slt_model.pt`

**Configuration** (in `src/phoenix_slt/config.py`):
- Epochs: 30
- Batch size: 8
- Encoder LR: 1e-4 | Decoder LR: 5e-6
- Early stopping patience: 8 epochs (based on BLEU improvement)

## W&B Logging

Both phases log to Weights & Biases automatically:

```bash
export WANDB_PROJECT="phoenix-slt"
export WANDB_RUN_NAME="my-experiment"
python scripts/train_vlp.py
python scripts/train_slt.py
```

Tracked metrics:
- **Phase 1**: `train_loss`, `val_loss`, `lr`
- **Phase 2**: `train_loss`, `val_loss`, `bleu`, `lr`

## Configuration

Edit `src/phoenix_slt/config.py` to adjust:
- Data paths (KPTS_DIR, SIGLIP_DIR, etc.)
- Model architecture (D_MODEL, N_HEADS, ENC_LAYERS, etc.)
- Training hyperparameters (batch sizes, learning rates, epochs)

## Model Architecture

### Encoder: SqueezeformerFusionEncoder
- **Input**: Keypoints (T, 225) + SigLIP embeddings (180, 768)
- **Process**:
  1. Project both modalities to shared dim (D_MODEL=384)
  2. Temporal Transformer on visual features
  3. Interpolate to keypoint length
  4. Gated fusion layer
  5. 4 Squeezeformer blocks (feedforward + MHA + conv)
- **Output**: (B, T, D_MODEL) sequence

### Phase 1: VLP_PretrainingModel
- Frozen mBART text encoder
- Contrastive loss: align visual embeddings with text embeddings
- Trains: encoder projection heads + logit temperature

### Phase 2: SignTranslationModel
- Encoder (trainable) → Adapter → mBART (frozen text enc, trainable decoder)
- Cross-entropy loss with label smoothing
- Differential learning rates: encoder(1e-4) vs decoder(5e-6)
- Beam search (num_beams=5) for generation

## Key Features

✅ **Professional structure**: Modular, versioned, documented
✅ **W&B integration**: Detailed logging per phase
✅ **Data efficiency**: Gradient accumulation, mixed precision (AMP)
✅ **Early stopping**: Patience-based with checkpoint saving
✅ **Flexible config**: Single source of truth for hyperparameters

## Troubleshooting

**GPU out of memory?**
- Reduce `BATCH_SIZE_PHASE1` / `BATCH_SIZE_PHASE2`
- Set `ACCUMULATE_STEPS` > 2 in Phase 2

**Data loading errors?**
- Verify paths in `src/phoenix_slt/config.py` point to your data
- Check `data/annotations/*.csv` exist with `name` and `translation` columns

**Missing W&B logs?**
- Install: `pip install wandb`
- Login: `wandb login` (requires API key)

## Citation

If you use this pipeline, please cite the original PHOENIX-2014-T dataset and relevant papers.

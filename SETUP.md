# Setup Guide: Phoenix 2014-T Sign Language Translation

## Prerequisites
- Python 3.8+
- CUDA 11.0+ (GPU recommended for training)
- ~50GB disk space for datasets + models

## Installation

### 1. Clone / Setup Project
```bash
cd /path/to/PHOENIX14T
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# Optional: editable install for package imports
pip install -e .
```

### 4. Verify CUDA (Optional but Recommended)
```bash
python scripts/gpu_check.py
```

## Directory Setup

Ensure your data is organized as:
```
data/
├── annotations/
│   ├── PHOENIX-2014-T.train.corpus.csv
│   ├── PHOENIX-2014-T.dev.corpus.csv
│   └── PHOENIX-2014-T.test.corpus.csv
├── kpts/
│   ├── train/
│   ├── dev/
│   └── test/
├── siglip_vitb16/
│   ├── train/
│   ├── dev/
│   └── test/
└── videos/ (optional)
```

## Quick Training

### Phase 1: Visual-Language Pre-training
Single GPU:
```bash
python scripts/train_vlp.py
```
Multi-GPU (DDP), e.g., 2 GPUs:
```bash
torchrun --nproc_per_node=2 scripts/train_vlp.py
```
Trains encoder with contrastive learning. Saves best to `checkpoints/vlp_best_encoder.pt`.

### Phase 2: Sign Language Translation
Single GPU:
```bash
python scripts/train_slt.py
```
Multi-GPU (DDP):
```bash
torchrun --nproc_per_node=2 scripts/train_slt.py
```
Fine-tunes full model for translation. Saves best to `checkpoints/best_slt_model.pt`.

## Configuration

Edit `src/phoenix_slt/config.py` to customize:
- **Batch sizes**: `BATCH_SIZE_PHASE1`, `BATCH_SIZE_PHASE2`
- **Learning rates**: `VLP_LR`, `ENCODER_LR`, `DECODER_LR`
- **Model dims**: `D_MODEL`, `N_HEADS`, `ENC_LAYERS`
- **Data paths**: `KPTS_DIR`, `SIGLIP_DIR`, `META_DIR`, etc.
- **Performance**: `NUM_WORKERS`, `MAX_FRAMES` (cap frames to stabilize memory)

## Weights & Biases Integration

Track experiments with W&B:
```bash
pip install wandb
wandb login  # Get API key from wandb.ai
export WANDB_PROJECT="phoenix-slt"
export WANDB_RUN_NAME="my-experiment"
python scripts/train_vlp.py
```

## Performance & Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'phoenix_slt'` | Ensure editable install `pip install -e .` or run from project root |
| `CUDA out of memory` | Reduce batch size in `config.py` or enable gradient accumulation |
| `FileNotFoundError: data/annotations/...` | Check data paths in `config.py` match your setup |
| `wandb not found` | Run `pip install wandb` |
| Multi-GPU hangs | Use `torchrun` and ensure NCCL is available; only rank 0 writes checkpoints |

## Performance Notes

- AMP (automatic mixed precision) reduces memory and speeds up training
- Use `MAX_FRAMES` to cap per-sample frames if memory spikes occur
- Set `NUM_WORKERS` according to CPU cores; keep `pin_memory=True` and `persistent_workers=True`

## Next Steps

1. ✅ Run Phase 1 to get a pre-trained encoder
2. ✅ Run Phase 2 to fine-tune on translation task
3. 📊 Monitor W&B dashboard for metrics
4. 🧪 Experiment with hyperparameters in `config.py`
5. 📈 Evaluate on test set using downstream script (to be added)

## References

- **Dataset**: [PHOENIX-2014-T](https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-os/signum/)
- **Models**: Squeezeformer, mBART-large-cc25
- **Features**: Keypoints (MediaPipe), SigLIP embeddings

---
For questions or issues, check the README.md or open an issue.

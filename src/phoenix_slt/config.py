"""Project-wide configuration constants (paths and hyperparameters)."""

import os
from pathlib import Path

# Project roots
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

# Data paths
KPTS_DIR = DATA_DIR / "kpts"
SIGLIP_DIR = DATA_DIR / "siglip_vitb16"
META_DIR = DATA_DIR / "annotations"
VIDEOS_DIR = DATA_DIR / "videos"
TIMESFORMER_DIR = DATA_DIR / "timesformer_sliding_window"

TRAIN_CSV = META_DIR / "PHOENIX-2014-T.train.corpus.csv"
DEV_CSV = META_DIR / "PHOENIX-2014-T.dev.corpus.csv"
TEST_CSV = META_DIR / "PHOENIX-2014-T.test.corpus.csv"

# Data dimensions
NUM_JOINTS = 75
NUM_COORDS = 3
KPTS_FEAT_DIM = NUM_JOINTS * NUM_COORDS
SIGLIP_DIM = 768
SIGLIP_LEN = 180
MAX_TOKENS = 64
MAX_FRAMES = 180

# Model hyperparameters
ENC_LAYERS = 4
N_HEADS = 4
FF_EXPANSION = 4
DROPOUT = 0.2  # Reduced from 0.4 to improve representation learning
D_MODEL = 384
MBART_DIM = 1024

# Modality switches
USE_KPTS = True
USE_SIGLIP = True  # ENABLED: Using both modalities for better performance
assert USE_KPTS or USE_SIGLIP, "At least one modality must be True."

# Batch sizes per phase (tuned for 2x A100 40GB; adjust as needed)
BATCH_SIZE_PHASE1 = 1500
BATCH_SIZE_PHASE2 = 50

# Phase 1 (VLP) settings
VLP_EPOCHS = 500
VLP_LR = 1e-4
VLP_PATIENCE = 15
VLP_CHECKPOINT_PATH = CHECKPOINTS_DIR / "vlp_best_encoder.pt"
VLP_FULL_CHECKPOINT_PATH = CHECKPOINTS_DIR / "vlp_best_full.pt"
ENCODER_CKPT = VLP_CHECKPOINT_PATH

# Phase 2 settings
ENCODER_LR = 1e-4
DECODER_LR = 5e-6
SLT_WEIGHT_DECAY = 0.05
SLT_EPOCHS = 30
LABEL_SMOOTHING = 0.1
ACCUMULATE_STEPS = 4  # Increased from 2 for larger effective batch (better contrastive learning)
WARMUP_EPOCHS = 5
PATIENCE = 8
BEST_SLT_CKPT = CHECKPOINTS_DIR / "best_slt_model.pt"

NUM_WORKERS = 2

# DataLoader performance/safety knobs
# Set a conservative default to avoid persistent worker edge cases on clusters.
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = NUM_WORKERS  # only applies when num_workers > 0
DATALOADER_TIMEOUT = 120  # seconds; helps surface stuck worker issues
DROP_LAST_TRAIN = True  # enforce equal batch counts across ranks

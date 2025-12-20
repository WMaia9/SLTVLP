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
ENC_LAYERS = 5  # Moderate increase from 4 (6 was too large)
N_HEADS = 8     # Increased from 4 for better attention
FF_EXPANSION = 4
DROPOUT = 0.3  # Increased from 0.2 to prevent overfitting
D_MODEL = 448   # Moderate increase from 384 (512 was too large)
MBART_DIM = 1024

# Modality switches
USE_KPTS = True
USE_SIGLIP = False  # Tested, no significant gain for this dataset
assert USE_KPTS or USE_SIGLIP, "At least one modality must be True."

# Batch sizes per phase (adjusted for larger model on 2x A100 40GB)
BATCH_SIZE_PHASE1 = 200  # Reduced from 250 due to larger model
BATCH_SIZE_PHASE2 = 15   # Can handle more without SigLIP

# Phase 1 (VLP) settings
VLP_EPOCHS = 100
VLP_LR = 2e-4
VLP_PATIENCE = 50
VLP_CHECKPOINT_PATH = CHECKPOINTS_DIR / "vlp_best_encoder.pt"
VLP_FULL_CHECKPOINT_PATH = CHECKPOINTS_DIR / "vlp_best_full.pt"
ENCODER_CKPT = VLP_CHECKPOINT_PATH

# Phase 2 settings
ENCODER_LR = 1e-4      # Restored conservative rate
DECODER_LR = 5e-6      # Restored conservative rate
SLT_WEIGHT_DECAY = 0.05  # Restored proper regularization
SLT_EPOCHS = 500
LABEL_SMOOTHING = 0.1    # Restored original value
ACCUMULATE_STEPS = 2  # Conservative for stability
WARMUP_EPOCHS = 10    # Increased from 5 for unfrozen encoder
PATIENCE = 300
BEST_SLT_CKPT = CHECKPOINTS_DIR / "best_slt_model.pt"

# Advanced training techniques
USE_CTC_LOSS = False      # DISABLED: Causing mode collapse, needs debugging
CTC_WEIGHT = 0.3          # Weight for CTC loss (0.3 * CTC + 0.7 * CE)
USE_CURRICULUM = False    # DISABLED: Not helpful without CTC
CURRICULUM_EPOCHS = 20    # Number of epochs to ramp up difficulty
USE_BACK_TRANSLATION = False  # Augment with back-translated data (requires prep)
ENSEMBLE_CHECKPOINTS = []     # List of checkpoint paths for ensemble decoding

# EMA for VLP encoder
EMA_DECAY = 0.999

NUM_WORKERS = 2

# DataLoader performance/safety knobs
# Set a conservative default to avoid persistent worker edge cases on clusters.
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = NUM_WORKERS  # only applies when num_workers > 0
DATALOADER_TIMEOUT = 120  # seconds; helps surface stuck worker issues
DROP_LAST_TRAIN = True  # enforce equal batch counts across ranks

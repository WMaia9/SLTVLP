"""Project-wide configuration constants (paths and hyperparameters)."""

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

# Model hyperparameters
ENC_LAYERS = 4
N_HEADS = 4
FF_EXPANSION = 4
DROPOUT = 0.4
D_MODEL = 384
MBART_DIM = 1024

# Batch sizes per phase
BATCH_SIZE_PHASE1 = 4
BATCH_SIZE_PHASE2 = 8

# Phase 1 (VLP) settings
VLP_EPOCHS = 100
VLP_LR = 1e-4
VLP_PATIENCE = 15
VLP_CHECKPOINT_PATH = CHECKPOINTS_DIR / "vlp_best_encoder.pt"
ENCODER_CKPT = VLP_CHECKPOINT_PATH

# Phase 2 settings
ENCODER_LR = 1e-4
DECODER_LR = 5e-6
SLT_WEIGHT_DECAY = 0.05
SLT_EPOCHS = 30
LABEL_SMOOTHING = 0.1
ACCUMULATE_STEPS = 2
WARMUP_EPOCHS = 5
PATIENCE = 8
BEST_SLT_CKPT = CHECKPOINTS_DIR / "best_slt_model.pt"

NUM_WORKERS = 0

"""Dataset and dataloader utilities for PHOENIX-2014-T SLT."""

import random
import unicodedata
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import MBartTokenizer

from phoenix_slt.config import (
    ACCUMULATE_STEPS,
    DEV_CSV,
    KPTS_DIR,
    KPTS_FEAT_DIM,
    DATALOADER_TIMEOUT,
    DROP_LAST_TRAIN,
    LABEL_SMOOTHING,
    MAX_TOKENS,
    MAX_FRAMES,
    NUM_COORDS,
    NUM_JOINTS,
    NUM_WORKERS,
    PERSISTENT_WORKERS,
    PREFETCH_FACTOR,
    SIGLIP_DIR,
    TRAIN_CSV,
    TEST_CSV,
)

# Keypoint augmentation hyperparameters (kept conservative to avoid semantic drift)
AUG_ROT_MAX_DEG = 7.5
AUG_TEMP_CROP_MIN_RATIO = 0.7
AUG_FRAME_DROP_PROB = 0.05
AUG_JOINT_DROP_PROB = 0.05
AUG_MIN_FRAMES = 8
AUG_NOISE_STD = 0.002
AUG_TEMP_MASK_PROB = 0.4
AUG_TEMP_MASK_MAX_RATIO = 0.2  # up to 20% of sequence masked
AUG_HAND_JITTER_STD = 0.003

# Keypoint normalization
KPTS_EPS = 1e-6


def load_tokenizer() -> MBartTokenizer:
    """Load mBART tokenizer configured for German."""
    tokenizer = MBartTokenizer.from_pretrained(
        "facebook/mbart-large-cc25", src_lang="de_DE", tgt_lang="de_DE"
    )
    return tokenizer


def load_splits(
    train_csv=TRAIN_CSV, dev_csv=DEV_CSV, test_csv=TEST_CSV
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/dev/test CSV splits with standardized columns."""
    video_col = "name"
    text_col = "translation"

    df_train = pd.read_csv(train_csv, sep="|")
    df_dev = pd.read_csv(dev_csv, sep="|")
    df_test = pd.read_csv(test_csv, sep="|")

    df_train = df_train[[video_col, text_col]].rename(columns={video_col: "name", text_col: "text"})
    df_dev = df_dev[[video_col, text_col]].rename(columns={video_col: "name", text_col: "text"})
    df_test = df_test[[video_col, text_col]].rename(columns={video_col: "name", text_col: "text"})

    return df_train, df_dev, df_test


def augment_kpts_tensor(kpts: torch.Tensor) -> torch.Tensor:
    """Apply temporal and geometric jitter for data augmentation."""
    T, D = kpts.shape
    kpts = kpts.view(T, NUM_JOINTS, NUM_COORDS)

    # Random temporal crop to vary start/end while keeping enough frames
    if T > AUG_MIN_FRAMES and random.random() < 0.5:
        max_T = min(T, MAX_FRAMES)
        crop_len = random.randint(int(max_T * AUG_TEMP_CROP_MIN_RATIO), max_T)
        start = random.randint(0, max(0, T - crop_len))
        kpts = kpts[start : start + crop_len]
        T = kpts.shape[0]

    if random.random() < 0.5 and T > 4:
        rate = random.uniform(0.8, 1.2)
        new_T = max(int(T * rate), 1)
        x = kpts.view(T, -1).permute(1, 0).unsqueeze(0)
        x = F.interpolate(x, size=new_T, mode="linear", align_corners=False)
        kpts = x.squeeze(0).permute(1, 0).view(new_T, NUM_JOINTS, NUM_COORDS)
        T = new_T

    # Frame dropout keeps temporal structure while adding robustness
    if T > AUG_MIN_FRAMES and random.random() < 0.5:
        keep_mask = torch.rand(T) > AUG_FRAME_DROP_PROB
        if keep_mask.sum() < AUG_MIN_FRAMES:
            keep_mask[:AUG_MIN_FRAMES] = True  # guarantee minimum length
        kpts = kpts[keep_mask]
        T = kpts.shape[0]

    # Temporal masking: zero-out a contiguous span to simulate occlusion/dropouts
    if T > AUG_MIN_FRAMES and random.random() < AUG_TEMP_MASK_PROB:
        span = max(1, int(T * random.uniform(0.05, AUG_TEMP_MASK_MAX_RATIO)))
        start = random.randint(0, max(0, T - span))
        kpts[start : start + span] = 0.0

    # Small in-plane rotation
    if random.random() < 0.5:
        theta = torch.empty(1).uniform_(
            -AUG_ROT_MAX_DEG, AUG_ROT_MAX_DEG
        ) * (np.pi / 180.0)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        x, y = kpts[..., 0].clone(), kpts[..., 1].clone()
        kpts[..., 0] = cos_t * x - sin_t * y
        kpts[..., 1] = sin_t * x + cos_t * y

    if random.random() < 0.5:
        scale = 1.0 + 0.05 * torch.randn(1).clamp(-0.1, 0.1)
        tx = 0.02 * torch.randn(1)
        ty = 0.02 * torch.randn(1)
        kpts[..., 0] = kpts[..., 0] * scale + tx
        kpts[..., 1] = kpts[..., 1] * scale + ty

    # Drop a subset of joints to force robustness to missing detections
    if random.random() < 0.5:
        joint_mask = torch.rand(NUM_JOINTS) > AUG_JOINT_DROP_PROB
        if not joint_mask.any():
            joint_mask[torch.randint(0, NUM_JOINTS, (1,))] = True
        kpts = kpts * joint_mask.view(1, NUM_JOINTS, 1)

    # Hand-focused jitter (assumes hand joints exist; applies to all joints for robustness)
    if random.random() < 0.5:
        kpts = kpts + AUG_HAND_JITTER_STD * torch.randn_like(kpts)

    if random.random() < 0.5:
        kpts = kpts + AUG_NOISE_STD * torch.randn_like(kpts)

    return kpts.view(-1, D)


class PhoenixDataset(Dataset):
    """PyTorch Dataset for PHOENIX-2014-T keypoints + SigLIP features."""
    def __init__(self, df, split: str, kpts_dir=KPTS_DIR, siglip_dir=SIGLIP_DIR, is_train: bool = False):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.kpts_dir = kpts_dir
        self.siglip_dir = siglip_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["name"]
        text = str(row["text"])

        kpts_path = self.kpts_dir / self.split / f"{name}.npy"
        siglip_path = self.siglip_dir / self.split / f"{name}.npy"

        kpts_np = np.load(kpts_path).astype(np.float32)
        kpts_np[..., 0] *= -1.0

        # Per-video normalization: center and scale by std to reduce subject variance
        kpts_center = kpts_np.mean(axis=1, keepdims=True)  # (T,1,3)
        kpts_np = kpts_np - kpts_center
        kpts_std = np.std(kpts_np, axis=(1, 2), keepdims=True)  # (T,1,1)
        kpts_std = np.clip(kpts_std, KPTS_EPS, None)
        kpts_np = kpts_np / kpts_std

        T = kpts_np.shape[0]
        kpts_np = kpts_np.reshape(T, -1)
        kpts = torch.from_numpy(kpts_np)

        siglip_np = np.load(siglip_path).astype(np.float32)
        siglip = torch.from_numpy(siglip_np)

        if self.is_train:
            kpts = augment_kpts_tensor(kpts)

        # Velocity features: frame-wise delta, pad first frame with zeros
        T = kpts.shape[0]
        kpts_view = kpts.view(T, NUM_JOINTS, NUM_COORDS)
        vel = torch.zeros_like(kpts_view)
        if T > 1:
            vel[1:] = kpts_view[1:] - kpts_view[:-1]
        kpts = torch.cat([kpts_view, vel], dim=-1).view(T, -1)

        return {
            "name": name,
            "text": text,
            "kpts": kpts,
            "siglip": siglip,
        }


def normalize_german(text: str) -> str:
    """Normalize German text (trim, NFC, lowercase)."""
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    return text.lower()


def phoenix_collate_fn(batch, tokenizer) -> Dict[str, Any]:
    """Pad sequences and tokenize batch for model consumption."""
    names = [x["name"] for x in batch]
    raw_texts = [x["text"] for x in batch]
    norm_texts = [normalize_german(t) for t in raw_texts]

    kpts_list = [x["kpts"] for x in batch]
    lengths = [k.shape[0] for k in kpts_list]
    max_len = min(max(lengths), MAX_FRAMES)

    B = len(batch)
    K = kpts_list[0].shape[1]

    kpts_padded = torch.zeros(B, max_len, K, dtype=kpts_list[0].dtype)
    kpts_mask = torch.zeros(B, max_len, dtype=torch.float32)

    for i, k in enumerate(kpts_list):
        T = k.shape[0]
        T_c = min(T, max_len)
        kpts_padded[i, :T_c, :] = k[:T_c]
        kpts_mask[i, :T_c] = 1.0

    siglip_batch = torch.stack([x["siglip"] for x in batch])

    # mBART tokenizer returns a dict with keys: input_ids, attention_mask
    text_enc = tokenizer(
        text_target=norm_texts,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )

    return {
        "names": names,
        "texts": raw_texts,
        "kpts": kpts_padded,
        "kpts_mask": kpts_mask,
        "siglip": siglip_batch,
        "labels": text_enc["input_ids"],
        "labels_mask": text_enc["attention_mask"],
    }


def build_loaders(
    df_train,
    df_dev,
    df_test,
    tokenizer,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/dev/test DataLoaders with shared collate_fn.

    If ``distributed`` is True, use DistributedSampler for deterministic sharding.
    """
    ds_train = PhoenixDataset(df_train, "train", is_train=True)
    ds_dev = PhoenixDataset(df_dev, "dev", is_train=False)
    ds_test = PhoenixDataset(df_test, "test", is_train=False)

    train_sampler = (
        DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    dev_sampler = (
        DistributedSampler(ds_dev, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )
    test_sampler = (
        DistributedSampler(ds_test, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )

    # timeout/prefetch_factor are only valid when num_workers > 0
    effective_timeout = DATALOADER_TIMEOUT if num_workers > 0 else 0
    common_loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS and num_workers > 0,
        timeout=effective_timeout,
        collate_fn=lambda b: phoenix_collate_fn(b, tokenizer),
    )

    if num_workers > 0 and PREFETCH_FACTOR is not None:
        common_loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=False if train_sampler is not None else True,
        sampler=train_sampler,
        drop_last=DROP_LAST_TRAIN,
        **common_loader_kwargs,
    )
    dev_loader = DataLoader(
        ds_dev,
        batch_size=batch_size,
        shuffle=False,
        sampler=dev_sampler,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        **common_loader_kwargs,
    )

    return train_loader, dev_loader, test_loader

import random
import unicodedata
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import MBartTokenizer

from phoenix_slt.config import (
    ACCUMULATE_STEPS,
    DEV_CSV,
    KPTS_DIR,
    KPTS_FEAT_DIM,
    LABEL_SMOOTHING,
    MAX_TOKENS,
    NUM_COORDS,
    NUM_JOINTS,
    NUM_WORKERS,
    SIGLIP_DIR,
    TRAIN_CSV,
    TEST_CSV,
)


def load_tokenizer() -> MBartTokenizer:
    tokenizer = MBartTokenizer.from_pretrained(
        "facebook/mbart-large-cc25", src_lang="de_DE", tgt_lang="de_DE"
    )
    return tokenizer


def load_splits(train_csv=TRAIN_CSV, dev_csv=DEV_CSV, test_csv=TEST_CSV):
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
    # Geometric augmentations: temporal resample, scale/translate, and jitter.
    T, D = kpts.shape
    kpts = kpts.view(T, NUM_JOINTS, NUM_COORDS)

    if random.random() < 0.5 and T > 4:
        rate = random.uniform(0.8, 1.2)
        new_T = max(int(T * rate), 1)
        x = kpts.view(T, -1).permute(1, 0).unsqueeze(0)
        x = F.interpolate(x, size=new_T, mode="linear", align_corners=False)
        kpts = x.squeeze(0).permute(1, 0).view(new_T, NUM_JOINTS, NUM_COORDS)
        T = new_T

    if random.random() < 0.5:
        scale = 1.0 + 0.05 * torch.randn(1).clamp(-0.1, 0.1)
        tx = 0.02 * torch.randn(1)
        ty = 0.02 * torch.randn(1)
        kpts[..., 0] = kpts[..., 0] * scale + tx
        kpts[..., 1] = kpts[..., 1] * scale + ty

    if random.random() < 0.5:
        kpts = kpts + 0.002 * torch.randn_like(kpts)

    return kpts.view(-1, D)


class PhoenixDataset(Dataset):
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
        T = kpts_np.shape[0]
        kpts_np = kpts_np.reshape(T, -1)
        kpts = torch.from_numpy(kpts_np)

        siglip_np = np.load(siglip_path).astype(np.float32)
        siglip = torch.from_numpy(siglip_np)

        if self.is_train:
            kpts = augment_kpts_tensor(kpts)

        return {
            "name": name,
            "text": text,
            "kpts": kpts,
            "siglip": siglip,
        }


def normalize_german(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    return text.lower()


def phoenix_collate_fn(batch, tokenizer):
    names = [x["name"] for x in batch]
    raw_texts = [x["text"] for x in batch]
    norm_texts = [normalize_german(t) for t in raw_texts]

    kpts_list = [x["kpts"] for x in batch]
    lengths = [k.shape[0] for k in kpts_list]
    max_len = max(lengths)

    B = len(batch)
    K = kpts_list[0].shape[1]

    kpts_padded = torch.zeros(B, max_len, K, dtype=kpts_list[0].dtype)
    kpts_mask = torch.zeros(B, max_len, dtype=torch.float32)

    for i, k in enumerate(kpts_list):
        T = k.shape[0]
        kpts_padded[i, :T, :] = k
        kpts_mask[i, :T] = 1.0

    siglip_batch = torch.stack([x["siglip"] for x in batch])

    with tokenizer.as_target_tokenizer():
        text_enc = tokenizer(
            norm_texts,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )

    return {
        "names": names,
        "kpts": kpts_padded,
        "kpts_mask": kpts_mask,
        "siglip": siglip_batch,
        "labels": text_enc["input_ids"],
        "labels_mask": text_enc["attention_mask"],
    }


def build_loaders(df_train, df_dev, df_test, tokenizer, batch_size: int, num_workers: int = NUM_WORKERS):
    train_loader = DataLoader(
        PhoenixDataset(df_train, "train", is_train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: phoenix_collate_fn(b, tokenizer),
    )

    dev_loader = DataLoader(
        PhoenixDataset(df_dev, "dev", is_train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: phoenix_collate_fn(b, tokenizer),
    )

    test_loader = DataLoader(
        PhoenixDataset(df_test, "test", is_train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: phoenix_collate_fn(b, tokenizer),
    )

    return train_loader, dev_loader, test_loader

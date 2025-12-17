#!/usr/bin/env python
"""Evaluate Phase 1 (VLP) encoder.

- Computes average contrastive loss on a split (dev/test)
- Computes retrieval metrics (R@1/5/10, MedR) between visual and text embeddings

Usage:
    python scripts/eval_vlp.py --split dev
    python scripts/eval_vlp.py --split test --ckpt checkpoints/vlp_best_encoder.pt

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from phoenix_slt.config import (
    VLP_CHECKPOINT_PATH,
    VLP_FULL_CHECKPOINT_PATH,
    BATCH_SIZE_PHASE1,
    NUM_WORKERS,
)
from phoenix_slt.data.datasets import load_splits, build_loaders, load_tokenizer
from phoenix_slt.models.modeling import SqueezeformerFusionEncoder, VLP_PretrainingModel
from phoenix_slt.train.phase1_vlp import evaluate_vlp


def compute_retrieval_metrics(
    vis_all: torch.Tensor, txt_all: torch.Tensor
) -> Tuple[float, float, float, float]:
    """Compute retrieval metrics given full-matrix similarities.

    Assumes rows are aligned pairs (i-th visual with i-th text).
    Returns (R@1, R@5, R@10, MedR)
    """
    device = vis_all.device
    with torch.no_grad():
        # Similarity matrix: (N, N)
        S = vis_all @ txt_all.t()
        # Higher is better; ranks via descending sort
        ranks = torch.argsort(S, dim=1, descending=True)
        idx = torch.arange(S.size(0), device=device)
        # Rank position of the correct text for each visual
        pos = (ranks == idx.unsqueeze(1)).nonzero(as_tuple=False)
        # pos contains [row, col]; the column is the rank position
        # For each row, get the column where match occurs
        order = pos[:, 0]
        rank_pos = torch.zeros(S.size(0), device=device, dtype=torch.long)
        rank_pos[order] = pos[:, 1]

        r1 = (rank_pos < 1).float().mean().item() * 100.0
        r5 = (rank_pos < 5).float().mean().item() * 100.0
        r10 = (rank_pos < 10).float().mean().item() * 100.0
        medr = rank_pos.float().median().item() + 1.0  # 1-based median rank
    return r1, r5, r10, medr


def main():
    p = argparse.ArgumentParser(description="Evaluate VLP encoder (Phase 1)")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--ckpt", type=str, default=str(VLP_CHECKPOINT_PATH))
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_PHASE1)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--examples", type=int, default=5, help="Number of qualitative examples to print")
    p.add_argument("--top-k", type=int, default=5, help="Top-K retrieval to display for each example")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eval device: {device}")

    tokenizer = load_tokenizer()
    df_train, df_dev, df_test = load_splits()
    # Ensure non-empty train df to satisfy DataLoader sampler even if unused
    if args.split == "dev":
        df_a, df_b, df_c = df_train, df_dev, df_test.iloc[:0]
    else:
        df_a, df_b, df_c = df_train, df_test, df_test.iloc[:0]

    train_loader, eval_loader, _ = build_loaders(
        df_a,
        df_b,
        df_c,
        tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=False,
        rank=0,
        world_size=1,
    )
    del train_loader  # unused

    # Build model and load encoder weights
    # Prefer full VLP checkpoint (encoder + heads + logit_scale); fallback to encoder-only
    encoder = SqueezeformerFusionEncoder()
    vlp = VLP_PretrainingModel(encoder).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    loaded_full = False
    if "visual_encoder.kpts_proj.weight" in state:
        # Looks like a full VLP state dict
        missing, unexpected = vlp.load_state_dict(state, strict=False)
        loaded_full = True
    else:
        # Encoder-only checkpoint
        missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_state] missing={missing} unexpected={unexpected}")
    if loaded_full:
        print(f"Loaded full VLP checkpoint: {ckpt_path}")
    else:
        print(f"Loaded encoder-only checkpoint: {ckpt_path} (heads are randomly init)")
    vlp = vlp.to(device)
    criterion = nn.CrossEntropyLoss()

    # 1) Contrastive loss on the chosen split
    val_loss = evaluate_vlp(vlp, eval_loader, criterion, device)
    print(f"[{args.split}] contrastive loss: {val_loss:.4f}")

    # 2) Retrieval metrics (collect embeddings)
    vlp.eval()
    all_vis, all_txt = [], []
    all_names, all_texts = [], []
    with torch.no_grad():
        for batch in eval_loader:
            for k in ["kpts", "kpts_mask", "siglip", "labels", "labels_mask"]:
                batch[k] = batch[k].to(device)
            vis_vec, txt_vec = vlp(batch)
            all_vis.append(vis_vec)
            all_txt.append(txt_vec)
            # Preserve order-aligned names/texts for qualitative display
            all_names.extend(batch["names"])  # list of strings
            all_texts.extend(batch["texts"])  # list of strings

    vis_all = torch.cat(all_vis, dim=0)
    txt_all = torch.cat(all_txt, dim=0)

    # Optional normalization for cosine-like similarity (kept disabled to match training)
    # vis_all = F.normalize(vis_all, dim=1)
    # txt_all = F.normalize(txt_all, dim=1)

    r1, r5, r10, medr = compute_retrieval_metrics(vis_all, txt_all)
    print(
        f"[{args.split}] Retrieval: R@1={r1:.2f}% | R@5={r5:.2f}% | R@10={r10:.2f}% | MedR={medr:.1f}"
    )

    # Qualitative examples: show top-K texts for a few visuals
    try:
        N = vis_all.size(0)
        K = max(1, min(args.top_k, N))
        n_examples = max(1, min(args.examples, N))
        S = vis_all @ txt_all.t()
        # Pick evenly spaced examples across dataset
        indices = torch.linspace(0, N - 1, steps=n_examples).round().long().tolist()
        print("\nExamples (visual → top-K text matches):")
        for i in indices:
            sim = S[i]
            top_idx = torch.argsort(sim, descending=True)[:K].tolist()
            gt_rank_pos = (torch.argsort(sim, descending=True) == i).nonzero(as_tuple=False)
            gt_rank = int(gt_rank_pos[0, 0].item()) + 1 if gt_rank_pos.numel() > 0 else -1
            print(f"\n[{i}] name={all_names[i]} | gt_rank={gt_rank}")
            print(f"gt_text: {all_texts[i]}")
            for j, idx in enumerate(top_idx, start=1):
                flag = "*" if idx == i else " "
                print(f"  {j:>2}{flag} name={all_names[idx]} | text={all_texts[idx]}")
    except Exception as e:
        print(f"[examples] skipped due to error: {e}")


if __name__ == "__main__":
    main()

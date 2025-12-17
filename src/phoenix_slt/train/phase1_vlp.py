"""Phase 1: Visual-language pretraining loop (contrastive alignment).

Supports both single-GPU, DataParallel, and DistributedDataParallel (torchrun).
"""

import os
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from phoenix_slt.config import (
    BATCH_SIZE_PHASE1,
    D_MODEL,
    VLP_CHECKPOINT_PATH,
    VLP_EPOCHS,
    VLP_LR,
    VLP_PATIENCE,
)
from phoenix_slt.data.datasets import build_loaders, load_splits, load_tokenizer
from phoenix_slt.models.modeling import SqueezeformerFusionEncoder, VLP_PretrainingModel


def evaluate_vlp(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute average validation loss for VLP contrastive alignment."""
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            for k in ["kpts", "kpts_mask", "siglip", "labels", "labels_mask"]:
                batch[k] = batch[k].to(device)
            vis_vec, txt_vec = model(batch)
            # Retrieve scalar logit scale from underlying module (handles DP/DDP)
            target = model.module if hasattr(model, "module") else model
            scale = target.logit_scale.exp()
            logits = scale * (vis_vec @ txt_vec.t())
            targets = torch.arange(len(logits), device=device)
            loss = (criterion(logits, targets) + criterion(logits.t(), targets)) / 2
            total_val_loss += loss.item()
    return total_val_loss / len(loader)


def main():
    """Run Phase 1 visual-language pretraining and save the best encoder."""
    # DDP setup
    ddp = False
    if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        ddp = True
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if rank == 0:
        print(f"Using device: {device} | GPUs: {n_gpu} | DDP: {ddp}")

    tokenizer = load_tokenizer()
    print(f"Tokenizer vocab: {len(tokenizer)} | PAD: {tokenizer.pad_token_id}")

    df_train, df_dev, df_test = load_splits()
    if rank == 0:
        print(
            f"Splits -> train: {len(df_train)}, dev: {len(df_dev)}, test: {len(df_test)}"
        )

    train_loader, dev_loader, _ = build_loaders(
        df_train,
        df_dev,
        df_test,
        tokenizer,
        batch_size=BATCH_SIZE_PHASE1,
        distributed=ddp,
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        print(f"Batches -> train: {len(train_loader)}, dev: {len(dev_loader)}")

    encoder = SqueezeformerFusionEncoder().to(device)
    vlp_model = VLP_PretrainingModel(encoder).to(device)
    if ddp:
        vlp_model = nn.parallel.DistributedDataParallel(
            vlp_model, device_ids=[device.index], output_device=device.index
        )
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if rank == 0:
            print("Multiple GPUs detected — enabling DataParallel for Phase 1.")
        vlp_model = nn.DataParallel(vlp_model)

    optimizer = optim.AdamW(vlp_model.parameters(), lr=VLP_LR, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=VLP_EPOCHS, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()
    amp_enabled = device.type == "cuda"
    scaler = GradScaler("cuda" if amp_enabled else "cpu", enabled=amp_enabled)

    if rank == 0:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "phoenix-slt"),
            name=os.environ.get("WANDB_RUN_NAME", "phase1-vlp"),
            config={
                "phase": "vlp",
                "batch_size": BATCH_SIZE_PHASE1,
                "epochs": VLP_EPOCHS,
                "lr": VLP_LR,
                "patience": VLP_PATIENCE,
                "weight_decay": 0.001,
                "d_model": D_MODEL,
                "ddp": ddp,
                "world_size": world_size,
            },
        )

    # Ensure checkpoint directory exists before any save
    VLP_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(VLP_EPOCHS):
        # Ensure proper shuffling across epochs with DistributedSampler
        if ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        # Optional sync to align ranks at epoch boundaries (helps avoid stragglers)
        if ddp and dist.is_initialized():
            dist.barrier()
        vlp_model.train()
        total_train_loss = 0.0
        pbar = tqdm(
            train_loader if rank == 0 else train_loader,
            desc=f"[VLP] Epoch {epoch + 1}/{VLP_EPOCHS}",
            leave=False if rank == 0 else False,
        )
        for batch in pbar:
            for k in ["kpts", "kpts_mask", "siglip", "labels", "labels_mask"]:
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            with autocast(
                device_type="cuda" if amp_enabled else "cpu",
                dtype=torch.float16 if amp_enabled else torch.bfloat16,
                enabled=True,
            ):
                vis_vec, txt_vec = vlp_model(batch)
                target = vlp_model.module if hasattr(vlp_model, "module") else vlp_model
                scale = target.logit_scale.exp()
                logits = scale * (vis_vec @ txt_vec.t())
                targets = torch.arange(len(logits), device=device)
                loss = (criterion(logits, targets) + criterion(logits.t(), targets)) / 2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train = total_train_loss / len(train_loader)
        scheduler.step()
        avg_val = evaluate_vlp(vlp_model, dev_loader, criterion, device)
        # Help mitigate fragmentation across long runs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if ddp and dist.is_initialized():
            # Keep all ranks in lockstep between phases to prevent hangs
            dist.barrier()
        if rank == 0:
            print(
                f"Epoch {epoch + 1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}"
            )
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train,
                    "val_loss": avg_val,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        if avg_val < best_val_loss and rank == 0:
            best_val_loss = avg_val
            patience = 0
            target = vlp_model.module if hasattr(vlp_model, "module") else vlp_model
            torch.save(target.visual_encoder.state_dict(), VLP_CHECKPOINT_PATH)
            print(f"  -> Saved encoder to {VLP_CHECKPOINT_PATH}")
            wandb.summary["best_val_loss"] = best_val_loss
        else:
            patience += 1
            if patience >= VLP_PATIENCE:
                if rank == 0:
                    print(f"Early stop at epoch {epoch + 1}")
                break

    if rank == 0:
        print(f"Phase 1 complete. Best encoder at {VLP_CHECKPOINT_PATH}")
        wandb.finish()
    if ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Phase 1: Visual-language pretraining loop (contrastive alignment).

Supports both single-GPU, DataParallel, and DistributedDataParallel (torchrun).
"""

import os
import copy
from datetime import timedelta
import math
import torch.distributed as dist

import torch
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_hooks
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from tqdm.auto import tqdm

from phoenix_slt.config import (
    BATCH_SIZE_PHASE1,
    D_MODEL,
    VLP_CHECKPOINT_PATH,
    VLP_FULL_CHECKPOINT_PATH,
    VLP_EPOCHS,
    VLP_LR,
    VLP_PATIENCE,
    ACCUMULATE_STEPS,
    USE_KPTS,
    USE_SIGLIP,
    WARMUP_EPOCHS,
    LABEL_SMOOTHING,
    EMA_DECAY,
)

# Gradient clipping to prevent exploding gradients
MAX_GRAD_NORM = 1.0
from phoenix_slt.data.datasets import build_loaders, load_splits, load_tokenizer
from phoenix_slt.models.modeling import SqueezeformerFusionEncoder, VLP_PretrainingModel


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.state_dict().items() if p.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.state_dict().items():
            if name in self.shadow and param.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(param, alpha=1.0 - self.decay)

    def clone_model(self, model: nn.Module) -> nn.Module:
        ema_model = copy.deepcopy(model)
        state = ema_model.state_dict()
        for name, value in self.shadow.items():
            if name in state:
                state[name].copy_(value)
        ema_model.load_state_dict(state, strict=True)
        return ema_model


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
                scale = target.logit_scale.clamp(min=math.log(1 / 1000), max=math.log(100)).exp()
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
            backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl")
            dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
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
        ddp_backend = (
            os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl") if ddp else "none"
        )
        print(f"Using device: {device} | GPUs: {n_gpu} | DDP: {ddp} (backend={ddp_backend})")

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
        ddp_backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl").lower()
        enable_static_graph = (
            os.environ.get("DDP_STATIC_GRAPH", "1") == "1" and ddp_backend == "nccl"
        )
        bucket_cap_mb = int(os.environ.get("DDP_BUCKET_CAP_MB", "25"))
        enable_fp16_hook = (
            os.environ.get("DDP_FP16_HOOK", "1") == "1" and ddp_backend == "nccl"
        )

        vlp_model = nn.parallel.DistributedDataParallel(
            vlp_model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=enable_static_graph,
            bucket_cap_mb=bucket_cap_mb,
        )
        # Compress gradients to fp16 to reduce bandwidth (NCCL only)
        if enable_fp16_hook:
            try:
                vlp_model.register_comm_hook(state=None, hook=ddp_hooks.fp16_compress_hook)
                if rank == 0:
                    print("DDP: fp16 gradient compression hook enabled")
            except Exception as e:
                if rank == 0:
                    print(f"DDP: could not enable fp16 comm hook: {e}")
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if rank == 0:
            print("Multiple GPUs detected — enabling DataParallel for Phase 1.")
        vlp_model = nn.DataParallel(vlp_model)

    optimizer = optim.AdamW(vlp_model.parameters(), lr=VLP_LR, weight_decay=0.001)
    # Warmup + cosine schedule (epoch-wise)
    warmup_epochs = max(0, min(WARMUP_EPOCHS, VLP_EPOCHS - 1))
    schedulers = []
    milestones = []
    if warmup_epochs > 0:
        schedulers.append(LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs))
        milestones.append(warmup_epochs)
    cosine_epochs = max(1, VLP_EPOCHS - warmup_epochs)
    schedulers.append(optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6))
    scheduler = SequentialLR(optimizer, schedulers, milestones=milestones) if milestones else schedulers[0]
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    # EMA tracker (on the non-wrapped model)
    ema_model_ref = vlp_model.module if hasattr(vlp_model, "module") else vlp_model
    ema = EMA(ema_model_ref, decay=EMA_DECAY)
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
                "use_kpts": USE_KPTS,
                "use_siglip": USE_SIGLIP,
                "warmup_epochs": warmup_epochs,
                "label_smoothing": LABEL_SMOOTHING,
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
        vlp_model.train()
        total_train_loss = 0.0
        accumulate_steps = max(1, int(ACCUMULATE_STEPS))
        step_counter = 0
        pbar = tqdm(
            train_loader if rank == 0 else train_loader,
            desc=f"[VLP] Epoch {epoch + 1}/{VLP_EPOCHS}",
            leave=False if rank == 0 else False,
        )
        for batch in pbar:
            for k in ["kpts", "kpts_mask", "siglip", "labels", "labels_mask"]:
                batch[k] = batch[k].to(device)
            if step_counter % accumulate_steps == 0:
                optimizer.zero_grad()
            with autocast(
                device_type="cuda" if amp_enabled else "cpu",
                dtype=torch.float16 if amp_enabled else torch.bfloat16,
                enabled=True,
            ):
                vis_vec, txt_vec = vlp_model(batch)
                target = vlp_model.module if hasattr(vlp_model, "module") else vlp_model
                scale = target.logit_scale.clamp(min=math.log(1 / 1000), max=math.log(100)).exp()
                logits = scale * (vis_vec @ txt_vec.t())
                targets = torch.arange(len(logits), device=device)
                loss = (criterion(logits, targets) + criterion(logits.t(), targets)) / 2
                # Scale loss for gradient accumulation
                loss = loss / accumulate_steps
            
            # Check for NaN/Inf before backward
            if not torch.isfinite(loss):
                if rank == 0:
                    print(f"WARNING: Non-finite loss detected at epoch {epoch+1}, skipping batch")
                continue
            
            scaler.scale(loss).backward()

            step_counter += 1
            if step_counter % accumulate_steps == 0:
                # Unscale and clip right before stepping to avoid multiple unscale_ calls
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vlp_model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                # EMA update on the base model
                ema.update(ema_model_ref)
            total_train_loss += loss.item() * accumulate_steps
            pbar.set_postfix({"loss": f"{(loss.item() * accumulate_steps):.4f}"})

        avg_train = total_train_loss / len(train_loader)
        scheduler.step()
        # Evaluate with EMA-smoothed weights for stability
        with torch.no_grad():
            ema_eval_model = ema.clone_model(ema_model_ref)
            avg_val = evaluate_vlp(ema_eval_model, dev_loader, criterion, device)
        # Help mitigate fragmentation across long runs
        if device.type == "cuda":
            torch.cuda.empty_cache()
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
            target = ema_eval_model if "ema_eval_model" in locals() else (vlp_model.module if hasattr(vlp_model, "module") else vlp_model)
            torch.save(target.visual_encoder.state_dict(), VLP_CHECKPOINT_PATH)
            torch.save(target.state_dict(), VLP_FULL_CHECKPOINT_PATH)
            print(f"  -> Saved encoder to {VLP_CHECKPOINT_PATH}")
            print(f"  -> Saved full VLP model to {VLP_FULL_CHECKPOINT_PATH}")
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

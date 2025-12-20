"""Phase 2: Sign language translation fine-tuning loop (BLEU-optimized)."""

import gc
import math
import os
from datetime import timedelta
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import wandb
from sacrebleu.metrics import BLEU
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from phoenix_slt.config import (
    ACCUMULATE_STEPS,
    BEST_SLT_CKPT,
    BATCH_SIZE_PHASE2,
    DECODER_LR,
    ENCODER_CKPT,
    ENCODER_LR,
    LABEL_SMOOTHING,
    PATIENCE,
    SLT_EPOCHS,
    SLT_WEIGHT_DECAY,
    WARMUP_EPOCHS,
    USE_CTC_LOSS,
    CTC_WEIGHT,
    USE_CURRICULUM,
    CURRICULUM_EPOCHS,
    ENSEMBLE_CHECKPOINTS,
)
from phoenix_slt.data.datasets import build_loaders, load_splits, load_tokenizer
from phoenix_slt.models.modeling import SignTranslationModel, SqueezeformerFusionEncoder


def build_param_groups(
    model: nn.Module, enc_lr: float, dec_lr: float, weight_decay: float
):
    """Create encoder/decoder param groups with differential LRs and decay."""
    enc_decay, enc_no_decay = [], []
    dec_decay, dec_no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder = name.startswith("encoder") or name.startswith("adapter")
        is_decoder = name.startswith("mbart.model.decoder")
        if not is_encoder and not is_decoder and name.startswith("mbart"):
            is_decoder = True
        if not is_encoder and not is_decoder:
            is_encoder = True
        is_bias_or_norm = len(param.shape) == 1 or name.endswith(".bias")
        if is_encoder:
            (enc_no_decay if is_bias_or_norm else enc_decay).append(param)
        else:
            (dec_no_decay if is_bias_or_norm else dec_decay).append(param)
    return [
        {"params": enc_decay, "weight_decay": weight_decay, "lr": enc_lr},
        {"params": enc_no_decay, "weight_decay": 0.0, "lr": enc_lr},
        {"params": dec_decay, "weight_decay": weight_decay, "lr": dec_lr},
        {"params": dec_no_decay, "weight_decay": 0.0, "lr": dec_lr},
    ]


def generate_from_batch(
    model: nn.Module, batch, tokenizer, device: torch.device
) -> torch.Tensor:
    """Greedily decode translations from a single batch using mBART generate."""
    model.eval()
    base = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        enc_out, _ = base.encoder(
            batch["kpts"], batch["kpts_mask"], batch["siglip"]
        )
        enc_out = base.adapter(enc_out)
        generated_ids = base.mbart.generate(
            inputs_embeds=enc_out,
            attention_mask=batch["kpts_mask"],
            num_beams=8,          # Increased from 5 for better search
            max_new_tokens=60,    # Increased from 50 to allow longer outputs
            length_penalty=1.0,   # Changed from 0.9 (was too short)
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=False, # Explore more candidates
            forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"],
            eos_token_id=tokenizer.eos_token_id,
        )
    return generated_ids


def compute_bleu(
    model: nn.Module,
    loader: DataLoader,
    tokenizer,
    device: torch.device,
    max_batches: int = 30,
) -> float:
    """Estimate BLEU on a subset of the dev loader for quick feedback."""
    bleu_metric = BLEU(effective_order=True)
    preds, refs = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            for k in ["kpts", "kpts_mask", "siglip", "labels"]:
                batch[k] = batch[k].to(device)
            generated_ids = generate_from_batch(model, batch, tokenizer, device)
            preds.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
            refs.extend(tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))
    return bleu_metric.corpus_score(preds, [refs]).score


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    accumulate_steps: int,
    criterion: nn.Module,
    device: torch.device,
    use_ctc: bool = False,
    ctc_weight: float = 0.3,
) -> float:
    """Train SLT for one epoch with AMP and gradient accumulation."""
    model.train()
    if hasattr(model, 'module'):
        model.module.use_ctc = use_ctc
    else:
        model.use_ctc = use_ctc
    
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="Train", leave=False)
    ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True) if use_ctc else None
    
    for step, batch in enumerate(pbar):
        for k in ["kpts", "kpts_mask", "siglip", "labels"]:
            batch[k] = batch[k].to(device)
        with autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.float16,
        ):
            ce_loss, logits, ctc_logits = model(batch, return_ctc=use_ctc)
            
            # Combined loss: cross-entropy + CTC
            if use_ctc and ctc_logits is not None:
                # CTC expects: (T, B, vocab), input_lengths, target_lengths
                log_probs = torch.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T, B, V)
                input_lengths = batch["kpts_mask"].sum(dim=1).long()  # (B,)
                target_lengths = (batch["labels"] != 0).sum(dim=1).long()  # (B,)
                ctc_loss = ctc_criterion(
                    log_probs,
                    batch["labels"],
                    input_lengths,
                    target_lengths,
                )
                loss = (1.0 - ctc_weight) * ce_loss + ctc_weight * ctc_loss
            else:
                loss = ce_loss
            
            loss = loss / accumulate_steps
        scaler.scale(loss).backward()
        if (step + 1) % accumulate_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * accumulate_steps
        pbar.set_postfix({"loss": f"{loss.item() * accumulate_steps:.4f}"})
    return total_loss / len(loader)


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute average cross-entropy loss on the dev set."""
    model.eval()
    if hasattr(model, 'module'):
        model.module.use_ctc = False
    else:
        model.use_ctc = False
    
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            for k in ["kpts", "kpts_mask", "siglip", "labels"]:
                batch[k] = batch[k].to(device)
            with autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.float16,
            ):
                ce_loss, logits, _ = model(batch, return_ctc=False)
                total_loss += ce_loss.item()
    return total_loss / len(loader)


def show_dev_examples(
    model: nn.Module,
    loader: DataLoader,
    tokenizer,
    device: torch.device,
    max_examples: int = 5,
):
    """Print a few dev-set predictions vs references for qualitative check."""
    model.eval()
    print("\n--- Generating Examples (Dev Set) ---")
    batch = next(iter(loader))
    for k in ["kpts", "kpts_mask", "siglip", "labels"]:
        batch[k] = batch[k].to(device)
    generated_ids = generate_from_batch(model, batch, tokenizer, device)
    preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    refs = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
    for i in range(min(len(preds), max_examples)):
        print(f"Example {i + 1}:")
        print(f"  GT:   {refs[i]}")
        print(f"  PRED: {preds[i]}")
        print("-" * 40)


def main():
    """Run Phase 2 SLT fine-tuning with DDP support, logging metrics and saving best BLEU model."""
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

    if rank == 0:
        print(f"Using device: {device} | DDP: {ddp}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if rank == 0:
            print("GPU cache cleared.")

    tokenizer = load_tokenizer()
    df_train, df_dev, df_test = load_splits()
    train_loader, dev_loader, _ = build_loaders(
        df_train, df_dev, df_test, tokenizer, batch_size=BATCH_SIZE_PHASE2,
        distributed=ddp, rank=rank, world_size=world_size
    )
    if rank == 0:
        print(f"Batches -> train: {len(train_loader)}, dev: {len(dev_loader)}")

    encoder = SqueezeformerFusionEncoder(dropout=0.3).to(device)
    model = SignTranslationModel(encoder).to(device)
    
    if ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
                   find_unused_parameters=True,
        )

    # Determine modality suffix for checkpoint naming
    modality_suffix = ""
    if USE_KPTS and USE_SIGLIP:
        modality_suffix = "_kpts_siglip"
    elif USE_KPTS:
        modality_suffix = "_kptsonly"
    elif USE_SIGLIP:
        modality_suffix = "_sigliponly"
    
    encoder_ckpt_path = CHECKPOINTS_DIR / f"vlp_best_encoder{modality_suffix}.pt"
    slt_ckpt_path = CHECKPOINTS_DIR / f"best_slt_model{modality_suffix}.pt"
    
    if rank == 0:
        print(f"Modality: {modality_suffix if modality_suffix else 'unknown'}")
        print(f"Will load encoder from: {encoder_ckpt_path}")
        print(f"Will save SLT model to: {slt_ckpt_path}")

    if os.path.exists(encoder_ckpt_path):
        enc_state = torch.load(encoder_ckpt_path, map_location=device)
        # Load into the encoder module (handle DDP wrapping)
        encoder_module = model.module if ddp else model
        encoder_module.encoder.load_state_dict(enc_state)
        if rank == 0:
            print(f"Loaded encoder weights from {encoder_ckpt_path}")
    else:
        if rank == 0:
            print(f"Warning: {encoder_ckpt_path} not found, encoder will train from scratch.")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING, ignore_index=tokenizer.pad_token_id
    )
    param_groups = build_param_groups(model, ENCODER_LR, DECODER_LR, SLT_WEIGHT_DECAY)
    optimizer = optim.AdamW(param_groups)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    def lr_lambda(epoch: int):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(WARMUP_EPOCHS)
        progress = float(epoch - WARMUP_EPOCHS) / float(max(1, SLT_EPOCHS - WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if rank == 0:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "phoenix-slt"),
            name=os.environ.get("WANDB_RUN_NAME", "phase2-slt"),
            config={
                "phase": "slt",
                "batch_size": BATCH_SIZE_PHASE2,
                "epochs": SLT_EPOCHS,
                "enc_lr": ENCODER_LR,
                "dec_lr": DECODER_LR,
                "weight_decay": SLT_WEIGHT_DECAY,
                "label_smoothing": LABEL_SMOOTHING,
                "warmup_epochs": WARMUP_EPOCHS,
                "accumulate_steps": ACCUMULATE_STEPS,
                "patience": PATIENCE,
                "ddp": ddp,
                "world_size": world_size,
                "use_ctc": USE_CTC_LOSS,
                "ctc_weight": CTC_WEIGHT if USE_CTC_LOSS else 0.0,
                "use_curriculum": USE_CURRICULUM,
                "curriculum_epochs": CURRICULUM_EPOCHS if USE_CURRICULUM else 0,
            },
        )

    # Load existing best BLEU if checkpoint exists
    best_bleu = 0.0
    best_dev_loss = float("inf")
    if rank == 0 and os.path.exists(slt_ckpt_path):
        try:
            existing_state = torch.load(slt_ckpt_path, map_location="cpu")
            print(f"Found existing checkpoint: {slt_ckpt_path}")
            print(f"Will only save if BLEU improves by +0.2")
        except Exception as e:
            print(f"Could not load existing checkpoint metadata: {e}")
    
    no_improve = 0

    if rank == 0:
        print(f"Starting SLT fine-tuning for {SLT_EPOCHS} epochs...")
    
    for epoch in range(SLT_EPOCHS):
        # Ensure proper shuffling across epochs with DistributedSampler
        if ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        # Curriculum learning: enable CTC gradually
        use_ctc_this_epoch = USE_CTC_LOSS
        if USE_CURRICULUM and epoch < CURRICULUM_EPOCHS:
            # Ramp up CTC weight linearly
            current_ctc_weight = CTC_WEIGHT * (epoch + 1) / CURRICULUM_EPOCHS
        else:
            current_ctc_weight = CTC_WEIGHT if USE_CTC_LOSS else 0.0
        
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            ACCUMULATE_STEPS,
            criterion,
            device,
            use_ctc=use_ctc_this_epoch,
            ctc_weight=current_ctc_weight,
        )
        dev_loss = evaluate_loss(model, dev_loader, criterion, device)
        dev_bleu = compute_bleu(model, dev_loader, tokenizer, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{SLT_EPOCHS} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {dev_loss:.4f} | BLEU: {dev_bleu:.2f} | LR: {current_lr:.2e}"
            )

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": dev_loss,
                "bleu": dev_bleu,
                "lr": current_lr,
            })

            if dev_bleu > best_bleu + 0.2:
                best_bleu = dev_bleu
                best_dev_loss = dev_loss
                no_improve = 0
                model_to_save = model.module if ddp else model
                torch.save(model_to_save.state_dict(), slt_ckpt_path)
                print(
                    f" -> New best saved to {slt_ckpt_path} (BLEU {best_bleu:.2f}, Val {best_dev_loss:.4f})"
                )
                wandb.summary["best_bleu"] = best_bleu
                wandb.summary["best_val_loss"] = best_dev_loss
            else:
                no_improve += 1
                print(f" -> No BLEU improvement ({no_improve}/{PATIENCE})")
                if no_improve >= PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1} (BLEU plateau)")
                    break

            show_dev_examples(model, dev_loader, tokenizer, device, max_examples=5)

    if rank == 0:
        print(
            f"Training complete. Best BLEU: {best_bleu:.2f} saved at {BEST_SLT_CKPT}. "
            f"Best Val Loss: {best_dev_loss:.4f}"
        )
        wandb.finish()
    
    if ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

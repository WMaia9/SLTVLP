"""Phase 2: Sign language translation fine-tuning loop (BLEU-optimized)."""

import gc
import math
import os
from typing import List, Tuple

import torch
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
    with torch.no_grad():
        enc_out, _ = model.encoder(
            batch["kpts"], batch["kpts_mask"], batch["siglip"]
        )
        enc_out = model.adapter(enc_out)
        generated_ids = model.mbart.generate(
            inputs_embeds=enc_out,
            attention_mask=batch["kpts_mask"],
            num_beams=5,
            max_new_tokens=50,
            length_penalty=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
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
) -> float:
    """Train SLT for one epoch with AMP and gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="Train", leave=False)
    for step, batch in enumerate(pbar):
        for k in ["kpts", "kpts_mask", "siglip", "labels"]:
            batch[k] = batch[k].to(device)
        with autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.float16,
        ):
            _, logits = model(batch)
            loss = criterion(
                logits.view(-1, logits.shape[-1]),
                batch["labels"].view(-1),
            )
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
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            for k in ["kpts", "kpts_mask", "siglip", "labels"]:
                batch[k] = batch[k].to(device)
            with autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.float16,
            ):
                _, logits = model(batch)
                loss = criterion(
                    logits.view(-1, logits.shape[-1]),
                    batch["labels"].view(-1),
                )
            total_loss += loss.item()
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
    """Run Phase 2 SLT fine-tuning, logging metrics and saving best BLEU model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared.")

    tokenizer = load_tokenizer()
    df_train, df_dev, df_test = load_splits()
    train_loader, dev_loader, _ = build_loaders(
        df_train, df_dev, df_test, tokenizer, batch_size=BATCH_SIZE_PHASE2
    )
    print(f"Batches -> train: {len(train_loader)}, dev: {len(dev_loader)}")

    encoder = SqueezeformerFusionEncoder(dropout=0.3).to(device)
    model = SignTranslationModel(encoder).to(device)

    if os.path.exists(ENCODER_CKPT):
        enc_state = torch.load(ENCODER_CKPT, map_location=device)
        model.encoder.load_state_dict(enc_state)
        print(f"Loaded encoder weights from {ENCODER_CKPT}")
    else:
        print(f"Warning: {ENCODER_CKPT} not found, encoder will train from scratch.")

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
        },
    )

    best_bleu = 0.0
    best_dev_loss = float("inf")
    no_improve = 0

    print(f"Starting SLT fine-tuning for {SLT_EPOCHS} epochs...")
    for epoch in range(SLT_EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            ACCUMULATE_STEPS,
            criterion,
            device,
        )
        dev_loss = evaluate_loss(model, dev_loader, criterion, device)
        dev_bleu = compute_bleu(model, dev_loader, tokenizer, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
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
            torch.save(model.state_dict(), BEST_SLT_CKPT)
            print(
                f" -> New best saved to {BEST_SLT_CKPT} (BLEU {best_bleu:.2f}, Val {best_dev_loss:.4f})"
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

    print(
        f"Training complete. Best BLEU: {best_bleu:.2f} saved at {BEST_SLT_CKPT}. "
        f"Best Val Loss: {best_dev_loss:.4f}"
    )
    wandb.finish()


if __name__ == "__main__":
    main()

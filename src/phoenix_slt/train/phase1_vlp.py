"""Phase 1: Visual-language pretraining loop (contrastive alignment)."""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
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
            vis_vec, txt_vec, scale = model(batch)
            logits = scale * (vis_vec @ txt_vec.t())
            targets = torch.arange(len(logits), device=device)
            loss = (criterion(logits, targets) + criterion(logits.t(), targets)) / 2
            total_val_loss += loss.item()
    return total_val_loss / len(loader)


def main():
    """Run Phase 1 visual-language pretraining and save the best encoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Using device: {device} | GPUs: {n_gpu}")

    tokenizer = load_tokenizer()
    print(f"Tokenizer vocab: {len(tokenizer)} | PAD: {tokenizer.pad_token_id}")

    df_train, df_dev, df_test = load_splits()
    print(
        f"Splits -> train: {len(df_train)}, dev: {len(df_dev)}, test: {len(df_test)}"
    )

    train_loader, dev_loader, _ = build_loaders(
        df_train, df_dev, df_test, tokenizer, batch_size=BATCH_SIZE_PHASE1
    )
    print(f"Batches -> train: {len(train_loader)}, dev: {len(dev_loader)}")

    encoder = SqueezeformerFusionEncoder().to(device)
    vlp_model = VLP_PretrainingModel(encoder).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Multiple GPUs detected — enabling DataParallel for Phase 1.")
        vlp_model = nn.DataParallel(vlp_model)

    optimizer = optim.AdamW(vlp_model.parameters(), lr=VLP_LR, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=VLP_EPOCHS, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()

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
        },
    )

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(VLP_EPOCHS):
        vlp_model.train()
        total_train_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"[VLP] Epoch {epoch + 1}/{VLP_EPOCHS}",
            leave=False,
        )
        for batch in pbar:
            for k in ["kpts", "kpts_mask", "siglip", "labels", "labels_mask"]:
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            vis_vec, txt_vec, scale = vlp_model(batch)
            logits = scale * (vis_vec @ txt_vec.t())
            targets = torch.arange(len(logits), device=device)
            loss = (criterion(logits, targets) + criterion(logits.t(), targets)) / 2
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train = total_train_loss / len(train_loader)
        scheduler.step()
        avg_val = evaluate_vlp(vlp_model, dev_loader, criterion, device)
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

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience = 0
            target = vlp_model.module if isinstance(vlp_model, nn.DataParallel) else vlp_model
            torch.save(target.visual_encoder.state_dict(), VLP_CHECKPOINT_PATH)
            print(f"  -> Saved encoder to {VLP_CHECKPOINT_PATH}")
            wandb.summary["best_val_loss"] = best_val_loss
        else:
            patience += 1
            if patience >= VLP_PATIENCE:
                print(f"Early stop at epoch {epoch + 1}")
                break

    print(f"Phase 1 complete. Best encoder at {VLP_CHECKPOINT_PATH}")
    wandb.finish()


if __name__ == "__main__":
    main()

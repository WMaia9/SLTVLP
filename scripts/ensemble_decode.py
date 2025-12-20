#!/usr/bin/env python
"""Ensemble decoding: average logits from multiple checkpoints for better BLEU.

Usage:
    python scripts/ensemble_decode.py \
        --ckpts checkpoints/best_slt_model.pt checkpoints/slt_epoch_25.pt checkpoints/slt_epoch_30.pt \
        --split dev
"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from phoenix_slt.config import BATCH_SIZE_PHASE2, NUM_WORKERS
from phoenix_slt.data.datasets import load_splits, build_loaders, load_tokenizer
from phoenix_slt.models.modeling import SqueezeformerFusionEncoder, SignTranslationModel


def ensemble_generate(
    models,
    batch,
    tokenizer,
    device,
    num_beams=8,
    max_new_tokens=60,
    length_penalty=1.0,
):
    """Generate translations by averaging encoder outputs from multiple models."""
    # Average encoder outputs
    enc_outs = []
    for model in models:
        model.eval()
        base = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            enc_out, _ = base.encoder(batch["kpts"], batch["kpts_mask"], batch["siglip"])
            enc_out = base.adapter(enc_out)
            enc_outs.append(enc_out)
    
    # Average
    enc_out_avg = torch.stack(enc_outs).mean(dim=0)
    
    # Decode with first model's mBART (all share same decoder)
    base = models[0].module if hasattr(models[0], "module") else models[0]
    with torch.no_grad():
        generated_ids = base.mbart.generate(
            inputs_embeds=enc_out_avg,
            attention_mask=batch["kpts_mask"],
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=False,
            forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"],
            eos_token_id=tokenizer.eos_token_id,
        )
    return generated_ids


def main():
    parser = argparse.ArgumentParser(description="Ensemble decoding for SLT")
    parser.add_argument("--ckpts", nargs="+", required=True, help="Checkpoint paths")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_PHASE2)
    parser.add_argument("--num-beams", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ensembling {len(args.ckpts)} checkpoints:")
    for ckpt in args.ckpts:
        print(f"  - {ckpt}")

    tokenizer = load_tokenizer()
    df_train, df_dev, df_test = load_splits()
    
    if args.split == "dev":
        df_a, df_b, df_c = df_train, df_dev, df_test.iloc[:0]
    else:
        df_a, df_b, df_c = df_train, df_test, df_test.iloc[:0]

    _, eval_loader, _ = build_loaders(
        df_a, df_b, df_c, tokenizer,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        distributed=False,
    )

    # Load all models
    models = []
    for ckpt_path in args.ckpts:
        encoder = SqueezeformerFusionEncoder(dropout=0.3).to(device)
        model = SignTranslationModel(encoder).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        print(f"Loaded: {ckpt_path}")

    # Evaluate ensemble
    bleu_metric = BLEU(effective_order=True)
    preds, refs = [], []
    
    print(f"\nGenerating translations on {args.split} set...")
    for batch in tqdm(eval_loader):
        for k in ["kpts", "kpts_mask", "siglip", "labels"]:
            batch[k] = batch[k].to(device)
        
        generated_ids = ensemble_generate(
            models, batch, tokenizer, device,
            num_beams=args.num_beams,
            max_new_tokens=args.max_tokens,
            length_penalty=args.length_penalty,
        )
        
        preds.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        refs.extend(tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))

    bleu_score = bleu_metric.corpus_score(preds, [refs]).score
    print(f"\n[{args.split}] Ensemble BLEU: {bleu_score:.2f}")
    
    # Show examples
    print("\nExamples:")
    for i in range(min(5, len(preds))):
        print(f"\n{i+1}. REF:  {refs[i]}")
        print(f"   PRED: {preds[i]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Evaluate Phase 2 (SLT) model on dev or test set.

Usage:
    python scripts/eval_slt.py --split dev --ckpt checkpoints/best_slt_model.pt
    python scripts/eval_slt.py --split test --ckpt checkpoints/best_slt_model.pt --examples 10
"""

import argparse
from pathlib import Path
import torch
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from phoenix_slt.config import BATCH_SIZE_PHASE2, NUM_WORKERS, BEST_SLT_CKPT
from phoenix_slt.data.datasets import load_splits, build_loaders, load_tokenizer
from phoenix_slt.models.modeling import SqueezeformerFusionEncoder, SignTranslationModel


def generate_translations(model, loader, tokenizer, device, num_beams=8, max_tokens=60):
    """Generate translations for entire dataset."""
    model.eval()
    base = model.module if hasattr(model, "module") else model
    
    preds, refs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating"):
            for k in ["kpts", "kpts_mask", "siglip", "labels"]:
                batch[k] = batch[k].to(device)
            
            enc_out, _ = base.encoder(batch["kpts"], batch["kpts_mask"], batch["siglip"])
            enc_out = base.adapter(enc_out)
            generated_ids = base.mbart.generate(
                inputs_embeds=enc_out,
                attention_mask=batch["kpts_mask"],
                num_beams=num_beams,
                max_new_tokens=max_tokens,
                length_penalty=1.0,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=False,
                forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"],
                eos_token_id=tokenizer.eos_token_id,
            )
            
            preds.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
            refs.extend(tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))
    
    return preds, refs


def main():
    parser = argparse.ArgumentParser(description="Evaluate SLT model")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--ckpt", type=str, default=str(BEST_SLT_CKPT))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_PHASE2)
    parser.add_argument("--num-beams", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--examples", type=int, default=5, help="Number of examples to show")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

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

    # Load model
    encoder = SqueezeformerFusionEncoder(dropout=0.3).to(device)
    model = SignTranslationModel(encoder).to(device)
    
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded model from {ckpt_path}")

    # Generate and evaluate
    preds, refs = generate_translations(
        model, eval_loader, tokenizer, device,
        num_beams=args.num_beams,
        max_tokens=args.max_tokens,
    )

    # Compute BLEU
    bleu_metric = BLEU(effective_order=True)
    bleu_score = bleu_metric.corpus_score(preds, [refs]).score
    
    print(f"\n{'='*60}")
    print(f"[{args.split.upper()}] BLEU Score: {bleu_score:.2f}")
    print(f"{'='*60}\n")

    # Show examples
    print(f"Sample Translations (first {args.examples}):")
    print("="*60)
    for i in range(min(args.examples, len(preds))):
        print(f"\n[{i+1}]")
        print(f"REF:  {refs[i]}")
        print(f"PRED: {preds[i]}")
        print("-"*60)


if __name__ == "__main__":
    main()

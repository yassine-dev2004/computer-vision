"""Train the gloss -> English Transformer from scratch on ASLG-PC12.

    python -m grammar.train                     # downloads ASLG-PC12 from Hugging Face
    python -m grammar.train --data pairs.tsv    # or a local CSV/TSV with gloss,text columns
"""
import argparse
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import PAD_IDX, GlossTextDataset, Vocab, collate, detokenize, load_pairs, split_pairs
from .model import GlossTranslator
from .translate import translate_batch


def bleu_chrf(hyps, refs):
    import sacrebleu

    hyps, refs = [h.lower() for h in hyps], [r.lower() for r in refs]
    return sacrebleu.corpus_bleu(hyps, [refs]).score, sacrebleu.corpus_chrf(hyps, [refs]).score


def evaluate_translations(model, vocab, pairs, beam_size=1, batch_size=64):
    model.eval()
    hyps = []
    for i in range(0, len(pairs), batch_size):
        hyps += translate_batch(model, vocab, [src for src, _ in pairs[i:i + batch_size]], beam_size)
    return bleu_chrf(hyps, [detokenize(tgt) for _, tgt in pairs]), hyps


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    total, batches = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1])
        total += loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1)).item()
        batches += 1
    return total / max(batches, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="local CSV/TSV with gloss,text columns (default: Hugging Face ASLG-PC12)")
    parser.add_argument("--out", default="checkpoints/grammar")
    parser.add_argument("--limit", type=int, help="use only the first N pairs (quick tests)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4, help="peak learning rate after warmup")
    parser.add_argument("--warmup", type=int, default=2000, help="warmup steps")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=64, help="drop pairs longer than this (tokens)")
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--mask-prob", type=float, default=0.1, help="chance an in-vocab shared word becomes an <fsN> slot")
    parser.add_argument("--bleu-samples", type=int, default=1000, help="validation sentences decoded each epoch")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.data)
    if args.limit:
        pairs = pairs[:args.limit]
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, seed=args.seed)
    vocab = Vocab.build([s for s, _ in train_pairs] + [t for _, t in train_pairs], min_freq=args.min_freq)
    vocab.save(out / "vocab.json")
    print(f"pairs: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test, vocab {len(vocab)}, device {device}")

    train_loader = DataLoader(GlossTextDataset(train_pairs, vocab, args.max_len, augment=True, mask_prob=args.mask_prob),
                              batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(GlossTextDataset(val_pairs, vocab, args.max_len),
                            batch_size=args.batch_size, collate_fn=collate)

    config = dict(vocab_size=len(vocab), d_model=args.d_model, nhead=args.heads,
                  num_encoder_layers=args.layers, num_decoder_layers=args.layers,
                  dim_feedforward=args.ff, dropout=args.dropout)
    model = GlossTranslator(**config).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    # linear warmup, then inverse square-root decay (the original Transformer schedule)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / args.warmup, math.sqrt(args.warmup / (step + 1))))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ASLG-PC12 glosses are close to English, so "output the gloss unchanged" is the baseline to beat
    bleu_pairs = val_pairs[:args.bleu_samples]
    refs = [detokenize(t) for _, t in bleu_pairs]
    base_bleu, base_chrf = bleu_chrf([detokenize(s) for s, _ in bleu_pairs], refs)
    print(f"copy-the-gloss baseline: BLEU {base_bleu:.2f}  chrF {base_chrf:.2f}")

    best_bleu = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total, batches = time.time(), 0.0, 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            with torch.autocast(device_type=device, enabled=use_amp):
                logits = model(src, tgt[:, :-1])
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
            batches += 1

        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        (bleu, chrf), hyps = evaluate_translations(model, vocab, bleu_pairs)
        print(f"epoch {epoch:3d} | train {total / max(batches, 1):.3f} | val {val_loss:.3f} | "
              f"BLEU {bleu:.2f} chrF {chrf:.2f} | {time.time() - start:.0f}s")
        print(f"    {' '.join(bleu_pairs[0][0]).upper()}  ->  {hyps[0]}")
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "bleu": bleu}, out / "model.pt")

    ckpt = torch.load(out / "model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    (bleu, chrf), _ = evaluate_translations(model, vocab, test_pairs, beam_size=5)
    print(f"test (best epoch {ckpt['epoch']}, beam 5): BLEU {bleu:.2f}  chrF {chrf:.2f}")


if __name__ == "__main__":
    main()

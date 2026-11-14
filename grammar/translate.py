"""Gloss -> English inference, used by the live pipeline and by evaluation.

    python -m grammar.translate YESTERDAY fs-JOHN STORE GO
    python -m grammar.translate              # interactive
"""
import argparse
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from .data import BOS_IDX, EOS_IDX, PAD_IDX, Vocab, detokenize, prepare_source, restore_slots
from .model import GlossTranslator


def translate_batch(model, vocab, gloss_batch, beam_size=1, max_len=80):
    """gloss_batch: list of gloss lists. beam_size=1 decodes greedily in one batch."""
    device = next(model.parameters()).device
    prepared = [prepare_source(glosses, vocab) for glosses in gloss_batch]
    if beam_size == 1:
        src = pad_sequence([torch.tensor(vocab.encode(tokens)) for tokens, _ in prepared],
                           batch_first=True, padding_value=PAD_IDX).to(device)
        outputs = model.greedy_decode(src, BOS_IDX, EOS_IDX, max_len).tolist()
    else:
        outputs = [model.beam_search(torch.tensor([vocab.encode(tokens)], device=device),
                                     BOS_IDX, EOS_IDX, beam_size, max_len)
                   for tokens, _ in prepared]
    return [detokenize(restore_slots(vocab.decode(out), slots))
            for out, (_, slots) in zip(outputs, prepared)]


class GlossToEnglish:
    """Wraps a trained checkpoint: GlossToEnglish()(["YESTERDAY", "fs-JOHN", "STORE", "GO"])."""

    def __init__(self, ckpt_dir="checkpoints/grammar", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_dir = Path(ckpt_dir)
        ckpt = torch.load(ckpt_dir / "model.pt", map_location=self.device)
        self.vocab = Vocab.load(ckpt_dir / "vocab.json")
        self.model = GlossTranslator(**ckpt["config"]).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])

    def __call__(self, glosses, beam_size=5):
        if isinstance(glosses, str):
            glosses = glosses.split()
        if not glosses:
            return ""
        return translate_batch(self.model, self.vocab, [glosses], beam_size)[0]


def main():
    parser = argparse.ArgumentParser(description="Translate ASL glosses to English.")
    parser.add_argument("glosses", nargs="*", help="e.g. YESTERDAY fs-JOHN STORE GO (none = interactive)")
    parser.add_argument("--ckpt", default="checkpoints/grammar")
    parser.add_argument("--beam", type=int, default=5)
    args = parser.parse_args()

    translator = GlossToEnglish(args.ckpt)
    if args.glosses:
        print(translator(args.glosses, args.beam))
        return
    while True:
        try:
            line = input("gloss> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line:
            print(translator(line, args.beam))


if __name__ == "__main__":
    main()

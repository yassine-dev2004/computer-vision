"""ASLG-PC12 loading, tokenization, vocabulary and fingerspelling placeholders.

The recognizer emits a mixed stream: word signs as plain glosses ("STORE") and
fingerspelled words with the usual ASL gloss prefix ("fs-JOHN"). Fingerspelled
words are mostly names and rare words the translator has never seen, so they are
swapped for <fsN> slots before translation and put back afterwards.
"""
import csv
import json
import random
import re
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
NUM_SLOTS = 8
SLOTS = [f"<fs{i}>" for i in range(NUM_SLOTS)]
SPECIALS = [PAD, BOS, EOS, UNK] + SLOTS
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = range(4)
FS_PREFIX = "fs-"

_TOKEN_RE = re.compile(r"<fs\d+>|[\w'-]+|[^\w\s]")
_GLOSS_PREFIX_RE = re.compile(r"\b(?:DESC|X)-")


def clean_gloss(gloss):
    """Drop ASLG-PC12 annotation prefixes (X-I -> I, DESC-SURE -> SURE) so training
    glosses look like what the sign recognizer outputs: plain words."""
    return _GLOSS_PREFIX_RE.sub("", gloss)


def tokenize(sentence):
    return _TOKEN_RE.findall(sentence.lower())


def detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r" ([.,!?;:%)])", r"\1", text)
    text = re.sub(r"([($]) ", r"\1", text)
    return text[:1].upper() + text[1:]


class Vocab:
    """Shared source/target vocabulary: glosses are mostly English words, so sharing
    lets the model tie embeddings and copy words straight across."""

    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, token_lists, min_freq=2, max_size=None):
        counter = Counter(t for tokens in token_lists for t in tokens)
        words = [w for w, c in counter.most_common(max_size) if c >= min_freq and w not in SPECIALS]
        return cls(SPECIALS + words)

    def __len__(self):
        return len(self.itos)

    def __contains__(self, token):
        return token in self.stoi

    def encode(self, tokens):
        return [BOS_IDX] + [self.stoi.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i == EOS_IDX:
                break
            if i not in (PAD_IDX, BOS_IDX):
                tokens.append(self.itos[i])
        return tokens

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


def load_pairs(path=None):
    """Tokenized (gloss, text) pairs from a local CSV/TSV with `gloss` and `text`
    columns, or from the Hugging Face copy of ASLG-PC12 when no path is given."""
    if path:
        delimiter = "\t" if str(path).endswith(".tsv") else ","
        with open(path, encoding="utf-8", newline="") as f:
            raw = [(r["gloss"], r["text"]) for r in csv.DictReader(f, delimiter=delimiter)]
    else:
        from datasets import load_dataset

        ds = load_dataset("achrafothman/aslg_pc12", split="train")
        raw = list(zip(ds["gloss"], ds["text"]))
    return [(tokenize(clean_gloss(g)), tokenize(t)) for g, t in raw if g.strip() and t.strip()]


def split_pairs(pairs, val=0.05, test=0.05, seed=0):
    pairs = pairs[:]
    random.Random(seed).shuffle(pairs)
    n_val, n_test = int(len(pairs) * val), int(len(pairs) * test)
    return pairs[n_val + n_test:], pairs[:n_val], pairs[n_val:n_val + n_test]


def mask_shared_words(src, tgt, vocab, prob, rng=random):
    """Training augmentation: words present on both sides become <fsN> slots, always
    when they are out of vocabulary and with probability `prob` otherwise. This
    teaches the model to carry slots through, which is what fingerspelled words
    need at inference."""
    tgt_set = set(tgt)
    chosen = [t for t in dict.fromkeys(src)
              if t in tgt_set and t[0].isalpha() and (t not in vocab or rng.random() < prob)]
    mapping = {w: SLOTS[i] for i, w in enumerate(chosen[:NUM_SLOTS])}
    return [mapping.get(t, t) for t in src], [mapping.get(t, t) for t in tgt]


def prepare_source(glosses, vocab):
    """Recognizer output -> model tokens. `glosses` is e.g. ["YESTERDAY", "fs-JOHN",
    "STORE", "GO"]. Fingerspelled and out-of-vocabulary words go into slots; returns
    (tokens, slots) where slots maps "<fsN>" back to the word to print."""
    tokens, slots = [], {}
    for gloss in glosses:
        if gloss.lower().startswith(FS_PREFIX):
            word = gloss[len(FS_PREFIX):]
            pieces = [(word.lower(), word.capitalize(), True)]
        else:
            pieces = [(t, t, False) for t in tokenize(gloss)]
        for token, shown, spelled in pieces:
            if (spelled or token not in vocab) and token[:1].isalpha():
                slot = next((s for s, w in slots.items() if w == shown), None)
                if slot is None and len(slots) < NUM_SLOTS:
                    slot = SLOTS[len(slots)]
                    slots[slot] = shown
                tokens.append(slot or token)
            else:
                tokens.append(token)
    return tokens, slots


def restore_slots(tokens, slots):
    return [slots.get(t, t) for t in tokens]


class GlossTextDataset(Dataset):
    def __init__(self, pairs, vocab, max_len=64, augment=False, mask_prob=0.1):
        self.pairs = [(s, t) for s, t in pairs if len(s) + 2 <= max_len and len(t) + 2 <= max_len]
        self.vocab = vocab
        self.augment = augment
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        src, tgt = self.pairs[i]
        if self.augment:
            src, tgt = mask_shared_words(src, tgt, self.vocab, self.mask_prob)
        return torch.tensor(self.vocab.encode(src)), torch.tensor(self.vocab.encode(tgt))


def collate(batch):
    src, tgt = zip(*batch)
    return (pad_sequence(src, batch_first=True, padding_value=PAD_IDX),
            pad_sequence(tgt, batch_first=True, padding_value=PAD_IDX))

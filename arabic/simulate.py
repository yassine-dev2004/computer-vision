"""Evaluate the corrector on simulated recitations with real recognizer noise.

    python -m arabic.simulate --cache ~/datasets/arabic_cache --ckpt checkpoints/arabic

Each verse is "signed" letter by letter with mistakes injected at a known rate (wrong letter,
skipped letter, extra sign). Every signed letter gets a probability vector the recognizer really
produced for that letter on a test signer it never saw (ensemble of the two models). The corrector
then reports mistakes, which are compared with the injected ones.

What matters for a revision aid:
    false alarms   correct letters reported as mistakes (should be close to zero)
    recall         injected mistakes that are reported
    repeat asks    letters marked "unclear" (the signer is asked to repeat: annoying, not harmful)
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from recognition.common import PROJECT_DIR

from .corrector import Corrector
from .quran import QURAN_DIR, Quran

SURAHS = (1, 103, 108, 112, 113, 114, 36)


def signed_letter(slot):
    """The sign a signer makes for an expected letter (hamza seats -> their base sign)."""
    return slot.letter if slot.letter in slot.accept else sorted(slot.accept)[0]


def test_probabilities(cache, ckpt):
    index = pd.read_csv(Path(cache) / "index.csv", encoding="utf-8")
    classes = sorted(index["label"].unique())
    lm, cnn = np.load(Path(ckpt) / "test_probs_landmarks.npz"), np.load(Path(ckpt) / "test_probs_cnn.npz")
    probs = 0.5 * lm["probs"] + 0.5 * cnn["probs"]
    labels = index["label"].to_numpy()[lm["index"]]
    pool = defaultdict(list)
    for p, l in zip(probs, labels):
        pool[l].append(p)
    return classes, {l: np.array(v) for l, v in pool.items()}


def simulate_verse(slots, classes, pool, rng, rate, ligatures=0.5):
    """-> list of probability vectors (one per sign), true status per slot, number of extra signs."""
    letters = [c for c in classes if len(c) == 1]
    signs, truth, extras, j = [], [], 0, 0
    while j < len(slots):
        s = slots[j]
        if s.optional:  # nothing to sign
            truth.append("skipped"); j += 1
            continue
        r = rng.random()
        if r < rate * 0.25:  # extra sign before this letter
            signs.append(rng.choice(letters)); extras += 1
        r = rng.random()
        if r < rate * 0.5:  # wrong letter
            wrong = rng.choice([l for l in letters if l not in s.accept])
            signs.append(wrong); truth.append("wrong"); j += 1
        elif r < rate * 0.75:  # skipped letter
            truth.append("missing"); j += 1
        else:
            nxt = slots[j + 1] if j + 1 < len(slots) else None
            pair = signed_letter(s) + (signed_letter(nxt) if nxt and nxt.word == s.word and not nxt.optional else "")
            if pair in ("ال", "لا") and rng.random() < ligatures:  # one sign for two letters
                signs.append(pair); truth += ["ok", "ok"]; j += 2
            else:
                signs.append(signed_letter(s)); truth.append("ok"); j += 1
    probs = [pool[sign][rng.randrange(len(pool[sign]))] for sign in signs]
    return probs, truth, extras


def evaluate(quran, classes, pool, rate, seed, repeats, **options):
    rng = random.Random(seed)
    counts = defaultdict(int)
    for _ in range(repeats):
        for surah in SURAHS:
            for key in quran.verses(surah):
                slots = quran.slots([key])
                probs, truth, extras = simulate_verse(slots, classes, pool, rng, rate)
                c = Corrector(slots, **options)
                for p in probs:
                    c.add_sign(p, classes)
                marks = {m.slot: m.status for m in c.marks() if m.slot >= 0}
                for j, t in enumerate(truth):
                    got = marks.get(j, "pending")
                    flagged = got in ("wrong", "missing")
                    if t == "ok":
                        counts["correct letters"] += 1
                        counts["false alarms"] += flagged
                        counts["repeat asks"] += got == "unclear"
                    elif t in ("wrong", "missing"):
                        counts["mistakes"] += 1
                        counts["caught"] += flagged
                        counts["caught as unclear"] += got == "unclear"
                counts["extra signs"] += extras
                counts["extras reported"] += sum(m.status == "extra" for m in c.marks())
    n_ok, n_bad = counts["correct letters"], max(counts["mistakes"], 1)
    return {"mistake rate": rate,
            "false alarms / 100 correct letters": 100 * counts["false alarms"] / n_ok,
            "repeat asks / 100 correct letters": 100 * counts["repeat asks"] / n_ok,
            "mistakes caught": counts["caught"] / n_bad,
            "caught or asked to repeat": (counts["caught"] + counts["caught as unclear"]) / n_bad,
            "extras reported / extra signs": counts["extras reported"] / max(counts["extra signs"], 1),
            "letters": n_ok + counts["mistakes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default=str(Path.home() / "datasets" / "arabic_cache"))
    parser.add_argument("--ckpt", default=str(PROJECT_DIR / "checkpoints" / "arabic"))
    parser.add_argument("--quran", default=str(QURAN_DIR))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    quran = Quran(args.quran)
    classes, pool = test_probabilities(args.cache, args.ckpt)
    rows = [evaluate(quran, classes, pool, rate, seed=0, repeats=args.repeats) for rate in (0.0, 0.02, 0.05, 0.1)]
    table = pd.DataFrame(rows).set_index("mistake rate")
    print(table.round(3).to_string())
    (Path(args.ckpt) / "corrector_simulation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

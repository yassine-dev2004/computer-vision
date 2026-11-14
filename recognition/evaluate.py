"""Compare the landmark MLP, the CNN and their ensemble on the unseen-signer test set.

    python -m recognition.evaluate
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .common import CACHE_DIR, CKPT_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(CACHE_DIR))
    parser.add_argument("--ckpt", default=str(CKPT_DIR))
    args = parser.parse_args()
    cache, ckpt = Path(args.cache), Path(args.ckpt)

    index = pd.read_csv(cache / "index.csv")
    classes = sorted(index["label"].unique())
    y = index["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()

    lm, cnn = np.load(ckpt / "test_probs_landmarks.npz"), np.load(ckpt / "test_probs_cnn.npz")
    assert (lm["index"] == cnn["index"]).all(), "models were evaluated on different test sets"
    test = lm["index"]
    y_test, groups = y[test], index["group"].to_numpy()[test]

    # the reported ensemble is the plain 0.5 average; other weights are shown for reference only,
    # picking the best one here would tune on the test set
    candidates = {"landmark MLP": lm["probs"], "CNN": cnn["probs"]}
    for w in (0.3, 0.5, 0.7):
        candidates[f"ensemble (CNN weight {w})"] = w * cnn["probs"] + (1 - w) * lm["probs"]

    results = {}
    for name, probs in candidates.items():
        pred = probs.argmax(-1)
        results[name] = {"test_acc": float((pred == y_test).mean()),
                         **{f"acc_{g}": float((pred[groups == g] == y_test[groups == g]).mean()) for g in sorted(set(groups))}}
    table = pd.DataFrame(results).T
    print(table.map(lambda v: f"{v:.2%}").to_string())

    best = "ensemble (CNN weight 0.5)"
    pred = candidates[best].argmax(-1)
    present = np.unique(np.concatenate([y_test, pred]))
    names = [classes[i] for i in present]
    report = classification_report(y_test, pred, labels=present, target_names=names, output_dict=True, zero_division=0)
    worst = sorted(((n, report[n]["f1-score"]) for n in names), key=lambda t: t[1])[:6]
    print("\nweakest classes (F1, ensemble):", ", ".join(f"{n} {f:.2f}" for n, f in worst))

    cm = confusion_matrix(y_test, pred, labels=present, normalize="true")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Unseen signers — {best}: {results[best]['test_acc']:.1%}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(ckpt / "confusion_matrix_ensemble.png", dpi=150)

    with open(ckpt / "report.json", "w") as f:
        json.dump({"results": results, "classification_report_ensemble": report}, f, indent=2)
    print(f"\nsaved {ckpt / 'report.json'} and {ckpt / 'confusion_matrix_ensemble.png'}")


if __name__ == "__main__":
    main()

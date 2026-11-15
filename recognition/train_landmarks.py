"""Train the landmark MLP on cached MediaPipe landmarks (run recognition.extract first).

    python -m recognition.train_landmarks

Also fits a Random Forest on the same features as a classical baseline for the report.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .common import CACHE_DIR, CKPT_DIR, split_indices
from .features import FEATURE_DIM, LandmarkMLP, landmark_features


def class_weights(y, num_classes):
    """Inverse square-root frequency: digits (~100 images) are not drowned by letters (~5000)."""
    counts = np.bincount(y, minlength=num_classes).clip(min=1)
    w = 1 / np.sqrt(counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict_proba(model, image_lm, world_lm, batch_size=8192):
    model.eval()
    out = [model(landmark_features(image_lm[i:i + batch_size], world_lm[i:i + batch_size])).softmax(-1)
           for i in range(0, len(image_lm), batch_size)]
    return torch.cat(out)


def random_forest_baseline(image_lm, world_lm, y, train, test):
    from sklearn.ensemble import RandomForestClassifier

    feats = landmark_features(image_lm, world_lm).cpu().numpy()
    mirrored = landmark_features(image_lm * torch.tensor([-1.0, 1, 1], device=image_lm.device),
                                 world_lm * torch.tensor([-1.0, 1, 1], device=image_lm.device)).cpu().numpy()
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    start = time.time()
    rf.fit(np.concatenate([feats[train], mirrored[train]]), np.concatenate([y[train], y[train]]))
    acc = float((rf.predict(feats[test]) == y[test]).mean())
    return acc, time.time() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default=str(CACHE_DIR))
    parser.add_argument("--out", default=str(CKPT_DIR))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-baseline", action="store_true", help="skip the Random Forest comparison")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(cache / "index.csv")
    lms = np.load(cache / "landmarks.npz")
    classes = sorted(index["label"].unique())
    y = index["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    train, val, test = split_indices(index["group"])
    print(f"{len(index)} hands, {len(classes)} classes | train {len(train)}  val {len(val)}  test {len(test)} | {device}")

    image_lm = torch.tensor(lms["image"], device=device)
    world_lm = torch.tensor(lms["world"], device=device)
    labels = torch.tensor(y, device=device)

    model = LandmarkMLP(len(classes), FEATURE_DIM, args.hidden, args.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y[train], len(classes)).to(device), label_smoothing=0.1)
    steps_per_epoch = int(np.ceil(len(train) / args.batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)

    train_t = torch.tensor(train, device=device)
    best_acc, best_state, start = -1.0, None, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_t[torch.randperm(len(train_t), device=device)].split(args.batch_size):
            x = landmark_features(image_lm[batch], world_lm[batch], augment=True)
            loss = loss_fn(model(x), labels[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        val_acc = (predict_proba(model, image_lm[val], world_lm[val]).argmax(-1) == labels[val]).float().mean().item()
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | loss {total / steps_per_epoch:.3f} | val (unseen signers) {val_acc:.2%} | {time.time() - start:.0f}s")

    model.load_state_dict(best_state)
    probs = predict_proba(model, image_lm[test], world_lm[test]).cpu().numpy()
    pred = probs.argmax(-1)
    test_acc = float((pred == y[test]).mean())
    groups = index["group"].to_numpy()[test]
    by_group = {g: float((pred[groups == g] == y[test][groups == g]).mean()) for g in sorted(set(groups))}
    print(f"\nbest epoch {best_epoch}: val {best_acc:.2%} | TEST (unseen signers) {test_acc:.2%}  {by_group}")

    config = dict(num_classes=len(classes), in_dim=FEATURE_DIM, hidden=args.hidden, dropout=args.dropout)
    torch.save({"model": best_state, "config": config, "classes": classes}, out / "landmark_mlp.pt")
    np.savez(out / "test_probs_landmarks.npz", index=test, probs=probs)

    metrics = {"model": "landmark MLP", "val_acc": best_acc, "best_epoch": best_epoch,
               "test_acc": test_acc, "test_acc_by_group": by_group,
               "train_size": int(len(train)), "val_size": int(len(val)), "test_size": int(len(test))}
    if not args.no_baseline:
        rf_acc, rf_time = random_forest_baseline(image_lm, world_lm, y, train, test)
        metrics["random_forest_test_acc"] = rf_acc
        print(f"Random Forest baseline (same features): TEST {rf_acc:.2%}  ({rf_time:.0f}s)")
    with open(out / "metrics_landmarks.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {out / 'landmark_mlp.pt'}")


if __name__ == "__main__":
    main()

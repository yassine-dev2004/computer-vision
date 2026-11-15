"""Fine-tune an ImageNet-pretrained CNN on the cached hand crops (run recognition.extract first).

    python -m recognition.train_cnn                          # ConvNeXt-Tiny
    python -m recognition.train_cnn --arch efficientnet_b0   # lighter, faster live
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from .common import CACHE_DIR, CKPT_DIR, split_indices

ARCHS = {
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_model(arch, num_classes, pretrained=True):
    factory, weights = ARCHS[arch]
    model = factory(weights=weights if pretrained else None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def eval_transform(size):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def train_transform(size):
    # no random erasing / heavy crops: a hidden fingertip can turn one letter into another
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
        transforms.RandomGrayscale(0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class CropDataset(Dataset):
    def __init__(self, root, files, labels, transform):
        self.root, self.files, self.labels, self.transform = Path(root), list(files), list(labels), transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.root / self.files[i]) as img:
            return self.transform(img.convert("RGB")), self.labels[i]


def balanced_weights(labels, groups):
    """Sampling weights: every class equally likely, and within a class every source/signer
    equally likely, so the single-signer Kaggle images do not dominate."""
    df = pd.DataFrame({"y": labels, "g": groups})
    per_cell = df.groupby(["y", "g"])["y"].transform("size")
    groups_per_class = df.groupby("y")["g"].transform("nunique")
    return (1.0 / (per_cell * groups_per_class)).to_numpy(copy=True)


@torch.no_grad()
def predict_proba(model, loader, device, tta=False):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            p = model(x).float().softmax(-1)
            if tta:  # a mirrored hand is the same letter (left-handed signer)
                p = (p + model(x.flip(-1)).float().softmax(-1)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default=str(CACHE_DIR))
    parser.add_argument("--out", default=str(CKPT_DIR))
    parser.add_argument("--arch", default="convnext_tiny", choices=ARCHS)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--epoch-size", type=int, default=60000, help="images sampled per epoch")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(cache / "index.csv")
    classes = sorted(index["label"].unique())
    y = index["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    groups = index["group"].to_numpy()
    train, val, test = split_indices(groups)
    print(f"{len(index)} crops, {len(classes)} classes | train {len(train)}  val {len(val)}  test {len(test)} | "
          f"{args.arch} on {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}")

    def loader(idx, transform, sampler=None):
        # On Windows every worker loads its own copy of the CUDA libraries (~0.7 GB of commit each):
        # only the training loader keeps its workers alive, evaluation uses a few short-lived ones.
        ds = CropDataset(cache, index["crop"].to_numpy()[idx], y[idx], transform)
        train_mode = sampler is not None
        return DataLoader(ds, batch_size=args.batch_size if train_mode else args.batch_size * 2, sampler=sampler,
                          num_workers=args.workers if train_mode else min(4, args.workers),
                          pin_memory=True, persistent_workers=train_mode)

    sampler = WeightedRandomSampler(balanced_weights(y[train], groups[train]), args.epoch_size, replacement=True)
    train_loader = loader(train, train_transform(args.img_size), sampler)
    val_loader = loader(val, eval_transform(args.img_size))
    test_loader = loader(test, eval_transform(args.img_size))

    model = build_model(args.arch, len(classes)).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total_steps = args.epochs * len(train_loader)
    warmup = len(train_loader) // 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / total_steps))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total, seen = time.time(), 0.0, 0
        for x, target in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = loss_fn(model(x), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(x)
            seen += len(x)
        val_acc = float((predict_proba(model, val_loader, device).argmax(-1) == y[val]).mean())
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d} | loss {total / seen:.3f} | val (unseen signers) {val_acc:.2%} | "
              f"{seen / (time.time() - start):.0f} img/s | {time.time() - start:.0f}s", flush=True)

    model.load_state_dict(best_state)
    probs = predict_proba(model, test_loader, device, tta=True)
    pred = probs.argmax(-1)
    test_acc = float((pred == y[test]).mean())
    by_group = {g: float((pred[groups[test] == g] == y[test][groups[test] == g]).mean()) for g in sorted(set(groups[test]))}
    print(f"\nbest epoch {best_epoch}: val {best_acc:.2%} | TEST (unseen signers, flip TTA) {test_acc:.2%}  {by_group}")

    torch.save({"model": best_state, "arch": args.arch, "img_size": args.img_size, "classes": classes},
               out / f"cnn_{args.arch}.pt")
    np.savez(out / "test_probs_cnn.npz", index=test, probs=probs)
    with open(out / "metrics_cnn.json", "w") as f:
        json.dump({"model": args.arch, "val_acc": best_acc, "best_epoch": best_epoch, "test_acc": test_acc,
                   "test_acc_by_group": by_group, "epochs": args.epochs, "epoch_size": args.epoch_size}, f, indent=2)
    print(f"saved {out / f'cnn_{args.arch}.pt'}")


if __name__ == "__main__":
    main()

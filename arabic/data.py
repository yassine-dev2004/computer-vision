"""Build the training manifest for the Arabic letter recognizer from the public datasets.

    python -m arabic.data --root C:/Users/itrab/datasets/arabic

Writes into --out (default ~/datasets/arabic_cache):
    manifest.csv   path,label,group,source for every image (label = the Arabic letter itself)
    splits.json    which ArSL21L signers are kept for validation and test
    karsl_frames/  frames taken from the middle of the KArSL letter videos

then run the usual recognition pipeline on that cache:
    python -m recognition.extract --manifest CACHE/manifest.csv --out CACHE --num-hands 2
    python -m recognition.train_landmarks --cache CACHE --out checkpoints/arabic
    python -m recognition.train_cnn --cache CACHE --out checkpoints/arabic
    python -m recognition.evaluate --cache CACHE --ckpt checkpoints/arabic
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import pandas as pd

from .letters import ARASL2018_ASLAD, ARSL21L, KARSL_EXCLUDED, KARSL_SIGNS, STATIC_LETTERS

CACHE_DIR = Path.home() / "datasets" / "arabic_cache"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VAL_SIGNERS, TEST_SIGNERS = 8, 12  # of the 50 ArSL21L signers, never seen in training


def arasl2018(root):
    """ArSL2018: 54k 64x64 grey photos, ~40 people, no signer ids -> training only."""
    for folder in sorted((root / "arasl2018" / "ArASL_Database_54K_Final").iterdir()):
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                yield f, ARASL2018_ASLAD[folder.name], "arasl2018", "arasl2018"


def aslad(root, per_class):
    """ASLAD-190K: 190k photos of 2 people. Near-duplicates, so a random subset is plenty."""
    rng = random.Random(0)
    for folder in sorted((root / "aslad190k" / "ASLAD-190K").iterdir()):
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        for f in rng.sample(files, min(per_class, len(files))):
            yield f, ARASL2018_ASLAD[folder.name], "aslad", "aslad"


def arsl21l(root):
    """ArSL21L: 14k photos, 50 signers; file name = SIGNER_AGE_SEX_class_n.jpg."""
    for f in sorted((root / "arsl21l" / "data" / "sign_data" / "images").glob("*.jpg")):
        signer, _, _, name, _ = f.stem.split("_")
        yield f, ARSL21L[name], f"arsl21l_{signer}", "arsl21l"


def karsl(root, out, fractions=(0.35, 0.5, 0.65)):
    """KArSL letter videos (Kinect, 1080p, ~20 frames each): a few frames around the middle,
    where the hand shape is held. Only the static letters that match the photo datasets."""
    for signer_dir in sorted((root / "karsl").glob("signer*")):
        for split_dir in sorted(p for p in signer_dir.iterdir() if p.is_dir()):
            for sign_id, letter in KARSL_SIGNS.items():
                if letter not in STATIC_LETTERS or letter in KARSL_EXCLUDED:
                    continue
                dst = out / "karsl_frames" / signer_dir.name / f"{sign_id:04d}"
                dst.mkdir(parents=True, exist_ok=True)
                for video in sorted((split_dir / f"{sign_id:04d}").glob("*_c.mp4")):
                    cap = cv2.VideoCapture(str(video))
                    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    for k, frac in enumerate(fractions):
                        frame_path = dst / f"{split_dir.name}_{video.stem}_{k}.jpg"
                        if not frame_path.exists():
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
                            ok, frame = cap.read()
                            if not ok:
                                continue
                            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        yield frame_path, letter, f"karsl_{signer_dir.name}", "karsl"
                    cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(Path.home() / "datasets" / "arabic"))
    parser.add_argument("--out", default=str(CACHE_DIR))
    parser.add_argument("--aslad-per-class", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [dict(path=str(p), label=l, group=g, source=s) for src in
            (arasl2018(root), aslad(root, args.aslad_per_class), arsl21l(root), karsl(root, out))
            for p, l, g, s in src]
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False, encoding="utf-8")

    signers = sorted(g for g in manifest["group"].unique() if g.startswith("arsl21l_"))
    random.Random(args.seed).shuffle(signers)
    splits = {"val": sorted(signers[:VAL_SIGNERS]), "test": sorted(signers[VAL_SIGNERS:VAL_SIGNERS + TEST_SIGNERS])}
    (out / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    print(manifest.groupby("source").agg(images=("path", "size"), classes=("label", "nunique")).to_string())
    print(f"{manifest['label'].nunique()} letters: {' '.join(sorted(manifest['label'].unique()))}")
    print(f"val signers {splits['val']}\ntest signers {splits['test']}\nsaved {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()

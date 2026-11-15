"""Run MediaPipe HandLandmarker over an image-folder dataset (one sub-folder per class) and cache,
for every image where a hand is found: its 21 image + world landmarks and a square hand crop.

    python -m recognition.extract --dataset C:/Users/itrab/datasets/combine_asl
    python -m recognition.extract --manifest manifest.csv --out CACHE --num-hands 2

--manifest takes a CSV with columns path,label,group[,source] instead of a class-folder dataset
(group = signer, used for the split; source = dataset, used to balance sampling).

Output in --out (default ~/datasets/signwave_cache):
    index.csv       one row per detected hand: crop, label, group, file
    landmarks.npz   image (N,21,3) in pixels, world (N,21,3) in metres, aligned with index.csv
    crops/          224x224 hand crops for the CNN
"""
import argparse
import os
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np
import pandas as pd

from .common import CACHE_DIR, HAND_MODEL, source_group

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_detector = None
_crop_dir = None
_crop_size = None


def raised_hand(result):
    """Index of the hand to keep: with two hands in view (e.g. KArSL signers rest the other
    hand on their lap) the signing hand is the one whose wrist is highest in the image."""
    return min(range(len(result.hand_landmarks)), key=lambda k: result.hand_landmarks[k][0].y)


def _init_worker(model_path, min_conf, crop_dir, crop_size, num_hands=1):
    global _detector, _crop_dir, _crop_size
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE, num_hands=num_hands,
        min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf)
    _detector = vision.HandLandmarker.create_from_options(options)
    _crop_dir, _crop_size = Path(crop_dir), crop_size


def _padded(img, min_side=256, pad_ratio=0.25):
    """Upscale small images and add a border: the palm detector misses hands that fill the frame."""
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        scale = min_side / min(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    pad = int(max(img.shape[:2]) * pad_ratio)
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)


def hand_crop(img, points_px, size, margin=1.6):
    """Square crop centred on the hand's landmarks, `margin` times its larger side."""
    (x0, y0), (x1, y1) = points_px.min(0), points_px.max(0)
    side = max(x1 - x0, y1 - y0) * margin
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    matrix = np.float32([[size / side, 0, size / 2 - cx * size / side],
                         [0, size / side, size / 2 - cy * size / side]])
    return cv2.warpAffine(img, matrix, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def _process(task):
    import mediapipe as mp

    idx, path = task
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    for candidate in (img, _padded(img)):
        rgb = np.ascontiguousarray(cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB))
        result = _detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if result.hand_landmarks:
            h, w = candidate.shape[:2]
            k = raised_hand(result)
            image_lm = np.array([[p.x * w, p.y * h, p.z * w] for p in result.hand_landmarks[k]], np.float32)
            world_lm = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[k]], np.float32)
            crop = hand_crop(candidate, image_lm[:, :2], _crop_size)
            cv2.imwrite(str(_crop_dir / f"{idx:06d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return idx, image_lm, world_lm
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source_arg = parser.add_mutually_exclusive_group(required=True)
    source_arg.add_argument("--dataset", help="folder with one sub-folder per class")
    source_arg.add_argument("--manifest", help="CSV with columns path,label,group[,source]")
    parser.add_argument("--num-hands", type=int, default=1, help="2: keep the raised hand when two are visible")
    parser.add_argument("--out", default=str(CACHE_DIR))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0, help="images per class, 0 = all (quick tests)")
    args = parser.parse_args()

    out = Path(args.out)
    crop_dir = out / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, groups, sources = [], [], [], []
    if args.manifest:
        manifest = pd.read_csv(args.manifest, encoding="utf-8")
        if args.limit:
            manifest = manifest.groupby("label").head(args.limit)
        paths = [Path(p) for p in manifest["path"]]
        labels, groups = list(manifest["label"]), list(manifest["group"])
        sources = list(manifest["source"]) if "source" in manifest else groups
    else:
        for folder in sorted(p for p in Path(args.dataset).iterdir() if p.is_dir()):
            files = sorted(f for f in folder.iterdir() if f.suffix.lower() in VALID_EXTS)
            files = files[:args.limit] if args.limit else files
            paths += files
            labels += [folder.name.upper()] * len(files)
        groups = sources = [source_group(p.name) for p in paths]
    print(f"{len(paths)} images, {len(set(labels))} classes, {args.workers} workers")

    image_lms, world_lms, rows = [], [], []
    start = time.time()
    with Pool(args.workers, initializer=_init_worker,
              initargs=(HAND_MODEL, args.min_confidence, crop_dir, args.crop_size, args.num_hands)) as pool:
        tasks = [(i, str(p)) for i, p in enumerate(paths)]
        for n, result in enumerate(pool.imap(_process, tasks, chunksize=32), 1):
            if result is not None:
                idx, image_lm, world_lm = result
                image_lms.append(image_lm)
                world_lms.append(world_lm)
                rows.append({"crop": f"crops/{idx:06d}.jpg", "label": labels[idx], "group": groups[idx],
                             "source": sources[idx], "file": f"{paths[idx].parent.name}/{paths[idx].name}"})
            if n % 5000 == 0 or n == len(tasks):
                rate = n / (time.time() - start)
                print(f"  {n}/{len(tasks)}  detected {len(rows)} ({len(rows) / n:.1%})  "
                      f"{rate:.0f} img/s  eta {(len(tasks) - n) / rate / 60:.1f} min", flush=True)

    index = pd.DataFrame(rows)
    index.to_csv(out / "index.csv", index=False, encoding="utf-8")
    np.savez_compressed(out / "landmarks.npz", image=np.stack(image_lms), world=np.stack(world_lms))

    totals = Counter(sources)
    print("\ndetection rate by source:")
    for group, total in sorted(totals.items()):
        found = int((index["source"] == group).sum())
        print(f"  {group:10s} {found:6d}/{total:<6d} {found / total:.1%}")
    rate_by_class = index["label"].value_counts() / pd.Series(Counter(labels))
    print("lowest detection rates by class:", rate_by_class.sort_values().head(6).round(3).to_dict())
    print(f"saved {len(index)} hands to {out} in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()

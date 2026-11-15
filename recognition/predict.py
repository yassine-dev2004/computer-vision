"""Static sign recognizer: MediaPipe hand landmarks -> landmark MLP + CNN on the hand crop -> averaged.

    python -m recognition.predict path/to/image.jpg [more images...]

The same SignRecognizer class is used by the webcam app (recognition.live).
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np
import torch
from PIL import Image

from .common import CKPT_DIR, HAND_MODEL
from .extract import _padded, hand_crop, raised_hand
from .features import LandmarkMLP, landmark_features
from .train_cnn import build_model, eval_transform

MODES = ("letters", "digits", "all")


class SignRecognizer:
    """Loads both trained models and the MediaPipe hand detector.

    video=True uses MediaPipe's tracking mode (webcam): frames must come with increasing timestamps.
    use_cnn=False keeps only the landmark model (fast enough for a CPU-only laptop).
    """

    def __init__(self, ckpt_dir=CKPT_DIR, arch="convnext_tiny", video=False, use_cnn=True,
                 min_confidence=0.5, device=None, num_hands=1):
        from mediapipe.tasks.python import BaseOptions, vision

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_dir = Path(ckpt_dir)

        lm = torch.load(ckpt_dir / "landmark_mlp.pt", map_location=self.device)
        self.classes = list(lm["classes"])
        self.mlp = LandmarkMLP(**lm["config"]).to(self.device).eval()
        self.mlp.load_state_dict(lm["model"])

        self.cnn = None
        if use_cnn:
            ck = torch.load(ckpt_dir / f"cnn_{arch}.pt", map_location=self.device)
            assert list(ck["classes"]) == self.classes, "the two checkpoints use different classes"
            self.cnn = build_model(ck["arch"], len(self.classes), pretrained=False)
            self.cnn.load_state_dict(ck["model"])  # works for float32 and float16 checkpoints
            self.cnn = self.cnn.to(self.device).eval().to(memory_format=torch.channels_last)
            self.img_size = ck["img_size"]
            self.transform = eval_transform(self.img_size)

        self.video = video
        self._last_ts = -1
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=num_hands, min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence, min_tracking_confidence=min_confidence)
        self.detector = vision.HandLandmarker.create_from_options(options)

        is_digit = np.array([c.isdigit() for c in self.classes])
        self.masks = {"letters": ~is_digit, "digits": is_digit, "all": np.ones(len(self.classes), bool)}

    # ---- detection -------------------------------------------------------------------------------
    def _detect_once(self, bgr, timestamp_ms):
        import mediapipe as mp

        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            ts = max(int(timestamp_ms), self._last_ts + 1)  # MediaPipe needs strictly increasing timestamps
            self._last_ts = ts
            result = self.detector.detect_for_video(image, ts)
        else:
            result = self.detector.detect(image)
        if not result.hand_landmarks:
            return None
        k = raised_hand(result)  # with num_hands=2: the signing hand, not the one resting below
        h, w = bgr.shape[:2]
        # same units as recognition.extract: image landmarks in pixels, world landmarks in metres
        image_lm = np.array([[p.x * w, p.y * h, p.z * w] for p in result.hand_landmarks[k]], np.float32)
        world_lm = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[k]], np.float32)
        return image_lm, world_lm

    def detect(self, bgr, timestamp_ms=0):
        """-> (frame the landmarks refer to, image_lm (21,3) px, world_lm (21,3) m) or None."""
        found = self._detect_once(bgr, timestamp_ms)
        if found is not None:
            return (bgr, *found)
        if not self.video:  # still photos: retry with a border, as during extraction
            padded = _padded(bgr)
            found = self._detect_once(padded, timestamp_ms)
            if found is not None:
                return (padded, *found)
        return None

    # ---- classification --------------------------------------------------------------------------
    @torch.no_grad()
    def classify(self, bgr, image_lm, world_lm, mode="all"):
        """-> (probabilities over self.classes, the 224x224 crop given to the CNN or None)."""
        feats = landmark_features(torch.from_numpy(image_lm)[None].to(self.device),
                                  torch.from_numpy(world_lm)[None].to(self.device))
        probs = self.mlp(feats).softmax(-1)[0]
        crop = None
        if self.cnn is not None:
            crop = hand_crop(bgr, image_lm[:, :2], self.img_size)
            x = self.transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))[None]
            x = torch.cat([x, x.flip(-1)]).to(self.device).to(memory_format=torch.channels_last)  # flip TTA
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                cnn_probs = self.cnn(x).float().softmax(-1).mean(0)
            probs = 0.5 * probs + 0.5 * cnn_probs  # same fixed average as recognition.evaluate
        probs = probs.float().cpu().numpy()
        probs = probs * self.masks[mode]  # letters-only / digits-only: removes 0/O, 2/V, 1/Z... clashes
        return probs / max(probs.sum(), 1e-9), crop

    def __call__(self, bgr, timestamp_ms=0, mode="all"):
        """Full pipeline on one BGR frame -> dict(probs, landmarks, crop) or None if no hand."""
        found = self.detect(bgr, timestamp_ms)
        if found is None:
            return None
        frame, image_lm, world_lm = found
        probs, crop = self.classify(frame, image_lm, world_lm, mode)
        return {"probs": probs, "image_lm": image_lm, "frame": frame, "crop": crop}

    def top(self, probs, k=3):
        order = np.argsort(probs)[::-1][:k]
        return [(self.classes[i], float(probs[i])) for i in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--mode", choices=MODES, default="all")
    parser.add_argument("--no-cnn", action="store_true", help="landmark model only")
    parser.add_argument("--ckpt", default=str(CKPT_DIR))
    args = parser.parse_args()

    recognizer = SignRecognizer(args.ckpt, use_cnn=not args.no_cnn)
    for path in args.images:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"{path}: cannot read")
            continue
        out = recognizer(img, mode=args.mode)
        if out is None:
            print(f"{path}: no hand found")
        else:
            print(f"{path}: " + "  ".join(f"{c} {p:.0%}" for c, p in recognizer.top(out["probs"])))


if __name__ == "__main__":
    main()

"""SignWaveAI live demo: fingerspell in front of the webcam, the letters are typed on screen.

    python -m recognition.live                 # letters mode, both models
    python -m recognition.live --no-cnn        # landmark model only (CPU-only laptops)
    python -m recognition.live --camera 1      # another webcam

Hold a sign steady for --hold seconds to type it. Lower your hand to end a word (a space is added
after --auto-space seconds without a hand). To type the same letter twice, drop the hand briefly.

Keys:  M switch letters / digits / all    SPACE space    BACKSPACE delete    C clear
       ENTER send the words to the grammar model (if trained)    Q or ESC quit
"""
import argparse
import sys
import time

import cv2
import numpy as np

from .common import CKPT_DIR, PROJECT_DIR
from .predict import MODES, SignRecognizer

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
ACCENT = (60, 76, 231)      # BGR red, close to the report's colour
INK, MUTED, PANEL = (40, 40, 40), (120, 120, 120), (245, 242, 239)
FONT = cv2.FONT_HERSHEY_DUPLEX


class Speller:
    """Turns a stream of (label, confidence) into typed text: a label is typed once it has been the
    confident top prediction for `hold` seconds, and is not repeated until the sign changes."""

    def __init__(self, hold=0.8, threshold=0.6, auto_space=1.5):
        self.hold, self.threshold, self.auto_space = hold, threshold, auto_space
        self.text = ""
        self.candidate, self.since, self.typed, self.last_hand = None, 0.0, None, None

    def update(self, label, confidence, now):
        """label=None when no hand is visible. Returns the progress (0..1) of the current hold."""
        if label is None:
            self.candidate, self.typed = None, None
            if (self.auto_space and self.last_hand is not None and now - self.last_hand > self.auto_space
                    and self.text and not self.text.endswith(" ")):
                self.text += " "
            return 0.0
        self.last_hand = now
        if confidence < self.threshold:
            self.candidate = None
            return 0.0
        if label != self.candidate:
            self.candidate, self.since = label, now
        if label == self.typed:
            return 1.0
        progress = (now - self.since) / self.hold
        if progress >= 1.0:
            self.text += label
            self.typed = label
        return min(progress, 1.0)

    def space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def backspace(self):
        self.text = self.text[:-1]
        self.typed = None

    def clear(self):
        self.text, self.typed = "", None


def load_translator():
    """Grammar model (gloss -> English) if it has been trained, else None."""
    ckpt = PROJECT_DIR / "checkpoints" / "grammar"
    if not (ckpt / "model.pt").exists():
        return None
    sys.path.insert(0, str(PROJECT_DIR))
    from grammar.translate import GlossToEnglish
    return GlossToEnglish(str(ckpt))


def words_to_glosses(text):
    """Fingerspelled words become fs- glosses (copied through by the translator), numbers stay as they are."""
    return [w if w.isdigit() else f"fs-{w}" for w in text.split()]


def draw_hand(img, points_xy):
    pts = [tuple(int(v) for v in p) for p in points_xy]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(img, pts[a], pts[b], ACCENT, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 4, ACCENT, -1, cv2.LINE_AA)


def render(frame, out, recognizer, speller, progress, mode, fps, translation, panel_w=300, bar_h=130):
    """Mirrored camera view + side panel (top-3, hold progress) + text bar."""
    h, w = frame.shape[:2]
    view = cv2.flip(frame, 1)  # mirror: moving your right hand moves the right side of the picture
    if out is not None:
        pts = out["image_lm"][:, :2].copy()
        pts[:, 0] = w - pts[:, 0]
        draw_hand(view, pts)

    canvas = np.full((h + bar_h, w + panel_w, 3), PANEL, np.uint8)
    canvas[:h, :w] = view
    x0 = w + 20

    cv2.putText(canvas, "SignWaveAI", (x0, 40), FONT, 0.9, ACCENT, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"mode: {mode}  (M)", (x0, 70), FONT, 0.5, MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{fps:4.1f} fps", (x0, 92), FONT, 0.5, MUTED, 1, cv2.LINE_AA)

    if out is None:
        cv2.putText(canvas, "show your hand", (x0, 150), FONT, 0.7, MUTED, 1, cv2.LINE_AA)
    else:
        top = recognizer.top(out["probs"], 3)
        label, conf = top[0]
        # big letter with the hold progress as a ring around it
        cx, cy = x0 + 70, 190
        cv2.circle(canvas, (cx, cy), 58, (220, 215, 210), 6, cv2.LINE_AA)
        if progress > 0:
            cv2.ellipse(canvas, (cx, cy), (58, 58), -90, 0, 360 * progress, ACCENT, 6, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, FONT, 2.2, 3)
        cv2.putText(canvas, label, (cx - tw // 2, cy + th // 2), FONT, 2.2, INK, 3, cv2.LINE_AA)
        for i, (c, p) in enumerate(top):
            y = 290 + i * 34
            cv2.putText(canvas, c, (x0, y + 14), FONT, 0.6, INK, 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (x0 + 30, y), (x0 + 30 + int(180 * p), y + 18),
                          ACCENT if i == 0 else (190, 180, 175), -1)
            cv2.putText(canvas, f"{p:.0%}", (x0 + 220, y + 14), FONT, 0.5, MUTED, 1, cv2.LINE_AA)
        if out["crop"] is not None:  # what the CNN sees
            crop = cv2.resize(out["crop"], (112, 112))
            y = min(h - 122, 400)
            canvas[y:y + 112, x0:x0 + 112] = crop
            cv2.putText(canvas, "CNN input", (x0 + 122, y + 60), FONT, 0.45, MUTED, 1, cv2.LINE_AA)

    # text bar
    cv2.line(canvas, (0, h), (w + panel_w, h), ACCENT, 2)
    text = speller.text[-48:] + ("|" if int(time.time() * 2) % 2 else "")
    cv2.putText(canvas, text, (20, h + 55), FONT, 1.3, INK, 2, cv2.LINE_AA)
    sub = translation or "SPACE space   BACKSPACE delete   C clear   ENTER translate   Q quit"
    cv2.putText(canvas, sub[:110], (20, h + 100), FONT, 0.55, ACCENT if translation else MUTED, 1, cv2.LINE_AA)
    return canvas


def open_camera(index, width, height):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    raise SystemExit(f"cannot open camera {index}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--mode", choices=MODES, default="letters")
    parser.add_argument("--no-cnn", action="store_true", help="landmark model only (faster on CPU)")
    parser.add_argument("--hold", type=float, default=0.8, help="seconds a sign must be held to be typed")
    parser.add_argument("--threshold", type=float, default=0.6, help="minimum smoothed confidence to type")
    parser.add_argument("--auto-space", type=float, default=1.5, help="seconds without a hand -> space (0 = off)")
    parser.add_argument("--smoothing", type=float, default=0.6, help="weight of the past in the running average")
    parser.add_argument("--ckpt", default=str(CKPT_DIR))
    parser.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)  # smoke tests
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    print("loading models...")
    recognizer = SignRecognizer(args.ckpt, video=True, use_cnn=not args.no_cnn)
    translator = load_translator()
    print(f"models on {recognizer.device} | CNN {'off' if args.no_cnn else 'on'} | "
          f"grammar model {'loaded' if translator else 'not trained yet'}")
    cap = open_camera(args.camera, args.width, args.height)

    speller = Speller(args.hold, args.threshold, args.auto_space)
    mode, smoothed, translation = args.mode, None, ""
    fps, last, frames = 0.0, time.perf_counter(), 0
    window = "SignWaveAI"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera stopped sending frames")
                break
            now = time.perf_counter()
            out = recognizer(frame, timestamp_ms=now * 1000, mode=mode)

            if out is None:
                smoothed = None
                progress = speller.update(None, 0.0, now)
            else:
                # running average over frames: one blurry frame does not flip the letter
                smoothed = out["probs"] if smoothed is None else args.smoothing * smoothed + (1 - args.smoothing) * out["probs"]
                out["probs"] = smoothed
                label, conf = recognizer.top(smoothed, 1)[0]
                progress = speller.update(label, conf, now)

            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            frames += 1
            if args.max_frames and frames >= args.max_frames:
                print(f"{frames} frames, {fps:.1f} fps, last prediction: "
                      f"{recognizer.top(out['probs']) if out else 'no hand'}")
                break
            if args.headless:
                continue

            cv2.imshow(window, render(frame, out, recognizer, speller, progress, mode, fps, translation))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("m"):
                mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
                smoothed = None
            elif key == 32:
                speller.space()
            elif key == 8:
                speller.backspace()
            elif key == ord("c"):
                speller.clear()
                translation = ""
            elif key == 13:
                if not speller.text.strip():
                    continue
                if translator is None:
                    translation = "grammar model not trained yet: " + speller.text.strip()
                else:
                    translation = translator(words_to_glosses(speller.text))
                print(f"{speller.text.strip()}  ->  {translation}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if speller.text.strip():
        print("typed:", speller.text.strip())


if __name__ == "__main__":
    main()

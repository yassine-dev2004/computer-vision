"""Quran revision with the Arabic finger alphabet: spell a surah in front of the webcam, letter by
letter; the app follows the verse and points out wrong, skipped and extra letters.

    python -m arabic.live --surah 1              # Al-Fatihah from the first verse
    python -m arabic.live --surah 112 --verse 2  # Al-Ikhlas from verse 2
    python -m arabic.live --surah 1 --no-cnn     # landmark model only (CPU-only laptops)

Hold each letter for --hold seconds. Words turn green when correct, red when a letter is wrong or
missing, amber when the recognizer was unsure (repeat the letter). A verse is checked once you are
two letters into the next one.

Keys:  BACKSPACE undo the last letter    R restart the verse    N / P next / previous verse
       H show the next expected letter (off by default: this is a revision aid)    Q or ESC quit
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from recognition.common import PROJECT_DIR
from recognition.live import Speller, draw_hand, open_camera
from recognition.predict import SignRecognizer

from .quran import QURAN_DIR, Quran
from .session import Session

CKPT_DIR = PROJECT_DIR / "checkpoints" / "arabic"
FONT_FILES = ["trado.ttf", "arial.ttf", "tahoma.ttf", "segoeui.ttf", "DejaVuSans.ttf", "NotoNaskhArabic-Regular.ttf"]
UI_FONT_FILES = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"]
FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/truetype/noto"),
             Path("/Library/Fonts"), Path("/System/Library/Fonts/Supplemental")]

# RGB (PIL draws in RGB, the canvas is converted from BGR)
COLORS = {"ok": (46, 139, 87), "wrong": (200, 40, 40), "missing": (200, 40, 40), "unclear": (214, 140, 0),
          "current": (30, 30, 30), "pending": (150, 150, 150), "skipped": (150, 150, 150),
          "not recited": (190, 190, 190), "extra": (200, 40, 40)}
INK, MUTED, ACCENT, PANEL = (30, 30, 30), (120, 120, 120), (231, 76, 60), (239, 242, 245)


def find_font(names=FONT_FILES):
    for name in names:
        for folder in FONT_DIRS:
            if (folder / name).exists():
                return str(folder / name)
    raise SystemExit("no suitable font found (looked for " + ", ".join(names) + ")")


class Arabic:
    """Draws Arabic text with PIL: letters shaped (joined forms) and laid out right to left."""

    def __init__(self, font_path):
        import arabic_reshaper
        from bidi import get_display
        self._shape = lambda t: get_display(arabic_reshaper.reshape(t))
        self.path, self.fonts = font_path, {}

    def font(self, size):
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(self.path, size)
        return self.fonts[size]

    def width(self, text, size):
        return self.font(size).getlength(self._shape(text))

    def draw(self, draw, xy, text, size, fill, anchor="ra"):
        draw.text(xy, self._shape(text), font=self.font(size), fill=fill, anchor=anchor)


def latin(draw, xy, text, size, fill, font_path, anchor="la"):
    draw.text(xy, text, font=ImageFont.truetype(font_path, size), fill=fill, anchor=anchor)


def mixed(draw, xy, parts, size, fill, ar, ui_font):
    """A left-to-right sentence mixing English and Arabic pieces: [("in ", False), ("كلمة", True)]."""
    x, y = xy
    latin_font = ImageFont.truetype(ui_font, size)
    for text, is_arabic in parts:
        if is_arabic:
            ar.draw(draw, (x, y + size // 2), text, int(size * 1.5), fill, anchor="lm")
            x += ar.width(text, int(size * 1.5))
        else:
            draw.text((x, y + size // 2), text, font=latin_font, fill=fill, anchor="lm")
            x += latin_font.getlength(text)


def wrap_rtl(words, widths, max_width, gap):
    """Lines of word indices, each line filled right to left."""
    lines, line, used = [], [], 0
    for i, w in enumerate(widths):
        if line and used + gap + w > max_width:
            lines.append(line)
            line, used = [], 0
        used += (gap if line else 0) + w
        line.append(i)
    return lines + ([line] if line else [])


def render(frame, out, recognizer, session, progress, ar, ui_font, fps, hint, panel_w=320, bar_h=230):
    h, w = frame.shape[:2]
    view = cv2.flip(frame, 1)
    if out is not None:
        pts = out["image_lm"][:, :2].copy()
        pts[:, 0] = w - pts[:, 0]
        draw_hand(view, pts)
    canvas = np.full((h + bar_h, w + panel_w, 3), PANEL[::-1], np.uint8)
    canvas[:h, :w] = view
    # progress ring drawn with OpenCV (anti-aliased arcs), text with PIL
    x0, cx, cy = w + 24, w + panel_w // 2, 215
    cv2.circle(canvas, (cx, cy), 62, (215, 210, 205), 6, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(canvas, (cx, cy), (62, 62), -90, 0, 360 * progress, ACCENT[::-1], 6, cv2.LINE_AA)
    cv2.line(canvas, (0, h), (w + panel_w, h), ACCENT[::-1], 2)

    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    v = session.view()
    latin(d, (x0, 18), "SignWave · Quran", 26, ACCENT, ui_font)
    latin(d, (x0, 54), v["surah"].split(" (")[0], 18, INK, ui_font)
    ar.draw(d, (w + panel_w - 24, 50), v["surah"].split("(")[-1].rstrip(")"), 30, INK)
    latin(d, (x0, 84), f"verse {v['key']}   ·   {v['done_verses']}/{v['total_verses']} checked   ·   {fps:4.1f} fps",
          14, MUTED, ui_font)

    if out is None:
        latin(d, (cx, cy), "show your hand", 18, MUTED, ui_font, anchor="mm")
    else:
        top = recognizer.top(out["probs"], 3)
        ar.draw(d, (cx, cy + 4), top[0][0], 64, INK, anchor="mm")
        for i, (letter, p) in enumerate(top):
            y = 300 + i * 36
            ar.draw(d, (x0 + 28, y + 12), letter, 26, INK, anchor="rm")
            d.rectangle([x0 + 40, y + 2, x0 + 40 + int(190 * p), y + 22], fill=ACCENT if i == 0 else (205, 195, 190))
            latin(d, (x0 + 240, y + 12), f"{p:.0%}", 14, MUTED, ui_font, anchor="lm")
    if hint and not v["finished"]:
        nxt = next((l.letter for l in v["current_letters"] if l.status == "pending"), None)
        if nxt:
            latin(d, (x0, 420), "next letter:", 16, MUTED, ui_font)
            ar.draw(d, (x0 + 150, 432), nxt, 36, ACCENT, anchor="lm")

    # verse, word by word, right to left; the lines around the current word
    size, gap, right = 40, 22, w + panel_w - 30
    widths = [ar.width(word, size) for word in v["words"]]
    lines = wrap_rtl(v["words"], widths, w + panel_w - 60, gap)
    current_line = next(i for i, line in enumerate(lines) if v["current_word"] in line)
    first = max(0, min(current_line - 1, len(lines) - 2))
    for row, line in enumerate(lines[first:first + 2]):
        x, y = right, h + 16 + row * 56
        for i in line:
            status = v["word_status"][i]
            ar.draw(d, (x, y), v["words"][i], size, COLORS.get(status, INK))
            if i == v["current_word"] and not v["finished"]:
                d.line([x - widths[i], y + size + 8, x, y + size + 8], fill=ACCENT, width=3)
            x -= widths[i] + gap

    # current word letter by letter, and what went wrong
    y = h + 140
    x = right
    for letter in v["current_letters"]:
        color = COLORS.get(letter.status, INK)
        ar.draw(d, (x, y), letter.letter, 34, color)
        if letter.status in ("wrong", "unclear") and letter.signed:
            ar.draw(d, (x - 6, y + 40), letter.signed, 18, color)
        x -= ar.width(letter.letter, 34) + 18
    if v["mistakes"]:
        word, l = v["mistakes"][-1]
        latin(d, (24, h + 136), f"{len(v['mistakes'])} mistake(s) in verse {v['key']} — last one:", 15,
              COLORS["wrong"], ui_font)
        if l.status == "wrong":
            parts = [("in ", False), (word, True), ("  signed ", False), (l.signed, True), ("  instead of ", False),
                     (l.letter, True)]
        else:
            parts = [("in ", False), (word, True), ("  the letter ", False), (l.letter, True), ("  was skipped", False)]
        mixed(d, (24, h + 166), parts, 16, COLORS["wrong"], ar, ui_font)
    elif v["message"]:
        latin(d, (24, h + 160), v["message"], 16, ACCENT, ui_font)
    latin(d, (24, h + bar_h - 24), "BACKSPACE undo   R restart verse   N/P next/prev verse   H hint   Q quit",
          13, MUTED, ui_font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--surah", type=int, default=1)
    parser.add_argument("--verse", type=int, default=1)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", help="read a video file instead of the webcam")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--no-cnn", action="store_true", help="landmark model only (faster on CPU)")
    parser.add_argument("--hold", type=float, default=0.7, help="seconds a letter must be held")
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum smoothed confidence to take a letter")
    parser.add_argument("--smoothing", type=float, default=0.6)
    parser.add_argument("--ckpt", default=str(CKPT_DIR))
    parser.add_argument("--quran", default=str(QURAN_DIR))
    parser.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--save-frame", help=argparse.SUPPRESS)
    args = parser.parse_args()

    ar, ui_font = Arabic(find_font()), find_font(UI_FONT_FILES)
    quran = Quran(args.quran)
    session = Session(quran, args.surah, args.verse)
    print("loading models...")
    recognizer = SignRecognizer(args.ckpt, video=True, use_cnn=not args.no_cnn, num_hands=2)
    classes = recognizer.classes
    print(f"{quran.name(args.surah)} from verse {args.verse} | models on {recognizer.device}")
    cap = cv2.VideoCapture(args.video) if args.video else open_camera(args.camera, args.width, args.height)

    speller = Speller(args.hold, args.threshold, auto_space=0)
    smoothed, held, hint = None, [], False
    fps, last, frames, window = 0.0, time.perf_counter(), 0, "SignWave Quran"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.video:  # keep the window a sensible size for 1080p files
                scale = args.height / frame.shape[0]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            now = time.perf_counter() if not args.video else frames / 30.0
            out = recognizer(frame, timestamp_ms=now * 1000, mode="all")
            if out is None:
                smoothed, held = None, []
                progress = speller.update(None, 0.0, now)
            else:
                smoothed = out["probs"] if smoothed is None else args.smoothing * smoothed + (1 - args.smoothing) * out["probs"]
                out["probs"] = smoothed
                label, conf = recognizer.top(smoothed, 1)[0]
                if label != speller.candidate:
                    held = []
                held.append(smoothed)
                before = len(speller.text)
                progress = speller.update(label, conf, now)
                if len(speller.text) > before:  # a letter was just taken: give the corrector its probabilities
                    session.add_sign(np.mean(held, axis=0), classes)
                    speller.text = speller.text[-50:]

            fps = 0.9 * fps + 0.1 / max(time.perf_counter() - last, 1e-6)
            last = time.perf_counter()
            frames += 1
            canvas = render(frame, out, recognizer, session, progress, ar, ui_font, fps, hint)
            if args.save_frame and (frames == args.max_frames or not args.max_frames):
                cv2.imwrite(args.save_frame, canvas)
            if args.max_frames and frames >= args.max_frames:
                break
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 8:
                session.undo()
            elif key == ord("r"):
                session.restart_verse()
            elif key == ord("n"):
                session.goto(session.verse + 1)
            elif key == ord("p"):
                session.goto(session.verse - 1)
            elif key == ord("h"):
                hint = not hint
    finally:
        cap.release()
        cv2.destroyAllWindows()
    for key, letters in session.results.items():
        bad = [l for l in letters if l.status in ("wrong", "missing")]
        print(f"{key}: {len(letters) - len(bad)}/{len(letters)} letters correct" + (f", mistakes: {[(l.letter, l.signed) for l in bad]}" if bad else ""))


if __name__ == "__main__":
    main()

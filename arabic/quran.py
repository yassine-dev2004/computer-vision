"""Quran text as the sequence of letters a signer spells (kitabah / finger spelling).

    python -m arabic.quran --download            # fetch the text from api.quran.com (once)
    python -m arabic.quran 1                     # show Al-Fatihah as expected letters

The text is the standard (imla'i) spelling. Diacritics and recitation marks are not spelled, and the
hamza seats the recognizer cannot tell apart yet are accepted as their base letter (see ACCEPT).
"""
import argparse
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .letters import STATIC_LETTERS

QURAN_DIR = Path.home() / "datasets" / "arabic" / "quran"
API = "https://api.quran.com/api/v4"

# harakat, shadda, sukun, tanwin, dagger alif, Quranic annotation and pause marks, tatweel
_MARKS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")

# letter as written -> recognizer classes accepted for it (phase 1: no hamza signs yet)
ACCEPT = {"أ": {"ا"}, "إ": {"ا"}, "آ": {"ا"}, "ٱ": {"ا"}, "ؤ": {"و"}, "ى": {"ي"}, "ئ": {"ئ", "ي"}}
OPTIONAL = {"ء"}  # lone hamza: no sign learned yet, so skipping it is never an error


@dataclass(frozen=True)
class Slot:
    """One letter the signer is expected to spell."""
    letter: str        # as written in the text, e.g. "أ"
    accept: frozenset  # recognizer classes that count as this letter
    optional: bool
    verse: str         # "1:2"
    word: int          # word index inside the verse


def strip_marks(text):
    return _MARKS.sub("", text)


def verse_slots(verse_key, text):
    slots = []
    for w, word in enumerate(strip_marks(text).split()):
        for ch in word:
            if ch in OPTIONAL:
                slots.append(Slot(ch, frozenset(), True, verse_key, w))
            else:
                accept = ACCEPT.get(ch, {ch})
                assert accept <= set(STATIC_LETTERS), f"no sign for {ch!r} in {verse_key}"
                slots.append(Slot(ch, frozenset(accept), False, verse_key, w))
    return slots


class Quran:
    def __init__(self, folder=QURAN_DIR):
        folder = Path(folder)
        verses = json.loads((folder / "imlaei.json").read_text(encoding="utf-8"))["verses"]
        chapters = json.loads((folder / "chapters.json").read_text(encoding="utf-8"))["chapters"]
        self.text = {v["verse_key"]: v["text_imlaei"] for v in verses}
        self.chapters = {c["id"]: c for c in chapters}

    def verses(self, surah, start=1, end=None):
        """Verse keys of a surah, from verse `start` to `end` (inclusive)."""
        end = end or self.chapters[surah]["verses_count"]
        return [f"{surah}:{v}" for v in range(start, end + 1)]

    def slots(self, verse_keys):
        return [s for key in verse_keys for s in verse_slots(key, self.text[key])]

    def words(self, verse_key):
        return strip_marks(self.text[verse_key]).split()

    def name(self, surah):
        c = self.chapters[surah]
        return f"{c['name_simple']} ({c['name_arabic']})"


def download(folder=QURAN_DIR):
    """Text from api.quran.com: imla'i (standard spelling) and Uthmani script, plus surah names."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "signwave-quran/1.0"}
    for name, url in [("imlaei.json", f"{API}/quran/verses/imlaei"), ("uthmani.json", f"{API}/quran/verses/uthmani"),
                      ("chapters.json", f"{API}/chapters")]:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
            (folder / name).write_bytes(r.read())
        print("saved", folder / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("surah", nargs="?", type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--folder", default=str(QURAN_DIR))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.download:
        download(args.folder)
    if args.surah:
        q = Quran(args.folder)
        print(q.name(args.surah))
        for key in q.verses(args.surah):
            slots = verse_slots(key, q.text[key])
            print(f"{key:>7}  {len(slots):3d} letters  {' '.join(s.letter for s in slots)}")


if __name__ == "__main__":
    main()

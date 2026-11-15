"""A revision session: the signer spells a surah letter by letter, the session follows along.

    s = Session(Quran(), surah=1)
    s.add_sign(probs, classes)      # every time the hold detector types a letter
    s.view()                        # what to show: current verse, word and letter statuses, message

The corrector only looks at the current verse and the next one, so long surahs stay fast. A verse
is frozen once the signer is two letters into the next one. If the signs stop fitting, the session
looks for them elsewhere in the surah and moves there when the match is unambiguous.
"""
from dataclasses import dataclass

from .corrector import Corrector, locate, tokens_from_probs

BAD = ("wrong", "missing")


@dataclass
class LetterView:
    letter: str
    status: str
    signed: str = ""


def word_status(statuses):
    statuses = [s for s in statuses if s != "not recited"]  # before the point where the signer started
    if not statuses:
        return "not recited"
    if any(s in BAD for s in statuses):
        return "wrong"
    if any(s == "unclear" for s in statuses):
        return "unclear"
    if all(s in ("ok", "skipped") for s in statuses):
        return "ok"
    if any(s in ("ok", "skipped") for s in statuses):
        return "current"
    return "pending"


class Session:
    def __init__(self, quran, surah, verse=1, **corrector_options):
        self.q, self.surah, self.options = quran, surah, corrector_options
        self.keys = quran.verses(surah)
        self.all_slots = quran.slots(self.keys)
        self.results = {}   # verse key -> [LetterView] for finished verses
        self.message = ""
        self._open(verse)

    # ---- window ---------------------------------------------------------------------------------
    def _open(self, verse, offset=0, signs=()):
        """Follow from letter `offset` of `verse` (1-based), replaying `signs`."""
        self.verse, self.offset = verse, offset
        keys = self.keys[verse - 1:verse + 1]
        self.current_slots = self.q.slots(keys[:1])
        self.corrector = Corrector(self.q.slots(keys)[offset:], **self.options)
        self.n_current = len(self.current_slots) - offset
        self.signs, self.token_sign = [], []
        for probs, classes in signs:
            self._feed(probs, classes)

    def _feed(self, probs, classes):
        self.signs.append((probs, classes))
        for tok in tokens_from_probs(probs, classes):
            self.corrector.add(tok)
            self.token_sign.append(len(self.signs) - 1)

    @property
    def key(self):
        return self.keys[self.verse - 1]

    @property
    def finished(self):
        return self.key in self.results

    def add_sign(self, probs, classes):
        if self.finished:
            return
        self._feed(probs, classes)
        self.message = ""
        if self.corrector.lost() and self._relocate():
            return
        self._advance()

    def undo(self):
        if self.signs:
            signs = self.signs[:-1]
            self._open(self.verse, self.offset, signs)

    def restart_verse(self):
        self._open(self.verse)

    def goto(self, verse):
        self.results.pop(self.keys[verse - 1], None)
        self._open(max(1, min(verse, len(self.keys))))

    def _advance(self):
        c = self.corrector
        last_verse = self.verse == len(self.keys)
        done = c.position() >= self.n_current + (0 if last_verse else 2)
        if not done:
            return
        marks = c.marks()
        self.results[self.key] = self._letters(marks)
        if last_verse:
            self.message = "end of the surah"
            return
        later = [m.token for m in marks if m.slot >= self.n_current and m.token >= 0]
        first_sign = self.token_sign[min(later)] if later else len(self.signs)
        self._open(self.verse + 1, 0, self.signs[first_sign:])

    def _relocate(self):
        """The last signs do not fit here: find them in the surah, move if the match is clear."""
        tail = [t.signed for t in self.corrector.tokens[-5:]]
        text = [s.letter if len(s.accept) != 1 else next(iter(s.accept)) for s in self.all_slots]
        found = locate(tail, text, top=2)
        if not found or found[0][1] > 0 or (len(found) > 1 and found[1][1] == 0):
            return False  # no exact match, or several: keep following here, the errors stand
        end = found[0][0]
        start = end - len(tail) + 1
        tokens, n_tail = self.corrector.tokens, len(tail)
        while start > 0 and n_tail < len(tokens) and tokens[-n_tail - 1].signed == text[start - 1]:
            start, n_tail = start - 1, n_tail + 1  # earlier signs that also fit: the word's beginning
        slot = self.all_slots[start]
        verse = int(slot.verse.split(":")[1])
        offset = start - next(i for i, s in enumerate(self.all_slots) if s.verse == slot.verse)
        if verse == self.verse and offset == self.offset:
            return False
        signs = self.signs[self.token_sign[len(tokens) - n_tail]:]
        self._open(verse, offset, signs)
        self.message = f"moved to verse {slot.verse}"
        return True

    # ---- display --------------------------------------------------------------------------------
    def _letters(self, marks):
        """Letters of the current verse with their status (letters before `offset` were not recited)."""
        out = [LetterView(s.letter, "not recited") for s in self.current_slots[:self.offset]]
        by_slot = {m.slot: m for m in marks if m.slot >= 0}
        for j, s in enumerate(self.current_slots[self.offset:]):
            m = by_slot.get(j)
            out.append(LetterView(s.letter, m.status if m else "pending", m.signed if m else ""))
        return out

    def view(self):
        """Everything the screen needs about the current verse."""
        if self.finished:
            letters = self.results[self.key]
        else:
            letters = self._letters(self.corrector.marks())
        words = self.q.words(self.key)
        statuses, k = [], 0
        for w, word in enumerate(words):
            n = sum(1 for s in self.current_slots if s.word == w)
            statuses.append(word_status([l.status for l in letters[k:k + n]]))
            k += n
        current = next((w for w, st in enumerate(statuses) if st in ("current", "pending")), len(words) - 1)
        start = sum(1 for s in self.current_slots if s.word < current)
        n = sum(1 for s in self.current_slots if s.word == current)
        extras = [m.signed for m in self.corrector.marks() if m.status == "extra"] if not self.finished else []
        mistakes = [(words[self.current_slots[j].word], l) for j, l in enumerate(letters) if l.status in BAD]
        return {"key": self.key, "surah": self.q.name(self.surah), "words": words, "word_status": statuses,
                "current_word": current, "current_letters": letters[start:start + n], "mistakes": mistakes,
                "extras": extras, "message": self.message, "finished": self.finished,
                "done_verses": len(self.results), "total_verses": len(self.keys)}

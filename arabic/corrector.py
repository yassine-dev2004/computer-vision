"""Follow a signer through a known passage and point out mistakes.

The recognizer gives, for every letter the signer holds, a probability for each of the 32 hand
shapes. The corrector aligns that stream with the expected letters (Needleman-Wunsch with
probability-based costs, updated one row per new sign) and labels every expected letter:

    ok        signed as expected
    wrong     another letter was signed clearly, and the expected one is very unlikely
    unclear   the recognizer is not sure: ask to repeat instead of claiming a mistake
    missing   skipped (letters after it were signed)
    pending   not reached yet

Signs that are not in the text are reported as `extra` only when they were clear.
A wrong correction is worse than a missed one in a revision aid, hence the thresholds.
"""
from dataclasses import dataclass, field

import numpy as np

LIGATURES = {"ال": ("ا", "ل"), "لا": ("ل", "ا")}


@dataclass
class Token:
    """One signed letter: the recognizer's probabilities, folded onto single letters."""
    probs: dict          # letter -> probability
    signed: str          # top letter
    confidence: float    # its probability


def tokens_from_probs(probs, classes):
    """A held sign -> one token, or two for the ligature signs ال and لا."""
    p = dict(zip(classes, map(float, probs)))
    top = max(p, key=p.get)
    if top not in LIGATURES:
        return [Token(p, top, p[top])]
    out = []
    for letter in LIGATURES[top]:
        q = dict(p)
        q[letter] = q.get(letter, 0.0) + p[top]  # the ligature sign counts as each of its letters
        out.append(Token(q, letter, p[top]))
    return out


@dataclass
class Mark:
    status: str          # ok / wrong / unclear / missing / pending / extra / skipped
    slot: int = -1       # index in the expected letters (-1 for extra signs)
    token: int = -1      # index in the signed tokens (-1 for missing / pending)
    signed: str = ""


@dataclass
class Corrector:
    slots: list                    # arabic.quran.Slot, the expected letters
    ins_cost: float = 2.5          # a sign that is not in the text
    del_cost: float = 2.5          # an expected letter that was not signed
    double_del_cost: float = 0.7   # second of two identical letters (hard to sign twice)
    max_sub_cost: float = 4.0      # < ins + del: a clear wrong letter is a substitution, not skip + extra
    wrong_below: float = 0.05      # expected letter's probability under which ...
    wrong_confidence: float = 0.6  # ... a confident other letter is a mistake
    ok_above: float = 0.25
    tokens: list = field(default_factory=list)

    def __post_init__(self):
        n = len(self.slots)
        dele = np.empty(n)
        for j, s in enumerate(self.slots):
            prev = self.slots[j - 1] if j else None
            doubled = prev is not None and prev.letter == s.letter and prev.word == s.word
            dele[j] = 0.0 if s.optional else self.double_del_cost if doubled else self.del_cost
        self._del = dele
        self._cum_del = np.concatenate([[0.0], np.cumsum(dele)])  # cost of skipping slots [0, j)
        self._rows = [self._cum_del.copy()]                       # D[0][j]: nothing signed, j slots skipped
        self._back = []

    # ---- alignment ------------------------------------------------------------------------------
    def _sub_costs(self, tok):
        p = np.array([sum(tok.probs.get(a, 0.0) for a in s.accept) for s in self.slots])
        cost = -np.log(np.clip(p, 1e-6, 1.0))
        cost[[s.optional for s in self.slots]] = np.inf  # nothing is signed for an optional letter
        return np.minimum(cost, self.max_sub_cost), p

    def add(self, token):
        """Feed one signed letter (a Token): one new row of the alignment table."""
        self.tokens.append(token)
        prev = self._rows[-1]
        sub, _ = self._sub_costs(token)
        diag = np.concatenate([[np.inf], prev[:-1] + sub])  # sign matches slot j-1
        ins = prev + self.ins_cost                           # sign is not in the text
        x = np.minimum(diag, ins)
        # then skip expected letters: D[j] = min over k<=j of x[k] + cost of skipping slots k..j-1
        v = x - self._cum_del
        running = np.minimum.accumulate(v)
        arg = np.maximum.accumulate(np.where(v == running, np.arange(len(v)), 0))
        skip = running + self._cum_del
        use_skip = skip < x - 1e-9
        self._rows.append(np.where(use_skip, skip, x))
        self._back.append((np.where(diag <= ins, 0, 1), np.where(use_skip, arg, -1)))

    def add_sign(self, probs, classes):
        for tok in tokens_from_probs(probs, classes):
            self.add(tok)

    def position(self):
        """Number of expected letters covered by the best alignment so far."""
        return int(np.argmin(self._rows[-1])) if self.tokens else 0

    def _path(self):
        """Best alignment as a list of ("sub", token, slot) / ("ins", token, None) / ("del", None, slot)."""
        i, j, ops = len(self.tokens), self.position(), []
        while i > 0:
            choice, skip_from = self._back[i - 1]
            if skip_from[j] >= 0:  # slots k..j-1 skipped after the last sign
                k = int(skip_from[j])
                ops += [("del", None, jj) for jj in range(j - 1, k - 1, -1)]
                j = k
            if choice[j] == 0:
                ops.append(("sub", i - 1, j - 1)); j -= 1
            else:
                ops.append(("ins", i - 1, None))
            i -= 1
        ops += [("del", None, jj) for jj in range(j - 1, -1, -1)]  # skipped before the first sign
        return ops[::-1]

    # ---- verdicts -------------------------------------------------------------------------------
    def marks(self):
        """One Mark per expected letter (in order), plus extra signs, from the best alignment."""
        out = []
        for op, i, j in self._path():
            if op == "ins":
                tok = self.tokens[i]
                if tok.confidence >= self.wrong_confidence:
                    out.append(Mark("extra", -1, i, tok.signed))
            elif op == "del":
                s = self.slots[j]
                status = "skipped" if s.optional else "ok" if self._del[j] < self.del_cost else "missing"
                out.append(Mark(status, j))
            else:
                tok, s = self.tokens[i], self.slots[j]
                p = sum(tok.probs.get(a, 0.0) for a in s.accept)
                if tok.signed in s.accept or p >= self.ok_above:
                    status = "ok"
                elif p < self.wrong_below and tok.confidence >= self.wrong_confidence:
                    status = "wrong"
                else:
                    status = "unclear"
                out.append(Mark(status, j, i, tok.signed))
        out += [Mark("pending", j) for j in range(self.position(), len(self.slots))]
        return out

    def lost(self, last=4):
        """True when most of the last signs do not fit the expected letters (the signer started
        elsewhere or jumped). The caller then checks with `locate` whether they fit elsewhere."""
        if len(self.tokens) < last:
            return False
        recent = [m for m in self.marks() if m.token >= len(self.tokens) - last]
        return sum(m.status in ("wrong", "extra") for m in recent) >= last // 2 + 1

    def mistakes(self):
        return [m for m in self.marks() if m.status in ("wrong", "missing", "extra")]


def locate(signed, text_letters, top=3):
    """Where in a long text does a short spelled sequence fit best? Approximate substring search
    (Sellers): free start and end in the text, unit costs. -> [(end index, edit distance)]"""
    text = np.asarray(text_letters)
    prev = np.zeros(len(text) + 1)
    for i, ch in enumerate(signed, 1):
        y = np.empty_like(prev)
        y[0] = i
        y[1:] = np.minimum(prev[:-1] + (text != ch), prev[1:] + 1)
        # a text letter left out by the signer: cur[j] = min over k<=j of y[k] + (j - k)
        idx = np.arange(len(y))
        prev = np.minimum.accumulate(y - idx) + idx
    order = np.argsort(prev[1:], kind="stable")
    found, used = [], []
    for e in order:
        if all(abs(e - u) > len(signed) for u in used):
            found.append((int(e), float(prev[e + 1])))
            used.append(e)
        if len(found) == top:
            break
    return found

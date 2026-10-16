"""Deterministic training-set augmentation: contiguous sentence-window crops of TRAINING texts (per text).

A crop keeps a contiguous run of the text's sentences (character span of the original string, so line breaks and
markdown inside the window are kept), covering a random fraction in [lo, hi] of the sentences and at least
`min_words` words. Several distinct windows can be drawn per text. The random generator of each text is seeded by
(seed, crc32(text id)), so the crops depend only on the text, its id and the CLI arguments: a replay (any row
order, any test size) reproduces them exactly. Only training-fold rows are ever augmented by the callers;
validation / test rows stay untouched.

Modes:
  window   any contiguous window (start anywhere)
  prefix   the window always starts at the first sentence (keeps the opener, drops the tail)
"""
from __future__ import annotations

import math
import re
import zlib

import numpy as np

# sentence end: . ! ? … (optionally followed by closing quotes / brackets) + whitespace; blank lines; a line break
# before a list item. Same boundaries as style2_features.SENT_SPLIT.
_SEP = re.compile(r"(?<=[.!?…])[\"”’)\]]*\s+|\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)")
_WORD = re.compile(r"\S+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SEP.finditer(text):
        end = m.start()
        # keep closing quotes / brackets with the sentence
        seg = m.group(0)
        k = 0
        while k < len(seg) and not seg[k].isspace():
            k += 1
        end += k
        if text[start:end].strip():
            spans.append((start, end))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    return spans


def crop_windows(text: str, lo: float, hi: float, min_words: int, mode: str) -> list[tuple[int, int]]:
    """All valid (first, last+1) sentence windows of the text."""
    spans = sentence_spans(text)
    n = len(spans)
    if n < 2:
        return []
    Lmin = max(1, math.ceil(lo * n))
    Lmax = min(n - 1, math.floor(hi * n))
    out = []
    for L in range(Lmin, Lmax + 1):
        starts = [0] if mode == "prefix" else range(0, n - L + 1)
        for s in starts:
            a, b = spans[s][0], spans[s + L - 1][1]
            if len(_WORD.findall(text[a:b])) >= min_words:
                out.append((s, s + L))
    return out


def make_crops(ids, texts, n_crops: int, lo: float = 0.5, hi: float = 0.9, min_words: int = 40,
               mode: str = "window", seed: int = 0):
    """Returns (src, crops): src[i] = row index (into ids / texts) of crop i's source text."""
    src, crops = [], []
    for i, (tid, text) in enumerate(zip(ids, texts)):
        if n_crops <= 0:
            break
        wins = crop_windows(text, lo, hi, min_words, mode)
        if not wins:
            continue
        rng = np.random.default_rng([seed, zlib.crc32(str(tid).encode("utf-8"))])
        pick = rng.choice(len(wins), size=min(n_crops, len(wins)), replace=False)
        spans = sentence_spans(text)
        for j in sorted(pick):
            s, e = wins[j]
            src.append(i)
            crops.append(text[spans[s][0]:spans[e - 1][1]].strip())
    return np.asarray(src, dtype=np.int64), crops


def crop_stats(texts, src, crops) -> str:
    nw = np.array([len(_WORD.findall(c)) for c in crops]) if crops else np.zeros(1)
    n_src = len(set(src.tolist())) if len(src) else 0
    return (f"{len(crops)} crops from {n_src}/{len(texts)} texts; crop words median {np.median(nw):.0f} "
            f"(min {nw.min():.0f}, max {nw.max():.0f})")


if __name__ == "__main__":
    from common import load_train
    tr = load_train()
    for mode in ("window", "prefix"):
        for k in (1, 2, 3):
            src, crops = make_crops(tr["id"], tr["text"], k, mode=mode)
            print(mode, k, crop_stats(tr["text"], src, crops))
    src, crops = make_crops(tr["id"][:3], tr["text"][:3], 2)
    for s, c in zip(src, crops):
        print(s, repr(tr["text"][s][:200]), "\n  ->", repr(c[:200]))

"""Hand-crafted style / formatting features of a text (no fitted state, so no fold leakage).

Used by the EDA (format traces per model) and by the style + LightGBM baseline. Counts are also given as
rates per 100 words where length would otherwise dominate.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

WORD = re.compile(r"[A-Za-zÀ-ɏ]+(?:['’][A-Za-z]+)?")
SENT_SPLIT = re.compile(r"(?<=[.!?…])[\"”’)\]]*\s+")
PARA_SPLIT = re.compile(r"\n\s*\n")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0001F900-\U0001F9FF]")

# (name, regex, flags): occurrence counts
PATTERNS = [
    ("md_header", r"^\s{0,3}#{1,6}\s", re.M),
    ("md_bold", r"\*\*[^*\n]+\*\*|__[^_\n]+__", 0),
    ("md_italic", r"(?<![*\w])\*[^*\n]+\*(?![*\w])|(?<![_\w])_[^_\n]+_(?![_\w])", 0),
    ("md_code", r"`[^`\n]+`", 0),
    ("md_fence", r"```", 0),
    ("md_table", r"^\s*\|.*\|\s*$", re.M),
    ("md_link", r"\[[^\]\n]+\]\([^)\n]+\)", 0),
    ("md_hr", r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.M),
    ("bullet", r"^\s*[-*•●▪◦]\s+", re.M),
    ("numbered", r"^\s*\d+[.)]\s+", re.M),
    ("em_dash", r"—", 0),
    ("em_dash_spaced", r"\s—\s", 0),
    ("em_dash_tight", r"\S—\S", 0),
    ("en_dash", r"–", 0),
    ("en_dash_spaced", r"\s–\s", 0),
    ("double_hyphen", r"--", 0),
    ("spaced_hyphen", r"\s-\s", 0),
    ("hyphen_word", r"\w-\w", 0),
    ("curly_double", r"[“”]", 0),
    ("curly_single", r"[‘’]", 0),
    ("curly_apostrophe", r"\w’\w", 0),
    ("straight_double", r"\"", 0),
    ("straight_apostrophe", r"\w'\w", 0),
    ("straight_single_quote", r"(?<!\w)'|'(?!\w)", 0),
    ("ellipsis_char", r"…", 0),
    ("three_dots", r"\.\.\.", 0),
    ("latex_inline", r"\$[^$\n]+\$", 0),
    ("latex_paren", r"\\\(|\\\[", 0),
    ("latex_cmd", r"\\[A-Za-z]+", 0),
    ("arrow", r"[→←↔⇒]", 0),
    ("math_sym", r"[×÷≈≠≤≥±∞√π°]", 0),
    ("nbsp", "[    ]", 0),
    ("zero_width", "[​‌‍﻿]", 0),
    ("comma", r",", 0),
    ("semicolon", r";", 0),
    ("colon", r":", 0),
    ("exclaim", r"!", 0),
    ("question", r"\?", 0),
    ("paren", r"\(", 0),
    ("bracket", r"\[", 0),
    ("slash", r"/", 0),
    ("ampersand", r"&", 0),
    ("percent", r"%", 0),
    ("digit", r"\d", 0),
    ("number_token", r"\b\d[\d,.]*\b", 0),
    ("not_x_but_y", r"\bnot (?:just|only|merely|simply)?\s*\w+(?:\s\w+)?,? but\b", re.I),
    ("isnt_just", r"\b(?:isn['’]t|is not|aren['’]t|are not|wasn['’]t|was not) (?:just|only|merely|simply)\b", re.I),
    ("contraction", r"\b\w+['’](?:t|s|re|ve|ll|d|m)\b", re.I),
    # v2: case-insensitive me/my/mine/myself (sentence-initial "My" was missed) and "I" except "Type I"
    ("first_person_sg", r"(?i:\b(?:me|my|mine|myself)\b)|(?<!Type )\bI\b", 0),
    ("first_person_pl", r"\b(?:we|us|our|ours|ourselves)\b", re.I),
    ("second_person", r"\b(?:you|your|yours|yourself)\b", re.I),
]
_COMPILED = [(n, re.compile(p, f)) for n, p, f in PATTERNS]
RATE_COLS = [n for n, _, _ in PATTERNS]


def _sent_lengths(text: str) -> np.ndarray:
    sents = [s for s in SENT_SPLIT.split(text.strip()) if s.strip()]
    return np.array([len(WORD.findall(s)) for s in sents] or [0], dtype=float)


def _last_char_class(text: str) -> str:
    t = text.rstrip()
    if not t:
        return "empty"
    c = t[-1]
    if c in ".!?…":
        return {".": "period", "!": "exclaim", "?": "question", "…": "ellipsis"}[c]
    if c in "\"”’')]*":
        return "quote_or_bracket"
    if c.isalnum():
        return "alnum"
    return "other"


def style_frame(texts) -> pd.DataFrame:
    rows = []
    for t in texts:
        words = WORD.findall(t)
        nw = max(len(words), 1)
        wl = np.array([len(w) for w in words] or [0], dtype=float)
        sl = _sent_lengths(t)
        paras = [p for p in PARA_SPLIT.split(t.strip()) if p.strip()]
        lines = [ln for ln in t.splitlines() if ln.strip()]
        lower = [w.lower() for w in words]
        nonascii = sum(ord(c) > 127 for c in t)
        r = dict(
            n_chars=len(t), n_words=len(words), n_lines=len(lines), n_paragraphs=len(paras),
            n_newlines=t.count("\n"), n_sentences=len(sl),
            sent_mean=sl.mean(), sent_std=sl.std(), sent_max=sl.max(), sent_min=sl.min(),
            word_len_mean=wl.mean(), word_len_std=wl.std(), long_word_rate=float((wl >= 9).mean()),
            ttr=len(set(lower)) / nw,
            hapax_rate=sum(1 for w, c in pd.Series(lower).value_counts().items() if c == 1) / nw if words else 0.0,
            nonascii_rate=nonascii / max(len(t), 1),
            upper_rate=sum(c.isupper() for c in t) / max(len(t), 1),
            titlecase_word_rate=sum(w[0].isupper() for w in words) / nw,
            emoji=len(EMOJI.findall(t)),
            lead_ws=int(t[:1].isspace()) if t else 0, trail_ws=int(t[-1:].isspace()) if t else 0,
            starts_quote=int(t.lstrip()[:1] in "\"“'‘"), starts_md=int(t.lstrip()[:1] in "#*-|"),
            ends_period=int(_last_char_class(t) == "period"), ends_question=int(_last_char_class(t) == "question"),
            ends_exclaim=int(_last_char_class(t) == "exclaim"), ends_quote=int(_last_char_class(t) == "quote_or_bracket"),
            ends_alnum=int(_last_char_class(t) == "alnum"),
            para_words_mean=float(np.mean([len(WORD.findall(p)) for p in paras])) if paras else 0.0,
            unicode_punct=sum(1 for c in t if ord(c) > 127 and unicodedata.category(c).startswith("P")),
        )
        for name, rx in _COMPILED:
            c = len(rx.findall(t))
            r[name] = c
            r[name + "_p100w"] = 100.0 * c / nw
        rows.append(r)
    return pd.DataFrame(rows)


def last_char_class(texts) -> list[str]:
    return [_last_char_class(t) for t in texts]

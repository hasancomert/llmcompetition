"""Second set of hand-crafted stylometric features (no fitted state: every column is a pure function of one text,
so they can be computed once for train and test without fold leakage).

Blocks (prefix of the column names):
  fw_*    relative frequency of ~150 fixed English function words (per word)
  sl_*    sentence-length distribution: histogram fractions, quantiles, coefficient of variation
  wl_*    word-length distribution: fraction of words of length 1..12 and 13+
  pb_*    punctuation bigrams: consecutive marks in the punctuation-only sequence of the text (per mark)
  tt_*    type-token variants: TTR of the first 50 / 100 words, MATTR(25), Yule's K, Honore R, Herdan C, Simpson D,
          hapax / dis legomena rates
  cl_*    clause stats: commas, coordinators, subordinators, relative pronouns per sentence; sentence openers
  cap_*   capitalisation: all-caps words, capitalised non-initial words, sentences starting lower case
  ch_*    specific characters per 1000 chars (curly quotes, dashes, ellipsis, semicolons, ...)
  sf_*    suffix rates (-ly, -ing, -tion, -ness, -ment, -ive, -ous, -al)
  pa_*    paragraph / line stats
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

WORD = re.compile(r"[A-Za-zÀ-ɏ]+(?:['’][A-Za-z]+)?")
SENT_SPLIT = re.compile(r"(?<=[.!?…])[\"”’)\]]*\s+|\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)")
PARA_SPLIT = re.compile(r"\n\s*\n")

FUNCTION_WORDS = """a about above across after again against all almost along already also although always am among
an and another any anyone anything are around as at be because been before being below beneath besides between
beyond both but by can cannot could did do does doing down during each either enough even ever every few for from
further had has have having he her here hers herself him himself his how however i if in indeed instead into is it
its itself just least less many may me might mine more moreover most much must my myself neither never
nevertheless no nobody none nor not nothing now of off often on once one only onto or other others otherwise our
ours ourselves out over own perhaps quite rather same several shall she should since so some something such than
that the their theirs them themselves then there therefore these they this those though through throughout thus to
too toward towards under unless until up upon us very was we were what whatever when whenever where whereas
whether which while who whom whose why will with within without would yet you your yours yourself""".split()
FW_SET = set(FUNCTION_WORDS)
COORD = {"and", "but", "or", "nor", "so", "yet", "for"}
SUBORD = {"because", "although", "though", "while", "whereas", "since", "unless", "until", "if", "when", "whenever",
          "where", "after", "before", "once", "as", "whether"}
RELPRON = {"which", "that", "who", "whom", "whose", "where"}
OPENERS = ["the", "this", "it", "in", "but", "and", "however", "so", "i", "we", "you", "a", "if", "when", "as",
           "there", "that", "these", "for", "what"]

PUNCT_MARKS = {".": "dot", ",": "com", ";": "sem", ":": "col", "!": "exc", "?": "que", "—": "em", "–": "en",
               "-": "hy", "(": "lp", ")": "rp", "\"": "dq", "“": "ldq", "”": "rdq", "'": "sq", "’": "rsq",
               "…": "ell", "*": "ast", "\n": "nl"}
PB_KEYS = [(a, b) for a in PUNCT_MARKS.values() for b in PUNCT_MARKS.values()]

CHARS = {"ldq": "“", "rdq": "”", "lsq": "‘", "rsq": "’", "dq": "\"", "sq": "'", "em": "—", "en": "–",
         "hy": "-", "ell": "…", "sem": ";", "col": ":", "exc": "!", "que": "?", "lpar": "(", "lbr": "[",
         "ast": "*", "hash": "#", "sl": "/", "amp": "&", "pct": "%", "dol": "$", "nl": "\n", "tab": "\t",
         "nbsp": " ", "arrow": "→", "times": "×", "bullet": "•", "us": "_", "bq": "`", "tilde": "~",
         "plus": "+", "eq": "=", "lt": "<", "gt": ">"}
SL_BINS = [0, 5, 10, 15, 20, 25, 30, 40, 60, 10_000]
SUFFIXES = ["ly", "ing", "tion", "ness", "ment", "ive", "ous", "al", "ed", "ity", "able", "est"]


def _ttr_prefix(lw, n):
    return len(set(lw[:n])) / max(min(n, len(lw)), 1)


def _mattr(lw, w=25):
    if len(lw) <= w:
        return len(set(lw)) / max(len(lw), 1)
    c = Counter(lw[:w])
    tot = len(c)
    s = tot
    for i in range(w, len(lw)):
        out, inn = lw[i - w], lw[i]
        c[out] -= 1
        if c[out] == 0:
            del c[out]
        c[inn] += 1
        s += len(c)
    return s / (len(lw) - w + 1) / w


def text_features(t: str) -> dict:
    r = {}
    words = WORD.findall(t)
    lw = [w.lower().replace("’", "'") for w in words]
    nw = max(len(lw), 1)
    nc = max(len(t), 1)
    cnt = Counter(lw)
    # function words
    for f in FUNCTION_WORDS:
        r["fw_" + f] = cnt.get(f, 0) / nw
    r["fw_total"] = sum(cnt.get(f, 0) for f in FW_SET) / nw
    # sentences
    sents = [s for s in SENT_SPLIT.split(t.strip()) if s and s.strip()]
    sw = [WORD.findall(s) for s in sents]
    sl = np.array([len(x) for x in sw if len(x) > 0] or [0], float)
    h, _ = np.histogram(sl, SL_BINS)
    for i, v in enumerate(h):
        r[f"sl_bin{SL_BINS[i]}"] = v / max(len(sl), 1)
    for q in (10, 25, 50, 75, 90):
        r[f"sl_q{q}"] = float(np.percentile(sl, q))
    r["sl_cv"] = float(sl.std() / sl.mean()) if sl.mean() > 0 else 0.0
    r["sl_n"] = len(sl)
    r["sl_logmean"] = float(np.log1p(sl).mean())
    r["sl_diff_mean"] = float(np.abs(np.diff(sl)).mean()) if len(sl) > 1 else 0.0
    # word lengths
    wl = np.array([len(w) for w in words] or [0])
    for L in range(1, 13):
        r[f"wl_{L}"] = float((wl == L).mean())
    r["wl_13p"] = float((wl >= 13).mean())
    r["wl_median"] = float(np.median(wl))
    r["wl_syll_proxy"] = float(np.mean([len(re.findall(r"[aeiouy]+", w)) for w in lw])) if lw else 0.0
    # punctuation bigrams
    seq = [PUNCT_MARKS[c] for c in t if c in PUNCT_MARKS]
    ns = max(len(seq), 1)
    pb = Counter(zip(seq, seq[1:]))
    for a, b in PB_KEYS:
        r[f"pb_{a}_{b}"] = pb.get((a, b), 0) / ns
    r["pb_marks_per_word"] = len(seq) / nw
    # type-token variants
    V = len(cnt)
    N = len(lw)
    v1 = sum(1 for c in cnt.values() if c == 1)
    v2 = sum(1 for c in cnt.values() if c == 2)
    r["tt_ttr50"] = _ttr_prefix(lw, 50)
    r["tt_ttr100"] = _ttr_prefix(lw, 100)
    r["tt_mattr25"] = _mattr(lw, 25)
    r["tt_mattr50"] = _mattr(lw, 50)
    s2 = sum(c * c for c in cnt.values())
    r["tt_yule_k"] = 1e4 * (s2 - N) / (N * N) if N > 0 else 0.0
    r["tt_honore"] = 100 * math.log(max(N, 2)) / max(1 - v1 / max(V, 1), 1e-3)
    r["tt_herdan"] = math.log(max(V, 2)) / math.log(max(N, 2))
    r["tt_simpson"] = sum(c * (c - 1) for c in cnt.values()) / max(N * (N - 1), 1)
    r["tt_hapax"] = v1 / max(N, 1)
    r["tt_dis"] = v2 / max(N, 1)
    r["tt_content_ttr"] = len({w for w in lw if w not in FW_SET}) / max(sum(1 for w in lw if w not in FW_SET), 1)
    # clause stats
    nsent = max(len(sents), 1)
    r["cl_comma_per_sent"] = t.count(",") / nsent
    r["cl_coord_per_sent"] = sum(cnt.get(w, 0) for w in COORD) / nsent
    r["cl_subord_per_sent"] = sum(cnt.get(w, 0) for w in SUBORD) / nsent
    r["cl_rel_per_sent"] = sum(cnt.get(w, 0) for w in RELPRON) / nsent
    r["cl_semicolon_per_sent"] = t.count(";") / nsent
    r["cl_colon_per_sent"] = t.count(":") / nsent
    r["cl_dash_per_sent"] = (t.count("—") + t.count("–")) / nsent
    r["cl_paren_per_sent"] = t.count("(") / nsent
    first = [x[0].lower().replace("’", "'") for x in sw if x]
    nf = max(len(first), 1)
    for o in OPENERS:
        r["cl_open_" + o] = sum(1 for f in first if f == o) / nf
    r["cl_open_fw"] = sum(1 for f in first if f in FW_SET) / nf
    r["cl_open_ing"] = sum(1 for f in first if f.endswith("ing")) / nf
    r["cl_open_ly"] = sum(1 for f in first if f.endswith("ly")) / nf
    r["cl_question_sent"] = sum(1 for s in sents if s.rstrip().endswith("?")) / nsent
    r["cl_exclaim_sent"] = sum(1 for s in sents if s.rstrip().endswith("!")) / nsent
    r["cl_ntt"] = (cnt.get("n't", 0) + sum(c for w, c in cnt.items() if w.endswith("n't"))) / nw
    r["cl_poss_s"] = sum(c for w, c in cnt.items() if w.endswith("'s")) / nw
    # capitalisation
    r["cap_allcaps"] = sum(1 for w in words if len(w) > 1 and w.isupper()) / nw
    initial = set()
    for x in sw:
        if x:
            initial.add(id(x[0]))
    noninit_caps = 0
    for x in sw:
        for w in x[1:]:
            if w[:1].isupper() and w != "I":
                noninit_caps += 1
    r["cap_noninit"] = noninit_caps / nw
    r["cap_sent_lower"] = sum(1 for s in sents if s.strip()[:1].islower()) / nsent
    r["cap_upper_char"] = sum(c.isupper() for c in t) / nc
    r["cap_title_line"] = sum(1 for ln in t.splitlines() if ln.strip() and ln.strip().istitle()) / max(len(t.splitlines()), 1)
    # characters per 1000 chars
    for k, c in CHARS.items():
        r["ch_" + k] = 1000 * t.count(c) / nc
    r["ch_digit"] = 1000 * sum(c.isdigit() for c in t) / nc
    r["ch_space"] = 1000 * t.count(" ") / nc
    r["ch_dblspace"] = 1000 * t.count("  ") / nc
    r["ch_nonascii"] = 1000 * sum(ord(c) > 127 for c in t) / nc
    r["ch_dblnl"] = 1000 * t.count("\n\n") / nc
    # suffixes
    for s in SUFFIXES:
        r["sf_" + s] = sum(1 for w in lw if len(w) > len(s) + 2 and w.endswith(s)) / nw
    # paragraphs / lines
    paras = [p for p in PARA_SPLIT.split(t.strip()) if p.strip()]
    pw = np.array([len(WORD.findall(p)) for p in paras] or [0], float)
    r["pa_n"] = len(paras)
    r["pa_words_std"] = float(pw.std())
    r["pa_words_max_frac"] = float(pw.max() / nw)
    r["pa_first_words"] = float(pw[0])
    r["pa_last_words"] = float(pw[-1])
    lines = [ln for ln in t.splitlines() if ln.strip()]
    r["pa_lines"] = len(lines)
    r["pa_sent_per_para"] = nsent / max(len(paras), 1)
    r["pa_n_words_log"] = math.log1p(N)
    r["pa_chars_per_word"] = len(t) / nw
    return r


def style2_frame(texts) -> pd.DataFrame:
    return pd.DataFrame([text_features(t) for t in texts]).astype(np.float32)


if __name__ == "__main__":
    import time
    from common import load_train
    tr = load_train()
    t0 = time.time()
    F = style2_frame(tr["text"].head(200))
    print(F.shape, f"{time.time() - t0:.1f}s")
    print(F.iloc[:3, :10])
    print((F.nunique() > 1).sum(), "non-constant of", F.shape[1])

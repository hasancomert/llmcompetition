"""Tokenizer features: how the same text is split by different labs' tokenizers (PLAN.md Adım 2.3).

Hypothesis: text sampled from a model is made of tokens of that model's vocabulary, so its own lab's tokenizer
splits it into relatively fewer / more frequent tokens. Features per tokenizer t:
  n_tok_t, chars_per_tok_t, words_per_tok_t, uniq_tok_frac_t, mean_log_id_t (BPE ids roughly follow merge
  order, so a higher id = a rarer merge), frac_id_gt_50k_t, frac_single_char_t
and for every pair (a, b): log(n_tok_a / n_tok_b).

Tokenizer files are loaded offline from TOKENIZER_DIR (default <repo>/models/tokenizers; on Kaggle an attached
dataset path). Sources, licenses and hashes: models/tokenizers/*/SOURCE.json and MODELS_LICENSES.md.
"""
from __future__ import annotations

import base64
import itertools
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from style_features import WORD

TOKENIZER_DIR = Path(os.environ.get("TOKENIZER_DIR", Path(__file__).resolve().parents[1] / "models" / "tokenizers"))

# name -> (subfolder, kind); kind "hf" = tokenizers.Tokenizer from tokenizer.json, "tiktoken" = ranks file + pat_str
TOKENIZERS = {
    "o200k": ("openai_o200k", "hf"),
    "qwen3": ("qwen", "hf"),
    "qwen35": ("qwen_new", "hf"),
    "glm": ("glm", "hf"),
    "kimi": ("kimi", "tiktoken"),
    "gemma": ("gemma", "hf"),
}


@lru_cache(maxsize=None)
def load(name: str):
    """Returns a function text -> list of token ids (no special tokens)."""
    sub, kind = TOKENIZERS[name]
    d = TOKENIZER_DIR / sub
    if kind == "hf":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        tok.no_truncation()
        tok.no_padding()
        return lambda text: tok.encode(text, add_special_tokens=False).ids
    import tiktoken
    ranks = {}
    with open(d / "tiktoken.model", "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                t, r = line.split()
                ranks[base64.b64decode(t)] = int(r)
    pat = (d / "pat_str.txt").read_text(encoding="utf-8")
    enc = tiktoken.Encoding(name=f"{name}_offline", pat_str=pat, mergeable_ranks=ranks, special_tokens={})
    return lambda text: enc.encode(text, disallowed_special=())


def tokenizer_frame(texts, names=None) -> pd.DataFrame:
    names = list(names or TOKENIZERS)
    texts = list(texts)
    n_chars = np.array([len(t) for t in texts], dtype=float)
    n_words = np.array([max(len(WORD.findall(t)), 1) for t in texts], dtype=float)
    cols = {}
    for nm in names:
        enc = load(nm)
        ids = [enc(t) for t in texts]
        n = np.array([max(len(i), 1) for i in ids], dtype=float)
        cols[f"n_tok_{nm}"] = n
        cols[f"chars_per_tok_{nm}"] = n_chars / n
        cols[f"words_per_tok_{nm}"] = n_words / n
        cols[f"uniq_tok_frac_{nm}"] = np.array([len(set(i)) / max(len(i), 1) for i in ids])
        cols[f"mean_log_id_{nm}"] = np.array([np.mean(np.log1p(i)) if i else 0.0 for i in ids])
        cols[f"frac_id_gt_50k_{nm}"] = np.array([np.mean(np.asarray(i) > 50_000) if i else 0.0 for i in ids])
        cols[f"frac_single_char_{nm}"] = _single_char_frac(nm, ids)
    for a, b in itertools.combinations(names, 2):
        cols[f"logratio_{a}_{b}"] = np.log(cols[f"n_tok_{a}"] / cols[f"n_tok_{b}"])
    return pd.DataFrame(cols)


BYTE_LEVEL = {"o200k", "qwen3", "qwen35", "glm"}  # GPT-2 style byte-to-unicode vocab: one char = one byte
BYTE_FALLBACK = re.compile(r"<0x[0-9A-Fa-f]{2}>")


@lru_cache(maxsize=None)
def _token_char_len(name: str):
    """UTF-8 length in bytes of every token id, as an array indexed by id (feature version v2).

    Byte-level vocabularies (o200k, Qwen, GLM) store one byte per character; tiktoken ranks (Kimi) are raw
    bytes. Gemma is SentencePiece-style ('▁' = space, '<0xNN>' byte-fallback tokens), so its pieces are
    converted to UTF-8 bytes; v1 counted characters there, which made an em dash a "single-char" token.
    """
    sub, kind = TOKENIZERS[name]
    d = TOKENIZER_DIR / sub
    if kind == "tiktoken":
        lens = {}
        with open(d / "tiktoken.model", "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    t, r = line.split()
                    lens[int(r)] = len(base64.b64decode(t))
        arr = np.zeros(max(lens) + 1, dtype=np.int32)
        for r, n in lens.items():
            arr[r] = n
        return arr
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    vocab = tok.get_vocab(with_added_tokens=True)
    arr = np.zeros(max(vocab.values()) + 1, dtype=np.int32)
    for s, i in vocab.items():
        if name in BYTE_LEVEL:
            arr[i] = len(s)
        elif BYTE_FALLBACK.fullmatch(s):
            arr[i] = 1
        else:
            arr[i] = len(s.replace("▁", " ").encode("utf-8"))
    return arr


def _single_char_frac(name: str, ids) -> np.ndarray:
    L = _token_char_len(name)
    out = np.empty(len(ids))
    for k, i in enumerate(ids):
        a = np.asarray(i, dtype=np.int64)
        a = a[a < len(L)]
        out[k] = float(np.mean(L[a] <= 1)) if len(a) else 0.0
    return out


if __name__ == "__main__":
    import time
    from common import OUT, load_train
    t0 = time.time()
    tr = load_train()
    F = tokenizer_frame(tr.text)
    print(F.shape, f"{time.time() - t0:.1f}s")
    print(F.groupby(tr.model.values)[[c for c in F.columns if c.startswith("chars_per_tok_")]].mean().round(3).to_string())

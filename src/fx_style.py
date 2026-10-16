"""Fit test, style side: generated text vs the real train text of the SAME prompt_id and label.

Per pair: length, sentence length, punctuation rates per 100 words (em/en dash, semicolon, colon, comma,
parentheses, ? !), apostrophe / double-quote typography (curly vs straight), markdown / newlines in the raw output,
opening word and closing character. Reports real mean, generated mean and the paired difference per run, plus the
same statistics for ALL train texts of the label (is the label's style stable across prompts?).

Usage: python fx_style.py --runs flash_low glm53_low kimik3_low glm53_default flash_default
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

import featherless as F

WORD = re.compile(r"\b\w[\w'’-]*\b")


def feats(t: str) -> dict:
    w = WORD.findall(t)
    nw = max(len(w), 1)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]
    per100 = lambda pat: 100.0 * len(re.findall(pat, t)) / nw
    first = re.sub(r"[^\w’']", "", t.split()[0]) if t.split() else ""
    return {
        "words": len(t.split()), "chars_per_word": len(re.sub(r"\s", "", t)) / max(len(t.split()), 1),
        "sentences": len(sents), "words_per_sentence": len(t.split()) / max(len(sents), 1),
        "emdash_100w": per100("—"), "spaced_emdash_100w": per100(" — "), "endash_100w": per100("–"),
        "semicolon_100w": per100(";"), "colon_100w": per100(":"), "comma_100w": per100(","),
        "paren_100w": per100(r"\("), "question_100w": per100(r"\?"), "exclaim_100w": per100("!"),
        "has_emdash": float("—" in t), "has_semicolon": float(";" in t),
        "curly_apos": float("’" in t), "straight_apos": float(bool(re.search(r"\w'\w", t))),
        "curly_dquote": float(bool(re.search(r"[“”]", t))), "straight_dquote": float('"' in t),
        "ends_question": float(t.rstrip().endswith("?")), "first_word": first, "last_char": t.rstrip()[-1:] or "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", default="fit_v1")
    a = ap.parse_args()
    tr = pd.read_csv(F.ROOT / "data" / "train.csv")
    num_cols = None
    out = []
    pd.set_option("display.width", 250)
    for run in a.runs:
        rows = [json.loads(l) for l in (F.EXT / f"gen_{run}.jsonl").read_text().splitlines() if l.strip()]
        g = pd.DataFrame(rows)
        lab = g["label"].iloc[0]
        real = tr[tr.model == lab].set_index("prompt_id")["text"]
        G = pd.DataFrame([feats(t) for t in g["text"]])
        R = pd.DataFrame([feats(real[p]) for p in g["prompt_id"]])
        A = pd.DataFrame([feats(t) for t in real])             # all train texts of the label
        num_cols = [c for c in G.columns if c not in ("first_word", "last_char")]
        tab = pd.DataFrame({"real_same_prompts": R[num_cols].mean(), "generated": G[num_cols].mean(),
                            "paired_diff": (G[num_cols] - R[num_cols]).mean(),
                            "real_all_label": A[num_cols].mean()}).round(3)
        extra = {"run": run, "label": lab, "n": len(g),
                 "raw_newlines_share": float((g["newlines_removed"] > 0).mean()),
                 "raw_markdown_share": float((g["markdown_removed"] > 0).mean()),
                 "finish_reason": g["finish_reason"].value_counts().to_dict(),
                 "len_ratio_median": float((G["words"] / R["words"]).median()),
                 "same_first_word_share": float((G["first_word"].str.lower() == R["first_word"].str.lower()).mean()),
                 "top_first_words_gen": G["first_word"].value_counts().head(6).to_dict(),
                 "top_first_words_real": R["first_word"].value_counts().head(6).to_dict(),
                 "last_char_gen": G["last_char"].value_counts().head(4).to_dict(),
                 "last_char_real": R["last_char"].value_counts().head(4).to_dict(),
                 "reasoning_chars_mean": float(g["reasoning_chars"].mean()),
                 "completion_tokens_mean": float(g["completion_tokens"].mean()),
                 "cost_usd": float(g["cost_usd"].sum())}
        print(f"\n===== {run} ({lab}, n={len(g)})")
        print(tab.to_string())
        print(json.dumps(extra, ensure_ascii=False, indent=None))
        out.append({**extra, "table": tab.to_dict()})
    (F.EXT / f"style_{a.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""Leakage check for the external corpus: does an external answer respond to the same question as an official
train prompt_id? Then a model trained on it would have seen that validation prompt, and CV would be inflated.

The official prompts are unknown, but every prompt_id has ~18 answers. Topic representation: TF-IDF over content
words (word 1-2-grams, English stop words removed, sublinear tf; fitted on official + external texts, no labels).
Each official prompt_id gets the centroid of its answers' unit vectors. An external answer is compared with every
centroid; its best cosine similarity and prompt_id are kept.

Calibration on the official data itself: for every official answer, the similarity to its own prompt's
leave-one-out centroid (same question) and the best similarity to any other prompt's centroid (different question).
Two thresholds are reported: strict = the smallest value above which no official answer matches a different
prompt (max other-prompt similarity + margin; detects only about half of the same-question answers, because answers
to one prompt use varied imagery), and inclusive = the 1st percentile of the own-prompt similarities (99% of
same-question answers reach it). Flagging uses the inclusive one: a false flag only removes an external row from one
fold's training set, a missed same-question row would inflate CV. An external prompt group is flagged for prompt_id p
if any of its answers reaches the threshold for p.

Output: data/external/ntl_corpus/leak_matches.csv (ext_id, prompt_group_id, best_pid, best_sim, flagged) and a
summary json; extdata.py excludes flagged rows from a training fold whenever the matched prompt_id is validated.

Usage: python ext_leak.py [--margin 0.02]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common import ROOT
import ext_corpus

OUT = ext_corpus.D / "leak_matches.csv"


def compute(tr: pd.DataFrame, ext: pd.DataFrame, own_quantile: float = 0.01, margin: float = 0.02):
    """-> (per-external-row match table, summary dict, official calibration arrays)."""
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, stop_words="english",
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]+\b", sublinear_tf=True, min_df=2, max_df=0.2,
                          dtype=np.float32)
    vec.fit(pd.concat([tr["text"], ext["text"]]).tolist())
    Xo = normalize(vec.transform(tr["text"]))
    Xe = normalize(vec.transform(ext["text"]))
    pids = np.array(sorted(tr["prompt_id"].unique()))
    pix = {p: i for i, p in enumerate(pids)}
    po = tr["prompt_id"].map(pix).to_numpy()
    S = np.zeros((len(pids), Xo.shape[1]), np.float32)          # sums of unit vectors per prompt
    for i, p in enumerate(po):
        S[p] += Xo[i].toarray()[0]
    C = S / np.linalg.norm(S, axis=1, keepdims=True)
    sims_all = np.asarray(Xo @ C.T)                                  # (n_official, n_prompts)
    xs = np.asarray(Xo @ S.T)
    own = np.empty(len(tr))
    other = np.empty(len(tr))
    for i, p in enumerate(po):                                       # own prompt: leave-one-out centroid
        s_loo = S[p] - Xo[i].toarray()[0]
        own[i] = (xs[i, p] - 1.0) / max(np.linalg.norm(s_loo), 1e-9)   # x . (S_p - x), |x| = 1
        row = sims_all[i].copy()
        row[p] = -1
        other[i] = row.max()
    strict = float(other.max() + margin)
    thr = float(np.quantile(own, own_quantile))
    Se = np.asarray(Xe @ C.T)
    best = Se.argmax(1)
    bsim = Se[np.arange(len(ext)), best]
    res = pd.DataFrame({"ext_id": ext["ext_id"].values, "prompt_group_id": ext["prompt_group_id"].values,
                        "label": ext["label"].values, "source": ext["source"].values, "best_pid": pids[best],
                        "best_sim": bsim.round(4)})
    res["flagged_answer"] = res["best_sim"] >= thr
    grp = res[res.flagged_answer].groupby("prompt_group_id")["best_pid"].agg(lambda s: ";".join(sorted(set(s))))
    res["group_pids"] = res["prompt_group_id"].map(grp).fillna("")
    res["flagged"] = res["group_pids"] != ""
    q = np.quantile(bsim, [0.5, 0.9, 0.99, 0.999])
    summ = {"threshold_inclusive": round(thr, 4), "own_detection": round(float((own >= thr).mean()), 4),
            "official_other_above_inclusive": round(float((other >= thr).mean()), 4),
            "threshold_strict": round(strict, 4), "own_detection_strict": round(float((own >= strict).mean()), 4),
            "flagged_answers_strict": int((bsim >= strict).sum()),
            "official_own_median": round(float(np.median(own)), 4), "official_own_q05": round(float(np.quantile(own, 0.05)), 4),
            "official_other_median": round(float(np.median(other)), 4),
            "official_other_q99": round(float(np.quantile(other, 0.99)), 4),
            "official_other_max": round(float(other.max()), 4),
            "external_best_sim_quantiles": {"q50": round(float(q[0]), 4), "q90": round(float(q[1]), 4),
                                            "q99": round(float(q[2]), 4), "q999": round(float(q[3]), 4)},
            "flagged_answers": int(res.flagged_answer.sum()), "flagged_rows_group_level": int(res.flagged.sum()),
            "flagged_groups": int(grp.shape[0]),
            "matched_prompt_ids": sorted(set(";".join(grp).split(";")) - {""})}
    return res, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--own-quantile", type=float, default=0.01, help="inclusive threshold = this quantile of own-prompt sims")
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    ext = ext_corpus.load()
    res, summ = compute(tr, ext, a.own_quantile, a.margin)
    res.to_csv(OUT, index=False)
    (ext_corpus.D / "leak_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    by_src = res[res.flagged].groupby("source").size().to_dict()
    print("flagged rows by source:", by_src)
    top = res.sort_values("best_sim", ascending=False).head(12)
    texts = ext.set_index("ext_id")
    prompts = pd.read_parquet(ext_corpus.D / "external_short_wide.parquet", columns=["id", "prompt"]).set_index("id")["prompt"]
    for _, r in top.iterrows():
        e = texts.loc[r.ext_id]
        o = tr[tr.prompt_id == r.best_pid]["text"].iloc[0]
        print(f"\n[{r.best_sim:.3f}] {r.source} {r.label} -> {r.best_pid}\n  EXT PROMPT: {str(prompts.get(e['id'], ''))[:160]!r}"
              f"\n  EXT: {e['text'][:160]!r}\n  OFFICIAL: {o[:160]!r}")


if __name__ == "__main__":
    main()

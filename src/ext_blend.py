"""Step 5 of the external-corpus test: the CPU per-text pipeline with and without external training rows.

Candidates = the CPU blend members (blend_pt.py set "cpu": tfidf_char_svc_C0.3, tfidf_charword_lr_C30,
lr_style_tok_v2, ng_char-word-fw_lr_C100_T, ng_char-word-fw-shapew_lr_C300_T, lpx_style2_nolm_C0.003_T). A variant
replaces every member that has an external-data version <member><suffix> (extdata.tag_suffix, e.g.
_ext-ntl_short-w0.3) by it; members without one stay as they are (listed). "union" offers both versions to the blend.

Same procedure as blend_pt_cpu_logloss (cpu2, nested 0.4038): per outer fold, hill-climb weights (log-prob space,
log loss) on the other four folds, applied to the held-out fold. Then the per-class bias correction for balanced
accuracy, nested the same way: per fold, additive log-prob biases fitted on the training-fold rows of that fold's
blend (family_combine.fit_bias, coordinate ascent on BA) and applied to the held-out rows. All OOF rows are official
texts (external rows only ever trained the members).

Variant "prior<suffix>": the external-data members with an analytic label-shift correction before blending:
p(c) * (1/18) / pi_train(c), renormalised, where pi_train(c) is the class share of the members' training weight
(official training rows + weight * external rows of class c). It removes the class prior the external rows bring in
(e.g. claude-opus-4-6 4.767 rows vs gemini flash 71-83) without fitting anything.

Reports per variant: nested BA / log loss without and with biases, paired bootstrap vs the baseline (same
treatment), per-class recall and its change, and the confusions of the glm / kimi / qwen / gemini pairs.
Usage: python ext_blend.py --suffix _ext-ntl_short-w0.3 [--suffix _ext-ntl_short-w1 ...] [--log]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import CLASSES, K, encode_y, iter_folds, load_folds, load_member, load_train, log_experiment, metrics, \
    scores
from compare import fmt, paired_bootstrap
from family_combine import apply_bias, fit_bias
import ext_corpus
import hill_climb  # noqa: E402  (reuse/airline)

BASE = ["tfidf_char_svc_C0.3", "tfidf_charword_lr_C30", "lr_style_tok_v2", "ng_char-word-fw_lr_C100_T",
        "ng_char-word-fw-shapew_lr_C300_T", "lpx_style2_nolm_C0.003_T"]
PAIRS = [("glm-5.3", "glm-5.3-flash"), ("glm-5.3-flash", "glm-5.3"), ("kimi-k3", "glm-5.3"),
         ("qwen3.8-max", "kimi-k3"), ("gemini-3.6-flash", "gemini-3.7-flash"), ("gemini-3.7-flash", "gemini-3.8-flash"),
         ("gpt-5.6-sol", "gpt-5.6-terra"), ("claude-opus-4-8", "claude-opus-5")]


def exists(tag):
    try:
        load_member(tag)
        return True
    except FileNotFoundError:
        return False


def nested(oofs, y, folds, bias):
    out_b, out = np.zeros((len(y), K)), np.zeros((len(y), K))
    ws = []
    for _, trn, val in iter_folds(folds):
        _, w = hill_climb.hill_climb(oofs, y, idx=trn, space="logprob", metric=metrics.mlogloss,
                                     higher_is_better=False)
        ws.append(w)
        B = hill_climb.blend(oofs, w, "logprob")
        out[val] = B[val]
        if bias:
            b = fit_bias(B[trn], y[trn])
            out_b[val] = apply_bias(B, b)[val]
    return out, (out_b if bias else None), np.mean(ws, 0)


def recall(y, P):
    pred = P.argmax(1)
    return np.array([np.mean(pred[y == c] == c) for c in range(K)])


def conf(y, P, a, b):
    pred = P.argmax(1)
    ia, ib = CLASSES.index(a), CLASSES.index(b)
    return int(((y == ia) & (pred == ib)).sum())


def prior_fix(P, w_ext, ext_counts, off_counts):
    """Divide out the training class prior (official rows weight 1 + external rows weight w_ext), uniform target."""
    pi = off_counts + w_ext * ext_counts
    pi = pi / pi.sum()
    Q = P * ((1.0 / K) / pi)[None, :]
    return Q / Q.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", action="append", required=True)
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y, folds, g = encode_y(tr["model"]), load_folds(tr), tr["prompt_id"].values
    variants = {"baseline": BASE}
    swapped, prior = {}, {}
    ext_counts = (np.bincount(encode_y(ext_corpus.load()["label"]), minlength=K).astype(float)
                  if any("ntl_short" in sfx for sfx in a.suffix) else None)
    off_counts = np.bincount(y, minlength=K) * 0.8                      # official rows in a training fold
    for sfx in a.suffix:
        rep = [t + sfx if exists(t + sfx) else t for t in BASE]
        swapped[sfx] = [t for t in BASE if exists(t + sfx)]
        variants[f"ext{sfx}"] = rep
        if "ntl_short" in sfx:                                            # corpus rows: strongly unbalanced labels
            variants[f"prior{sfx}"] = rep
            prior[f"prior{sfx}"] = float(sfx.split("-w")[1].split("-")[0])
        variants[f"union{sfx}"] = BASE + [t + sfx for t in swapped[sfx]]
    members = {}
    for sfx in a.suffix:                                                  # member-level effect of the external rows
        for t in swapped[sfx]:
            r = paired_bootstrap(y, load_member(t)[0], load_member(t + sfx)[0], g)
            members[t + sfx] = {"ba": f"{r['ba_a']:.4f} -> {r['ba_b']:.4f}", "diff": round(r["diff"], 4),
                                "ci": [round(r["ci_lo"], 4), round(r["ci_hi"], 4)],
                                "logloss": f"{r['ll_a']:.4f} -> {r['ll_b']:.4f}"}
            print(f"member {t}{sfx}: {fmt(r)}", flush=True)
    res, preds = {}, {}
    for name, tags in variants.items():
        oofs = [load_member(t)[0] for t in tags]
        if name in prior:                                                 # analytic prior fix of the ext members
            oofs = [prior_fix(o, prior[name], ext_counts, off_counts) if "_ext-" in t else o
                    for o, t in zip(oofs, tags)]
        P, Pb, w = nested(oofs, y, folds, bias=True)
        preds[name] = (P, Pb)
        s, sb = scores(y, P, folds), scores(y, Pb, folds)
        res[name] = {"members": tags, "weights": {t: round(float(x), 3) for t, x in zip(tags, w) if x > 0},
                     "ba": round(s["ba"], 4), "logloss": round(s["logloss"], 4),
                     "ba_bias": round(sb["ba"], 4), "logloss_bias": round(sb["logloss"], 4)}
    P0, P0b = preds["baseline"]
    rows = []
    for name in variants:
        P, Pb = preds[name]
        r = res[name]
        if name != "baseline":
            r["paired_vs_baseline"] = fmt(paired_bootstrap(y, P0, P, g))
            r["paired_vs_baseline_bias"] = fmt(paired_bootstrap(y, P0b, Pb, g))
        r["recall_bias"] = dict(zip(CLASSES, np.round(recall(y, Pb), 3).tolist()))
        r["confusions_bias"] = {f"{x}->{z}": conf(y, Pb, x, z) for x, z in PAIRS}
        rows.append(name)
    rec = pd.DataFrame({n: res[n]["recall_bias"] for n in variants})
    for n in variants:
        if n != "baseline":
            rec[f"d_{n}"] = (rec[n] - rec["baseline"]).round(3)
    pd.set_option("display.width", 250)
    for n in variants:
        r = res[n]
        print(f"\n[{n}] nested BA {r['ba']:.4f} ll {r['logloss']:.4f} | with per-class bias: BA {r['ba_bias']:.4f} "
              f"ll {r['logloss_bias']:.4f}\n  weights {r['weights']}")
        if n != "baseline":
            print(f"  vs baseline (no bias): {r['paired_vs_baseline']}\n  vs baseline (bias): {r['paired_vs_baseline_bias']}")
        print(f"  confusions (bias): {r['confusions_bias']}")
    print("\nper-class recall (nested blend + nested bias):")
    print(rec.to_string())
    out = ext_corpus.D / f"ext_blend{'_'.join(s.replace('_ext-', '') for s in a.suffix)}.json"
    out.write_text(json.dumps({"members": members, "variants": res, "swapped": swapped}, indent=1))
    if a.log:
        for n in variants:
            if n == "baseline":
                continue
            r = res[n]
            sb = scores(y, preds[n][1], folds)
            log_experiment(f"extblend_{n.replace('_ext-ntl_short-', '_').replace('_ext-', '_')}",
                           f"CPU per-text blend (nested hill-climb, logprob/logloss) + nested per-class BA biases; "
                           f"members {', '.join(r['members'])} (ext_blend.py; external rows in training folds only)",
                           sb, time.time() - t0,
                           note=(f"without biases: BA {r['ba']:.4f} ll {r['logloss']:.4f}; vs baseline (no bias) "
                                 f"{r['paired_vs_baseline']}; vs baseline (bias, BA {res['baseline']['ba_bias']:.4f}) "
                                 f"{r['paired_vs_baseline_bias']}; weights {r['weights']}; confusions "
                                 f"{r['confusions_bias']}; not saved"))


if __name__ == "__main__":
    main()

"""Fit test of the generated texts: classify every generated text with the fold model that never saw its prompt_id.

Classifier = the three strongest CPU per-text members (same code and settings as the saved members):
  ng_char-word-fw_lr_C100_T, ng_char-word-fw-shapew_lr_C300_T (train_ngram.run_fold) and lpx_style2_nolm_C0.003_T
  (train_lpx.fit_predict on style+tok v2 + style2 columns), blended with their cpu2 log-probability weights
  (0.173 / 0.307 / 0.173, renormalised). For outer fold k the members are fitted on the real train rows of the
  other four folds and applied to (a) the real rows of fold k (OOF; must reproduce the saved members) and (b) the
  generated texts whose prompt_id is in fold k. The real-text OOF recall of the same label is the yardstick.

Usage: python fx_predict.py --runs flash_low glm53_low kimik3_low [--tag fit_v1]
"""
from __future__ import annotations

import argparse
import json
from argparse import Namespace

import numpy as np
import pandas as pd

from common import CLASSES, K, encode_y, lab_of, load_folds, load_member, load_train, metrics
import featherless as F
import train_lpx
import train_ngram

WEIGHTS = {"ng_fw": 0.173, "ng_fwsh": 0.307, "style2": 0.173}
SAVED = {"ng_fw": "ng_char-word-fw_lr_C100_T", "ng_fwsh": "ng_char-word-fw-shapew_lr_C300_T",
         "style2": "lpx_style2_nolm_C0.003_T"}


def ng_args(feats, C):
    return Namespace(feats=feats, model="lr", C=C, alpha=1.0, nb_base="svc", temp=True, binary=False,
                     no_sublinear=False, no_idf=False, min_df=2, max_features=0, stream_weights="", n_jobs=2)


def load_gen(runs):
    rows = []
    for r in runs:
        for line in (F.EXT / f"gen_{r}.jsonl").read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                e["run"] = r
                rows.append(e)
    g = pd.DataFrame(rows)
    return g[g["text"].str.split().str.len() >= 5].reset_index(drop=True)


def member_predictions(tr, y, folds, g_ids, gen):
    """{member: (oof (N,K), gen_pred (M,K))} with the fold that excludes each generated text's prompt."""
    gfold = gen["fold"].to_numpy()
    out = {}
    for name, (feats, C) in (("ng_fw", ("char+word+fw", 100.0)), ("ng_fwsh", ("char+word+fw+shapew", 300.0))):
        args = ng_args(feats, C)
        T_all = train_ngram.transformed_texts(feats, tr["text"].tolist())
        T_gen = train_ngram.transformed_texts(feats, gen["text"].tolist())
        take = lambda d, idx: {k: [v[i] for i in idx] for k, v in d.items()}
        oof, pg = np.zeros((len(y), K)), np.zeros((len(gen), K))
        for k in range(int(folds.max()) + 1):
            trn, val, gk = np.flatnonzero(folds != k), np.flatnonzero(folds == k), np.flatnonzero(gfold == k)
            gk_ = gk if len(gk) else np.array([0])               # a fold without generated texts: dummy row
            p_va, p_g, _ = train_ngram.run_fold(args, take(T_all, trn), y[trn], g_ids[trn], take(T_all, val),
                                                take(T_gen, gk_))
            oof[val] = p_va
            if len(gk):
                pg[gk] = p_g
        out[name] = (oof, pg)
        print(f"member {name} done", flush=True)
    Xtr, Xg, names = train_lpx.build_blocks(["style", "style2"], tr, gen[["id", "text"]])
    keep = [i for i, n in enumerate(names) if not n.startswith("s2_pb_")]
    Xtr, Xg = Xtr[:, keep], Xg[:, keep]
    oof, pg = np.zeros((len(y), K)), np.zeros((len(gen), K))
    for k in range(int(folds.max()) + 1):
        trn, val, gk = np.flatnonzero(folds != k), np.flatnonzero(folds == k), np.flatnonzero(gfold == k)
        gk_ = gk if len(gk) else np.array([0])
        (pv, pgk), _ = train_lpx.fit_predict("lr", 0.003, Xtr[trn], y[trn], g_ids[trn], [Xtr[val], Xg[gk_]], True)
        oof[val] = pv
        if len(gk):
            pg[gk] = pgk
    out["style2"] = (oof, pg)
    print("member style2 done", flush=True)
    return out


def blend(P):
    z = sum(w * np.log(np.clip(P[m], 1e-9, 1)) for m, w in WEIGHTS.items()) / sum(WEIGHTS.values())
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", default="fit_v1")
    a = ap.parse_args()
    tr = load_train()
    y, folds = encode_y(tr["model"]), load_folds(tr)
    g_ids = tr["prompt_id"].to_numpy()
    pfold = dict(zip(tr["prompt_id"], folds))
    gen = load_gen(a.runs)
    gen["fold"] = gen["prompt_id"].map(pfold).astype(int)
    M = member_predictions(tr, y, folds, g_ids, gen)
    for m, (oof, _) in M.items():                      # sanity: the fold logic reproduces the saved members
        saved = load_member(SAVED[m])[0]
        print(f"check {m}: OOF BA {metrics.balanced_accuracy(y, oof):.4f}, max |diff| vs saved {SAVED[m]}: "
              f"{np.abs(oof - saved).max():.2e}", flush=True)
    oof_b = blend({m: v[0] for m, v in M.items()})
    gen_b = blend({m: v[1] for m, v in M.items()})
    print(f"real-text OOF BA of the 3-member CPU blend: {metrics.balanced_accuracy(y, oof_b):.4f}")
    gen["pred"] = np.asarray(CLASSES)[gen_b.argmax(1)]
    gen["p_label"] = gen_b[np.arange(len(gen)), [CLASSES.index(l) for l in gen["label"]]]
    for m, (_, pg) in M.items():
        gen[f"pred_{m}"] = np.asarray(CLASSES)[pg.argmax(1)]
    real_pred = np.asarray(CLASSES)[oof_b.argmax(1)]
    rep = []
    for run, sub in gen.groupby("run", sort=False):
        lab = sub["label"].iloc[0]
        li = CLASSES.index(lab)
        real_rows = (tr["model"] == lab).to_numpy() & tr["prompt_id"].isin(sub["prompt_id"]).to_numpy()
        conf = sub["pred"].value_counts()
        rep.append({"run": run, "label": lab, "n": len(sub),
                    "gen_acc": round(float((sub["pred"] == lab).mean()), 3),
                    "real_recall_same_prompts": round(float((real_pred[real_rows] == lab).mean()), 3),
                    "real_recall_all": round(float((real_pred[tr["model"].to_numpy() == lab] == lab).mean()), 3),
                    "gen_lab_acc": round(float(sub["pred"].map(lab_of).eq(lab_of(lab)).mean()), 3),
                    "real_lab_acc_same_prompts": round(float(pd.Series(real_pred[real_rows]).map(lab_of).eq(lab_of(lab)).mean()), 3),
                    "gen_mean_p_label": round(float(sub["p_label"].mean()), 3),
                    "real_mean_p_label": round(float(oof_b[real_rows, li].mean()), 3),
                    "gen_acc_members": {m: round(float((sub[f"pred_{m}"] == lab).mean()), 3) for m in M},
                    "top_predicted": conf.head(5).to_dict()})
    R = pd.DataFrame(rep)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 200)
    print(R.to_string(index=False))
    gen[["id", "run", "prompt_id", "label", "fold", "pred", "p_label"] + [f"pred_{m}" for m in M]].to_csv(
        F.EXT / f"pred_{a.tag}.csv", index=False)
    R.to_json(F.EXT / f"report_{a.tag}.json", orient="records", indent=1)


if __name__ == "__main__":
    main()

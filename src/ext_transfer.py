"""Transfer tests for the external corpus (no CV model is changed; diagnostics only).

  ext2off   train on external rows only (all, or one source), predict EVERY official train text: BA, per-class
            recall, confusion matrix. Head: char+word+fw n-gram streams (train_ngram.py); vocabulary fitted on the
            official texts without labels (--vocab official, default; the feature space of the CV runs) or on the
            external texts; LogisticRegression C=--C with class-balanced sample weights (every label has the same total
            weight, so recall is not driven by the corpus' label counts).
  off2ext   train on ALL official rows (same streams, LR C=100 = the ng_char-word-fw member settings), predict every
            external row: per source x label accuracy, lab accuracy and mean P(claimed label). A source whose texts
            are recognised as their label far less often than other sources of the same label is suspect.

Outputs (gitignored, derived from competition data): data/external/ntl_corpus/transfer_<name>.json / .csv

  lift      off2ext rows relative to the base rate of each predicted label (claude-opus-5 attracts external texts).

Usage: python ext_transfer.py ext2off [--source speechmap] [--C 30]
       python ext_transfer.py off2ext
       python ext_transfer.py lift
"""
from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from common import CLASSES, K, encode_y, lab_of, load_folds, load_member, load_train, metrics
import ext_corpus
from train_ngram import make_vectorizer, transformed_texts, STREAMS

FEATS = "char+word+fw"
NG = Namespace(no_sublinear=False, min_df=2, binary=False, no_idf=False, max_features=0)


def vectorize(fit_texts: dict, apply: list, hashed: bool = False):  # noqa: C901
    """Same streams as train_ngram.py. hashed=True: HashingVectorizer (2^21 columns) + TfidfTransformer instead of a
    vocabulary (the char vocabulary of 24k texts needs >5 GB before pruning)."""
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    out = [[] for _ in apply]
    for f in FEATS.split("+"):
        t = STREAMS[f][0]
        if hashed:
            kw = {k: v for k, v in STREAMS[f][1].items() if k != "max_features"}
            hv = HashingVectorizer(n_features=2 ** 21, alternate_sign=False, norm=None, dtype=np.float32, **kw)
            tf = TfidfTransformer(sublinear_tf=True).fit(hv.transform(fit_texts[t]))
            for i, d in enumerate(apply):
                out[i].append(tf.transform(hv.transform(d[t])))
            continue
        v = make_vectorizer(f, NG).fit(fit_texts[t])
        for i, d in enumerate(apply):
            out[i].append(v.transform(d[t]))
    return [sp.hstack(b).tocsr() for b in out]


def recall_table(y, pred):
    return {c: round(float((pred[y == i] == i).mean()), 4) for i, c in enumerate(CLASSES) if np.any(y == i)}


def ext2off(a):
    t0 = time.time()
    tr = load_train()
    y = encode_y(tr["model"])
    ext = ext_corpus.load()
    if a.source:
        ext = ext[ext["source"] == a.source].reset_index(drop=True)
    if a.drop_source:
        ext = ext[~ext["source"].isin(a.drop_source.split(";"))].reset_index(drop=True)
    ye = encode_y(ext["label"]) if set(ext["label"]) == set(CLASSES) else np.array([CLASSES.index(l) for l in ext["label"]])
    cnt = np.bincount(ye, minlength=K).astype(float)
    sw = (len(ye) / max((cnt > 0).sum(), 1)) / cnt[ye]                     # equal total weight per present label
    Te = transformed_texts(FEATS, ext["text"].tolist())
    To = transformed_texts(FEATS, tr["text"].tolist())
    fit = To if a.vocab == "official" else Te                              # official: texts only, no labels
    Ae, Ao = vectorize(fit, [Te, To], hashed=a.hashed)
    m = LogisticRegression(C=a.C, max_iter=3000, tol=1e-4).fit(Ae, ye, sample_weight=sw)
    P = np.zeros((len(tr), K))
    P[:, m.classes_] = m.predict_proba(Ao)
    pred = P.argmax(1)
    present = sorted(set(ye))
    rec = recall_table(y, pred)
    cm = pd.crosstab(pd.Series(np.asarray(CLASSES)[y], name="true"), pd.Series(np.asarray(CLASSES)[pred], name="pred"))
    cm = cm.reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    name = (f"ext2off_{a.source.replace('/', '_') if a.source else 'all'}"
            f"{'_drop' + str(len(a.drop_source.split(';'))) if a.drop_source else ''}_C{a.C:g}{'_hash' if a.hashed else ''}"
            f"_v{a.vocab}")
    cm.to_csv(ext_corpus.D / f"transfer_{name}_confusion.csv")
    ba_present = float(np.mean([rec[CLASSES[i]] for i in present]))
    top_conf = {}
    for c in CLASSES:
        row = cm.loc[c].drop(c)
        top_conf[c] = f"{row.idxmax()} {int(row.max())}" if row.max() > 0 else ""
    lab_acc = float(np.mean([lab_of(CLASSES[p]) == lab_of(CLASSES[t]) for p, t in zip(pred, y)]))
    out = {"name": name, "train_rows": int(len(ext)), "labels_in_training": [CLASSES[i] for i in present],
           "ba_all18": round(metrics.balanced_accuracy(y, P), 4), "ba_present_labels": round(ba_present, 4),
           "logloss": round(metrics.mlogloss(y, np.clip(P, 1e-9, 1)), 4), "lab_accuracy": round(lab_acc, 4),
           "recall": rec, "top_confusion": top_conf,
           "pred_share": {c: round(float((pred == i).mean()), 4) for i, c in enumerate(CLASSES)},
           "seconds": round(time.time() - t0)}
    (ext_corpus.D / f"transfer_{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("recall", "top_confusion", "pred_share")}))
    print(pd.DataFrame({"recall": rec, "top_confusion": top_conf, "pred_share": out["pred_share"]}).to_string())


def off2ext(a):
    t0 = time.time()
    tr = load_train()
    y = encode_y(tr["model"])
    ext = ext_corpus.load()
    To = transformed_texts(FEATS, tr["text"].tolist())
    Te = transformed_texts(FEATS, ext["text"].tolist())
    Ao, Ae = vectorize(To, [To, Te])
    m = LogisticRegression(C=100.0, max_iter=3000, tol=1e-4).fit(Ao, y)
    P = m.predict_proba(Ae)
    li = np.array([CLASSES.index(l) for l in ext["label"]])
    ext["pred"] = np.asarray(CLASSES)[P.argmax(1)]
    ext["hit"] = ext["pred"] == ext["label"]
    ext["lab_hit"] = ext["pred"].map(lab_of) == ext["label"].map(lab_of)
    ext["p_label"] = P[np.arange(len(ext)), li]
    oof = load_member("ng_char-word-fw_lr_C100_T")[0]                      # official OOF recall per label (yardstick)
    rec_off = recall_table(y, oof.argmax(1))
    g = ext.groupby(["source", "label"]).agg(n=("hit", "size"), acc=("hit", "mean"), lab_acc=("lab_hit", "mean"),
                                             mean_p=("p_label", "mean")).reset_index()
    g["official_oof_recall"] = g["label"].map(rec_off)
    lab = ext.groupby("label").agg(n=("hit", "size"), acc=("hit", "mean"), lab_acc=("lab_hit", "mean"),
                                   mean_p=("p_label", "mean")).reset_index()
    lab["official_oof_recall"] = lab["label"].map(rec_off)
    src = ext.groupby("source").agg(n=("hit", "size"), acc=("hit", "mean"), lab_acc=("lab_hit", "mean"),
                                    mean_p=("p_label", "mean"), labels=("label", "nunique")).reset_index()
    for d in (g, lab, src):
        for c in ("acc", "lab_acc", "mean_p"):
            d[c] = d[c].round(3)
    g.to_csv(ext_corpus.D / "transfer_off2ext_source_label.csv", index=False)
    ext[["ext_id", "source", "label", "pred", "p_label"]].to_csv(ext_corpus.D / "transfer_off2ext_rows.csv", index=False)
    out = {"by_label": lab.to_dict("records"), "by_source": src.sort_values("n", ascending=False).to_dict("records"),
           "chance": round(1 / K, 4), "seconds": round(time.time() - t0)}
    (ext_corpus.D / "transfer_off2ext.json").write_text(json.dumps(out, indent=1))
    pd.set_option("display.width", 250)
    print(lab.to_string(index=False))
    print(src.sort_values("n", ascending=False).to_string(index=False))
    print(g[g.n >= 30].sort_values(["label", "acc"]).to_string(index=False))


def lift(a):
    """Recognition relative to the base rate: the official model predicts some labels (claude-opus-5) for a large share
    of ALL external texts, so P(pred = label | claimed label) / P(pred = label) is the fairer reliability signal
    (1 = no better than an arbitrary external text; < 1 = the texts look less like that label than average)."""
    r = pd.read_csv(ext_corpus.D / "transfer_off2ext_rows.csv")
    share = r["pred"].value_counts(normalize=True)
    lab_share = r["pred"].map(lab_of).value_counts(normalize=True)
    r["hit"] = r["pred"] == r["label"]
    r["lab_hit"] = r["pred"].map(lab_of) == r["label"].map(lab_of)
    rows = []
    for (src, lab), g in [(("ALL", l), g) for l, g in r.groupby("label")] + list(r.groupby(["source", "label"])):
        if len(g) < 30:
            continue
        rows.append({"source": src, "label": lab, "n": len(g), "acc": round(g.hit.mean(), 3),
                     "pred_share_label": round(float(share.get(lab, 0)), 3),
                     "lift": round(g.hit.mean() / max(share.get(lab, 0), 1e-9), 2),
                     "lab_acc": round(g.lab_hit.mean(), 3),
                     "lab_lift": round(g.lab_hit.mean() / max(lab_share.get(lab_of(lab), 0), 1e-9), 2),
                     "top_pred": g["pred"].value_counts().index[0]})
    t = pd.DataFrame(rows)
    t.to_csv(ext_corpus.D / "transfer_off2ext_lift.csv", index=False)
    pd.set_option("display.width", 250)
    print("pred share over all external rows:", share.round(3).head(6).to_dict())
    print(t[t.source == "ALL"].to_string(index=False))
    print(t[t.source != "ALL"].sort_values(["label", "lift"]).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ext2off", "off2ext", "lift"])
    ap.add_argument("--source", default="")
    ap.add_argument("--drop-source", default="")
    ap.add_argument("--C", type=float, default=30.0)
    ap.add_argument("--hashed", action="store_true", help="hashed n-gram features (low memory)")
    ap.add_argument("--vocab", default="official", choices=["official", "external"],
                    help="ext2off vocabulary: fitted on the official texts (no labels; same feature space as the CV "
                         "runs, low memory) or on the external training texts (~6 GB)")
    a = ap.parse_args()
    {"ext2off": ext2off, "off2ext": off2ext, "lift": lift}[a.cmd](a)


if __name__ == "__main__":
    main()

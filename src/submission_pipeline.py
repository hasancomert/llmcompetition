"""End-to-end submission (runs offline inside the Kaggle notebook, and locally for testing).

Members = the current best nested blend (EXPERIMENTS.md #38 blend_v2_tok_ba):
  A  char+word TF-IDF + LogReg C=30          (train_tfidf.py --feats char+word --model lr --C 30)
  B  style+tok v2 features + QuantileTransformer + LogReg C=0.1   (train_gbdt.py --model lr)
Both are fitted on the 5 shared folds (folds.csv from the private dataset, not regenerated: the Kaggle image has a
different scikit-learn); test probabilities are fold averages; blend = log-probability weights A 2/3, B 1/3.
Then the test texts are clustered into prompts and assigned with the soft one-model-per-prompt Hungarian step
(matching.py, lambda 2, threshold 0.92), skipped for tiny tests (the 1-3 row interactive test).
The in-kernel OOF scores are printed so that the Kaggle environment can be compared with the local CV.

Usage: python submission_pipeline.py --data-dir D --folds F --tokenizers-flat T --out submission.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

W_TFIDF, W_STYLE = 2 / 3, 1 / 3
MATCH_MIN_ROWS = 40
MATCH_LAMBDA = 2.0
MATCH_THRESHOLD = 0.92   # matching.cluster_prompts v2 (raw TF-IDF, max_df 40 docs), scale study + nested eval


def materialize_tokenizers(flat_dir: Path, out_dir: Path) -> Path:
    """The dataset stores <folder>__<file>; rebuild <folder>/<file> with symlinks (no copies)."""
    for f in flat_dir.iterdir():
        if "__" in f.name:
            sub, name = f.name.split("__", 1)
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
            dst = out_dir / sub / name
            if not dst.exists():
                os.symlink(f.resolve(), dst)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--tokenizers-flat", required=True)
    ap.add_argument("--out", default="submission.parquet")
    ap.add_argument("--fake-test", type=int, default=0, help="LOCAL TIMING ONLY: use N train texts as the test")
    args = ap.parse_args()
    t0 = time.time()
    tok_dir = materialize_tokenizers(Path(args.tokenizers_flat), Path("/tmp/nllm_tokenizers"))
    os.environ["TOKENIZER_DIR"] = str(tok_dir)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from common import CLASSES, K, encode_y, metrics
    from matching import cluster_prompts, match
    from train_gbdt import fit_fold_lr
    from style_features import style_frame
    from tokenizer_features import tokenizer_frame
    from train_tfidf import vectorize
    from sklearn.linear_model import LogisticRegression
    import submission as sub_check  # reuse/airline/submission.py

    d = Path(args.data_dir)
    tr = pd.read_csv(d / "train.csv")
    te = pd.read_csv(d / "test.csv", dtype={"id": str})
    te["text"] = te["text"].fillna("").astype(str)
    if args.fake_test:
        te = tr.sample(args.fake_test, replace=args.fake_test > len(tr), random_state=0)[["id", "text"]].reset_index(drop=True)
        te["id"] = [f"fake{i}" for i in range(len(te))]
    fdf = pd.read_csv(args.folds)
    assert (fdf["id"].values == tr["id"].values).all(), "folds.csv does not match train.csv row order"
    folds = fdf["fold"].to_numpy()
    y = encode_y(tr["model"])
    print(f"train {len(tr)}, test {len(te)}, folds {np.bincount(folds).tolist()}  ({time.time() - t0:.0f}s)", flush=True)

    # style + tokenizer features (pure per-text functions)
    def feats(texts):
        F = pd.concat([style_frame(texts), tokenizer_frame(texts)], axis=1)
        return F
    Ftr, Fte = feats(tr["text"].tolist()), feats(te["text"].tolist())
    keep = [c for c in Ftr.columns if Ftr[c].nunique() > 1]
    Xtr, Xte = Ftr[keep].to_numpy(np.float32), Fte[keep].to_numpy(np.float32)
    print(f"features {len(keep)} cols  ({time.time() - t0:.0f}s)", flush=True)

    n_te = len(te)
    oof_a, oof_b = np.zeros((len(tr), K)), np.zeros((len(tr), K))
    te_a, te_b = np.zeros((n_te, K)), np.zeros((n_te, K))
    texts = tr["text"].tolist()
    for k in range(int(folds.max()) + 1):
        trn, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        A_tr, A_va, A_te = vectorize("char+word", [texts[i] for i in trn],
                                     [[texts[i] for i in trn], [texts[i] for i in val], te["text"].tolist()])
        m = LogisticRegression(C=30.0, max_iter=3000, tol=1e-4).fit(A_tr, y[trn])
        oof_a[val], p = m.predict_proba(A_va), m.predict_proba(A_te)
        te_a += p / 5
        oof_b[val], p = fit_fold_lr(Xtr[trn], y[trn], [Xtr[val], Xte], 0.1)
        te_b += p / 5
        print(f"fold {k}: A {metrics.balanced_accuracy(y[val], oof_a[val]):.4f}  "
              f"B {metrics.balanced_accuracy(y[val], oof_b[val]):.4f}  ({time.time() - t0:.0f}s)", flush=True)

    def blend(pa, pb):
        z = W_TFIDF * np.log(np.clip(pa, 1e-9, 1)) + W_STYLE * np.log(np.clip(pb, 1e-9, 1))
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)
    oof, P = blend(oof_a, oof_b), blend(te_a, te_b)
    print(f"in-kernel OOF BA: A {metrics.balanced_accuracy(y, oof_a):.4f}, B {metrics.balanced_accuracy(y, oof_b):.4f}, "
          f"blend {metrics.balanced_accuracy(y, oof):.4f} (local CV: 0.3471 / 0.2993; blend #38 0.3923 nested)", flush=True)
    try:  # diagnostic only
        oof_match = match(oof, cluster_prompts(texts, MATCH_THRESHOLD), MATCH_LAMBDA)
        print(f"in-kernel OOF BA after clustering + matching (lambda {MATCH_LAMBDA}): "
              f"{metrics.balanced_accuracy(y, np.eye(K)[oof_match]):.4f} (full-structure upper bound: every train prompt "
              f"has all 18 models; test-like drop simulations give +0.03 to +0.05, EXPERIMENTS #43/#44)", flush=True)
    except Exception as e:
        print(f"OOF matching diagnostic failed: {type(e).__name__}: {e}", flush=True)

    if n_te >= MATCH_MIN_ROWS:
        try:
            groups = cluster_prompts(te["text"].tolist(), MATCH_THRESHOLD)
            pred = match(P, groups, MATCH_LAMBDA)
            sizes = np.bincount(groups)
            print(f"test clusters {len(sizes)} for {n_te} texts (sizes: median {np.median(sizes):.0f}, max {sizes.max()}, "
                  f"singletons {(sizes == 1).sum()}); labels changed by matching {(pred != P.argmax(1)).mean():.3f}",
                  flush=True)
        except Exception as e:  # never lose the submission to the optional matching step
            pred = P.argmax(1)
            print(f"matching failed ({type(e).__name__}: {e}); falling back to argmax", flush=True)
    else:
        pred = P.argmax(1)
        print(f"test has {n_te} rows (< {MATCH_MIN_ROWS}): clustering/matching skipped", flush=True)

    labels = np.asarray(CLASSES)[pred]
    sample = None
    sp = d / "sample_submission.parquet"
    if sp.exists() and not args.fake_test:
        ss = pd.read_parquet(sp)
        if set(ss["id"]) == set(te["id"]):
            sample = ss
        else:
            print("sample_submission ids differ from test ids: writing in test order", flush=True)
    counts = sub_check.write_submission(args.out, te["id"].values, labels, sample=sample, id_col="id",
                                        target_col="model", allowed=CLASSES,
                                        min_classes=1 if n_te < MATCH_MIN_ROWS else 2)
    out = pd.read_parquet(args.out)
    assert list(out.columns) == ["id", "model"] and len(out) == n_te and out["id"].is_unique
    assert out["model"].notna().all() and set(out["model"]) <= set(CLASSES) and set(out["id"]) == set(te["id"])
    print(f"wrote {args.out}: {len(out)} rows, {out['model'].nunique()} labels; total {time.time() - t0:.0f}s", flush=True)
    print(counts.to_string(), flush=True)


if __name__ == "__main__":
    main()

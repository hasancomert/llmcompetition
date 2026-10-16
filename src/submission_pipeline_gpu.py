"""GPU submission (v2): members of the nested blend EXPERIMENTS.md blend_v3_emb_logloss + prompt matching.

Members (all fitted on the 5 shared folds; test = fold average):
  svc   char TF-IDF + LinearSVC C=0.3 + inner-fold softmax temperature   (train_tfidf.py --feats char --model svc)
  lp    Qwen3 0.6B/1.7B/4B/8B token log-prob stats + style+tok v2 -> QuantileTransformer + LogReg C=0.1
  emb4  Qwen3-4B-base pooled hidden states -> StandardScaler + LogReg C=C_EMB
  emb8  Qwen3-8B-base pooled hidden states -> StandardScaler + LogReg C=C_EMB
LM features of train AND test are extracted here with src/gpu_embed.py (same code as the CV features), one
subprocess per model. Blend = weighted log-probabilities; then clustering + soft Hungarian (matching.py).
If any GPU step fails, the notebook falls back to the CPU-only v1 members (char+word LR + style+tok LR) so that a
submission is always written.

Usage: python submission_pipeline_gpu.py --data-dir D --folds F --tokenizers-flat T --out submission.parquet
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LM_MODELS = [("qwen3_0.6b_base", "0.6b-base"), ("qwen3_1.7b_base", "1.7b-base"),
             ("qwen3_4b_base", "4b-base"), ("qwen3_8b_base", "8b-base")]
WEIGHTS = {"svc": 0.257, "lp": 0.257, "emb4": 0.229, "emb8": 0.257}   # blend_v3_emb_logloss final weights
C_EMB, C_LP = 0.003, 0.1
MATCH_MIN_ROWS, MATCH_LAMBDA, MATCH_THRESHOLD = 40, 4.0, 0.92
V1_WEIGHTS = {"charword": 2 / 3, "style": 1 / 3}


def log(msg, t0):
    print(f"{msg}  ({time.time() - t0:.0f}s)", flush=True)


def blend(members, weights):
    z = sum(w * np.log(np.clip(members[k], 1e-9, 1)) for k, w in weights.items()) / sum(weights.values())
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def extract_lm_features(tr, te, work: Path, t0):
    """Run gpu_embed.py for every LM on clean train/test csv copies; returns {name: (emb_tr, emb_te, lp_tr, lp_te)}."""
    work.mkdir(parents=True, exist_ok=True)
    tr[["id", "text"]].to_csv(work / "train.csv", index=False)
    te[["id", "text"]].to_csv(work / "test.csv", index=False)
    here = Path(__file__).resolve().parent
    out = {}
    for name, hint in LM_MODELS:
        cfg = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if f"/{hint}/" in p and "qwen" in p]
        if not cfg:
            raise FileNotFoundError(f"model {hint} not attached")
        r = subprocess.run([sys.executable, str(here / "gpu_embed.py"), "--model-dir", os.path.dirname(cfg[0]),
                            "--name", name, "--texts", str(work / "train.csv"), "--texts", str(work / "test.csv"),
                            "--out", str(work / "emb")])
        if r.returncode != 0:
            raise RuntimeError(f"gpu_embed failed for {name} (exit {r.returncode})")
        f = lambda kind, split: np.load(work / "emb" / f"{kind}_{name}_{split}.npy").astype(np.float32)
        out[name] = (f("emb", "train"), f("emb", "test"), f("lp", "train"), f("lp", "test"))
        log(f"LM features {name}: emb {out[name][0].shape}, lp {out[name][2].shape}", t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--tokenizers-flat", required=True)
    ap.add_argument("--out", default="submission.parquet")
    ap.add_argument("--work", default="/kaggle/temp/nllm")
    ap.add_argument("--cpu-only", action="store_true", help="skip the GPU members (fallback path test)")
    ap.add_argument("--precomputed-lm", default="", help="LOCAL TEST: load emb/lp npy from this dir instead of the GPU")
    args = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from submission_pipeline import materialize_tokenizers
    tok_dir = materialize_tokenizers(Path(args.tokenizers_flat), Path("/tmp/nllm_tokenizers"))
    os.environ["TOKENIZER_DIR"] = str(tok_dir)

    from common import CLASSES, K, cv_preds, encode_y, metrics
    from matching import cluster_prompts, match
    from train_emb import fold_fit
    from train_gbdt import build
    import train_tfidf
    import submission as sub_check  # reuse/airline/submission.py

    d = Path(args.data_dir)
    tr = pd.read_csv(d / "train.csv")
    te = pd.read_csv(d / "test.csv", dtype={"id": str})
    te["text"] = te["text"].fillna("").astype(str)
    fdf = pd.read_csv(args.folds)
    assert (fdf["id"].values == tr["id"].values).all(), "folds.csv does not match train.csv row order"
    folds = fdf["fold"].to_numpy()
    y = encode_y(tr["model"])
    n_te = len(te)
    log(f"train {len(tr)}, test {n_te}", t0)

    S_tr, S_te, _ = build("style+tok", tr, te)             # style + tokenizer v2 features (pure per-text)
    log(f"style+tok features {S_tr.shape[1]} cols", t0)

    members_oof, members_te = {}, {}
    gpu_ok = False
    if not args.cpu_only:
        try:
            if args.precomputed_lm:
                pdir = Path(args.precomputed_lm)
                g = lambda kind, n, split: np.load(pdir / f"{kind}_{n}_{split}.npy").astype(np.float32)
                F = {n: (g("emb", n, "train"), g("emb", n, "test"), g("lp", n, "train"), g("lp", n, "test"))
                     for n, _ in LM_MODELS}
            else:
                F = extract_lm_features(tr, te, Path(args.work), t0)
            lp_tr = np.hstack([F[n][2] for n, _ in LM_MODELS] + [S_tr])
            lp_te = np.hstack([F[n][3] for n, _ in LM_MODELS] + [S_te])
            members_oof["lp"], members_te["lp"] = fold_fit("lp", C_LP, lp_tr, lp_te, y, folds)
            log(f"member lp: OOF BA {metrics.balanced_accuracy(y, members_oof['lp']):.4f} (local CV 0.3279)", t0)
            for key, name, ref in (("emb4", "qwen3_4b_base", "0.3956"), ("emb8", "qwen3_8b_base", "0.4008")):
                members_oof[key], members_te[key] = fold_fit("emb", C_EMB, F[name][0], F[name][1], y, folds)
                log(f"member {key}: OOF BA {metrics.balanced_accuracy(y, members_oof[key]):.4f} (local CV {ref})", t0)
            gpu_ok = True
        except Exception as e:  # never lose the submission to the GPU path
            print(f"GPU members failed ({type(e).__name__}: {e}); falling back to the CPU v1 blend", flush=True)

    texts = tr["text"]
    if gpu_ok:
        svc_args = argparse.Namespace(feats="char", model="svc", C=0.3, transductive=False)
        fp = train_tfidf.make_fit_predict(svc_args, tr["prompt_id"].to_numpy())
        members_oof["svc"], members_te["svc"] = cv_preds.cv_predict(fp, texts, y, te["text"], folds, K)[:2]
        log(f"member svc: OOF BA {metrics.balanced_accuracy(y, members_oof['svc']):.4f} (local CV 0.3249)", t0)
        weights = WEIGHTS
    else:
        cw_args = argparse.Namespace(feats="char+word", model="lr", C=30.0, transductive=False)
        fp = train_tfidf.make_fit_predict(cw_args, tr["prompt_id"].to_numpy())
        members_oof["charword"], members_te["charword"] = cv_preds.cv_predict(fp, texts, y, te["text"], folds, K)[:2]
        members_oof["style"], members_te["style"] = fold_fit("lp", C_LP, S_tr, S_te, y, folds)
        weights = V1_WEIGHTS
    oof = blend(members_oof, weights)
    P = blend(members_te, weights)
    log(f"in-kernel OOF BA of the blend ({'v3 GPU' if gpu_ok else 'v1 CPU fallback'}): "
        f"{metrics.balanced_accuracy(y, oof):.4f} (local nested CV {'0.4430' if gpu_ok else '0.3923'})", t0)
    try:  # diagnostic only
        om = match(oof, cluster_prompts(texts.tolist(), MATCH_THRESHOLD), MATCH_LAMBDA)
        log(f"in-kernel OOF BA after clustering + matching (lambda {MATCH_LAMBDA}): "
            f"{metrics.balanced_accuracy(y, np.eye(K)[om]):.4f} (full-structure upper bound; test-like drop "
            f"simulations: EXPERIMENTS.md match_* rows)", t0)
    except Exception as e:
        print(f"OOF matching diagnostic failed: {type(e).__name__}: {e}", flush=True)

    if n_te >= MATCH_MIN_ROWS:
        try:
            groups = cluster_prompts(te["text"].tolist(), MATCH_THRESHOLD)
            pred = match(P, groups, MATCH_LAMBDA)
            sizes = np.bincount(groups)
            log(f"test clusters {len(sizes)} for {n_te} texts (median {np.median(sizes):.0f}, max {sizes.max()}, "
                f"singletons {(sizes == 1).sum()}); labels changed by matching {(pred != P.argmax(1)).mean():.3f}", t0)
        except Exception as e:
            pred = P.argmax(1)
            print(f"matching failed ({type(e).__name__}: {e}); falling back to argmax", flush=True)
    else:
        pred = P.argmax(1)
        print(f"test has {n_te} rows (< {MATCH_MIN_ROWS}): clustering/matching skipped", flush=True)

    labels = np.asarray(CLASSES)[pred]
    sample = None
    sp = d / "sample_submission.parquet"
    if sp.exists():
        ss = pd.read_parquet(sp)
        if set(ss["id"].astype(str)) == set(te["id"]):
            sample = ss.assign(id=ss["id"].astype(str))
    counts = sub_check.write_submission(args.out, te["id"].values, labels, sample=sample, id_col="id",
                                        target_col="model", allowed=CLASSES,
                                        min_classes=1 if n_te < MATCH_MIN_ROWS else 2)
    out = pd.read_parquet(args.out)
    assert list(out.columns) == ["id", "model"] and len(out) == n_te and out["id"].is_unique
    assert out["model"].notna().all() and set(out["model"]) <= set(CLASSES) and set(out["id"]) == set(te["id"])
    log(f"wrote {args.out}: {len(out)} rows, {out['model'].nunique()} labels", t0)
    print(counts.to_string(), flush=True)


if __name__ == "__main__":
    main()

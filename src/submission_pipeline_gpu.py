"""GPU submission (v3): prompt-centered members + nested blend + prompt clustering + soft matching.

Members (all fitted on the 5 shared folds; test = fold average). "Centered" = every text's feature row minus the
mean row of its unsupervised prompt cluster (matching.cluster_prompts, thr 0.92, clusters of < 3 texts unchanged):
train texts are clustered together, test texts ON THEIR OWN; no labels or prompt ids are used for centering.
  tfc    char TF-IDF (50k cols) centered + LinearSVC C=0.3 + inner-fold temperature  (train_tfidf_ctr.py)
  lpc    Qwen3 0.6B/1.7B/4B/8B token log-prob stats + style+tok v2, centered -> Quantile + LogReg C=0.1
  emb4c  Qwen3-4B-base pooled hidden states, centered -> StandardScaler + LogReg C=0.001
  emb8c  Qwen3-8B-base pooled hidden states, centered -> StandardScaler + LogReg C=0.001
LM features of train AND test are extracted here with src/gpu_embed.py (same code as the CV features), one
subprocess per model. Blend = weighted log-probabilities (nested hill-climb weights); then clustering + soft
Hungarian (matching.py). If any GPU step fails, the notebook falls back to the CPU-only v1 members (char+word LR +
style+tok LR) so that a submission is always written.

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
WEIGHTS = {"tfc": 0.167, "lpc": 0.167, "emb4c": 0.167, "emb8c": 0.5}   # nested logloss hill-climb (EXPERIMENTS.md)
C_EMB, C_LP = 0.001, 0.1
TFC = argparse.Namespace(feats="char", max_features=50_000, thr=0.92, C=0.3, model="svc")
CENTER_THRESHOLD = 0.92
MATCH_MIN_ROWS, MATCH_LAMBDA, MATCH_THRESHOLD = 40, 2.0, 0.92
V1_WEIGHTS = {"charword": 2 / 3, "style": 1 / 3}
CTR_CPU_WEIGHTS = {"tfc": 0.5, "stylec": 0.5}   # nested logloss hill-climb of tfc + style+tok_ctr: CV 0.4675


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
    ap.add_argument("--members", default="v3", choices=["v3", "ctr-cpu"],
                    help="v3: tfc+lpc+emb4c+emb8c (GPU); ctr-cpu: tfc + centered style+tok only (no LM, CPU)")
    ap.add_argument("--match-lambda", type=float, default=MATCH_LAMBDA, help="0 = plain argmax (no matching)")
    ap.add_argument("--precomputed-lm", default="", help="LOCAL TEST: load emb/lp npy from this dir instead of the GPU")
    args = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from submission_pipeline import materialize_tokenizers
    tok_dir = materialize_tokenizers(Path(args.tokenizers_flat), Path("/tmp/nllm_tokenizers"))
    os.environ["TOKENIZER_DIR"] = str(tok_dir)

    from common import CLASSES, K, cv_preds, encode_y, metrics
    from matching import cluster_prompts, match
    from train_emb import center_groups, fold_fit
    from train_gbdt import build
    import train_tfidf
    import train_tfidf_ctr
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
    texts = tr["text"]
    if not args.cpu_only:
        try:
            g_tr = cluster_prompts(texts.tolist(), CENTER_THRESHOLD)       # unsupervised prompt clusters
            if n_te >= MATCH_MIN_ROWS:
                g_te = cluster_prompts(te["text"].tolist(), CENTER_THRESHOLD)
                ctr_te = lambda X: center_groups(X, g_te)
            else:                                                       # too few test rows to cluster
                ctr_te = lambda X: X - X.mean(0)
            ctr_tr = lambda X: center_groups(X, g_tr)
            log(f"centering clusters: train {len(np.unique(g_tr))}"
                + (f", test {len(np.unique(g_te))}" if n_te >= MATCH_MIN_ROWS else ", test: global mean"), t0)
            if args.members == "ctr-cpu":                                # ablation: no LM features
                members_oof["stylec"], members_te["stylec"] = fold_fit("lp", C_LP, ctr_tr(S_tr), ctr_te(S_te), y,
                                                                       folds)
                log(f"member stylec: OOF BA {metrics.balanced_accuracy(y, members_oof['stylec']):.4f} "
                    f"(local CV 0.3782)", t0)
                F = None
            elif args.precomputed_lm:
                pdir = Path(args.precomputed_lm)
                g = lambda kind, n, split: np.load(pdir / f"{kind}_{n}_{split}.npy").astype(np.float32)
                F = {n: (g("emb", n, "train"), g("emb", n, "test"), g("lp", n, "train"), g("lp", n, "test"))
                     for n, _ in LM_MODELS}
            else:
                F = extract_lm_features(tr, te, Path(args.work), t0)
            if F is not None:
                lp_tr = ctr_tr(np.hstack([F[n][2] for n, _ in LM_MODELS] + [S_tr]).astype(np.float32))
                lp_te = ctr_te(np.hstack([F[n][3] for n, _ in LM_MODELS] + [S_te]).astype(np.float32))
                members_oof["lpc"], members_te["lpc"] = fold_fit("lp", C_LP, lp_tr, lp_te, y, folds)
                log(f"member lpc: OOF BA {metrics.balanced_accuracy(y, members_oof['lpc']):.4f} (local CV 0.4016)",
                    t0)
                for key, name, ref in (("emb4c", "qwen3_4b_base", "0.5498"), ("emb8c", "qwen3_8b_base", "0.5583")):
                    members_oof[key], members_te[key] = fold_fit("emb", C_EMB, ctr_tr(F[name][0]),
                                                                 ctr_te(F[name][1]), y, folds)
                    log(f"member {key}: OOF BA {metrics.balanced_accuracy(y, members_oof[key]):.4f} "
                        f"(local CV {ref})", t0)
            del F
            fp = train_tfidf_ctr.make_fit_predict(TFC, tr["prompt_id"].to_numpy())
            members_oof["tfc"], members_te["tfc"] = cv_preds.cv_predict(fp, texts, y, te["text"], folds, K)[:2]
            log(f"member tfc: OOF BA {metrics.balanced_accuracy(y, members_oof['tfc']):.4f} (local CV 0.3956)", t0)
            gpu_ok = True
        except Exception as e:  # never lose the submission to the GPU path
            import traceback
            traceback.print_exc()
            print(f"GPU members failed ({type(e).__name__}: {e}); falling back to the CPU v1 blend", flush=True)
            members_oof, members_te = {}, {}

    if gpu_ok:
        weights = WEIGHTS if args.members == "v3" else CTR_CPU_WEIGHTS
    else:
        cw_args = argparse.Namespace(feats="char+word", model="lr", C=30.0, transductive=False)
        fp = train_tfidf.make_fit_predict(cw_args, tr["prompt_id"].to_numpy())
        members_oof["charword"], members_te["charword"] = cv_preds.cv_predict(fp, texts, y, te["text"], folds, K)[:2]
        members_oof["style"], members_te["style"] = fold_fit("lp", C_LP, S_tr, S_te, y, folds)
        weights = V1_WEIGHTS
    oof = blend(members_oof, weights)
    P = blend(members_te, weights)
    ref = {"v3": "0.5931", "ctr-cpu": "0.4675"}[args.members] if gpu_ok else "0.3923"
    log(f"in-kernel OOF BA of the blend ({args.members if gpu_ok else 'v1 CPU fallback'}): "
        f"{metrics.balanced_accuracy(y, oof):.4f} (local nested CV {ref})", t0)
    try:  # diagnostic only
        om = match(oof, cluster_prompts(texts.tolist(), MATCH_THRESHOLD), args.match_lambda)
        log(f"in-kernel OOF BA after clustering + matching (lambda {args.match_lambda:g}): "
            f"{metrics.balanced_accuracy(y, np.eye(K)[om]):.4f} (full-structure upper bound; test-like drop "
            f"simulations: EXPERIMENTS.md match_* rows)", t0)
    except Exception as e:
        print(f"OOF matching diagnostic failed: {type(e).__name__}: {e}", flush=True)

    if args.match_lambda <= 0:
        pred = P.argmax(1)
        print("match lambda 0: plain argmax (matching skipped)", flush=True)
    elif n_te >= MATCH_MIN_ROWS:
        try:
            groups = cluster_prompts(te["text"].tolist(), MATCH_THRESHOLD)
            pred = match(P, groups, args.match_lambda)
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

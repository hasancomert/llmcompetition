"""LoRA24 member and twin evidence on CPU, from the saved features of the trained jobs (gpu_lora24.py, lora24_infer.py).

Member (--member TAG): the OOF row of a held-out text is the plain readout of its fold job, softmax(Z W^T + b) on the
job's valid_features; the test row is the mean over the fold jobs of the plain readout on that job's test features.
Saved with common.save_member (OOF + test, class order checked).

Twin evidence (--twins TAG, replay): test texts in topic clusters of min..max texts (matching.cluster_prompts, cut
--thr; fewer than 40 test texts: all singletons) get, per fold job, the centered readout on features centered by
their cluster; evidence = log of the fold-mean probability, 0 elsewhere (outputs/twins/<TAG>.npz, the format of
adaptive_twins.py, so replay.py adds weight * evidence).

Usage: python lora24_member.py --jobs-dir lora24 --features-dir outputs/lora24 --member lora24_plain
       python lora24_member.py --jobs-dir lora24 --features-dir outputs/lora24 --twins twins_lora24 [--thr 0.85]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import CLASSES, K, OUT, encode_y, load_folds, load_test, load_train, metrics, run_member

FOLD_JOBS = [f"f{k}" for k in range(5)]


def readout(Z, ro, kind):
    """Probabilities of a saved linear head ('plain' or 'center') on features Z (rows x dims)."""
    assert list(ro["classes"]) == CLASSES, "class order of the readout differs"
    lg = Z.astype(np.float32) @ ro[f"{kind}_coef"].T + ro[f"{kind}_intercept"]
    lg -= lg.max(1, keepdims=True)
    P = np.exp(lg)
    return P / P.sum(1, keepdims=True)


def group_center(Z, groups):
    """Each row minus the mean of its group's rows."""
    Zc = Z.astype(np.float32).copy()
    for g in np.unique(groups):
        ix = groups == g
        Zc[ix] -= Zc[ix].mean(0)
    return Zc


def load_job(jobs_dir, job):
    d = Path(jobs_dir) / job
    assert (d / "COMPLETE.json").exists(), f"{d}: job not complete"
    return np.load(d / "readouts.npz"), d


def build_member(a, tr, te, y, folds, t0):
    """--oof-only (CV before any test inference): test rows are uniform placeholders; the tag must end in _oofonly
    so that such a member can never be mistaken for a submission member."""
    assert not a.oof_only or a.member.endswith("_oofonly"), "--oof-only members must be tagged *_oofonly"
    oof = np.zeros((len(tr), K))
    test = np.full((len(te), K), 1.0 / K) if a.oof_only else np.zeros((len(te), K))
    for k, job in enumerate(FOLD_JOBS):
        ro, d = load_job(a.jobs_dir, job)
        idx = np.load(d / "valid_idx.npy")
        assert (np.sort(idx) == np.flatnonzero(folds == k)).all(), f"{job}: held-out rows are not fold {k}"
        oof[idx] = readout(np.load(d / "valid_features.npy"), ro, "plain")
        if a.oof_only:
            continue
        meta = json.loads((Path(a.features_dir) / f"test_meta_{job}.json").read_text())
        assert meta["ids"] == te["id"].tolist(), f"{job}: test features not in test.csv order"
        test += readout(np.load(Path(a.features_dir) / f"test_features_{job}.npy"), ro, "plain") / len(FOLD_JOBS)
    lg = json.loads((Path(a.jobs_dir) / "f0" / "log.json").read_text())
    pre = Path(a.jobs_dir) / "pre" / "log.json"
    stage1 = ""
    if pre.exists():
        lp = json.loads(pre.read_text())
        stage1 = (f"two-stage: stage 1 on {lp['fit_rows']} public pool rows, {lp['updates_done']}/{lp['updates']} "
                  f"updates, class-weighted; ")
    run_member(a.member, f"LoRA24 (Qwen3 base truncated to {lg['n_layers']} layers, LoRA r16 on 7 projections, "
               f"pooled layers {lg['pool_layers']} + LayerNorm, plain + prompt-centered heads on same-prompt triples; "
               f"{stage1}folds {lg['epochs']} epochs at lr {lg['lr']:g} / {lg['head_lr']:g}; gpu_lora24.py): plain "
               f"readout{' (OOF ONLY: test rows uniform placeholders)' if a.oof_only else ''}",
               oof, test, y, folds, t0, note=a.note)


def build_twins(a, te, t0):
    from matching import cluster_prompts
    g = cluster_prompts(te["text"].tolist(), a.thr)
    sizes = np.bincount(g)[g]
    adapt = (sizes >= a.min_size) & (sizes <= a.max_size)
    rows = np.flatnonzero(adapt)
    ev = np.zeros((len(te), K))
    if len(rows):
        P = np.zeros((len(rows), K))
        for job in FOLD_JOBS:
            ro, _ = load_job(a.jobs_dir, job)
            Z = np.load(Path(a.features_dir) / f"test_features_{job}.npy")[rows]
            P += readout(group_center(Z, g[rows]), ro, "center") / len(FOLD_JOBS)
        ev[rows] = np.log(np.clip(P, 1e-9, 1))
    meta = {"views": "lora24", "thr": a.thr, "min_size": a.min_size, "max_size": a.max_size, "n_test": len(te),
            "n_adapted": int(len(rows)), "n_clusters": int(len(np.unique(g)))}
    (OUT / "twins").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "twins" / f"{a.twins}.npz", evidence=ev, adapt=adapt, clusters=g, meta=json.dumps(meta))
    print(f"[lora24 twins] {meta} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-dir", required=True, help="directory with f0..f4/ of the training kernel")
    ap.add_argument("--features-dir", default=str(OUT / "lora24"), help="test_features_<job>.npy (lora24_infer.py)")
    ap.add_argument("--member", default="", help="save the plain-readout member under this tag")
    ap.add_argument("--oof-only", action="store_true", help="CV only: no test features yet (tag must end in _oofonly)")
    ap.add_argument("--note", default="")
    ap.add_argument("--twins", default="", help="write test twin evidence outputs/twins/<tag>.npz")
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=18)
    a = ap.parse_args()
    t0 = time.time()
    tr, te = load_train(), load_test()
    y, folds = encode_y(tr["model"]), load_folds(tr)
    if a.member:
        build_member(a, tr, te, y, folds, t0)
    if a.twins:
        build_twins(a, te, t0)


if __name__ == "__main__":
    main()

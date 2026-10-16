"""Blend-level check of augmented members: does replacing a reference member improve the nested blends?

(a) --cpu   CPU-only candidate set of blend_pt_cpu_logloss (blend_pt.py 'cpu': tfidf_char_svc_C0.3,
            tfidf_charword_lr_C30, lr_style_tok_v2, ng_char-word-fw_lr_C100_T, ng_char-word-fw-shapew_lr_C300_T,
            lpx_style2_nolm_C0.003_T) with the --replace members swapped in; nested hill-climb (logprob, weights by
            log loss, same procedure as blend_pt.py / blend_baselines.py) vs the same procedure on the original set.
(b) --famc  (n-gram member only) family_combine.py procedure re-run in-process with the new member in place of
            ng_char-word-fw_lr_C100_T as the 6th base member: base6 nested hill-climb, the lab and super-family
            combiners (specialists emb 8B C0.0003, emb 4B C0.001, lpstyle C0.03, ng char+word+fw C30, per family),
            and the GPU blend of blend_pt_gpu_ba (nested hill-climb, weights by BA, over the blend_pt.py 'gpu_noglm'
            candidates, with the recomputed combiners and the new n-gram member swapped in).
Each --replace is OLD=NEW where NEW is a saved member tag or a scratch file outputs/aug_scratch/<NEW>.npz.
Every comparison is paired (compare.paired_bootstrap over prompts) against the reference recomputed with the same code.
Usage: python aug_blend_check.py --cpu --replace ng_char-word-fw_lr_C100_T=ngaug_..._T [--famc] [--log]
"""
from __future__ import annotations

import argparse
import importlib
import os
import time

import numpy as np

from common import OUT, encode_y, iter_folds, load_folds, load_member, load_train, log_experiment, metrics, scores
from compare import fmt, paired_bootstrap

import hill_climb  # noqa: E402  (reuse/airline)

CPU = ["tfidf_char_svc_C0.3", "tfidf_charword_lr_C30", "lr_style_tok_v2", "ng_char-word-fw_lr_C100_T",
       "ng_char-word-fw-shapew_lr_C300_T", "lpx_style2_nolm_C0.003_T"]
NG_REF = "ng_char-word-fw_lr_C100_T"
SPECS = ["emb_qwen3_8b_base_C0.0003", "emb_qwen3_4b_base_C0.001", "lpstyle_C0.03", "ng_char+word+fw_C30"]
FAMC_SAVED = {"lab": "famc6_lab_s4q_pf", "super": "famcsup6_s4q_pf"}


def load_any(name):
    p = OUT / "aug_scratch" / f"{name}.npz"
    try:
        return load_member(name)[0]
    except FileNotFoundError:
        return np.load(p)["oof"]


def nested(oofs, y, folds, metric):
    meta = [(trn, val) for _, trn, val in iter_folds(folds)]
    kw = dict(space="logprob")
    if metric == "logloss":
        kw.update(metric=metrics.mlogloss, higher_is_better=False)
    r = hill_climb.nested_hill(oofs, y, meta, avg_weights=True, **kw)
    return r["nested_pred"], r["weights"]


def famc_oof(grp, base_oofs, y, folds):
    """family_combine.py main (OOF part only, --per-family, no offsets / bias) for grouping grp."""
    os.environ["FAMSPEC_GROUPS"] = grp
    import family_spec
    importlib.reload(family_spec)
    import family_combine as fc
    importlib.reload(fc)
    pre = "" if grp == "lab" else "sup_"
    LS = [fc.lg(fc.load_spec(pre + s)[0]) for s in SPECS]
    out, base = np.zeros_like(base_oofs[0]), np.zeros_like(base_oofs[0])
    for _, trn, val in iter_folds(folds):
        _, w = hill_climb.hill_climb(base_oofs, y, idx=trn, space="logprob", metric=metrics.mlogloss,
                                     higher_is_better=False)
        B = hill_climb.blend(base_oofs, w, "logprob")
        base[val] = B[val]
        params = fc.fit_combiner(B, LS, y, trn, True, None)
        out[val] = fc.apply_family(B, LS, params)[val]
    return out, base


def report(name, P_ref, P_new, y, folds, g, log, extra=""):
    s_ref, s_new = scores(y, P_ref, folds), scores(y, P_new, folds)
    r = paired_bootstrap(y, P_ref, P_new, g)
    line = (f"[{name}] ref BA {s_ref['ba']:.4f} ll {s_ref['logloss']:.4f} -> new BA {s_new['ba']:.4f} "
            f"ll {s_new['logloss']:.4f}; paired {fmt(r)}{('; ' + extra) if extra else ''}")
    print(line, flush=True)
    if log:
        log_experiment(f"augchk_{name}_{log}", f"blend check (aug_blend_check.py): {name}", s_new, None,
                       note=f"replacements {log}; reference recomputed BA {s_ref['ba']:.4f} ll {s_ref['logloss']:.4f}; "
                            f"paired vs reference: {fmt(r)}{('; ' + extra) if extra else ''}; not saved")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="append", required=True, help="OLD=NEW")
    ap.add_argument("--ref-replace", action="append", default=[],
                    help="OLD=NEW applied to the REFERENCE set too (e.g. a no-temperature screening control)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--famc", action="store_true")
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rep = dict(x.split("=", 1) for x in a.replace)
    rrep = dict(x.split("=", 1) for x in a.ref_replace)
    label = ",".join(f"{k}->{v}" for k, v in rep.items()) + (
        "; reference set with " + ",".join(f"{k}->{v}" for k, v in rrep.items()) if rrep else "")
    tr = load_train()
    y, folds, g = encode_y(tr["model"]), load_folds(tr), tr["prompt_id"].values
    log = label if a.log else None
    if a.cpu:
        for metric in ("logloss", "ba"):
            ref_tags = [rrep.get(t, t) for t in CPU]
            P_ref, w_ref = nested([load_any(t) for t in ref_tags], y, folds, metric)
            new_tags = [rep.get(t, t) for t in CPU]
            P_new, w_new = nested([load_any(t) for t in new_tags], y, folds, metric)
            wd = lambda tags, w: {t: round(float(x), 3) for t, x in zip(tags, w)}  # noqa: E731
            report(f"cpu_{metric}", P_ref, P_new, y, folds, g, log,
                   extra=f"weights ref {wd(ref_tags, w_ref)}, new {wd(new_tags, w_new)}")
    if a.famc:
        assert NG_REF in rep, "--famc needs a replacement of " + NG_REF
        import family_combine as fc0
        base_ref = [load_member(t)[0] for t in fc0.BASE] + [load_any(rrep.get(NG_REF, NG_REF))]
        base_new = base_ref[:-1] + [load_any(rep[NG_REF])]
        fam = {}
        for grp in ("lab", "super"):
            F_ref, B_ref = famc_oof(grp, base_ref, y, folds)
            F_new, B_new = famc_oof(grp, base_new, y, folds)
            dev = np.abs(F_ref - load_member(FAMC_SAVED[grp])[0]).max()   # 0 unless --ref-replace
            if grp == "lab":
                report("base6", B_ref, B_new, y, folds, g, log)
            report(f"famc_{grp}", F_ref, F_new, y, folds, g, log,
                   extra=f"max |recomputed ref - saved {FAMC_SAVED[grp]}| {dev:.1e}")
            fam[grp] = (F_ref, F_new)
        # GPU blend of blend_pt_gpu_ba: nested hill-climb (BA) over the blend_pt.py gpu_noglm candidates
        import blend_pt
        cand = blend_pt.SETS["gpu_noglm"]
        for metric in ("ba", "logloss"):
            def arrays(new):
                out = []
                for t in cand:
                    if t == FAMC_SAVED["lab"]:
                        out.append(fam["lab"][1 if new else 0])
                    elif t == FAMC_SAVED["super"]:
                        out.append(fam["super"][1 if new else 0])
                    elif new and t in rep:
                        out.append(load_any(rep[t]))
                    elif not new and t in rrep:
                        out.append(load_any(rrep[t]))
                    else:
                        out.append(load_member(t)[0])
                return out
            P_ref, w_ref = nested(arrays(False), y, folds, metric)
            P_new, w_new = nested(arrays(True), y, folds, metric)
            used = lambda w: {t: round(float(x), 3) for t, x in zip(cand, w) if x > 0}  # noqa: E731
            report(f"gpu_noglm_{metric}", P_ref, P_new, y, folds, g, log,
                   extra=f"weights ref {used(w_ref)}, new {used(w_new)}")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

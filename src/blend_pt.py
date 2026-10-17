"""Per-text blend exploration (post-LB finding: only per-text members transfer): nested hill-climb of candidate
member sets, both weight metrics (log loss / BA), paired bootstrap vs a reference blend.

Same procedure as blend_baselines.py (reuse/airline/hill_climb.nested_hill over the shared folds, logprob space,
final weights = mean of the fold weights); this script only loops over several candidate sets and logs one
EXPERIMENTS.md row per (set, metric) without saving a member. The chosen sets are then saved with
blend_baselines.py --log --tag blend_pt_<variant>_<metric>.
Usage: python blend_pt.py --set gpu_all [--set cpu ...] [--log]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import encode_y, iter_folds, load_folds, load_member, load_train, log_experiment, metrics, scores
from compare import fmt, paired_bootstrap

import hill_climb  # noqa: E402  (reuse/airline)

TFIDF = ["tfidf_char_svc_C0.3", "tfidf_charword_lr_C30"]
LP = ["lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base_style_C0.1"]
EMBQ = ["emb_qwen3_0.6b_base_C0.003", "emb_qwen3_1.7b_base_C0.003", "emb_qwen3_4b_base_C0.001",
        "emb_qwen3_4b_base_C0.003", "emb_qwen3_8b_base_C0.001", "emb_qwen3_8b_base_C0.003"]
STYLE = ["lr_style_tok_v2"]
NG = ["ng_char-word-fw_lr_C100_T", "ng_char-word-fw-shapew_lr_C300_T"]
RAWCAT = ["emb_raw_4b+8b_concat_C0.001"]
LPX_NOLM = ["lpx_style2_nolm_C0.003_T"]
GLM = ["emb_glm4_9b_C0.001", "lpx_lp4glm_style2_C0.003_T"]
FAMC_GLM = ["famcsup6_s5_pf", "famc6_lab_s5_pf"]
FAMC_Q = ["famcsup6_s4q_pf", "famc6_lab_s4q_pf"]      # same combiners without the GLM-4-9B specialist
FAMC_CPU = ["famcpu_sup_ng_s", "famcpu_lab_ngs2_pf"]  # CPU-only combiners (family_combine_cpu.py)
EMB12 = ["emb_qwen3_4b_base_l12_C0.0003", "emb_qwen3_4b_base_l12_C0.001", "emb_qwen3_8b_base_l12_C0.0003",
         "emb_qwen3_8b_base_l12_C0.001"]                 # 12-layer embeddings (layers 3..36 step 3)
FAMC_L12 = ["famcsup6_l12", "famc6_lab_l12"]          # the FAMC_Q combiners rebuilt on the 12-layer embeddings

SETS = {
    # (A) GPU notebook: Qwen3 LM features + TF-IDF + style (+ GLM-4-9B variants to test whether it is needed)
    "gpu_all": TFIDF + LP + EMBQ + STYLE + NG + RAWCAT + LPX_NOLM + GLM + FAMC_GLM + FAMC_Q,
    "gpu_glm": TFIDF + LP + EMBQ + STYLE + NG + RAWCAT + LPX_NOLM + GLM + FAMC_GLM,
    "gpu_noglm": TFIDF + LP + EMBQ + STYLE + NG + RAWCAT + LPX_NOLM + FAMC_Q,
    "gpu_nofamc": TFIDF + LP + EMBQ + STYLE + NG + RAWCAT + LPX_NOLM,
    "gpu_l12": TFIDF + LP + EMBQ + EMB12 + STYLE + NG + RAWCAT + LPX_NOLM + FAMC_L12,
    "gpu_l12_all": TFIDF + LP + EMBQ + EMB12 + STYLE + NG + RAWCAT + LPX_NOLM + FAMC_Q + FAMC_L12,
    # (B) CPU only: no LM features at all
    "cpu": TFIDF + STYLE + NG + LPX_NOLM,
    "cpu_famc": TFIDF + STYLE + NG + LPX_NOLM + FAMC_CPU,
}
REFS = {"gpu": "blend_v3_emb_logloss", "cpu": "blend_v2_tok_ba"}


def run_set(name, tags, y, folds, g, metric, log):
    t0 = time.time()
    meta = [(trn, val) for _, trn, val in iter_folds(folds)]
    oofs = [load_member(t)[0] for t in tags]
    kw = dict(space="logprob")
    if metric == "logloss":
        kw.update(metric=metrics.mlogloss, higher_is_better=False)
    r = hill_climb.nested_hill(oofs, y, meta, avg_weights=True, **kw)
    nested, w = r["nested_pred"], r["weights"]
    s = scores(y, nested, folds)
    ref = REFS["cpu" if name.startswith("cpu") else "gpu"]
    cmp = fmt(paired_bootstrap(y, load_member(ref)[0], nested, g))
    used = {t: round(float(x), 3) for t, x in zip(tags, w) if x > 0}
    print(f"[{name}/{metric}] nested BA {s['ba']:.4f} (folds {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) "
          f"ll {s['logloss']:.4f}\n  weights {used}\n  vs {ref}: {cmp}", flush=True)
    if log:
        log_experiment(f"blendpt_{name}_{metric}",
                       f"nested hill-climb (logprob, weights by {metric}) over {len(tags)} per-text candidates "
                       f"({name}: {', '.join(tags)}) (blend_pt.py)", s, time.time() - t0,
                       note=f"weights {used}; paired vs {ref}: {cmp}; exploration row, not saved")
    return s, nested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, choices=list(SETS))
    ap.add_argument("--metric", nargs="+", default=["logloss", "ba"])
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    tr = load_train()
    y, folds, g = encode_y(tr["model"]), load_folds(tr), tr["prompt_id"].values
    for name in a.set:
        for m in a.metric:
            run_set(name, SETS[name], y, folds, g, m, a.log)


if __name__ == "__main__":
    main()

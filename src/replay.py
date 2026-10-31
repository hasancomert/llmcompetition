"""Submission by REPLAY: re-run the exact experiment scripts (same arguments as the local CV runs) inside the
notebook, then blend the members' test predictions with fixed weights and write the plain argmax.

Why replay: the members, family specialists and family combiners were built by separate scripts that read and write
the project layout (data/, outputs/folds.csv, outputs/emb, outputs/preds, outputs/famspec). Re-running the same
scripts on the notebook's copy of the code reproduces the local CV numbers exactly and predicts the hidden test with
the same code path. Everything is per text: nothing clusters, centers or matches test texts.

Recipes (CV = nested, EXPERIMENTS.md):
  cpu2  char TF-IDF SVC + style+tok LR + n-gram fw + n-gram fw+shapew + style2 (blend_pt_cpu_logloss, 0.4038)
  v4    blend_v3 members + n-gram fw member, lab and super-family specialists (8B emb, 4B emb, lp+style, n-gram fw),
        two nested family combiners blended 0.533 / 0.467 (blend_pt_gpu_ba, 0.4619)   [needs the 4 Qwen3 LMs]
  v5    v4's super-family combiner rebuilt on 12-layer embeddings of the 4B / 8B LMs (layers 3..36 step 3): base
        members with emb_*_l12_C0.0003, specialists sup_emb_*_l12_C0.0003 (blend_pt_gpu_l12_ba, 0.4719; the nested
        hill-climb gives it all the weight)                                          [4 Qwen3 LMs + 2 12-layer passes]
  v6    v5 + test-time adaptive prompt-centering: centered twins of the 8B / 4B 12-layer views (trained on true-prompt
        centered training rows) for test texts in topic clusters of 2..18 (cut 0.85), evidence added with weight 0.5
        per twin (adaptive_twins.py; simulated in adaptive_sim.py: +0.009 at 1-2 texts per prompt, +0.05 at 1-8)
  v7    v6 with (a) the three CPU members of the super combiner trained with the public generated passages (ext_pub.py,
        5 CC BY 4.0 Kaggle datasets attached as inputs; weight 0.3, class-balanced, training folds only) and (b) test-set
        standardization of the 12-layer LM members and specialists (held-out fold / test rows scaled by their own
        statistics). Combiner CV 0.4719 -> 0.4871 (+0.015 [+0.003, +0.027]); twins as in v6
  v7p   v7g + the 4-view n-gram SVC with the public passage pool in the super combiner (CV 0.5030), WITHOUT the
        Hungarian step (public LB A/Bs: -0.009 to -0.017 in 5 tests)
  v7q   v7p with the 4-view n-gram SVC trained with the 120k pool (ext_pool5.py) instead of the 38k one (CV 0.5405)
  v8    v7q + the two-stage 4B LoRA24 member (stage 1 on the 120k pool, kernel nllm-lora2s5 attached; CV 0.5875)
  v8b   v8 + the two-stage 8B LoRA24 member (kernel nllm-lora2s58 attached; CV 0.6064)
  v8c   v8b + the 8B LoRA twins (centered readout in the test clusters, weight 0.25; simulated +0.002..+0.010)
  v9    v8c with a second 8B LoRA (seed 2, more stage-1 data; kernel nllm-lora2s58b) instead of the 4B (CV 0.6186)
  v7g   v7 + shared-prompt Gaussian group evidence of the same two views on the same test clusters (gauss_twins.py,
        weight 0.25, T 3) + one-to-one model assignment inside test clusters of 3..18 texts ("hungarian": 3, on the
        final log-posteriors). Simulated on held-out prompts (EXPERIMENTS gauss_sim_*): over the twins alone +0.009
        (1-2 texts per prompt), +0.013 (1-5), +0.012 (2-4), +0.024 (3-8), +0.090 (18)

External training rows (extdata.py; OFF by default, cpu2 only): --ext ntl_short --ext-weight W re-runs the CPU members
with the external corpus rows added to their training folds (member tags get the extdata suffix; blend weights stay
those of the recipe). The corpus is read from the attached Kaggle dataset (--ext-dir auto: the directory under
/kaggle/input that holds external_short_wide.parquet); it is never copied into the notebook or re-uploaded.

Commit-only path (--check-inputs; build_submission.py --cheap-commit adds it in the notebook's save run only): the save
run's GPU time counts against the weekly quota, the competition's hidden-test rerun does not. The save run then only
checks what the rerun needs (attached inputs, GPU, imports of every step script) and writes a placeholder; the full
replay runs in the rerun. Its in-kernel OOF check is then not visible: run the recipe's local e2e replay first.

Usage (notebook): python replay.py --recipe v4 --data-dir D --folds F --tokenizers-flat T --out submission.parquet
                  [--work /tmp/nllm_work --train-lm auto]       (GPU: LM features of the test are extracted here)
Local check:      python replay.py --recipe cpu2 ... --precomputed-lm outputs/emb
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
Q4 = "qwen3_0.6b_base,qwen3_1.7b_base,qwen3_4b_base,qwen3_8b_base"
LM_MODELS = [("qwen3_0.6b_base", "0.6b-base"), ("qwen3_1.7b_base", "1.7b-base"),
             ("qwen3_4b_base", "4b-base"), ("qwen3_8b_base", "8b-base")]
LP4_STYLE = "lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base_style_C0.1"
L12_FRACS = "0.0833,0.1667,0.25,0.3333,0.4167,0.5,0.5833,0.6667,0.75,0.8333,0.9167,1.0"   # layers 3, 6, .., 36 of 36
LM_L12 = [("qwen3_4b_base_l12", "4b-base", L12_FRACS), ("qwen3_8b_base_l12", "8b-base", L12_FRACS)]
E18_FRACS = ",".join(f"{i / 36:.6f}" for i in range(1, 19))                              # layers 1, 2, .., 18 of 36
LM_E18 = [("qwen3_4b_base_e18", "4b-base", E18_FRACS), ("qwen3_8b_base_e18", "8b-base", E18_FRACS)]

# (env overrides, script, args) — copied from the commands that produced the saved members (meta json / agent logs)
STEPS = {
    "svc": ({}, "train_tfidf.py", ["--feats", "char", "--model", "svc", "--C", "0.3"]),
    "charword": ({}, "train_tfidf.py", ["--feats", "char+word", "--model", "lr", "--C", "30"]),
    "style_lr": ({}, "train_gbdt.py", ["--feats", "style+tok", "--model", "lr", "--C", "0.1", "--tag", "lr_style_tok_v2"]),
    "ng_fw": ({}, "train_ngram.py", ["--feats", "char+word+fw", "--model", "lr", "--C", "100", "--temp", "--save"]),
    "ng_fwsh": ({}, "train_ngram.py", ["--feats", "char+word+fw+shapew", "--model", "lr", "--C", "300", "--temp", "--save"]),
    "style2": ({}, "train_lpx.py", ["--blocks", "style+style2", "--head", "lr", "--C", "0.003", "--drop", "s2_pb_",
                                    "--temp", "--save", "--tag", "lpx_style2_nolm_C0.003_T"]),
    "lp4": ({}, "train_emb.py", ["--kind", "lp", "--names", Q4, "--C", "0.1", "--with-style"]),
    "emb4": ({}, "train_emb.py", ["--kind", "emb", "--names", "qwen3_4b_base", "--C", "0.003"]),
    "emb8": ({}, "train_emb.py", ["--kind", "emb", "--names", "qwen3_8b_base", "--C", "0.003"]),
}
for grp, pre in (("lab", ""), ("super", "sup_")):
    env = {"FAMSPEC_GROUPS": grp}
    STEPS[f"spec_{grp}_e8"] = (env, "family_spec.py", ["--view", "emb:qwen3_8b_base", "--Cs", "0.0003"])
    STEPS[f"spec_{grp}_e4"] = (env, "family_spec.py", ["--view", "emb:qwen3_4b_base", "--Cs", "0.001"])
    STEPS[f"spec_{grp}_lps"] = (env, "family_spec.py", ["--view", "lpstyle", "--Cs", "0.03"])
    STEPS[f"spec_{grp}_ng"] = (env, "family_spec.py", ["--view", "ng:char+word+fw", "--Cs", "30"])
    STEPS[f"spec_{grp}_e8_l12"] = (env, "family_spec.py", ["--view", "emb:qwen3_8b_base_l12", "--Cs", "0.0003"])
    STEPS[f"spec_{grp}_e4_l12"] = (env, "family_spec.py", ["--view", "emb:qwen3_4b_base_l12", "--Cs", "0.0003"])
    STEPS[f"famc_{grp}"] = (env, "family_combine.py", [
        "--specs", f"{pre}emb_qwen3_8b_base_C0.0003", f"{pre}emb_qwen3_4b_base_C0.001", f"{pre}lpstyle_C0.03",
        f"{pre}ng_char+word+fw_C30", "--per-family", "--extra-members", "ng_char-word-fw_lr_C100_T",
        "--save", "famc6_lab_s4q_pf" if grp == "lab" else "famcsup6_s4q_pf"])

STEPS["emb4_l12"] = ({}, "train_emb.py", ["--kind", "emb", "--names", "qwen3_4b_base_l12", "--C", "0.0003"])
STEPS["emb8_l12"] = ({}, "train_emb.py", ["--kind", "emb", "--names", "qwen3_8b_base_l12", "--C", "0.0003"])
for tag in ("e18",):
    STEPS[f"emb4_{tag}"] = ({}, "train_emb.py", ["--kind", "emb", "--names", f"qwen3_4b_base_{tag}", "--C", "0.0003"])
    STEPS[f"emb8_{tag}"] = ({}, "train_emb.py", ["--kind", "emb", "--names", f"qwen3_8b_base_{tag}", "--C", "0.0003"])
    for m in ("8", "4"):
        STEPS[f"spec_super_e{m}_{tag}"] = ({"FAMSPEC_GROUPS": "super"}, "family_spec.py",
                                          ["--view", f"emb:qwen3_{m}b_base_{tag}", "--Cs", "0.0003"])
    STEPS[f"famc_super_{tag}"] = ({"FAMSPEC_GROUPS": "super"}, "family_combine.py", [
        "--specs", f"sup_emb_qwen3_8b_base_{tag}_C0.0003", f"sup_emb_qwen3_4b_base_{tag}_C0.0003", "sup_lpstyle_C0.03",
        "sup_ng_char+word+fw_C30", "--per-family", "--extra-members", "ng_char-word-fw_lr_C100_T",
        "--base", "tfidf_charword_lr_C30", "tfidf_char_svc_C0.3", LP4_STYLE, f"emb_qwen3_4b_base_{tag}_C0.0003",
        f"emb_qwen3_8b_base_{tag}_C0.0003", "--save", f"famcsup6_{tag}"])
# public generated passages (ext_pub.py; CC BY 4.0 Kaggle datasets attached as notebook inputs, never copied):
# class-balanced weight 0.3, training folds only; run_step adds --ext-dir (the attached inputs) when --pub-dir is set
PUB = ["--ext", "pub", "--ext-balance", "--ext-weight", "0.3"]
STEPS["charword_pub"] = ({}, "train_tfidf.py", ["--feats", "char+word", "--model", "lr", "--C", "30", *PUB])
STEPS["svc_pub"] = ({}, "train_tfidf.py", ["--feats", "char", "--model", "svc", "--C", "0.3", *PUB])
STEPS["ng_fw_pub"] = ({}, "train_ngram.py", ["--feats", "char+word+fw", "--model", "lr", "--C", "100", "--temp", "--save",
                                             "--n-jobs", "4", *PUB])
STEPS["spec_super_ng_pub"] = ({"FAMSPEC_GROUPS": "super"}, "family_spec.py", ["--view", "ng:char+word+fw", "--Cs", "30",
                                                                              *PUB])
# test-set standardization of the frozen-LM features (held-out fold / test rows scaled by their own statistics)
STEPS["emb4_l12_ts"] = ({}, "train_emb.py", ["--kind", "emb", "--names", "qwen3_4b_base_l12", "--C", "0.0003", "--test-std"])
STEPS["emb8_l12_ts"] = ({}, "train_emb.py", ["--kind", "emb", "--names", "qwen3_8b_base_l12", "--C", "0.0003", "--test-std"])
for m in ("8", "4"):
    STEPS[f"spec_super_e{m}_l12_ts"] = ({"FAMSPEC_GROUPS": "super"}, "family_spec.py",
                                       ["--view", f"emb:qwen3_{m}b_base_l12", "--Cs", "0.0003", "--test-std"])
STEPS["famc_super_v7"] = ({"FAMSPEC_GROUPS": "super"}, "family_combine.py", [
    "--specs", "sup_emb_qwen3_8b_base_l12_ts_C0.0003", "sup_emb_qwen3_4b_base_l12_ts_C0.0003", "sup_lpstyle_C0.03",
    "sup_ng_char+word+fw_C30", "--per-family", "--extra-members", "ng_char-word-fw_lr_C100_T_ext-pub-w0.3-bal",
    "--base", "tfidf_charword_lr_C30_ext-pub-w0.3-bal", "tfidf_char_svc_C0.3_ext-pub-w0.3-bal", LP4_STYLE,
    "emb_qwen3_4b_base_l12_C0.0003_ts", "emb_qwen3_8b_base_l12_C0.0003_ts", "--save", "famcsup6_l12_pub03_ts"])
STEPS["twins_l12"] = ({}, "adaptive_twins.py", ["--views", "qwen3_8b_base_l12,qwen3_4b_base_l12", "--C", "0.001",
                                                 "--thr", "0.85", "--tag", "twins_l12"])
# the 4-view n-gram SVC of the public two-stage notebook, trained with the public passage pool (ext_pool.py; weight 1,
# class-balanced, training folds only; EXPERIMENTS row 606: 0.4101; in the v7 combiner 0.4871 -> 0.5030)
STEPS["ng4_pool"] = ({}, "train_ngram.py", ["--feats", "char+word13+shapec+fw", "--stream-weights", "1,0.7,0.5,0.5",
                                            "--model", "svc", "--C", "0.5", "--save", "--n-jobs", "2", "--ext", "pool",
                                            "--ext-balance", "--ext-weight", "1.0"])
STEPS["famc_super_v7p"] = ({"FAMSPEC_GROUPS": "super"}, "family_combine.py", [
    "--specs", "sup_emb_qwen3_8b_base_l12_ts_C0.0003", "sup_emb_qwen3_4b_base_l12_ts_C0.0003", "sup_lpstyle_C0.03",
    "sup_ng_char+word+fw_C30", "--per-family", "--extra-members", "ng_char-word-fw_lr_C100_T_ext-pub-w0.3-bal",
    "ng_char-word13-shapec-fw_svc_C0.5_w1_0.7_0.5_0.5_ext-pool-w1-bal",
    "--base", "tfidf_charword_lr_C30_ext-pub-w0.3-bal", "tfidf_char_svc_C0.3_ext-pub-w0.3-bal", LP4_STYLE,
    "emb_qwen3_4b_base_l12_C0.0003_ts", "emb_qwen3_8b_base_l12_C0.0003_ts", "--save", "famcsup6_l12_pub03_ts_ng4pool"])
# the same SVC with the 120k pool (ext_pool5; EXPERIMENTS 625: 0.4101 -> 0.4649): ~14 GB and 3.4 h on a Kaggle CPU
STEPS["ng4_pool5"] = ({}, "train_ngram.py", ["--feats", "char+word13+shapec+fw", "--stream-weights", "1,0.7,0.5,0.5",
                                             "--model", "svc", "--C", "0.5", "--save", "--n-jobs", "1", "--ext", "pool5",
                                             "--ext-balance", "--ext-weight", "1.0", "--ext-min-labels", "1"])
STEPS["famc_super_v7q"] = ({"FAMSPEC_GROUPS": "super"}, "family_combine.py", [
    "--specs", "sup_emb_qwen3_8b_base_l12_ts_C0.0003", "sup_emb_qwen3_4b_base_l12_ts_C0.0003", "sup_lpstyle_C0.03",
    "sup_ng_char+word+fw_C30", "--per-family", "--extra-members", "ng_char-word-fw_lr_C100_T_ext-pub-w0.3-bal",
    "ng_char-word13-shapec-fw_svc_C0.5_w1_0.7_0.5_0.5_ext-pool5-w1-ml1-bal",
    "--base", "tfidf_charword_lr_C30_ext-pub-w0.3-bal", "tfidf_char_svc_C0.3_ext-pub-w0.3-bal", LP4_STYLE,
    "emb_qwen3_4b_base_l12_C0.0003_ts", "emb_qwen3_8b_base_l12_C0.0003_ts", "--save", "famcsup6_l12_pub03_ts_ng4pool5"])
# two-stage LoRA24 members (gpu_lora24.py; adapters = the attached output of the training kernel): test features on
# the GPU (lora24_infer.py, anchor parity checked), then the plain-readout member (OOF from the jobs' held-out features)
STEPS["lora4_infer"] = ({}, "lora24_infer.py", ["--model-dir", "{LM:4b-base}", "--jobs-dir", "{JOBS:nllm-lora2s5}",
                                                "--texts", "{ROOT}/data/test.csv", "--train", "{ROOT}/data/train.csv",
                                                "--out", "{OUT}/lora24_2s5_4b"])
STEPS["lora4_member"] = ({}, "lora24_member.py", ["--jobs-dir", "{JOBS:nllm-lora2s5}", "--features-dir",
                                                  "{OUT}/lora24_2s5_4b", "--member", "lora24_2s5_4b"])
STEPS["lora8_infer"] = ({}, "lora24_infer.py", ["--model-dir", "{LM:8b-base}", "--jobs-dir", "{JOBS:nllm-lora2s58}",
                                                "--texts", "{ROOT}/data/test.csv", "--train", "{ROOT}/data/train.csv",
                                                "--out", "{OUT}/lora24_2s5_8b"])
STEPS["lora8_member"] = ({}, "lora24_member.py", ["--jobs-dir", "{JOBS:nllm-lora2s58}", "--features-dir",
                                                  "{OUT}/lora24_2s5_8b", "--member", "lora24_2s5_8b"])
STEPS["lora8_twins"] = ({}, "lora24_member.py", ["--jobs-dir", "{JOBS:nllm-lora2s58}", "--features-dir",
                                                 "{OUT}/lora24_2s5_8b", "--twins", "twins_lora8", "--thr", "0.85"])
STEPS["lora8b_infer"] = ({}, "lora24_infer.py", ["--model-dir", "{LM:8b-base}", "--jobs-dir", "{JOBS:nllm-lora2s58b}",
                                                 "--texts", "{ROOT}/data/test.csv", "--train", "{ROOT}/data/train.csv",
                                                 "--out", "{OUT}/lora24_2s5_58b"])
STEPS["lora8b_member"] = ({}, "lora24_member.py", ["--jobs-dir", "{JOBS:nllm-lora2s58b}", "--features-dir",
                                                   "{OUT}/lora24_2s5_58b", "--member", "lora24_2s5_58b"])
STEPS["famc_super_v9"] = ({"FAMSPEC_GROUPS": "super"}, "family_combine.py", [
    "--specs", "sup_emb_qwen3_8b_base_l12_ts_C0.0003", "sup_emb_qwen3_4b_base_l12_ts_C0.0003", "sup_lpstyle_C0.03",
    "sup_ng_char+word+fw_C30", "--per-family", "--extra-members", "ng_char-word-fw_lr_C100_T_ext-pub-w0.3-bal",
    "ng_char-word13-shapec-fw_svc_C0.5_w1_0.7_0.5_0.5_ext-pool5-w1-ml1-bal", "lora24_2s5_8b", "lora24_2s5_58b",
    "--base", "tfidf_charword_lr_C30_ext-pub-w0.3-bal", "tfidf_char_svc_C0.3_ext-pub-w0.3-bal", LP4_STYLE,
    "emb_qwen3_4b_base_l12_C0.0003_ts", "emb_qwen3_8b_base_l12_C0.0003_ts", "--save", "famcsup6_l12_pub03_ts_ng4pool5_lora8_8b"])
STEPS["famc_super_v8b"] = ({"FAMSPEC_GROUPS": "super"}, "family_combine.py", [
    "--specs", "sup_emb_qwen3_8b_base_l12_ts_C0.0003", "sup_emb_qwen3_4b_base_l12_ts_C0.0003", "sup_lpstyle_C0.03",
    "sup_ng_char+word+fw_C30", "--per-family", "--extra-members", "ng_char-word-fw_lr_C100_T_ext-pub-w0.3-bal",
    "ng_char-word13-shapec-fw_svc_C0.5_w1_0.7_0.5_0.5_ext-pool5-w1-ml1-bal", "lora24_2s5_4b", "lora24_2s5_8b",
    "--base", "tfidf_charword_lr_C30_ext-pub-w0.3-bal", "tfidf_char_svc_C0.3_ext-pub-w0.3-bal", LP4_STYLE,
    "emb_qwen3_4b_base_l12_C0.0003_ts", "emb_qwen3_8b_base_l12_C0.0003_ts", "--save", "famcsup6_l12_pub03_ts_ng4pool5_lora48"])
STEPS["famc_super_v8"] = ({"FAMSPEC_GROUPS": "super"}, "family_combine.py", [
    "--specs", "sup_emb_qwen3_8b_base_l12_ts_C0.0003", "sup_emb_qwen3_4b_base_l12_ts_C0.0003", "sup_lpstyle_C0.03",
    "sup_ng_char+word+fw_C30", "--per-family", "--extra-members", "ng_char-word-fw_lr_C100_T_ext-pub-w0.3-bal",
    "ng_char-word13-shapec-fw_svc_C0.5_w1_0.7_0.5_0.5_ext-pool5-w1-ml1-bal", "lora24_2s5_4b",
    "--base", "tfidf_charword_lr_C30_ext-pub-w0.3-bal", "tfidf_char_svc_C0.3_ext-pub-w0.3-bal", LP4_STYLE,
    "emb_qwen3_4b_base_l12_C0.0003_ts", "emb_qwen3_8b_base_l12_C0.0003_ts", "--save", "famcsup6_l12_pub03_ts_ng4pool5_lora4"])
STEPS["gauss_l12"] = ({}, "gauss_twins.py", ["--views", "qwen3_8b_base_l12;qwen3_4b_base_l12", "--T", "3",
                                              "--thr", "0.85", "--tag", "gauss_l12"])
STEPS["famc_super_l12"] = ({"FAMSPEC_GROUPS": "super"}, "family_combine.py", [
    "--specs", "sup_emb_qwen3_8b_base_l12_C0.0003", "sup_emb_qwen3_4b_base_l12_C0.0003", "sup_lpstyle_C0.03",
    "sup_ng_char+word+fw_C30", "--per-family", "--extra-members", "ng_char-word-fw_lr_C100_T",
    "--base", "tfidf_charword_lr_C30", "tfidf_char_svc_C0.3", LP4_STYLE, "emb_qwen3_4b_base_l12_C0.0003",
    "emb_qwen3_8b_base_l12_C0.0003", "--save", "famcsup6_l12"])

RECIPES = {
    "cpu2": {"needs_lm": False,
             "steps": ["svc", "style_lr", "ng_fw", "ng_fwsh", "style2"],
             "weights": {"tfidf_char_svc_C0.3": 0.17333, "lr_style_tok_v2": 0.17333, "ng_char-word-fw_lr_C100_T": 0.17333,
                         "ng_char-word-fw-shapew_lr_C300_T": 0.30667, "lpx_style2_nolm_C0.003_T": 0.17333},
             "cv": 0.4038},
    "v4": {"needs_lm": True,
           "steps": ["charword", "svc", "lp4", "emb4", "emb8", "ng_fw",
                     "spec_lab_e8", "spec_lab_e4", "spec_lab_lps", "spec_lab_ng", "famc_lab",
                     "spec_super_e8", "spec_super_e4", "spec_super_lps", "spec_super_ng", "famc_super"],
           "weights": {"famcsup6_s4q_pf": 0.53333, "famc6_lab_s4q_pf": 0.46667},
           "cv": 0.4619},
    "v5": {"needs_lm": True, "lm_extra": LM_L12,
           "steps": ["charword", "svc", "lp4", "emb4_l12", "emb8_l12", "ng_fw",
                     "spec_super_e8_l12", "spec_super_e4_l12", "spec_super_lps", "spec_super_ng", "famc_super_l12"],
           "weights": {"famcsup6_l12": 1.0},
           "cv": 0.4719},
    "v6": {"needs_lm": True, "lm_extra": LM_L12,
           "steps": ["charword", "svc", "lp4", "emb4_l12", "emb8_l12", "ng_fw",
                     "spec_super_e8_l12", "spec_super_e4_l12", "spec_super_lps", "spec_super_ng", "famc_super_l12",
                     "twins_l12"],
           "weights": {"famcsup6_l12": 1.0},
           "twins": {"twins_l12": 0.5},
           "cv": 0.4719},
    "v7": {"needs_lm": True, "lm_extra": LM_L12,
           # same steps as a sequential list, run in 3 stages (4 CPUs): ~half the wall time of the sequential order
           "stages": [[["charword_pub@1"], ["svc_pub@1"], ["lp4@2", "emb4_l12_ts@2", "emb8_l12_ts@2"]],
                      [["ng_fw_pub@1"], ["spec_super_e8_l12_ts@1", "spec_super_e4_l12_ts@1", "spec_super_lps@1",
                                         "spec_super_ng@1", "twins_l12@1"]],
                      [["famc_super_v7"]]],
           "weights": {"famcsup6_l12_pub03_ts": 1.0},
           "twins": {"twins_l12": 0.5},
           "cv": 0.4871},
    "v7g": {"needs_lm": True, "lm_extra": LM_L12,
            # gauss_l12 in a stage of its own (memory: the 2-chain stage 2 of v7 is the peak; local e2e replay
            # on a 539-text pseudo test: gauss 2.1 GB, whole replay 10.3 GB)
            "stages": [[["charword_pub@1"], ["svc_pub@1"], ["lp4@2", "emb4_l12_ts@2", "emb8_l12_ts@2"]],
                       [["ng_fw_pub@1"], ["spec_super_e8_l12_ts@1", "spec_super_e4_l12_ts@1", "spec_super_lps@1",
                                          "spec_super_ng@1", "twins_l12@1"]],
                       [["famc_super_v7"]],
                       [["gauss_l12"]]],
            "weights": {"famcsup6_l12_pub03_ts": 1.0},
            "twins": {"twins_l12": 0.5, "gauss_l12": 0.25},
            "hungarian": 3,
            "cv": 0.4871},
    "v7p": {"needs_lm": True, "lm_extra": LM_L12, "needs_pool": True,
            # v7g + the 4-view n-gram SVC with the public pool in the super combiner (CV 0.4871 -> 0.5030). The SVC needs
            # no LM features: it starts before the GPU feature extraction and is awaited before the combiner stage
            # (local e2e replay inside stage 1: stage 1 1,731 s instead of 770 s). n-jobs 2: with 3 workers next to
            # stage 1 the local 15 GB machine ran out of memory
            "early": [["ng4_pool"]], "early_join": 2,
            "stages": [[["charword_pub@1"], ["svc_pub@1"], ["lp4@2", "emb4_l12_ts@2", "emb8_l12_ts@2"]],
                       [["ng_fw_pub@1"], ["spec_super_e8_l12_ts@1", "spec_super_e4_l12_ts@1", "spec_super_lps@1",
                                          "spec_super_ng@1", "twins_l12@1"]],
                       [["famc_super_v7p"]],
                       [["gauss_l12"]]],
            "weights": {"famcsup6_l12_pub03_ts_ng4pool": 1.0},
            "twins": {"twins_l12": 0.5, "gauss_l12": 0.25},
            # no Hungarian: public LB A/Bs lost 0.009-0.017 with it every time (5 tests, 'Full Ensemble + What
            # Failed'); the 0.66 public notebooks leave it out; our 16 Oct soft matching lost 0.006 on the LB
            "cv": 0.5030},
    "v7q": {"needs_lm": True, "lm_extra": LM_L12, "needs_pool5": True,
            # v7p with the 4-view n-gram SVC trained with the 120k pool instead of the 38k one (combiner CV 0.5030 ->
            # 0.5405, EXPERIMENTS 624). The SVC needs ~14 GB and ~3.4 h (one fold at a time; Kaggle CPU kernel
            # nllm-cpu-ng4pool5): it runs alone after stage 2 (next to the GPU extraction or the other stages the
            # memory could run out), ~4.8 h in all. No local e2e (15 GB is too little); the steps it shares with v7p
            # passed v7p's local e2e, the SVC step passed on the Kaggle CPU kernel
            "stages": [[["charword_pub@1"], ["svc_pub@1"], ["lp4@2", "emb4_l12_ts@2", "emb8_l12_ts@2"]],
                       [["ng_fw_pub@1"], ["spec_super_e8_l12_ts@1", "spec_super_e4_l12_ts@1", "spec_super_lps@1",
                                          "spec_super_ng@1", "twins_l12@1"]],
                       [["ng4_pool5@4"]],
                       [["famc_super_v7q"]],
                       [["gauss_l12"]]],
            "weights": {"famcsup6_l12_pub03_ts_ng4pool5": 1.0},
            "twins": {"twins_l12": 0.5, "gauss_l12": 0.25},
            "cv": 0.5405},
    "v8": {"needs_lm": True, "lm_extra": LM_L12, "needs_pool5": True,
           # v7q + the two-stage 4B LoRA24 member (stage 1 on the 120k pool; nllm-lora2s5, OOF 0.5568) in the super
           # combiner (CV 0.5405 -> 0.5875, EXPERIMENTS); its test features are computed on the GPU in stage 1
           "stages": [[["charword_pub@1"], ["svc_pub@1"], ["lp4@2", "emb4_l12_ts@2", "emb8_l12_ts@2"],
                       ["lora4_infer@2", "lora4_member@1"]],
                      [["ng_fw_pub@1"], ["spec_super_e8_l12_ts@1", "spec_super_e4_l12_ts@1", "spec_super_lps@1",
                                         "spec_super_ng@1", "twins_l12@1"]],
                      [["ng4_pool5@4"]],
                      [["famc_super_v8"]],
                      [["gauss_l12"]]],
           "weights": {"famcsup6_l12_pub03_ts_ng4pool5_lora4": 1.0},
           "twins": {"twins_l12": 0.5, "gauss_l12": 0.25},
           "cv": 0.5875},
    "v8b": {"needs_lm": True, "lm_extra": LM_L12, "needs_pool5": True,
            # v8 + the two-stage 8B LoRA24 member (nllm-lora2s58, OOF 0.5894) in the super combiner (CV 0.5875 ->
            # 0.6064); the 4B and 8B test features are computed one after the other on the GPU in stage 1
            "stages": [[["charword_pub@1"], ["svc_pub@1"], ["lp4@2", "emb4_l12_ts@2", "emb8_l12_ts@2"],
                        ["lora4_infer@2", "lora4_member@1", "lora8_infer@2", "lora8_member@1"]],
                       [["ng_fw_pub@1"], ["spec_super_e8_l12_ts@1", "spec_super_e4_l12_ts@1", "spec_super_lps@1",
                                          "spec_super_ng@1", "twins_l12@1"]],
                       [["ng4_pool5@4"]],
                       [["famc_super_v8b"]],
                       [["gauss_l12"]]],
            "weights": {"famcsup6_l12_pub03_ts_ng4pool5_lora48": 1.0},
            "twins": {"twins_l12": 0.5, "gauss_l12": 0.25},
            "cv": 0.6064},
    "v8c": {"needs_lm": True, "lm_extra": LM_L12, "needs_pool5": True,
            # v8b + the 8B LoRA twins: its centered head on test features centered within the topic clusters of the LM
            # twins, weight 0.25 next to the LM twins (0.5) and the Gaussian heads (0.25). Simulated on the v8b plain
            # combiner (EXPERIMENTS lora_twin_sim_v8b_*): +0.002 / +0.008 / +0.003 / +0.010 / +0.005 at 1-2 / 2-4 /
            # 1-5 / 3-8 / 18 texts per prompt over v8b's post-steps, the only setting positive in every structure
            "stages": [[["charword_pub@1"], ["svc_pub@1"], ["lp4@2", "emb4_l12_ts@2", "emb8_l12_ts@2"],
                        ["lora4_infer@2", "lora4_member@1", "lora8_infer@2", "lora8_member@1", "lora8_twins@1"]],
                       [["ng_fw_pub@1"], ["spec_super_e8_l12_ts@1", "spec_super_e4_l12_ts@1", "spec_super_lps@1",
                                          "spec_super_ng@1", "twins_l12@1"]],
                       [["ng4_pool5@4"]],
                       [["famc_super_v8b"]],
                       [["gauss_l12"]]],
            "weights": {"famcsup6_l12_pub03_ts_ng4pool5_lora48": 1.0},
            "twins": {"twins_l12": 0.5, "gauss_l12": 0.25, "twins_lora8": 0.25},
            "cv": 0.6064},
    "v9": {"needs_lm": True, "lm_extra": LM_L12, "needs_pool5": True,
           # v8c with a second 8B two-stage LoRA (nllm-lora2s58b: seed 2, stage 1 on the 3+ label blocks of the 120k
           # pool; OOF 0.5949) and without the 4B (no weight next to the 8B): super combiner 0.6064 -> 0.6186 (+0.0122
           # [+0.0033, +0.0211]); a third 8B (seed 3) lowered it (0.6157). Twins as in v8c (8B LoRA twins w 0.25)
           "stages": [[["charword_pub@1"], ["svc_pub@1"], ["lp4@2", "emb4_l12_ts@2", "emb8_l12_ts@2"],
                       ["lora8_infer@2", "lora8_member@1", "lora8_twins@1", "lora8b_infer@2", "lora8b_member@1"]],
                      [["ng_fw_pub@1"], ["spec_super_e8_l12_ts@1", "spec_super_e4_l12_ts@1", "spec_super_lps@1",
                                         "spec_super_ng@1", "twins_l12@1"]],
                      [["ng4_pool5@4"]],
                      [["famc_super_v9"]],
                      [["gauss_l12"]]],
           "weights": {"famcsup6_l12_pub03_ts_ng4pool5_lora8_8b": 1.0},
           "twins": {"twins_l12": 0.5, "gauss_l12": 0.25, "twins_lora8": 0.25},
           "cv": 0.6186},
}


def log(msg, t0):
    print(f"[replay] {msg}  ({time.time() - t0:.0f}s)", flush=True)


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy(src, dst)


def setup_project(a, t0):
    """data/, outputs/folds.csv and tokenizers under ROOT (the notebook's code copy, or a local scratch copy)."""
    d = Path(a.data_dir)
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(d / "train.csv")
    te = pd.read_csv(d / "test.csv", dtype={"id": str})
    te["text"] = te["text"].fillna("").astype(str)
    if (ROOT / "data" / "train.csv").resolve() != (d / "train.csv").resolve():
        tr.to_csv(ROOT / "data" / "train.csv", index=False)
        te[["id", "text"]].to_csv(ROOT / "data" / "test.csv", index=False)
    if (ROOT / "outputs" / "folds.csv").resolve() != Path(a.folds).resolve():
        shutil.copy(a.folds, ROOT / "outputs" / "folds.csv")
    fdf = pd.read_csv(ROOT / "outputs" / "folds.csv")
    assert (fdf["id"].values == tr["id"].values).all(), "folds.csv does not match train.csv row order"
    sys.path.insert(0, str(HERE))
    from submission_pipeline import materialize_tokenizers
    tok = materialize_tokenizers(Path(a.tokenizers_flat), Path("/tmp/nllm_tokenizers"))
    os.environ["TOKENIZER_DIR"] = str(tok)
    log(f"project at {ROOT}: train {len(tr)}, test {len(te)}", t0)
    return tr, te


def lm_dir(hint: str) -> str:
    """Directory of the attached Qwen3 model (Kaggle Models) whose path holds /<hint>/."""
    cfg = [q for q in glob.glob("/kaggle/input/**/config.json", recursive=True) if f"/{hint}/" in q and "qwen" in q]
    if not cfg:
        raise FileNotFoundError(f"model {hint} not attached")
    return os.path.dirname(cfg[0])


def setup_lm(a, tr, te, t0, extra=()):
    """outputs/emb/{emb,lp,meta}_<name>_{train,test}: precomputed (local), or test extracted here (GPU) with the
    train split taken from the private dataset nllm-lmfeats when it matches train.csv row order.
    extra: further (name, model hint, layer fracs) passes, e.g. the 12-layer embeddings of v5."""
    emb = ROOT / "outputs" / "emb"
    emb.mkdir(parents=True, exist_ok=True)
    models = [(n, h, None) for n, h in LM_MODELS] + list(extra)
    for f in glob.glob("/kaggle/input/**/*_ext_pub.*", recursive=True):   # LM features of the public passages
        if Path(f).name.startswith(("emb_", "lp_", "meta_")):           # (output of nllm-gpu-embed-ext, attached)
            link(Path(f), emb / Path(f).name)
    if a.precomputed_lm:
        p = Path(a.precomputed_lm)
        for name, _, _ in models:
            for split in ("train", "test"):
                for kind, ext in (("emb", "npy"), ("lp", "npy"), ("meta", "json")):
                    link(p / f"{kind}_{name}_{split}.{ext}", emb / f"{kind}_{name}_{split}.{ext}")
        log("LM features linked from " + str(p), t0)
        return
    from submission_pipeline_gpu import find_train_lm, load_split
    train_dir = find_train_lm(a.train_lm)
    for name, hint, fracs in models:
        model_dir = lm_dir(hint)
        pre = load_split(train_dir, name, "train", tr["id"].tolist()) if train_dir else None
        if pre is not None:
            for kind, ext in (("emb", "npy"), ("lp", "npy"), ("meta", "json")):
                link(train_dir / f"{kind}_{name}_train.{ext}", emb / f"{kind}_{name}_train.{ext}")
        texts = ([] if pre is not None else ["--texts", str(ROOT / "data" / "train.csv")]) + \
            ["--texts", str(ROOT / "data" / "test.csv")]
        r = subprocess.run([sys.executable, str(HERE / "gpu_embed.py"), "--model-dir", model_dir,
                            "--name", name, *texts, "--out", str(emb)] + (["--layer-fracs", fracs] if fracs else []))
        if r.returncode != 0:
            raise RuntimeError(f"gpu_embed failed for {name} (exit {r.returncode})")
        log(f"LM features {name}: train {'from ' + str(train_dir) if pre is not None else 'extracted'}, test extracted",
            t0)


EXT_STEPS = {"svc", "charword", "style_lr", "ng_fw", "ng_fwsh", "style2"}   # steps that accept extdata flags


def find_ext_dir(spec: str) -> Path:
    if spec != "auto":
        return Path(spec)
    hits = sorted(glob.glob("/kaggle/input/**/external_short_wide.parquet", recursive=True))
    assert hits, "external corpus not attached (add olegzholobov/name-that-llm-response-corpus as a notebook input)"
    return Path(hits[0]).parent


def find_pub_dir(spec: str) -> str:
    """Root of the attached public generated-passage datasets (ext_pub.load searches below it), '' locally."""
    if spec != "auto":
        return spec
    return "/kaggle/input" if glob.glob("/kaggle/input/**/passages_clean.csv", recursive=True) else ""


def find_pool_dir(spec: str) -> str:
    """Directory of the attached public passage pool (ext_pool.py), '' locally (data/external/pools/prvsiyan)."""
    if spec != "auto":
        return spec
    hits = sorted(p for p in glob.glob("/kaggle/input/**/pool_v1.parquet", recursive=True)
                  if "ntl-public-in-style-pool" in p)
    return os.path.dirname(hits[0]) if hits else ""


def find_pool5_dir(spec: str) -> str:
    """Directory of the attached 120k pool (ext_pool5.py), '' locally (data/external/pools/dhruv)."""
    if spec != "auto":
        return spec
    hits = sorted(p for p in glob.glob("/kaggle/input/**/nlm_pool_v5.parquet", recursive=True)
                  if "nlm-matched-clean" in p)
    return os.path.dirname(hits[0]) if hits else ""


POOL_DIR = ""                                        # set in main(); run_step passes it to --ext pool steps
POOL5_DIR = ""                                       # same for --ext pool5 steps


def jobs_dir(slug: str) -> str:
    """lora24/ of an attached LoRA training kernel's output (Kaggle: kernel sources under /kaggle/input; locally the
    weight-free copy fetched by build_gpu_lora24.py --fetch, enough for lora24_member.py but not for inference)."""
    hits = sorted(os.path.dirname(os.path.dirname(p)) for p in
                  glob.glob("/kaggle/input/**/lora24/f0/COMPLETE.json", recursive=True) if f"/{slug}/" in p)
    if hits:
        return hits[0]
    local = Path(os.environ.get("NLLM_KAGGLE_BUILD", ROOT / "outputs" / "kaggle_build")) / slug / "output" / "lora24"
    assert local.exists(), f"LoRA jobs of {slug} not attached (build with --kernel-sources hasancmert/{slug})"
    return str(local)


def resolve(arg: str) -> str:
    """Run-time paths in step arguments: {LM:<hint>} attached model, {JOBS:<slug>} LoRA jobs, {ROOT} / {OUT}."""
    if arg.startswith("{LM:"):
        return lm_dir(arg[4:-1])
    if arg.startswith("{JOBS:"):
        return jobs_dir(arg[6:-1])
    return arg.replace("{ROOT}", str(ROOT)).replace("{OUT}", str(ROOT / "outputs"))


def run_step(key, t0, extra=(), pub_dir="", threads=None):
    env_over, script, args = STEPS[key]
    args = [resolve(x) for x in args] + list(extra)
    if pub_dir and "pub" in args and "--ext" in args:
        args += ["--ext-dir", pub_dir]
    if POOL_DIR and "pool" in args and "--ext" in args:
        args += ["--ext-dir", POOL_DIR]
    if POOL5_DIR and "pool5" in args and "--ext" in args:
        args += ["--ext-dir", POOL5_DIR]
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"), NLLM_KGRAM_CACHE="0", **env_over)
    if threads:                                     # steps running side by side (recipe stages): fixed thread budgets
        env.update(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    r = subprocess.run([sys.executable, str(HERE / script), *args], cwd=str(HERE), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = "\n".join(l for l in r.stdout.splitlines() if l.startswith("[") or "Error" in l or "Traceback" in l)[-1500:]
    if r.returncode != 0:
        print(r.stdout[-4000:], flush=True)
        raise RuntimeError(f"step {key} ({script}) failed with exit {r.returncode}")
    log(f"step {key}: {script} {' '.join(args)[:120]} -> ok\n{tail}", t0)


def run_chain(entries, t0, ext_args, pub_dir):
    """One chain: its steps in order; an entry is 'step' or 'step@threads' (OMP / BLAS threads of that step)."""
    for e in entries:
        key, _, th = e.partition("@")
        run_step(key, t0, ext_args if key in EXT_STEPS else (), pub_dir, int(th) if th else None)


def start_early(chains, t0, ext_args, pub_dir):
    """'early' chains of a recipe: CPU-only steps (no LM features) started before the GPU feature extraction, so that
    they overlap with it; run_stages waits for them before the stage 'early_join'. Returns (executor, futures)."""
    from concurrent.futures import ThreadPoolExecutor
    if not chains:
        return None, []
    ex = ThreadPoolExecutor(len(chains))
    log(f"early chains started: {chains}", t0)
    return ex, [ex.submit(run_chain, c, t0, ext_args, pub_dir) for c in chains]


def run_stages(stages, t0, ext_args, pub_dir, early=(), early_join=None):
    """Recipe stages: the chains of a stage run side by side (each chain's steps in order), stages one after the other.
    Steps only exchange files through outputs/, so independent members can run concurrently; the first failure is
    re-raised after the stage. early: futures of the early chains, awaited before stage index early_join."""
    from concurrent.futures import ThreadPoolExecutor
    for i, stage in enumerate(stages):
        if early and i == early_join:
            errs = [f.exception() for f in early]
            if any(errs):
                raise next(e for e in errs if e)
            log("early chains done", t0)
        with ThreadPoolExecutor(len(stage)) as ex:
            futs = [ex.submit(run_chain, c, t0, ext_args, pub_dir) for c in stage]
            errs = [f.exception() for f in futs]
        if any(errs):
            raise next(e for e in errs if e)
        log(f"stage {i + 1}/{len(stages)} done", t0)


def recipe_steps(rec):
    """Step keys of a recipe in run order (sequential 'steps' or the chains of 'stages')."""
    if "stages" in rec:
        return [e.partition("@")[0] for c in rec.get("early", []) for e in c] + \
            [e.partition("@")[0] for stage in rec["stages"] for c in stage for e in c]
    return list(rec["steps"])


def check_inputs(a, rec, tr, te, pub_dir, t0):
    """Commit run of a --cheap-commit notebook: checks what the hidden-test rerun needs and returns placeholder labels
    (the most frequent training label; the save run is never scored). Every failure is raised (no fallback).
    a.check_inputs 'local' (tests outside Kaggle) skips the LM model and GPU checks and accepts the local data copies."""
    assert len(te) < 40, "--check-inputs is for the commit run on the public 3-row test only"
    kaggle = a.check_inputs == "kaggle"
    keys = recipe_steps(rec)
    if any("pub" in STEPS[k][2] and "--ext" in STEPS[k][2] for k in keys):
        assert pub_dir or not kaggle, "public generated passages not attached (build with --pub-datasets)"
    if rec.get("needs_pool"):
        assert POOL_DIR or not kaggle, "public passage pool not attached (build with --pool-dataset)"
        from ext_pool import load                       # prints a warning if the attached pool is another version
        log(f"pool: {len(load(root=POOL_DIR or None))} training rows", t0)
    if rec.get("needs_pool5"):
        assert POOL5_DIR or not kaggle, "120k pool not attached (build with --pool5-dataset)"
        from ext_pool5 import load as load5             # prints a warning if the attached pool is another version
        log(f"pool v5: {len(load5(root=POOL5_DIR or None))} training rows", t0)
    scripts = {STEPS[k][1] for k in keys}
    for slug in sorted({x[6:-1] for k in keys for x in STEPS[k][2] if x.startswith("{JOBS:")}):
        d = jobs_dir(slug)                              # attached LoRA training output (adapters needed on Kaggle)
        assert not kaggle or all(os.path.isdir(f"{d}/f{k}/adapter") for k in range(5)), f"{slug}: adapters missing"
        log(f"LoRA jobs {slug}: {d}", t0)
    if rec["needs_lm"]:
        scripts.add("gpu_embed.py")
        from submission_pipeline_gpu import find_train_lm
        for hint in sorted({h for _, h in LM_MODELS} | {h for _, h, _ in rec.get("lm_extra", ())}):
            log(f"LM {hint}: {lm_dir(hint) if kaggle else 'not checked (local)'}", t0)
        log(f"train LM features: {find_train_lm(a.train_lm) or 'not attached (the rerun extracts them)'}", t0)
        import torch
        assert torch.cuda.is_available() or not kaggle, "no GPU in the save run: the rerun needs one"
        log(f"GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}", t0)
    import importlib                                    # local tests passed with other versions than Kaggle's
    pkgs = ["numpy", "pandas", "scipy", "sklearn"] + (["torch", "transformers"] if rec["needs_lm"] else []) + \
        (["peft"] if "lora24_infer.py" in scripts else [])
    log("versions: " + ", ".join(f"{m} {importlib.import_module(m).__version__}" for m in pkgs), t0)
    mods = sorted(s[:-3] for s in scripts)
    r = subprocess.run([sys.executable, "-c", "import " + ", ".join(mods)], cwd=str(HERE), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"step scripts do not import:\n{r.stdout[-3000:]}")
    log(f"step scripts import: {', '.join(mods)}", t0)
    log("commit run: inputs checked; placeholder submission (the full replay runs in the competition rerun)", t0)
    return np.full(len(te), tr["model"].mode()[0], dtype=object)


def write_out(a, te, labels, t0):
    """Write and validate a.out (ids of test.csv in the order of sample_submission.parquet when it matches)."""
    from common import CLASSES
    import submission as sub_check  # reuse/airline/submission.py
    sample = None
    sp = Path(a.data_dir) / "sample_submission.parquet"
    if sp.exists():
        ss = pd.read_parquet(sp)
        if set(ss["id"].astype(str)) == set(te["id"]):
            sample = ss.assign(id=ss["id"].astype(str))
    counts = sub_check.write_submission(a.out, te["id"].values, labels, sample=sample, id_col="id", target_col="model",
                                        allowed=CLASSES, min_classes=1 if len(te) < 40 else 2)
    out = pd.read_parquet(a.out)
    assert list(out.columns) == ["id", "model"] and len(out) == len(te) and out["id"].is_unique
    assert out["model"].notna().all() and set(out["model"]) <= set(CLASSES) and set(out["id"]) == set(te["id"])
    log(f"wrote {a.out}: {len(out)} rows, {out['model'].nunique()} labels (recipe {a.recipe})", t0)
    print(counts.to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--tokenizers-flat", required=True)
    ap.add_argument("--out", default="submission.parquet")
    ap.add_argument("--work", default="/tmp/nllm_work")
    ap.add_argument("--train-lm", default="")
    ap.add_argument("--precomputed-lm", default="", help="LOCAL: link emb/lp/meta npy+json from this dir")
    ap.add_argument("--no-fallback", action="store_true", help="LOCAL CHECKS: raise instead of falling back")
    ap.add_argument("--ext", default="none", choices=["none", "ntl_short"], help="external training rows (off)")
    ap.add_argument("--ext-weight", type=float, default=0.3)
    ap.add_argument("--ext-dir", default="auto")
    ap.add_argument("--pub-dir", default="auto", help="root of the attached public passage datasets (auto: /kaggle/input "
                                                       "when they are attached, else the local data/external/pub)")
    ap.add_argument("--pool-dir", default="auto", help="directory of the attached public passage pool (auto: found under "
                                                        "/kaggle/input, else the local data/external/pools/prvsiyan)")
    ap.add_argument("--pool5-dir", default="auto", help="directory of the attached 120k pool (auto: found under "
                                                         "/kaggle/input, else the local data/external/pools/dhruv)")
    ap.add_argument("--check-inputs", nargs="?", const="kaggle", default="", choices=["kaggle", "local"],
                    help="commit run of a --cheap-commit notebook: check the inputs, GPU and step imports, write a "
                         "placeholder (3-row public test only); 'local': without the LM model and GPU checks")
    a = ap.parse_args()
    t0 = time.time()
    rec = RECIPES[a.recipe]
    ext_args, sfx = [], ""
    if a.ext != "none":
        if a.recipe != "cpu2":
            ap.error("--ext is wired for the cpu2 recipe only (the v4 combiners load fixed member tags)")
        ext_args = ["--ext", a.ext, "--ext-weight", f"{a.ext_weight:g}", "--ext-dir", str(find_ext_dir(a.ext_dir))]
        sfx = f"_ext-{a.ext}-w{a.ext_weight:g}"
        log(f"EXTERNAL training rows ON: {' '.join(ext_args)}", t0)
    tr, te = setup_project(a, t0)
    pub_dir = find_pub_dir(a.pub_dir)
    global POOL_DIR, POOL5_DIR
    POOL_DIR, POOL5_DIR = find_pool_dir(a.pool_dir), find_pool5_dir(a.pool5_dir)
    if rec.get("needs_pool"):
        log(f"public passage pool read from {POOL_DIR or 'data/external/pools/prvsiyan (local)'}", t0)
    if rec.get("needs_pool5"):
        log(f"120k public passage pool read from {POOL5_DIR or 'data/external/pools/dhruv (local)'}", t0)
    if a.check_inputs:
        write_out(a, te, check_inputs(a, rec, tr, te, pub_dir, t0), t0)
        return
    try:
        if any("pub" in STEPS[k][2] for k in recipe_steps(rec)):
            log(f"public generated passages read from {pub_dir or 'data/external/pub (local)'}", t0)
        early_ex, early = start_early(rec.get("early", []), t0, ext_args, pub_dir)
        if rec["needs_lm"]:
            setup_lm(a, tr, te, t0, rec.get("lm_extra", ()))
        if "stages" in rec:
            run_stages(rec["stages"], t0, ext_args, pub_dir, early, rec.get("early_join"))
        if early_ex is not None:
            early_ex.shutdown(wait=True)
        for key in rec.get("steps", []):
            run_step(key, t0, ext_args if key in EXT_STEPS else (), pub_dir)
        from common import CLASSES, K, encode_y, load_member, metrics
        w ={m + sfx: wt for m, wt in rec["weights"].items()}
        oofs = {m: load_member(m)[0] for m in w}
        tests = {m: load_member(m)[1] for m in w}
        assert all(t.shape == (len(te), K) for t in tests.values()), "member test predictions have the wrong shape"

        def blend(P):
            z = sum(wt * np.log(np.clip(P[m], 1e-9, 1)) for m, wt in w.items()) / sum(w.values())
            return z.argmax(1), z
        y = encode_y(tr["model"])
        for m in w:
            log(f"member {m}: in-kernel OOF BA {metrics.balanced_accuracy(y, oofs[m]):.4f}", t0)
        oof_pred, _ = blend(oofs)
        log(f"in-kernel OOF BA of the fixed-weight blend: {metrics.balanced_accuracy(y, np.eye(K)[oof_pred]):.4f} "
            f"(local nested CV {rec['cv']})", t0)
        pred, z = blend(tests)
        clusters = None
        for tag, wt in rec.get("twins", {}).items():          # test-time adaptive centering (no labels used)
            tw = np.load(ROOT / "outputs" / "twins" / f"{tag}.npz")
            assert clusters is None or (tw["clusters"] == clusters).all(), "twin files with different clusterings"
            clusters = tw["clusters"]
            z = z + wt * tw["evidence"]
            pred = z.argmax(1)
            log(f"twins {tag}: weight {wt} on {int(tw['adapt'].sum())} of {len(z)} test texts; "
                f"{int((pred != blend(tests)[0]).sum())} predictions changed", t0)
        if rec.get("hungarian"):                              # one model per text inside each test cluster of m..18
            from adaptive_sim import hungarian
            assert clusters is not None, "hungarian needs the twins' test clusters"
            before = pred
            pred = hungarian(z, clusters, rec["hungarian"], 18)
            sizes = np.bincount(clusters)[clusters]
            log(f"hungarian: {int(((sizes >= rec['hungarian']) & (sizes <= 18)).sum())} test texts in clusters of "
                f"{rec['hungarian']}..18; {int((pred != before).sum())} predictions changed", t0)
        labels = np.asarray(CLASSES)[pred]
    except Exception as e:
        if a.no_fallback:
            raise
        import traceback
        traceback.print_exc()
        print(f"[replay] recipe {a.recipe} FAILED ({type(e).__name__}: {e}); falling back", flush=True)
        fb = (["submission_pipeline_gpu.py", "--work", a.work, "--train-lm", a.train_lm, "--members", "v2raw",
               "--match-lambda", "0"] if rec["needs_lm"] else ["submission_pipeline.py", "--no-match"])
        r = subprocess.run([sys.executable, str(HERE / fb[0]), "--data-dir", a.data_dir, "--folds", a.folds,
                            "--tokenizers-flat", a.tokenizers_flat, "--out", a.out, *fb[1:]])
        sys.exit(r.returncode)
    write_out(a, te, labels, t0)


if __name__ == "__main__":
    main()

"""Fit test v2 evaluation (rules fixed in PLAN.md before generation).

Generated rows: Kaggle Benchmarks outputs (data/external/ft2/kaggle_out/**/ft2_<split>_<label>.jsonl) and Featherless
outputs (data/external/ft2/featherless_<split>.jsonl), joined to rows_<split>.csv. Kaggle texts get the same cleaning
as the Featherless ones (newlines -> spaces, markdown markers removed; counts recorded). Each generated text is
classified by the fold model that never saw its prompt (fx_predict.member_predictions: ng_fw + ng_fwsh + style2 CPU
blend); the official texts of the same prompts are scored by the same members' OOF predictions.

Per (label, template): n, own-label accuracy, own-lab accuracy, mean P(own label) for generated vs real (same prompts)
and r = mean P(own label | generated) / mean P(own label | real).
  dev:  template with the highest median r over the labels (ties within 0.05 -> T0)
  eval: a label passes if r >= 0.7
Outputs: data/external/ft2/eval_<split>.json, pred_<split>.csv

Usage: python ft2_eval.py dev
       python ft2_eval.py eval
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from common import CLASSES, encode_y, lab_of, load_folds, load_train
from ft2_rows import OUT
from fx_generate import clean
import fx_predict

R_PASS = 0.7
TIE = 0.05


def load_generated(split: str) -> pd.DataFrame:
    rows = pd.read_csv(OUT / f"rows_{split}.csv", keep_default_na=False).set_index("row_id")
    recs = []
    for p in sorted(glob.glob(str(OUT / "kaggle_out" / "**" / f"ft2_{split}_*.jsonl"), recursive=True)):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("row_id") in rows.index and r.get("text"):
                    text, info = clean(r["text"])
                    recs.append({"row_id": r["row_id"], "text": text, "source": "kaggle", **info,
                                 "cost_usd": (r.get("cost_nanodollars") or 0) / 1e9, "reasoning": r.get("reasoning")})
    fp = OUT / f"featherless_{split}.jsonl"
    if fp.exists():
        for line in open(fp, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                recs.append({"row_id": r["row_id"], "text": r["text"], "source": "featherless",
                             "newlines_removed": r.get("newlines_removed", 0),
                             "markdown_removed": r.get("markdown_removed", 0),
                             "cost_usd": 0.0 if r.get("reused_from") else r.get("cost_usd", 0.0),
                             "reasoning": "low"})
    g = pd.DataFrame(recs).drop_duplicates("row_id", keep="last")
    g = g.join(rows[["template", "prompt_id", "fold", "label", "target_words"]], on="row_id")
    g["words"] = g["text"].str.split().str.len()
    g = g[g["words"] >= 5].reset_index(drop=True)
    g["id"] = g["row_id"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "eval"])
    a = ap.parse_args()
    tr = load_train()
    y, folds = encode_y(tr["model"]), load_folds(tr)
    g_ids = tr["prompt_id"].to_numpy()
    gen = load_generated(a.split)
    print(f"{len(gen)} generated texts: {gen.groupby(['template', 'source']).size().to_dict()}", flush=True)
    M = fx_predict.member_predictions(tr, y, folds, g_ids, gen)
    oof = fx_predict.blend({m: v[0] for m, v in M.items()})
    pg = fx_predict.blend({m: v[1] for m, v in M.items()})
    gen["pred"] = np.asarray(CLASSES)[pg.argmax(1)]
    gen["p_label"] = pg[np.arange(len(gen)), [CLASSES.index(l) for l in gen["label"]]]
    real_pred = np.asarray(CLASSES)[oof.argmax(1)]
    rep = []
    for (lab, t), s in gen.groupby(["label", "template"]):
        li = CLASSES.index(lab)
        rr = (tr["model"] == lab).to_numpy() & tr["prompt_id"].isin(s["prompt_id"]).to_numpy()
        real_p = float(oof[rr, li].mean())
        rep.append({"label": lab, "template": t, "n": len(s),
                    "gen_acc": round(float((s["pred"] == lab).mean()), 3),
                    "real_acc": round(float((real_pred[rr] == lab).mean()), 3),
                    "gen_lab_acc": round(float(s["pred"].map(lab_of).eq(lab_of(lab)).mean()), 3),
                    "real_lab_acc": round(float(pd.Series(real_pred[rr]).map(lab_of).eq(lab_of(lab)).mean()), 3),
                    "gen_p": round(float(s["p_label"].mean()), 4), "real_p": round(real_p, 4),
                    "r": round(float(s["p_label"].mean()) / max(real_p, 1e-9), 3),
                    "top_pred": s["pred"].value_counts().index[0],
                    "len_ratio": round(float((s["words"] / s["target_words"]).median()), 2),
                    "md_share": round(float((s["markdown_removed"] > 0).mean()), 2),
                    "cost_usd": round(float(s["cost_usd"].sum()), 4)})
    R = pd.DataFrame(rep)
    out = {"split": a.split, "rows": R.to_dict("records")}
    pd.set_option("display.width", 250)
    if a.split == "dev":
        med = R.groupby("template")["r"].median().sort_values(ascending=False)
        best = med.index[0]
        if best != "T0" and "T0" in med and med[best] - med["T0"] < TIE:
            best = "T0"
        out.update({"median_r": med.round(3).to_dict(), "chosen_template": best})
        print(R.pivot(index="label", columns="template", values="r").to_string())
        print("median r per template:", med.round(3).to_dict(), "-> chosen", best)
    else:
        R["pass"] = R["r"] >= R_PASS
        out.update({"passing_labels": R.loc[R["pass"], "label"].tolist()})
        print(R.to_string(index=False))
        print("passing labels (r >= 0.7):", out["passing_labels"])
    (OUT / f"eval_{a.split}.json").write_text(json.dumps(out, indent=1))
    gen[["row_id", "label", "template", "prompt_id", "fold", "pred", "p_label", "words", "source"]].to_csv(
        OUT / f"pred_{a.split}.csv", index=False)


if __name__ == "__main__":
    main()

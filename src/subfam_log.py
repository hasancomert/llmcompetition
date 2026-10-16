"""Log the standalone hard-group specialist heads (subfam_spec.py outputs) to EXPERIMENTS.md, one row per view.

A view alone is scored as an 18-class per-text prediction: P = blend_pt_gpu_ba (nested OOF), and inside each of the
non-overlapping groups gem78, glm, gptlst, claude the group mass of P is kept and the head replaces P inside the
group, P(m) = P(G) * S(m | G). The note lists the in-group balanced accuracy / conditional NLL of every group file
of the view (all C of the grid) next to the in-group BA of blend_pt_gpu_ba itself.

Usage: python subfam_log.py --view emb_qwen3_8b_base --C Cin [--view ... --C ...]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from common import encode_y, load_folds, load_member, load_train, log_experiment, scores
from family_spec import SPEC
from subfam_spec import GROUPS, group_cols, in_group

REF = "blend_pt_gpu_ba"
REPLACE = ["gem78", "glm", "gptlst", "claude"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="append", required=True)
    ap.add_argument("--C", action="append", required=True)
    a = ap.parse_args()
    tr = load_train()
    y, folds = encode_y(tr["model"]), load_folds(tr)
    B, _ = load_member(REF)
    ref_ig = {gn: in_group(y, B, group_cols(gn))[0] for gn in GROUPS}
    for view, C in zip(a.view, a.C):
        P = B.copy()
        for gn in REPLACE:
            cols = group_cols(gn)
            S = np.load(SPEC / f"grp_{gn}_{view}_C{C}.npz")["oof"][:, cols]
            P[:, cols] = B[:, cols].sum(1, keepdims=True) * S / S.sum(1, keepdims=True)
        s = scores(y, P, folds)
        parts = []
        for gn in GROUPS:
            files = sorted(Path(SPEC).glob(f"grp_{gn}_{re.escape(view)}_C*.npz"))
            files = [f for f in files if re.fullmatch(rf"grp_{re.escape(gn)}_{re.escape(view)}_C[0-9.e+-]+|grp_{re.escape(gn)}_{re.escape(view)}_Cin", f.stem)]
            if not files:
                continue
            vals = []
            for f in files:
                ba, nll = in_group(y, np.load(f)["oof"], group_cols(gn))
                vals.append(f"{f.stem.split('_C')[-1]}:{ba:.3f}/{nll:.3f}")
            parts.append(f"{gn} (blend {ref_ig[gn]:.3f}) " + " ".join(vals))
        note = ("standalone group heads, in-group BA/cond NLL per C: " + "; ".join(parts)
                + f". 18-class score = {REF} with the C{C} heads replacing it inside gem78/glm/gptlst/claude. "
                + "NOT helpful alone: in-group BA at the level of the blend (saturated); used only as combiner views")
        log_experiment(f"grpspec_{view}", f"hard-group specialist heads {view} (subfam_spec.py; LogReg on the group's "
                       f"training-fold rows only, groups {', '.join(GROUPS)})", s, None, note=note)
        print(f"{view}: BA {s['ba']:.4f} ll {s['logloss']:.4f}")


if __name__ == "__main__":
    main()

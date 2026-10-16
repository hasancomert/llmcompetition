"""Family-specialist heads on the style2 tables of train_lpx.py (per-text), as family_spec.py's lpstyle view.

Blocks are train_lpx.build_blocks names (style, style2, lp4, glm); columns with the dropped prefixes are removed
(default s2_pb_, as in the saved lpx members). For every outer fold and family: QuantileTransformer + scaler +
LogReg fit on that family's training-fold rows only (family_spec.table_view), applied to every held-out / test row.
Output: outputs/famspec/[sup_]<view>_C<C>.npz, view = 'lpx_' + blocks joined by '+'.
FAMSPEC_GROUPS=super for super-families. Nothing clusters, centers or matches held-out / test texts.
Usage: python family_spec_style2.py --blocks style+style2 --Cs 0.003,0.01,0.03
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import encode_y, load_folds, load_test, load_train
from family_spec import save, table_view, within_family_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="style+style2")
    ap.add_argument("--Cs", default="0.003,0.01,0.03")
    ap.add_argument("--drop", default="s2_pb_")
    a = ap.parse_args()
    from train_lpx import build_blocks
    t0 = time.time()
    tr, te = load_train(), load_test()
    y, folds = encode_y(tr["model"]), load_folds(tr)
    Xtr, Xte, names = build_blocks(a.blocks.split("+"), tr, te)
    drops = tuple(p for p in a.drop.split(",") if p)
    keep = [i for i, n in enumerate(names) if not (drops and n.startswith(drops))]
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    print(f"{a.blocks}: {Xtr.shape[1]} columns", flush=True)
    res = table_view(Xtr, Xte, y, folds, [float(c) for c in a.Cs.split(",")])
    for C, (oof, test) in res.items():
        name = save("lpx_" + a.blocks, C, oof, test)
        ba, nll, txt = within_family_report(y, oof, name)
        print(f"{txt} | mean in-family BA {ba:.4f} cond NLL {nll:.4f}", flush=True)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

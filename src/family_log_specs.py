"""Log the standalone family-specialist heads (family_spec.py outputs) to EXPERIMENTS.md, one row per view.

A specialist alone is scored as an 18-class per-text prediction p(m) = B(F) * S(m | F), with B = the nested OOF of
blend_v3_emb_logloss (its family masses are kept, the specialist replaces the blend inside the family). The note
lists, for every C of the grid, the in-family balanced accuracy / conditional NLL (rows of multi-model families,
block of the TRUE family) and the 18-class BA / log loss; the row's scores are those of the C with the best
in-family BA. Reference in-family numbers of blend_v3_emb_logloss itself are printed too.

Usage: python family_log_specs.py [--groups lab|super]   (set FAMSPEC_GROUPS accordingly)
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from common import encode_y, load_folds, load_member, load_train, log_experiment, scores
from family_spec import FAM_OF_CLASS, FAMS, GROUPING, SPEC, within_family_report


def main():
    tr = load_train()
    y, folds = encode_y(tr["model"]), load_folds(tr)
    B, _ = load_member("blend_v3_emb_logloss")
    FM = np.stack([B[:, FAM_OF_CLASS == f].sum(1) for f in range(len(FAMS))], 1)
    rba, rnll, _ = within_family_report(y, B)
    print(f"blend_v3_emb_logloss in-family BA {rba:.4f} cond NLL {rnll:.4f} ({GROUPING})")
    pre = "sup_" if GROUPING == "super" else ""
    views = defaultdict(list)
    for p in sorted(SPEC.glob("*.npz")):
        nm = p.stem
        if (GROUPING == "super") != nm.startswith("sup_"):
            continue
        m = re.match(r"(.*)_C([0-9.e-]+)$", nm)
        views[m.group(1)].append((float(m.group(2)), nm))
    for view, items in views.items():
        rows = []
        for C, nm in sorted(items):
            S = np.load(SPEC / f"{nm}.npz")["oof"]
            P = FM[:, FAM_OF_CLASS] * S
            s = scores(y, P, folds)
            ba, nll, _ = within_family_report(y, S)
            rows.append((ba, C, nll, s))
        best = max(rows, key=lambda r: r[0])
        grid = "; ".join(f"C{C:g}: in-family BA {ba:.4f} / NLL {nll:.4f}, 18-class BA {s['ba']:.4f} / ll {s['logloss']:.4f}"
                         for ba, C, nll, s in sorted(rows, key=lambda r: r[1]))
        note = (f"standalone specialist ({GROUPING} families) = blend_v3 family mass x specialist; blend_v3 itself: "
                f"in-family BA {rba:.4f} / NLL {rnll:.4f}; {grid}; alone never better than the blend in-family; "
                f"used only as combiner inputs (family_combine.py)")
        log_experiment(f"famspec_{view}", f"family-specialist head {view} (family_spec.py), best C {best[1]:g}",
                       best[3], None, note=note)
        print(f"{view}: best C {best[1]:g} in-family BA {best[0]:.4f}; 18-class BA {best[3]['ba']:.4f} "
              f"ll {best[3]['logloss']:.4f}")


if __name__ == "__main__":
    main()

"""Nested hill-climbing blend of saved members (preview of PLAN.md step E; used here only to measure whether
the baseline families complement each other).

Meta folds = the shared folds.csv split (grouped by prompt), so every blend weight is chosen on 4 folds and
scored on the 5th (reuse/airline/hill_climb.nested_hill, final weights = mean of the fold weights).
Usage: python blend_baselines.py --tags a b c [--space logprob] [--metric ba|logloss] [--log]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import encode_y, iter_folds, load_folds, load_member, load_train, metrics, run_member  # sets sys.path
from compare import fmt, paired_bootstrap

import hill_climb  # noqa: E402  (reuse/airline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--space", default="logprob", choices=["prob", "logprob"])
    ap.add_argument("--metric", default="logloss", choices=["ba", "logloss"])
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    meta = [(trn, val) for _, trn, val in iter_folds(folds)]
    members = [load_member(t) for t in args.tags]
    oofs, tests = [m[0] for m in members], [m[1] for m in members]
    kw = dict(space=args.space)
    if args.metric == "logloss":
        kw.update(metric=metrics.mlogloss, higher_is_better=False)
    r = hill_climb.nested_hill(oofs, y, meta, avg_weights=True, **kw)
    w = r["weights"]
    single = [metrics.balanced_accuracy(y, o) for o in oofs]
    best = int(np.argmax(single))
    print("singles BA:", dict(zip(args.tags, np.round(single, 4))))
    print("final weights (mean of fold weights):", dict(zip(args.tags, np.round(w, 3))))
    print("fold weights:", [dict(zip(args.tags, np.round(fw, 2))) for fw in r["fold_weights"]])
    nested = r["nested_pred"]
    print(f"nested blend: BA {metrics.balanced_accuracy(y, nested):.4f}, log loss {metrics.mlogloss(y, nested):.4f}")
    print(f"best single ({args.tags[best]}) vs nested blend: "
          + fmt(paired_bootstrap(y, oofs[best], nested, tr['prompt_id'].values)))
    if args.log:
        test = hill_climb.blend(tests, w, args.space)
        tag = args.tag or "blend_" + "+".join(args.tags)
        run_member(tag, f"nested hill-climb blend ({args.space}, weights by {args.metric}) of {', '.join(args.tags)}",
                   nested, test, y, folds, t0, note="nested OOF (weights chosen on the other 4 folds); preview of step E",
                   members=args.tags, weights=[float(x) for x in w])


if __name__ == "__main__":
    main()

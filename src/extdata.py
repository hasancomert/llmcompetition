"""Optional external training rows for the per-text members (the on/off flag of the extra-data workstream).

  --ext none        (default) official train only: every member behaves exactly as before
  --ext ntl_short   Kaggle olegzholobov/name-that-llm-response-corpus, short view (ext_corpus.load: competition
                    format, 40-250 words, de-duplicated; quarantine_luna_* never used)
  --ext ft2         fit test v2 generated texts (ft2_eval.load_generated, dev + eval splits, one --ext-template);
                    each answers a reconstructed TRAIN prompt, so it is left out whenever that prompt_id is validated

Rules (EXTERNAL_DATA.md): external rows only ever enter TRAINING sets. Validation / OOF rows are official texts only,
and inner temperature splits validate on official rows only. Rows whose topic matches an official prompt_id
(ext_leak.compute, inclusive threshold) are left out of every training set whose validation part contains that
prompt_id. Each external row has weight --ext-weight (official rows 1); --ext-licenses / --ext-drop-sources select
rows. In a Kaggle notebook the dataset is attached as an input and --ext-dir points at it (no copy, no re-upload).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import CLASS_INDEX, ROOT

CHOICES = ("none", "ntl_short", "ft2")


def add_args(ap):
    g = ap.add_argument_group("external training data (training folds only)")
    g.add_argument("--ext", default="none", choices=CHOICES)
    g.add_argument("--ext-weight", type=float, default=0.3, help="sample weight of an external row (official rows 1)")
    g.add_argument("--ext-licenses", default="open,open_terms,unknown",
                   help="license classes to keep (ext_corpus.LICENSE_CLASS)")
    g.add_argument("--ext-drop-sources", default="", help="';'-separated sources to leave out")
    g.add_argument("--ext-drop-labels", default="", help="';'-separated labels whose external rows are left out")
    g.add_argument("--ext-dir", default="", help="dataset directory (default data/external/ntl_corpus)")
    g.add_argument("--ext-limit", type=int, default=0, help="smoke tests: keep a random subset of this size")
    g.add_argument("--ext-template", default="T0", help="--ext ft2: generation template to use")


def tag_suffix(a) -> str:
    if getattr(a, "ext", "none") == "none":
        return ""
    s = f"_ext-{a.ext}{'-' + a.ext_template if a.ext == 'ft2' else ''}-w{a.ext_weight:g}"
    if a.ext_licenses != "open,open_terms,unknown":
        s += "-" + a.ext_licenses.replace(",", "+")
    if a.ext_drop_sources:
        s += f"-drop{len(a.ext_drop_sources.split(';'))}"
    if getattr(a, "ext_drop_labels", ""):
        s += f"-droplab{len(a.ext_drop_labels.split(';'))}"
    if a.ext_limit:
        s += f"-n{a.ext_limit}"
    return s


def describe(a) -> str:
    if getattr(a, "ext", "none") == "none":
        return ""
    return (f"; + EXTERNAL training rows ({a.ext}, weight {a.ext_weight:g}, licenses {a.ext_licenses}"
            f"{', dropped sources ' + a.ext_drop_sources if a.ext_drop_sources else ''}"
            f"{', dropped labels ' + a.ext_drop_labels if getattr(a, 'ext_drop_labels', '') else ''}; training folds only, "
            f"topic-matched rows left out when their prompt_id is validated)")


@dataclass
class Ext:
    df: pd.DataFrame          # ext_id, text, label, source, license_class, prompt_group_id
    y: np.ndarray
    groups: np.ndarray        # 'ext:<prompt_group_id>' (never equal to an official prompt_id)
    w: np.ndarray
    leak: list                # per row: set of official prompt_ids it may answer

    def __len__(self):
        return len(self.df)

    @property
    def texts(self) -> list:
        return self.df["text"].tolist()

    def allowed(self, val_pids) -> np.ndarray:
        """Rows that may train a model whose validation part contains the official prompt_ids val_pids."""
        v = set(val_pids)
        return np.array([not (s & v) for s in self.leak], dtype=bool)


def load(a, tr: pd.DataFrame | None = None) -> Ext | None:
    if getattr(a, "ext", "none") == "none":
        return None
    if a.ext == "ft2":
        return _load_ft2(a)
    import ext_corpus
    import ext_leak
    root = a.ext_dir or None
    df = ext_corpus.load(root=root)
    keep = df["license_class"].isin(a.ext_licenses.split(","))
    if a.ext_drop_sources:
        keep &= ~df["source"].isin(a.ext_drop_sources.split(";"))
    if getattr(a, "ext_drop_labels", ""):
        keep &= ~df["label"].isin(a.ext_drop_labels.split(";"))
    df = df[keep].reset_index(drop=True)
    if a.ext_limit and a.ext_limit < len(df):
        df = df.sample(n=a.ext_limit, random_state=0).reset_index(drop=True)
    tr = tr if tr is not None else pd.read_csv(ROOT / "data" / "train.csv")
    cache = ext_corpus.D / "leak_matches.csv"
    if not root and cache.exists():
        lm = pd.read_csv(cache, keep_default_na=False)
    else:                                                     # notebook / fresh root: recompute (deterministic)
        lm, _ = ext_leak.compute(tr, ext_corpus.load(root=root))
    gp = dict(zip(lm["ext_id"], lm["group_pids"]))
    leak = [set(p for p in str(gp.get(e, "")).split(";") if p) for e in df["ext_id"]]
    y = df["label"].map(CLASS_INDEX).to_numpy()
    w = np.full(len(df), float(a.ext_weight))
    n_leak = sum(bool(s) for s in leak)
    print(f"[extdata] {a.ext}: {len(df)} external training rows (weight {a.ext_weight:g}; licenses {a.ext_licenses}); "
          f"{n_leak} topic-matched to official prompts (left out when validated)", flush=True)
    return Ext(df=df, y=y, groups=("ext:" + df["prompt_group_id"].astype(str)).to_numpy(), w=w, leak=leak)


def _load_ft2(a) -> Ext:
    import ft2_eval
    from ft2_rows import OUT
    parts = [ft2_eval.load_generated(sp) for sp in ("dev", "eval") if (OUT / f"rows_{sp}.csv").exists()]
    df = pd.concat(parts, ignore_index=True)
    df = df[df["template"] == a.ext_template]
    if getattr(a, "ext_drop_labels", ""):
        df = df[~df["label"].isin(a.ext_drop_labels.split(";"))]
    df = df.rename(columns={"row_id": "ext_id"}).reset_index(drop=True)
    df["prompt_group_id"] = df["prompt_id"]
    df["license_class"] = "generated"
    leak = [{p} for p in df["prompt_id"]]                     # answers that train prompt: out while it is validated
    y = df["label"].map(CLASS_INDEX).to_numpy()
    w = np.full(len(df), float(a.ext_weight))
    print(f"[extdata] ft2 ({a.ext_template}): {len(df)} generated training rows over {df['prompt_id'].nunique()} prompts, "
          f"{df['label'].nunique()} labels (weight {a.ext_weight:g}); each left out while its prompt is validated",
          flush=True)
    return Ext(df=df, y=y, groups=("ext:" + df["prompt_id"].astype(str)).to_numpy(), w=w, leak=leak)

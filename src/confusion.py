"""Confusion analysis of an OOF member (EDA.md §8): per-class recall, lab-level accuracy, most confused pairs.

Usage: python confusion.py <tag> [<tag> ...]   ("a,b" averages members first)
Writes outputs/eda/confusion_<tag>.md and prints it.
"""
from __future__ import annotations

import sys

import numpy as np

from common import CLASSES, OUT, encode_y, lab_of, load_member, load_train


def report(tag: str) -> str:
    tr = load_train()
    y = encode_y(tr["model"])
    P, _ = load_member(tag)
    pred = P.argmax(1)
    k = len(CLASSES)
    C = np.zeros((k, k), dtype=int)
    np.add.at(C, (y, pred), 1)
    rec = C.diagonal() / C.sum(1)
    labs = np.array([lab_of(c) for c in CLASSES])
    lab_ok = labs[y] == labs[pred]
    lines = [f"### Karışıklık: `{tag}`", "",
             f"OOF BA {rec.mean():.4f}; tahmin edilen lab doğru: %{100 * lab_ok.mean():.1f} "
             f"(lab içi yanlış: %{100 * (lab_ok & (y != pred)).mean():.1f}, lab dışı yanlış: %{100 * (~lab_ok).mean():.1f}).", "",
             "| model | recall | en çok karıştırıldığı (sayı) | bu sınıfa tahmin edilen toplam |", "|---|---|---|---|"]
    for i in np.argsort(rec):
        off = C[i].copy()
        off[i] = 0
        top = np.argsort(-off)[:2]
        lines.append(f"| {CLASSES[i]} | {rec[i]:.3f} | " + "; ".join(f"{CLASSES[j]} ({off[j]})" for j in top) +
                     f" | {C[:, i].sum()} |")
    S = C + C.T
    np.fill_diagonal(S, 0)
    iu = np.triu_indices(k, 1)
    order = np.argsort(-S[iu])[:12]
    lines += ["", "En sık karışan 12 çift (iki yönün toplamı; ✓ = aynı lab):", ""]
    lines.append(", ".join(f"{CLASSES[iu[0][o]]} ↔ {CLASSES[iu[1][o]]} {S[iu][o]}"
                           f"{' ✓' if labs[iu[0][o]] == labs[iu[1][o]] else ''}" for o in order))
    lab_names = sorted(set(labs))
    L = np.zeros((len(lab_names), len(lab_names)), dtype=int)
    li = {l: i for i, l in enumerate(lab_names)}
    for a, b in zip(labs[y], labs[pred]):
        L[li[a], li[b]] += 1
    lines += ["", "Lab düzeyinde karışıklık (satır = gerçek lab, sütun = tahmin; satır yüzdesi):", "",
              "| gerçek \\ tahmin | " + " | ".join(lab_names) + " |", "|---|" + "---|" * len(lab_names)]
    for i, l in enumerate(lab_names):
        lines.append(f"| {l} | " + " | ".join(f"{100 * v / L[i].sum():.0f}" for v in L[i]) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        txt = report(tag)
        (OUT / "eda").mkdir(parents=True, exist_ok=True)
        (OUT / "eda" / f"confusion_{tag.replace(',', '+')}.md").write_text(txt, encoding="utf-8")
        print(txt)

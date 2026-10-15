"""Every number in EDA.md comes from this script. Output: outputs/eda/eda_tables.md (+ csv matrices).

Sections: (1) (prompt, model) pairs, (2) length per model, (3) format traces per model, (4) start / end
patterns, (5) similarity between models (same lab vs other labs), (6) distinctive words per model.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from common import CLASSES, OUT, lab_of, load_test, load_train
from style_features import last_char_class, style_frame

EDA = OUT / "eda"
md: list[str] = []


def h(title):
    md.append(f"\n## {title}\n")


def table(df: pd.DataFrame, floatfmt="{:.2f}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    md.append("| " + " | ".join(map(str, [df.index.name or ""] + list(df.columns))) + " |")
    md.append("|" + "---|" * (len(df.columns) + 1))
    for idx, r in df.iterrows():
        md.append("| " + " | ".join(map(str, [idx] + list(r.values))) + " |")
    md.append("")


def pairs(tr):
    h("1. (prompt_id, model) çiftleri")
    ct = pd.crosstab(tr.prompt_id, tr.model)
    vals, cnts = np.unique(ct.values, return_counts=True)
    md.append(f"- Satır {len(tr)}, prompt {tr.prompt_id.nunique()}, model {tr.model.nunique()}; "
              f"tam ızgara {ct.shape[0]} × {ct.shape[1]} = {ct.size} hücre.")
    md.append("- Hücre değerleri: " + ", ".join(f"{v} → {c} hücre" for v, c in zip(vals, cnts)) + ".")
    miss = [(p, m) for p in ct.index for m in ct.columns if ct.loc[p, m] == 0]
    dup = [(p, m, int(ct.loc[p, m])) for p in ct.index for m in ct.columns if ct.loc[p, m] > 1]
    md.append(f"- Eksik çiftler: {miss if miss else 'yok'}; birden fazla olan: {dup if dup else 'yok'}.")
    per_prompt = tr.groupby("prompt_id").size().value_counts().sort_index()
    md.append("- Prompt başına metin sayısı: " + ", ".join(f"{k} metin → {v} prompt" for k, v in per_prompt.items()) + ".")
    md.append(f"- Tekrarlanan metin (birebir aynı): {int(tr.text.duplicated().sum())}; id tekil: {tr.id.is_unique}.")
    md.append(f"- Prompt kimlikleri: {sorted(tr.prompt_id.unique())[:3]} … {sorted(tr.prompt_id.unique())[-3:]}.")


def lengths(tr, S):
    h("2. Model başına metin sayısı ve uzunluk (kelime = boşlukla ayrılmış parça)")
    w = tr.text.str.split().str.len()
    tr = tr.assign(words=w, chars=tr.text.str.len())
    g = tr.groupby("model")
    t = pd.DataFrame({
        "n": g.size(), "mean": g.words.mean(), "std": g.words.std(), "min": g.words.min(),
        "p10": g.words.quantile(0.1), "median": g.words.median(), "p90": g.words.quantile(0.9),
        "max": g.words.max(), "chars_mean": g.chars.mean(),
        "<40 %": g.words.apply(lambda s: 100 * (s < 40).mean()), ">250 %": g.words.apply(lambda s: 100 * (s > 250).mean()),
        "paragraf ort.": g.apply(lambda d: S.loc[d.index, "n_paragraphs"].mean()),
        "cümle ort. kelime": g.apply(lambda d: S.loc[d.index, "sent_mean"].mean()),
    })
    t.index.name = "model"
    table(t.sort_values("mean"), "{:.1f}")
    hist, edges = np.histogram(w, bins=list(range(30, 290, 20)))
    md.append("Kelime sayısı histogramı (20'lik kutular): " + ", ".join(f"{a}–{b - 1}: {c}" for a, b, c in zip(edges[:-1], edges[1:], hist)) + ".")
    md.append("")
    md.append(f"Genel: ortalama {w.mean():.1f}, medyan {w.median():.0f}, min {w.min()}, max {w.max()} kelime; "
              f"<40 kelime {int((w < 40).sum())}, >250 kelime {int((w > 250).sum())} metin.")
    # does the prompt set the length? share of length variance explained by prompt vs model
    lw = np.log(w)
    tot = ((lw - lw.mean()) ** 2).sum()
    by_p = (tr.assign(lw=lw).groupby("prompt_id").lw.transform("mean") - lw.mean()) ** 2
    by_m = (tr.assign(lw=lw).groupby("model").lw.transform("mean") - lw.mean()) ** 2
    md.append(f"log(kelime) varyansının prompt ortalamalarıyla açıklanan payı {by_p.sum() / tot:.3f}, "
              f"model ortalamalarıyla açıklanan payı {by_m.sum() / tot:.3f}.")
    return tr


FORMAT_COLS = [
    ("md_bold", "**kalın**"), ("md_italic", "*italik*"), ("md_header", "# başlık"), ("bullet", "madde (-,*,•)"),
    ("numbered", "numaralı liste"), ("multi_para", ">1 paragraf"), ("em_dash", "em-dash —"),
    ("em_dash_spaced", "boşluklu ' — '"), ("em_dash_tight", "bitişik a—b"), ("en_dash", "en-dash –"),
    ("double_hyphen", "--"), ("spaced_hyphen", "' - '"), ("curly_double", "kıvrık “ ”"), ("curly_single", "kıvrık ‘ ’"),
    ("straight_double", "düz \""), ("straight_apostrophe", "düz a'b"), ("latex_any", "LaTeX"), ("emoji", "emoji"),
    ("ellipsis_char", "…"), ("three_dots", "..."), ("semicolon", ";"), ("colon", ":"), ("exclaim", "!"),
    ("question", "?"), ("paren", "("), ("math_sym", "×≈π° vb."), ("nbsp", "nbsp/ince boşluk"),
    ("not_x_but_y", "not X but Y"), ("contraction", "kısaltma (n't, 's)"), ("first_person_pl", "we/our"),
    ("first_person_sg", "I/my"), ("second_person", "you"),
]


def formats(tr, S):
    h("3. Model başına biçim izleri (% metin: en az bir kez içeren)")
    S = S.copy()
    S["multi_para"] = (S.n_paragraphs > 1).astype(int)
    S["latex_any"] = S.latex_inline + S.latex_paren + S.latex_cmd
    pres = (S[[c for c, _ in FORMAT_COLS]] > 0).groupby(tr.model.values).mean() * 100
    pres.columns = [lab for _, lab in FORMAT_COLS]
    pres.index.name = "model"
    keep = [c for c in pres.columns if pres[c].max() >= 1.0]
    dropped = [c for c in pres.columns if c not in keep]
    table(pres[keep], "{:.0f}")
    md.append(f"Hiçbir modelde %1'e ulaşmayan izler (tablodan çıkarıldı): {', '.join(dropped) if dropped else 'yok'}.")
    md.append(f"Satır sonu (\\n) içeren metin: {int(tr.text.str.contains(chr(10)).sum())}; '#' içeren: "
              f"{int(tr.text.str.contains('#', regex=False).sum())}; '*' içeren: {int(tr.text.str.contains('*', regex=False).sum())}; "
              f"emoji içeren: {int((S.emoji > 0).sum())}; '$' içeren: {int(tr.text.str.contains('$', regex=False).sum())}.")
    md.append("")
    md.append("### 3a. Lab düzeyinde özet (% metin)")
    labs = tr.model.map(lab_of).values
    lab_t = (S[["em_dash", "em_dash_spaced", "curly_single", "curly_double", "straight_apostrophe", "curly_apostrophe", "semicolon"]] > 0).groupby(labs).mean() * 100
    lab_t.columns = ["em-dash", "boşluklu em-dash", "kıvrık ‘ ’", "kıvrık “ ”", "düz a'b", "kıvrık a’b", ";"]
    lab_t.index.name = "lab"
    table(lab_t, "{:.0f}")
    md.append("")
    md.append("### 3b. 100 kelime başına ortalama sayım (noktalama ve tire)")
    rate_cols = ["em_dash_p100w", "comma_p100w", "semicolon_p100w", "colon_p100w", "paren_p100w", "question_p100w",
                 "curly_apostrophe_p100w", "straight_apostrophe_p100w", "hyphen_word_p100w", "contraction_p100w",
                 "not_x_but_y_p100w"]
    rates = S[rate_cols].groupby(tr.model.values).mean()
    rates.columns = [c.replace("_p100w", "") for c in rate_cols]
    rates.index.name = "model"
    table(rates, "{:.2f}")
    extra = S[["nonascii_rate", "ttr", "word_len_mean", "long_word_rate", "titlecase_word_rate"]].groupby(tr.model.values).mean()
    extra["nonascii_rate"] *= 100
    extra.columns = ["ASCII dışı karakter %", "tip/token", "ort. kelime uzunluğu", "≥9 harf kelime oranı", "büyük harfle başlayan kelime oranı"]
    extra.index.name = "model"
    md.append("### 3c. Sözcük düzeyi")
    table(extra, "{:.3f}")
    return S


def starts_ends(tr, S):
    h("4. Başlangıç ve bitiş kalıpları")
    # curly apostrophes normalised so that "it’s" (GPT/Gemini/Kimi/Qwen) and "it's" (Claude/GLM) tokenise alike
    norm = tr.text.str.replace("[’‘]", "'", regex=True).str.lower()
    first2 = norm.str.findall(r"[a-z']+").str[:2].str.join(" ")
    first3 = norm.str.findall(r"[a-z']+").str[:3].str.join(" ")
    # tie-robust: the 3-word opening is shared with at least one OTHER text of the same prompt
    n_same = tr.assign(f3=first3).groupby(["prompt_id", "f3"]).f3.transform("size")
    same3 = n_same >= 2
    lc = pd.Series(last_char_class(tr.text), index=tr.index)
    rows = {}
    for m in CLASSES:
        ix = tr.model == m
        top = first2[ix].value_counts().head(3)
        rows[m] = {
            "en sık ilk 2 kelime (pay %)": "; ".join(f"{k} ({100 * v / ix.sum():.0f})" for k, v in top.items()),
            "ilk 3 kelimesi prompt'taki başka bir metinle ortak %": 100 * same3[ix].mean(),
            "nokta ile biter %": 100 * (lc[ix] == "period").mean(),
            "? ile %": 100 * (lc[ix] == "question").mean(),
            "! ile %": 100 * (lc[ix] == "exclaim").mean(),
            "tırnak/parantez ile %": 100 * (lc[ix] == "quote_or_bracket").mean(),
            "harf/rakamla (yarım) %": 100 * (lc[ix] == "alnum").mean(),
        }
    t = pd.DataFrame(rows).T
    for c in t.columns[1:]:
        t[c] = t[c].astype(float)
    t.index.name = "model"
    table(t, "{:.1f}")
    top_share = tr.assign(n=n_same).groupby("prompt_id").n.max() / tr.groupby("prompt_id").size()
    md.append(f"Bütün metinlerde ilk 3 kelimesi (’ → ' normalize) aynı prompt'taki en az bir başka metinle ortak olanlar: "
              f"%{100 * same3.mean():.1f}. Prompt'ların {int((top_share >= 0.5).sum())}/{tr.prompt_id.nunique()}'ünde "
              f"metinlerin en az yarısı aynı 3 kelimeyle başlıyor; {int((top_share <= 1 / 18 + 1e-9).sum())} prompt'ta "
              f"hiçbir açılış ortak değil.")
    last2 = tr.text.str.rstrip().str[-2:]
    md.append(f"Son iki karakterin en sık değerleri: "
              + ", ".join(f"{k!r} {v}" for k, v in last2.value_counts().head(8).items()) + ".")


def similarity(tr, S):
    h("5. Modeller arası benzerlik (aynı lab ile farklı lab)")
    out = {}
    for name, vec in [
        ("kelime 1–2 gram TF-IDF", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("karakter 3–5 gram TF-IDF", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3)),
    ]:
        X = vec.fit_transform(tr.text)
        sim = np.zeros((len(CLASSES), len(CLASSES)))
        cnt = np.zeros_like(sim)
        for _, d in tr.groupby("prompt_id"):
            idx = d.index.to_numpy()
            mi = np.array([CLASSES.index(m) for m in d.model])
            G = (X[idx] @ X[idx].T).toarray()
            for a, b in itertools.product(range(len(idx)), repeat=2):
                if a != b:
                    sim[mi[a], mi[b]] += G[a, b]
                    cnt[mi[a], mi[b]] += 1
        sim = sim / np.maximum(cnt, 1)
        out[name] = sim
        pd.DataFrame(sim, index=CLASSES, columns=CLASSES).to_csv(EDA / f"sim_{name.split()[0]}.csv")
    # style profile similarity: correlation of z-scored per-model mean feature vectors
    num = S.select_dtypes("number")
    num = num.loc[:, num.std() > 0]
    Z = (num - num.mean()) / num.std()
    prof = Z.groupby(tr.model.values).mean().loc[CLASSES]
    out["stil profili korelasyonu"] = np.corrcoef(prof.values)
    pd.DataFrame(out["stil profili korelasyonu"], index=CLASSES, columns=CLASSES).to_csv(EDA / "sim_style.csv")

    labs = np.array([lab_of(m) for m in CLASSES])
    iu = np.triu_indices(len(CLASSES), 1)
    same = labs[iu[0]] == labs[iu[1]]
    md.append("Aynı prompt'a verilen cevaplar arasında ortalama kosinüs benzerliği (model çifti başına 149–150 prompt "
              "ortalaması) ve model başına ortalama stil profili korelasyonu.")
    md.append("")
    md.append("| ölçü | aynı lab çiftleri ort. | farklı lab çiftleri ort. | en benzer 6 çift |")
    md.append("|---|---|---|---|")
    for name, M in out.items():
        v = M[iu]
        order = np.argsort(-v)[:6]
        tops = "; ".join(f"{CLASSES[iu[0][i]]} ~ {CLASSES[iu[1][i]]} {v[i]:.3f}" for i in order)
        md.append(f"| {name} | {v[same].mean():.3f} | {v[~same].mean():.3f} | {tops} |")
    md.append("")
    md.append("### 5b. Her modelin en yakın komşusu (kelime TF-IDF / karakter TF-IDF / stil profili)")
    md.append("")
    md.append("| model | lab | kelime TF-IDF | karakter TF-IDF | stil profili |")
    md.append("|---|---|---|---|---|")
    names = list(out)
    for i, m in enumerate(CLASSES):
        cells = []
        for name in names:
            M = out[name].copy()
            np.fill_diagonal(M, -np.inf)
            j = int(np.argmax(M[i]))
            cells.append(f"{CLASSES[j]} ({M[i, j]:.3f}){' ✓' if labs[j] == labs[i] else ''}")
        md.append(f"| {m} | {labs[i]} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("✓ = en yakın komşu aynı lab'dan.")
    # within-prompt length agreement between the same-lab pairs of interest
    md.append("")
    md.append("### 5c. Aynı lab içi seçili çiftler: prompt içi uzunluk ilişkisi")
    w = tr.text.str.split().str.len()
    piv = tr.assign(w=w).pivot_table(index="prompt_id", columns="model", values="w")
    md.append("")
    allsp = piv.corr(method="spearman").to_numpy()[np.triu_indices(len(CLASSES), 1)]
    md.append(f"Bütün 153 model çiftinde prompt içi uzunluk Spearman: min {allsp.min():.2f}, max {allsp.max():.2f}, "
              f"ortalama {allsp.mean():.3f}.")
    md.append("")
    md.append("| çift | kelime TF-IDF | karakter TF-IDF | stil korelasyonu | prompt içi uzunluk Spearman |")
    md.append("|---|---|---|---|---|")
    fam = [("gemini-3.6-flash", "gemini-3.7-flash"), ("gemini-3.7-flash", "gemini-3.8-flash"),
           ("gemini-3.6-flash", "gemini-3.8-flash"), ("gemini-3.1-pro-preview", "gemini-3.8-flash"),
           ("glm-5.3", "glm-5.3-flash"), ("gpt-5.6-luna", "gpt-5.6-sol"), ("gpt-5.6-sol", "gpt-5.6-terra"),
           ("gpt-5.6-luna", "gpt-5.6-terra"), ("gpt-5.5", "gpt-6-astra"), ("claude-opus-4-6", "claude-opus-4-8"),
           ("claude-opus-4-8", "claude-opus-5"), ("claude-opus-5", "claude-sonnet-5"), ("claude-fable-5", "claude-opus-5")]
    for a, b in fam:
        i, j = CLASSES.index(a), CLASSES.index(b)
        sp = piv[[a, b]].dropna().corr(method="spearman").iloc[0, 1]
        md.append(f"| {a} ~ {b} | {out[names[0]][i, j]:.3f} | {out[names[1]][i, j]:.3f} | {out[names[2]][i, j]:.3f} | {sp:.2f} |")


def distinctive_words(tr, top=10):
    h("6. Modele özgü kelimeler (log-odds, bilgilendirici Dirichlet önseli; Monroe vd. 2008)")
    md.append("Her model kendi metinleri ile diğer 17 modelin metinleri arasında karşılaştırıldı; z-skoru en yüksek "
              f"{top} kelime. Her model aynı 149–150 prompt'a cevap verdiği için konu kelimeleri dengelenir.")
    md.append("")
    cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-z][a-z']+\b", min_df=5)
    X = cv.fit_transform(tr.text.str.replace("[’‘]", "'", regex=True))  # curly apostrophes normalised
    vocab = np.array(cv.get_feature_names_out())
    tot = np.asarray(X.sum(0)).ravel().astype(float)
    alpha = tot / tot.sum() * 1000.0
    a0 = alpha.sum()
    md.append("| model | kelimeler (z) |")
    md.append("|---|---|")
    for m in CLASSES:
        ix = (tr.model == m).to_numpy()
        yi = np.asarray(X[ix].sum(0)).ravel().astype(float)
        yj = tot - yi
        ni, nj = yi.sum(), yj.sum()
        d = (np.log((yi + alpha) / (ni + a0 - yi - alpha)) - np.log((yj + alpha) / (nj + a0 - yj - alpha)))
        var = 1 / (yi + alpha) + 1 / (yj + alpha)
        z = d / np.sqrt(var)
        o = np.argsort(-z)[:top]
        md.append(f"| {m} | " + ", ".join(f"{vocab[k]} ({z[k]:.1f})" for k in o) + " |")


def order_checks(tr):
    h("8. id sırası denetimi (reuse/ev/adversarial.order_check)")
    from ev.adversarial import order_check
    md.append("Ardışık satırların aynı sınıfta olma oranı ile şans düzeyi (sum p_c^2); |z| > 3 yapı demek.")
    md.append("")
    md.append("| sıra | gecikme 1 (gözlenen / şans / z) | gecikme 2 | gecikme 10 |")
    md.append("|---|---|---|---|")
    for name, order in [("train.csv dosya sırası", np.arange(len(tr))), ("id'ye göre sıralı", np.argsort(tr.id.values)),
                        ("prompt_id'ye göre sıralı", np.argsort(tr.prompt_id.values, kind="stable"))]:
        r = order_check(tr.model.values[order])
        md.append(f"| {name} | " + " | ".join(f"{d['observed']:.4f} / {d['chance']:.4f} / {d['z']:+.2f}" for d in r) + " |")
    md.append("")
    md.append(f"train.csv satırları id'ye göre sıralı mı: {bool((np.diff(np.argsort(tr.id.values)) == 1).all())}; "
              f"prompt_id'ye göre sıralı mı: {bool(tr.prompt_id.is_monotonic_increasing)}.")


def test_sample():
    h("7. Örnek test dosyası")
    te = load_test()
    md.append(f"- Satır {len(te)}, sütunlar {list(te.columns)}; kelime sayıları {te.text.str.split().str.len().tolist()}.")


def main():
    EDA.mkdir(parents=True, exist_ok=True)
    tr = load_train()
    S = style_frame(tr.text)
    S.index = tr.index
    md.append("# EDA tabloları (src/eda.py çıktısı)")
    pairs(tr)
    lengths(tr, S)
    S2 = formats(tr, S)
    starts_ends(tr, S2)
    similarity(tr, S)
    distinctive_words(tr)
    test_sample()
    order_checks(tr)
    (EDA / "eda_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

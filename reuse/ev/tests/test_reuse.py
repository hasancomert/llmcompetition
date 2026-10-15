"""reuse/ modüllerinin testleri: yalnız rastgele üretilmiş küçük sahte veri, diske yalnız geçici klasörlere yazılır.

Çalıştırma (depo kökünden): python reuse/tests/test_reuse.py   (pytest kuruluysa: python -m pytest reuse/tests)
Sahte veri: 18 sınıf; her "prompt" grubunda sınıfların neredeyse hepsinden birer satır (0-2 sınıf eksik).
"""
import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reuse import adversarial, bias, blend, cv_preds, kaggle_runner, refit_sim, residual, residual_scan  # noqa: E402
from reuse import target_encoding, tokenizer_feats  # noqa: E402

C = 18


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def fake_groups(n_groups=36, n_feat=10, seed=0, sep=1.6):
    """Prompt benzeri gruplar: her grupta sınıfların neredeyse hepsi birer kez; grup etkisi + sınıf merkezi + gürültü."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(C, n_feat)) * sep
    X, y, g = [], [], []
    for grp in range(n_groups):
        drop = rng.choice(C, size=rng.integers(0, 3), replace=False)
        shift = rng.normal(size=n_feat) * 0.5
        for c in np.setdiff1d(np.arange(C), drop):
            X.append(centers[c] + shift + rng.normal(size=n_feat))
            y.append(c)
            g.append(grp)
    return np.array(X), np.array(y), np.array(g)


def fake_xor(n_groups=40, seed=1):
    """Sınıf çiftleri (2k, 2k+1) doğrusal özelliklerde aynı merkezde; ikisini yalnız XOR(u > 0, v > 0) ayırır."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(C // 2, 8)) * 2.0
    X, y, g = [], [], []
    for grp in range(n_groups):
        for c in range(C):
            bit = c % 2
            u = abs(rng.normal()) * rng.choice([-1, 1])
            v = abs(rng.normal()) * (1 if (u > 0) != bool(bit) else -1)  # XOR(u > 0, v > 0) == bit
            X.append(np.r_[centers[c // 2] + rng.normal(size=8) * 0.7, u, v])
            y.append(c)
            g.append(grp)
    return np.array(X), np.array(y), np.array(g)


def lr_fit_predict(X_tr, y_tr, X_va, X_te):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)

    def full(X):  # eğitimde olmayan sınıfın sütunu 0
        out = np.zeros((len(X), C))
        out[:, m.classes_] = m.predict_proba(X)
        return out
    return full(X_va), full(X_te)


# ---------------------------------------------------------------- cv_preds
def test_cv_preds():
    from sklearn.metrics import balanced_accuracy_score
    X, y, g = fake_groups()
    folds = cv_preds.make_folds(y, g, n_splits=5, seed=42)
    assert all(len(np.unique(folds[g == k])) == 1 for k in np.unique(g)), "bir grup iki kata bölündü"
    for k, tr, va in cv_preds.iter_folds(folds):
        assert len(np.unique(y[tr])) == C
    relabeled = (folds + 2) % 5
    assert cv_preds.same_partition(folds, relabeled)
    assert not cv_preds.same_partition(folds, cv_preds.make_folds(y, g, n_splits=5, seed=7))
    rng = np.random.default_rng(3)
    yy, pp = rng.integers(0, C, 500), rng.integers(0, C, 500)
    assert abs(cv_preds.balanced_accuracy(yy, pp) - balanced_accuracy_score(yy, pp)) < 1e-12
    X_te = X[:50] + 0.1
    oof, test, scores = cv_preds.cv_predict(lr_fit_predict, X, y, X_te, folds, C, verbose=False)
    assert oof.shape == (len(y), C) and test.shape == (50, C) and len(scores) == 5
    ba = cv_preds.balanced_accuracy(y, oof)
    assert ba > 3 / C, ba
    with tempfile.TemporaryDirectory() as d:
        cls = [f"model_{i}" for i in range(C)]
        cv_preds.save_preds("a", oof, test, cls, d, cv=ba)
        cv_preds.save_preds("b", oof * 0.5 + 0.5 / C, test, cls, d)
        o, t, c = cv_preds.load_preds("a,b", d, classes=cls)
        assert np.allclose(o, (oof + oof * 0.5 + 0.5 / C) / 2) and c == cls
        with open(os.path.join(d, "meta_a.json"), encoding="utf-8") as f:
            assert json.load(f)["cv"] == ba
        cv_preds.save_preds("c", oof, test, cls[::-1], d)
        try:
            cv_preds.load_preds("a,c", d)
            raise RuntimeError("sınıf sırası farkı yakalanmadı")
        except AssertionError:
            pass
    return f"OOF balanced accuracy {ba:.3f} (şans {1 / C:.3f})"


# ---------------------------------------------------------------- blend
def _fake_models(y, seed=0):
    rng = np.random.default_rng(seed)
    oh = np.eye(C)[y]
    strong = softmax(2.2 * oh + rng.normal(size=oh.shape))
    mid = softmax(1.6 * oh + rng.normal(size=oh.shape))
    noise = softmax(rng.normal(size=oh.shape))
    return [strong, mid, noise]


def test_blend():
    from sklearn.metrics import balanced_accuracy_score
    _, y, g = fake_groups(n_groups=60)
    preds = _fake_models(y)
    singles = [blend.balanced_accuracy(y, p) for p in preds]
    assert abs(blend.balanced_accuracy(y, preds[0]) - balanced_accuracy_score(y, preds[0].argmax(1))) < 1e-12
    w, s = blend.hill_climb(preds, y)
    assert abs(w.sum() - 1) < 1e-9 and w[2] <= w[0] and s >= max(singles) - 1e-12
    assert abs(blend.balanced_accuracy(y, blend.blend(preds, w)) - s) < 1e-12
    w_lp, s_lp = blend.hill_climb(preds, y, space="logprob")
    w_ll, s_ll = blend.hill_climb(preds, y, metric="logloss")
    assert s_ll <= 0 and abs(w_lp.sum() - 1) < 1e-9
    r = blend.nested_hill(preds, y, groups=g, n_splits=5, seed=42)
    assert r["fold_weights"].shape == (5, 3) and abs(r["avg_weights"].sum() - 1) < 1e-9
    assert np.allclose(r["blind_pred"].sum(1), 1) and 0 < r["blind_score"] <= 1
    t_noise = blend.add_candidate_test(preds[0], preds[2], y, groups=g)
    t_mid = blend.add_candidate_test(preds[0], preds[1], y, groups=g)
    assert t_noise["full_gain"] >= 0 and t_mid["full_gain"] >= 0
    st = blend.stack_logreg(preds, y, preds_test=[p[:30] for p in preds], groups=g, C=1.0)
    assert np.allclose(st["blind_pred"].sum(1), 1) and st["test_pred"].shape == (30, C)
    assert blend.label_agreement(preds[0], preds[0]) == 1.0 and blend.label_agreement(preds[0], preds[2]) < 0.2
    return (f"tek modeller {[round(x, 3) for x in singles]}; harman {s:.3f} w={np.round(w, 2).tolist()}; "
            f"iç içe {r['blind_score']:.3f}; gürültü adayı iç içe kazanç {t_noise['nested_gain']:+.3f}, "
            f"orta aday {t_mid['nested_gain']:+.3f}; LR istifleme iç içe {st['blind_score']:.3f}")


# ---------------------------------------------------------------- residual
def test_residual_identity():
    """Öğrenme oranı ~0 iken artık modelin tahmini softmax(başlangıç logit'i) olmalı."""
    rng = np.random.default_rng(0)
    X, y, M = rng.normal(size=(400, 6)), np.tile(np.arange(C), 23)[:400], rng.normal(size=(400, C)) * 2
    out = {}
    (p,), _ = residual.fit_xgb_residual(X, y, M, [X], [M], C, params={"eta": 1e-9}, num_boost_round=3)
    out["xgb"] = np.abs(p - residual.softmax(M)).max()
    (p,), _ = residual.fit_lgbm_residual(X, y, M, [X], [M], C, params={"learning_rate": 1e-9}, num_boost_round=3)
    out["lgbm"] = np.abs(p - residual.softmax(M)).max()
    (p,), _ = residual.fit_cat_residual(X, y, M, [X], [M], C, params={"learning_rate": 1e-6, "iterations": 2})
    out["cat"] = np.abs(p - residual.softmax(M)).max()
    assert out["xgb"] < 1e-5 and out["lgbm"] < 1e-5 and out["cat"] < 1e-3, out
    return "en büyük fark " + ", ".join(f"{k} {v:.1e}" for k, v in out.items())


def test_residual_leak_free():
    """Ezberleyen taban modelle (1-NN) örneklem içi logit'ler eğitim etiketini verir, iç-kat logit'leri vermez."""
    from sklearn.neighbors import KNeighborsClassifier
    X, y, g = fake_groups()

    def knn_logits(X_fit, y_fit, X_apps):
        m = KNeighborsClassifier(n_neighbors=1).fit(X_fit, y_fit)
        res = []
        for Xa in X_apps:
            z = np.full((len(Xa), C), -30.0)
            z[:, m.classes_] = np.log(np.clip(m.predict_proba(Xa), 1e-12, 1))
            res.append(z)
        return res

    folds = cv_preds.make_folds(y, g, 5, 42)
    mg = residual.leak_free_margins(knn_logits, X, y, folds, X[:20], C, groups=g)
    tr = mg[0]["tr"]
    in_sample = knn_logits(X[tr], y[tr], [X[tr]])[0]
    acc_in = np.mean(in_sample.argmax(1) == y[tr])
    acc_inner = np.mean(mg[0]["fit"].argmax(1) == y[tr])
    assert acc_in == 1.0 and acc_inner < 0.95, (acc_in, acc_inner)
    return f"eğitim satırlarında doğruluk: örneklem içi {acc_in:.2f}, iç-kat logit'i {acc_inner:.2f}"


def test_residual_xor():
    """Doğrusal taban XOR'u göremez; ondan başlayan artık GBDT'ler görmeli (gruplu katlar, OOF balanced accuracy)."""
    X, y, g = fake_xor()
    folds = cv_preds.make_folds(y, g, 5, 42)
    X_te = X[:36]
    mg = residual.leak_free_margins(residual.logreg_logits(C), X, y, folds, X_te, C, groups=g)
    base_oof = np.zeros((len(y), C))
    for d in mg:
        base_oof[d["va"]] = residual.softmax(d["valid"])
    ba_base = blend.balanced_accuracy(y, base_oof)
    res = {"taban LR": ba_base}
    oof, _ = residual.residual_cv(residual.fit_xgb_residual, X, y, X_te, mg, C, num_boost_round=150,
                                  params={"eta": 0.1, "max_depth": 3})
    res["xgb"] = blend.balanced_accuracy(y, oof)
    oof, _ = residual.residual_cv(residual.fit_lgbm_residual, X, y, X_te, mg, C, num_boost_round=150,
                                  params={"learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 5})
    res["lgbm"] = blend.balanced_accuracy(y, oof)
    oof, _ = residual.residual_cv(residual.fit_cat_residual, X, y, X_te, mg, C,
                                  params={"iterations": 150, "learning_rate": 0.1, "depth": 3})
    res["cat"] = blend.balanced_accuracy(y, oof)
    oof, _ = residual.residual_cv(residual.fit_xgb_residual, X, y, X_te, mg, C, es_on_valid=True,
                                  num_boost_round=300, early_stopping_rounds=30, params={"eta": 0.1, "max_depth": 3})
    res["xgb (doğrulama katında erken durdurma)"] = blend.balanced_accuracy(y, oof)
    assert all(res[k] > ba_base + 0.1 for k in ("xgb", "lgbm", "cat")), res
    return ", ".join(f"{k} {v:.3f}" for k, v in res.items())


# ---------------------------------------------------------------- target_encoding
def test_target_encoding():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, C, n)
    coarse = target_encoding.encode_keys(rng.choice(["a", "b", "c", None], n))
    fine = target_encoding.encode_keys([f"{c}-{i}" for i, c in enumerate(coarse)])  # her satıra tekil anahtar
    fit, app = np.arange(0, 200), np.arange(200, 210)
    S = 10.0
    F = target_encoding.chain_logp([coarse], y, fit, app, C, S=S)
    prior = np.bincount(y[fit], minlength=C) / 200
    i = app[0]
    cnt = np.bincount(y[fit][coarse[fit] == coarse[i]], minlength=C)
    expect = np.log((cnt + S * prior) / (cnt.sum() + S))
    assert np.allclose(F[0], expect), "elle hesaplanan sonsalla aynı değil"
    tr_feat, app_feat = target_encoding.cross_fit_chain([coarse, fine], y, np.arange(250), np.arange(250, 300), C,
                                                        S=S, inner_splits=5, seed=42)
    # tekil ince anahtar başka satırda yok: ince seviye sonsalı kaba seviyeninkine eşit kalmalı (kendi etiketi yok)
    assert np.allclose(tr_feat[:, :C], tr_feat[:, C:]), "satır kendi etiketini gördü"
    g = rng.integers(0, 30, n)
    tr_g, _ = target_encoding.cross_fit_chain([coarse], y, np.arange(250), np.arange(250, 300), C, groups=g)
    assert tr_g.shape == (250, C) and np.isfinite(tr_g).all()
    return "elle hesaplanan sonsal tuttu; tekil anahtarda kendi etiketi sızmadı; gruplu iç katlar çalıştı"


# ---------------------------------------------------------------- tokenizer_feats
def test_tokenizer_feats():
    vocab = {}
    words = lambda t: [vocab.setdefault(w, len(vocab)) for w in t.split()]
    chars3 = lambda t: [hash(t[i:i + 3]) % 50021 for i in range(0, len(t), 3)]
    texts = ["Certainly! Here is the answer.", "Certainly! Here it is.", "Sure, the answer is 42.", ""]
    df = tokenizer_feats.token_table(texts, {"w": words, "c3": chars3})
    assert df["n_tok_w"].tolist() == [5, 4, 5, 0]
    assert df.loc[0, "key_w_L2"] == df.loc[1, "key_w_L2"] and df.loc[3, "key_w_LAST"] == ""
    df = tokenizer_feats.add_key_counts(df)
    assert np.isclose(df.loc[0, "key_w_L2_cnt"], np.log1p(2))
    msg = "sahte tokenizer'larla özellikler doğru"
    try:
        import transformers  # noqa: F401
        msg += "; transformers kurulu ama yerel tokenizer klasörü olmadan load_hf_tokenizer denenmedi"
    except ImportError:
        msg += "; transformers kurulu değil: load_hf_tokenizer test edilmedi"
    return msg


# ---------------------------------------------------------------- bias
def test_bias():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(C), 100)
    logits = 1.5 * np.eye(C)[y] + rng.normal(size=(len(y), C))
    logits[:, :6] -= 1.2  # modelin 6 sınıfa karşı sistematik eğilimi
    logp = np.log(softmax(logits))
    r = bias.nested_bias(logp, y, n_splits=5, seed=42, grid=np.linspace(-2, 2, 41), n_rounds=2)
    assert r["blind_score"] > r["base_score"] + 0.02, r
    assert abs(r["full_bias"].mean()) < 1e-9 and r["full_bias"][:6].mean() > r["full_bias"][6:].mean()
    yt = np.r_[np.zeros(200, int), np.repeat(np.arange(1, C), 100)]
    pb = bias.prior_bias(yt, C)
    assert pb[0] < pb[1] and abs(pb.mean()) < 1e-9
    return (f"kaydırmasız {r['base_score']:.3f} -> iç içe {r['blind_score']:.3f} "
            f"(tüm OOF'ta {r['full_score']:.3f})")


# ---------------------------------------------------------------- refit_sim
def test_refit_sim():
    from sklearn.linear_model import LogisticRegression
    X, y, g = fake_groups(n_groups=40)
    fp = lambda X_tr, y_tr, X_te: LogisticRegression(max_iter=2000).fit(X_tr, y_tr).predict_proba(X_te)
    r = refit_sim.simulate_refit(fp, X, y, groups=g, holds=(0, 1), n_splits=5)
    assert len(r["rows"]) == 2 and set(r["mean_diff"]) == {"full", "mix_0.25", "mix_0.5", "mix_0.75"}
    return "ortalama fark (kat ortalamasına göre): " + ", ".join(f"{k} {v:+.3f}" for k, v in r["mean_diff"].items())


# ---------------------------------------------------------------- residual_scan
def test_residual_scan():
    rng = np.random.default_rng(0)
    n = 6000
    P = softmax(rng.normal(size=(n, C)) * 1.2)
    key, rkey = rng.integers(0, 40, n), rng.integers(0, 40, n)
    draw = lambda Q: (Q.cumsum(1) > rng.random(len(Q))[:, None]).argmax(1)
    # boş hipotez: etiketler modelin kendi (kalibre) olasılıklarından çekilir
    z_null = residual_scan.scan(draw(P), P, {"rastgele": rkey})[0][0]
    assert z_null < 4.5, z_null
    # yapılı: sekiz grupta sınıf 0 modelin sandığından çok daha sık; model bu gruplarda kalibre değil
    P_true = P.copy()
    P_true[key < 8, 0] += 1.0
    P_true /= P_true.sum(1, keepdims=True)
    y = draw(P_true)
    res = residual_scan.scan(y, P, {"yapılı": key, "rastgele": rkey})
    top = res[0]
    z_rand = max(z for z, c, G, name in res if name == "rastgele")
    assert top[3] == "yapılı" and top[1] == 0 and top[0] > 3 * z_rand, (top, z_rand)
    cm = residual_scan.chance_max_z(225)
    assert 2.4 < cm < 3.2, cm
    g_ba, _ = residual_scan.residual_te_gain(y, P, key)
    g_ba_r, _ = residual_scan.residual_te_gain(y, P, rkey)
    g_ll, _ = residual_scan.residual_te_gain(y, P, key, metric="logloss")
    g_ll_r, _ = residual_scan.residual_te_gain(y, P, rkey, metric="logloss")
    assert g_ba > g_ba_r and g_ll > g_ll_r > -1e-9 and g_ll > 0, (g_ba, g_ba_r, g_ll, g_ll_r)
    return (f"boş hipotezde en yüksek z {z_null:.2f}; yapılı anahtar z {top[0]:.0f} (sınıf {top[1]}), "
            f"rastgele {z_rand:.0f} (genel kalibrasyon bozukluğu her gruplamada görünür); 225 testte şans "
            f"düzeyi {cm:.2f}; artık kodlama kazancı BA {g_ba:+.4f} (rastgele {g_ba_r:+.4f}), "
            f"log-olasılık {g_ll:+.4f} (rastgele {g_ll_r:+.4f})")


# ---------------------------------------------------------------- adversarial
def test_adversarial():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(800, 6)), rng.normal(size=(600, 6))
    auc_same, _ = adversarial.adversarial_auc(A, B, params={"n_estimators": 100})
    B2 = B.copy()
    B2[:, 0] += 1.2  # testte kayma
    auc_shift, _ = adversarial.adversarial_auc(A, B2, params={"n_estimators": 100})
    assert abs(auc_same - 0.5) < 0.06 and auc_shift > 0.65, (auc_same, auc_shift)
    y = rng.integers(0, C, 3000)
    z_rand = max(abs(r["z"]) for r in adversarial.order_check(y))
    z_sorted = adversarial.order_check(np.sort(y))[0]["z"]
    assert z_rand < 4 and z_sorted > 20, (z_rand, z_sorted)
    return (f"aynı dağılım AUC {auc_same:.3f}, kaymalı {auc_shift:.3f}; id sırası rastgele en büyük |z| "
            f"{z_rand:.2f}, sınıfa göre sıralı z {z_sorted:.0f}")


# ---------------------------------------------------------------- kaggle_runner
def test_kaggle_runner():
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "kb")
        meta = kaggle_runner.build_kernel(b, "print('merhaba')\n", "deneme-slug", "kullanici",
                                          competition="ornek-yarisma", datasets=["kullanici/veri"],
                                          kernel_sources=["kullanici/onceki"], gpu=False)
        with open(os.path.join(b, "kernel-metadata.json"), encoding="utf-8") as f:
            m = json.load(f)
        with open(os.path.join(b, "run.py"), encoding="utf-8") as f:
            code = f.read()
    assert m == meta and m["id"] == "kullanici/deneme-slug" and m["enable_internet"] is False
    assert "machine_shape" not in m and m["competition_sources"] == ["ornek-yarisma"]
    assert "def find_input" in code and code.rstrip().endswith("print('merhaba')")
    return "kernel-metadata.json ve run.py doğru; push / wait / logs / fetch test edilmedi (Kaggle hesabı gerekir)"


TESTS = [test_cv_preds, test_blend, test_residual_identity, test_residual_leak_free, test_residual_xor,
         test_target_encoding, test_tokenizer_feats, test_bias, test_refit_sim, test_residual_scan,
         test_adversarial, test_kaggle_runner]

if __name__ == "__main__":
    import importlib
    vers = []
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "catboost"):
        try:
            vers.append(f"{mod} {importlib.import_module(mod).__version__}")
        except ImportError:
            vers.append(f"{mod} yok")
    print("Sürümler:", ", ".join(vers))
    failed = 0
    for t in TESTS:
        t0 = time.time()
        try:
            msg = t()
            print(f"GEÇTİ  {t.__name__} ({time.time() - t0:.1f} sn): {msg}", flush=True)
        except Exception:
            failed += 1
            print(f"KALDI  {t.__name__} ({time.time() - t0:.1f} sn)", flush=True)
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} test geçti")
    sys.exit(1 if failed else 0)

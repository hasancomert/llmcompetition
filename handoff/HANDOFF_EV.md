# HANDOFF: Kaggle Playground S6E9 → Name That LLM

Taranan kaynaklar: README.md, LESSONS.md, HANDOFF_playground-series-s6e9.md, kod (train.py, blend.py, blend_ext.py,
kaggle_run.py, gpt2tok.py, public/, notebooks/), 76 commit'lik git log, Kaggle not defteri günlükleri
(kernel_build*/out, git'te yok) ve oturumun scratchpad betikleri ve günlükleri (repo'da yok). Kodda ya da günlükte
bulunamayan her şey "bulunamadı" diye yazıldı. Yeniden kullanılabilir kod `reuse/` altında (bölüm 9).

S6E9 özeti: sentetik tablo verisinde ikili sınıflandırma (`Will_Buy_EV`), ROC AUC; train 668.665, test 286.571
satır, 13 sütun. Son gönderim: CV 0.94673, iç içe CV 0.94672, public 0.94676, private 0.94570; public sıra 133 / 3.576.

Etiketler, Name That LLM (NLLM) koşullarına göre: 18 sınıf; 2.699 train metni, 150 prompt; ~4.000 test metni,
görülmemiş prompt'lardan; balanced accuracy; internetsiz, en fazla 9 saatlik kod yarışması.
- **[TAŞINIR]** olduğu gibi kullanılabilir.
- **[UYARLANMALI]** fikir geçerli, ama prompt grubu, çok sınıf, metin ya da kod yarışması kısıtı için değişmeli.
- **[TAŞINMAZ]** S6E9'a özgü (sentetik tablo verisi, ikili AUC).

İşe yaramayan denemelerde etiket, "bunu yapma" dersinin taşınıp taşınmadığını gösterir.

## 1. Validation kurulumu ve CV–public–private uyumu

| Madde | S6E9'da ne yapıldı / ölçüm | Etiket | Neden |
|---|---|---|---|
| Kat bölmesi | `StratifiedKFold(K, shuffle=True, random_state=42)`, satır düzeyinde | [UYARLANMALI] | NLLM testi train'de olmayan prompt'lardan geliyor, bu yüzden bir prompt'un metinleri tek katta kalmalı (`cv_preds.make_folds(y, groups=prompt_id)`). |
| Tek bölme, bütün modeller aynı katlarda | megayak OOF kütüphanesinin `fold` sütunu bizim bölmemizle birebir aynı (çapraz tabloyla doğrulandı); GLR not defteri aynı çağrıyı kullanıyor | [TAŞINIR] | OOF'lar ancak aynı bölmeden geldiklerinde sızıntısız harmanlanır (`cv_preds.same_partition`). |
| Kat sayısı | 5 → 10 kat: model başına +0.00005 … +0.00012; 10 → 20: +0.00001 … +0.00004 (LightGBM'de 0) | [UYARLANMALI] | 150 prompt'ta 10 kat, kat başına 15 prompt demek; her kat modeli test çıkarımında ayrıca çalışacaksa kat sayısını 9 saat sınırı belirler. |
| Hedef-türevli özellikler kat içinde | `TargetEncoder(cv=5)` eğitim satırlarını iç katlarla kodlar; `chain` iç 5 kat; GLR etiket özellikleri iç 5 kat | [UYARLANMALI] | İlke aynı ama iç katlar da prompt'a göre gruplu olmalı; sklearn `TargetEncoder`'ın iç katları grup bilmiyor (`target_encoding.cross_fit_chain(groups=...)`). |
| Erken durdurma / epoch seçimi | GBDT'lerde doğrulama katının kendisiyle; `nn()` ve glr_mlp'de en iyi epoch da doğrulama AUC'siyle; OOF'a etkisi ölçülmedi | [UYARLANMALI] | S6E9'da bile ölçülmeyen bu iyimserlik riski 2.699 satırda ayrı bir durdurma kümesiyle önlenmeli. |
| Harman ağırlıkları için iç içe CV | 5 kat; tüm-OOF skoru ile iç içe skor farkı 6–19. gönderimlerde 0 … 0.00002 | [UYARLANMALI] | Prompt gruplu katlarla kurulmalı (`blend.nested_hill(groups=...)`). |
| Düşmanca doğrulama | train/test ayrımı LightGBM ile, AUC 0.50005 (kod repoda yok, oturum günlüğünde) | [TAŞINIR] | Görülmemiş prompt'lardan gelen testte kayma olup olmadığını ucuzca gösterir (`adversarial.adversarial_auc`). |
| id sırası denetimi | ardışık satırlarda etiket korelasyonu +0.001 (şans sınırı ±0.0012); id–hedef AUC 0.49999 | [TAŞINIR] | Ucuz bir sızıntı denetimi; çok sınıflı sürümü `adversarial.order_check`. |
| Private'ın sabit kayması | 2–19. gönderimlerde public'in 0.00092 … 0.00106, CV'nin 0.00080 … 0.00104 altında | [TAŞINMAZ] | Sayılar 286 bin satırlık AUC testine ait; NLLM'de test ~4.000 metin ve public/private model karışımı farklı. |
| Sıra uyumu | CV–private Pearson 0.9975, Spearman 0.9854 (19 gönderim); en iyi private (0.94570) en iyi public'li 18 ve 19'da | [UYARLANMALI] | Gönderim başına CV, iç içe CV, public ve private kaydı tutmak taşınır; uyumun kendisi NLLM'de varsayılamaz. |
| Görünür adım büyüklüğü | +0.00019'luk CV adımı private'a +0.00020 taşındı; ≤ +0.00005'lik adımlar tutarsız (4→5, 7→9, 12→15 görünmedi; 16→19'da +0.00002 → +0.00003) | [UYARLANMALI] | Bu eşik AUC'ye ve bu test boyutuna ait; balanced accuracy ve ~4.000 metin için yeniden ölçülmeli. |

## 2. Deney tablosu

Ölçüm düzeyleri farklı: tek model 5 kat lr 0.1, tek model 10 kat lr 0.02 ya da harman iç içe CV. Katkılar toplanamaz.

### İşe yarayanlar (büyükten küçüğe)

| Değişiklik | Ölçüm | Etiket | Neden |
|---|---|---|---|
| `te1`: 13 sütunun hedef kodlaması | LightGBM 5 kat 0.94381 → 0.94543 (+0.00162) | [UYARLANMALI] | Metinde kodlanacak anahtarlar parça önekleri gibi şeyler olur; kodlama çok sınıflı ve prompt gruplu iç katlarla yapılmalı (`target_encoding`). |
| `freq`: değerin train + test + orijinaldeki sayımı (etiketsiz) | 0.94178 → 0.94301 (+0.00123) | [UYARLANMALI] | Sütun değeri yerine metin anahtarlarının sayımı olarak kurulur (`tokenizer_feats.add_key_counts`). |
| `dig`: gelir mod 10 / 100 / 1000, mesafenin ondalığı | 0.94301 → 0.94381 (+0.00080) | [TAŞINMAZ] | Sentetik üretecin sayısal hane izleri; metinde karşılığı bulunamadı. |
| `tok`: gelirin GPT-2 BPE parçaları | 5 kat +0.00035; 10 katta ≈ +0.00021 … +0.00026 / model | [UYARLANMALI] | Ayrıntı bölüm 3'te. |
| `bins`: gelir // 100, // 1000, tam km | +0.00027 / +0.00031 (LightGBM / XGBoost) | [TAŞINMAZ] | Sayısal sütun kabalaştırması; metinde karşılığı yok. |
| Tohum ortalaması | RealMLP 6 tohum +0.00015 … +0.00025; XGBoost 3 tohum +0.00006 … +0.00008 (5 kat) | [TAŞINIR] | Ucuz varyans azaltma: tohumlar harmandan önce ortalanır (`cv_preds.load_preds("a,b,c")`). |
| Optuna ile XGBoost ayarı (50 deneme) | 0.94591 → 0.94606 (+0.00015); en iyi denemelerin çoğu derinlik 4, tepe düz | [UYARLANMALI] | Yalnız hızlı eğitilen modellerde (ör. stil özellikleri üstündeki GBDT) aynı şekilde uygulanabilir. |
| Açık not defteri ayarları (LightGBM `pubp`) | 0.94593 → 0.94608 (+0.00015) | [TAŞINIR] | Güçlü bir açık not defterinin ayarlarından başlamak kendi ayarlarımızdan iyi çıktı. |
| Harmana açık OOF kütüphaneleri | iç içe 0.94626 → 0.94640 (+0.00014) | [UYARLANMALI] | Yalnız aynı prompt gruplu bölmeyle üretilmiş OOF'lar harmanlanabilir; NLLM'de paylaşılan OOF bulunamadı. |
| 5 → 10 kat | +0.00005 … +0.00012 / model | [UYARLANMALI] | Bkz. bölüm 1. |
| `dig2` (tek tek haneler) + `te3` (üç yumuşatma) | +0.00003 / +0.00009; `te3` tek başına −0.00002 / +0.00004 | [TAŞINMAZ] | Hane özellikleri sentetik veriye özgü, `te3`'ün tek başına katkısı yok. |
| GLR logit'inden artık GBDT (`margin=1`) | +0.00006 … +0.00007 / model; harmana iç içe +0.00001 | [UYARLANMALI] | Ayrıntı bölüm 4'te. |
| `chain`: parça hiyerarşisinde hedef oranı | `tok`'suz 5 kat 0.94573 → 0.94605; `tok` ile birlikte 0.94605 (ek kazanç yok); 10 katta `tok` üstüne +0.00002 … +0.00004 | [UYARLANMALI] | Kaba → ince anahtar zinciri metinde parça önekleriyle kurulabilir (`target_encoding.chain_logp`). |
| 10 → 20 kat | +0.00001 … +0.00004 (LightGBM 0) | [UYARLANMALI] | Bkz. bölüm 1. |

### İşe yaramayanlar

| Deneme | Ölçüm | Etiket | Neden |
|---|---|---|---|
| Orijinal veriyi ek eğitim satırı yapmak | 0.94178 → 0.94177; 0.94543 → 0.94534 | [TAŞINMAZ] | Orijinal veri sentetik verinin kaynağıydı; NLLM için böyle bir kaynak veri seti bulunamadı. |
| Küçük tam sayıları kategori yapmak | −0.00074 | [TAŞINMAZ] | Tablo sütunlarına özgü. |
| Kesişim anahtarlarının hedef kodlaması: `te2` −0.00021, `te2s` −0.00008, `tokx` (parça × bağlam) −0.00017 | LightGBM 5 kat | [TAŞINIR] | Kesişim hücreleri iki ayrı denemede zarar verdi; 2.699 satırda bu hücreler daha da seyrek kalır. |
| `mix`: aynı anahtarı paylaşan train + test satırlarının diğer sütun ortalamaları | 5 kat −0.00011; 10 katta −0.00004 | [UYARLANMALI] | Test metinlerini prompt'a göre kümeleyip grup özelliği çıkarma fikrine en yakın deneyimiz bu ve zarar verdi; grup özellikleri tek tek ölçülerek eklenmeli. |
| `recipe`, `omean`, `max_bin=1024`, `bins2`, `tedig` / `freq2` / `flag` | −0.00005 … +0.00004 | [TAŞINMAZ] | S6E9 sütunlarına özgü. |
| Tam veri modeli (kat ortalaması yerine) | 4 sahte testte ortalama −0.00004 AUC; %25 karışım +0.00001 | [UYARLANMALI] | Karar NLLM modelleriyle yeniden benzetilmeli (`refit_sim.simulate_refit`). |
| Sözde etiketleme (kat başına sızıntısız) | XGBoost 10 kat 0.94625 → 0.94626 | [UYARLANMALI] | S6E9'da test train'in %43'üydü, NLLM'de test (~4.000) train'den (2.699) büyük, yani sonuç taşınamaz ve yeniden ölçülmeli. |
| Örneklemde bulunan %0 / %100 aralık kuralları | görülmemiş katta −0.00037 (≥ 100 satırlık kurallar) | [TAŞINIR] | OOF'ta bulunan bir kural, görmediği katta sınanmadan kullanılmamalı. |
| Her ağacı tek sütunla sınırlamak (`solo=1`) | ham sütunlarda +0.00045, zengin sette −0.00022 | [TAŞINMAZ] | S6E9 sütun yapısına özgü bir deney. |
| GLR logit'ini RealMLP'ye özellik olarak vermek | 0.94636 → 0.94596 | [TAŞINIR] | Taban logit'i artık modele başlangıç değeri olarak verilmeli; sinir ağına özellik olarak verilince belirgin zarar verdi. |
| Kendi gömmeli MLP'miz (`train.py nn`) | 0.94525; harmanda ağırlık 0 | [TAŞINMAZ] | Tablo için yazılmış bir MLP, metin modeli değil. |
| İkinci seviye LightGBM istifleme | 0.94575 (o sıradaki harman 0.94644) | [TAŞINIR] | Ağaç istifleyici tahminleri kutuladı ve sırayı kaybetti; ağırlıklı ortalama daha iyiydi. |
| Harmandan başlayıp ham sütunlarla düzeltme modeli | 1–8 ağaçta durdu; 0.94644 değişmedi | [UYARLANMALI] | Harmanın ötesinde öğrenilecek yapı olup olmadığının ucuz testi; çok sınıfta `residual.fit_*_residual` ile kurulur. |
| LR istifleme | tepe tırmanmayla aynı iç içe skor (0.94639–0.94640; sonra 0.94668–0.94669) | [UYARLANMALI] | `blend.stack_logreg` log-olasılıklarla çalışır; S6E9'da kazanç vermedi. |
| `tok` sonrası XGBoost'u yeniden ayarlamak (40 deneme, `t2`) | 10 kat 0.94650 / 0.94646; `t1` ile aynı düzey | [TAŞINIR] | Özellik değişince ayarı baştan aramak burada bir şey getirmedi. |
| GLR L2 = 3 / 10 / 30; LightGBM 20 kat | değişiklik yok | [TAŞINMAZ] | Modele özgü sonuçlar. |
| 25 dış OOF'u harmana tek tek eklemek | 3'ü katkı verdi (+0.000006 … +0.000014), 22'si 0 | [UYARLANMALI] | Yöntem taşınır (`blend.add_candidate_test`, iç içe ağırlık seçimi); hangi adayın uygun olduğunu prompt gruplu bölme belirler. |
| Kaçırılmış desen arayışı | ~300 gruplamada en yüksek z 2.47 (225 testte şans düzeyi ~2.8); id sırası, birebir tekrar, en yakın orijinal satır: sonuç yok | [UYARLANMALI] | Çok sınıflı tarama `residual_scan.scan` ile yapılır; olasılıklar önce kalibre edilmeli (S6E9'da izotonik). |

## 3. Tokenizer tabanlı özellikler

**Nasıl hesaplandı** (`gpt2tok.py`, `train.py:136–146`, `kaggle_run.py`):

1. Gelirin train + test + orijinaldeki 16.426 farklı değeri `" " + str(int(gelir))` olarak GPT-2 tokenizer'ıyla
   (`AutoTokenizer.from_pretrained("gpt2")`) parçalandı: 11.873 değer 3 parçaya, 4.553 değer 2 parçaya bölündü.
2. Tutulanlar: parça sayısı, ilk üç parça kimliği, son parça kimliği ve parça metinleri (`gpt2_income_tokens.csv`).
3. Betik internetli bir Kaggle CPU not defterinde (`pvep-gpt2tok`) çalıştı. CSV, internetsiz eğitim not defterlerine
   `kernel_sources` ile girdi olarak bağlandı.
4. Özellikler:
   - L1 = ilk parça;
   - L2 = ilk iki parça (t1 · 60000 + t2);
   - LAST = parça sayısı · 60000 + son parça.

   Her biri hedef kodlama anahtarı oldu (te1 / te3 ile). Ayrıca her anahtarın train + test + orijinaldeki log1p
   sayımı ve parça sayısı (`tok_n`) eklendi.
5. Mesafe değerleri için de eşleme üretildi (`gpt2_commute_tokens.csv`) ama hiçbir modelde kullanılmadı.

**Ne kazandırdı:**

| Ölçüm | Sonuç |
|---|---|
| LightGBM 5 kat lr 0.1, taban `base,freq,dig,dig2,te1,te3,bins` | 0.94573 → 0.94608 |
| Aynı tabana daha çok ondalık kutu (`bins2`) | +0.00000 |
| XGBoost `t1` 10 kat, 3 tohum ortalaması | 0.94631 → 0.94652 |
| CatBoost d5 10 kat | 0.94622 → 0.94645 |
| RealMLP 10 kat, tek tohum | 0.9461 → 0.94636 |
| `tokx` (parça × bağlam hücreleri), `tok` üstüne | 0.94608 → 0.94591 |
| Leaderboard | `tok`'lu modellerin ilk girdiği 11. gönderim: public 0.94667, private 0.94561. Önceki en iyi (9.): 0.94645 / 0.94544. Harmanın diğer üyeleri de değişti. |
| Tek başına hedef kodlama AUC'si (5 kat, yumuşatma 20; bu oturumda ölçüldü) | ilk parça (110 anahtar) 0.68357, gelir // 1000 (150 anahtar) 0.68388; ilk iki parça (3.790) 0.70591, gelir // 100 (1.142) 0.70344; tam gelir (13.214) 0.71059 |

| Madde | Etiket | Neden |
|---|---|---|
| Üretenin tokenizer izini özellik yapmak | [UYARLANMALI] | Burada sentetik üretecin sayıları nasıl yazdığını yakaladı; NLLM'de 18 modelin tokenizer'larıyla aynı fikir denenebilir ama metinde hiç ölçülmedi. |
| Tokenizer başına parça sayısı (`tok_n`) | [UYARLANMALI] | Her lab tokenizer'ı için ayrı sayım (`tokenizer_feats.token_table`); metinde parça sayısı uzunlukla birlikte değiştiği için ham sayım uzunluğu da taşır. |
| İlk / ilk iki / son parça anahtarları + hedef kodlama | [UYARLANMALI] | Metinde açılış ve kapanış parçaları olarak kurulur, kodlama çok sınıflı ve prompt gruplu iç katlarla (`target_encoding`). |
| 60000 çarpanlı sayısal anahtar | [TAŞINMAZ] | Kimliklerin 60000'in altında olduğunu varsayar (GPT-2'de doğru); büyük sözlüklü tokenizer'larda çakışır, bu yüzden `reuse`'da string anahtar kullanıldı. |
| LAST'ı parça sayısıyla birleştirmek | [TAŞINMAZ] | Sayılarda 2–3 parça vardı; 40–250 kelimelik metinde parça sayısı satırı neredeyse tekil yapar. |
| Etiketsiz anahtar sayımı train + test üstünden | [UYARLANMALI] | Test metinleri yalnız gönderim çalıştırmasında görülüyorsa sayım orada yeniden hesaplanmalı. |
| İnternetli not defterinde ön-hesap, internetsiz not defterine girdi | [UYARLANMALI] | Final not defteri internetsiz olduğu için tokenizer dosyaları veri seti olarak eklenmeli (`load_hf_tokenizer` yerel klasörden yükler; bu ortamda test edilmedi). |

## 4. Residual / base_margin yaklaşımı

**Kurulum:**

| Parça | Nasıl | Kod |
|---|---|---|
| Taban model | heuljax'ın üreteç-farkında L2 lojistik regresyonu (GLR), L2 = 10, LBFGS | `notebooks/glr` |
| Sızıntısız logit'ler | Doğrulama ve test satırlarının logit'i dış kat modelinden. Eğitim satırlarınınki, o katın eğitim satırlarında iç 5 katla (`StratifiedKFold(5, shuffle, 43)`) yeniden eğitilen GLR'den. Çıktı `glr_margins.npz`: fit (K, N), oof (N), test (K, M). 20 katlı sürümü ayrı. | `notebooks/glr_margin`, `glr_margin_f20` |
| XGBoost `t1` | `fit(base_margin=eğitim logit'i, base_margin_eval_set=[doğrulama logit'i])`; tahmin `predict_proba(X, base_margin=...)` | `train.py:271` |
| LightGBM `pubp` | `fit(init_score=..., eval_init_score=[...])`; tahmin sigmoid(raw_score + logit) | `train.py:252` |
| CatBoost d5 | `Pool(baseline=...)`; tahmin sigmoid(RawFormulaVal + logit) | `train.py:292` |
| Güvenlik | `assert MG["fit"].shape[0] == FOLDS`; her eğitim satırının iç-kat logit'i sonlu olmalı (aynı bölme şart) | `train.py:430–437` |
| İkinci tür: MLP artığı | Son katmanı sıfırla başlatılmış küçük bir MLP, GLR'nin aynı katta eğitilmiş (örneklem içi) logit'ine eklenir; epoch doğrulama AUC'siyle seçilir | `notebooks/glr_mlp` |

**Ne kazandırdı:**

| Model | Düz | Artık |
|---|---|---|
| XGBoost 10 kat, iki tohum | 0.94652 / 0.94648 | 0.94658 / 0.94654 |
| LightGBM 10 kat | 0.94650 | 0.94656 |
| CatBoost d5 10 kat | 0.94648 | 0.94655 |
| XGBoost 20 kat (20 katlı GLR logit'inden) | 0.94651 / 0.94653 | 0.94659 / 0.94657; 2 tohum ortalaması 0.94661 |
| LightGBM 20 kat | 0.94650 | 0.94659 |
| GLR + MLP artığı (10 kat / 3 bölme tohumu ortalaması / 20 kat) | GLR 0.94640 / 0.94643 / 0.94644 | 0.94642 / 0.94646 / 0.94647 |
| Harmana eklemek (16 → 17) | iç içe 0.94670; public 0.94673; private 0.94567 | iç içe 0.94671; public 0.94675; private 0.94569 |
| Son harman (19) | | En büyük ağırlık 20 katlı artık XGBoost'ta (0.281) |

| Madde | Etiket | Neden |
|---|---|---|
| İç-kat (sızıntısız) başlangıç logit'leri | [TAŞINIR] | Eğitim satırının logit'i kendi etiketini görmüş bir modelden gelmemeli; sahte veride ezberleyen tabanla örneklem içi doğruluk 1.00, iç-kat 0.82 çıktı (`residual.leak_free_margins`). |
| Taban model olarak GLR | [TAŞINMAZ] | GLR'nin özellikleri sentetik üretece göre tasarlanmıştı; NLLM'de tabanı LLM sınıflandırıcısının logit'leri oluşturur. |
| LLM logit'leri üstüne stil özellikli artık GBDT | [UYARLANMALI] | Çok sınıflı başlangıç logit'leri (n, 18) bu ortamda üç kütüphanede doğrulandı (`residual.fit_*_residual`); kat başına taban model 1 + 5 kez eğitildiği için LLM'de bu maliyet hesaba katılmalı. |
| CatBoost'la çok sınıflı artık model | [UYARLANMALI] | catboost 1.2.10, eğitim katında bir sınıf eksikse hata veriyor (bu oturumda denendi); prompt gruplu katlarda her sınıfın her eğitim katında olduğu denetlenmeli. |
| Logit'i sinir ağına özellik olarak vermek | [TAŞINMAZ] | Burada −0.00040; logit başlangıç değeri olarak kullanılmalı. |
| Sıfır başlatılmış MLP artık başı | [UYARLANMALI] | GLR üstünde +0.00002 … +0.00003 verdi; LLM logit'leri üstünde ölçülmedi. |
| Artık modelde erken durdurma | [UYARLANMALI] | S6E9'da doğrulama katıyla yapıldı ve etkisi ölçülmedi; `residual.residual_cv` varsayılan olarak sabit tur sayısı kullanır. |

## 5. Blend yöntemi

**Nasıl yapıldı** (`blend_ext.py`; ilk sürüm `blend.py`):

1. Her OOF ve test tahmini sıraya çevrildi (`rankdata / n`); tohumlar önce olasılık uzayında ortalandı.
2. Tepe tırmanma (`hill`): en iyi tek modelden başlar ve her adımda AUC'yi en çok artıran modeli bir kez daha ekler.
   Aynı model tekrar seçilebilir. Kazanç 1e-7'nin altına inince durur (en çok 100 adım); ağırlık = seçilme
   sayısı / toplam. `blend.py`'de eşik 1e-6'ydı ve adım sınırı yoktu.
3. İç içe 5 kat (`StratifiedKFold(5, shuffle, 42)`): ağırlıklar OOF'un 4/5'inde seçilir, kalan 1/5'te ölçülür.
   18. ve 19. gönderimde son ağırlıklar bu 5 vektörün ortalaması (`AVGW=1`).
4. Aday ekleme testi (scratchpad `addtest.py`): (1 − w) · harman + w · aday, w ∈ {0, 0.05, …, 0.5}; w iç içe seçilir.
5. LR istifleme (`STACK=1`): probit(sıra) özellikleri, C = 1. scratchpad `stack.py`'de C 0.001 / 0.01 / 1 denendi.
6. Sınır kuralları harmandan sonra uygulandı (bölüm 6).

**Ölçümler:**
- Harman, en iyi tek modelin 0.00012 üstünde (0.94661 → 0.94673).
- Açık OOF kütüphaneleri iç içe +0.00014 getirdi; LR istifleme tepe tırmanmayla aynı skoru verdi.
- 19. gönderimde 28 adaydan 12'si ağırlık aldı; en büyük ağırlık 0.281.

| Madde | Etiket | Neden |
|---|---|---|
| Sıra uzayı | [TAŞINMAZ] | AUC yalnız sıraya bakar; balanced accuracy her satırda sınıflar arasında argmax alır, sütunları ayrı ayrı sıralamak sınıfları karşılaştırılamaz yapar. |
| Tekrar seçilebilir açgözlü tepe tırmanma | [UYARLANMALI] | Olasılık ya da log-olasılık ortalamasıyla ve balanced accuracy (ya da log loss) ile çalışacak şekilde uyarlandı (`blend.hill_climb`). |
| İç içe CV ile ağırlık ölçümü | [UYARLANMALI] | Prompt gruplu katlarla kurulmalı (`blend.nested_hill(groups=...)`). |
| Ağırlıkları iç içe katlarda ortalamak (`AVGW`) | [TAŞINIR] | Tek bir tüm-veri seçimi yerine 5 seçimin ortalaması; S6E9'da son iki gönderimde kullanıldı. |
| Aday ekleme testi | [TAŞINIR] | `blend.add_candidate_test`. |
| LR istifleme | [UYARLANMALI] | `blend.stack_logreg` log-olasılık özellikleriyle; S6E9'da tepe tırmanmayı geçmedi. |
| Tohumları harmandan önce ortalamak | [TAŞINIR] | `cv_preds.load_preds("a,b,c")`. |
| Balanced accuracy ile tepe tırmanma | [UYARLANMALI] | BA basamaklıdır (bir metnin sınıfı değişince skor bir adım değişir); log loss seçeneği var, NLLM verisinde hiçbiri ölçülmedi. |

## 6. Eşik ya da sınıf ağırlığı optimizasyonu

S6E9'da eşik ya da sınıf ağırlığı optimizasyonu **bulunamadı**:
- Metrik AUC'ydi, yani eşiksiz.
- Kodda eşik, `scale_pos_weight` ya da sınıf başına kaydırma araması yok.
- `public/pub_realmlp.py` "balanced" sınıf ağırlığını hesaplıyor, ama kayba `pos_weight=None` veriyor; yani kullanılmadı.

En yakın iki uygulama:
- **Sınır kuralları** (`RULES=1`, `DEADZONE=1`): sıra grup içinde korunarak satırlar en üste ya da en alta itildi.
  Kurallar trende istisnasızdı (ör. gelir ≥ 170.537: 393 satırın 393'ü alım). OOF kazancı +0.000001 / +0.000004.
- **Örneklemde bulunan aralık kuralları:** görülmemiş katta −0.00037.

| Madde | Etiket | Neden |
|---|---|---|
| Eşik / sınıf ağırlığı optimizasyonu | [TAŞINMAZ] | S6E9'da yapılmadığı için taşınacak deneyim yok; `reuse/bias.py` NLLM için sıfırdan yazıldı ve yalnız sahte veriyle denendi (kaydırmasız 0.352, iç içe 0.376, tüm OOF'ta seçilince iyimser biçimde 0.416). |
| Sınır kuralları (`RULES`, `DEADZONE`) | [TAŞINMAZ] | S6E9'un gelir ve mesafe eşiklerine özgü. |
| Kuralı ya da kaydırmayı görmediği katta sınamak | [TAŞINIR] | S6E9'da örneklem içinde bulunan kurallar −0.00037 kaybettirdi; `bias.nested_bias` aynı ilkeyle çalışır. |

## 7. Final 2 submission seçimi

Önerilen seçim 19 ve 11'di:
- **19:** public'te 18 ile aynı skoru aldı (0.94676), iç içe CV'si biraz daha yüksekti (0.94672 / 0.94671).
- **11:** iyi gönderimler içinde en farklı olan, test tahminlerinin sıra korelasyonu 0.9991. 17, 18 ve 19 birbirinin
  neredeyse kopyası olduğu için ikisini birlikte seçmek yedek işlevi görmezdi.

Kaggle'da işaretlenen ikili **bulunamadı**: API seçili gönderimleri vermiyor. Sonuç: 19 private 0.94570, 11 private
0.94561; en yüksek private 0.94570 (18 ve 19).

| Madde | Etiket | Neden |
|---|---|---|
| "En iyi iç içe CV + en farklı iyi gönderim" kuralı | [UYARLANMALI] | Farklılık ölçüsü sıra korelasyonuydu; çok sınıfta tahmin edilen sınıfların uyuşma oranı kullanılır (`blend.label_agreement`). |
| Yedeğe güvenmek | [UYARLANMALI] | S6E9'da yedek gerekmedi çünkü en iyi private yine en iyi CV'li gönderimdi; NLLM'de public/private model karışımı farklı olduğu için bu varsayılamaz. |
| Seçimi kayda geçirmek | [TAŞINIR] | API seçili gönderimleri vermediği için seçim yapıldığı anda elle not edilmeli. |

## 8. Zaman kaybettiren çıkmaz sokaklar ve sık hatalar

| Ne oldu | Kayıt | Etiket | Neden |
|---|---|---|---|
| Küçük CV adımlarına gönderim harcamak | 12–15: dört gönderim public'te 0.94671'de kaldı (CV +0.00002); 7–10 arası 0.94644–0.94645 | [TAŞINIR] | Günlük gönderim hakkı, CV'de görünür bir adım olmadan harcanmamalı. |
| Tam veri yeniden eğitimi | XGBoost 4 ve RealMLP 2 sahte testle benzetim, ayrıca tam veri RealMLP, XGBoost ve CatBoost modelleri; sonuç −0.00004 ve kullanılmadı; süre kaydı bulunamadı | [UYARLANMALI] | Büyük eğitimlerden önce küçük bir benzetim (`refit_sim`) kararı verir. |
| Desen arayışı | ~300 gruplama, id sırası, tekrar eden satır, en yakın orijinal satır; sonuç yok; süre kaydı bulunamadı | [UYARLANMALI] | Kalibre OOF artıklarıyla tek betikte tarama (`residual_scan`) yeterli. |
| 25 dış OOF'u tek tek denemek | 22'si 0 katkı | [UYARLANMALI] | Aday önce ucuz ekleme testinden (`blend.add_candidate_test`) geçmeli. |
| CatBoost'u GPU'da eğitmek | aynı katlarda CPU'dan ~0.00026 düşük; GPU varsayılanı `border_count=128` ile ~0.0008 düşük. CPU'da 10 kat: yerelde 79 dk, Kaggle CPU'da 5.441–6.951 sn | [UYARLANMALI] | GPU'ya geçmeden önce aynı katlarda CPU ile küçük bir karşılaştırma yapılmalı; varsayılanlar farklı olabiliyor. |
| Leaderboard sırasını tahmin etmek | "~20–25" dedim; indirilen leaderboard'da 133 / 3.576 çıktı | [TAŞINIR] | Sıra tahmin edilmemeli, `kaggle competitions leaderboard -d` ile indirilip okunmalı. |
| Son seçimi doğrulayamamak | API seçili gönderimleri vermiyor | [TAŞINIR] | Seçim yapıldığı anda not edilmeli. |
| Logit dosyası ile modelin bölmesi uyuşmazsa | `train.py`'de assert'ler: kat sayısı ve sonlu logit | [TAŞINIR] | Artık model ile logit dosyası aynı bölmeyi kullanmalı; bu assert'ler sessiz hatayı önler. |
| RealMLP kategorik sütunsuz veride çöktü | sabit `dummy_` kategorik sütun eklendi (`notebooks/glr_realmlp`) | [UYARLANMALI] | Gömme gibi tamamen sayısal girdiyle RealMLP kullanılırsa aynı sorun çıkar. |
| Kaggle altyapısı | Aynı anda en çok 2 GPU oturumu açılabildi (push "Maximum batch GPU" hatasında yeniden denedi). Çıktı indirmek www.kaggleusercontent.com erişimi istedi. Kapsayıcı yeniden başlayınca yerel işler öldü; ağır işler Kaggle not defterlerine taşındı. | [UYARLANMALI] | `reuse/kaggle_runner.py` aynı akışı parametreli verir; NLLM'nin final not defteri ayrıca internetsiz ve en fazla 9 saat. |
| XGBoost'ta logit'i elle eklemek | `output_margin` çıktısı base_score'u da içeriyor; elle eklenince olasılık 0.09'a kadar saptı (bu oturumda ölçüldü). S6E9 kodu doğru yolu kullanıyordu. | [TAŞINIR] | Tahmin `DMatrix(base_margin=...)` ile alınmalı. |
| Doğrulama katında erken durdurma ve epoch seçimi | etkisi ölçülmedi | [UYARLANMALI] | 2.699 satırda ayrı bir durdurma kümesi kullanılmalı. |
| OOF'suz açık harmanlar | Public tablonun tepesi (0.94657'de 98 takım) OOF'suz açık harmanların yeniden gönderimiydi; biz yalnız OOF'u olan bileşenleri harmanladık. | [UYARLANMALI] | Kod yarışmasında da yalnız kendi bölmemizde OOF'u olan bileşenler ölçülebilir. |

## 9. reuse/ modülleri

Hepsi çok sınıflı (N × 18 olasılık) ve balanced accuracy'ye uyarlandı. Modüller birbirini içe aktarmaz. Testler yalnız
rastgele sahte veri kullanır ve depoya bir şey yazmaz: `python reuse/tests/test_reuse.py` → 12/12 geçti
(numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.2.0, lightgbm 4.7.0, catboost 1.2.10). Tablodaki sayılar
sahte veriye ait; NLLM hakkında bir şey söylemez.

| Modül | S6E9 kaynağı | Uyarlama | Sahte veri testi |
|---|---|---|---|
| `cv_preds.py` | `train.py` kat döngüsü ve kaydetme; `blend.py` tohum ortalaması; megayak bölme denetimi | prompt gruplu katlar, (N, C) diziler, sınıf sırası `meta_<etiket>.json`'da | OOF BA 0.893 (şans 0.056); kaydet/yükle; sınıf sırası farkı yakalandı |
| `blend.py` | `blend_ext.py` `hill`, iç içe CV, `AVGW`, `STACK`; `addtest.py` | olasılık / log-olasılık, BA / log loss, gruplu katlar, `label_agreement` | tekler 0.653 / 0.445 / 0.055 → harman 0.763, iç içe 0.763; LR istifleme 0.677 |
| `residual.py` | `glr_margin` not defteri; `train.py margin=1` | (n, C) logit, XGBoost ve LightGBM yerel API, sklearn LR taban örneği | lr ≈ 0'da softmax(logit) ile fark ≤ 4e-7; 1-NN tabanda örneklem içi 1.00, iç-kat 0.82; XOR verisinde taban 0.513 → XGBoost 0.682, LightGBM 0.710, CatBoost 0.853 |
| `target_encoding.py` | `train.py` `chain_feats`, `fold_data` | Dirichlet yumuşatma, log-olasılık, gruplu iç katlar | elle hesaplanan sonsalla aynı; tekil anahtarda kendi etiketi sızmadı |
| `tokenizer_feats.py` | `gpt2tok.py`, `train.py` `tok` | metin, birden çok tokenizer, string anahtarlar | sahte tokenizer'larla doğru; `load_hf_tokenizer` test edilmedi (transformers yok) |
| `bias.py` | YENİ (S6E9'da karşılığı yok) | sınıf başına kaydırma, `prior_bias`, iç içe doğrulama | kaydırmasız 0.352 → iç içe 0.376 (tüm OOF'ta 0.416) |
| `refit_sim.py` | `train.py hold=` / `full=`; `sim_eval.py` | olasılık karışımı; ağaç sayısı ölçekleme genelleştirilmedi | çalıştı |
| `residual_scan.py` | `resid_base.py`, `resid_scan1-3.py` | sınıf başına bire-karşı-hepsi z; log loss kazanç seçeneği | boş hipotezde en yüksek z 2.31; yapılı anahtar 647, rastgele 131 (genel kalibrasyon bozukluğu); 225 testte şans 2.79 |
| `adversarial.py` | oturum günlüğündeki komut | özellik matrisiyle; çok sınıflı `order_check` | aynı dağılım AUC 0.515, kaymalı 0.767; sınıfa göre sıralı id z 224 |
| `kaggle_runner.py` | `kaggle_run.py` | yarışma, veri seti, not defteri kaynakları parametre | yalnız `build_kernel` test edildi |

**Uyarlanamayanlar / taşınmayanlar:**
- **`train.py`'nin kendisi:** S6E9 sütun adları ve özellik grupları gömülü. Kat döngüsü, kaydetme ve artık model
  kısımları `reuse`'a alındı.
- **`train.py nn()` ve `public/pub_realmlp.py`:** ikisi de ikili (BCE kaybı). Çok sınıf için çıkış katmanı ve kayıp
  değişmeli; uyarlanmadı.
- **`notebooks/glr*`:** özellik matrisi sentetik üretece göre tasarlanmıştı; çok terimli sürümü yazılmadı.
  `reuse`'da taban örneği sklearn `LogisticRegression`.
- **`train.py pl=` (sözde etiketleme):** ikili yumuşak etiketle çalışıyor (test satırları p ve 1 − p ağırlıklı iki
  kopya). Çok sınıf için C kopya gerekir; uyarlanmadı.
- **Optuna araması (`train.py trials=`):** ikili XGBoost için yazıldı; uyarlanmadı.
- **`blend_ext.py`'deki `RULES`, `DEADZONE`, `FULLMIX`:** S6E9 eşiklerine ve sıra uzayına özgü.
- **scratchpad `nn_orig.py` (en yakın orijinal satır):** NLLM'de orijinal veri seti yok.
- **`kaggle_runner` push / wait / logs / fetch:** Kaggle hesabı olmadan test edilmedi.
- **`tokenizer_feats.load_hf_tokenizer`:** transformers bu ortamda kurulu olmadığı için test edilmedi.
- **Test metinlerini prompt'a göre kümeleyip modelleri birebir eşleştirme:** S6E9'da karşılığı bulunamadı; yazılmadı.
- **Eşik optimizasyonu:** S6E9'da bulunamadı; `bias.py` yeni yazıldı.
- **scratchpad betikleri:** repo'da değiller ve oturum kapanınca kaybolacaklar. İşe yarayan kısımları
  `residual_scan`, `blend.add_candidate_test`, `refit_sim` ve `adversarial` olarak taşındı.

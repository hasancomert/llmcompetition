# PLAN: Kaggle "Name That LLM"

Görev: 40–250 kelimelik bir metni 18 modelden hangisinin yazdığını bulmak. Metrik balanced accuracy (BA). Train 2.699
metin / 150 prompt; gizli test ~4.000 metin, hepsi train'de olmayan prompt'lardan, model başına sayılar eşit değil,
public (%40) ve private (%60) arasında model karışımı farklı. Kod yarışması: internetsiz, ≤ 9 saat, çıktı
`submission.parquet` (`id`, `model`). Son tarih 13.01.2027.

Bu plan iki önceki projenin devir notlarını (`handoff/`), onlardan çıkarılan kodu (`reuse/`) ve kullanıcının fikirlerini
birleştirir. Etiketler handoff'lardaki gibi: **[TAŞINIR]**, **[UYARLANMALI]**, **[TAŞINMAZ]**. Kaynak sütununda
"EV §n" = `handoff/HANDOFF_EV.md` bölüm n, "Airline §n" = `handoff/HANDOFF_AIRLINE.md` bölüm n; iki el kitabı aynı
dersi farklı etiketlediyse ikisi de yazıldı.

## 0. Kaynaklar: hangi dosya hangi projeden

Gelen iki zip'te de el kitabı aynı adı taşıyordu (`HANDOFF_NAME_THAT_LLM.md`). Hangi projeye ait oldukları içerikten
belirlendi; dosyaların içeriği değiştirilmedi.

| Repo yolu | Kaynak | Nasıl anlaşıldı |
|---|---|---|
| `handoff/HANDOFF_EV.md` | `…handoffpvep.zip` içindeki el kitabı | Başlık "Kaggle Playground S6E9 → Name That LLM"; hedef `Will_Buy_EV`, gelir/mesafe sütunları; "pvep" = S6E9 proje kısaltması |
| `handoff/HANDOFF_AIRLINE.md` | `…name_that_llm_handoff.zip` içindeki el kitabı (ayrıca tek başına gönderilen `.md` ile bayt bayt aynı) | Başlık "Kaggle S6E10 (+ S6E9 dersleri)"; Age, Flight Distance, gecikme sütunları = Airline Passenger Satisfaction |
| `reuse/ev/` | `…handoffpvep.zip` → `reuse/` (10 modül + `tests/`) | aynı zip |
| `reuse/airline/` | `…name_that_llm_handoff.zip` → `reuse/` (8 modül + `run_selftests.py`) | aynı zip |

Airline el kitabı S6E9 derslerini de içeriyor; onları S6E9'un `LESSONS.md` dosyasından alıyor. EV el kitabı ise doğrudan
S6E9 kodundan ve günlüklerinden yazılmış. İki el kitabı S6E9 hakkında şu yerlerde ayrışıyor:
- Ç1 (`tok` kazancı),
- Ç12–Ç14 (yedek gönderim, logit-özellik etiketi, harman üstü düzeltme),
- S6E9 tepe tırmanma adım sınırı: EV §5 100 adım, Airline §5.1 200 adım (Ç7).

Artık model sayıları da farklı koşulardan aktarılıyor:

| Ölçüm | EV | Airline |
|---|---|---|
| Artık XGBoost | 20 kat, 2 tohum ortalaması 0.94661 | 4 tohum ortalaması 0.94659 |
| Artık LightGBM | 10 kat 0.94656, 20 kat 0.94659 | 10 kat 0.94656 (aynı); son harmandaki üye 0.94657 |

Bunlar karar etkilemez. Diğer ortak S6E9 sayıları iki el kitabında aynı.

Gelen dosyalarda yarışma verisi (csv/parquet) ya da model ağırlığı **yoktu**; yalnız `.md` ve `.py`.

Bu ortamda iki paketin kendi testleri de çalıştırıldı:

| Paket | Komut | Sonuç |
|---|---|---|
| `reuse/airline` | `python reuse/airline/run_selftests.py` | 8/8 geçti |
| `reuse/ev` | `python reuse/tests/test_reuse.py` | 12/12 geçti |

`reuse/ev` testleri `from reuse import …` beklediği için, içeriğe dokunmamak adına paket `reuse/` adlı geçici bir kopyada
çalıştırıldı. Sürümler: numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10.
El kitaplarında xgboost 3.2.0, EV'de numpy 2.4.6 yazıyor. Yalnız XGBoost'lu iki sahte veri sayısı değişti; neden
muhtemelen xgboost sürümü:

| Test | Handoff | Bu ortam |
|---|---|---|
| airline XGBoost artığı | 0.5349 | 0.5351 |
| ev XOR testi, XGBoost | 0.682 | 0.696 |

Diğer bütün sahte veri sayıları aynı çıktı.

## 1. Taşıdıklarımız [TAŞINIR]

| # | Ne | Kaynak | Bu projede |
|---|---|---|---|
| T1 | Tek sabit bölme, bütün modeller aynı katlarda | EV §1 [TAŞINIR]; Airline §1.1 [UYARLANMALI]. S6E10 istifi 5/10/15 katlı üyeleri karıştırdı, Airline §5.1 | `folds.csv` bir kez yazılır, her betik onu okur. Bizim kuralımız S6E10'dan katı: harman/istif yalnız bu bölmeden gelen OOF'larla |
| T2 | Hedeften türeyen her şey (istifçi, artık modelin taban logit'i, kaydırma, kodlama) dış katın yalnız eğitim satırlarında öğrenilir | Airline §1.1 [TAŞINIR]; EV §4, §6 [TAŞINIR]; EV §1 [UYARLANMALI]: iç katlar da gruplu olmalı | iç katlar da prompt'a göre gruplu |
| T3 | Her seçim (ağırlık, C, kaydırma, λ) iç içe ölçülür; gönderim iç içe skora göre seçilir | Airline §1.1, §5.2 [TAŞINIR]; EV §6 [TAŞINIR]; EV §5 iç içe ağırlık [UYARLANMALI]: gruplu olmalı | `reuse/airline/hill_climb.nested_hill`, `stacker.nested`, `reuse/ev/bias.nested_bias`; her çağrıda gruplu meta katlar |
| T4 | Kararı CV verir, public LB vermez; küçük adım için gönderim harcanmaz; küçük ama iç içe ölçülmüş adımlar birikir, LB'de görünmemeleri yanlış oldukları anlamına gelmez | Airline §1.4, §8; EV §8 | public %40 ve karışımı private'tan farklı: daha da zayıf sinyal |
| T5 | Aynı katlarda eşli karşılaştırma + birden çok meta tohum; kaynak ablasyonu | Airline §5.2 | `stacker.paired_compare`, `ablate`; tek modeller için `src/compare.py` |
| T6 | En güçlü değil en farklı üye ağırlık alır; yeni üye, harmana katkısıyla seçilir | Airline §2.2, §5.1, §5.2 (en farklı üye); EV §5 `add_candidate_test` [TAŞINIR] | TF-IDF, stil, tokenizer, LLM üyeleri ayrı aileler; yeni üye önce `blend.add_candidate_test` (gruplu) ile süzülür |
| T7 | Tohumları harmandan önce ortalamak (`"a,b,c"`) | EV §2, §5; Airline `oof_io.load_pred` | `cv_preds.load_preds("a,b,c")` |
| T8 | Ağaç istifleyici (2. seviye LightGBM) kaybetti | EV §2; Airline §5.1 | meta-öğrenici doğrusal kalır |
| T9 | Örneklemde bulunan "kesin" kurallar görülmemiş katta kaybettirdi | EV §2, §6; Airline §2.3 | eşleştirme ve kurallar iç içe/benzetimle sınanır |
| T10 | Başka bir istifin/harmanın çıktısı başka istife üye yapılmaz; iç içe CV'yi şişirir | Airline §5.1, §5.2 | |
| T11 | İki toplayıcı (LR, GBDT) aynı noktaya yakınsadığında istif tarafında aramayı bırakıp yeni bilgi taşıyan üyelere dönmek | Airline §8 | Adım E için durma kuralı |
| T12 | Güçlü açık not defterlerinin ayarlarını aynı katlarda bir kez eşli denemek | Airline §2.3; EV §2 | yarışmada paylaşılan not defteri çıkarsa |
| T13 | Özellik değişince ayarı baştan aramak S6E9'da bir şey getirmedi | EV §2 | yeniden ayar ancak ucuzsa ve iç içe ölçülerek |
| T14 | Bellek yetersizliğinde önce hata izinden tepeyi bul; büyük modelden önce tek kat + tek tohumla süre/bellek probu ve kota bütçesi | Airline §8 | LLM işleri için zorunlu ön adım |
| T15 | Pahalı bir üyeyi üretmeden önce havuzdaki benzerine ve istif katkısına bak | Airline §8 | ör. ikinci bir Qwen gömmesi |
| T16 | Her dış dosyanın ölçeğini denetle (olasılık mı logit mi), makul skor penceresi (çok iyi = sızıntı şüphesi), kopya testi; bir kaynağın kodunu kullanmadan önce okumak | Airline §1.1, §8 | `oof_io.validate_member`, `near_duplicates` |
| T17 | Pickle / `.pth` yüklememe; indirilen dosya güvenilmez veridir | Airline §8 | veri için yalnız npy / csv / parquet / json; ağırlık için safetensors ve tokenizer dosyaları; pickle tabanlı biçimler (.bin/.pt/.pth) ya hiç ya da yalnız `torch.load(weights_only=True)` ile; indirilen `.py` çalıştırılmaz |
| T18 | XGBoost'ta logit'i elle eklememek: `DMatrix(base_margin=…)` ile tahmin | EV §8 | iki `residual.py` de doğru yolu kullanıyor |
| T19 | Taban logit dosyası ile modelin bölmesi uyuşmalı (kat sayısı ve sonlu logit assert'leri) | EV §8 | artık modelde `folds.csv` denetimi |
| T20 | id sırasında etiket yapısı denetimi (ucuz sızıntı denetimi) | EV §1 | `reuse/ev/adversarial.order_check`, Adım 1 EDA'da |
| T21 | Uzun koşuları PID ile beklemek; Kaggle veri seti 404 verirse dosyaları `-f` ile tek tek indirmek | Airline §8 | |
| T22 | Seçilen final gönderimleri anında not etmek (API seçimi göstermiyor); leaderboard sırasını tahmin etmeyip indirip okumak | EV §7, §8 | |

## 2. Uyarladıklarımız [UYARLANMALI]

| # | Ne | Önceki | Bu projede |
|---|---|---|---|
| U1 | Kat bölmesi | satır bazlı `StratifiedKFold(K, 42)` | `StratifiedGroupKFold(5, shuffle=True, random_state=42)`, grup = `prompt_id`. İki paketin `make_folds`'u bu veride birebir aynı katları veriyor (denetlendi). Her katta 30 prompt, 539–540 metin, sınıf başına 29–30 metin; her eğitim katında 18 sınıfın hepsi var |
| U2 | Kat sayısı | 5 → 10 kat +0.0001 civarı (AUC) | şimdilik kullanıcı kararıyla 5. 10 kat = katta 15 prompt; ileride yalnız ucuz üyelerde ve eşli ölçümle denenebilir |
| U3 | Gürültü tabanı | AUC'de ±0.000004 eşli | Havuzlanmış OOF'ta bir metin, sınıfının recall'unu ~1/150, BA'yı ~0.00037 oynatır. Tek bir doğrulama katında (sınıf başına 29–30 metin) bir metin kat BA'sını ~0.0019 oynatır. Tek bir OOF BA'nın standart hatası ≈0.01; bu i.i.d. alt sınır, prompt kümelenmesiyle daha büyük. Gerçek tahmin `src/compare.py`'deki prompt düzeyi bootstrap. Karar: aynı katlarda **eşli** fark + prompt'lar üstünde eşli bootstrap. Seçim gerektiren adımlarda 3+ **meta-kat** tohumu; taban bölme T1 gereği sabit |
| U4 | Erken durdurma | doğrulama katında; S6E10 RealMLP'de yanlılık ~0.00001 ölçüldü (700 bin satır), S6E9'da ve küçük katta ölçülmedi | doğrulama katında **yok**: sabit tur sayısı ya da dış katın eğitim satırlarında gruplu iç ayrım. İki pakette de iç gruplu ES yok, `src/` sarmalayıcısı yazıldı (`train_gbdt.fit_fold`: iç `StratifiedGroupKFold(4)` ile tur seçimi, sonra tüm eğitim katında yeniden fit). Bir kez doğrulama katında ES ile karşılaştırılır |
| U5 | Tepe tırmanma | sıra uzayında AUC | olasılık / log-olasılık uzayı; BA basamaklı olduğu için log loss ile ağırlık seçme seçeneği de ölçülecek |
| U6 | L2 lojistik istif | C=0.001, 700 bin satır, 267 üye | 18×18×üye ağırlık, 2.699 satır: C iç içe yeniden seçilir; aile bazlı L2 (LLM / TF-IDF / stil) |
| U7 | Artık model (S6E9 GLR margin) | ikili GLR logit'i | taban: LLM sınıflandırıcısının (N×18) logit'i; üstüne stil + tokenizer özellikli LightGBM. Eğitim satırlarının logit'i iç katlardan, doğrulama/test satırlarınınki dış kat modelinden (`reuse/ev/residual.leak_free_margins`; kaydedilmiş OOF dosyası değil): dış 5 × iç 5 |
| U8 | Logit'i başlangıç değeri yapmak ile özellik yapmak | S6E9: RealMLP'ye özellik olarak verilince düştü, GBDT'lerde başlangıç olarak yükseldi | iki kullanım seçilen model ailesinde ayrı ölçülür (Ç13) |
| U9 | Tokenizer özellikleri | GPT-2 parçaları kategorik anahtar | farklı lab tokenizer'larıyla token sayıları, karakter/token ve tokenizer'lar arası oranlar. Fayda birden çok model ailesinde ayrı ayrı (LightGBM ve doğrusal model) ve sonra harmana katkıyla ölçülür (Airline §3 [TAŞINIR]). İlk/son parça anahtarları ikinci aşamada (`tokenizer_feats.token_table`); açılış prompt'tan etkilendiği için dikkatle (EDA §7) |
| U10 | Tohum ortalaması | NN'de büyük, GBDT'de küçük kazanç | LoRA üyelerinde tohum varyansı ayrıca ölçülür |
| U11 | Düşmanca doğrulama | train/test ayrımı | test gizli (etkileşimde 1–3 satır) → yerelde yapılamaz. Yalnız dış veri / train katları arasında, `groups=prompt_id` ile |
| U12 | Tam veri modeli ile kat ortalaması | S6E9'da kat ortalaması kazandı (−0.00004) | test tahmini kat ortalaması (`cv_preds.cv_predict`); 9 saat sınırına göre `refit_sim` ile yeniden ölçülür |
| U13 | Sözde etiket / transdüktif yöntemler / grup özellikleri | gürültü düzeyinde; en yakın deney `mix` (aynı anahtarlı train+test satırlarının grup ortalamaları) zarar verdi: EV §2 −0.00011, Airline §3 0.94608 → 0.94597 | test (~4.000) train'den büyük ve gönderim çalıştırmasında elde: kümeleme, eşleştirme ve TF-IDF sözlüğü testte kurulabilir. Küme/grup özellikleri tek tek, her biri eşli CV ile eklenir; yalnız kat içi sızıntısız benzetimle değerlendirilir |
| U14 | Final seçiminde public'e bakmak | public–private uyumu yüksekti | burada public ve private model karışımı farklı: public yalnız kaba bir bozulma alarmı |
| U15 | Kaggle runner | iki ayrı sürüm | `reuse/airline/kaggle_runner.py` (süre sınırlı birim zamanlayıcı + birleştirme) temel; EV'nin "Maximum batch GPU → bekle" ve "çıktı indirmek www.kaggleusercontent.com ister" notları eklenir. İnternetsiz gönderim not defteri iki projede de hiç kurulmadı: yeni iş. `build()` varsayılanı `internet=True`, gönderimde açıkça `False` |
| U16 | Kütüphane varsayılanları ve ayar araması | S6E10'da LightGBM varsayılanı +0.00099 geride kaldı; Optuna 700 bin satırda | stil GBDT'sinde varsayılan referans alınmaz; ayar araması yapılırsa sonucu iç içe doğrulanır |
| U17 | CatBoost GPU / CPU | aynı katlarda GPU ~0.00026 düşük (`border_count=128`) | CatBoost GPU'da koşacaksa aynı katlarda bir kez CPU ile karşılaştırılır |
| U18 | Büyük matrislerde bellek | 10 katlı kodlama bellekten öldü → parça parça | gömme çıkarımı ve TF-IDF matrisleri parça parça |
| U19 | Final ikili kuralı ("en iyi iç içe CV + iyiler içinde en farklı") | Airline §7 [TAŞINIR]; EV §7 [UYARLANMALI] (farklılık ölçüsü) | farklılık = iki gönderimin aynı etiketi verdiği satır oranı (`blend.label_agreement`); yedeğin rolü için Ç12 |

Taşınmayanlar [TAŞINMAZ]:
- tablo hedef kodlamaları (te1/te2/te3; bkz. Ç3) ve sayısal hane/kutu özellikleri (dig/dig2/bins),
- sentetik üretim izleri,
- sıra uzayı ve AUC'ye özgü istif ayarları,
- orijinal veri satırları,
- private kayma sayıları,
- S6E9 sınır kuralları (`RULES`, `DEADZONE`).

## 3. İki el kitabı / iki kod paketi arasındaki çelişkiler ve kararlar

| # | Konu | HANDOFF_EV / `reuse/ev` | HANDOFF_AIRLINE / `reuse/airline` | Karar |
|---|---|---|---|---|
| Ç1 | S6E9 `tok` kazancı, LightGBM 5 kat | "0.94573 → 0.94608" (+0.00035, taban `base,freq,dig,dig2,te1,te3,bins`, lr 0.1); ayrıca §2: `chain` `tok`'suz 0.94573 → 0.94605, `tok` ile birlikte ek kazanç yok | "LightGBM 5 kat 0.94606 → 0.94608" (+0.00002) | Kısmen uzlaşıyor: EV'nin +0.00035'i `chain`'siz tabana göre. Airline'ın tabanı (≈0.94605) `chain` kazancını zaten içeriyor ve EV'ye göre `tok` `chain`'in üstüne neredeyse bir şey katmıyor. Sonuç değişmez: tokenizer fikrinin kazancı bu projede sıfırdan ölçülür |
| Ç2 | OOF / test kayıt düzeni | `oof_<tag>.npy`, `pred_<tag>.npy`, `meta_<tag>.json` (sınıf sırası); dtype korunur (`cv_predict` çıktısı float64) | `oof/<tag>.npy`, `preds/<tag>.npy`, float32'ye çevrilir, sınıf sırası yok. İkisi de satır başına kat numarası verebiliyor (`oof_io.fold_ids`) | **EV düzeni** (`cv_preds.save_preds/load_preds`): 18 sınıf adının sırası dosyayla birlikte saklanmalı. Yer: `outputs/preds/`. `folds.csv` satır başına kat numarası. EXPERIMENTS.md'ye yalnız `common.log_experiment` yazar; Airline `log_row` (6 sütun, numarasız) kullanılmaz. `FoldCheckpoint` gerekirse `root=outputs/` ile |
| Ç3 | Hedef kodlamanın taşınabilirliği | `te1` [UYARLANMALI]: metin anahtarları (parça önekleri) için, gruplu iç katlarla (`target_encoding.py`) | `te1` [TAŞINMAZ]: sentetik tablonun değer kimliği artefaktı | Tablo sütunu kodlaması taşınmaz (Airline haklı). Metinde tek uygulanabilir yer açılış/kapanış token anahtarları; düşük öncelik, yalnız ölçülerek |
| Ç4 | Düşmanca doğrulama | 3 kat, grup yok, lr 0.1; ayrıca `order_check` (id sırası) | 5 kat, `groups=` desteği, lr 0.05 | `adversarial_auc` için **Airline sürümü** (gruplar olmadan sınıflandırıcı prompt'un konusunu öğrenir); `order_check` EV'den kullanılır |
| Ç5 | Artık modelde erken durdurma | varsayılan sabit tur (300), ES isteğe bağlı | `lgb_residual` / `xgb_residual`, `yva` verilince doğrulama katında ES (2000 tur, 100 sabır); `yva` yoksa 2000 sabit tur | **Sabit tur ya da iç gruplu ES**; doğrulama katında ES kullanılmaz (U4). Airline fonksiyonları sabit turla kullanılırsa `rounds` açıkça verilir |
| Ç6 | İç kat tohumu (artık taban) | `inner_seed=43` | `seed=0` | Önemsiz; sabitlenip kayda yazılır (43) |
| Ç7 | Tepe tırmanma | en çok 100 adım; `hill_climb` → `(ağırlık, skor)`, metrik string, `nested_hill` katları kendi kurar (`groups=` ile) ve `full_weights` + `avg_weights` döner. EV §5'e göre S6E9 sınırı 100 adım | en çok 200 adım; `hill_climb` → `(sayımlar, ağırlıklar)`, metrik çağrılabilir + `higher_is_better`, `nested_hill` meta katları dışarıdan alır, varsayılan `avg_weights=True`. Airline §5.1'e göre S6E9 sınırı 200 | API'ler karıştırılırsa sessiz hata çıkar (`w, s = …` öbüründe `(sayım, ağırlık)` olur). Adım E için **Airline `hill_climb` / `nested_hill`** kullanılır; meta katlar `folds.csv`'den, adım sınırı 200. EV'den yalnız `add_candidate_test` ve `label_agreement` |
| Ç8 | LR istif girdisi / C | ham log-olasılık, C=1.0; S6E9'da tepe tırmanmayı geçmedi | satır içi merkezlenmiş log-olasılık, C=0.001; S6E10'da s08'den itibaren kullanılan yöntem (s03'te tepe tırmanmayla aynı düzey) | Merkezleme her üyeden bir yönü (satırın ortalama log-olasılığı, yani güven düzeyi) atan doğrusal bir izdüşüm. Etkisi ölçülmedi, ikisi iç içe karşılaştırılır. C bu veride iç içe seçilir; tepe tırmanma ile istif eşli karşılaştırılır |
| Ç9 | CatBoost artık modeli | var; eğitim katında sınıf eksikse hata | yok | Gruplu katlarda her eğitim katında 18 sınıfın hepsi var (denetlendi), CatBoost kullanılabilir; öncelik düşük |
| Ç10 | Testin OOF'tan yumuşak olması | geçmiyor | test = kat ortalaması; kontrast testte OOF'un 0.967 katı (S6E10'da etkisiz bulundu) | Sınıf kaydırması ve eşleştirme kat ortalaması olan test olasılıklarına uygulanacak: oran ölçülür, gerekirse sıcaklık düzeltmesi |
| Ç11 | Sınıf başına kaydırma kodu | `bias.py` yeni yazıldı (sahte veri: 0.352 → iç içe 0.376) | "`reuse/` altında bir eşik/bias modülü uyarlanamadı, çünkü uyarlanacak kod yok" | `reuse/ev/bias.py` kullanılır (gruplu); kazanç yalnız iç içe |
| Ç12 | S6E9'da yedek gönderimin rolü | §7: "S6E9'da yedek gerekmedi çünkü en iyi private yine en iyi CV'li gönderimdi" | §7: "ikincisi yedek işlevi gördü" | Aynı olgu (en iyi private = en iyi CV'li gönderim) iki farklı yorumla aktarılıyor. Bu yarışmada public/private karışımı farklı olduğu için yedek gerçekten gerekebilir: ikinci seçim "iyiler içinde en farklı" kuralıyla yapılır |
| Ç13 | Logit'i özellik olarak vermek | §2 [TAŞINIR] ("taban logit'i başlangıç değeri olarak verilmeli"), §4 aynı ders [TAŞINMAZ] etiketiyle | §4 [UYARLANMALI]: "başlangıç noktası yapmak ile özellik yapmak ayrı ölçülmeli … iki ölçüm farklı model ailelerinde" | Varsayılan başlangıç değeri (`init_score`); seçilen ailede bir kez özellik kullanımıyla karşılaştırılır (U8) |
| Ç14 | Harmanın üstüne artık düzeltme | §2 [UYARLANMALI]: "harmanın ötesinde öğrenilecek yapı olup olmadığının ucuz testi" | §4 [TAŞINIR]: "bir harmanın üstüne `init_score` ile düzeltme getirmedi; artık model tek güçlü tabanın üstünde işe yaradı" | Artık model üye olarak tek güçlü tabanın üstüne kurulur; son harmanın üstüne düzeltme yalnız ucuz bir tanı olarak koşulur |
| Ç15 | Kesişim anahtarlarının kodlaması | §2 [TAŞINIR]: S6E9'da `te2` −0.00021, `tokx` −0.00017 zarar verdi | §2.1: S6E10'da `te2` +0.00030, `tefd` +0.00021 kazandırdı ([TAŞINMAZ], tabloya özgü) | Sonuç projeye göre değişmiş. Karar seyreklik gerekçesine dayanır: 2.699 metinde anahtar×anahtar hücreleri çok seyrek, yapılmaz |
| Ç16 | Gruplu yardımcıların varsayılanı (iki pakette de) | `make_folds`, `nested_hill`, `add_candidate_test`, `stack_logreg`, `leak_free_margins`, `nested_bias`, `simulate_refit`, `cross_fit_chain`, `residual_scan.residual_te_gain` hepsi `groups=None` ile satır bazlı bölmeye düşer; `adversarial_auc`'ta `groups` yok | `make_folds`, `adversarial_auc`, `paired_compare`, `residual.nested_margins` de `groups` verilmezse grupsuz böler; `nested_hill` ve `stacker.nested` meta katları dışarıdan alır | Proje kodunda her çağrı `groups=prompt_id` (ya da `folds.csv`'den kurulmuş gruplu katlar) ile yapılır; `src/` sarmalayıcıları bunu zorunlu kılar |
| Ç17 | Log loss tanımı | `blend.log_loss`: `clip(P, 1e-9, 1)`, yeniden normalize yok; `hill_climb(metric="logloss")` ve `add_candidate_test` bunu kullanır | `metrics.mlogloss`: `clip(P, 1e-15, 1)` sonra satırlar yeniden normalize | Sıfıra yakın olasılıklarda iki tanım farklı amacı optimize eder. Bütün raporlama ve log loss ile seçim **Airline `metrics.mlogloss`** ile; EV fonksiyonları `metric="balanced_accuracy"` ile ya da hiç kullanılmaz |
| Ç18 | Kaggle runner çıktı düzeni | `fetch` istenen klasöre her dosyayı indirir | `build(out_dirs=("oof","preds"))`, `fetch(subdirs=("oof","preds"))` yalnız `*.npy` kopyalar; günlük olarak küçük harfli `experiments.md` | Airline düzeni `meta_<tag>.json`'u taşımaz, `cv_preds.load_preds` açamaz. GPU çekirdekleri `outputs/preds/` düzeninde (`*.npy` + `*.json`) yazar; `src/` sarmalayıcısı ikisini de indirir ve sınıf sırasını `CLASSES` ile denetler. `reuse/` değiştirilmez |

Ayrıca: iki pakette de `balanced_accuracy` aynı tanımda (y'de olan sınıfların recall ortalaması). `reuse/airline/metrics.py`
sklearn ile birebir eşitliği test ediyor; proje kodu onu kullanır.

## 4. Kod düzeni

```
data/            yarışma verisi (git'e girmez)
models/          tokenizer dosyaları, ağırlıklar (git'e girmez; Kaggle'a dataset/model olarak yüklenir)
outputs/         OOF/test tahminleri, özellik önbellekleri (git'e girmez)
  preds/         oof_<tag>.npy, pred_<tag>.npy, meta_<tag>.json   (reuse/ev/cv_preds düzeni)
  folds.csv      id, prompt_id, model, fold                         (git'e girmez: *.csv)
handoff/, reuse/ değiştirilmeden saklanır
src/             proje kodu
  common.py      yollar, sınıf sırası, veri/kat okuma, kayıt, deney satırı
  make_folds.py  folds.csv üretimi (bir kez; yeniden koşunca aynı dosyayı vermek zorunda)
  eda.py, confusion.py   EDA.md'deki bütün sayılar
  style_features.py, tokenizer_features.py   özellikler (durumsuz: tek metnin fonksiyonu)
  train_*.py     her üye ailesi için bir betik; her koşu EXPERIMENTS.md'ye satır yazar
  compare.py     iki üyenin eşli karşılaştırması (prompt üstünde bootstrap)
kaggle/          GPU not defterleri ve gönderim not defteri
```

İçe aktarma düzeni: `src/common.py` hem `reuse/airline`'ı hem `reuse/`'u `sys.path`'e ekler.
- `reuse/airline` modülleri birbirini düz adla içe aktarır. Bu yüzden düz `import residual` / `import adversarial` /
  `import kaggle_runner` **airline** sürümüdür.
- EV modülleri paket yoluyla gelir: `from ev import cv_preds, blend, residual, bias`.
- Aynı adlı modüllerin (adversarial, residual, kaggle_runner) hangisinin kullanıldığı böylece içe aktarma satırından
  bellidir.

Sınıf sırası 18 etiketin alfabetik sırasıdır ve sabittir; `meta_<tag>.json`'da saklanır, her yüklemede denetlenir.
Test tahmini kat modellerinin ortalamasıdır (U12; `cv_preds.cv_predict`). Yerelde gerçek test yok (3 satır), bu yüzden
yerel `pred_<tag>.npy` yalnız örnek test içindir. Gerçek test olasılıkları gönderim not defterinde üretilir.

## 5. Yol haritası

### Adım 0 (bu commit)
Dosya yerleştirme, `.gitignore`, bu plan.

### Adım 1: Veri ve doğrulama (CPU)
- `EDA.md`: (prompt, model) sayımları, model başına uzunluk, biçim izleri (markdown, madde, em-dash, kıvrık tırnak, LaTeX,
  emoji, paragraf, başlangıç/bitiş kalıpları), aynı lab modellerinin benzerliği, id sırası denetimi (T20). Her sayı
  `src/eda.py` ile üretilir.
- `folds.csv`: `StratifiedGroupKFold(5, shuffle=True, random_state=42)` / `prompt_id`. Her betik bunu okur.
- Kayıt: `cv_preds.save_preds` + BA ve log loss + EXPERIMENTS.md satırı.
- GPU çekirdekleri (A, C) aynı katları şöyle alır: `folds.csv` etiket içerdiği için git'e girmez; private bir Kaggle
  dataset'iyle bağlanır. Çekirdek ayrıca katları yeniden üretip sha256 önekini (`647ff7926782e8eb`) ve
  `cv_preds.same_partition`'ı denetler. `StratifiedGroupKFold` çıktısı sklearn sürümleri arasında garanti değil, bu
  yüzden yeniden üretilen değil bağlanan dosya esastır. `folds.csv` ya da veri hiçbir zaman `code_files` ile
  `run.py` içine gömülmez; gömülürse base64 olarak commit'lenen dosyaya girer.

### Adım 2: Hızlı baseline'lar (CPU), sonra DUR
1. Karakter ve kelime n-gram TF-IDF + LogisticRegression / LinearSVC. TF-IDF sözlüğü dış katın eğitim satırlarında fit.
   Testte gönderim sırasında train + test üstünde fit etmek transdüktif bir seçenektir, ayrıca ölçülür. LinearSVC'nin
   olasılığı yok: marjlar, eğitim katının gruplu iç bölmesinde fit edilen bir softmax sıcaklığıyla N×18 olasılığa
   çevrilir. Bu kalibrasyon da hedeften türediği için T2 geçerli; varsayılan (grupsuz) `CalibratedClassifierCV` kullanılmaz.
2. Stil özellikleri (Adım 1'deki izler, noktalama oranları, cümle/paragraf istatistikleri) + LightGBM (sabit tur ya da iç ES).
3. Tokenizer özellikleri: o200k (OpenAI), Qwen3 ve Qwen3.5, GLM, Kimi, Gemma (Gemini yerine) token sayıları ve oranları.
   - Dosyalar Kaggle Models'tan yalnız tokenizer dosyası olarak alınır (HF bu ortamda kapalı). `models/tokenizers/`
     altında SHA-256 ile tutulur, gönderimde private Kaggle dataset girdisi olur (`MODELS_LICENSES.md`).
   - `reuse/ev/tokenizer_feats.load_hf_tokenizer` test edilmemişti (transformers yok). Proje kendi yükleyicisini
     (`src/tokenizer_features.py`: `tokenizers` + `tiktoken`, transformers'sız) kullanır. Her tokenizer'ın internetsiz
     yüklendiği ve örnek metindeki token sayısının kaynakla tuttuğu denetlenir.
   - Fayda CV'de birden çok ailede ölçülür (U9): stil + tokenizer ile yalnız stil, aynı katlar, eşli.

### Adım 3+ (onaydan sonra; şimdilik plan)
Önerilen sıra Adım 2 sonuçlarına göre güncellenecek:

| Sıra | İş | CV protokolü | Maliyet |
|---|---|---|---|
| A | Dondurulmuş LLM gömmeleri (Qwen3 ailesi; birkaç katmanın ortalama havuzu) + LR | 5 kat; gömmeler bir kez çıkarılır, LR kat içinde | Kaggle GPU, model başına < 1 saat (önce prob, T14) |
| B | Prompt centering: gömmeden prompt ortalamasını çıkarmak. Testte prompt yok → kümeleme. CV'de doğrulama katı da gerçek `prompt_id` değil kümeleme ile merkezlenir. Eğitim satırları da aynı kümeleme ile merkezlenir; gerçek `prompt_id` ile merkezlenmez, böylece train/test uyumsuzluğu olmaz. Kümelemenin `prompt_id`'yi ne kadar bulduğu (ARI, saflık) raporlanır. Testte küme sayısı bilinmiyor (~4.000 metin, ≥~220 prompt, düzensiz boyutlar). Bu yüzden k verilmez, tahmin edilir. Ölçüm 30 prompt'luk katta değil, havuzlanmış 2.699 OOF metninde (150 prompt) ve model düşürmeli, test benzeri boyutlarda yapılır; kat düzeyinde ARI testi olduğundan iyi gösterir | A'nın üstüne eşli; U13 (grup özellikleri tek tek) | CPU |
| C | LoRA ince ayar (Qwen3 küçük/orta) | 5 kat (+ artık model için iç katlar: 5 × 5) | GPU kotasının çoğu; önce tek kat probu |
| D | Artık model: taban (C ya da A) logit'i `init_score` (N×18); eğitim satırları için iç kat logit'leri, doğrulama/test için dış kat modeli (`leak_free_margins`); üstüne stil + tokenizer GBDT | sızıntısız taban (T2, U7), kat denetimi (T19) | A tabanıyla ucuz, C tabanıyla pahalı |
| E | Harman: log-olasılık tepe tırmanma ve L2 LR istif, iç içe; aile bazlı L2 | iç içe + eşli meta tohumlar; durma kuralı T11 | CPU |
| F | Sınıf başına logit kaydırma (`bias.nested_bias`, gruplu) | iç içe | CPU |
| G | Prompt eşleştirme: test kümelerinde Hungarian; aynı modele ikinci atamaya ceza λ (yumuşak kısıt) | CV'de gerçek `prompt_id` yerine kümeleme; doğrulama katında model başına eşit olmayan sayıları taklit etmek için rastgele model düşürme benzetimi; λ iç içe | CPU |
| H | Gönderim not defteri (yeni iş, U15) | Ayrıntılar aşağıda | |

Gönderim not defteri (H) şartları:
- 1–3 satırda çökmez; kümeleme ve eşleştirme atlanır.
- Çıktı denetimi: 18 geçerli etiket, her id bir kez, null yok. `reuse/airline/submission.check`'in `min_classes=2`
  varsayılanı 1 satırlık testte not defterini düşürür. `min_classes=min(2, n_test)` verilir ve gerçek denetleyici 1, 2
  ve 3 satırlık testlerle sınanır.
- 4.000 satır için süre tahmini.
- İnternetsiz bağımlılıklar açıkça listelenir. Kaggle imajında olmayan Python paketleri (ör. `tokenizers`, `tiktoken`,
  PEFT) tekerlek olarak bir dataset'te gelir. LLM/LoRA ağırlıkları Model ya da Dataset girdisi, tokenizer'lar ayrı bir
  dataset, `src/` + `reuse/` bir kod paketi olarak bağlanır.
- İnternetsiz deneme koşusu (internet kapalı, gerçek 3 satırlık test) erkenden yapılır: Adım 2'den hemen sonra, en
  basit üyeyle.

BA ile ilgili bir not (F ve G için):
- BA sınıf başına recall ortalaması olduğundan, kalibre olasılıklarda en iyi karar `argmax p(c|x) / π_train(c)` olur.
  Bu kural test sınıf oranlarına bağlı değildir.
- Train dengeli (sınıf başına 149–150), bu yüzden düz argmax zaten doğru kuraldır. Kaydırma yalnız sınıflar arası
  kalibrasyon farkını düzeltir; büyük kazanç beklenmez.
- Eşleştirmenin kazancı ise prompt yapısından gelir: train'de her prompt'a her model tam bir kez cevap vermiş (2.699
  çiftin tamamı tekil, yalnız p002 / claude-fable-5 eksik).
- Testte model sayıları eşit değil, yani bu kısıt testte gevşek; katı birebir atama zarar verebilir. Bu yüzden yumuşak
  kısıt ve benzetim.

GPU bütçesi:
- Kaggle haftalık kotası ~30 saat; S6E10'da 30,17/30 saatte bitti (Airline §8).
- Aynı anda en çok 2 GPU oturumu açılabiliyor (EV §8).
- Her büyük işten önce tek kat + tek tohum probu yapılır (T14).

## 6. Çalışma kuralları (kullanıcıdan)

- Kararlar CV'ye göre; her deney `EXPERIMENTS.md`'de (no, değişiklik, CV BA, CV log loss, varsa public LB). İşe yaramayanlar da yazılır.
- Gönderimi kullanıcı yapar. Ben not defterini push'lar, gönderilecek sürümü, beklenen CV'yi ve gerekçeyi bildiririm.
- Dış veri yalnız ücretsiz ve herkese açık, yalnız eğitimde; `EXTERNAL_DATA.md`'ye kaynak + lisans. Validation her zaman yarışma katları.
- Para harcayan, bir şeyi public yapan ya da büyük yön değiştiren her adımdan önce sorulur. Kaggle dataset ve not defterleri private.
- Kullanılan kod/model lisansları ticari kullanımı kısıtlamamalı; `MODELS_LICENSES.md`'ye yazılır.
- Yalnız gerçekten çalıştırılıp ölçülen sonuçlar raporlanır.

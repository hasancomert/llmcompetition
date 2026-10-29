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

### Adım 2 sonucu (ayrıntı: EXPERIMENTS.md, EDA.md)

- En iyi tek model: char+word TF-IDF + LR, CV BA 0.347 / log loss 2.036.
- En iyi iç içe harman (TF-IDF + stil/tokenizer LR): BA 0.392 / log loss 1.856.
- Tokenizer özellikleri tek başına zayıf (v2: LR 0.180, LightGBM 0.163). Stil özelliklerine eklenince log loss her ailede iyileşiyor,
  BA katkısı küçük ve gürültü sınırında.
- Lab tahmini ~%73 doğru. Hataların ~%55'i aile içinde (Gemini flash üçlüsü ve pro ↔ flash, GLM ikilisi, gpt-5.6 üçlüsü),
  ~%45'i lab dışında (GLM → Claude, Kimi/Qwen → GPT, claude-fable-5 ↔ GLM).

### Adım 2 sonrası önerilen sıra (onay bekliyor)

| sıra | iş | neden şimdi | kaynak |
|---|---|---|---|
| 1 | H-lite: mevcut harmanla internetsiz gönderim not defteri. Tokenizer + kod private dataset olarak; 1–3 satır testi, çıktı denetimi, süre ölçümü | İnternetsiz kurulum iki projede de hiç yapılmadı (U15). Riskini en ucuz üyeyle erkenden çözmek; ilk LB okuması da gelir (karar yine CV'den) | Kaggle CPU, GPU kotası harcamaz |
| 2 | G (+ B'nin kümeleme kısmı): prompt kümeleme + yumuşak Hungarian eşleştirmenin mevcut harman OOF'u üzerinde benzetimi. k tahmini, model düşürme benzetimi | Train'de her prompt'a her model bir kez cevap vermiş. Eşleştirme aile içi karışıklığı (en büyük kayıp) doğrudan hedefliyor; GPU gerekmez | CPU |
| 3 | A + B: dondurulmuş Qwen3 gömmeleri (küçük model, birkaç katman) + LR; prompt merkezleme kümeleme ile | Ucuz GPU probu; gömmelerin aile içi farkı yakalayıp yakalamadığını gösterir | ~1–2 GPU saati (tahmin, prob ile netleşecek) |
| 4 | C: LoRA ince ayar (Qwen3 1.7B/4B), önce tek kat probu, sonra 5 kat | Muhtemel en büyük sıçrama; en pahalı adım | GPU kotasının büyük kısmı |
| 5 | I (yeni öneri): çapraz-model log-olasılık özellikleri. Metnin token başına log-olasılık / sıra istatistikleri, aynı lab'ların açık akrabalarıyla (gpt-oss-20b, Qwen3.5 küçük, Gemma 4, GLM-4.7-Flash, Moonlight). Artık GBDT'ye (D) girdi | Tokenizer hipotezinin güçlü hâli: tokenizer sayıları zayıf ama tutarlı bir sinyal verdi | GPU; modeller gönderim not defterinde de koşmalı (9 saat bütçesine yazılır) |
| 6 | D, E, F | Üyeler oluştukça | CPU |

### Adım 3+ (onaydan sonra; şimdilik plan)
Plandaki iş listesi (sıra yukarıdaki öneriye göre):

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

### Adım 3 durumu (16 Ekim 2026; ayrıntı ve bütün sayılar: EXPERIMENTS.md)

| iş | durum | ölçülen sonuç |
|---|---|---|
| H-lite | bitti | `nllm-submit` v1 (CPU) ve `nllm-submit-gpu` v1 (GPU, 4 Qwen3) internetsiz koştu, çıktı denetimi geçti |
| G | bitti, bırakıldı | kümeleme v2 + yumuşak Hungarian; CV'de büyük kazanç, gizli testte etkisiz (LB ablasyonu) |
| A + B | A sürüyor, B bırakıldı | dondurulmuş Qwen3 gömmeleri ham 0.40. Merkezleme CV'de 0.55–0.56 verdi ama gizli testte zararlı |
| B her aileye | bitti | merkezleme her özellik ailesinde işe yaradı: lp + stil 0.33 → 0.40, stil + tok 0.30 → 0.38, char TF-IDF 0.33 → 0.41 |
| E (kısmen) | v3 harman | iç içe tepe tırmanma: BA 0.5931 / log loss 1.2539. Test benzeri tam boru hattı benzetimi (eşleştirmeli): hafif düşürme 0.616, sert düşürme 0.557 |
| C | ertelendi | LoRA probu (Qwen3-1.7B, 2 kat, 2 epoch) BA 0.265; donmuş gömmenin çok altında, eksik öğrenmiş |
| I | kısmen | 4 Qwen3'ün log-olasılık istatistikleri lp üyesinde. GLM-4-9B harmana +0.0015 (anlamsız), alınmadı. Gemma 4 T4'te çalışmadı |

- GPU kotası: Kaggle haftalık 30 saatlik GPU kotası 16 Ekim'de doldu.
- **16 Ekim public LB ablasyonları planı değiştirdi** (EXPERIMENTS.md "Public LB bulguları"):
  - Eşleştirme gizli testte etkisiz, prompt merkezleme zararlı. Gizli testte prompt başına az metin var.
  - v3 (merkezlenmiş) gönderilmeyecek. G ve B bırakıldı.
- 16 Ekim metin başına workflow'u (EXPERIMENTS.md "Metin başına iyileştirmeler"):
  - GPU blend 0.4430 → 0.4619 (fonksiyon kelimesi n-gram üyesi + aile uzmanları).
  - CPU blend 0.3923 → 0.4038.
  - Gönderim, deney betiklerini notebook'ta aynen yeniden oynatarak yapılıyor (`src/replay.py`, tarifler cpu2 / v4).
- Dış derlem (olegzholobov/name-that-llm-response-corpus) CV'de zarar verdi; kullanılmıyor.
  - 16 Ekim'deki ikinci değerlendirme (dal `featherless-extra-data`) de her ayarda zarar verdi.
  - Kullanıcı kararı: bırakıldı. Bayrak kapalı kalıyor, kod silinmiyor.
- Uyum testi v2 (dal `featherless-extra-data`; Kaggle Benchmarks + Featherless ile üretilmiş metinler):
  - Eşiği 15 etiketin 10'u geçti.
  - CV'de anlamlı kazanç yok: blend +0.000 ile +0.006 arası, gürültü içinde; zarar da yok.
  - Kullanıcı kararı (17 Ekim): ölçekleme yok. `--ext ft2` bayrağı kapalı kalıyor, kod silinmiyor.
- 16 Ekim akşamı `nllm-submit-cpu2` gönderildi: CV 0.4038, public LB **0.375**. Sabahki CPU v1 0.362 idi (CV 0.392);
  metin başına kazanç LB'ye taşındı.
- **17 Ekim durumu:** en iyi gönderim v5, 12 katmanlı gömmeler: CV 0.4719, public LB **0.435**. v4: 0.4619 / 0.422.
  - Katman 1–18 ham üyeleri ve uzmanları iyileştirdi ama birleştirici BA'sını değil; v5 kaldı.
  - Birleştirici BA'sı ~0.47'de doymuş görünüyor.
  - Kalan büyük kaldıraç LoRA. Tek başına zayıf ama tamamlayıcı: 2 fold'da +~0.009.
    - Kullanıcı kararıyla sonraya bırakıldı.
    - Kaggle GPU'su S6E10 TabPFN kararına bağlı; o oturum kullanıcının seçimini bekliyor.

### Yeni yön (18 Ekim, kullanıcı onayı): public SOTA'dan öğrenilenler

- Durum: public LB 0.435, 1.642 takımda 931. sıra. 425 takım ≥ 0.55; en iyi public notebook'lar 0.55–0.56.
- Public SOTA'nın temeli Lucas Boesen'in "LoRA x Adaptive Centering" notebook'u (0.549); fork'ları 0.552–0.563.
  Bizden farkları:
  1. Uyarlamalı prompt merkezleme.
     - Yalnız 2–18 metinlik test kümelerine, merkezlenmiş bir "ikiz" model uygulanıyor.
     - Düz tahmin herkes için korunuyor; ikizin kanıtı onun üstüne ekleniyor.
     - Tek kalan metinler hiç merkezlenmiyor. Bizim 16 Ekim'deki başarısızlığımız tam bu hataydı.
  2. Test setiyle standartlaştırma ve Sinkhorn sınıf dengeleme.
  3. Daha güçlü üyeler: LoRA24 (Qwen3-8B, 24 katman, tek başına ~0.48), dondurulmuş DeBERTa/ELECTRA, Claude içi kapı.
  4. Ek veri: Kaggle proxy'si ve API'lerle üretilmiş public dataset'ler (CC BY 4.0).
- Sıra:
  1. Uyarlamalı merkezleme. CPU'da yapılır; 1–2 / 1–5 / 3–8 metin/prompt simülasyonuyla ölçülür.
  2. Public ek verinin CV değerlendirmesi. Yalnız eğitim fold'larında; sızıntı kontrolü ve lisans kaydıyla.
     Paralel yürür.
  3. Test standartlaştırma + Sinkhorn.
  4. Kazanç varsa v6 notebook'u ve LB kontrolü.
  5. GPU açılınca (S6E10 31 Ekim'de kapanıyor): LoRA24, DeBERTa/ELECTRA, gerekirse kendi üretimimiz.
- Durum (18 Ekim gece; sayılar EXPERIMENTS.md ve EXTERNAL_DATA.md'de):
  1. Uyarlamalı merkezleme yapıldı. Kesim 0.85, C=0.001, w=0.5; benzetimde her yapıda olumlu (+0.011 … +0.072).
     v6 notebook'u (`nllm-submit-gpu` v4) 21:33 UTC'de push'landı; LB bekleniyor.
  2. Public ek veri (5 dataset, CC BY 4.0): CPU üyelerinde +0.022 … +0.043. v5 süper aile birleştiricisinde 3 CPU üyesi
     dış veriyle 0.4719 → 0.4831. Ek veri dalı ana dala birleştirildi (`featherless-extra-data`, 18 Ekim).
  3. Sinkhorn: nötr ya da olumsuz, benimsenmedi. Test standartlaştırma: tek üyede tam fold'da +0.006 / +0.010;
     yığın düzeyinde CV sürüyor.
  4. v6 public LB 0.479 (v5 0.435). v7 = v6 + dış veriyle CPU üyeleri + LM girdilerinde test standartlaştırma
     (birleştirici CV 0.4719 → 0.4871). Yerel replay'de notebook içi OOF 0.4871. Replay artık paralel aşamalarla
     koşuyor (süre yarıya indi). 19 Ekim 10:59'da push'landı.

### GPU planı: 17–24 Ekim haftası (kota 30 saat; kullanıcı kararı: öncelik S6E10)

Süreler, geçen haftaki koşuların loglarından tahmin. Kaggle'da aynı anda en fazla 2 GPU oturumu var.

| iş | proje | GPU-saat (tahmin) | zaman (UTC) |
|---|---|---|---|
| `s6e10-pfn15-og-s1a` (fold 0–7, seed 1) | S6E10 | 11.1–11.8 | Cmt 00:20 → ~11:30–12:10 |
| `s6e10-pfn15-og-s1b` (fold 8–14, seed 1) | S6E10 | 10.5–11.8 | Cmt 00:20 → ~10:50–12:10 |
| v4 gönderim not defteri (`nllm-submit-gpu`, replay v4) | NLLM | ~1.0 | ilk boşalan slotta, ~Cmt 11:00 |
| 12 katmanlı gömme (`nllm-gpu-embed-l12`) | NLLM | ~0.8 | ikinci slot boşalınca, ~Cmt 12:00 |
| LoRA denemesi (`nllm-lora-probe2`, 2 fold) | NLLM | ~1.0 | ~Cmt 13:00 |
| toplam | | ~24.4–26.4 | |
| tampon (S6E10'un eksik birimleri için bir tekrar turu ~2.9 sa dahil) | | ~3.6–5.6 | |

- **Gerçekleşen (Cmt 17 Ekim gece):**
  - S6E10 oturumu TabPFN seed-1 çekirdeklerini 00:20'de başlatmadı. Kullanıcıya üç seçenek sundu (hiç koşmamak,
    hafta içinde koşmak, şimdi koşmak) ve GPU'nun NLLM'e kalmasını önerdi.
  - 00:36'da GPU'da çalışan oturum yoktu; kota 30 saat. NLLM işleri kısa ve S6E10 sonradan koşsa da kota korumasına
    sığıyor: 30 − 2.8 ≥ 23.6 + 3. Bu yüzden 00:37'de v4 gönderim notebook'u (`nllm-submit-gpu` v2) ve 12 katmanlı
    gömme (`nllm-gpu-embed-l12` v1) push'landı.
  - LoRA probe2 ilk boşalan slota konacak.
  - Gerçekleşen GPU süreleri:
    - `nllm-submit-gpu` v2 (v4): 00:37–01:12, ~0.6 sa.
    - `nllm-gpu-embed-l12` v1: 00:37–00:56, ~0.3 sa.
    - `nllm-lora-probe2` v1: 00:56–02:00 arası bitti, ≤1.1 sa.
    - `nllm-submit-gpu` v3 (v5): 03:39–04:45, ~1.1 sa. 12 katmanlı train öznitelikleri notebook'ta çıkarıldı (+~8 dk).
      Dataset güncellemesi gece Kaggle'ın 429 sınırı yüzünden yapılamadı. 17 Ekim öğleden sonra yapıldı:
      `nllm-lmfeats` yeni sürüm, 12 katmanlı 4B/8B train öznitelikleri eklendi. Dataset private kaldı
      (isPrivate true, anonim erişim 403).
    - Kaggle kotasına göre toplam 2.95 sa (04:46). Kalan 27.05 sa.
    - Gizli test için yeniden koşma kotadan düşmüyor.
    - `nllm-gpu-embed-e18` v1 (8B/4B, katman 1–18): 09:02–09:18, ~0.27 sa. Toplam ~3.2 sa.
    - `nllm-submit-gpu` v4 (v6, 18 Ekim): 21:33–22:28, ~0.9 sa. Toplam ~4.1 sa. S6E10 bu hafta GPU kullanmadı (18 Ekim 22:00'de kernel listesi).
    - `nllm-submit-gpu` v5 (v7, 19 Ekim): 10:59 UTC'de push'landı. Kullanıcı onayı (19 Ekim): "Şimdi gönder". Tahmin ~0.8 sa (paralel aşamalar); v7 sonrası S6E10'a ~25 sa kalıyor (planlı 23.6 sa).
    - 19 Ekim 13:00 UTC: v7 hâlâ Kaggle kuyruğunda (QUEUED, 10:59'dan beri). S6E10 oturumu `s6e10-pfn15-og-s1a`'yı 11:11'de başlattı (RUNNING; s1b henüz yok). Kuyrukta beklemek kotadan düşmüyor; 8B modeli tek P100'e sığmadığı için T4×2 bekleniyor.
    - v7 kuyruktan 15:01'den sonra çıktı, 19:34'te COMPLETE. Notebook içi süre 3.362 sn (~1.0 GPU-saat; Kaggle CPU'ları yerelden ~2.3× yavaş: 1. aşama 1.797 sn, yerelde 770). Hafta toplamı NLLM ~5.1 sa. v7 19:35'te gönderildi.
  - LoRA probe2 sonucu:
    - Tek başına zayıf: 2 fold'da 0.4085; dondurulmuş 4B gömmeleri 0.4354.
    - Ama tamamlayıcı (EXPERIMENTS satır 493).
    - 5 fold'luk tam LoRA (kalan 3 fold ~2.1 sa) ve onunla gönderim (~1.3 sa) toplam ~3.4 sa.
    - S6E10 üst sınırla koşarsa (23.6 + 3 sa tampon) NLLM'e ~0.45 sa kalıyor; o durumda sığmaz.
    - Kullanıcı kararı (17 Ekim sabah): LoRA sonraya bırakıldı.
      - Kullanıcının bilgisayarında eğitim olmayacak; ~8 GB ağırlık indirilemez.
      - Geliştirme LoRA dışındaki yönlerle sürüyor.
  - 12 katmanlı gömmeler CV'de v4'ü geçti (0.4719 vs 0.4619; EXPERIMENTS.md). Sırada v5 gönderim notebook'u var,
    ~0.7 sa.
- Uyum testi v2'nin Kaggle tarafı (Benchmarks task'ları) CPU not defterinde çalışıyor; GPU kotası kullanmıyor.
- Sığmayabilecek iş: LoRA deneme sonucu iyiyse 5 fold'luk tam LoRA koşusu (~2.5 sa) ve onunla yeni gönderim not
  defteri (~1–1.5 sa).
  - S6E10 üst sınırı kullanırsa bu ikisi bu haftaya sığmıyor; haftaya kalır.
  - Kaydırmadan önce kullanıcıya sorulur.
- Zamanlanmış görev (`trig_01D8v1ENzCjSmBXhwSz7P57e`, Cmt 00:35):
  - Boş slot ve kota kontrolü yapıp sırayla push'lar.
  - Slot yoksa S6E10 kernel'lerinin beklenen bitişinde tekrar dener.
  - Gerçek GPU saatleri bu tabloya işlenir.
- Yeni yön: metin başına sınıflandırıcıyı iyileştirmek. Sıra:
  1. Ham GPU harmanı, düz argmax (`--members v2raw --match-lambda 0`); kota açılınca gönderilecek.
  2. 12 katmanlı gömmeler (ham). **Yapıldı (17 Ekim):** iç içe CV 0.4619 → 0.4719. Tarif v5 hazır; yerel replay
     0.4723. Gönderim notebook'u, Kaggle API'sinin 429 sınırı kalkınca push'lanacak.
  3. Düzgün eğitilmiş LoRA (C).
     - probe2 (4B, 2 fold) tek başına 0.4085; gömmelerden zayıf ama tamamlayıcı.
     - Kullanıcı kararıyla sonraya bırakıldı.
  4. Ham üyelerle argmax odaklı harman (E/F).


### Public çözümler incelemesi (21 Ekim; kullanıcı isteği: "İnsanların paylaştığı çözümleri kontrol et")

- Durum: LB zirvesi 0.667. En iyi public notebook'lar 0.597–0.612. Bizim v7: 0.503.
- Soy ağacı (her adımın public LB'si; notebook'lar okundu, çalıştırılmadı):
  - 0.563 "in-style" yığını (Travis Nguyen). Çok sayıda üye:
    - n-gram SVC ve stilometri;
    - dondurulmuş decoder/encoder'lar ve merkezli ikizleri;
    - üç LoRA24, UltraBERT;
    - Sinkhorn.
  - 0.592 Nishant Dahal, "Two-stage LoRA on the public pool".
    - 1. aşama: ~35 bin public pasaj (14 dataset), 1 epoch, sınıf ağırlıklı CE.
    - 2. aşama: resmi 2.699 metin, 2 epoch, yarı lr.
    - Qwen3-8B (24 katman) ve 14B (27 katman); ikisi de tek başına CV 0.569. Yalnız 1. aşama 0.483.
    - Yığın: 8B + 14B + 4 görünümlü n-gram SVC (train + havuz, CV 0.435) + merkezli ikiz + Sinkhorn. CV 0.593, LB 0.592.
    - İşe yaramayan dedikleri: tohum ortalaması, aile kapıları, kümelerde Hungarian, 2. aşamada daha çok epoch.
  - 0.600 kirilltym: %85 iki aşamalı dal + %15 eski yığın; 8B LoRA özniteliklerinde paylaşılan-prompt Gaussian kafası
    (prvsiyan).
  - 0.604, iki ayrı notebook:
    - prvsiyan v4: kendi iki aşamalı tohumları (38.642 pasajlık havuz) ve her LoRA'da Gaussian kafa.
    - sangam21: 0.600 + 14B'de Gaussian + boşluklu uzun tire kuralı + 12–18 metinlik kümelerde Hungarian.
  - 0.607 josephadamski: 0.600 + %10 ATLAS (Qwen3-8B adaptörü, kaynak koşullu token olabilirliği, metin başına 18 geçiş).
  - 0.612 kiza123123: sangam21 + kıvrık tırnak kuralı + Hungarian 9–18.
  - 0.597 lightningv08: + OLMo2, SmolLM3 ve Qwen3.8-27B (%20).
- Bizde ölçülenler (EXPERIMENTS.md):
  - Tipografi kuralları (satır 598):
    - Train'de doğru: boşluklu uzun tire yalnız 4 sınıfta; Claude ve GLM'de kıvrık tırnak 1.049 metinde 0.
    - v7 OOF'unda 1 satır değişiyor, +0.0004. Char n-gram üyelerimiz bu işaretleri zaten taşıyor.
  - Hungarian (satır 591–597): ≥4 metinlik kümelerde birebir atama, ikizlerin üstüne.
    - Kazanç: 1–2 +0.0006, 1–5 +0.007, 2–4 +0.002, 3–8 +0.0135, 1–18 +0.057, 9–18 +0.059, 18 +0.082.
    - Hiçbir yapıda zarar yok. m = 4 her yapıda en iyisi (6/9/12 daha az).
    - Eskiden başarısız olan yumuşak eşleştirmeden farkı: güncel kümeleme (0.85) ve yalnız büyük kümeler.
  - Gaussian grup kafası (`gauss_head.py`; çift ve üçlü sonsalı brute-force integral ile birebir), satır 599–603.
    - İkizlerin üstüne Gaussian (w 0.25, T 3) + Hungarian (≥3 metinlik kümeler).
    - İkizlere göre kazanç: 1–2 +0.009, 1–5 +0.013, 2–4 +0.012, 3–8 +0.024, 18 +0.090.
    - Her yapıda olumlu; en kötü durumu en iyi olan ayar seçildi.
    - Gaussian tek başına (ikizsiz) 2–4'te ikizlerden kötü: ikizlerin yerine değil, yanına.
  - Havuzla CPU üyeleri (`--ext pool`): CV sürüyor.
- Alınacaklar (öncelik sırasıyla):
  1. İki aşamalı LoRA (GPU). Kod hazır:
     - `gpu_lora24.py`'de 'pre' işi: havuz, sızıntılı bloklar atılı, sınıf ağırlıklı.
     - Fold işleri 'pre'den başlıyor (`--init-pre`); iki T4'te DDP.
     - CPU duman testi (Kaggle `nllm-lora2s-smoke`, 21 Ekim) iki kez geçti: havuz, sha256 ve sızıntı ayıklaması kernel
       içinde; iki gloo işlemiyle DDP; 'pre' sınıf ağırlıklı, f0 'pre'den başladı. İkinci sürümde işlemler arası
       ağırlık farkı 0.0 (her iş sonunda kontrol ediliyor). Çıktılar ağırlıksız indirildi (`--fetch`).
  2. Gaussian grup kafası + Hungarian m = 3: replay tarifi `v7g` hazır (`gauss_twins.py`); v8'in parçası.
  3. Havuz satırlarıyla CPU üyeleri: 4 görünümlü n-gram SVC birleştiricide 0.4871 → 0.5030 (+0.016, P 0.006).
     Tarif `v7p` = v7g + bu üye. char+word LR'de havuz belirgin kazanç vermedi, alınmadı.
  - Alınmayanlar:
    - ATLAS: metin başına 18 geçiş, kazanç +0.007.
    - OLMo2 / SmolLM3 / 27B: pahalı, kazanç küçük.
    - Başkalarının public checkpoint'leri: OOF yok, CV ile doğrulanamaz; kullanıcı kararı gerekir.
- GPU planı (24–30 Ekim, ~7 sa; **kullanıcı 21 Ekim'de onayladı**):
  1. İki aşamalı LoRA, Qwen3-4B (24 katman).
     - 1. aşama: havuzun 10+ etiketli kısmı (17.860 satır, 1 epoch).
     - 2. aşama: 5 fold × 2 epoch, iki T4'te DDP.
     - Tahmin ~2.6 sa (probe2 hızından), sert sınır 3.5 sa.
  2. `nllm-gpu-embed-ext` (~0.7 sa).
  3. v8 gönderimi (~1.3–1.7 sa): v7p (v7g + havuzlu n-gram SVC) + LoRA üyesi ve ikizi. LoRA'nın CV kazancı
     yoksa v7p tek başına gönderilir.
  - 8B iki aşamalı LoRA 31 Ekim haftasına kalıyor: S6E10 kapanıyor, kota tamamen NLLM'e.
  - Eski plandaki "yalnız resmi veriyle 8B LoRA24" (~4.5 sa) bırakılıyor. Public CV'ye göre 1. aşamasız LoRA ~0.48,
    iki aşamalı 0.569.

### Yeni public notebook'lar (29 Ekim; 0.636–0.666; kullanıcı isteği, GPU'suz)

- Okunanlar: alexchilton 0.660 / 0.651, lightningv08 0.661 / 0.642, omerfarukmerey 0.663, ozertuu 0.666, kirilltym 0.642,
  fleongg 0.636, mathleteutd 0.649, mercurydust ("Who am I? V31"), romantamrazov, huangkx666, xianyu3, avazaku,
  dhruvtibarewal. Notebook'lar okundu, çalıştırılmadı; aşağıdaki LB sayıları onların beyanı.
- Ne oldukları: başkalarının eğittiği public checkpoint'lerin dev yığınları (Qwen3-14B / 8B / Llama fold
  adaptörleri, iki aşamalı 14B ve Qwen2.5-14B, %10–20 ATLAS, UltraBERT). OOF'ları bizim fold'larımızda yok.
  - 120 bin satırlık public derlem 1. aşamada / Qwen2.5-14B yeniden eğitiminde kullanılmış (beyan: +0.006).
  - alexchilton'ın kendi 5 omurgalı iki aşamalı harmanı tabloda yalnız 0.559 (ikizlerle 0.588); iç içe CV'si 0.6865
    tam 18'li gruplarla şişmiş.
  - Güçlü yığınlara üye eklemek: ±0.004.
- Gürültü: aynı kod farklı koşularda 0.651 / 0.643 / 0.651 aldı. Public LB'de 0.01'in altındaki farklar anlamsız.
- Hungarian: public LB A/B'lerinde 5 denemenin hepsinde kaybettirmiş (−0.009 … −0.017; "Full Ensemble + What
  Failed"). 0.66 notebook'ları kullanmıyor; bizim 16 Ekim yumuşak eşleştirmemiz de LB'de −0.006 almıştı.
  **v7p'den çıkarıldı** (v7p = v7 + Gaussian grup kanıtı + havuzlu n-gram SVC). Benzetimdeki kazancı gizli testin
  yapısına taşınmıyor.
- V31'in iki son adımı (`post_steps.py`, yeniden yazıldı), bizde ölçüldü (EXPERIMENTS 618–623):
  - Yumuşak Sinkhorn (s 0.45, ρ 0.35) ve Bayesçi tipografi (37 işaret, κ 24).
  - Tam OOF'ta her ayar düz birleştiricinin ±0.002'si içinde (0.5012–0.5038).
  - v7p ikizleri + Gaussian üstüne, benzetimde: yapıya göre −0.005 … +0.002; hiçbir ayar her yapıda olumlu değil.
    **Alınmadı.**
- Commit'te ucuz yol (alexchilton, mathleteutd, omerfarukmerey kullanıyor) **alındı**:
  - `build_submission.py --cheap-commit`: kayıt (save) koşusu `replay.py --check-inputs` ile yalnız ekli girdileri,
    GPU'yu, paket sürümlerini ve her adım betiğinin import'unu denetler, yer tutucu yazar. Tam replay yalnız gizli test
    koşusunda çalışır (`KAGGLE_IS_COMPETITION_RERUN` var ya da test 40+ satır).
  - Kaggle personeli: puanlama koşusu GPU kotasından düşmüyor, kayıt koşusu düşüyor (Q&A 453105). Gönderim başına
    ~1 GPU-saat kazanılıyor.
  - Bedeli: Kaggle'da notebook içi OOF kontrolü görünmüyor. Bu yüzden her push'tan önce tarifin yerel uçtan uca replay'i
    koşulur. Kaggle 15 Ekim'de sklearn 1.6.1 / numpy 2.1.3 kullanıyordu, yerelde 1.9.1 / 2.5.4 var; kod ikisinde de
    geçiyor.
  - Yerel testler: denetim modu 3 satırlık testte; notebook hücresi sahte Kaggle girdileriyle iki dalda (commit:
    girdiler bulundu, GPU yok diye durdu; rerun: tam replay başladı).

### Featherless ek veri iş kolu (16 Ekim'de açıldı; dal `featherless-extra-data`)

**Kurallar (kullanıcıdan):**
- Bütçe 8 $; her çağrı `data/external/featherless/costs.jsonl`'a yazılır. 8 $'a yaklaşınca durulur ve sorulur.
- En fazla 2 eşzamanlı istek. API anahtarı hiçbir yerde görünmez.
- Üretilen veri `data/external/`'da kalır, git'e girmez ve satır başına kaynak bilgisi taşır.
- Dış veri kuralı netleşene kadar bu veriyle eğitilmiş model final submission'a konmaz; pipeline'da bayrakla açılır,
  varsayılan kapalıdır.

**Adımlar ve durum:**

| adım | durum | not |
|---|---|---|
| 1. Hangi etiketler üretilebilir | bitti | glm-5.3, glm-5.3-flash, kimi-k3 birebir var. qwen3.8-max yok: `Qwen3.8-2.4T-A95B` Max'in "temeli" olarak geçiyor, kullanıcı kararı |
| 2. Format ve prompt'lar | bitti | train metinleri tam ve tek paragraf, kırpma yok. 150 prompt Qwen3-32B ile geri kuruldu |
| 3. Uyum testi | bitti, rapor edildi | GLM'lerde tam etiket uyumu gerçeğe yakın ama lab düzeyinde zayıf. Kimi zayıf. Sebep üslup farkı. Toplam harcama 1.217 $ |
| 4. Ölçekleme | onay bekliyor | uyum zayıf; önce üslup talimatlı ikinci uyum testi önerildi |
| 5. CV testi | bekliyor | ek veri yalnız eğitim katlarında, ağırlıkla dengelenecek, bayrakla açılacak |

### Kaggle Benchmarks model proxy (16 Ekim: yalnız kontrol, üretim yok)

- **CLI ve kota:**
  - Kaggle CLI 2.2.4 (`pip install -U kaggle` bunu değiştirmedi; PyPI'deki son sürüm bu). `kaggle benchmarks quota`
    komutu bu sürümde yok.
  - Kota SDK'daki salt-okunur uç noktadan doğrulandı (`/api/v1/models/proxy/quota`, `get_model_proxy_quotas`):
    günlük 10 $ (kullanılan 0), aylık 100 $ (kullanılan 0.013 $; yenilenme 5 Kasım).
- **Katalog** (`kaggle benchmarks tasks models`, 43 model): 18 etiketten 12'si birebir var.

| etiket | Kaggle slug | liste fiyatı $/M giriş/çıkış (kaynak) |
|---|---|---|
| gpt-5.5 | `gpt-5.5-2026-04-23` (GPT-5.5'in tarihli sürümü) | 5 / 30 (üçüncü taraf) |
| gpt-5.6-luna | `gpt-5.6-luna` | 0.20 / 1.20 (üçüncü taraf) |
| gpt-5.6-sol | `gpt-5.6-sol` | 4 / 20 kampanya, liste 5 / 30 (üçüncü taraf) |
| gpt-5.6-terra | `gpt-5.6-terra` | 2 / 12 (üçüncü taraf) |
| gpt-6-astra | `gpt-6-astra` | 10 / 50 (üçüncü taraf) |
| gemini-3.1-pro-preview | `gemini-3.1-pro-preview` | 2 / 12 (üçüncü taraf) |
| gemini-3.6-flash | `gemini-3.6-flash` | bulunamadı |
| gemini-3.8-flash | `gemini-3.8-flash` | 0.75 / 3.75 (tek kaynak) |
| claude-opus-4-6 | `claude-opus-4-6-default` | 5 / 25 (Anthropic) |
| claude-opus-4-8 | `claude-opus-4-8-default` | 5 / 25 (Anthropic) |
| claude-opus-5 | `claude-opus-5-default` | 5 / 25 (Anthropic) |
| claude-sonnet-5 | `claude-sonnet-5-default` | 2 / 10 (Anthropic) |

- **Katalogda olmayanlar:**
  - gemini-3.7-flash, claude-fable-5 ve qwen3.8-max.
  - glm-5.3, glm-5.3-flash ve kimi-k3 de yok; bu üçü Featherless'ta var.
  - Aynı ailenin başka sürümleri (glm-5, qwen3-235b, gpt-6-sol, claude-opus-4-7, gemini-3.5-flash-lite vb.) sayılmadı.
- **Maliyet:** katalogda fiyat alanı yok. Proxy her çağrının maliyetini `usage.cost` (nanodolar) içinde döndürüyor;
  gerçek fiyat ilk küçük koşuda ölçülür.
- **Plan (onay bekliyor; hiçbir şey çalıştırılmadı):**
  1. **Prompt dataset'i:** private bir Kaggle dataset'i.
     - Uyum testi için 30 geri kurulmuş train prompt'u ve hedef kelime sayıları. Yarışmadan türediği için yalnız private.
     - Ölçekleme gerekirse train prompt'larını kopyalamayan yeni prompt'lar.
  2. **Task kodu:** `kaggle_benchmarks` ile tek bir task.
     - Her satır için `llm.prompt(prompt + uzunluk talimatı)` çağrılıyor.
     - Metin, token ve maliyet run dosyasına yazılıyor.
     - `evaluate(..., n_jobs=2)` kullanılıyor.
  3. **Çalıştırma:** yalnız Kaggle altyapısında; yerel proxy token'ı ile toplu üretim yok.
     - `kaggle benchmarks tasks push nllm-gen -f kaggle/bench_gen_task.py -d <private prompts>`
     - `kaggle benchmarks tasks run nllm-gen -m <slug> --wait`: model model, her model arasında kota kontrolü.
     - `kaggle benchmarks tasks download nllm-gen -o data/external/kaggle_bench/`
     - Task asla `publish` edilmez.
  4. **Bütçe:**
     - Tahmin: ortalama uzunlukta (115 kelime ≈ 155 token, giriş ≈ 120 token) cevap başına 0.0002 $ (gpt-5.6-luna) ile
       0.009 $ (gpt-6-astra) arası. Varsayılan düşünme ~500 token eklerse 0.0008–0.034 $.
     - Fiyatı bilinen 11 modelin her birinden bir cevap: düşünmesiz 0.038 $, düşünmeli 0.145 $. Günlük 10 $ ile
       ≈260 ya da ≈70 prompt (≈2.900 ya da ≈760 cevap).
     - Uyum testi: 30 prompt × 12 model ≈ 360 cevap. Tahmin 1.2–4.4 $, tavan 5 $.
     - Model başına cevap maliyeti tahminin 2 katını geçerse durulur.
     - Günlük kota 10 $ zaten sert sınır.
  5. **Değerlendirme:** Featherless adım 3 ile aynı.
     - Fold-safe sınıflandırma (`fx_predict.py`) ve stil karşılaştırması (`fx_style.py`).
     - Ölçekleme yalnız üretilen metinler gerçek metinlere yakın oranda tanınırsa yapılır.
  6. **Veri kuralları:** çıktılar `data/external/` altında kalır ve her satır kaynak bilgisi taşır. Dış veri kuralı
     netleşene kadar bu veriyle eğitilen model final submission'a girmez.

### Uyum testi v2 (kullanıcı onayı 16 Ekim akşamı; kurallar üretimden ÖNCE yazıldı)

- **Kapsam:** 15 etiket.
  - Kaggle Benchmarks (12 etiket): üretim Kaggle altyapısında private bir task olarak; yerel token ile toplu üretim yok.
  - Featherless (3 etiket): glm-5.3, glm-5.3-flash, kimi-k3.
- **Tavanlar:** Kaggle 5 $, Featherless 1 $.
- **Prompt'lar:** 30 geri kurulmuş train prompt'u, fold başına 6 (`src/ft2_rows.py`, tohum 20261017).
  - Yalnız step-3 T0 çıktıları üç Featherless etiketinde de kullanılabilir olan 38 prompt arasından seçildi.
  - Fold başına 2'si DEV (10 prompt): şablon burada seçilir.
  - 4'ü EVAL (20 prompt): eşik yalnız burada, seçilen şablonla sınanır.
  - Bu prompt'lar ve onlardan üretilen metinler yarışma verisinden türüyor: hiçbir yere public konmaz.
    Kaggle dataset'i ve task private.
- **Şablonlar** (noktalama ya da kelime alışkanlığı dayatmıyor):
  - T0: "Answer in a single paragraph of about N words." (v1 ile aynı)
  - T1: edebi üslup notu (canlı düzyazı, somut imge, değişken cümle ritmi, düz metin)
  - T2: denemeci sistem prompt'u + "Keep it to one paragraph of about N words."
  - N, aynı prompt ve etiketin train metninin kelime sayısı.
- **Ayarlar:** temperature 1.0 (model kabul ediyorsa), düşünme `low` (kabul edilmezse modelin varsayılanı).
- **Ölçüm:** her üretilen metni, prompt'unu görmemiş fold modeli tahmin eder.
  - Model: 3 CPU üyesinin blend'i (ng_fw + ng_fwsh + style2; `fx_predict.py`).
  - Ana ölçüt, etiket ve şablon başına: r = ort. P(kendi etiketi | üretilen) / ort. P(kendi etiketi | aynı prompt'ların
    gerçek metni).
  - Ek olarak raporlanır: kendi etiketi ve kendi lab'ı doğruluğu (üretilen ve gerçek).
- **Şablon seçimi (yalnız DEV):** 15 etiketin r medyanı en yüksek olan şablon.
  - 0.05'ten küçük farkta T0 seçilir (en basit olan).
  - Bütün etiketlere tek şablon.
- **Eşik (yalnız EVAL):** etiket r ≥ 0.7 ise geçer. Geçmeyen etiket ölçeklenmez.
- **Uygulamada çıkan sapmalar (16 Ekim gece, maliyet yüzünden):**
  - Kaggle'daki ilk koşular tahminden pahalı çıktı; Gemini modelleri "low" ayarında bile cevap başına 600–2.000 token
    düşünüyor.
  - 12 Kaggle modelinin hepsiyle DEV ve EVAL ~5.9 $ tutuyordu (tavan 5 $).
  - Bu yüzden en pahalı iki model (gpt-5.5, gpt-6-astra) DEV'e girmedi. Şablon 13 etiketin r medyanıyla seçiliyor; bu
    ikisi yalnız EVAL'de seçilen şablonla sınanıyor.
  - Proxy her çağrıdan önce modelin en uzun çıktısına göre maliyet ayırıyor ve kota aşılınca 403 veriyor.
    claude-opus-4-6 ve gpt-5.6-sol böyle başarısız oldu.
  - Task'ın 2. sürümü çıktıyı 8.000 token ile sınırlıyor. Normal kullanım 200–2.000 token olduğundan metinleri
    etkilemiyor.
- **Gerçekleşen harcama (16 Ekim gece, EVAL sonrası):**
  - Kaggle: proxy kotasına göre 4,933 $ (tavan 5 $). 12 Kaggle etiketinin hepsi EVAL'de 20/20 metinle koştu.
    - Task'ın kaydettiği çağrı maliyetleri toplamı, ilk sürümlerde kotadan 0,72 $ az. Fark büyük olasılıkla 1. sürümün
      403'le biten koşularından ve task oluşturma koşularından geliyor.
    - 3. sürümden itibaren kota artışı kayıtlı maliyetle birebir aynı. Ölçekleme tahminleri bu yüzden kayıtlı maliyetle
      yapılabilir.
    - gemini-3.1-pro EVAL'de DEV'den pahalı çıktı: 0,62 $; cevap başına ortalama 2.575 çıktı token'ı.
    - Kalan bütçeye sığması için sıra: önce CV recall'u en düşük olan gpt-5.5, sonra gpt-6-astra, en son gemini-3.1-pro.
  - Featherless: 1,0105 $ (tavan 1 $).
    - 0,0105 $'lık aşım, aynı anda giden iki isteğin bütçe kontrolünü birlikte geçmesinden. Kontrol düzeltildi:
      uçuştaki istekler en kötü maliyetleriyle sayılıyor (commit 7650a59).
    - kimi-k3 EVAL'de tavana takıldı: 20 metnin 5'i üretildi. Kimi uzun düşünüyor; 5 metin için 9 çağrı, çağrı başına
      ~0,027 $. Kimi'nin EVAL r'si 5 metne dayanıyor.
- **CV kontrolü:**
  - Seçilen şablonla üretilen metinler (30 prompt) eğitim fold'larına eklenir. Prompt'u doğrulamadaki satırlar o
    fold'dan çıkarılır.
  - Ağırlıklar 0.3 ve 1.0; cpu2 üyeleri ve iç içe blend (`ext_blend.py`).
  - BA, log loss ve etiket bazında recall raporlanır. Ölçekleme kararını kullanıcı CV'ye göre verir.
- **Kullanıcı kararı (17 Ekim):** ölçekleme yok (seçenek A; ~2.000 satırlık pilot önerisi alınmadı).
  - CV'de anlamlı kazanç yoktu; sonuçlar EXPERIMENTS.md'de.
  - `--ext ft2` bayrağı kapalı kalıyor, kod silinmiyor.
  - Private Kaggle dataset'i ve task'ı private olarak duruyor.

## 6. Çalışma kuralları (kullanıcıdan)

- Kararlar CV'ye göre; her deney `EXPERIMENTS.md`'de (no, değişiklik, CV BA, CV log loss, varsa public LB). İşe yaramayanlar da yazılır.
- Gönderim (16 Ekim'de değişti): kullanıcı gönderimleri benim yapıp skoru okumamı istedi. Not defterini push'lar,
  `kaggle competitions submit -k hasancmert/<slug> -v <sürüm> -f submission.parquet` ile gönderir, beklenen CV'yi,
  gerekçeyi ve gelen public LB'yi bildiririm; LB `EXPERIMENTS.md`'ye yazılır. Günlük hak 5 (Kaggle her gönderimde
  kalanı yazıyor), final 2. Public LB yalnız kontrol amaçlı; karar yine CV'ye göre.
- Gönderim serbestliği (16 Ekim akşamı): günde 1–5 gönderimi kendi kararımla yapabilirim. Sorun olduğunda kullanıcıya
  sorarım.
- GPU: S6E10 ile ortak haftalık 30 saat. Bu hafta öncelik S6E10'da. Kota yetmeyecekse LoRA ve gömme işleri haftaya
  kaydırılmadan önce kullanıcıya söylenir.
- GPU (19 Ekim akşamı, kullanıcı kararı): bu hafta artık GPU yok; kota 24 Ekim Cumartesi sıfırlanınca kullanılabilir. O zamana kadar yalnız CPU işleri: dış veri pasajlarının LM öznitelikleri için GPU işinin ve onları kullanan eğitimin kodu, LoRA hazırlığı. Plan onaylandı.
- GPU planı değişikliği (kullanıcı onayı, 21 Ekim, "Onaylıyorum"): yalnız resmi veriyle 8B LoRA24 yerine iki aşamalı LoRA (1. aşama public pasaj havuzu, CC BY-SA 4.0). 24–30 Ekim: Qwen3-4B (sert sınır 3.75 sa) + `nllm-gpu-embed-ext` + v8 (v7g + LoRA üyesi). 8B iki aşamalı LoRA 31 Ekim haftasına. Ayrıntı: "Public çözümler incelemesi" bölümü.
- GPU bütçesi, 24–30 Ekim haftası (kullanıcı kararı, 20 Ekim): S6E10 ~23 saat kullanacak; NLLM'e ~7 GPU-saat kalıyor. Bu bütçe içinde serbestim. Plan:
  1. Dış veri pasajlarının LM öznitelikleri (`nllm-gpu-embed-ext`, ~0.7 sa) → CPU'da dış veriyle LM üyeleri, uzmanlar, ikizler (CV).
  2. LoRA24 8B, 5 fold + final (`nllm-lora24`), sert süre sınırı 5 sa (`LORA24_BUDGET_H`). Tahmin ~4.5 sa: probe2'nin ölçümünden ölçeklendi (4B, 27 katman, 4 epoch: fold başına ~1 sa).
  3. CV kazancı varsa birleşik gönderim (v8, ~1.3 sa). Toplam ~6.5 sa.
  - S6E10 iki GPU slotunu doldurursa NLLM işleri slot boşalınca push'lanır.
  - Zamanlanmış görev (`trig_01QPMvJH5scmKNDHHNvE1KQz`, 24 Ekim 00:30 UTC): kota sıfırlanınca bu planı başlatır (önce kernel listesiyle S6E10 slotları kontrol edilir).
    21 Ekim: istemi yeni plana göre güncellendi; kullanıcı planı aynı gün onayladı (iki aşamalı 4B LoRA, ext-embed, v8).
    23 Ekim 22:13 UTC: kullanıcı 2 gün ara istedi ("Önümüzdeki 2 gün bir şey yapma"). Görev 25 Ekim 22:30 UTC'ye alındı; o zaman da işe kendiliğinden başlamaz, önce kullanıcıya sorar (GPU planına devam ve GPU kiralama kararı).
    28 Ekim: kullanıcı kararlarından sonra görev 31 Ekim 00:30 UTC'ye alındı. 29 Ekim: istemine `--cheap-commit` ve Hungarian'sız v7p eklendi.
- Kullanıcı kararları (28 Ekim): bu hafta GPU kullanılmıyor, 31 Ekim'deki kota sıfırlanması bekleniyor. GPU kiralama yok;
  bütün eğitim Kaggle'da. S6E10 31 Ekim'de bitiyor; ondan sonra haftalık 30 saatin tamamı NLLM'e kalıyor.
- GPU planı, 31 Ekim haftası (~30 sa, yalnız Kaggle T4; sıra):
  1. Kernel listesiyle S6E10'un son koşularına bakılır; slotlarına dokunulmaz.
  2. v7p gönderimi (Hungarian'sız; kayıt koşusu birkaç dakika, puanlama koşusu kotadan düşmüyor):
     `python kaggle/build_submission.py --gpu --pipeline replay.py --extra "--recipe v7p" --pub-datasets --pool-dataset
     --cheap-commit --push`. COMPLETE olunca logdaki denetimler okunur, sonra yarışmaya gönderilir. LB, Gaussian'ın ve
     havuz SVC'sinin gizli testteki etkisini gösterir.
  3. İki aşamalı 4B LoRA, 1. aşama 120 binlik havuzla (29 Ekim güncellemesi, aşağıda):
     `--slug nllm-lora2s5 --two-stage --pool5 --hint 4b-base --pre-max-h 3 --budget-h 5.5`.
  4. İki aşamalı 8B LoRA, aynı havuzla: `--slug nllm-lora2s58 --two-stage --pool5 --hint 8b-base --micro 1
     --pre-max-h 5.5 --fold-max-h 0.6 --budget-h 9` (T4'te 8B için üçlü başına bir ileri geçiş). 3 ve 4 iki ayrı GPU
     oturumunda aynı anda koşabilir (v7p'nin kısa kayıt koşusundan sonra).
  - 29 Ekim güncellemesi: 1. aşama havuzu `prvsiyan/ntl-public-in-style-pool-v1` yerine
    `dhruvtibarewal/nlm-matched-clean` (`nlm_pool_v5`; lisans denetimi ve kurallar EXTERNAL_DATA.md'de).
    - Gerekçe: public notebook'lar aynı tarifle 38 binlik havuzda ~0.57, 120 binlikte 0.634 (14B) bildiriyor.
    - Sızıntı süzgecinden sonra 1. aşamada 73.050 satır (3.212 blok), eski havuzda 17.860.
    - Süre tahmini (probe2 hızından, belirsiz): 4B ~27 bin satır/sa (1 epoch ~2,7 sa), 8B ~yarısı (~5,4 sa).
      `--fit-cap`, ilk güncellemelerin hızını ölçüp öğrenme oranı takvimini süre sınırına sığdırıyor. Hız
      düşükse epoch'un rastgele bir kısmı görülür, takvim yine sıfıra iner.
    - Toplam ~14,5 GPU-saat (önceki plan ~10).
    - Kaggle CPU duman testleri (29 Ekim, `nllm-lora2s5-smoke` v1–v2) geçti: havuz v5, sha256, sızıntı süzgeci,
      DDP (gloo, iki süreç, ağırlık farkı 0.0), f0 1. aşamadan başladı. v2'de kısa sınır zorlandı: 34 sn/güncelleme
      ölçüldü, takvim 23'ten 11 güncellemeye indi ve 1. aşama 396 sn'lik sınırın içinde (386 sn) bitti.
    - LoRA üyeleri yalnız CV'leri artırırsa gönderime girer (kural).
- Kullanıcı kararı (29 Ekim, ~16:30 UTC): "gpuyu kullanabilirsin". Kota (`KaggleApi().quota_view()`): bu hafta 30
  saatin 3,0 saati kullanılmış, 27 saat kalmış, sıfırlanma 31 Ekim 00:00 UTC. Kullanılmayan saat sıfırlanınca
  yanacağı için plan 29 Ekim'e çekildi:
  - 16:39 `nllm-submit-gpu` v6 (v7p, ucuz commit). Kayıt koşusu 36 sn: denetimlerin hepsi ok. Havuz sha'sı tuttu,
    4 LM ve train LM öznitelikleri bulundu, 2 × T4. Paket sürümleri 15 Ekim'deki ortamla aynı (sklearn 1.6.1,
    numpy 2.1.3). 16:41'de gönderildi (ref 57777088).
  - 16:39 `nllm-lora2s58` (8B, havuz v5, sınır 9 sa) ve 16:41 `nllm-lora2s5` (4B, havuz v5, sınır 5,5 sa) push'landı.
  - 31 Ekim görevi buna göre değişti: bu işleri yeniden başlatmıyor; sonuçları topluyor, üyeleri ve CV'yi kuruyor,
    yeni haftanın 30 saatini sonuçlara göre planlıyor.
  - Aynı gün CPU'da 120 binlik havuzun etkisi ölçüldü (EXPERIMENTS 624–627). 4 görünümlü SVC 0.4101'den 0.4649'a,
    v7p birleştiricisi 0.5030'dan 0.5405'e çıktı. Tarif `v7q` = v7p + havuz v5'li SVC.
    - Kaggle aynı anda en fazla 2 toplu GPU oturumu açıyor; ikisi LoRA'larda.
    - Arka plandaki bir betik (`scratchpad/push_v7q_when_free.sh`) 4B LoRA bitince v7q'yu ucuz commit'le push'luyor,
      kayıt koşusunun denetimlerini kontrol edip gönderiyor (log: `scratchpad/push_v7q.log`).
    - Yeniden koşu tahmini ~4,8 sa, sınır 9 sa; SVC ayrı aşamada tek başına koşuyor (~14 GB).
  - Akşam (22:30 UTC):
    - v7p public LB **0.516** (v7 0.503).
    - Konteyner iki kez yeniden başladı, bekleme betiği öldü. v7q elle push'landı (sürüm 7, kayıt denetimleri ok) ve
      gönderildi (ref 57784052).
    - 4B LoRA bitti (4,1 sa). Üye OOF 0.5568; v7q birleştiricisine eklenince 0.5405 → **0.5875**.
    - v8 (= v7q + 4B LoRA) tam kayıt koşusuyla push'landı (sürüm 8, 22:35 UTC, ~5 sa). Amaç LoRA çıkarım yolunu
      Kaggle'da doğrulamak ve notebook içi OOF'u görmek. Bitince gönderilecek.
    - 8B LoRA ~01:40 UTC'de bitecek; üyesi CV'de ölçülüp v8'e eklenirse v8b.
    - Kota 22:37 UTC'de 13,1 / 30 sa.
  5. Çıktılar ağırlıksız indirilir (`--fetch`), LoRA üyeleri ve ikizleri CPU'da (`lora24_member.py`), CV ölçülür;
     replay'e LoRA çıkarım adımı eklenir, yerelde test edilir; CV artarsa v8 gönderilir.
  6. Kota kalırsa `nllm-gpu-embed-ext` (~0.7 sa) ve 14B / ikinci tohum.
  - Gerçek GPU saatleri bu bölüme işlenir.
- 24 Ekim sonrası GPU planı (hazır kod):
  1. `nllm-gpu-embed-ext` (hazır, push'lanmadı): 7.700 public pasajın LM öznitelikleri. 0.6B / 1.7B 4 katman, 4B / 8B
     12 katman. Tahmin ~0.6–0.8 GPU-saat. Sonra CPU'da: dış veriyle LM üyeleri (`train_emb --ext pub`), uzmanlar
     (`family_spec --ext pub`, primal), dış veri bloklarıyla ikizler (`adaptive_sim --ext pub`, TwinPrimal).
  2. LoRA24 (`kaggle/build_gpu_lora24.py`, `src/gpu_lora24.py`): public iki başlı 24 katmanlı tarifin yeniden yazımı
     (Qwen3-8B ilk 24 katman, LoRA r16, 8 havuz katmanı + LayerNorm, düz + prompt merkezli baş, aynı prompt'tan
     üçlüler, 3 epoch). 5 fold + final, iki T4'te 3'er iş. Public notebook'un üst sınırı ~10 sa; probe2'den ölçeklenen tahmin ~1.4 sa/iş, toplam **~4.5 GPU-saat** (5 sa sert sınırla).
     - CPU duman testleri geçti (19 Ekim): eğitim (0.6B, 12 katman, 6 prompt, 1 epoch; çıktıların hepsi) ve çıkarım
       (`lora24_infer.py`; çapa eşleşmesi birebir). GPU'nun fp16 + ölçekleyici yolu yalnız gerçek koşuda denenecek.
     - Replay için: `lora24_infer.py` (test öznitelikleri), `lora24_member.py` (OOF üyesi ve merkezli ikiz kanıtı).
  3. Alternatif (kullanıcı kararı gerekir): lightningv08'in public LoRA24 checkpoint'i (eğitim maliyeti yok; başka bir
     katılımcının modeli, OOF'u onların fold'larında; lisans kontrolü gerekir).
- Dış veri yalnız ücretsiz ve herkese açık, yalnız eğitimde; `EXTERNAL_DATA.md`'ye kaynak + lisans. Validation her zaman yarışma katları.
- Para harcayan, bir şeyi public yapan ya da büyük yön değiştiren her adımdan önce sorulur. Kaggle dataset ve not defterleri private.
- Kullanılan kod/model lisansları ticari kullanımı kısıtlamamalı; `MODELS_LICENSES.md`'ye yazılır.
- Yalnız gerçekten çalıştırılıp ölçülen sonuçlar raporlanır.
- Kaggle Benchmarks model proxy'si (18 Ekim): Kaggle'ın ücretsiz AI kotası, para harcanmıyor.
  - Günlük limit (10 $ birim) içinde serbestçe kullanılabilir.
  - Aylık kullanım limitin yarısına (50 $) ulaşınca kullanıcıya sorulur.
  - Üretim Kaggle altyapısında private task olarak yapılır; yerel proxy token'ıyla toplu üretim yok.
  - Yarışma prompt'ları kopyalanmaz; yalnız tür ve uzunluk taklit edilir.

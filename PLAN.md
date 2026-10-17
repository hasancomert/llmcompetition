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
- Dış veri yalnız ücretsiz ve herkese açık, yalnız eğitimde; `EXTERNAL_DATA.md`'ye kaynak + lisans. Validation her zaman yarışma katları.
- Para harcayan, bir şeyi public yapan ya da büyük yön değiştiren her adımdan önce sorulur. Kaggle dataset ve not defterleri private.
- Kullanılan kod/model lisansları ticari kullanımı kısıtlamamalı; `MODELS_LICENSES.md`'ye yazılır.
- Yalnız gerçekten çalıştırılıp ölçülen sonuçlar raporlanır.

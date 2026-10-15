# HANDOFF: Kaggle S6E10 (+ S6E9 dersleri) → Kaggle "Name That LLM"

**Kaynaklar.** Bu repo (S6E10): `README.md`, `experiments.md`, `HANDOFF_playground-series-s6e10.md`, `src/`, `kaggle/`, `scripts/`
ve git log. Bir önceki yarışma S6E9: yalnız `LESSONS.md`. S6E9'un kodu (`train.py`, `blend_ext.py`, `gpt2tok.py`) bu repoda
yok, S6E9 sayıları LESSONS.md'den alındı. Kodda ya da günlükte karşılığı olmayan her şey için "bulunamadı" yazıldı.

**Okurken dikkat.** Buradaki bütün skorlar **ROC AUC** (iki yarışma da ikili sınıflandırmaydı, ~700 bin satırlık sentetik tablo verisi).
Yeni yarışmanın metriği balanced accuracy, verisi 2.699 metin. AUC farkları balanced accuracy farklarına çevrilemez; buradaki sayılar
"neyin ne kadar işe yaradığının" kaydı, yeni yarışma için beklenti değil. Yeni yarışmayla ilgili bilgiler yalnız kullanıcının özetinden.

**Etiketler.** **[TAŞINIR]** olduğu gibi kullanılabilir. **[UYARLANMALI]** fikir taşınır, ayrıntı değişmeli. **[TAŞINMAZ]** önceki
yarışmaya özgüydü ya da yeni görevde karşılığı yok. Her etiketin yanında tek cümlelik neden var.

**Kod.** `reuse/` altında 8 bağımsız modül (+ test çalıştırıcısı), N×18 olasılık matrisi ve balanced accuracy için uyarlandı, her biri rastgele sahte veriyle
test edildi: `python reuse/run_selftests.py` (bölüm 9).

---

## 1. Validation kurulumu ve CV / public / private uyumu

### 1.1 Ne yapıldı (S6E10)

- **Tek bölme, her modelde aynı katlar.** `StratifiedKFold(K, shuffle=True, random_state=42)` (`src/common.py: folds`); `seed=` yalnız
  model tohumunu değiştirir. K=5 (s01–s04), sonra 10; TabPFN üyesi 15 kat (abdullahsafwan333'ün 15 katlı üyeleriyle aynı bölme olsun diye).
  **[UYARLANMALI]** "her modelde aynı sabit bölme" aynen kalmalı, çünkü harman ve istif OOF'ları buna dayanıyor; ama bölme
  `StratifiedGroupKFold(prompt_id)` olmalı, çünkü test prompt'ları train'de yok ve satır bazlı bölme aynı prompt'un cevaplarını katın iki
  tarafına koyar (`reuse/oof_io.make_folds`, grup çakışmasını assert ile denetler).
- **Hedeften türeyen her şey kat içinde.** sklearn `TargetEncoder(cv=5)` her dış katın yalnız eğitim satırlarında fit edildi, doğrulama
  ve test o katın kodlayıcısıyla dönüştü (`src/te_cache.py`). **[TAŞINIR]** ilke veri türünden bağımsız: istifçi, artık modelin taban
  logit'i, sınıf önselleri, sınıf başına kaydırma da dış katın eğitim satırlarında öğrenilmeli.
- **Erken durdurma doğrulama katında.** pytabkit en iyi epoch'u puanlanan katta seçiyor; ölçülen yanlılık: son epoch 0.96126, en iyi
  epoch 0.96127 / 0.96128 (~0.00001, s11 kontrolü). `nn.py`'deki aynı alışkanlığın etkisi ölçülmedi. **[UYARLANMALI]** 700 bin satırda
  yanlılık 0.00001'di; yeni yarışmada bir doğrulama katı ~540 metin (2.699 / 5) ve bu yanlılık ölçülmedi, iç bölme ya da sabit tur
  sayısıyla bir kez karşılaştırılmalı.
- **Gürültü tabanı.** Tek model: aynı özellik seti, 3 model tohumu → 0.95986 / 0.95993 / 0.95997 (std ≈ 0.00005); 0.0001 altı farklar
  gürültü sayıldı. İstif: aynı istif, meta-kat tohumu 7 / 1 / 2 / 3 → 0.962247 / 0.962241 / 0.962244 / 0.962256 (aralık 0.000015);
  aynı katlarda iki istifin **eşli** farkının gürültüsü ≈ ±0.000004. Karar kuralı (`experiments.md`): eşli fark birkaç meta tohumda tutarlı
  biçimde ≥ +0.00001 ya da 4/4 pozitif değilse gürültü. **[UYARLANMALI]** yöntem (aynı katlarda eşli fark, birden çok bölme tohumu) taşınır,
  eşik sayıları taşınmaz: ~150 metin/sınıfta (2.699 / 18) tek bir metin o sınıfın recall'unu ~0.0067, balanced accuracy'yi ~0.00037
  değiştirir (`reuse/stacker.paired_compare`).
- **İç içe ölçüm.** Her harman/istif ağırlığı OOF'un 4/5'inde öğrenildi, kalan 1/5'te puanlandı. 700 bin satırda OOF içi skor ile iç içe
  skor aynıydı (s03: 0.96159 / 0.96160). **[TAŞINIR]** ağırlık, C, kaydırma gibi her seçim iç içe ölçülmeli; küçük veride seçim yanlılığı
  bu projede ölçülmedi, yok sayılmamalı (`reuse/hill_climb.nested_hill`, `reuse/stacker.nested`).
- **Düşmanca doğrulama** (`src/adversarial.py`): train/test AUC 0.500, train/orijinal veri 0.834 → orijinal satırlara `is_orig` bayrağı.
  **[UYARLANMALI]** train ve test prompt kümeleri ayrık olduğu için ayrılabilirlik metin özelliklerinde (gömme, stil istatistikleri)
  ölçülmeli ve train satırları prompt'a göre gruplanmalı, yoksa sınıflandırıcı prompt'un konusunu öğrenir (`reuse/adversarial.py`, `groups=`).
- **Dış OOF dosyalarının denetimi** (`src/ext_pub.py`): id hizalama, şekil / sonluluk, OOF AUC'si (0.90, 0.9625) dışındakini reddetme
  (çok iyi = sızıntı şüphesi), her kaynağın kodunu okuma, pickle yüklememe. **[UYARLANMALI]** yeni yarışmada paylaşılan OOF olup olmayacağı
  bilinmiyor; aynı denetimler (olasılık mı logit mi, şekil, makul skor penceresi) kendi üyelerimiz için de geçerli
  (`reuse/oof_io.validate_member`).

### 1.2 S6E10: iç içe CV ile public LB (22 gönderim: s01–s21 + s03-alt)

| Ölçü | Değer |
|---|---|
| Spearman / Pearson (iç içe CV, public) | 0.998 / 0.988 |
| public − iç içe CV, yalnız kendi modellerimiz (s01–s07 + s03-alt, 8 gönderim) | −0.00067 … −0.00054 |
| public − iç içe CV, açık OOF'lu istifler (s08–s21, 14 gönderim) | −0.00048 … −0.00040 |
| yalnız s08–s21 içinde Spearman / Pearson | 0.993 / 0.979 |
| büyük adımlar | s01→s02 iç içe +0.00033 / public +0.00028; s07→s08 +0.00029 / +0.00046; s08→s21 +0.00016 / +0.00022 |
| küçük adımlar | ≤ +0.00001'lik iç içe adımlar public'te 0 ya da +0.00001 göründü (s09→10, s12→13, s13→14, s17→18, s19→20, s20→21) |
| public sıralaması | aynı public skorla s13→s14 17/968 → 90/1086, s19→s20 9/1487 → 76/1694; s21 19/1696 |
| private | **bulunamadı** (yarışma 31.10.2026'da bitiyor) |

### 1.3 S6E9: CV / public / private (LESSONS.md §1, 19 gönderim)

- Private, public'in sabit bir kaymayla altında: 2–19. gönderimlerde −0.00092 … −0.00106; CV'nin de ~0.001 altında.
- Pearson CV–private 0.9975, public–private 0.9997; Spearman CV–private 0.9854, public–private 0.9845.
- En iyi private (0.94570) en iyi CV ve public'li gönderimlerde (18, 19).
- Büyük adımlar private'a taşındı (10→11 GPT-2 parçaları: CV +0.00019 / public +0.00023 / private +0.00020; 2→19: +0.00098 / +0.00089 /
  +0.00075); küçükler görünmedi (4→5: +0.00005 / 0 / −0.00001; 7→9: +0.00003 / +0.00001 / −0.00001; 12→15: +0.00002 / 0 / 0).
- 5 → 10 kat her modelde ~+0.0001; 10 → 20 kat XGBoost, GLR ve artık modellerde +0.00001 … +0.00004, LightGBM'de 0.

### 1.4 Yeni yarışmaya

- **[TAŞINIR]** Kararı CV verir, public vermez: iki yarışmada da sıralama korundu ve küçük adımlar public'te görünmedi; yeni yarışmada
  public ve private'ın model karışımı farklı olduğu için public tek başına daha da zayıf bir sinyal.
- **[TAŞINMAZ]** Kayma sayıları (public ≈ CV −0.0004 … −0.0007; private ≈ public −0.001): train ve testin aynı dağılımdan geldiği
  (düşmanca AUC 0.500) sentetik tabloda AUC ile ölçüldü; görülmemiş prompt'lar ve farklı sınıf karışımıyla bir karşılığı yok.
- **[TAŞINIR]** Sıralamaya bakıp yön değiştirmemek: aynı public skorla sıra 17 → 90 ve 9 → 76 oynadı; bir günde 46 takım aynı skordaydı
  (kopya not defterleri).
- **[UYARLANMALI]** 5 → 10 kat kazancı (model başına +0.0001–0.0002, harman 0.96165 → 0.96181): kazanç modelin daha çok veri
  görmesinden geldi; 150 prompt'ta kat sayısı doğrulama katındaki prompt sayısını da belirler ve bu etki ölçülmedi.

---

## 2. Deney tablosu

### 2.1 S6E10: işe yarayanlar (AUC)

| Değişiklik | Ölçüm | Yeni yarışma |
|---|---|---|
| `te1`: her kolonun kat içi hedef kodlaması (değer kimliği) | LightGBM lr 0.1, 5 kat: 0.95879 → 0.95997 (+0.00118) | **[TAŞINMAZ]** sentetik tablonun değer kimliği artefaktıydı, metinde karşılığı yok. |
| LightGBM varsayılan (100 ağaç) → 63 yaprak + erken durdurma | 0.95780 → 0.95879 (+0.00099) | **[UYARLANMALI]** stil özellikli GBDT'de de kütüphane varsayılanı referans alınmamalı; erken durdurma yanlılığı küçük katta ayrıca ölçülmeli. |
| Optuna (`t1`, 2 katta 30 deneme) + lr 0.1 → 0.03 | LightGBM FS3 0.96089 → 0.96130 (+0.00041) | **[UYARLANMALI]** ayar 700 bin satırda 2 kat üzerinde yapıldı; 2.699 satırda ayar aramasının CV'ye uyumu ölçülmedi, sonucu iç içe doğrulanmalı. |
| Orijinal veriyi her eğitim katına ek satır (`is_orig`) | FS1 adımı +0.00031; RealMLP 0.96110 → 0.96127 | **[TAŞINMAZ]** yeni yarışmada dış veri için kullanıcı kuralı geçerli: yalnız kurallar açıkça izin verirse, emin değilse sorulacak. |
| `te2` (kolon ikilileri) / `tefd` (mesafe × kolon) | +0.00030 / +0.00021 | **[TAŞINMAZ]** tablo kolonlarına özgü kodlamalar. |
| Farklı model ailelerini harmanlamak | tek en iyi 0.96130 → istif 0.96160 (5 kat) | **[TAŞINIR]** kazanç model çeşitliliğinden geldi, veri türünden değil; gömme sınıflandırıcısı, stil GBDT'si, tokenizer özellikli modeller aynı katlarda OOF üretmeli. |
| Tohum ortalaması, MLP 1 → 7 koşu | 0.95907 → 0.96017 (+0.0011); GBDT'de 2. tohum +0.00001–0.00003 | **[UYARLANMALI]** tohum etkisi sinir ağlarında büyük, GBDT'de küçüktü; ince ayarlı LLM üyelerinde tohum varyansı ayrıca ölçülmeli. |
| TabPFN: 3 tohumun logit ortalaması ile 1 tohum (15 katın 0. katı, aynı ayar) | 0.962293 (bizim, 1 tohum) / 0.962675 (abdullahsafwan333, 3 tohum): +0.00038 | **[UYARLANMALI]** büyük önceden eğitilmiş modellerde tohum/ensemble etkisi büyüktü; aynı katta bir kez ölçmek yeter. |
| Kendi + açık OOF üyeleri, L2 lojistik istif (C=0.001) | iç içe 0.96184 → 0.96213; public 0.96119 → 0.96165 | **[UYARLANMALI]** istif yöntemi taşınır (bölüm 5); aynı katlarla OOF paylaşan bir ekosistem yeni yarışmada var mı bilinmiyor. |
| Tablo temel modeli OOF'ları (cdeotte TFM seti) | s14 → s15: 0.962170 → 0.962200 | **[TAŞINMAZ]** tablo modelleri. |
| TabM (3 tohum) | s06 → s07 harman 0.96181 → 0.96184, en büyük ağırlık | **[TAŞINMAZ]** tablo ağı. |
| Kendi 15 katlı TabPFN-3.5, bağlamda orijinal satırlar | OOF 0.961422; istif eşli +0.0000088 (4/4 meta tohum) → s18 | **[TAŞINMAZ]** model tabloya özgü; bellekle ilgili ders bölüm 8'de. |
| Açık havuz yenilemeleri | 09.10 +0.00004, 10.10 +0.00001, 11.10 +0.000012, 13.10 +0.000010 | **[UYARLANMALI]** düzenli tarama + kodu okuma + kopya testi rutini taşınır; yeni yarışmada paylaşım olup olmayacağı bilinmiyor. |
| İstifte aile bazlı L2 (TabPFN ×3, diğer TFM ×1.5) | eşli +0.0000085, 4/4 meta tohum → s21 | **[UYARLANMALI]** ailelere göre ceza (ör. LLM gömme modelleri / stil modelleri) taşınır; katsayılar yayınlanmış değerlerdi, bizim veride ayarlanmadı. |

### 2.2 S6E10: işe yaramayanlar

| Deneme | Ölçüm | Yeni yarışma |
|---|---|---|
| Elle özetler: gecikme toplamları, puan ortalaması/min/max, grup ortalamaları, 0 bayrakları, puanların kategorik kopyası | 0.95843–0.95880 (referans 0.95879) | **[TAŞINMAZ]** ağaçlar ham kolonları zaten görüyordu; metinde ham girdi yok, stil istatistikleri girdinin kendisi. |
| Orijinal veride değer başına oran (`omean`), `ofd`, FS2 üstüne `opred` | `te1` varken 0 (0.96049, 0.96047, 0.96082) | **[TAŞINMAZ]** dış veri ve tablo kodlamasına özgü. |
| Orijinal satır ağırlığı 0.5 / 2 | 0.96074 → 0.96063 / 0.96065 | **[TAŞINMAZ]** dış veri ağırlığı. |
| `te3` üçlü kodlamalar | 0.96098 → 0.96105 (gürültü) | **[TAŞINMAZ]** tablo kodlaması. |
| Basamaklar / rota ortalamasından sapma / etiketsiz puan beklentisi (hedef kodlamasız LightGBM, 2 kat) | 0.96041 / 0.96015 / 0.96035 (temel 0.96040) | **[TAŞINMAZ]** tablo özellikleri. |
| TE girdili MLP | tek başına 0.9591 → 0.9598, harmanda ağırlık 0.003 | **[TAŞINIR]** tek başına güçlü ama mevcut üyelere benzeyen model harmana bir şey katmadı; üye seçimi istif katkısıyla yapılmalı. |
| RealMLP + etiketsiz beklenti kolonları (`aux`) | en iyi tek modelimiz 0.961494 (+0.000075); istif 0.962247 → 0.962245 | **[TAŞINIR]** aynı ders: en iyi tek model ≠ istife katkı. |
| Kendi TabPFN (5 kat, 22 sütun) | 0.96120; açık TabPFN ile korelasyon 0.998, istife ≈0 | **[TAŞINIR]** havuzda zaten olan bir modelin benzerini üretmek Kaggle 2×T4'te 4,4 saat sürdü ve katkı vermedi; önce korelasyona bakılmalı. |
| Büyük ağırlıklı kendi üyelerine ek tohumlar (s10) | iç içe 0.962149 → 0.962150, public aynı (0.96171) | **[UYARLANMALI]** istif düzeyinde ek tohum ölçülebilir kazanç getirmedi; tek model düzeyindeki tohum kazancından ayrı ölçülmeli. |
| Hedef kodlamasız CatBoost, LightGBM `extra_trees`, rota içi kNN (0.92249) | istife 0 | **[TAŞINMAZ]** tabloya özgü üyeler. |
| LightGBM `rank_xendcg` sıralama hedefi | 0.9601 / 0.9591 (ikili hedef 0.9613) | **[TAŞINMAZ]** AUC'ye özgü hedef denemesi. |
| Bütün kolonlar LightGBM'e kategorik | 0.95995 | **[TAŞINMAZ]** tablo. |
| Harman ağırlıkları segment başına; segment duyarlı istif (`stack_seg.py`) | 0.96184 → 0.96184; −0.000001 / −0.000010 / +0.000001 | **[UYARLANMALI]** global ağırlıklar yetti; yeni veride segment (ör. uzunluk) denenecekse yalnız iç içe ölçümle. |
| İstif ayarları: probit(sıra) girdisi, negatif olmayan ağırlıklar, LightGBM meta-öğrenici, kırpma, standardizasyon, yakınsamış çözücü, %20 LightGBM harmanı | bölüm 5'te sayılarıyla | bölüm 5 |
| Kumo-Tabular / TabFM (tablo temel modelleri, T4) | büyük model her bağlamda bellek yetersiz; küçük 0.958366 (karşılaştırma 0.958421); TabFM ~8.000 bağlamla sınırlı | **[TAŞINMAZ]** tablo modelleri; bellek dersi bölüm 8'de. |

### 2.3 S6E9: kayda değer ölçümler (LESSONS.md §2)

| Değişiklik | Ölçüm | Yeni yarışma |
|---|---|---|
| Özellik grupları sırayla (LightGBM 5 kat): base → freq → dig → te1 → bins → dig2+te3 | 0.94178 → 0.94301 → 0.94381 → 0.94543 → 0.94570 → 0.94573 | **[TAŞINMAZ]** gelir/mesafe sütunlarının sentetik üretim izleri. |
| GPT-2 BPE parçaları (`tok`) | 10 kat: model başına +0.00021–0.00026 (LightGBM 5 kat +0.00002); gönderim 10→11 private +0.00020 | bölüm 3 |
| GLR logit'inden artık model | +0.00006–0.00007 | bölüm 4 |
| Tohum ortalaması: XGBoost 5 kat 1 → 3 tohum; RealMLP 1 → 6 tohum | 0.94611–0.94613 → 0.94619; 0.94594–0.94604 → 0.94619 | **[UYARLANMALI]** iki yarışmada da sinir ağı tohumları GBDT'ninkinden fazla kazandırdı; LLM üyelerinde ayrıca ölçülmeli. |
| Açık not defterinin LightGBM ayarları bizimkinin yerine | 0.94593 → 0.94608 | **[TAŞINIR]** başkalarının yayınladığı ayarları aynı katlarda bir kez denemek ucuzdu ve S6E9'da bizimkini geçti (S6E10'da RealMLP ayarları da açık not defterinden alındı, karşılaştırma ölçülmedi). |
| Tam veri modeli (kat ortalaması yerine), 4 sahte testli benzetim | ortalama −0.00004 (üçünde −0.00005 / −0.00007, birinde +0.00003) | **[UYARLANMALI]** 9 saatlik internetsiz notebook'ta K kat modeli sığmazsa tam veri modeli seçeneği aynı benzetimle ölçülmeli; S6E9'da kat ortalaması kazandı. |
| Kat başına sızıntısız sözde etiketleme | 0.94164 → 0.94172; XGBoost 0.94625 → 0.94626 | **[UYARLANMALI]** etkisi gürültü düzeyindeydi; görülmemiş prompt'lu testte transdüktif yöntemler yalnız kat içi sızıntısız benzetimle değerlendirilmeli. |
| Orijinal veri ek satır; `te2`; `ncat`; `solo=1` | 0.94178 → 0.94177; 0.94543 → 0.94522; 0.94381 → 0.94307; 0.94576 → 0.94554 | **[TAŞINMAZ]** tabloya özgü. |
| GLR logit'ini RealMLP'ye özellik olarak vermek | 0.94636 → 0.94596 | bölüm 4 |
| İkinci seviye LightGBM istifleme | 0.94575 | bölüm 5 |
| Trende %0 / %100 olan aralıklardan kural çıkarmak (4 katta bul, 5.'de uygula) | ≥100 satırlık kurallar −0.00037, ≥300 satır −0.00004 | **[TAŞINIR]** veriden "kesin" kural madenciliği iç içe ölçülünce kaybettirdi; kurallar dışarıda bırakılan katta sınanmalı. |
| CatBoost GPU ile CPU (aynı katlar) | GPU ~0.00026 düşük; GPU varsayılanı `border_count=128` ile ~0.0008 düşük; XGBoost GPU = CPU | **[UYARLANMALI]** stil özellikleri için CatBoost GPU'da koşacaksa aynı katlarda bir kez CPU ile karşılaştırılmalı. |

---

## 3. Tokenizer tabanlı özellikler

- **S6E10 (kendi kodumuz): bulunamadı.** `src/` altında tokenizer kodu yok.
- **S6E10 (açık havuz üyesi):** denpugovkin'in "numeric keys CatBoost CTR" üyesinin (`dp_catboost_numeric_ctr`) tarifi, dört tam sayı
  alanının (Age, Flight Distance, iki gecikme) `" " + sayı` biçimindeki GPT-2 BPE parçalarından 12 sütunluk bir özellik grubu içeriyor
  ("View-B"); tokenizer dosyaları internetsiz kullanım için sabit varlık olarak taşınıyor ve SHA-256 özetleriyle sabitleniyor
  (`ext/pub/.../numeric_tokens.py`, `campaign.py`). Üye bir CatBoost CTR modeli; parçaların bu üyeye katkısı ayrıca ölçülmedi. s11 kaynak
  ablasyonunda üye çıkarılınca istif −0.000005 (gürültü ±0.000004'ün hemen üstü). Tek başına OOF AUC'si kayıtlarda yok (bulunamadı).
- **S6E9 (LESSONS.md):** `" " + str(gelir)` Hugging Face `gpt2` tokenizer'ıyla parçalandı. İlk parça, ilk iki parça ve son parça (parça
  sayısıyla birlikte) kat içi hedef kodlama anahtarı oldu; ayrıca bu anahtarların train + test + orijinaldeki log sayımı ve parça sayısı eklendi.
  Tokenizer internetli bir Kaggle CPU not defterinde çalıştı, çıktısı eşleme tablolarıydı (`gpt2tok.py` →
  `gpt2_income_tokens.csv`, `gpt2_commute_tokens.csv`; kod bu repoda yok). Fikir heuljax'ın GLR not defterinden.
  - Kazanç: LightGBM 5 kat 0.94606 → 0.94608; 10 kat XGBoost 3 tohum ortalaması 0.94631 → 0.94652, CatBoost d5 0.94622 → 0.94645,
    RealMLP tek tohum 0.9461 → 0.94636; gönderim 10 → 11: CV +0.00019, public +0.00023, private +0.00020.
  - Üstüne işe yaramayanlar: `mix` 0.94608 → 0.94597; `tokx` (parça × bağlam hücreleri) 0.94608 → 0.94591.

Yeni yarışmaya:
- **[UYARLANMALI]** S6E9'da tokenizer, sayının nasıl parçalandığını kategorik anahtara çevirmek için kullanıldı (sentetik üretimin izini
  yakalıyordu); yeni plandaki "farklı lab tokenizer'larıyla token sayıları" başka bir kullanım ve bu projelerde ölçümü yok (bulunamadı).
- **[TAŞINIR]** Tokenizer'ı internetsiz çalıştırmak: denpugovkin'in kodu varlıkları sabit dosya olarak taşıyıp özetle doğruluyordu; internetsiz
  notebook'ta her tokenizer ekli bir veri seti olarak gelmeli.
- **[TAŞINIR]** Yeni bir özellik ailesini birkaç model ailesinde ayrı ayrı ve sonra harmanda ölçmek: S6E9'da `tok` her modelde +0.0002
  civarı getirdi ve harmandaki kazanç private'a taşındı.
- `reuse/` altında tokenizer modülü yok: uyarlanacak kod bulunamadı.

---

## 4. Residual / base_margin yaklaşımı

- **S6E10: bulunamadı.** `src/` altında `base_margin` / `init_score` kullanılmadı (`stack.py gbw=` LightGBM istifçisi artık model değil).
- **S6E9 (LESSONS.md):** XGBoost `base_margin`, LightGBM `init_score`, CatBoost `baseline` GLR'nin logit'inden başladı. Eğitim satırlarının
  logit'i, o dış katın eğitim satırlarında **iç 5 katla** yeniden eğitilen GLR'den; doğrulama ve test satırlarınınki dış kat GLR'sinden.
  Fikir goodpjw2008'in "LR-Margin GBDT" not defterinden.

  | Model | Önce → sonra (10 kat) |
  |---|---|
  | XGBoost `t1` tok+chain, iki tohum | 0.94652 / 0.94648 → 0.94658 / 0.94654 |
  | LightGBM | 0.94650 → 0.94656 |
  | CatBoost d5 | 0.94648 → 0.94655 |
  | Artık XGBoost, 4 tohum ortalaması | 0.94659 |
  | Artık modelleri harmana eklemek | +0.00001 |
  | GLR logit'i üstüne artık MLP (256-64) | 0.94642; 3 bölme tohumu 0.94646; 20 kat 0.94647 |
  | GLR logit'ini RealMLP'ye **özellik** olarak vermek | 0.94636 → 0.94596 |
  | Harmandan `init_score` ile başlayıp ham sütunlar + üye tahminleriyle düzeltme | 1–8 ağaçta durdu, 0.94644 değişmedi |

  Son harmanda (19) en büyük ağırlık artık XGBoost'undaydı (20 kat, 0.281); artık CatBoost 0.074, 10 katlı artık XGBoost 0.053. Kendi artık
  LightGBM'imiz (0.94657) ve goodpjw2008'in artık modelleri (0.94646) ağırlık almadı.

Yeni yarışmaya:
- **[UYARLANMALI]** Taban modelin logit'inden başlayan GBDT kullanıcının planıyla (LLM logit'leri + stil özellikleri) aynı yapıda; çok sınıfta
  taban bir (N, 18) logit matrisi, LightGBM'de 2 boyutlu `init_score`, XGBoost'ta `multi:softprob` ile `base_margin` (`reuse/residual.py`).
- **[TAŞINIR]** Eğitim satırlarının taban logit'i iç katlardan gelmeli: S6E9'daki sızıntısız tarifin parçasıydı; LLM ince ayarı için bu dış
  kat başına iç K model demek (5 dış × 5 iç = 25 ek eğitim + 5 dış), süre bütçesine baştan yazılmalı.
- **[UYARLANMALI]** Logit'i başlangıç noktası yapmak ile özellik yapmak ayrı ölçülmeli: S6E9'da özellik olarak verilince RealMLP düştü
  (0.94636 → 0.94596), aynı logit'ten başlayan GBDT'ler yükseldi; iki ölçüm farklı model ailelerinde.
- **[TAŞINIR]** Bir harmanın (birden çok üyenin) üstüne `init_score` ile düzeltme getirmedi (1–8 ağaçta durdu): artık model tek ve güçlü bir
  tabanın üstünde işe yaradı.
- `reuse/residual.py` S6E9 tarifinin **yeniden kurulumu** (orijinal kod repoda yok): `nested_margins` (dış kat başına iç K katla taban
  logit'leri), `lgb_residual`, `xgb_residual`. CatBoost `baseline` uygulanmadı. Sahte veride (taban doğrusal, verinin bir kısmı doğrusal değil)
  OOF balanced accuracy taban 0.4263 → LightGBM artığı 0.5339, XGBoost artığı 0.5349; bu yalnız kodun çalıştığını gösterir, gerçek
  veride ölçüm değildir.

---

## 5. Blend yöntemi ve ağırlık seçimi

### 5.1 Ne yapıldı

- **Tepe tırmanma** (S6E10 `src/blend.py: fit_hill`, S6E9 `blend_ext.py`): her üye önce sıraya çevrilir (`rankdata / n`); en iyi tek modelden
  başlanır; AUC'yi en çok artıran üye (tekrar seçilebilir) bir kez daha eklenir; kazanç ≤ 1e-7 olunca ya da 200 adımda durur;
  ağırlık = seçilme sayısı / toplam.
- **Yöntem seçimi iç içe** (S6E10 `blend.py`): dört yöntem (tepe tırmanma, eşit sıra ortalaması, eşit olasılık ortalaması, logit üstünde
  lojistik regresyon) `StratifiedKFold(5, 42)` ile iç içe ölçülür, en iyisi seçilir; gönderim ağırlıkları tüm OOF satırlarında yeniden fit edilir.
  S6E9'da 18. ve 19. gönderimde son ağırlıklar 5 iç içe ağırlık vektörünün **ortalaması** (`AVGW=1`).
- **s03 karşılaştırması (8 aile, S6E10):**

  | Yöntem | OOF AUC | İç içe CV |
  |---|---|---|
  | Lojistik istif | 0.96159 | 0.96160 |
  | Tepe tırmanma | 0.96161 | 0.96159 |
  | Eşit sıra ortalaması | 0.96153 | 0.96153 |
  | Eşit olasılık ortalaması | 0.96146 | 0.96147 |

- **s08 ve sonrası: L2 lojistik istif** (`src/stack.py`): her üyenin logit'i bir sütun, `LogisticRegression(C=0.001)`; iç içe meta-CV
  `StratifiedKFold(5, shuffle, seed 7)`; gönderim için tüm OOF satırlarında fit, aynı katsayılar test logit'lerine. C taraması (s08):
  1 → 0.962103, 0.1 → 0.962106, 0.01 → 0.962121, **0.001 → 0.962127**, 0.0003 → 0.962102 (s09'da 0.0005 / 0.001 / 0.002 →
  0.962136 / 0.962149 / 0.962150). `ablate=1`: kaynak bırak-bir-dışarı (ör. s09'da kendi üyelerimiz −0.000087, goodpjw2008 −0.000031).
- **İstif üstünde denenip işe yaramayanlar (S6E10):** probit(sıra) girdisi 0.961265 (logit girdisi 0.962127; S6E9'da probit(sıra) üstünde
  lojistik istif tepe tırmanmayla aynı düzeydeydi);
  negatif olmayan ağırlıklar 0.962006 (serbest 0.962149, 184 üyenin 81'i negatif ağırlıklı); LightGBM meta-öğrenici 0.961997 ve
  0.962079–0.962088; logit kırpma ve sütun standardizasyonu ≤ 0.962151; yakınsamış çözücü (tol 1e-8) +0.000003 / 0; %20 LightGBM harmanı
  +0.000002 / 0. S6E9'da ikinci seviye LightGBM 0.94575 (tepe tırmanma 0.9464 düzeyinde).
- **Diğer ölçümler:** s05a'da lojistik istif aynı tarifli iki RealMLP arasında ±0.59 ağırlıkla fark aldı ve sağlam bulunmadı (o turda tepe
  tırmanma kullanıldı); s10'da
  ise 81 negatif ağırlığın iç içe CV'de gerçek bilgi taşıdığı görüldü (yukarıdaki negatif olmayan karşılaştırma). Farklı kat düzenlerinde
  üretilmiş üyeler (5, 10, 15 kat; farklı kat tohumu) aynı doğrusal istifte kullanıldı: her satırın tahmini o satırı görmemiş bir modelden.
- **S6E9 harmanında açık OOF'lar tek tek:** 25'inden 3'ü katkı yaptı (+0.000014, +0.000007, +0.000006); ağırlık alanların ikisi tek başına en
  zayıflar arasındaydı (residual-stack v19 0.94596 → 0.094, heuljax XGB 0.94631 → 0.094).
- **Yayınlanmış istiflerin OOF'unu üye yapmak** iç içe CV'yi şişirir (ewtherka'nın uyarısı); yapılmadı, başka üyelerin harmanı olan
  najiama 02/03 istif dışında tutuldu.

### 5.2 Yeni yarışmaya

- **[TAŞINIR]** Her ağırlık ve yöntem seçimini iç içe yapmak, gönderimi OOF içi skora göre değil iç içe skora göre seçmek.
- **[UYARLANMALI]** Tepe tırmanma: balanced accuracy argmax'a dayalı basamaklı bir metrik, 1e-7 eşiği ve tek satırlık değişimler
  anlamını yitirir; `reuse/hill_climb.py` balanced accuracy'nin yanında log loss ile ağırlık seçmeye de izin veriyor (hangisinin daha iyi
  olduğu ölçülmedi).
- **[TAŞINMAZ]** Sıra uzayı (rank space): AUC sıralamaya bakar, balanced accuracy her satırda sınıflar arası argmax'a; sınıf sütunlarının
  sıraları birbiriyle karşılaştırılamaz. `reuse/hill_climb.py` olasılık ya da log-olasılık uzayında çalışıyor.
- **[UYARLANMALI]** L2 lojistik istif: çok sınıfta her üye 18 sütun (satır içi merkezlenmiş log-olasılık, logit'in karşılığı; ölçülmüş bir
  seçim değil) ve multinomial LR'de 18 × 18 × üye sayısı ağırlık var (10 üyede 3.240 ağırlık, 2.699 satır); C=0.001 700 bin satırda
  267 üyeyle seçildi, yeniden iç içe seçilmeli (`reuse/stacker.py`).
- **[TAŞINIR]** Eşli meta-tohum testi ve kaynak ablasyonu (`reuse/stacker.paired_compare`, `ablate`): küçük farkları gürültüden ayırmanın
  ölçülmüş yolu buydu.
- **[UYARLANMALI]** Aile bazlı L2 (`design(..., family, factors)`): aile tanımı yeni üyelere göre yapılmalı; katsayılar bizim veride
  ayarlanmadan alınmıştı.
- **[TAŞINIR]** En güçlü değil en farklı üye ağırlık alır: S6E10'da hedef kodlamasız LightGBM tek başına 0.00024 zayıfken 2. büyük
  ağırlığı aldı, TE girdili MLP daha iyi olduğu hâlde 0.003 aldı; S6E9'da da ağırlık alanların ikisi en zayıflardandı.
- **[TAŞINMAZ]** AUC'ye özgü istif ayarları (probit(sıra) girdisi, logit kırpma): bir yarışmadan ötekine bile taşınmadı (S6E9'da tepe
  tırmanmayla aynı düzeydeydi, S6E10'da logit girdisinden 0.00086 düşük kaldı).
- **[TAŞINIR]** Başka bir istifin çıktısını üye yapmamak: iç içe CV'yi şişirir.

---

## 6. Eşik ya da sınıf ağırlığı optimizasyonu

- **S6E10: bulunamadı.** Metrik ROC AUC (eşikten bağımsız); kodda eşik, sınıf ağırlığı (`class_weight`, `scale_pos_weight`,
  `is_unbalance`) ya da olasılık kaydırma yok. `train.py`'deki `sample_weight` yalnız orijinal satırların ağırlığı (0.5 / 2 denendi, işe
  yaramadı).
- **S6E9: bulunamadı.** LESSONS.md'de eşik ya da sınıf ağırlığı optimizasyonu yok. En yakın şey sıra sonrası kurallar: belirli gelir /
  mesafe aralıklarındaki satırları en üste / en alta itmek (`RULES=1` OOF +0.000001, `DEADZONE=1` +0.000004; veriden madencilikle bulunan
  kurallar −0.00037 / −0.00004). **[TAŞINMAZ]** bu kurallar sentetik verideki istisnasız aralıklara ve AUC'nin sıralamasına özgüydü.

Yeni yarışmaya:
- **[TAŞINMAZ]** Taşınacak kod ya da ölçüm yok: planlanan "balanced accuracy için sınıf başına logit kaydırma"nın bu projelerde öncülü
  bulunamadı; `reuse/` altında bir eşik/bias modülü **uyarlanamadı**, çünkü uyarlanacak kod yok. Yazılırsa iç içe değerlendirme kabuğu olarak `reuse/hill_climb.nested_hill` ve
  `reuse/stacker.nested` düzeni (4/5'te öğren, 1/5'te ölç, birkaç bölme tohumu) kullanılabilir.
- **[UYARLANMALI]** Test tahminlerinin OOF'tan daha yumuşak olması: S6E10'da test tahminleri kat ortalaması olduğu için istif skorunun
  kontrast kısmı testte OOF'un 0.967 katıydı (etkisiz bulundu); OOF'ta ayarlanan her kaydırma kat ortalaması olan test olasılıklarına
  uygulanacağı için bu oran yeni projede de ölçülmeli.

---

## 7. Final 2 submission seçimi

- **S6E10: bulunamadı (henüz yapılmadı).** Yarışma 31.10.2026'da bitiyor; README'ye göre seçim kapanıştan önce kullanıcıyla birlikte
  yapılacak. Şu anki en iyi: s21, iç içe 0.962285, public 0.96187.
- **S6E9 (LESSONS.md §4):** 19 ve 11 önerildi.
  - 19: en iyi iç içe CV (0.94672) ve en iyi public (0.94676, 18 ile aynı).
  - 11: iyi gönderimler içinde en farklısı (19 ile sıra korelasyonu 0.9991); 17, 18 ve 19 birbirinin neredeyse kopyası olduğundan ikisini
    birlikte seçmek yedek işlevi görmezdi.
  - Sonuç: 19 private 0.94570, 11 private 0.94561; en yüksek private 0.94570 (18 ve 19). Kaggle'da hangi ikisinin işaretlendiği API
    çıktısında yok (doğrulanmadı).

Yeni yarışmaya:
- **[TAŞINIR]** "En iyi iç içe CV + iyiler arasında en farklı olan" kuralı: S6E9'da en iyi private yine en iyi CV'li gönderimdi, ikincisi
  yedek işlevi gördü.
- **[UYARLANMALI]** "Farklılık" ölçüsü: AUC'de sıra korelasyonu kullanıldı; etiket gönderiminde iki gönderimin aynı etiketi verdiği satır
  oranı gibi bir ölçü gerekir.
- **[UYARLANMALI]** Public'e bakarak seçmek: S6E9'da public–private Pearson 0.9997 idi ama orada dağılımlar aynıydı; yeni yarışmada
  public ve private'ın model karışımı farklı.

---

## 8. Zaman kaybettiren çıkmaz sokaklar ve sık hatalar

| Ne oldu | Kayıt | Yeni yarışma |
|---|---|---|
| 15 katlı TabPFN tam bağlam T4'te bellek yetersiz; sütun atılmış "lite" sürümler de üç bellek modunda yetersiz. Hata izleri iki tepe gösterdi ve ikisi de **satır** sayısına bağlıydı (karar anahtarı projeksiyonu 2,49 GiB; SDPA sarmalayıcısının K/V kopyaları 2 × 1,49 GiB). İzleri okuyup `lean_patch.py` v2 yazılınca 783 k × 23 bağlam 12,36 GB'ta sığdı | experiments.md 10.10 | **[TAŞINIR]** bellek yetersizliğinde önce hata izinden tepeyi ve neye bağlı olduğunu bulmak; parametre küçültmeyi körlemesine denemek boşa GPU saati harcadı. |
| Kumo-Tabular: büyük model her bağlamda bellek yetersiz (100 k bağlamda 55,9 GB istiyor), küçük model tam bağlam 0.958366 (karşılaştırma 0.958421, 130 dk); bu denemeyle haftalık GPU kotası bitti (30,17 / 30 saat) | experiments.md 09.10 | **[TAŞINIR]** büyük bir modeli denemeden önce tek kat + tek tohumla bellek/süre probu ve kota bütçesi; yeni yarışmada GPU kotası ve 9 saatlik sınır aynı kısıt. |
| Kendi 5 katlı TabPFN'imiz (Kaggle 2×T4'te 4,4 saat) havuzdaki açık TabPFN'le 0.998 korelasyonlu çıktı, istife ≈0 | README s09 | **[TAŞINIR]** pahalı bir üyeyi üretmeden önce havuzdaki benzerine ve istif katkısına bakmak. |
| Açık bir üye (arhancanli12 `*_5f`) olasılık değil [−2, 2,4] aralığında skor saklıyordu; istifçi bunları sessizce [1e-6, 1−1e-6]'ya kırpıyordu → sigmoid ile düzeltildi | experiments.md 09.10 | **[TAŞINIR]** her dış dosyanın ölçeğini denetlemek (`reuse/oof_io.to_probs`, `validate_member`). |
| Havuzda kopyalar: `gx_lgb_v3_k` = `gp_lgb_v3` (korelasyon 1,0), `o_tabpfn35` = `gp_tabpfn_catfd10`, yeni sürümler eskisiyle aynı | experiments.md 09.10, 11.10 | **[TAŞINIR]** yeni üyeyi almadan önce kopya testi (`reuse/oof_io.near_duplicates`). |
| Sızıntılı kaynaklar: tüm eğitim etiketleriyle kurulmuş istiflerin test tahminlerinden üretilmiş sözde etiket modelleri (OOF sızdırıyor), yayınlanmış istiflerin OOF'u, test satırlarını orijinal veriyle eşleştiren "parent row" yöntemi → alınmadı | experiments.md 13.10, 15.10 | **[TAŞINIR]** bir kaynağı almadan önce kodunu okumak; kullanıcı kuralı: eşleştirip etiket kopyalamak yok. |
| Pickle / `.pth` dosyaları (AutoGluon OOF, denetim noktaları) yüklenmedi | experiments.md 07.10, 10.10 | **[TAŞINIR]** indirilen dosya güvenilmez veridir; yalnız npy/csv/parquet. |
| Kaggle veri seti tam indirmesi sürüm işlenirken 404 verdi → dosyalar `-f` ile tek tek indirildi | experiments.md 13.10, 15.10 | **[TAŞINIR]** Kaggle API davranışı aynı. |
| 10 katlı FS3 hedef kodlaması bellekten öldü → kodlama 40'lık sütun parçalarına bölündü | git log 2026-10-02 | **[UYARLANMALI]** büyük gömme matrislerinde de parça parça hesap. |
| CatBoost'ta bütün kolonlar kategorik: 4 çekirdekte bir kat 25 dakikada bitmedi, durduruldu | README | **[TAŞINMAZ]** tabloya özgü ayar. |
| En iyi tek modeller istife katkı vermedi (aux RealMLP 0.961494, istif −0.000002); istif s21'den sonra doydu (yakınsamış çözücü, LightGBM harmanı, segment, golem v6 üyeleri: hepsi gürültü) | experiments.md | **[TAŞINIR]** iki toplayıcı (LR, GBDT) aynı noktaya yakınsadığında istif tarafında aramayı bırakıp yeni bilgi taşıyan üyelere dönmek. |
| Public sıralamasında aynı skorla 9 → 76, 17 → 90 oynama; bir günde 46 takım aynı 0.96176'daydı (kopya not defterleri) | experiments.md, HANDOFF_playground-series-s6e10.md | **[TAŞINIR]** sıralamaya göre karar vermemek. |
| S6E9'da 12 → 19 arası adımların her biri küçüktü (CV ≤ +0.00002, private'ta çoğu 0), toplamda CV +0.00006, private +0.00004 oldu | LESSONS.md §1 tablosu | **[TAŞINIR]** küçük ama iç içe ölçülmüş adımlar birikir; tek tek LB'de görünmemeleri yanlış oldukları anlamına gelmedi. |
| Arka plan kuyruğunu bekleyen betikler süreç adını regex ile aradı; desen kendi komut satırına da uydu (ve `aux+aux2`'deki `+` regex karakteriydi) → bekleyiciler hemen çıktı; açık PID'lerle beklemek çözdü | oturum kaydı (repoda iz yok) | **[TAŞINIR]** uzun koşuları ad desenine göre değil PID'e göre beklemek. |
| Doğrulama katında erken durdurma (her iki projede alışkanlık) | yanlılık ~0.00001 (S6E10 RealMLP) | **[UYARLANMALI]** bkz. 1.1: küçük katta ölçülmeli. |

---

## 9. `reuse/` modülleri

Hepsi tek dosya, birbirini düz adla içe aktarır (`from metrics import ...`); `python reuse/<modül>.py` kendi testini çalıştırır,
`python reuse/run_selftests.py` hepsini. Testler yalnız rastgele üretilmiş sahte veri kullanır; çıktıdaki skorlar kodun çalıştığını gösterir,
yöntemin değerini değil. Bağımlılıklar: numpy, scikit-learn, lightgbm, xgboost, pandas + pyarrow (bu ortamda sklearn 1.9.1, lightgbm 4.7.0,
xgboost 3.2.0, pandas 3.0.6, pyarrow 25.0.1).

| Modül | Kaynağı | Çok sınıf / yeni görev için değişen | Sahte veri testi (son koşu) |
|---|---|---|---|
| `metrics.py` | yeni (eski projelerde yalnız AUC) | balanced accuracy = argmax'ın sınıf başına recall ortalaması (y'de olmayan sınıflar ortalamaya girmez, sklearn gibi), log loss, kat skorları | sklearn `balanced_accuracy_score` / `log_loss` ile birebir: BA 0.3806, log loss 2.0295 |
| `oof_io.py` | S6E10 `common.py` (katlar, kayıt), `train.py` (kat checkpoint'i, tamamlananı atlama, deney satırı), `blend.py` ("a,b,c" tohum ortalaması), `ext_pub.py` (denetimler) | `StratifiedGroupKFold` + çakışma assert'i; (N, K) dosyalar; logit gelirse softmax; makul skor penceresi; kopya testi (log-olasılık korelasyonu ≥ 0.99999) | 40 sahte prompt'la 5 grup katı çakışmasız; kayıt/yükleme/tohum ortalaması, checkpoint, sızıntı penceresi ve şekil reddi, kopya bulma geçti |
| `hill_climb.py` | S6E10 `blend.py: fit_hill` + iç içe karşılaştırma, S6E9 `AVGW` | olasılık ya da log-olasılık uzayı; metrik balanced accuracy (ya da log loss, `higher_is_better=False`); sıra uzayı yok | zayıf üye seçilmedi (ağırlık 0); iç içe BA 0.4817 (eşit ortalama 0.4695) |
| `stacker.py` | S6E10 `stack.py` (L2 LR, C, iç içe meta-CV, `mseed`, `ablate`, `fam`) | her üye 18 sütun (satır içi merkezlenmiş log-olasılık), multinomial LR, gruplu meta katlar, BA + log loss | iç içe BA 0.4395; eşli karşılaştırma, ablasyon, son fit çalıştı |
| `residual.py` | S6E9 tarifi (LESSONS.md), yeniden kurulum | (N, K) taban logit'leri, iç katlı sızıntısız taban, LightGBM 2 boyutlu `init_score`, XGBoost `base_margin` | taban 0.4263 → LightGBM artığı 0.5339 / XGBoost artığı 0.5349 |
| `adversarial.py` | S6E10 `adversarial.py` | herhangi bir sayısal matris (gömme, stil), `groups=` ile gruplu katlar, OOF skor döner | aynı dağılım AUC 0.496; bir özellik kaydırılınca 0.801 ve en önemli özellik doğru |
| `submission.py` | S6E10 `submit_check.py` | etiket gönderimi (argmax → sınıf adı), parquet, örnek dosya sırası, izinli etiket kümesi, tahmin edilen sınıf sayımları | parquet örnek sırasında yazıldı; yabancı etiket ve tek sınıflı gönderim reddedildi |
| `kaggle_runner.py` | S6E10 `kaggle/gpu_kernel.py` (paketleme, push / status / fetch), `tabpfn/build_kernel.py` (süre sınırlı birim zamanlayıcı), `tabpfn/aggregate.py` (birimleri birleştirme) | yarışma, veri setleri, model kaynakları, internet, dosya bağlantıları parametre; (n, K) birimler; çekirdek içi yollar ortam değişkeniyle değişebilir | üretilen `run.py` sahte `/kaggle/input` ile yerelde çalıştı; zamanlayıcı süre dolunca 6 birim bitirip 6'sını bıraktı; iki "çekirdekten" gelen birimler birleşti, eksik kat yakalandı |

**Uyarlanamayanlar / taşınmayanlar:**
- **Eşik / bias optimizasyonu:** iki projede de yok (bölüm 6), uyarlanacak kod bulunamadı.
- **Tokenizer özellikleri:** S6E10'da kendi kodumuz yok; S6E9'un `gpt2tok.py`'si bu repoda yok (bölüm 3).
- **Sıra uzayında harman:** AUC'ye özgü, balanced accuracy'de anlamı yok (bölüm 5).
- **CatBoost `baseline` ile artık model:** S6E9'da vardı; `residual.py` yalnız LightGBM ve XGBoost'u kapsıyor.
- **İnternetsiz çalıştırma:** S6E10'daki bütün Kaggle çekirdekleri internetliydi (pip, Hugging Face indirmesi); internetsiz bir gönderim
  notebook'u (tekerlekler ve ağırlıklar ekli girdi olarak) bu projelerde hiç kurulmadı, `kaggle_runner.py` yalnız `internet=False`
  bayrağını ve `model_sources` alanını taşıyor.
- **Tabloya özgü kod:** `te_cache.py` / `features.py` (kat içi hedef kodlama, etiketsiz profiller), `realmlp.py`, `nn.py`, `glm.py`,
  TabPFN `worker.py` / `lean_patch.py`, `ext_pub.py`'nin kaynak başına ayrıştırıcıları, `tune.py` (Optuna; gerekirse doğrudan
  kullanılabilir, uyarlanmadı).
- **S6E9 kodu** (`train.py margin=1`, `blend_ext.py`, tam veri ve sözde etiket benzetimleri): bu repoda yok.

---

## 10. Planlanan fikirlerin bu projelerdeki karşılığı

| Plan | Bu projelerde | `reuse/` |
|---|---|---|
| `prompt_id` ile GroupKFold | S6E10 grup kullanmadı (satır bazlı `StratifiedKFold`); ilke (tek sabit bölme, her şey kat içinde) bölüm 1 | `oof_io.make_folds` (StratifiedGroupKFold) |
| LLM gömme sınıflandırıcısı (Qwen3 vb.) | bulunamadı (metin yoktu); büyük önceden eğitilmiş model + T4 bellek/süre dersleri TabPFN'den (bölüm 8) | `kaggle_runner` |
| Farklı lab tokenizer'larıyla token sayıları | bulunamadı; S6E9'da GPT-2 parçaları başka amaçla (bölüm 3) | yok |
| LLM logit'leri üstünde stil özellikli artık GBDT | S6E9 artık modelleri +0.00006–0.00007 (bölüm 4) | `residual` |
| OOF üzerinde tepe tırmanma | S6E10 / S6E9 (bölüm 5); lojistik istif ile aynı düzey (s03: 0.96159 / 0.96160) | `hill_climb`, `stacker` |
| Balanced accuracy için sınıf başına logit kaydırma | bulunamadı (bölüm 6) | yok |
| Test metinlerini prompt'a göre kümeleyip modelleri birebir eşleştirme | bulunamadı | yok |

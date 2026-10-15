# EDA

Bütün sayılar `src/eda.py` çıktısından; tam tablolar bu dosyanın sonunda, "Tablolar" bölümünde. Baseline karışıklık
analizi (`src/confusion.py`) "Baseline karışıklığı" bölümünde. Veri: `data/train.csv` (2.699 satır), örnek
`data/test.csv` (3 satır). Bulgular ve tablolar bağımsız bir denetimde başka kodla yeniden hesaplandı. Denetimin
düzelttiği noktalar burada işlendi (açılış istatistiği, kesme işareti normalizasyonu, `I/my` regex'i).

## Önemli bulgular

1. **Izgara tam:** 150 prompt × 18 model = 2.700 hücre. Her hücrede tam bir metin var, tek eksik **p002 / claude-fable-5**.
   Tekrarlanan metin yok. Yani her prompt'a her model bir kez cevap vermiş. Bu, testte prompt kümeleri içinde eşleştirme
   (PLAN.md §5-G) için güçlü bir kısıt. Ancak testte model başına sayılar eşit değil, bu yüzden kısıt orada gevşek olacak.
2. **Biçim neredeyse yok:** hiçbir metinde satır sonu yok (0/2.699). Markdown başlığı, madde işareti, numaralı liste,
   emoji ve `**kalın**` hiçbir modelde %1'e ulaşmıyor. `$` yalnız 1 metinde, `*` 15 metinde geçiyor. LaTeX (`\( \)`)
   yalnız gpt-5.6-luna (%3) ve gpt-5.6-terra (%1) metinlerinde, markdown italik en çok %3. Neredeyse hepsi tek paragraf
   düz yazı. "Biçim" sinyali pratikte **noktalama ve karakter seçimine** iniyor.
3. **Uzunluk zayıf bir sinyal:**
   - 36–274 kelime, ortalama 114,7. Dağılım 40'tan 250'ye azalarak iniyor (log-düzgüne benzer).
   - log(kelime) varyansının yalnız %0,9'unu model, %5,2'sini prompt açıklıyor.
   - Aynı prompt'ta iki modelin uzunlukları ilişkisiz: 153 model çiftinde prompt içi Spearman −0,21 … 0,16, ortalama ≈ 0.
   - Uzunluk büyük olasılıkla örnek başına rastgele verilmiş bir hedef. Model ortalamaları 100,7 (gemini-3.8-flash) ile
     123,6 (claude-opus-4-8) arasında.
4. **Lab düzeyinde güçlü karakter izleri** (% metin):

   | iz | claude | glm | kimi | gpt | qwen | gemini |
   |---|---|---|---|---|---|---|
   | em-dash `—` | 75 | 81 | 52 | 22 | 3 | 0 |
   | düz kesme `a'b` | 62 | 61 | 22 | 5 | 12 | 14 |
   | kıvrık kesme `a’b` | 0 | 0 | 32 | 46 | 27 | 20 |
   | kıvrık çift tırnak `“ ”` | 0 | 0 | 9 | 10 | 1 | 0 |

   - Boşluklu em-dash (`a — b`) neredeyse yalnız claude-opus-4-6'da (%27) ve claude-opus-5'te (%25) görülüyor. Diğer
     Claude'lar ve GLM bitişik (`a—b`) yazıyor.
   - GPT ailesi içinde em-dash kullanımı: luna %37, sol %27, terra %23, gpt-5.5 %13, gpt-6-astra %9.
   - Noktalı virgül içeren metin payı modelden modele %15 (claude-opus-4-6) ile %62 arasında değişiyor. Yüksek olanlar
     GPT (%43–62), kimi-k3 (%61), claude-opus-5 (%54) ve glm-5.3-flash (%48). Tek başına bir lab ayıracı değil.
   - Virgül/100 kelime en yüksek qwen3.8-max (11,7) ve gpt-5.5 (11,0).
5. **Sözcük izleri** (log-odds z; kesme işaretleri normalize edildi):
   - Gemini "of", "an", "profound", "expanse", "microscopic" kelimelerini fazla kullanıyor.
   - gpt-5.6-luna/terra çekinceli: "may", "can".
   - claude-fable-5 anlatı dilinde: "she", "her", "told".
   - kimi-k3'te "keeps", "patience", "debt"; qwen3.8-max'te "turning", "luminous", "becomes" öne çıkıyor.

   Her model aynı prompt'lara cevap verdiği için bunlar konu değil üslup farkı.
6. **Aynı lab modelleri birbirine daha yakın:**
   - Prompt içi kosinüs benzerliği, kelime TF-IDF'te aynı lab 0,207 / farklı lab 0,158; karakter TF-IDF'te 0,359 / 0,305.
   - Stil profili korelasyonunda aynı lab 0,665 / farklı lab −0,211.
   - Aynı lab'dan kardeşi olan 16 modelin 14'ünde kelime TF-IDF'teki en yakın komşu aynı lab'dan.
   - İstisnalar: glm-5.3 ve glm-5.3-flash'ın en yakın komşusu claude-fable-5. Kardeşi olmayan kimi-k3'ün de en yakını
     claude-fable-5, qwen3.8-max'inki gpt-5.6-luna.
   - En sıkı aileler stil profilinde: glm-5.3 ~ glm-5.3-flash 0,946; gemini-3.7 ~ 3.8-flash 0,936; gemini-3.6 ~ 3.7-flash 0,924.
   - Claude kendi içinde en dağınık aile (stil korelasyonu 0,32–0,73); claude-fable-5 stil olarak glm-5.3'e (0,934) daha yakın.

   Beklenen karışıklık bölgeleri: Gemini flash üçlüsü, GLM ikilisi, gpt-5.6 üçlüsü, claude-fable-5 ↔ GLM. Baseline
   karışıklığında hepsi çıktı, ayrıca öngörülmeyen birkaç bölge de var (aşağıda).
7. **Açılış prompt'tan güçlü biçimde etkileniyor:**
   - Kesme işaretleri normalize edilince metinlerin %53,9'unun ilk 3 kelimesi aynı prompt'taki en az bir başka metinle ortak.
   - 150 prompt'un 29'unda metinlerin en az yarısı aynı 3 kelimeyle açılıyor.
     Yalnız 6 prompt'ta hiçbir açılış ortak değil.
   - Açılış token'ları kısmen konu bilgisi. Bu yüzden ilk-parça anahtarlı özellikler (S6E9 `tok` L1/L2) prompt sızdırabilir;
     yalnız gruplu katlarda ölçülerek kullanılmalı.
8. **Bitiş:** modele göre metinlerin %96,7–100'ü nokta ile bitiyor. Yarım biten (harf/rakamla) metin yalnız kimi-k3'te var
   (%1,3, 2 metin).
9. **id sırası:** train.csv id'ye göre sıralı; id sırasında etiket yapısı yok (ardışık aynı sınıf z: +0,60 / −0,24 / +1,31).
   prompt_id sırasındaki güçlü negatif z (−11,9) beklenen prompt yapısı (her prompt'ta her model bir kez), sızıntı değil.

## Sonuçları etkileyen notlar

- Uzunluk ve paragraf yapısı neredeyse bilgi taşımıyor. Asıl sinyal kelime seçimi, noktalama alışkanlıkları ve aile içi
  ince farklar. Bu da karakter n-gram (Adım 2 baseline'ları), LLM gömmesi ve ince ayarın (PLAN §5 A, C) ana yük
  olacağını gösteriyor.
- Gemini, GPT ve Claude için lab ayırımı kolay. GLM (→ Claude) ve Kimi/Qwen (→ GPT) için lab da zor. Aile içi ayırım
  genel olarak zor.

## Baseline karışıklığı

En iyi tek model (`tfidf_charword_lr_C30`, EXPERIMENTS.md #5) ve en iyi iç içe harman (`blend_v2_tok_ba`) OOF'larından.
Özet:
- Lab tahmini %70,5 (tek model) ve %72,7 (harman) doğru. Hataların ~%55'i aile içinde, ~%45'i lab dışında.
- Gemini lab olarak %95 doğru bulunuyor. Kaybı büyük ölçüde lab içinde: ~%60'ı flash üçlüsünde, yaklaşık üçte biri
  gemini-3.1-pro-preview ↔ flash (özellikle 3.6-flash).
- GLM metinlerinin %47'si Claude olarak tahmin ediliyor.
- Kimi'nin %35–39'u ve Qwen'in %38–39'u GPT'ye kayıyor.
- EDA'da (bulgu 6) öngörülen karışıklık bölgelerinin hepsi çıktı. Öngörülmeyenler de var: gpt-5.5 ↔ kimi-k3,
  gpt-5.5 ↔ qwen3.8-max, claude-opus-5 ↔ glm-5.3. Harmanda claude-fable-5 en çok claude-opus-5 ve claude-sonnet-5 ile
  karışıyor.

#### Karışıklık: `tfidf_charword_lr_C30`

OOF BA 0.3471; tahmin edilen lab doğru: %70.5 (lab içi yanlış: %35.8, lab dışı yanlış: %29.5).

| model | recall | en çok karıştırıldığı (sayı) | bu sınıfa tahmin edilen toplam |
|---|---|---|---|
| claude-fable-5 | 0.094 | glm-5.3-flash (25); claude-opus-5 (24) | 89 |
| glm-5.3 | 0.173 | claude-opus-5 (35); glm-5.3-flash (27) | 124 |
| glm-5.3-flash | 0.173 | glm-5.3 (30); claude-opus-5 (22) | 140 |
| gpt-5.6-terra | 0.213 | gpt-5.6-luna (29); gpt-6-astra (23) | 138 |
| gpt-5.6-sol | 0.220 | gpt-5.6-terra (36); gpt-5.6-luna (29) | 115 |
| claude-opus-4-8 | 0.247 | claude-opus-5 (17); claude-sonnet-5 (16) | 111 |
| gpt-5.5 | 0.280 | kimi-k3 (22); gpt-5.6-terra (17) | 149 |
| gpt-5.6-luna | 0.307 | gpt-5.6-terra (23); gpt-5.6-sol (21) | 192 |
| kimi-k3 | 0.307 | gpt-5.5 (28); glm-5.3 (10) | 165 |
| gemini-3.8-flash | 0.333 | gemini-3.7-flash (46); gemini-3.6-flash (25) | 151 |
| claude-sonnet-5 | 0.367 | claude-opus-4-8 (17); glm-5.3-flash (14) | 165 |
| qwen3.8-max | 0.380 | gpt-5.6-luna (17); gpt-5.5 (16) | 124 |
| gemini-3.7-flash | 0.393 | gemini-3.8-flash (39); gemini-3.6-flash (27) | 161 |
| claude-opus-4-6 | 0.453 | claude-sonnet-5 (19); claude-opus-5 (14) | 130 |
| gemini-3.6-flash | 0.460 | gemini-3.7-flash (30); gemini-3.1-pro-preview (23) | 152 |
| gpt-6-astra | 0.507 | gpt-5.6-luna (26); gpt-5.6-sol (9) | 178 |
| claude-opus-5 | 0.667 | claude-sonnet-5 (9); claude-opus-4-8 (8) | 247 |
| gemini-3.1-pro-preview | 0.673 | gemini-3.6-flash (19); gemini-3.8-flash (13) | 168 |

En sık karışan 12 çift (iki yönün toplamı; ✓ = aynı lab):

gemini-3.7-flash ↔ gemini-3.8-flash 85 ✓, gpt-5.6-sol ↔ gpt-5.6-terra 58 ✓, glm-5.3 ↔ glm-5.3-flash 57 ✓, gemini-3.6-flash ↔ gemini-3.7-flash 57 ✓, gpt-5.6-luna ↔ gpt-5.6-terra 52 ✓, gpt-5.5 ↔ kimi-k3 50, gpt-5.6-luna ↔ gpt-5.6-sol 50 ✓, gemini-3.6-flash ↔ gemini-3.8-flash 47 ✓, gpt-5.6-luna ↔ gpt-6-astra 47 ✓, claude-opus-5 ↔ glm-5.3 43, gemini-3.1-pro-preview ↔ gemini-3.6-flash 42 ✓, claude-fable-5 ↔ glm-5.3-flash 38

Lab düzeyinde karışıklık (satır = gerçek lab, sütun = tahmin; satır yüzdesi):

| gerçek \ tahmin | claude | gemini | glm | gpt | kimi | qwen |
|---|---|---|---|---|---|---|
| claude | 69 | 5 | 16 | 4 | 4 | 2 |
| gemini | 2 | 95 | 0 | 1 | 1 | 1 |
| glm | 47 | 4 | 36 | 6 | 5 | 1 |
| gpt | 4 | 1 | 2 | 80 | 8 | 5 |
| kimi | 18 | 1 | 12 | 35 | 31 | 3 |
| qwen | 9 | 5 | 2 | 39 | 7 | 38 |

#### Karışıklık: `blend_v2_tok_ba`

OOF BA 0.3923; tahmin edilen lab doğru: %72.7 (lab içi yanlış: %33.4, lab dışı yanlış: %27.3).

| model | recall | en çok karıştırıldığı (sayı) | bu sınıfa tahmin edilen toplam |
|---|---|---|---|
| claude-fable-5 | 0.148 | claude-opus-5 (25); claude-sonnet-5 (22) | 99 |
| glm-5.3-flash | 0.180 | glm-5.3 (33); claude-sonnet-5 (20) | 125 |
| glm-5.3 | 0.187 | glm-5.3-flash (32); claude-opus-5 (24) | 132 |
| gpt-5.6-sol | 0.207 | gpt-5.6-luna (32); gpt-5.6-terra (26) | 104 |
| gpt-5.6-terra | 0.233 | gpt-5.6-luna (34); gpt-5.6-sol (21) | 123 |
| claude-opus-4-8 | 0.273 | claude-sonnet-5 (21); glm-5.3 (14) | 108 |
| kimi-k3 | 0.293 | gpt-5.5 (28); gpt-5.6-terra (11) | 132 |
| gpt-5.6-luna | 0.353 | gpt-5.6-sol (23); gpt-5.6-terra (16) | 172 |
| gemini-3.8-flash | 0.367 | gemini-3.7-flash (43); gemini-3.6-flash (23) | 153 |
| gpt-5.5 | 0.407 | kimi-k3 (18); qwen3.8-max (17) | 183 |
| qwen3.8-max | 0.413 | gpt-5.5 (24); gpt-5.6-luna (10) | 143 |
| gemini-3.7-flash | 0.433 | gemini-3.8-flash (37); gemini-3.6-flash (30) | 159 |
| gemini-3.6-flash | 0.453 | gemini-3.7-flash (31); gemini-3.8-flash (23) | 155 |
| claude-sonnet-5 | 0.467 | claude-opus-4-8 (14); claude-fable-5 (12) | 188 |
| claude-opus-4-6 | 0.553 | claude-sonnet-5 (16); claude-opus-5 (12) | 142 |
| gpt-6-astra | 0.633 | gpt-5.6-luna (17); gpt-5.5 (8) | 191 |
| claude-opus-5 | 0.720 | claude-fable-5 (9); glm-5.3 (9) | 221 |
| gemini-3.1-pro-preview | 0.740 | gemini-3.6-flash (15); gemini-3.8-flash (14) | 169 |

En sık karışan 12 çift (iki yönün toplamı; ✓ = aynı lab):

gemini-3.7-flash ↔ gemini-3.8-flash 80 ✓, glm-5.3 ↔ glm-5.3-flash 65 ✓, gemini-3.6-flash ↔ gemini-3.7-flash 61 ✓, gpt-5.6-luna ↔ gpt-5.6-sol 55 ✓, gpt-5.6-luna ↔ gpt-5.6-terra 50 ✓, gpt-5.6-sol ↔ gpt-5.6-terra 47 ✓, gpt-5.5 ↔ kimi-k3 46, gemini-3.6-flash ↔ gemini-3.8-flash 46 ✓, gpt-5.5 ↔ qwen3.8-max 41, claude-fable-5 ↔ glm-5.3 39, gpt-5.5 ↔ gpt-5.6-terra 36 ✓, claude-opus-4-8 ↔ claude-sonnet-5 35 ✓

Lab düzeyinde karışıklık (satır = gerçek lab, sütun = tahmin; satır yüzdesi):

| gerçek \ tahmin | claude | gemini | glm | gpt | kimi | qwen |
|---|---|---|---|---|---|---|
| claude | 74 | 4 | 14 | 3 | 3 | 1 |
| gemini | 2 | 95 | 0 | 2 | 1 | 1 |
| glm | 47 | 2 | 40 | 4 | 3 | 3 |
| gpt | 2 | 2 | 1 | 81 | 6 | 7 |
| kimi | 15 | 1 | 13 | 39 | 29 | 3 |
| qwen | 7 | 8 | 2 | 38 | 4 | 41 |

## Tablolar (`src/eda.py` çıktısı)

### 1. (prompt_id, model) çiftleri

- Satır 2699, prompt 150, model 18; tam ızgara 150 × 18 = 2700 hücre.
- Hücre değerleri: 0 → 1 hücre, 1 → 2699 hücre.
- Eksik çiftler: [('p002', 'claude-fable-5')]; birden fazla olan: yok.
- Prompt başına metin sayısı: 17 metin → 1 prompt, 18 metin → 149 prompt.
- Tekrarlanan metin (birebir aynı): 0; id tekil: True.
- Prompt kimlikleri: ['p000', 'p001', 'p002'] … ['p147', 'p148', 'p149'].

### 2. Model başına metin sayısı ve uzunluk (kelime = boşlukla ayrılmış parça)

| model | n | mean | std | min | p10 | median | p90 | max | chars_mean | <40 % | >250 % | paragraf ort. | cümle ort. kelime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini-3.8-flash | 150 | 100.7 | 52.8 | 41 | 47.0 | 84.0 | 191.0 | 238 | 706.2 | 0.0 | 0.0 | 1.0 | 22.7 |
| kimi-k3 | 150 | 107.4 | 53.7 | 39 | 49.9 | 89.0 | 189.3 | 239 | 690.1 | 0.7 | 0.0 | 1.0 | 26.0 |
| qwen3.8-max | 150 | 108.3 | 56.8 | 39 | 46.0 | 98.5 | 192.1 | 265 | 707.7 | 0.7 | 0.7 | 1.0 | 22.3 |
| gpt-5.5 | 150 | 110.3 | 54.7 | 39 | 47.8 | 103.5 | 187.2 | 245 | 698.6 | 0.7 | 0.0 | 1.0 | 26.4 |
| glm-5.3 | 150 | 110.3 | 56.8 | 36 | 46.9 | 96.0 | 203.0 | 244 | 710.3 | 2.0 | 0.0 | 1.0 | 23.6 |
| gemini-3.6-flash | 150 | 111.2 | 54.1 | 40 | 53.0 | 95.5 | 201.2 | 238 | 758.6 | 0.0 | 0.0 | 1.0 | 24.7 |
| claude-fable-5 | 149 | 112.5 | 58.2 | 39 | 48.0 | 100.0 | 212.4 | 252 | 731.0 | 0.7 | 1.3 | 1.0 | 24.5 |
| gpt-5.6-terra | 150 | 113.3 | 54.5 | 43 | 50.0 | 100.5 | 200.1 | 239 | 746.6 | 0.0 | 0.0 | 1.0 | 26.4 |
| claude-opus-4-6 | 150 | 115.1 | 60.9 | 42 | 51.9 | 98.5 | 212.5 | 274 | 771.2 | 0.0 | 4.7 | 1.0 | 27.4 |
| gpt-5.6-luna | 150 | 115.4 | 57.8 | 38 | 50.9 | 104.5 | 203.1 | 246 | 766.6 | 0.7 | 0.0 | 1.0 | 23.7 |
| gemini-3.7-flash | 150 | 117.1 | 59.8 | 40 | 51.0 | 101.5 | 213.1 | 252 | 828.7 | 0.0 | 0.7 | 1.0 | 23.9 |
| gpt-6-astra | 150 | 117.8 | 59.1 | 36 | 47.0 | 105.5 | 211.0 | 256 | 750.3 | 0.7 | 0.7 | 1.0 | 21.1 |
| gpt-5.6-sol | 150 | 118.9 | 60.5 | 40 | 45.9 | 111.5 | 211.1 | 261 | 779.4 | 0.0 | 0.7 | 1.0 | 24.9 |
| glm-5.3-flash | 150 | 119.2 | 58.2 | 37 | 50.9 | 112.0 | 201.2 | 256 | 777.6 | 2.7 | 0.7 | 1.0 | 23.6 |
| gemini-3.1-pro-preview | 150 | 119.3 | 59.6 | 38 | 51.9 | 104.0 | 211.5 | 262 | 783.0 | 0.7 | 2.7 | 1.0 | 20.3 |
| claude-sonnet-5 | 150 | 120.0 | 64.3 | 39 | 48.0 | 99.5 | 214.2 | 267 | 816.6 | 1.3 | 2.0 | 1.0 | 26.6 |
| claude-opus-5 | 150 | 123.4 | 62.3 | 39 | 45.0 | 116.5 | 217.0 | 257 | 762.8 | 1.3 | 0.7 | 1.0 | 24.0 |
| claude-opus-4-8 | 150 | 123.6 | 61.6 | 42 | 51.0 | 109.0 | 214.5 | 251 | 809.8 | 0.0 | 0.7 | 1.0 | 24.1 |

Kelime sayısı histogramı (20'lik kutular): 30–49: 301, 50–69: 482, 70–89: 397, 90–109: 294, 110–129: 249, 130–149: 218, 150–169: 195, 170–189: 164, 190–209: 157, 210–229: 140, 230–249: 77, 250–269: 24.

Genel: ortalama 114.7, medyan 101, min 36, max 274 kelime; <40 kelime 18, >250 kelime 23 metin.
log(kelime) varyansının prompt ortalamalarıyla açıklanan payı 0.052, model ortalamalarıyla açıklanan payı 0.009.

### 3. Model başına biçim izleri (% metin: en az bir kez içeren)

| model | *italik* | em-dash — | boşluklu ' — ' | bitişik a—b | en-dash – | kıvrık “ ” | kıvrık ‘ ’ | düz " | düz a'b | LaTeX | ; | : | ! | ? | ( | not X but Y | kısaltma (n't, 's) | we/our | I/my | you |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-fable-5 | 0 | 83 | 0 | 83 | 0 | 0 | 0 | 9 | 58 | 0 | 34 | 59 | 0 | 5 | 3 | 10 | 58 | 30 | 3 | 18 |
| claude-opus-4-6 | 3 | 75 | 27 | 49 | 0 | 0 | 0 | 7 | 57 | 0 | 15 | 42 | 1 | 3 | 0 | 17 | 57 | 32 | 3 | 19 |
| claude-opus-4-8 | 3 | 63 | 0 | 63 | 0 | 0 | 0 | 7 | 58 | 0 | 24 | 49 | 0 | 5 | 1 | 16 | 58 | 37 | 6 | 24 |
| claude-opus-5 | 1 | 81 | 25 | 56 | 1 | 0 | 0 | 9 | 62 | 0 | 54 | 69 | 1 | 3 | 2 | 22 | 62 | 32 | 6 | 24 |
| claude-sonnet-5 | 1 | 73 | 0 | 73 | 0 | 0 | 0 | 10 | 73 | 0 | 23 | 51 | 1 | 1 | 1 | 16 | 73 | 28 | 5 | 15 |
| gemini-3.1-pro-preview | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 3 | 10 | 0 | 19 | 17 | 1 | 5 | 0 | 5 | 14 | 42 | 7 | 15 |
| gemini-3.6-flash | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 3 | 20 | 0 | 23 | 20 | 0 | 4 | 1 | 3 | 37 | 31 | 5 | 15 |
| gemini-3.7-flash | 1 | 1 | 0 | 1 | 0 | 0 | 31 | 0 | 14 | 0 | 23 | 23 | 0 | 3 | 1 | 3 | 39 | 31 | 4 | 15 |
| gemini-3.8-flash | 0 | 0 | 0 | 0 | 0 | 2 | 25 | 1 | 11 | 0 | 28 | 18 | 0 | 2 | 0 | 4 | 35 | 27 | 3 | 15 |
| glm-5.3 | 0 | 82 | 3 | 79 | 0 | 0 | 0 | 9 | 57 | 0 | 34 | 62 | 1 | 4 | 1 | 13 | 57 | 31 | 7 | 21 |
| glm-5.3-flash | 0 | 79 | 2 | 77 | 1 | 0 | 0 | 12 | 65 | 0 | 48 | 65 | 0 | 4 | 0 | 17 | 65 | 37 | 6 | 20 |
| gpt-5.5 | 0 | 13 | 0 | 13 | 0 | 9 | 42 | 1 | 11 | 0 | 61 | 65 | 1 | 2 | 0 | 19 | 53 | 29 | 10 | 14 |
| gpt-5.6-luna | 1 | 37 | 0 | 37 | 1 | 11 | 51 | 0 | 1 | 3 | 43 | 71 | 0 | 4 | 5 | 13 | 51 | 27 | 8 | 9 |
| gpt-5.6-sol | 0 | 27 | 0 | 27 | 1 | 15 | 53 | 0 | 0 | 0 | 62 | 64 | 1 | 3 | 1 | 13 | 51 | 27 | 9 | 7 |
| gpt-5.6-terra | 0 | 23 | 0 | 23 | 1 | 8 | 47 | 0 | 5 | 1 | 52 | 73 | 0 | 1 | 2 | 13 | 50 | 21 | 5 | 7 |
| gpt-6-astra | 0 | 9 | 0 | 9 | 0 | 7 | 44 | 0 | 9 | 0 | 53 | 65 | 0 | 5 | 1 | 11 | 50 | 35 | 7 | 19 |
| kimi-k3 | 0 | 52 | 0 | 52 | 1 | 9 | 32 | 1 | 22 | 0 | 61 | 77 | 0 | 2 | 1 | 21 | 54 | 27 | 7 | 16 |
| qwen3.8-max | 0 | 3 | 0 | 3 | 0 | 1 | 29 | 0 | 12 | 0 | 39 | 47 | 0 | 1 | 0 | 23 | 39 | 30 | 5 | 13 |

Hiçbir modelde %1'e ulaşmayan izler (tablodan çıkarıldı): **kalın**, # başlık, madde (-,*,•), numaralı liste, >1 paragraf, --, ' - ', emoji, …, ..., ×≈π° vb., nbsp/ince boşluk.
Satır sonu (\n) içeren metin: 0; '#' içeren: 0; '*' içeren: 15; emoji içeren: 0; '$' içeren: 1.

### 3a. Lab düzeyinde özet (% metin)
| lab | em-dash | boşluklu em-dash | kıvrık ‘ ’ | kıvrık “ ” | düz a'b | kıvrık a’b | ; |
|---|---|---|---|---|---|---|---|
| claude | 75 | 10 | 0 | 0 | 62 | 0 | 30 |
| gemini | 0 | 0 | 20 | 0 | 14 | 20 | 23 |
| glm | 81 | 3 | 0 | 0 | 61 | 0 | 41 |
| gpt | 22 | 0 | 47 | 10 | 5 | 46 | 54 |
| kimi | 52 | 0 | 32 | 9 | 22 | 32 | 61 |
| qwen | 3 | 0 | 29 | 1 | 12 | 27 | 39 |


### 3b. 100 kelime başına ortalama sayım (noktalama ve tire)
| model | em_dash | comma | semicolon | colon | paren | question | curly_apostrophe | straight_apostrophe | hyphen_word | contraction | not_x_but_y |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-fable-5 | 1.36 | 9.15 | 0.35 | 0.67 | 0.03 | 0.04 | 0.00 | 0.98 | 0.57 | 0.98 | 0.07 |
| claude-opus-4-6 | 1.13 | 7.75 | 0.12 | 0.37 | 0.00 | 0.03 | 0.00 | 0.95 | 0.75 | 0.95 | 0.16 |
| claude-opus-4-8 | 0.86 | 9.50 | 0.19 | 0.50 | 0.01 | 0.04 | 0.00 | 0.87 | 0.57 | 0.87 | 0.12 |
| claude-opus-5 | 1.22 | 8.36 | 0.58 | 0.80 | 0.01 | 0.05 | 0.00 | 0.99 | 0.59 | 0.99 | 0.17 |
| claude-sonnet-5 | 1.22 | 9.62 | 0.23 | 0.59 | 0.00 | 0.01 | 0.00 | 1.47 | 0.84 | 1.46 | 0.15 |
| gemini-3.1-pro-preview | 0.00 | 7.57 | 0.16 | 0.16 | 0.00 | 0.05 | 0.04 | 0.12 | 0.11 | 0.16 | 0.03 |
| gemini-3.6-flash | 0.00 | 8.36 | 0.20 | 0.17 | 0.01 | 0.04 | 0.20 | 0.24 | 0.13 | 0.45 | 0.02 |
| gemini-3.7-flash | 0.01 | 8.25 | 0.20 | 0.17 | 0.01 | 0.03 | 0.35 | 0.17 | 0.23 | 0.52 | 0.04 |
| gemini-3.8-flash | 0.00 | 8.52 | 0.28 | 0.18 | 0.00 | 0.03 | 0.31 | 0.15 | 0.17 | 0.47 | 0.03 |
| glm-5.3 | 1.51 | 9.02 | 0.37 | 0.75 | 0.01 | 0.04 | 0.00 | 1.01 | 0.63 | 1.00 | 0.11 |
| glm-5.3-flash | 1.38 | 8.71 | 0.53 | 0.74 | 0.00 | 0.03 | 0.00 | 1.05 | 0.63 | 1.05 | 0.15 |
| gpt-5.5 | 0.18 | 11.00 | 0.75 | 0.82 | 0.00 | 0.02 | 0.67 | 0.15 | 0.71 | 0.82 | 0.17 |
| gpt-5.6-luna | 0.48 | 10.28 | 0.45 | 0.87 | 0.14 | 0.05 | 0.68 | 0.02 | 0.50 | 0.70 | 0.11 |
| gpt-5.6-sol | 0.34 | 10.12 | 0.64 | 0.79 | 0.01 | 0.03 | 0.76 | 0.00 | 0.51 | 0.76 | 0.12 |
| gpt-5.6-terra | 0.37 | 10.57 | 0.52 | 0.89 | 0.06 | 0.01 | 0.68 | 0.05 | 0.59 | 0.73 | 0.12 |
| gpt-6-astra | 0.14 | 7.87 | 0.59 | 0.76 | 0.01 | 0.06 | 0.65 | 0.14 | 0.35 | 0.79 | 0.13 |
| kimi-k3 | 0.88 | 9.98 | 0.82 | 0.98 | 0.00 | 0.02 | 0.45 | 0.41 | 0.60 | 0.85 | 0.19 |
| qwen3.8-max | 0.05 | 11.73 | 0.38 | 0.49 | 0.00 | 0.00 | 0.48 | 0.16 | 0.53 | 0.63 | 0.24 |

### 3c. Sözcük düzeyi
| model | ASCII dışı karakter % | tip/token | ort. kelime uzunluğu | ≥9 harf kelime oranı | büyük harfle başlayan kelime oranı |
|---|---|---|---|---|---|
| claude-fable-5 | 0.230 | 0.812 | 5.303 | 0.127 | 0.061 |
| claude-opus-4-6 | 0.184 | 0.815 | 5.590 | 0.166 | 0.056 |
| claude-opus-4-8 | 0.141 | 0.802 | 5.403 | 0.144 | 0.058 |
| claude-opus-5 | 0.207 | 0.782 | 5.026 | 0.107 | 0.060 |
| claude-sonnet-5 | 0.190 | 0.823 | 5.596 | 0.157 | 0.056 |
| gemini-3.1-pro-preview | 0.016 | 0.804 | 5.498 | 0.153 | 0.065 |
| gemini-3.6-flash | 0.036 | 0.828 | 5.712 | 0.171 | 0.057 |
| gemini-3.7-flash | 0.059 | 0.841 | 5.931 | 0.197 | 0.055 |
| gemini-3.8-flash | 0.061 | 0.860 | 5.890 | 0.188 | 0.059 |
| glm-5.3 | 0.254 | 0.814 | 5.288 | 0.126 | 0.063 |
| glm-5.3-flash | 0.228 | 0.806 | 5.323 | 0.127 | 0.063 |
| gpt-5.5 | 0.188 | 0.826 | 5.157 | 0.110 | 0.055 |
| gpt-5.6-luna | 0.227 | 0.824 | 5.485 | 0.146 | 0.058 |
| gpt-5.6-sol | 0.244 | 0.826 | 5.398 | 0.126 | 0.058 |
| gpt-5.6-terra | 0.204 | 0.831 | 5.379 | 0.130 | 0.053 |
| gpt-6-astra | 0.162 | 0.819 | 5.245 | 0.113 | 0.059 |
| kimi-k3 | 0.260 | 0.835 | 5.260 | 0.117 | 0.058 |
| qwen3.8-max | 0.097 | 0.839 | 5.423 | 0.134 | 0.058 |


### 4. Başlangıç ve bitiş kalıpları

| model | en sık ilk 2 kelime (pay %) | ilk 3 kelimesi prompt'taki başka bir metinle ortak % | nokta ile biter % | ? ile % | ! ile % | tırnak/parantez ile % | harf/rakamla (yarım) % |
|---|---|---|---|---|---|---|---|
| claude-fable-5 | in the (6); at the (3); beneath the (3) | 69.1 | 99.3 | 0.7 | 0.0 | 0.0 | 0.0 |
| claude-opus-4-6 | in the (6); at the (3); imagine a (1) | 60.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| claude-opus-4-8 | in the (7); imagine a (3); at the (3) | 59.3 | 98.7 | 0.7 | 0.0 | 0.7 | 0.0 |
| claude-opus-5 | in the (3); there is (3); beneath the (2) | 32.7 | 99.3 | 0.7 | 0.0 | 0.0 | 0.0 |
| claude-sonnet-5 | in the (8); beneath the (3); in a (2) | 60.7 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gemini-3.1-pro-preview | in the (5); imagine a (3); deep within (3) | 42.0 | 98.7 | 1.3 | 0.0 | 0.0 | 0.0 |
| gemini-3.6-flash | in the (5); beneath the (5); deep within (4) | 42.0 | 98.7 | 1.3 | 0.0 | 0.0 | 0.0 |
| gemini-3.7-flash | beneath the (6); deep within (5); at the (4) | 44.7 | 99.3 | 0.7 | 0.0 | 0.0 | 0.0 |
| gemini-3.8-flash | beneath the (5); across the (4); in the (4) | 42.7 | 99.3 | 0.7 | 0.0 | 0.0 | 0.0 |
| glm-5.3 | beneath the (3); deep within (3); at the (2) | 43.3 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| glm-5.3-flash | beneath the (3); in the (3); deep in (3) | 53.3 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gpt-5.5 | across the (3); at the (2); in a (2) | 63.3 | 98.7 | 0.7 | 0.0 | 0.7 | 0.0 |
| gpt-5.6-luna | in the (4); at the (3); imagine a (3) | 58.7 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gpt-5.6-sol | in the (4); at the (3); when a (2) | 60.0 | 98.7 | 0.0 | 0.0 | 1.3 | 0.0 |
| gpt-5.6-terra | at the (4); across the (3); when the (3) | 59.3 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gpt-6-astra | in the (3); across the (3); imagine a (3) | 62.0 | 96.7 | 1.3 | 0.0 | 2.0 | 0.0 |
| kimi-k3 | in the (5); at the (3); within the (2) | 56.0 | 98.0 | 0.7 | 0.0 | 0.0 | 1.3 |
| qwen3.8-max | in the (5); in a (3); deep in (2) | 60.7 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Bütün metinlerde ilk 3 kelimesi (’ → ' normalize) aynı prompt'taki en az bir başka metinle ortak olanlar: %53.9. Prompt'ların 29/150'ünde metinlerin en az yarısı aynı 3 kelimeyle başlıyor; 6 prompt'ta hiçbir açılış ortak değil.
Son iki karakterin en sık değerleri: 'e.' 604, 's.' 470, 't.' 248, 'n.' 239, 'y.' 225, 'd.' 222, 'r.' 136, 'g.' 108.

### 5. Modeller arası benzerlik (aynı lab ile farklı lab)

Aynı prompt'a verilen cevaplar arasında ortalama kosinüs benzerliği (model çifti başına 149–150 prompt ortalaması) ve model başına ortalama stil profili korelasyonu.

| ölçü | aynı lab çiftleri ort. | farklı lab çiftleri ort. | en benzer 6 çift |
|---|---|---|---|
| kelime 1–2 gram TF-IDF | 0.207 | 0.158 | claude-fable-5 ~ claude-opus-4-8 0.249; gemini-3.7-flash ~ gemini-3.8-flash 0.232; gemini-3.6-flash ~ gemini-3.7-flash 0.229; gpt-5.6-sol ~ gpt-5.6-terra 0.228; claude-fable-5 ~ glm-5.3-flash 0.224; gpt-5.6-luna ~ gpt-5.6-sol 0.223 |
| karakter 3–5 gram TF-IDF | 0.359 | 0.305 | claude-fable-5 ~ claude-opus-4-8 0.396; gpt-5.6-sol ~ gpt-5.6-terra 0.386; gpt-5.6-luna ~ gpt-5.6-sol 0.383; gemini-3.6-flash ~ gemini-3.7-flash 0.383; gemini-3.7-flash ~ gemini-3.8-flash 0.382; gpt-5.6-luna ~ gpt-5.6-terra 0.376 |
| stil profili korelasyonu | 0.665 | -0.211 | glm-5.3 ~ glm-5.3-flash 0.946; gemini-3.7-flash ~ gemini-3.8-flash 0.936; claude-fable-5 ~ glm-5.3 0.934; gemini-3.6-flash ~ gemini-3.7-flash 0.924; gemini-3.6-flash ~ gemini-3.8-flash 0.910; claude-fable-5 ~ glm-5.3-flash 0.901 |

### 5b. Her modelin en yakın komşusu (kelime TF-IDF / karakter TF-IDF / stil profili)

| model | lab | kelime TF-IDF | karakter TF-IDF | stil profili |
|---|---|---|---|---|
| claude-fable-5 | claude | claude-opus-4-8 (0.249) ✓ | claude-opus-4-8 (0.396) ✓ | glm-5.3 (0.934) |
| claude-opus-4-6 | claude | claude-fable-5 (0.212) ✓ | claude-opus-4-8 (0.368) ✓ | claude-opus-5 (0.572) ✓ |
| claude-opus-4-8 | claude | claude-fable-5 (0.249) ✓ | claude-fable-5 (0.396) ✓ | claude-sonnet-5 (0.728) ✓ |
| claude-opus-5 | claude | claude-fable-5 (0.192) ✓ | claude-fable-5 (0.334) ✓ | glm-5.3 (0.650) |
| claude-sonnet-5 | claude | claude-opus-4-8 (0.217) ✓ | claude-opus-4-8 (0.376) ✓ | claude-opus-4-8 (0.728) ✓ |
| gemini-3.1-pro-preview | gemini | gemini-3.6-flash (0.213) ✓ | gemini-3.6-flash (0.369) ✓ | gemini-3.6-flash (0.790) ✓ |
| gemini-3.6-flash | gemini | gemini-3.7-flash (0.229) ✓ | gemini-3.7-flash (0.383) ✓ | gemini-3.7-flash (0.924) ✓ |
| gemini-3.7-flash | gemini | gemini-3.8-flash (0.232) ✓ | gemini-3.6-flash (0.383) ✓ | gemini-3.8-flash (0.936) ✓ |
| gemini-3.8-flash | gemini | gemini-3.7-flash (0.232) ✓ | gemini-3.7-flash (0.382) ✓ | gemini-3.7-flash (0.936) ✓ |
| glm-5.3 | glm | claude-fable-5 (0.212) | claude-fable-5 (0.363) | glm-5.3-flash (0.946) ✓ |
| glm-5.3-flash | glm | claude-fable-5 (0.224) | claude-fable-5 (0.374) | glm-5.3 (0.946) ✓ |
| gpt-5.5 | gpt | gpt-5.6-sol (0.214) ✓ | gpt-5.6-sol (0.366) ✓ | gpt-5.6-sol (0.829) ✓ |
| gpt-5.6-luna | gpt | gpt-5.6-sol (0.223) ✓ | gpt-5.6-sol (0.383) ✓ | gpt-5.6-terra (0.775) ✓ |
| gpt-5.6-sol | gpt | gpt-5.6-terra (0.228) ✓ | gpt-5.6-terra (0.386) ✓ | gpt-5.5 (0.829) ✓ |
| gpt-5.6-terra | gpt | gpt-5.6-sol (0.228) ✓ | gpt-5.6-sol (0.386) ✓ | gpt-5.6-sol (0.812) ✓ |
| gpt-6-astra | gpt | gpt-5.6-sol (0.220) ✓ | gpt-5.6-sol (0.369) ✓ | gpt-5.6-sol (0.704) ✓ |
| kimi-k3 | kimi | claude-fable-5 (0.187) | claude-fable-5 (0.325) | gpt-5.5 (0.565) |
| qwen3.8-max | qwen | gpt-5.6-luna (0.199) | gpt-5.6-luna (0.359) | gpt-5.5 (0.601) |

✓ = en yakın komşu aynı lab'dan.

### 5c. Aynı lab içi seçili çiftler: prompt içi uzunluk ilişkisi

Bütün 153 model çiftinde prompt içi uzunluk Spearman: min -0.21, max 0.16, ortalama -0.003.

| çift | kelime TF-IDF | karakter TF-IDF | stil korelasyonu | prompt içi uzunluk Spearman |
|---|---|---|---|---|
| gemini-3.6-flash ~ gemini-3.7-flash | 0.229 | 0.383 | 0.924 | -0.00 |
| gemini-3.7-flash ~ gemini-3.8-flash | 0.232 | 0.382 | 0.936 | 0.09 |
| gemini-3.6-flash ~ gemini-3.8-flash | 0.206 | 0.360 | 0.910 | -0.06 |
| gemini-3.1-pro-preview ~ gemini-3.8-flash | 0.192 | 0.346 | 0.655 | -0.05 |
| glm-5.3 ~ glm-5.3-flash | 0.200 | 0.346 | 0.946 | -0.07 |
| gpt-5.6-luna ~ gpt-5.6-sol | 0.223 | 0.383 | 0.754 | 0.16 |
| gpt-5.6-sol ~ gpt-5.6-terra | 0.228 | 0.386 | 0.812 | 0.08 |
| gpt-5.6-luna ~ gpt-5.6-terra | 0.216 | 0.376 | 0.775 | 0.09 |
| gpt-5.5 ~ gpt-6-astra | 0.186 | 0.335 | 0.643 | -0.09 |
| claude-opus-4-6 ~ claude-opus-4-8 | 0.207 | 0.368 | 0.439 | 0.05 |
| claude-opus-4-8 ~ claude-opus-5 | 0.191 | 0.330 | 0.407 | 0.02 |
| claude-opus-5 ~ claude-sonnet-5 | 0.173 | 0.311 | 0.325 | 0.06 |
| claude-fable-5 ~ claude-opus-5 | 0.192 | 0.334 | 0.549 | -0.02 |

### 6. Modele özgü kelimeler (log-odds, bilgilendirici Dirichlet önseli; Monroe vd. 2008)

Her model kendi metinleri ile diğer 17 modelin metinleri arasında karşılaştırıldı; z-skoru en yüksek 10 kelime. Her model aynı 149–150 prompt'a cevap verdiği için konu kelimeleri dengelenir.

| model | kelimeler (z) |
|---|---|
| claude-fable-5 | she (6.2), her (4.0), nothing (4.0), complete (4.0), insisting (3.7), shorter (3.7), told (3.6), zoom (3.5), conversation (3.5), everything (3.5) |
| claude-opus-4-6 | roughly (4.9), that (4.8), comfortable (4.5), most (4.3), deeply (4.0), percent (3.8), underlying (3.8), doesn't (3.7), emerges (3.7), traverse (3.7) |
| claude-opus-4-8 | you (6.7), quietly (5.2), we (5.0), glimpse (4.3), that (4.2), immensity (3.8), here (3.7), isn't (3.7), illuminate (3.7), retreat (3.6) |
| claude-opus-5 | the (8.5), is (7.9), he (7.2), hundred (6.2), never (5.7), it (5.6), that (5.6), so (5.5), thing (5.3), rumor (5.1) |
| claude-sonnet-5 | itself (5.2), it's (5.2), isn't (4.4), this (4.0), requires (4.0), minimal (4.0), entirely (3.8), atop (3.7), since (3.7), mid (3.7) |
| gemini-3.1-pro-preview | of (14.2), profound (9.1), our (7.9), absolute (7.3), the (7.2), this (6.7), biological (6.5), microscopic (6.3), ultimate (6.1), relentless (6.1) |
| gemini-3.6-flash | of (8.4), expanse (6.6), dynamic (6.5), void (6.2), breathtaking (6.0), mathematical (5.9), microscopic (5.6), an (5.6), into (5.6), delicate (5.5) |
| gemini-3.7-flash | an (13.5), of (9.6), profound (7.5), fundamentally (6.4), expanse (5.6), biological (5.5), these (5.3), intricate (5.3), primordial (5.1), perpetual (5.1) |
| gemini-3.8-flash | an (13.0), transforming (6.0), enduring (5.8), abyss (5.6), into (5.5), permanently (5.4), across (5.3), architecture (5.3), absolute (5.2), tempest (5.1) |
| glm-5.3 | itself (5.0), every (4.9), balls (4.7), now (4.6), wire (4.5), billion (4.3), believe (4.0), nothing (3.9), fragments (3.8), so (3.7) |
| glm-5.3-flash | overnight (4.9), scientists (4.0), never (3.9), lived (3.9), neighbor (3.8), organ (3.7), crucially (3.6), ago (3.6), she (3.5), knelt (3.4) |
| gpt-5.5 | speech (4.9), like (4.9), and (4.5), less (4.4), habits (4.4), is (4.1), not (3.8), seems (3.8), warm (3.7), heat (3.6) |
| gpt-5.6-luna | may (6.2), while (5.7), can (5.4), therefore (5.2), records (5.1), and (4.9), sometimes (4.7), gathers (4.5), making (4.4), become (4.3) |
| gpt-5.6-sol | and (5.4), or (4.5), travel (4.5), continually (4.4), can (4.4), limited (4.2), another (4.0), fled (4.0), combine (4.0), quarrel (4.0) |
| gpt-5.6-terra | may (5.8), crowded (5.4), can (4.5), entered (4.3), institutions (4.3), moon (4.0), strong (3.9), be (3.9), relationships (3.8), assembled (3.8) |
| gpt-6-astra | familiar (6.2), can (6.1), helps (5.3), could (5.2), helped (5.2), still (5.2), its (5.2), what (5.0), shelter (5.0), turn (4.6) |
| kimi-k3 | keeps (5.5), patience (5.2), bright (5.1), debt (4.5), before (4.4), after (4.3), steering (4.2), patiently (4.1), shining (4.1), becomes (4.1) |
| qwen3.8-max | turning (5.8), luminous (5.2), becomes (5.1), and (4.5), shape (4.5), lift (4.0), compassion (3.9), becoming (3.9), delayed (3.8), pressing (3.6) |

### 7. Örnek test dosyası

- Satır 3, sütunlar ['id', 'text']; kelime sayıları [48, 111, 149].

### 8. id sırası denetimi (reuse/ev/adversarial.order_check)

Ardışık satırların aynı sınıfta olma oranı ile şans düzeyi (sum p_c^2); |z| > 3 yapı demek.

| sıra | gecikme 1 (gözlenen / şans / z) | gecikme 2 | gecikme 10 |
|---|---|---|---|
| train.csv dosya sırası | 0.0582 / 0.0556 / +0.60 | 0.0545 / 0.0556 / -0.24 | 0.0614 / 0.0556 / +1.31 |
| id'ye göre sıralı | 0.0582 / 0.0556 / +0.60 | 0.0545 / 0.0556 / -0.24 | 0.0614 / 0.0556 / +1.31 |
| prompt_id'ye göre sıralı | 0.0030 / 0.0556 / -11.93 | 0.0096 / 0.0556 / -10.41 | 0.0316 / 0.0556 / -5.42 |

train.csv satırları id'ye göre sıralı mı: True; prompt_id'ye göre sıralı mı: False.

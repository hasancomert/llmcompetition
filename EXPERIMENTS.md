# EXPERIMENTS

Her satır gerçekten çalıştırılmış bir deney. Ortak katlar: `outputs/folds.csv`
(StratifiedGroupKFold(5, shuffle=True, random_state=42) / `prompt_id`, sha256 önek `647ff7926782e8eb`).
CV BA = havuzlanmış OOF balanced accuracy (2.699 satır); yanında kat başına BA ortalaması ± std; log loss havuzlanmış OOF.
Şans düzeyi BA = 1/18 = 0.0556. Tek bir OOF BA'nın standart hatası ≈0.01 (i.i.d. alt sınır; prompt kümelenmesiyle daha büyük). İki modelin farkı
`src/compare.py` (prompt üstünde eşli bootstrap) ile değerlendirilir. Public LB yalnız gönderilen deneylerde dolu.

## Okuma notları

- **Std:** "kat BA ort. ± std" sütunundaki std, 5 kat üzerinde popülasyon std'si (ddof=0). Örnek std (ddof=1) bunun
  ~1,12 katı.
- **v1 / v2 özellikler:**
  - #8–#13, #16–#24 v1 stil/tokenizer özellikleriyle koşuldu (#16 ve #17–#20 harmanları v1 stil/tokenizer üyelerini içerir).
  - Bağımsız kod denetimi iki hata buldu. Gemma'da `frac_single_char` bayt değil karakter sayıyordu (em-dash `—` ve `’`
    "tek karakterli token" sayılıyordu). `first_person_sg` regex'i cümle başındaki "My"ı kaçırıyor, "Type I"ı sayıyordu.
  - Düzeltilmiş özelliklerle koşular #25–#38 (`_v2`). Karar v2 satırlarına göre; v1 satırları kayıt için duruyor.
  - Sayılar v1 ile v2 arasında çok az değişti.
- **Süreler:** #7, #8 (415 s) ve #10 (192 s) koşarken doğrulama ajanlarının yeniden hesaplama işleri de aynı 4 çekirdeği
  kullanıyordu; bu süreler şişkin. LightGBM tek başına ~65–140 s (#21–#34).
- **Harman satırları** (#16–#20, #35–#38): Adım E'nin önizlemesi.
  - Ağırlıklar her seferinde diğer 4 katta seçildi (`hill_climb.nested_hill`, meta katlar = `folds.csv`). Satırdaki BA
    iç içe OOF skorudur.
  - 4 harman biçimi (tokenizer'lı/tokenizersız × ağırlık ölçütü) arasından en iyisini seçmek küçük bir seçim iyimserliği
    taşır.
- **LinearSVC (#4):** sıcaklık iç gruplu bölmeyle fit edildi. İç modeller dış katın vektörleştiricisini paylaşıyor
  (sözlük iç doğrulama metinlerini görmüş). Bu yüzden sıcaklık biraz keskin olabilir; yalnız log loss'u etkiler.
- **Kayıt denetimi:** bütün `oof_*.npy` dosyaları sklearn ile yeniden puanlandı ve tablodaki sayılarla birebir tuttu.
  Bundan sonraki her satır `src` özetini taşır; aynı etiketin üzerine yazmak engellenmiştir.

## Adım 2 özeti ve eşli karşılaştırmalar

Eşli karşılaştırma = aynı katlar, prompt'lar üstünde 2.000 tekrarlı bootstrap (`src/compare.py`), BA farkı ve %95 aralık.

| karşılaştırma | BA farkı [%95 aralık] | log loss | sonuç |
|---|---|---|---|
| char TF-IDF (#2) → char+word (#5) | +0.0163 [+0.0029, +0.0300] | 2.119 → 2.036 | kelime n-gram'ı ekle |
| char (#2) → transdüktif sözlük (#7) | −0.0022 [−0.0070, +0.0026] | 2.119 → 2.128 | kazanç yok, kat içi sözlük |
| char (#2) → charwb (#6) | −0.0207 [−0.0348, −0.0078] | 2.119 → 2.123 | kelimeler arası noktalama desenleri önemli (`a—b`, `, `) |
| stil LR (#25) → stil+tok LR (#26) | +0.0163 [+0.0004, +0.0311] | 2.060 → 2.028 | tokenizer küçük katkı |
| stil LGB 3 tohum (#29,31,33) → stil+tok LGB 3 tohum (#30,32,34) | +0.0022 [−0.0126, +0.0177] | 2.145 → 2.125 | BA'da belirsiz, log loss iyileşiyor |
| harman tok'suz → tok'lu, log loss ağırlıklı (#35 → #36) | −0.0030 [−0.0148, +0.0085] | 1.816 → 1.793 | BA'da belirsiz, log loss iyileşiyor |
| harman tok'suz → tok'lu, BA ağırlıklı (#37 → #38) | +0.0089 [−0.0022, +0.0200] | 1.897 → 1.856 | küçük, gürültü sınırında |
| en iyi tek (#5) → en iyi harman (#38) | +0.0452 [+0.0304, +0.0601] | 2.036 → 1.856 | aileler birbirini tamamlıyor |

Diğer gözlemler:
- Aynı özelliklerde LR, LightGBM'den iyi: stil LR 0.283 / 2.060, stil LightGBM 0.264–0.267 / 2.150–2.160 (v2, 3 tohum).
- LightGBM tohum gürültüsü ~0.01 BA: aynı ayarda v1 stil+tok 0.2652–0.2767, v2 0.2667–0.2782.
- Tokenizer kimliği TF-IDF'leri (#14 o200k, #15 Gemma) kelime TF-IDF'iyle aynı düzeyde. Harmanda ağırlık almadılar (#18).
- Yalnız tokenizer özellikleri tek başına zayıf: v2'de LR 0.180, LightGBM 0.163 (şans 0.056). En önemli özellikler
  Gemma oranları (`chars_per_tok_gemma`, `frac_single_char_gemma`), ardından o200k ve Kimi'nin tek-karakter token oranları.

## Adım 3 özeti (GPU üyeleri, prompt merkezleme, eşleştirme)

- **Prompt merkezleme** (her metnin özellik satırından, etiketsiz prompt kümesinin ortalaması çıkarılır; test kendi
  içinde kümelenir) en büyük kaldıraç oldu:

| üye | ham BA / log loss | merkezlenmiş BA / log loss |
|---|---|---|
| Qwen3-8B gömme, LR C 0.001 | 0.4045 / 1.735 | 0.5583 / 1.3239 |
| Qwen3-4B gömme, LR C 0.001 | 0.4063 / 1.750 | 0.5498 / 1.3538 |
| 4×Qwen3 lp + stil+tok, LR C 0.1 | 0.3279 / 1.991 | 0.4016 / 1.7484 |
| stil+tok, LR C 0.1 | 0.2993 / ~2.03 | 0.3782 / 1.8353 |
| char TF-IDF (SVC C 0.3; merkezlenmişte 50k sütun) | 0.3249 / 2.0606 | 0.3956 / 1.8215 |

- Tam OOF'ta emb/lp üyeleri bütün train metinlerinin kümeleriyle merkezlendi (prompt'lar katlara gruplu olduğu için
  doğrulama metinlerinin kümeleri çoğunlukla yalnız kendi katından). Test benzeri benzetim (eğitim katı bütün, doğrulama
  katı düşürülüp kendi içinde kümelenir) 8B için 0.53–0.55 verdi; yani tam OOF hafif iyimser.
- **v3 harman** (merkezlenmiş tfc + lpc + emb4c + emb8c, ağırlıklar .167/.167/.167/.5): iç içe BA 0.5931 / log loss
  1.2539. En iyi tek üyeye göre +0.0348 [+0.0215, +0.0482].
- **Test benzeri tam boru hattı** (`src/pipeline_sim.py`: model başına düşürme, test kendi içinde kümelenir ve
  merkezlenir, sonra eşleştirme): λ = 2'de hafif düşürme U(0.35, 1) 0.6162, sert U(0.2, 0.6) 0.5567. Aynı benzetimde
  gönderilmeye hazır v2 GPU harmanı 0.559 / 0.519.
- **İşe yaramayanlar:** LoRA probu (0.265), GLM-4-9B'yi eklemek (+0.0015, anlamsız), merkezlenmiş LR TF-IDF
  üyeleri (log loss'u yüksek; harmanda ağırlık 0).

## Deney tablosu

| no | tag | değişiklik | CV BA | kat BA ort. ± std | CV log loss | public LB | süre | not |
|---|---|---|---|---|---|---|---|---|
| 1 | tfidf_char_lr_C10 | TF-IDF char + LogReg C=10 | 0.3275 | 0.3274 ± 0.0168 | 2.2109 | | 151s |  |
| 2 | tfidf_char_lr_C30 | TF-IDF char + LogReg C=30 | 0.3308 | 0.3308 ± 0.0133 | 2.1191 | | 169s |  |
| 3 | tfidf_word_lr_C10 | TF-IDF word + LogReg C=10 | 0.3297 | 0.3297 ± 0.0138 | 2.2111 | | 27s |  |
| 4 | tfidf_char_svc_C0.3 | TF-IDF char + LinearSVC (+inner temperature) C=0.3 | 0.3249 | 0.3248 ± 0.0114 | 2.0606 | | 126s |  |
| 5 | tfidf_charword_lr_C30 | TF-IDF char+word + LogReg C=30 | 0.3471 | 0.3471 ± 0.0154 | 2.0359 | | 219s |  |
| 6 | tfidf_charwb_lr_C30 | TF-IDF charwb + LogReg C=30 | 0.3101 | 0.3100 ± 0.0059 | 2.1232 | | 213s |  |
| 7 | tfidf_char_lr_C30_td | TF-IDF char + LogReg C=30, vectorizer fit incl. val/test texts | 0.3286 | 0.3285 ± 0.0142 | 2.1278 | | 253s |  |
| 8 | lgb_style_s0 | LightGBM on style features (87 cols), rounds from inner grouped CV | 0.2678 | 0.2679 ± 0.0151 | 2.1583 | | 415s | rounds [122, 125, 128, 129, 118] |
| 9 | lgb_tok_s0 | LightGBM on tok features (57 cols), rounds from inner grouped CV | 0.1711 | 0.1711 ± 0.0177 | 2.6035 | | 79s | rounds [70, 79, 71, 77, 72] |
| 10 | lgb_style_tok_s0 | LightGBM on style+tok features (144 cols), rounds from inner grouped CV | 0.2763 | 0.2764 ± 0.0273 | 2.1397 | | 192s | rounds [134, 124, 121, 144, 114] |
| 11 | lr_style_C0.1 | QuantileTransformer + LogReg C=0.1 on style features (87 cols) | 0.2838 | 0.2838 ± 0.0212 | 2.0597 | | 10s |  |
| 12 | lr_style_tok_C0.1 | QuantileTransformer + LogReg C=0.1 on style+tok features (144 cols) | 0.2990 | 0.2990 ± 0.0218 | 2.0276 | | 23s |  |
| 13 | lr_tok_C0.1 | QuantileTransformer + LogReg C=0.1 on tok features (57 cols) | 0.1833 | 0.1833 ± 0.0157 | 2.5040 | | 17s |  |
| 14 | tfidf_tok_o200k_lr_C10 | TF-IDF tok_o200k + LogReg C=10 | 0.3286 | 0.3286 ± 0.0119 | 2.2627 | | 50s |  |
| 15 | tfidf_tok_gemma_lr_C10 | TF-IDF tok_gemma + LogReg C=10 | 0.3282 | 0.3282 ± 0.0202 | 2.2136 | | 49s |  |
| 16 | blend_notok | nested hill-climb blend (logprob, weights by logloss) of tfidf_charword_lr_C30, tfidf_char_svc_C0.3, tfidf_word_lr_C10, lr_style_C0.1, lgb_style_s0 | 0.3827 | 0.3827 ± 0.0150 | 1.8157 | | 1s | nested OOF (weights chosen on the other 4 folds); preview of step E |
| 17 | blend_tok | nested hill-climb blend (logprob, weights by logloss) of tfidf_charword_lr_C30, tfidf_char_svc_C0.3, tfidf_word_lr_C10, lr_style_tok_C0.1, lgb_style_tok_s0 | 0.3775 | 0.3775 ± 0.0198 | 1.7902 | | 1s | nested OOF (weights chosen on the other 4 folds); preview of step E |
| 18 | blend_tok_ids | nested hill-climb blend (logprob, weights by logloss) of tfidf_charword_lr_C30, tfidf_char_svc_C0.3, tfidf_word_lr_C10, lr_style_tok_C0.1, lgb_style_tok_s0, tfidf_tok_o200k_lr_C10, tfidf_tok_gemma_lr_C10 | 0.3775 | 0.3775 ± 0.0198 | 1.7902 | | 1s | nested OOF (weights chosen on the other 4 folds); preview of step E |
| 19 | blend_notok_ba | nested hill-climb blend (logprob, weights by ba) of tfidf_charword_lr_C30, tfidf_char_svc_C0.3, tfidf_word_lr_C10, lr_style_C0.1, lgb_style_s0 | 0.3841 | 0.3841 ± 0.0193 | 1.9165 | | 1s | nested OOF (weights chosen on the other 4 folds); preview of step E |
| 20 | blend_tok_ba | nested hill-climb blend (logprob, weights by ba) of tfidf_charword_lr_C30, tfidf_char_svc_C0.3, tfidf_word_lr_C10, lr_style_tok_C0.1, lgb_style_tok_s0 | 0.3912 | 0.3912 ± 0.0177 | 1.8614 | | 2s | nested OOF (weights chosen on the other 4 folds); preview of step E |
| 21 | lgb_style_s1 | LightGBM on style features (87 cols), rounds from inner grouped CV | 0.2664 | 0.2664 ± 0.0082 | 2.1579 | | 104s | rounds [118, 129, 116, 136, 119] |
| 22 | lgb_style_tok_s1 | LightGBM on style+tok features (144 cols), rounds from inner grouped CV | 0.2652 | 0.2652 ± 0.0117 | 2.1421 | | 142s | rounds [133, 130, 140, 134, 129] |
| 23 | lgb_style_s2 | LightGBM on style features (87 cols), rounds from inner grouped CV | 0.2686 | 0.2686 ± 0.0084 | 2.1540 | | 69s | rounds [123, 131, 116, 127, 121] |
| 24 | lgb_style_tok_s2 | LightGBM on style+tok features (144 cols), rounds from inner grouped CV | 0.2767 | 0.2767 ± 0.0215 | 2.1330 | | 136s | rounds [139, 128, 132, 139, 124] |
| 25 | lr_style_v2 | QuantileTransformer + LogReg C=0.1 on style features (87 cols) | 0.2830 | 0.2830 ± 0.0210 | 2.0597 | | 13s | src 6802457d4c72 |
| 26 | lr_style_tok_v2 | QuantileTransformer + LogReg C=0.1 on style+tok features (144 cols) | 0.2993 | 0.2993 ± 0.0210 | 2.0281 | | 28s | src 6802457d4c72 |
| 27 | lr_tok_v2 | QuantileTransformer + LogReg C=0.1 on tok features (57 cols) | 0.1800 | 0.1800 ± 0.0113 | 2.5179 | | 15s | src 6802457d4c72 |
| 28 | lgb_tok_v2_s0 | LightGBM on tok features (57 cols), rounds from inner grouped CV | 0.1634 | 0.1634 ± 0.0156 | 2.6156 | | 63s | rounds [70, 75, 72, 70, 67]; src 6802457d4c72 |
| 29 | lgb_style_v2_s0 | LightGBM on style features (87 cols), rounds from inner grouped CV | 0.2638 | 0.2638 ± 0.0130 | 2.1586 | | 70s | rounds [124, 120, 124, 131, 118]; src 6802457d4c72 |
| 30 | lgb_style_tok_v2_s0 | LightGBM on style+tok features (144 cols), rounds from inner grouped CV | 0.2708 | 0.2708 ± 0.0265 | 2.1445 | | 120s | rounds [131, 131, 130, 149, 119]; src 6802457d4c72 |
| 31 | lgb_style_v2_s1 | LightGBM on style features (87 cols), rounds from inner grouped CV | 0.2671 | 0.2671 ± 0.0106 | 2.1595 | | 65s | rounds [117, 135, 120, 130, 117]; src 6802457d4c72 |
| 32 | lgb_style_tok_v2_s1 | LightGBM on style+tok features (144 cols), rounds from inner grouped CV | 0.2667 | 0.2667 ± 0.0170 | 2.1409 | | 117s | rounds [130, 129, 141, 142, 123]; src 6802457d4c72 |
| 33 | lgb_style_v2_s2 | LightGBM on style features (87 cols), rounds from inner grouped CV | 0.2671 | 0.2672 ± 0.0155 | 2.1502 | | 68s | rounds [126, 126, 121, 125, 120]; src 6802457d4c72 |
| 34 | lgb_style_tok_v2_s2 | LightGBM on style+tok features (144 cols), rounds from inner grouped CV | 0.2782 | 0.2782 ± 0.0229 | 2.1325 | | 115s | rounds [138, 130, 123, 141, 122]; src 6802457d4c72 |
| 35 | blend_v2_notok_logloss | nested hill-climb blend (logprob, weights by logloss) of tfidf_charword_lr_C30, tfidf_char_svc_C0.3, tfidf_word_lr_C10, lr_style_v2, lgb_style_v2_s0,lgb_style_v2_s1,lgb_style_v2_s2 | 0.3819 | 0.3819 ± 0.0162 | 1.8155 | | 2s | nested OOF (weights chosen on the other 4 folds); preview of step E; src 6802457d4c72 |
| 36 | blend_v2_tok_logloss | nested hill-climb blend (logprob, weights by logloss) of tfidf_charword_lr_C30, tfidf_char_svc_C0.3, tfidf_word_lr_C10, lr_style_tok_v2, lgb_style_tok_v2_s0,lgb_style_tok_v2_s1,lgb_style_tok_v2_s2 | 0.3789 | 0.3789 ± 0.0178 | 1.7928 | | 1s | nested OOF (weights chosen on the other 4 folds); preview of step E; src 6802457d4c72 |
| 37 | blend_v2_notok_ba | nested hill-climb blend (logprob, weights by ba) of tfidf_charword_lr_C30, tfidf_char_svc_C0.3, tfidf_word_lr_C10, lr_style_v2, lgb_style_v2_s0,lgb_style_v2_s1,lgb_style_v2_s2 | 0.3834 | 0.3834 ± 0.0217 | 1.8973 | | 1s | nested OOF (weights chosen on the other 4 folds); preview of step E; src 6802457d4c72 |
| 38 | blend_v2_tok_ba | nested hill-climb blend (logprob, weights by ba) of tfidf_charword_lr_C30, tfidf_char_svc_C0.3, tfidf_word_lr_C10, lr_style_tok_v2, lgb_style_tok_v2_s0,lgb_style_tok_v2_s1,lgb_style_tok_v2_s2 | 0.3923 | 0.3923 ± 0.0203 | 1.8560 | | 1s | nested OOF (weights chosen on the other 4 folds); preview of step E; src 6802457d4c72 |
| 39 | match_blend_v2_tok_ba_full_v1 | prompt clustering v1 (LSA-128, thr 0.5) + soft Hungarian on blend_v2_tok_ba, scenario full; lambda nested | 0.4986 | 0.4986 ± 0.0000 | nan | | 1s | argmax BA 0.3923 on the same rows; ± = std over 1 drop simulations; log loss n/a (labels); src 89638098d089 |
| 40 | match_blend_v2_tok_ba_drop035-10_v1 | prompt clustering v1 (LSA-128, thr 0.5) + soft Hungarian on blend_v2_tok_ba, scenario drop 0.35-1.0; lambda nested | 0.4478 | 0.4478 ± 0.0094 | nan | | 4s | argmax BA 0.3961 on the same rows; ± = std over 5 drop simulations; log loss n/a (labels); src 89638098d089 |
| 41 | match_blend_v2_tok_ba_drop02-06_v1 | prompt clustering v1 (LSA-128, thr 0.5) + soft Hungarian on blend_v2_tok_ba, scenario drop 0.2-0.6; lambda nested | 0.4309 | 0.4309 ± 0.0086 | nan | | 6s | argmax BA 0.4017 on the same rows; ± = std over 5 drop simulations; log loss n/a (labels); src 89638098d089 |
| 42 | match_blend_v2_tok_ba_full_v2 | prompt clustering v2 (raw TF-IDF, max_df 40 docs, thr 0.92) + soft Hungarian on blend_v2_tok_ba, scenario full; lambda nested | 0.4993 | 0.4993 ± 0.0000 | nan | | 1s | argmax BA 0.3923 on the same rows; ± = std over 1 drop simulations; log loss n/a (labels); src 569948b7656c |
| 43 | match_blend_v2_tok_ba_drop035-10_v2 | prompt clustering v2 (raw TF-IDF, max_df 40 docs, thr 0.92) + soft Hungarian on blend_v2_tok_ba, scenario drop 0.35-1.0; lambda nested | 0.4448 | 0.4448 ± 0.0086 | nan | | 3s | argmax BA 0.3961 on the same rows; ± = std over 5 drop simulations; log loss n/a (labels); src 569948b7656c |
| 44 | match_blend_v2_tok_ba_drop02-06_v2 | prompt clustering v2 (raw TF-IDF, max_df 40 docs, thr 0.92) + soft Hungarian on blend_v2_tok_ba, scenario drop 0.2-0.6; lambda nested | 0.4311 | 0.4311 ± 0.0108 | nan | | 4s | argmax BA 0.4017 on the same rows; ± = std over 5 drop simulations; log loss n/a (labels); src 569948b7656c |
| 45 | emb_qwen3_0.6b_base_C0.05 | frozen LM embeddings (qwen3_0.6b_base, 4096 cols), LogReg C=0.05 | 0.3634 | 0.3634 ± 0.0221 | 2.4315 | | 81s | src 1e28e4e3c715 |
| 46 | emb_qwen3_1.7b_base_C0.05 | frozen LM embeddings (qwen3_1.7b_base, 8192 cols), LogReg C=0.05 | 0.3752 | 0.3752 ± 0.0291 | 2.3756 | | 100s | src 1e28e4e3c715 |
| 47 | emb_qwen3_4b_base_C0.05 | frozen LM embeddings (qwen3_4b_base, 10240 cols), LogReg C=0.05 | 0.3938 | 0.3938 ± 0.0278 | 2.3116 | | 96s | src 1e28e4e3c715 |
| 48 | emb_qwen3_8b_base_C0.05 | frozen LM embeddings (qwen3_8b_base, 16384 cols), LogReg C=0.05 | 0.3926 | 0.3926 ± 0.0305 | 2.3074 | | 104s | src 1e28e4e3c715 |
| 49 | lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base_C0.1 | LM token log-prob stats (qwen3_0.6b_base, qwen3_1.7b_base, qwen3_4b_base, qwen3_8b_base, 52 cols), LogReg C=0.1 | 0.1678 | 0.1678 ± 0.0079 | 2.6280 | | 8s | src 1e28e4e3c715 |
| 50 | lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base_style_C0.1 | LM token log-prob stats (qwen3_0.6b_base, qwen3_1.7b_base, qwen3_4b_base, qwen3_8b_base, 196 cols) + style+tok v2, LogReg C=0.1 | 0.3279 | 0.3279 ± 0.0151 | 1.9911 | | 32s | src 1e28e4e3c715 |
| 51 | emb_qwen3_4b_base_C0.01 | frozen LM embeddings (qwen3_4b_base, 10240 cols), LogReg C=0.01 | 0.3949 | 0.3949 ± 0.0294 | 1.9791 | | 132s | src 8e9e94a5ebb4 |
| 52 | emb_qwen3_8b_base_C0.01 | frozen LM embeddings (qwen3_8b_base, 16384 cols), LogReg C=0.01 | 0.3967 | 0.3967 ± 0.0305 | 1.9967 | | 155s | src 6a24e4a02b9e |
| 53 | emb_qwen3_4b_base_C0.003 | frozen LM embeddings (qwen3_4b_base, 10240 cols), LogReg C=0.003 | 0.3956 | 0.3956 ± 0.0310 | 1.8024 | | 143s | src 6a24e4a02b9e |
| 54 | emb_qwen3_8b_base_C0.003 | frozen LM embeddings (qwen3_8b_base, 16384 cols), LogReg C=0.003 | 0.4008 | 0.4008 ± 0.0295 | 1.8206 | | 178s | src 6a24e4a02b9e |
| 55 | blend_v3_emb_logloss | nested hill-climb blend (logprob, weights by logloss) of tfidf_charword_lr_C30, tfidf_char_svc_C0.3, lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base_style_C0.1, emb_qwen3_4b_base_C0.003, emb_qwen3_8b_base_C0.003 | 0.4430 | 0.4430 ± 0.0222 | 1.5775 | | 2s | nested OOF (weights chosen on the other 4 folds); preview of step E; src 6a24e4a02b9e |
| 56 | blend_v3_emb_ba | nested hill-climb blend (logprob, weights by ba) of tfidf_charword_lr_C30, tfidf_char_svc_C0.3, lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base_style_C0.1, emb_qwen3_4b_base_C0.003, emb_qwen3_8b_base_C0.003 | 0.4363 | 0.4363 ± 0.0241 | 1.5980 | | 1s | nested OOF (weights chosen on the other 4 folds); preview of step E; src 6a24e4a02b9e |
| 57 | match_blend_v3_emb_logloss_full | prompt clustering (thr 0.92) + soft Hungarian on blend_v3_emb_logloss, scenario full; lambda nested | 0.6423 | 0.6423 ± 0.0000 | nan | | 2s | argmax BA 0.4430 on the same rows; ± = std over 1 drop simulations; log loss n/a (labels); src 6a24e4a02b9e |
| 58 | match_blend_v3_emb_logloss_drop035-10 | prompt clustering (thr 0.92) + soft Hungarian on blend_v3_emb_logloss, scenario drop 0.35-1.0; lambda nested | 0.5594 | 0.5594 ± 0.0120 | nan | | 3s | argmax BA 0.4406 on the same rows; ± = std over 5 drop simulations; log loss n/a (labels); src 6a24e4a02b9e |
| 59 | match_blend_v3_emb_logloss_drop02-06 | prompt clustering (thr 0.92) + soft Hungarian on blend_v3_emb_logloss, scenario drop 0.2-0.6; lambda nested | 0.5194 | 0.5194 ± 0.0036 | nan | | 4s | argmax BA 0.4469 on the same rows; ± = std over 5 drop simulations; log loss n/a (labels); src 6a24e4a02b9e |
| 60 | emb_qwen3_4b_base_C0.001 | frozen LM embeddings (qwen3_4b_base, 10240 cols), LogReg C=0.001 | 0.4063 | 0.4063 ± 0.0349 | 1.7498 | | 118s | src 2bacb7ef9c88 |
| 61 | emb_qwen3_8b_base_C0.001 | frozen LM embeddings (qwen3_8b_base, 16384 cols), LogReg C=0.001 | 0.4045 | 0.4045 ± 0.0314 | 1.7353 | | 325s | src bdb8fc10231c |
| 62 | emb_qwen3_0.6b_base_C0.003 | frozen LM embeddings (qwen3_0.6b_base, 4096 cols), LogReg C=0.003 | 0.3752 | 0.3752 ± 0.0165 | 1.8348 | | 144s | src bdb8fc10231c |
| 63 | emb_qwen3_1.7b_base_C0.003 | frozen LM embeddings (qwen3_1.7b_base, 8192 cols), LogReg C=0.003 | 0.3841 | 0.3841 ± 0.0321 | 1.8391 | | 170s | src bdb8fc10231c |
| 64 | emb_qwen3_4b_base_C0.0003 | frozen LM embeddings (qwen3_4b_base, 10240 cols), LogReg C=0.0003 | 0.3889 | 0.3889 ± 0.0260 | 1.8812 | | 34s | src bdb8fc10231c |
| 65 | emb_qwen3_8b_base_C0.0003 | frozen LM embeddings (qwen3_8b_base, 16384 cols), LogReg C=0.0003 | 0.3978 | 0.3978 ± 0.0276 | 1.7985 | | 63s | src bdb8fc10231c |
| 66 | emb_qwen3_4b_base_ctr_C0.001 | frozen LM embeddings (qwen3_4b_base, 10240 cols) + prompt centering, LogReg C=0.001 | 0.5498 | 0.5497 ± 0.0306 | 1.3538 | | 136s | centered by 220 unsupervised clusters (thr 0.92); src db38de9c2a96 |
| 67 | emb_qwen3_8b_base_ctr_C0.001 | frozen LM embeddings (qwen3_8b_base, 16384 cols) + prompt centering, LogReg C=0.001 | 0.5583 | 0.5582 ± 0.0272 | 1.3239 | | 130s | centered by 220 unsupervised clusters (thr 0.92); src db38de9c2a96 |
| 68 | emb_qwen3_1.7b_base_ctr_C0.001 | frozen LM embeddings (qwen3_1.7b_base, 8192 cols) + prompt centering, LogReg C=0.001 | 0.5257 | 0.5256 ± 0.0237 | 1.4118 | | 106s | centered by 220 unsupervised clusters (thr 0.92); src db38de9c2a96 |
| 69 | emb_qwen3_0.6b_base_ctr_C0.001 | frozen LM embeddings (qwen3_0.6b_base, 4096 cols) + prompt centering, LogReg C=0.001 | 0.5242 | 0.5241 ± 0.0217 | 1.4742 | | 73s | centered by 220 unsupervised clusters (thr 0.92); src db38de9c2a96 |
| 70 | emb_qwen3_4b_base_ctr_C0.003 | frozen LM embeddings (qwen3_4b_base, 10240 cols) + prompt centering, LogReg C=0.003 | 0.5475 | 0.5475 ± 0.0294 | 1.3189 | | 131s | centered by 220 unsupervised clusters (thr 0.92); src db38de9c2a96 |
| 71 | emb_qwen3_8b_base_ctr_C0.003 | frozen LM embeddings (qwen3_8b_base, 16384 cols) + prompt centering, LogReg C=0.003 | 0.5560 | 0.5560 ± 0.0246 | 1.3021 | | 121s | centered by 220 unsupervised clusters (thr 0.92); src db38de9c2a96 |
| 72 | lora_qwen3_1.7b_probe | LoRA seq-cls fine-tune, Qwen3-1.7B-base (r16, lr 1e-4, 2 epochs, eff. batch 16, max_len 384), Kaggle T4, folds 0-1 only | 0.2651 | 0.2652 ± 0.0144 | 1.9803 | | 2049s | NOT useful at this config: same 2 folds raw frozen 1.7B emb C0.003 = 0.4159 / 1.7645, centered 1.7B = 0.5465; train loss still falling (3.42 -> 2.00, ln18 = 2.89) = underfit on ~2.2k rows; ~1000s/fold/2 epochs; 2-fold BA/ll, not comparable to 5-fold rows; src db38de9c2a96 |
| 73 | psim_ctr_v1_keep0.35-1.0 | pipeline sim, keep U(0.35,1.0): raw lp (w .152) + centered emb4c (C .001, w .271) + centered emb8c (C .001, w .576) [svc weight 0], clustering thr .92 + matching | 0.6054 | 0.6054 ± 0.0042 | nan | | 1250s | 3 sims, mean n 1935; members emb4c 0.5419, emb8c 0.5444, lp 0.3235; argmax blend 0.5558 / ll 1.2998; matched by lambda 0:0.5558, 1:0.5989, 2:0.6054, 4:0.6052, 8:0.6049 (lambda 2 picked on these sims); weights from nested logloss hill-climb on full OOF; same protocol for the v3 GPU blend gave 0.559; src 1cd23a2122ac |
| 74 | psim_ctr_v1_keep0.2-0.6 | pipeline sim, keep U(0.2,0.6): raw lp (w .152) + centered emb4c (C .001, w .271) + centered emb8c (C .001, w .576) [svc weight 0], clustering thr .92 + matching | 0.5538 | 0.5538 ± 0.0071 | nan | | 1250s | 3 sims, mean n 1154; members emb4c 0.5108, emb8c 0.5159, lp 0.3291; argmax blend 0.5364 / ll 1.3618; matched by lambda 0:0.5364, 1:0.5539, 2:0.5538, 4:0.5517, 8:0.5517; same protocol for the v3 GPU blend gave 0.519; src 1cd23a2122ac |
| 75 | lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base_ctr_stylec_C0.1 | LM token log-prob stats (qwen3_0.6b_base, qwen3_1.7b_base, qwen3_4b_base, qwen3_8b_base, 196 cols) + prompt centering + style+tok v2 (centered too), LogReg C=0.1 | 0.4016 | 0.4016 ± 0.0240 | 1.7484 | | 38s | centered by 220 unsupervised clusters (thr 0.92); src 1cd23a2122ac |
| 76 | lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base_ctr_style_C0.1 | LM token log-prob stats (qwen3_0.6b_base, qwen3_1.7b_base, qwen3_4b_base, qwen3_8b_base, 196 cols) + prompt centering + style+tok v2 (not centered), LogReg C=0.1 | 0.3590 | 0.3590 ± 0.0137 | 1.8778 | | 27s | centered by 220 unsupervised clusters (thr 0.92); src 1cd23a2122ac |
| 77 | style+tok_ctr_C0.1 | style+tok v2 features (144 cols) + prompt centering, LogReg C=0.1 | 0.3782 | 0.3782 ± 0.0066 | 1.8353 | | 20s | centered by 220 unsupervised clusters (thr 0.92); src a4011b771cb6 |
| 78 | tfidf_char_ctr_mf50k_lr_C10 | TF-IDF char (max_features 50000) + prompt centering (train-fold texts and held-out texts clustered separately, thr 0.92) + LogReg C=10 | 0.4079 | 0.4079 ± 0.0232 | 2.0820 | | 123s | src 45e8dea1a02e |
| 79 | tfidf_char_ctr_mf50k_lr_C30 | TF-IDF char (max_features 50000) + prompt centering (train-fold texts and held-out texts clustered separately, thr 0.92) + LogReg C=30 | 0.4045 | 0.4045 ± 0.0222 | 1.9632 | | 139s | src 45e8dea1a02e |
| 80 | tfidf_char_ctr_mf50k_lr_C3 | TF-IDF char (max_features 50000) + prompt centering (train-fold texts and held-out texts clustered separately, thr 0.92) + LogReg C=3 | 0.4123 | 0.4123 ± 0.0232 | 2.2958 | | 107s | src 45e8dea1a02e |
| 81 | tfidf_char_ctr_mf50k_svc_C0.3 | TF-IDF char (max_features 50000) + prompt centering (train-fold texts and held-out texts clustered separately, thr 0.92) + LinearSVC (+inner temperature) C=0.3 | 0.3956 | 0.3956 ± 0.0169 | 1.8215 | | 452s | src 25ac70dba3c1 |
| 82 | tfidf_char_ctr_mf50k_lr_C100 | TF-IDF char (max_features 50000) + prompt centering (train-fold texts and held-out texts clustered separately, thr 0.92) + LogReg C=100 | 0.4027 | 0.4027 ± 0.0205 | 1.8804 | | 137s | src 25ac70dba3c1 |
| 83 | tfidf_charword_ctr_mf40k_lr_C10 | TF-IDF char+word (max_features 40000) + prompt centering (train-fold texts and held-out texts clustered separately, thr 0.92) + LogReg C=10 | 0.4234 | 0.4234 ± 0.0209 | 2.0010 | | 146s | src 25ac70dba3c1 |
| 84 | psim_ctr_v3_keep0.35-1.0 | pipeline sim, keep U(0.35,1.0): centered tfc (char TF-IDF 50k + LinearSVC C .3, w .167) + centered lpc (4x Qwen3 lp + style+tok, C .1, w .167) + centered emb4c (w .167) + emb8c (w .5), clustering thr .92 + matching | 0.6162 | 0.6162 ± 0.0060 | nan | | 1500s | 3 sims (same drops as psim_ctr_v1, paired): members tfc 0.3892, lpc 0.4017, emb4c 0.5419, emb8c 0.5444; argmax blend 0.5765 / ll 1.2678 (v1 0.5558); matched by lambda 0:0.5765, 1:0.6123, 2:0.6162, 4:0.6155, 8:0.6152 (v1 at lambda 2: 0.6054); src bf5f06d10438 |
| 85 | psim_ctr_v3_keep0.2-0.6 | pipeline sim, keep U(0.2,0.6): centered tfc (char TF-IDF 50k + LinearSVC C .3, w .167) + centered lpc (4x Qwen3 lp + style+tok, C .1, w .167) + centered emb4c (w .167) + emb8c (w .5), clustering thr .92 + matching | 0.5567 | 0.5567 ± 0.0084 | nan | | 1500s | 3 sims (same drops as psim_ctr_v1, paired): members tfc 0.3849, lpc 0.3750, emb4c 0.5108, emb8c 0.5159; argmax blend 0.5426 / ll 1.3405 (v1 0.5364); matched by lambda 0:0.5426, 1:0.5597, 2:0.5567, 4:0.5545, 8:0.5533 (reported: lambda 2, the submission setting; v1 at lambda 2: 0.5538); src bf5f06d10438 |
| 86 | emb_glm4_9b_ctr_C0.001 | frozen LM embeddings (glm4_9b, 16384 cols) + prompt centering, LogReg C=0.001 | 0.5242 | 0.5241 ± 0.0220 | 1.3991 | | 124s | centered by 220 unsupervised clusters (thr 0.92); src bf5f06d10438 |
| 87 | emb_glm4_9b_C0.001 | frozen LM embeddings (glm4_9b, 16384 cols), LogReg C=0.001 | 0.3715 | 0.3715 ± 0.0161 | 1.8642 | | 169s | src bf5f06d10438 |
| 88 | lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base+glm4_9b_ctr_stylec_C0.1 | LM token log-prob stats (qwen3_0.6b_base, qwen3_1.7b_base, qwen3_4b_base, qwen3_8b_base, glm4_9b, 209 cols) + prompt centering + style+tok v2 (centered too), LogReg C=0.1 | 0.4108 | 0.4109 ± 0.0217 | 1.7461 | | 32s | centered by 220 unsupervised clusters (thr 0.92); src bf5f06d10438 |
| 89 | blend_ctr_v3_logloss | nested hill-climb blend (logprob, weights by logloss) of tfidf_char_ctr_mf50k_svc_C0.3, lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base_ctr_stylec_C0.1, emb_qwen3_4b_base_ctr_C0.001, emb_qwen3_8b_base_ctr_C0.001 | 0.5931 | 0.5931 ± 0.0271 | 1.2539 | | 1s | nested OOF (weights chosen on the other 4 folds); preview of step E; src bf5f06d10438 |
| 90 | blend_ctr_v3glm_logloss | nested hill-climb blend (logprob, weights by logloss) of tfidf_char_ctr_mf50k_svc_C0.3, lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base+glm4_9b_ctr_stylec_C0.1, emb_qwen3_4b_base_ctr_C0.001, emb_qwen3_8b_base_ctr_C0.001, emb_glm4_9b_ctr_C0.001 | 0.5946 | 0.5946 ± 0.0222 | 1.2504 | | 1s | nested OOF (weights chosen on the other 4 folds); preview of step E; src bf5f06d10438 |
| 91 | e2e_submit_gpu_v3_fold0 | END-TO-END check of submission_pipeline_gpu.py v3 (tests/e2e_gpu_pipeline.py): train = folds 1-4, fake test = fold 0 (id/text only), precomputed LM features; score of the written submission.parquet | 0.6820 | 0.6820 ± 0.0000 | nan | | 1000s | fold 0 only, full structure 539 rows: 0.6820; with keep U(0.35,1.0) drops (351 rows): 0.6765; in-kernel 4-fold OOF blend 0.5639, matched 0.6528; code path check, not a CV estimate; src bf5f06d10438 |
| 92 | blend_ctr_v3glm vs v3 | adding centered GLM-4-9B-0414 embeddings (emb 0.5242) and GLM lp stats to the v3 blend | 0.5946 | 0.5946 ± 0.0222 | 1.2504 | | 2s | NOT adopted: paired vs blend_ctr_v3_logloss (0.5931 / 1.2539) diff +0.0015, 95% CI [-0.0056, +0.0089]; costs a 9B extraction in the notebook; src bf5f06d10438 |

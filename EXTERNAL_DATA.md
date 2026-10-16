# Dış veri

Kural: yalnız ücretsiz ve herkese açık veri, yalnız eğitimde. Validation her zaman yarışma verisinin `folds.csv`
katlarıdır ve bir dış verinin faydası aynı katlarda eşli olarak gösterilmeden kullanılmaz. Ücretli API ile veri
üretilmez.

| kaynak | lisans | ne için | CV etkisi | durum |
|---|---|---|---|---|
| (yok) | | | | Adım 2 sonuna kadar dış veri kullanılmadı. |
| Kaggle `olegzholobov/name-that-llm-response-corpus` (snapshot 15 Ekim 2026), dosya `external_short_wide.parquet`: 18 modelin herkese açık kaynaklardan toplanmış 40–250 kelimelik cevapları (24.025 hücre) | Dataset "Other": kaynak başına karışık lisans. Apache-2.0, MIT, CC-BY-4.0 olanlar var; eqbench, artificialanalysis, glm_flash_analysis gibi bazılarının lisansı "unknown". Birçoğunda model çıktıları ayrıca sağlayıcı şartlarına tabi. Kaynak listesi: `data/external/ntl_corpus/source_catalog.csv` | Yalnız eğitim (TF-IDF üyesine ek eğitim satırı) | **Zararlı:** char+word TF-IDF LR 0.3471 → 0.2845, eşli −0.063 [−0.077, −0.047]. Öncül düzeltmesiyle 0.2874 (`src/ext_tfidf_probe.py`) | **Kullanılmıyor.** Hiçbir gönderimde yok. |

Dış derlem notları (16 Ekim):
- Biçim yarışma metinlerinden çok farklı: metinlerin %94'ünde satır sonu, %55'inde markdown işareti var. Deneyde metinler
  düzleştirildi (satır sonları boşluğa çevrildi, markdown işaretleri silindi), 40–250 kelimeye yeniden süzüldü ve
  tekilleştirildi; 23.658 satır kaldı.
- Sınıflar çok dengesiz: claude-opus-4-6 4.767, Gemini flash modelleri 71–83.
- Derlem yazarına göre 2.699 train cevabı ve 3 örnek test satırıyla birebir ya da normalize edilmiş çakışma yok.
- Dosyalar `data/external/` altında; `data/` git'e girmez.


### Dış derlemin ikinci değerlendirmesi (16 Ekim, dal `featherless-extra-data`)

Kullanıcının adımlarıyla yeniden değerlendirildi. Yalnız `external_short_wide.parquet` kullanıldı; `external_all_wide` ile
birleştirilmedi, `quarantine_luna_*` dışarıda. Kod: `src/ext_corpus.py` (yükleme, temizlik, lisans sınıfı, profil),
`src/ext_leak.py` (sızıntı), `src/ext_transfer.py` (transfer), `src/extdata.py` (bayrak), `src/ext_blend.py` (CV).
Çıktılar `data/external/ntl_corpus/` altında (git'e girmez).

- **Sayılar:** 24.025 hücre. Temizlik, 40–250 kelime süzgeci ve tekilleştirmeden sonra 23.658 satır kaldı.
  - Zayıf sınıflar: gemini-3.6-flash 83, gemini-3.7-flash 77, gemini-3.8-flash 71, glm-5.3 603, gpt-6-astra 722,
    qwen3.8-max 799, claude-opus-5 818.
  - En kalabalık sınıflar: claude-opus-4-6 4.767, gpt-5.6-sol 3.246, gpt-5.6-luna 2.259.
- **Uzunluk:** dış veride medyan 155 kelime (q05–q95: 49–238), resmi train'de 101 kelime (44–224).
- **Biçim:** dış metinlerin ancak %0–10'u ham haliyle tek düz paragraf (Gemini flash'larda %30–54).
- **Lisans sınıfları** (kaynak kataloğuna göre; derlem hiçbir şeyi yeniden lisanslamıyor):
  - açık (Apache-2.0 / MIT / CC-BY-4.0): 3.654 satır.
  - açık depo lisansı ama çıktılar sağlayıcı şartlarına tabi: 10.525 satır (speechmap, petergpt, woog, novigens,
    periospot).
  - **lisansı bilinmeyen ya da belirsiz: 9.479 satır.** Bunlar: eqbench4 7.221, glm_flash_analysis 832,
    Crownelius 1.096 (üst kaynak lisansı belirsiz), artificialanalysis 169, PawanKrd 105, copilot_model_value_benchmark
    51, eqbench longform 5.
  - Kısa görünümde kısıtlayıcı lisanslı (ör. GPL) kaynak yok. RISE (yalnız yazılım GPL-3.0) ve Postman APIFlow
    (çıktı kullanımı koşullu) yalnız tam görünümde var.
- **Sızıntı (konu benzerliği):** resmi cevaplarla kalibre edilen kapsayıcı eşik 0.162 (aynı soruya verilen resmi
  cevapların %99'u bu eşiği geçiyor).
  - 37 dış prompt grubunun 51 satırı ~20 resmi prompt_id ile eşleşti. Eşleşmelerin çoğu angrygiraffe (Monty Hall,
    Gödel, Fermi paradoksu, entropi gibi aynı konular); geri kalanı woog, Crownelius ve speechmap.
  - Bu satırlar, eşleştikleri prompt_id doğrulamadayken o fold'un eğitiminden çıkarılıyor.
  - Dış verinin geneli resmi konulardan uzak: en iyi benzerliğin medyanı 0.03.


Tokenizer dosyaları veri değil, ön-eğitimli varlık sayılır; kaynakları ve lisansları `MODELS_LICENSES.md`'de.

## Featherless ile üretilen etiketli metinler (dal `featherless-extra-data`; 16 Ekim)

- **Kaynak:** Featherless API (`https://api.featherless.ai/v1`, ücretli; iş kolu bütçesi 8 $). Kullanıcı bu iş kolunu açıkça
  istedi; eski "ücretli API ile veri üretme" kuralının bu iş kolu için istisnası.
- **Modeller (etiketin birebir aynısı):**

| model | etiket | lisans |
|---|---|---|
| `zai-org/GLM-5.3` | glm-5.3 | "GLM-5.3 License", MIT benzeri |
| `zai-org/GLM-5.3-Flash` | glm-5.3-flash | MIT |
| `moonshotai/Kimi-K3` | kimi-k3 | "Kimi K3 License", MIT benzeri |
| `Qwen/Qwen3-32B` | yalnız prompt kurgulama | Apache-2.0 |

  - GLM-5.3 lisansının şartı: geliri 10 milyar $'ı aşan "Model as a Service" işletmeleri için güvenlik incelemesi.
  - Kimi-K3 lisansının şartları: 20 milyon $ MaaS geliri ya da 100 milyon MAU üstü üründe isim gösterme.
- **Dosyalar** (`data/external/`, git'e girmez, hiçbir yere public konmaz):
  - `reconstructed_prompts.csv`: yarışma verisinden türedi.
  - `featherless/gen_<koşu>.jsonl`.
  - `featherless/costs.jsonl`.
  - `featherless/models.json`.
- **Her satırda:** model_id, etiket, örnekleme ayarları, tarih, prompt kaynağı, hedef kelime, bitiş nedeni, token ve maliyet.
- **Durum:** yalnız uyum testi (EXPERIMENTS.md). Hiçbir üyede, blend'de ya da gönderimde kullanılmadı. Dış veri kuralı
  netleşene kadar bu veriyle eğitilmiş model final submission'a konmayacak; pipeline'a eklenirse bayrakla (varsayılan
  kapalı) açılacak.
- **Kurallar açısından not:** prompt kurgulama ve üretim için yarışma train metinlerinden türeyen içerik Featherless'a
  gönderildi (150 prompt × 4 cevap ve kurulan prompt'lar). Yarışma kurallarındaki veri paylaşımı maddesi açısından
  değerlendirilmeli.

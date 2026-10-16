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

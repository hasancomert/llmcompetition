# Dış veri

Kural: yalnız ücretsiz ve herkese açık veri, yalnız eğitimde. Validation her zaman yarışma verisinin `folds.csv`
katlarıdır ve bir dış verinin faydası aynı katlarda eşli olarak gösterilmeden kullanılmaz. Ücretli API ile veri
üretilmez.

| kaynak | lisans | ne için | CV etkisi | durum |
|---|---|---|---|---|
| (yok) | | | | Adım 2 sonuna kadar dış veri kullanılmadı. |
| Kaggle `olegzholobov/name-that-llm-response-corpus` (snapshot 15 Ekim 2026), dosya `external_short_wide.parquet`: 18 modelin herkese açık kaynaklardan toplanmış 40–250 kelimelik cevapları (24.025 hücre) | Dataset "Other": kaynak başına karışık lisans. Apache-2.0, MIT, CC-BY-4.0 olanlar var; eqbench, artificialanalysis, glm_flash_analysis gibi bazılarının lisansı "unknown". Birçoğunda model çıktıları ayrıca sağlayıcı şartlarına tabi. Kaynak listesi: `data/external/ntl_corpus/source_catalog.csv` | Yalnız eğitim (TF-IDF üyesine ek eğitim satırı) | ölçülüyor (`src/ext_tfidf_probe.py`) | **Yalnız CV deneyi; hiçbir gönderimde kullanılmadı.** Lisans karışık olduğu için gönderimde kullanmadan önce kullanıcı onayı istenecek; kullanılırsa yalnız açık lisanslı kaynaklarla sınırlanacak. |

Dış derlem notları (16 Ekim):
- Biçim yarışma metinlerinden çok farklı: metinlerin %94'ünde satır sonu, %55'inde markdown işareti var. Deneyde metinler
  düzleştirildi (satır sonları boşluğa çevrildi, markdown işaretleri silindi), 40–250 kelimeye yeniden süzüldü ve
  tekilleştirildi; 23.658 satır kaldı.
- Sınıflar çok dengesiz: claude-opus-4-6 4.767, Gemini flash modelleri 71–83.
- Derlem yazarına göre 2.699 train cevabı ve 3 örnek test satırıyla birebir ya da normalize edilmiş çakışma yok.
- Dosyalar `data/external/` altında; `data/` git'e girmez.


Tokenizer dosyaları veri değil, ön-eğitimli varlık sayılır; kaynakları ve lisansları `MODELS_LICENSES.md`'de.

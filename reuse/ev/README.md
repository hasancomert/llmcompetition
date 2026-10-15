# reuse/

S6E9 (pvep) çalışmasından çıkarılan yardımcılar. Hepsi çok sınıflı (N x C olasılık matrisi) ve balanced accuracy'ye
uyarlandı. Modüller birbirini içe aktarmaz; her dosya tek başına kopyalanabilir. Ayrıntılar, S6E9'daki ölçümler ve
neyin uyarlanamadığı: `HANDOFF_NAME_THAT_LLM.md`.

| Modül | Ne yapar |
|---|---|
| `cv_preds.py` | Gruplu / katmanlı kat ataması, kat döngüsü, OOF ve test tahminini sınıf sırasıyla kaydetme ve yükleme, bölme eşitliği denetimi |
| `blend.py` | Tekrar seçilebilir tepe tırmanma (olasılık ya da log-olasılık), iç içe CV, ağırlık ortalaması, aday ekleme testi, LR istifleme |
| `residual.py` | Sızıntısız iç-kat başlangıç logit'leri; XGBoost / LightGBM / CatBoost artık modelleri |
| `target_encoding.py` | Hiyerarşik (kaba -> ince anahtar) Dirichlet zinciri, kat içinde çapraz, gruplu iç katlar |
| `tokenizer_feats.py` | Birden çok tokenizer ile parça sayısı, ilk / son parça anahtarları ve etiketsiz anahtar sayımları |
| `bias.py` | Balanced accuracy için sınıf başına logit kaydırma ve iç içe doğrulaması (YENİ: S6E9'da karşılığı yok) |
| `refit_sim.py` | Tam veri modeli ile kat ortalamasını sahte testte karşılaştırma |
| `residual_scan.py` | OOF artıklarında grup yapısı taraması (sınıf başına z), çoklu test şans düzeyi, artık kodlama kazancı |
| `adversarial.py` | Train / test ayırt etme AUC'si ve id sırasında etiket yapısı denetimi |
| `kaggle_runner.py` | Kaggle not defteri kurma, gönderme, bekleme, günlük ve çıktı indirme |

Testler (yalnız rastgele sahte veri, depo köküne bir şey yazmaz): `python reuse/tests/test_reuse.py`

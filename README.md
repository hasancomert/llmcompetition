# Kaggle "Name That LLM"

40–250 kelimelik bir metni 18 LLM'den hangisinin yazdığını tahmin etmek. Metrik balanced accuracy. Kod yarışması:
internetsiz, en fazla 9 saat. Son tarih 13.01.2027.

| belge | içerik |
|---|---|
| `PLAN.md` | taşınan / uyarlanan yöntemler, iki önceki projenin çelişkileri, yol haritası |
| `EDA.md` | veri analizi ve baseline karışıklığı |
| `EXPERIMENTS.md` | her deneyin CV skoru (eşli karşılaştırmalarla) |
| `MODELS_LICENSES.md` | tokenizer / model / kütüphane kaynakları ve lisansları |
| `EXTERNAL_DATA.md` | dış veri (şimdilik yok) |
| `handoff/`, `reuse/` | önceki iki projenin devir notları ve kodu (değiştirilmeden) |

## Kurulum

```bash
python3 -m venv ~/.venvs/nllm
~/.venvs/nllm/bin/pip install kaggle kagglehub numpy pandas scikit-learn lightgbm xgboost catboost pyarrow scipy tiktoken tokenizers sentencepiece
kaggle competitions download -c name-that-llm -p data/ && (cd data && unzip -o name-that-llm.zip)
```

Tokenizer dosyaları `models/tokenizers/<lab>/` altına Kaggle'dan tek dosya olarak indirilir (`SOURCE.json`'lardaki
kaynaklar, `MODELS_LICENSES.md`). `data/`, `models/`, `outputs/` git'e girmez.

## Çalıştırma (`src/` içinden)

```bash
python make_folds.py                                   # outputs/folds.csv (bir kez; yeniden koşunca aynı dosyayı vermeli)
python eda.py                                          # outputs/eda/eda_tables.md
python train_tfidf.py --feats char+word --model lr --C 30
python train_gbdt.py --feats style+tok --model lr --C 0.1 --tag lr_style_tok_v2
python blend_baselines.py --tags tfidf_charword_lr_C30 lr_style_tok_v2 --metric ba
python compare.py <tagA> <tagB>                        # prompt üstünde eşli bootstrap
python confusion.py <tag>
```

Her eğitim betiği OOF ve test olasılıklarını `outputs/preds/{oof,pred,meta}_<tag>.*` olarak yazar ve
`EXPERIMENTS.md`'ye bir satır ekler. Var olan bir etiketin üzerine yazmak için `NLLM_FORCE=1` gerekir.

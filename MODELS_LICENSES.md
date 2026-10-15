# Modeller, tokenizer'lar, kütüphaneler ve lisansları

Kural: kullanılan kod, model ve dosyaların lisansı ticari kullanımı kısıtlamamalı. Model ağırlıkları ne bu repoya ne
yerel bilgisayara indirilir; ağırlıklar yalnız Kaggle Models / private Kaggle dataset olarak not defterine bağlanır.

## Tokenizer dosyaları (yalnız tokenizer, ağırlık yok)

Yer: `models/tokenizers/<klasör>/` (git'e girmez). Her klasörde `SOURCE.json` var: Kaggle kaynağı, dosya SHA-256'ları,
notlar. Gönderim not defterinde bu klasör private bir Kaggle dataset olarak bağlanacak (`TOKENIZER_DIR`).
huggingface.co bu ortamda kapalı olduğu için bütün dosyalar Kaggle'dan (`kagglehub.model_download(..., path=...)`
ile tek dosya) alındı.

| klasör | lab | Kaggle kaynağı | resmi yayıncı mı | lisans | ticari kullanım | `tokenizer.json` / model dosyası sha256 (ilk 16) |
|---|---|---|---|---|---|---|
| `openai_o200k` | OpenAI (o200k_base) | `danielhanchen/gpt-oss-20b/transformers/default/1` (Unsloth aynası); `o200k_base.tiktoken`: dataset `abdullahsabrifayed/tiktoken-encodings` | hayır (Kaggle'da resmi OpenAI sahibi yok) | gpt-oss Apache-2.0; o200k_base openai/tiktoken ile MIT | evet | `0614fe83cadab421`; `.tiktoken` `446a9538cb6c348e` (tiktoken'ın resmi beklenen özetiyle aynı) |
| `qwen` | Qwen3 | `qwen-lm/qwen-3/transformers/0.6b/1` | evet | Apache-2.0 | evet | `aeb13307a71acd8f` |
| `qwen_new` | Qwen3.5 (= 3.6; 3.8 ile aynı BPE) | `qwen-lm/qwen-3-5/transformers/qwen3.5-27b/1` | evet | Apache-2.0 | evet | `5f9e4d4901a92b99` |
| `glm` | GLM-5.3-Flash (= GLM-4.7-Flash) | `jeannkouagou/glm-5-3-flash-nvfp4/transformers/nvfp4/1` | hayır (topluluk aynası; aslı zai-org) | MIT (Z.AI) | evet | `19e773648cb4e65d` |
| `kimi` | Kimi (K3; Kimi-Linear / Kimi-VL / Moonlight ile aynı BPE) | `helium990/kimi-k3/pyTorch/k3-mxfp4-v2-p1/1`; aynı `tiktoken.model` MIT etiketli `manojkumarcs28/moonlight-16b-series/pyTorch/moonlight-16b-a3b/1` ve `mirzamilanfarabi/kimi` aynalarında da var | hayır | K3 aynası "Apache 2.0" diyor (doğrulanamadı); aynı dosyayı taşıyan Moonlight / Kimi-Linear / Kimi-VL aynaları MIT | büyük olasılıkla evet: MIT. Pakete Moonlight (MIT) kaynağından konulacak | `b6c497a7469b33ce` (`tiktoken.model`) |
| `gemma` | Gemma 4 (Gemini için vekil) | `google/gemma-4/transformers/gemma-4-e2b-it/1` | evet | Apache-2.0 | evet | `cc8d3a0ce36466cc` |

Notlar:
- Gemini'nin herkese açık tokenizer'ı yok; Gemma 4 tokenizer'ı yaklaşık bir vekil.
- Gemma 3 / 3n "Gemma Terms of Use" lisanslı ve onaya bağlı. Bu hesap onaylamamış (403). Hiçbir şey onaylanmadı,
  onlar yerine Gemma 4 (Apache-2.0) kullanıldı.
- Claude için herkese açık güncel tokenizer yok, kullanılmadı.
- Kimi K3'ün kendi lisansı doğrulanamadı (huggingface.co kapalı). Aynı tokenizer dosyası MIT etiketli Moonlight aynasında
  birebir (aynı sha256) bulunduğu için gönderim paketine o kaynak yazılacak.
- Topluluk aynalarından alınan dosyaların bütünlüğü başka kaynaklarla karşılaştırıldı (ör. o200k tabanı resmi
  tiktoken özetiyle, GLM dosyası RedHat aynasının üst akış özetiyle, Kimi dosyası üç aynada aynı).
- Hiçbir indirilen `.py` dosyası çalıştırılmadı. Kimi'nin `tokenization_kimi.py` dosyası `.txt` olarak saklandı ve
  `pat_str` yalnız AST ile okundu.

## Python kütüphaneleri (bu ortamda kullanılan)

| paket | sürüm | lisans |
|---|---|---|
| numpy | 2.5.4 | BSD-3 |
| pandas | 3.0.6 | BSD-3 |
| scikit-learn | 1.9.1 | BSD-3 |
| scipy | 1.18.1 | BSD-3 |
| lightgbm | 4.7.0 | MIT |
| xgboost | 3.4.1 | Apache-2.0 |
| catboost | 1.2.10 | Apache-2.0 |
| pyarrow | 26.0.0 | Apache-2.0 |
| tokenizers (Hugging Face) | 0.23.3 | Apache-2.0 |
| tiktoken | 0.14.0 | MIT |
| sentencepiece | 0.2.2 | Apache-2.0 |
| kaggle / kagglehub | 2.2.4 / (kurulu) | Apache-2.0 |

## Önceki projelerden gelen kod

`reuse/ev/`, `reuse/airline/`: kullanıcının kendi önceki projelerinden; içerikleri değiştirilmeden kullanılıyor.

## LLM ağırlıkları (Adım 3 ve sonrası, henüz kullanılmadı)

Planlanan: Qwen3 ailesi (Apache-2.0, resmi `qwen-lm` Kaggle Models). Eklendikçe bu tabloya yazılacak.

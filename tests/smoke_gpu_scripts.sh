#!/bin/bash
# CPU smoke test of src/gpu_lora.py and src/gpu_embed.py with a tiny random-weight Qwen3 (no weight download).
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
S=${SMOKE_DIR:-$R/outputs/smoke}

PY=${PY:-python}
rm -rf $S && mkdir -p $S/tiny
$PY - <<EOF
from transformers import Qwen3Config, Qwen3ForCausalLM
import shutil
cfg = Qwen3Config(vocab_size=151669, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
                  tie_word_embeddings=True)
Qwen3ForCausalLM(cfg).save_pretrained("$S/tiny")
for f in ("tokenizer.json", "tokenizer_config.json"):
    shutil.copy("$R/models/tokenizers/qwen/" + f, "$S/tiny/" + f)
print("tiny model saved")
EOF
$PY -c "import pandas as pd; df=pd.read_csv('$R/data/train.csv').head(60); df[['id','text']].to_csv('$S/t60.csv', index=False)"
echo "=== gpu_embed"
$PY $R/src/gpu_embed.py --model-dir $S/tiny --name tiny --texts $S/t60.csv --out $S/emb --batch 8
$PY -c "import numpy as np; e=np.load('$S/emb/emb_tiny_t60.npy'); l=np.load('$S/emb/lp_tiny_t60.npy'); print('emb', e.shape, np.isfinite(e).all(), 'lp', l.shape, np.isfinite(l).all())"
echo "=== gpu_lora (1 fold, 1 epoch, 120 rows)"
$PY $R/src/gpu_lora.py --device cpu --model-dir $S/tiny --train $R/data/train.csv --test $R/data/test.csv \
    --folds-csv $R/outputs/folds.csv --folds 0 --epochs 1 --bs 4 --accum 3 --max-len 64 --limit 120 \
    --out $S/lora --adapter-out $S/adapters
ls $S/lora $S/adapters/adapter_f0
cat $S/adapters/adapter_f0/nllm_meta.json | head -5
$PY - <<EOF
# reload the adapter the way the submission notebook would, and check predictions match the saved test preds
import json, numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import sys; sys.path.insert(0, "$R/src")
import gpu_lora; gpu_lora.disable_torchao_check()
meta = json.load(open("$S/adapters/adapter_f0/nllm_meta.json"))
tok = AutoTokenizer.from_pretrained("$S/tiny"); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
base = AutoModelForSequenceClassification.from_pretrained("$S/tiny", num_labels=meta["num_labels"])
base.config.pad_token_id = meta["pad_token_id"]
m = PeftModel.from_pretrained(base, "$S/adapters/adapter_f0").eval()
import pandas as pd
te = pd.read_csv("$R/data/test.csv")
enc = tok(te["text"].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=meta["max_len"])
with torch.no_grad():
    P = torch.softmax(m(**enc).logits.float(), -1).numpy()
saved = np.load("$S/lora/test_f0.npy")
print("reloaded adapter max |diff| vs saved test preds:", float(np.abs(P - saved).max()))
EOF
echo SMOKE_OK

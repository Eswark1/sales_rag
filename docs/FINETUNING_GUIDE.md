# Fine-Tuning Open Source Models for Sales Intelligence

A step-by-step guide to downloading weights from Hugging Face, practising with pre-trained models, and fine-tuning them to improve SQL generation for this sales database project.

## Table of Contents

1. [Why Fine-Tune?](#1-why-fine-tune)
2. [Model Selection](#2-model-selection)
3. [Prerequisites & Environment Setup](#3-prerequisites--environment-setup)
4. [Practising with Pre-Trained Weights](#4-practising-with-pre-trained-weights)
5. [Preparing Training Data](#5-preparing-training-data)
6. [LoRA Fine-Tuning Workflow](#6-lora-fine-tuning-workflow)
7. [Evaluation & Testing](#7-evaluation--testing)
8. [Deploying the Fine-Tuned Model](#8-deploying-the-fine-tuned-model)
9. [Troubleshooting & Best Practices](#9-troubleshooting--best-practices)
10. [Resources](#10-resources)

---

## 1. Why Fine-Tune?

The project uses `deepseek-coder-v2` via Ollama with schema injection at inference time. Fine-tuning lets you:

| Benefit | Description |
|---------|-------------|
| **Schema memorisation** | Model knows your tables/columns without needing schema in every prompt |
| **Domain vocabulary** | Learns your sales stages, product names, KPI terminology |
| **Output discipline** | Enforces clean SQL — no markdown fences, correct aliases, consistent formatting |
| **Smaller prompts** | No schema injection = lower latency and token cost |
| **Higher accuracy** | Trains on your exact query patterns (aggregations, joins, date arithmetic) |

**When to skip fine-tuning:**
- Base model already produces correct SQL with in-context schema injection
- Fewer than ~200 unique question-SQL pairs available
- Schema changes frequently — fine-tuned model becomes stale

---

## 2. Model Selection

### Recommended Hugging Face Models

| Model | Hub ID | Params | VRAM (4-bit) | SQL Quality | Notes |
|-------|--------|--------|--------------|-------------|-------|
| **DeepSeek-Coder-V2-Lite** | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 16B | ~10 GB | ★★★★★ | Best for this project; drop-in for current Ollama model |
| **Qwen2.5-Coder-7B** | `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | ~6 GB | ★★★★☆ | Lightweight; good for smaller GPUs |
| **CodeLlama-13B** | `codellama/CodeLlama-13b-Instruct-hf` | 13B | ~8 GB | ★★★★☆ | Solid SQL; broad community support |
| **Llama-3.1-8B** | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 8B | ~5 GB | ★★★☆☆ | General purpose; needs more prompting |
| **Mistral-7B** | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | ~5 GB | ★★★☆☆ | Fast inference; less SQL-focused |

### Decision Guide

```
Do you have a 24 GB GPU?
  ├── Yes → deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct (16B, best quality)
  └── No
       ├── 12 GB GPU → Qwen/Qwen2.5-Coder-7B-Instruct
       ├── 8–10 GB GPU → codellama/CodeLlama-7b-Instruct-hf
       └── CPU only → Use Ollama directly; fine-tuning not practical
```

---

## 3. Prerequisites & Environment Setup

> Fine-tuning runs on your **GPU host**, not inside Docker.

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX 3060 (12 GB VRAM) | RTX 3090 / 4090 (24 GB VRAM) |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB free | 250 GB SSD |
| OS | Windows WSL2 / Linux | Ubuntu 22.04 |

Check your GPU and CUDA version:
```bash
nvidia-smi
nvcc --version
```

### Python Environment

```bash
# Create an isolated conda environment (Python 3.11 required)
conda create -n sales-finetune python=3.11
conda activate sales-finetune
```

### Install PyTorch

```bash
# Replace cu121 with your CUDA version (e.g. cu118 for CUDA 11.8)
pip install torch==2.3.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

### Install Fine-Tuning Libraries

```bash
pip install transformers==4.40.0    # Hugging Face core
pip install peft==0.11.0            # LoRA and PEFT methods
pip install trl==0.8.6              # SFTTrainer
pip install datasets==2.19.0        # Dataset utilities
pip install bitsandbytes==0.43.1    # 4-bit/8-bit quantisation
pip install accelerate==0.30.0      # Multi-GPU, mixed precision
pip install sentencepiece protobuf  # Tokeniser dependencies
pip install huggingface_hub         # Model downloads
```

### Verify Setup

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

Expected:
```
PyTorch: 2.3.0+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
VRAM: 24.0 GB
```

### Hugging Face Authentication

Some models (e.g. Llama-3) require accepting a licence on the Hub first:

```bash
pip install -q huggingface_hub
huggingface-cli login   # paste your HF token from hf.co/settings/tokens
```

---

## 4. Practising with Pre-Trained Weights

Before fine-tuning, experiment with the raw model weights to build intuition.

### Step 1 — Download Weights

```python
from huggingface_hub import snapshot_download

# Downloads the full model to ~/.cache/huggingface/hub/
snapshot_download(
    repo_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    ignore_patterns=["*.msgpack", "*.h5"],
)
```

### Step 2 — Load with 4-bit Quantisation

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded. Memory:", torch.cuda.memory_allocated() / 1e9, "GB")
```

### Step 3 — Zero-Shot SQL Generation

Test the base model on your schema before any fine-tuning:

```python
SCHEMA = """
Tables:
  orders(order_id, customer_name, total_amount, status, order_date, rep_id, region_id, product_id)
  regions(region_id, name)
  sales_reps(rep_id, full_name, region_id)
  products(product_id, name, category, unit_price)
  leads(lead_id, company_name, status, source, rep_id, region_id, notes, created_at)
  opportunities(opportunity_id, lead_id, rep_id, stage, estimated_value, probability, notes)
  activities(activity_id, entity_type, entity_id, rep_id, activity_type, occurred_at, summary)
"""

def generate_sql(question: str) -> str:
    prompt = (
        f"You are a SQLite expert. Return ONLY a valid SQLite SELECT — no markdown.\n\n"
        f"Schema:\n{SCHEMA}\n\n"
        f"Question: {question}\n\nSQL:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# Try out different question types
questions = [
    "What is total revenue from delivered orders?",
    "Show revenue broken down by region.",
    "Which sales rep closed the most deals this quarter?",
    "What is the weighted pipeline value?",
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"SQL: {generate_sql(q)}")
```

### Step 4 — Few-Shot Prompting (No Training Needed)

Add 2–3 examples to the prompt to dramatically improve base model output:

```python
FEW_SHOT = """
Example 1:
Question: Total revenue from delivered orders?
SQL: SELECT ROUND(SUM(total_amount),2) AS total_revenue FROM orders WHERE status='Delivered';

Example 2:
Question: Revenue by region?
SQL: SELECT r.name AS region, ROUND(SUM(o.total_amount),2) AS revenue
     FROM orders o JOIN regions r ON o.region_id=r.region_id
     GROUP BY r.name ORDER BY revenue DESC;
"""
```

Record the accuracy on your validation questions — this is your **baseline** to beat with fine-tuning.

---

## 5. Preparing Training Data

### Generate Base Pairs

The project ships with a generator script:

```bash
# From project root
python training/generate_pairs.py
# Output: training/data/sql_pairs.jsonl  (~30 pairs)
```

Each line is a JSON object:
```json
{"instruction": "What is total revenue from delivered orders?", "output": "SELECT ROUND(SUM(o.total_amount),2) AS total_revenue FROM orders o WHERE o.status='Delivered';"}
```

### Expand to 500–2000 Pairs

Thirty examples is not enough for meaningful fine-tuning. Expand coverage:

#### A. Add More Hand-Written Queries

Cover these categories:

| Category | Example Questions |
|----------|------------------|
| Revenue aggregation | Total, by region, by rep, by product, by month |
| Pipeline analysis | Stage breakdown, weighted value, win rate, avg deal size |
| Lead funnel | Conversion rate, by source, by region, disqualification rate |
| Activity tracking | Calls per rep, activity types, follow-up cadence |
| Order fulfilment | Pending orders, avg discount, top customers |
| Time-series | QTD, YTD, last 30/90 days, MoM growth |
| Ranking | TOP N, above average, percentile |
| Multi-table joins | 3+ table joins, subqueries, CTEs |

#### B. Paraphrase Existing Questions

```python
# training/augment_pairs.py
import json, ollama

with open("training/data/sql_pairs.jsonl") as f:
    pairs = [json.loads(l) for l in f]

augmented = list(pairs)  # keep originals

for pair in pairs:
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{
            "role": "user",
            "content": (
                f"Rephrase this question in 3 different ways, one per line, "
                f"same meaning, vary the word order and vocabulary.\n\n"
                f"Question: {pair['instruction']}"
            ),
        }],
    )
    for line in response["message"]["content"].strip().split("\n"):
        line = line.strip().lstrip("0123456789.-) ")
        if line:
            augmented.append({"instruction": line, "output": pair["output"]})

with open("training/data/sql_pairs_augmented.jsonl", "w") as f:
    for p in augmented:
        f.write(json.dumps(p) + "\n")

print(f"Total pairs: {len(augmented)}")
```

#### C. Embed Schema in Each Example

Models learn better when schema is always present:

```python
# training/add_schema.py
import json

SCHEMA = """Tables:
  orders(order_id, customer_name, total_amount, status, order_date, rep_id, region_id, product_id)
  regions(region_id, name)
  sales_reps(rep_id, full_name, region_id)
  products(product_id, name, category, unit_price)
  leads(lead_id, company_name, status, source, created_at, notes, rep_id, region_id)
  opportunities(opportunity_id, lead_id, rep_id, stage, estimated_value, probability, notes)
  activities(activity_id, entity_type, entity_id, rep_id, activity_type, occurred_at, summary)
"""

with open("training/data/sql_pairs_augmented.jsonl") as f:
    pairs = [json.loads(l) for l in f]

for p in pairs:
    p["instruction"] = f"{SCHEMA}\n\nQuestion: {p['instruction']}"

with open("training/data/sql_pairs_final.jsonl", "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

print(f"Schema-enriched pairs written: {len(pairs)}")
```

#### D. Train / Validation Split

```python
import json, random

with open("training/data/sql_pairs_final.jsonl") as f:
    pairs = [json.loads(l) for l in f]

random.seed(42)
random.shuffle(pairs)

split = int(0.9 * len(pairs))

with open("training/data/train.jsonl", "w") as f:
    for p in pairs[:split]: f.write(json.dumps(p) + "\n")

with open("training/data/val.jsonl", "w") as f:
    for p in pairs[split:]: f.write(json.dumps(p) + "\n")

print(f"Train: {split}  Val: {len(pairs)-split}")
```

---

## 6. LoRA Fine-Tuning Workflow

### What is LoRA?

LoRA (Low-Rank Adaptation) freezes all original weights and adds small trainable matrices into each attention layer:

```
Original weight: W  (d × k)  — frozen
LoRA delta:      ΔW = B·A   where B is (d × r), A is (r × k), r << d
Effective:       W' = W + ΔW
```

Only the A and B matrices (~0.1% of parameters) are trained, greatly reducing memory and time.

### Run the Included Script

```bash
python training/finetune_lora.py \
  --model  deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --data   training/data/train.jsonl \
  --output training/output/lora-adapter \
  --epochs 3 \
  --lr     2e-4 \
  --batch  4 \
  --rank   16
```

### Hyperparameter Reference

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `--rank` | 16 | Start at 16. Increase to 32–64 if underfitting. |
| `--lr` | 2e-4 | Standard for LoRA. Lower to 1e-4 if loss is unstable. |
| `--epochs` | 3 | Monitor val loss; stop if it starts rising (overfitting). |
| `--batch` | 4 | Increase with more VRAM. Pair with `gradient_accumulation=4`. |
| `lora_alpha` | 2×rank | Scaling factor — keep at 2×r. |
| `lora_dropout` | 0.05 | Increase to 0.1 if overfitting. |

### Expected Training Output

```
Loading base model: deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
trainable params: 13,107,200 || all params: 16,284,217,344 || trainable%: 0.0805

Starting LoRA fine-tuning…
Epoch 1/3 ██████████ 100% | loss: 0.523 | step 112/112
Epoch 2/3 ██████████ 100% | loss: 0.341 | step 112/112
Epoch 3/3 ██████████ 100% | loss: 0.278 | step 112/112

LoRA adapter saved → training/output/lora-adapter
```

### Memory Optimisation

If you get CUDA OOM errors:

```python
# In finetune_lora.py — reduce memory footprint

# 1. Smaller batch + more gradient accumulation (same effective batch)
per_device_train_batch_size = 2
gradient_accumulation_steps = 8    # effective batch = 2 × 8 = 16

# 2. Enable gradient checkpointing (trades compute for memory)
model.gradient_checkpointing_enable()
model.config.use_cache = False

# 3. Nested quantisation
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # saves ~0.4 bits per param extra
)

# 4. Shorten sequences
max_seq_length = 512   # reduce from 1024
```

---

## 7. Evaluation & Testing

### Load the Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    load_in_4bit=True, device_map="auto", trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "training/output/lora-adapter")
tokenizer = AutoTokenizer.from_pretrained("training/output/lora-adapter")
model.eval()
```

### Metric 1 — Execution Accuracy

Does the generated SQL return identical results to the gold SQL?

```python
import sqlite3

def execution_match(pred_sql, gold_sql, db_path="db/sales.db"):
    try:
        conn = sqlite3.connect(db_path)
        pred_rows = conn.execute(pred_sql).fetchall()
        gold_rows = conn.execute(gold_sql).fetchall()
        conn.close()
        return sorted(pred_rows) == sorted(gold_rows)
    except Exception:
        return False
```

### Metric 2 — Syntactic Validity

```python
def is_valid_sql(sql, db_path="db/sales.db"):
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(f"EXPLAIN QUERY PLAN {sql}")
        conn.close()
        return True
    except sqlite3.Error:
        return False
```

### Metric 3 — Exact Match

```python
import re

def normalise(sql):
    sql = sql.lower().strip()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.rstrip(";")
    return sql

def exact_match(pred, gold):
    return normalise(pred) == normalise(gold)
```

### Run Full Evaluation

```python
# training/evaluate.py
import json

with open("training/data/val.jsonl") as f:
    val = [json.loads(l) for l in f]

em_hits, exec_hits, valid_hits = 0, 0, 0
for ex in val:
    pred = generate_sql(ex["instruction"])   # use your generation function
    gold = ex["output"]
    if exact_match(pred, gold):   em_hits += 1
    if is_valid_sql(pred):        valid_hits += 1
    if execution_match(pred, gold): exec_hits += 1

n = len(val)
print(f"Exact Match:          {em_hits/n:.1%}")
print(f"Execution Accuracy:   {exec_hits/n:.1%}")
print(f"Syntactic Validity:   {valid_hits/n:.1%}")
```

### Target Benchmarks

| Metric | Baseline (base model) | Good | Excellent |
|--------|----------------------|------|-----------|
| Exact Match | ~40–60% | >70% | >85% |
| Execution Accuracy | ~60–75% | >80% | >92% |
| Syntactic Validity | ~85% | >95% | >99% |

---

## 8. Deploying the Fine-Tuned Model

### Option A — Merge & Convert to GGUF (Recommended for Ollama)

This integrates seamlessly with the existing `docker-compose.yml`.

#### Step 1 — Merge Adapter into Base Model

```python
# training/merge_adapter.py
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "training/output/lora-adapter")
merged = model.merge_and_unload()

print("Saving merged model...")
merged.save_pretrained("training/output/merged-model", safe_serialization=True)

tok = AutoTokenizer.from_pretrained("training/output/lora-adapter")
tok.save_pretrained("training/output/merged-model")
print("Done.")
```

#### Step 2 — Convert to GGUF

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

python convert_hf_to_gguf.py \
  ../training/output/merged-model \
  --outtype q4_K_M \
  --outfile ../training/output/sales-sql-agent-q4.gguf
```

#### Step 3 — Register with Ollama

```bash
# Modelfile
cat > training/output/Modelfile << 'EOF'
FROM ./sales-sql-agent-q4.gguf

PARAMETER temperature 0.0
PARAMETER stop "### Instruction"
PARAMETER stop "###"

SYSTEM """You are a SQLite expert for a sales database. Return ONLY a valid SQLite SELECT — no markdown fences, no explanation."""

TEMPLATE """### Instruction:
{{ .Prompt }}

### Response:
"""
EOF

ollama create sales-sql-agent -f training/output/Modelfile
ollama list   # confirm it appears
```

#### Step 4 — Update the Project Config

In your `.env` file:
```env
OLLAMA_MODEL=sales-sql-agent
```

Or to test without touching the environment:
```bash
docker compose exec app env OLLAMA_MODEL=sales-sql-agent uvicorn app.main:app --port 8001
```

### Option B — Direct Transformers Inference

For more control or if you prefer not to use Ollama:

```python
# app/agents/finetuned_sql_agent.py
import re, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.db import get_schema

class FineTunedSQLAgent:
    _instance = None

    def __init__(self, adapter_path="training/output/lora-adapter"):
        base = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            load_in_4bit=True, device_map="auto", trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    def generate(self, question: str) -> str:
        schema = get_schema()
        prompt = (
            f"{schema}\n\n"
            f"### Instruction:\n{question}\n\n### Response:\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=200,
                temperature=0.0, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        raw = re.sub(r"```sql\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"```\s*", "", raw)
        return raw.strip()
```

Then swap it into `app/agents/sql_agent.py`:
```python
from app.agents.finetuned_sql_agent import FineTunedSQLAgent
_agent = FineTunedSQLAgent()

def question_to_sql(question: str) -> str:
    return _agent.generate(question)
```

### Option C — vLLM (High-Throughput Production)

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
  --model training/output/merged-model \
  --dtype half \
  --max-model-len 2048 \
  --port 11434   # matches OLLAMA_BASE_URL port
```

The existing Ollama client in `sql_agent.py` is then compatible via the OpenAI-compatible endpoint.

---

## 9. Troubleshooting & Best Practices

### Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| CUDA OOM | Batch too large or model too big | Reduce `batch`, enable gradient checkpointing, use double quantisation |
| Model generates markdown fences | Base model habit | Add post-processing `re.sub` or add more clean-output training pairs |
| Loss NaN after first step | Learning rate too high | Lower `--lr` to `5e-5` |
| Validation loss rises after epoch 1 | Overfitting | Reduce epochs, increase `lora_dropout`, add more training data |
| SQL has non-existent column names | Schema not in training data | Re-run `add_schema.py` and retrain |
| `trust_remote_code` warnings | Model uses custom code | Add `--trust-remote-code` flag; inspect the code first |

### Best Practices

**Version your adapters**
```bash
git lfs install
git lfs track "*.safetensors" "*.bin" "*.gguf"
git add training/output/lora-adapter
git commit -m "feat: LoRA adapter v1 — 800 SQL pairs, 87% exec accuracy"
```

**Tag your training experiments**
```bash
# Descriptive output directories
--output training/output/v1-rank16-lr2e4-3ep
--output training/output/v2-rank32-augmented
```

**Track metrics before deploying**
Maintain a simple log:

```
| Version | Pairs | Rank | Epochs | EM    | Exec Acc | Notes                    |
|---------|-------|------|--------|-------|----------|--------------------------|
| v1      | 30    | 16   | 3      | 52%   | 68%      | Baseline — original pairs |
| v2      | 800   | 16   | 3      | 74%   | 87%      | Augmented + schema        |
| v3      | 800   | 32   | 5      | 81%   | 93%      | Higher rank, more epochs  |
```

**A/B test in staging before updating `.env`**
```bash
# Run fine-tuned model on port 8001 alongside the production 8000
OLLAMA_MODEL=sales-sql-agent uvicorn app.main:app --port 8001
```

---

## 10. Resources

### Papers
- [LoRA: Low-Rank Adaptation of Large Language Models (2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs (2023)](https://arxiv.org/abs/2305.14314)
- [PEFT: State of the Art Parameter-Efficient Fine-Tuning (2023)](https://arxiv.org/abs/2303.15647)
- [Text-to-SQL Benchmarks Survey (2024)](https://arxiv.org/abs/2406.08426)

### Documentation
- [Hugging Face PEFT Docs](https://huggingface.co/docs/peft)
- [TRL SFTTrainer Guide](https://huggingface.co/docs/trl/sft_trainer)
- [BitsAndBytes Quantisation](https://huggingface.co/docs/bitsandbytes)
- [Ollama Modelfile Syntax](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [llama.cpp GGUF Conversion](https://github.com/ggerganov/llama.cpp/blob/master/docs/development/HOWTO-create-model.md)

### Model Hubs
- [Hugging Face — Text Generation Models](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending)
- [DeepSeek on Hugging Face](https://huggingface.co/deepseek-ai)
- [Ollama Library](https://ollama.com/library)

---

## Quick-Reference Checklist

```
Setup
  [ ] Conda environment created (Python 3.11)
  [ ] PyTorch + CUDA verified
  [ ] PEFT / TRL / BitsAndBytes installed
  [ ] Hugging Face token set (if using gated models)

Data
  [ ] Run training/generate_pairs.py (base 30 pairs)
  [ ] Expand to 500+ pairs via augmentation / manual curation
  [ ] Add schema to each instruction
  [ ] Train/val split created (90/10)

Training
  [ ] Base model tested zero-shot → baseline metrics recorded
  [ ] finetune_lora.py run, loss converging
  [ ] Validation loss not rising (no overfitting)

Evaluation
  [ ] Exact Match, Execution Accuracy, Validity computed on val set
  [ ] Fine-tuned model beats baseline on all three metrics

Deployment
  [ ] Adapter merged into base model
  [ ] GGUF file created (Q4_K_M recommended)
  [ ] Ollama Modelfile created and model registered
  [ ] .env OLLAMA_MODEL updated to new model name
  [ ] End-to-end /ask API tested with new model

Housekeeping
  [ ] Adapter version committed to git (git-lfs)
  [ ] Metrics table updated with v-number and scores
```

"""
finetune_lora.py – LoRA fine-tune a DeepSeek (or any causal LM) on SQL Q&A pairs.

Prerequisites (GPU host, not inside the main Docker container):
  pip install torch transformers peft trl datasets bitsandbytes accelerate

Usage:
  python training/finetune_lora.py \
    --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --data   training/data/sql_pairs.jsonl \
    --output training/output/lora-adapter
"""

import argparse
import json
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    p.add_argument("--data",   default="training/data/sql_pairs.jsonl")
    p.add_argument("--output", default="training/output/lora-adapter")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr",     type=float, default=2e-4)
    p.add_argument("--batch",  type=int, default=4)
    p.add_argument("--rank",   type=int, default=16, help="LoRA rank")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Lazy imports (only available on GPU training host) ─────────────────
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    # ── Load dataset ───────────────────────────────────────────────────────
    with open(args.data) as f:
        raw = [json.loads(l) for l in f if l.strip()]

    def format_prompt(ex):
        return {
            "text": (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Response:\n{ex['output']}"
            )
        }

    dataset = Dataset.from_list(raw).map(format_prompt)

    # ── Quantisation (4-bit NF4 – fits 7-16B on a 16 GB GPU) ─────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ── Load base model ────────────────────────────────────────────────────
    print(f"Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA config ────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training ───────────────────────────────────────────────────────────
    train_cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=512,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=train_cfg,
        tokenizer=tokenizer,
    )

    print("Starting LoRA fine-tuning…")
    trainer.train()

    # ── Save adapter ───────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA adapter saved → {args.output}")
    print()
    print("Next steps:")
    print("  1. Merge adapter:  model.merge_and_unload() → save_pretrained()")
    print("  2. Convert to GGUF: llama.cpp/convert_hf_to_gguf.py <merged_dir>")
    print("  3. Copy GGUF to ollama models dir and register with a Modelfile")


if __name__ == "__main__":
    main()

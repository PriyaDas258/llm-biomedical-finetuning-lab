"""
src/finetune.py
---------------
Fine-tunes a pre-trained LLM using LoRA/QLoRA via HuggingFace PEFT + TRL.

LoRA (Low-Rank Adaptation) adds small trainable matrices to frozen model
weights, reducing GPU memory by ~70% vs full fine-tuning while achieving
comparable performance.

QLoRA extends this with 4-bit quantization — enables fine-tuning 7B models
on a single 16GB GPU, or smaller models on CPU.

Supports:
  - Any HuggingFace causal LM (Phi-2, TinyLlama, Mistral-7B, Llama-3)
  - LoRA and QLoRA modes
  - Optional Weights & Biases experiment tracking
  - Checkpoint saving and resuming
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class FineTuningConfig:
    """All hyperparameters in one place — easy to experiment with."""

    # Model
    base_model: str = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    use_4bit: bool = True           # QLoRA: quantize to 4-bit
    use_nested_quant: bool = False  # Double quantization

    # LoRA
    lora_r: int = int(os.getenv("LORA_R", 16))          # Rank — higher = more params
    lora_alpha: int = int(os.getenv("LORA_ALPHA", 32))  # Scaling factor
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training
    output_dir: str = "outputs/checkpoints"
    max_steps: int = int(os.getenv("MAX_STEPS", 100))
    batch_size: int = int(os.getenv("BATCH_SIZE", 4))
    gradient_accumulation_steps: int = 2
    learning_rate: float = float(os.getenv("LEARNING_RATE", 2e-4))
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    save_steps: int = 50
    logging_steps: int = 10
    eval_steps: int = 50

    # Data
    data_dir: str = "data"
    dataset_text_field: str = "text"

    # Tracking
    use_wandb: bool = bool(os.getenv("WANDB_API_KEY"))
    run_name: str = "biomedical-lora-finetune"


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str):
    """Loads tokenizer with padding token set."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(config: FineTuningConfig):
    """
    Loads the base model with optional 4-bit quantization (QLoRA).

    4-bit quantization reduces memory from ~14GB (7B fp16) to ~5GB,
    enabling fine-tuning on consumer GPUs or even CPU.
    """
    if config.use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )
        print(f"Loading {config.base_model} with 4-bit quantization (QLoRA)...")
    else:
        bnb_config = None
        print(f"Loading {config.base_model} in full precision (CPU/no quantization)...")

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    return model


# ---------------------------------------------------------------------------
# LoRA Configuration
# ---------------------------------------------------------------------------

def apply_lora(model, config: FineTuningConfig):
    """
    Applies LoRA adapters to the model.

    LoRA adds rank-decomposition matrices (A, B) to attention layers:
      W_new = W_frozen + alpha/r * B @ A
    Only A and B are trained — typically 0.1-1% of original parameters.
    """
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Print trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train(config: Optional[FineTuningConfig] = None) -> str:
    """
    Runs the full fine-tuning pipeline.

    Returns:
        Path to the saved fine-tuned model
    """
    if config is None:
        config = FineTuningConfig()

    # Setup output dir
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(f"{config.data_dir}/pubmedqa_processed")
    except Exception:
        print("Processed dataset not found. Using synthetic sample dataset...")
        from data.prepare_dataset import get_sample_dataset
        dataset = get_sample_dataset(n_samples=100)

    print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

    # Load tokenizer and model
    tokenizer = load_tokenizer(config.base_model)
    model = load_base_model(config)
    model = apply_lora(model, config)

    # Wandb setup
    if config.use_wandb:
        import wandb
        wandb.init(project="biomedical-llm-finetune", name=config.run_name)

    # Training arguments
    training_args = SFTConfig(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=torch.cuda.is_available(),
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        load_best_model_at_end=True,
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.run_name,
        max_seq_length=config.max_seq_length,
        dataset_text_field=config.dataset_text_field,
        packing=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        tokenizer=tokenizer,
    )

    print(f"\nStarting fine-tuning: {config.max_steps} steps...")
    trainer.train()

    # Save final model
    final_path = Path(config.output_dir) / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Save config for reference
    config_path = Path(config.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    print(f"\nFine-tuning complete! Model saved to: {final_path}")
    return str(final_path)


if __name__ == "__main__":
    cfg = FineTuningConfig()
    print(f"Fine-tuning {cfg.base_model} with LoRA")
    print(f"LoRA rank: {cfg.lora_r}, alpha: {cfg.lora_alpha}")
    print(f"Steps: {cfg.max_steps}, LR: {cfg.learning_rate}")
    train(cfg)

# 🧠 LLM Fine-tuning Lab — Biomedical Domain Adaptation

Fine-tune open-source LLMs (TinyLlama, Phi-2, Mistral-7B) on biomedical QA using **LoRA** and **QLoRA** via HuggingFace PEFT + TRL. Includes full evaluation pipeline comparing base vs fine-tuned model.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FINE-TUNING PIPELINE                     │
│                                                             │
│  PubMedQA Dataset                                           │
│  (1000 expert QA pairs)                                     │
│         │                                                   │
│         ▼                                                   │
│  data/prepare_dataset.py                                    │
│  • Format as instruction template                           │
│  • Train / Val / Test split                                 │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────┐                           │
│  │  Base LLM (frozen weights)   │                           │
│  │  TinyLlama / Phi-2 / Mistral │                           │
│  │                              │                           │
│  │  + LoRA Adapters (trainable) │                           │
│  │    W_new = W + α/r · B @ A   │                           │
│  │    Only 0.1–1% params trained│                           │
│  └──────────────────────────────┘                           │
│         │                                                   │
│         ▼                                                   │
│  SFTTrainer (TRL)                                           │
│  • Supervised fine-tuning                                   │
│  • Gradient checkpointing                                   │
│  • Optional: 4-bit QLoRA (GPU)                              │
│         │                                                   │
│         ▼                                                   │
│  outputs/checkpoints/final_model/                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                      │
│                                                             │
│  Test Set (100 questions)                                   │
│         │                                                   │
│         ├─────────────► Base Model ──────────┐             │
│         │                                    │             │
│         └─────────────► Fine-tuned Model ────┤             │
│                                              ▼             │
│                                    Metrics Comparison      │
│                                    • ROUGE-1/2/L           │
│                                    • BLEU                  │
│                                    • Exact Match           │
│                                    • Biomedical Coverage   │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

- **LoRA fine-tuning** — trains only 0.1–1% of parameters, fits on CPU/consumer GPU
- **QLoRA support** — 4-bit quantization for 7B models on 16GB GPU
- **PubMedQA dataset** — 1,000 expert-annotated biomedical QA pairs (free, auto-downloaded)
- **Side-by-side comparison** — base vs fine-tuned on same questions
- **Full evaluation suite** — ROUGE, BLEU, exact match, biomedical term coverage
- **Streamlit UI** — launch training, compare models, visualize metrics

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/llm-finetuning-lab.git
cd llm-finetuning-lab
pip install -r requirements.txt

cp .env.example .env
# Add HF_TOKEN (free from huggingface.co/settings/tokens)

# 1. Prepare dataset
python data/prepare_dataset.py

# 2. Fine-tune
python src/finetune.py

# 3. Compare models
python src/inference.py

# 4. Evaluate
python src/evaluate.py

# 5. Launch UI
streamlit run app.py
```

---

## LoRA Explained

```
Standard fine-tuning:  Update all W (billions of params, huge memory)

LoRA:
  W_original  (frozen, e.g. 4096×4096 = 16M params)
       +
  B @ A       (trainable, e.g. 4096×16 + 16×4096 = 131K params)
              ← Only these are trained: 0.8% of the layer
```

**Key hyperparameters:**
| Param | Effect | Default |
|-------|--------|---------|
| `r` (rank) | More params, higher capacity | 16 |
| `lora_alpha` | Scaling; set to 2×r | 32 |
| `lora_dropout` | Regularisation | 0.05 |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base models | TinyLlama, Phi-2, Mistral-7B |
| Fine-tuning | HuggingFace PEFT + TRL |
| Quantization | BitsAndBytes (4-bit QLoRA) |
| Dataset | PubMedQA via HuggingFace Datasets |
| Evaluation | ROUGE, BLEU, custom metrics |
| Tracking | Weights & Biases (optional) |
| UI | Streamlit + Plotly |

---

## Author

**Priya Das** — GenAI Engineer & Data Scientist | Toronto, Canada

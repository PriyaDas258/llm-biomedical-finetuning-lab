# 🧠 LLM Fine-tuning Lab — Biomedical Domain Adaptation

Fine-tune open-source LLMs (TinyLlama, Phi-2, Mistral-7B) on biomedical QA using LoRA and QLoRA with Hugging Face PEFT + TRL. Includes full training + evaluation pipeline and a beautiful Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers%20%7C%20PEFT%20%7C%20TRL-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Features
- LoRA & 4-bit QLoRA support (fits on consumer GPUs)
- PubMedQA biomedical QA dataset (expert-annotated)
- Side-by-side base vs. fine-tuned comparison
- Comprehensive evaluation (ROUGE, BLEU, Exact Match, Biomedical Coverage)
- Interactive Streamlit dashboard

## Project Structure
llm-biomedical-finetuning-lab/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py                     # Streamlit UI
│
├── data/
│   └── prepare_dataset.py
│
├── src/
│   ├── init.py
│   ├── finetune.py
│   ├── inference.py
│   └── evaluate.py
│
├── tests/
│   └── test_evaluate.py
│
└── outputs/                   # (gitignored)
text## Quickstart

```bash
git clone https://github.com/PriyaDas258/llm-biomedical-finetuning-lab.git
cd llm-biomedical-finetuning-lab

pip install -r requirements.txt

cp .env.example .env
# → Add your HF_TOKEN from huggingface.co/settings/tokens

# 1. Prepare dataset
python data/prepare_dataset.py

# 2. Fine-tune (optional: reduce max_steps for testing)
python -m src.finetune

# 3. Launch UI
streamlit run app.py

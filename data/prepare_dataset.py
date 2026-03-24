"""
data/prepare_dataset.py
-----------------------
Loads and prepares the PubMedQA dataset for fine-tuning.

PubMedQA is a free, open biomedical QA dataset with:
  - 1,000 expert-annotated QA pairs
  - Questions derived from PubMed paper titles
  - Long-form answers from abstracts
  - Yes/No/Maybe labels

Formats data into instruction-following format compatible
with LoRA fine-tuning of chat/instruction-tuned LLMs.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATE = """### Instruction:
You are a biomedical AI assistant. Answer the following clinical or research question accurately and concisely based on scientific evidence.

### Question:
{question}

### Context:
{context}

### Answer:
{answer}"""


INFERENCE_TEMPLATE = """### Instruction:
You are a biomedical AI assistant. Answer the following clinical or research question accurately and concisely based on scientific evidence.

### Question:
{question}

### Context:
{context}

### Answer:
"""


def format_pubmedqa_sample(sample: Dict) -> Dict:
    """
    Converts a PubMedQA sample into instruction-following format.

    PubMedQA fields used:
      - question: the research question
      - context.contexts: list of abstract sentences
      - long_answer: the expert answer
      - final_decision: yes/no/maybe label
    """
    question = sample.get("question", "")
    contexts = sample.get("context", {}).get("contexts", [])
    context_text = " ".join(contexts[:3]) if contexts else "No context provided."
    long_answer = sample.get("long_answer", "")
    decision = sample.get("final_decision", "")

    if long_answer:
        answer = f"{long_answer} (Conclusion: {decision})" if decision else long_answer
    else:
        answer = decision if decision else "Insufficient information to answer."

    text = INSTRUCTION_TEMPLATE.format(
        question=question,
        context=context_text[:800],
        answer=answer,
    )
    return {"text": text, "question": question, "answer": answer, "context": context_text[:800]}


def load_and_prepare(
    dataset_name: str = "qiaojin/PubMedQA",
    subset: str = "pqa_labeled",
    train_size: int = 800,
    val_size: int = 100,
    test_size: int = 100,
    save_dir: str = "data",
) -> DatasetDict:
    """
    Loads PubMedQA and splits into train/val/test sets.

    Args:
        dataset_name: HuggingFace dataset identifier
        subset: PubMedQA subset (pqa_labeled = expert annotated)
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        save_dir: Directory to save processed datasets

    Returns:
        DatasetDict with train/validation/test splits
    """
    print(f"Loading {dataset_name} ({subset})...")
    dataset = load_dataset(dataset_name, subset)

    # PubMedQA labeled set has ~1000 samples in 'train'
    raw = dataset["train"]
    total = len(raw)
    print(f"Total samples: {total}")

    # Format all samples
    formatted = [format_pubmedqa_sample(raw[i]) for i in range(min(total, train_size + val_size + test_size))]

    # Split
    train_data = formatted[:train_size]
    val_data = formatted[train_size:train_size + val_size]
    test_data = formatted[train_size + val_size:train_size + val_size + test_size]

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })

    # Save to disk
    save_path = Path(save_dir) / "pubmedqa_processed"
    save_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(save_path))
    print(f"Dataset saved to {save_path}")

    # Also save test set as JSON for easy evaluation
    test_path = Path(save_dir) / "test_samples.json"
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"Test samples saved to {test_path}")

    return dataset_dict


def load_local_dataset(data_dir: str = "data") -> DatasetDict:
    """Loads previously saved processed dataset from disk."""
    from datasets import load_from_disk
    path = Path(data_dir) / "pubmedqa_processed"
    if not path.exists():
        raise FileNotFoundError(f"No processed dataset found at {path}. Run prepare_dataset.py first.")
    return load_from_disk(str(path))


def get_sample_dataset(n_samples: int = 50) -> DatasetDict:
    """
    Creates a tiny synthetic dataset for quick testing without downloading.
    Useful for CI/CD or demo purposes.
    """
    samples = [
        {
            "question": "Does seizure frequency correlate with cognitive decline in epilepsy patients?",
            "context": "A longitudinal study followed 200 epilepsy patients over 5 years. Neuropsychological testing was performed annually. Patients with drug-resistant epilepsy showed greater cognitive decline.",
            "answer": "Yes, higher seizure frequency is associated with greater cognitive decline, particularly in drug-resistant epilepsy patients. (Conclusion: yes)",
            "text": INSTRUCTION_TEMPLATE.format(
                question="Does seizure frequency correlate with cognitive decline in epilepsy patients?",
                context="A longitudinal study followed 200 epilepsy patients over 5 years.",
                answer="Yes, higher seizure frequency is associated with greater cognitive decline.",
            ),
        },
        {
            "question": "Is deep brain stimulation effective for Parkinson's disease tremor?",
            "context": "Randomized controlled trials have demonstrated that deep brain stimulation of the subthalamic nucleus significantly reduces tremor in Parkinson's disease patients.",
            "answer": "Yes, deep brain stimulation of the subthalamic nucleus effectively reduces tremor in Parkinson's disease. (Conclusion: yes)",
            "text": INSTRUCTION_TEMPLATE.format(
                question="Is deep brain stimulation effective for Parkinson's disease tremor?",
                context="Randomized controlled trials demonstrated DBS effectiveness.",
                answer="Yes, DBS effectively reduces tremor in Parkinson's disease.",
            ),
        },
    ]
    # Replicate to reach n_samples
    extended = (samples * (n_samples // len(samples) + 1))[:n_samples]
    split = int(n_samples * 0.8)
    return DatasetDict({
        "train": Dataset.from_list(extended[:split]),
        "validation": Dataset.from_list(extended[split:split + int(n_samples * 0.1)]),
        "test": Dataset.from_list(extended[split + int(n_samples * 0.1):]),
    })


if __name__ == "__main__":
    print("Preparing PubMedQA dataset for fine-tuning...")
    dataset = load_and_prepare()
    print(f"\nDataset splits:")
    for split, ds in dataset.items():
        print(f"  {split}: {len(ds)} samples")
    print("\nSample training example:")
    print(dataset["train"][0]["text"][:400])

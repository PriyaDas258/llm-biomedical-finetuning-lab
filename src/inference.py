"""
src/inference.py
----------------
Runs inference with both base and fine-tuned models side by side.
Enables direct comparison of responses on biomedical questions.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

INFERENCE_TEMPLATE = """### Instruction:
You are a biomedical AI assistant. Answer the following clinical or research question accurately and concisely based on scientific evidence.

### Question:
{question}

### Context:
{context}

### Answer:
"""


def load_model_for_inference(model_path: str, is_peft: bool = False, base_model: str = None):
    """
    Loads a model for inference.

    Args:
        model_path: Path to model (HF hub name or local directory)
        is_peft: Whether this is a LoRA fine-tuned model
        base_model: Base model name (required if is_peft=True)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if not is_peft else base_model,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_peft and base_model:
        print(f"Loading LoRA fine-tuned model from {model_path}...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        print(f"Loading base model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )

    model.eval()
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    question: str,
    context: str = "",
    max_new_tokens: int = 200,
    temperature: float = 0.1,
) -> str:
    """Generates an answer for a given question."""
    prompt = INFERENCE_TEMPLATE.format(
        question=question,
        context=context[:500] if context else "No additional context provided.",
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    return answer.strip()


def compare_models(
    questions: List[Dict],
    base_model_name: str,
    finetuned_path: str,
    output_path: str = "outputs/comparison_results.json",
) -> List[Dict]:
    """
    Runs both models on the same questions and saves comparison.

    Args:
        questions: List of dicts with 'question', 'context', 'reference_answer'
        base_model_name: HF model name for base model
        finetuned_path: Local path to fine-tuned model
        output_path: Where to save results JSON

    Returns:
        List of comparison results
    """
    print("Loading base model...")
    base_model, base_tokenizer = load_model_for_inference(base_model_name)

    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = load_model_for_inference(
        finetuned_path, is_peft=False
    )

    results = []
    for i, item in enumerate(questions):
        print(f"\nQuestion {i+1}/{len(questions)}: {item['question'][:60]}...")

        base_answer = generate_answer(
            base_model, base_tokenizer,
            item["question"], item.get("context", "")
        )
        ft_answer = generate_answer(
            ft_model, ft_tokenizer,
            item["question"], item.get("context", "")
        )

        result = {
            "question": item["question"],
            "context": item.get("context", ""),
            "reference_answer": item.get("answer", ""),
            "base_model_answer": base_answer,
            "finetuned_answer": ft_answer,
        }
        results.append(result)
        print(f"  Base: {base_answer[:100]}...")
        print(f"  Fine-tuned: {ft_answer[:100]}...")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison saved to {output_path}")

    # Cleanup
    del base_model, ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# Sample questions for quick demo
SAMPLE_QUESTIONS = [
    {
        "question": "Does high seizure frequency correlate with cognitive decline in epilepsy?",
        "context": "Longitudinal studies show that patients with drug-resistant epilepsy experience greater cognitive decline.",
        "answer": "Yes, high seizure frequency is associated with cognitive decline, especially in drug-resistant epilepsy.",
    },
    {
        "question": "What is the role of the subthalamic nucleus in Parkinson's disease?",
        "context": "Deep brain stimulation of the subthalamic nucleus reduces motor symptoms in Parkinson's patients.",
        "answer": "The subthalamic nucleus shows hyperactivity in Parkinson's disease; DBS of this region effectively reduces motor symptoms.",
    },
    {
        "question": "Are EEG biomarkers reliable for seizure prediction?",
        "context": "Several ML studies have shown EEG features can predict seizure onset minutes before clinical manifestation.",
        "answer": "Yes, EEG biomarkers including spectral features and connectivity measures show promise for seizure prediction.",
    },
]

if __name__ == "__main__":
    import os
    base = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ft_path = "outputs/checkpoints/final_model"

    if Path(ft_path).exists():
        results = compare_models(SAMPLE_QUESTIONS, base, ft_path)
    else:
        print(f"Fine-tuned model not found at {ft_path}")
        print("Run src/finetune.py first, then re-run this script.")
        print("\nDemo: running base model only...")
        model, tokenizer = load_model_for_inference(base)
        for q in SAMPLE_QUESTIONS[:2]:
            ans = generate_answer(model, tokenizer, q["question"], q["context"])
            print(f"\nQ: {q['question']}\nA: {ans}\n")

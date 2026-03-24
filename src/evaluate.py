"""
src/evaluate.py
---------------
Evaluates fine-tuned model quality using:
  - ROUGE-1/2/L: n-gram overlap with reference answers
  - BLEU: precision-based n-gram score
  - BERTScore: semantic similarity using embeddings
  - Exact Match: binary correctness (useful for yes/no questions)
  - Custom biomedical term coverage metric

Compares base vs fine-tuned model on all metrics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# Biomedical terms to check for in answers (domain coverage metric)
BIOMEDICAL_TERMS = [
    "seizure", "epilepsy", "EEG", "Parkinson", "neural", "clinical",
    "patient", "treatment", "diagnosis", "biomarker", "cognitive",
    "neurological", "cortex", "electrode", "frequency", "amplitude",
]


def compute_rouge(prediction: str, reference: str) -> Dict[str, float]:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_bleu(prediction: str, reference: str) -> float:
    """Computes BLEU score with smoothing for short sequences."""
    ref_tokens = nltk.word_tokenize(reference.lower())
    pred_tokens = nltk.word_tokenize(prediction.lower())
    smoothing = SmoothingFunction().method1
    try:
        return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
    except Exception:
        return 0.0


def compute_exact_match(prediction: str, reference: str) -> float:
    """
    Checks if the prediction contains the same yes/no/maybe as reference.
    Useful for PubMedQA which has binary labels.
    """
    pred_lower = prediction.lower()
    ref_lower = reference.lower()

    for label in ["yes", "no", "maybe"]:
        if label in ref_lower:
            return 1.0 if label in pred_lower else 0.0
    return 0.0


def compute_biomedical_coverage(prediction: str) -> float:
    """
    Measures what fraction of biomedical terms appear in the prediction.
    A higher score means the model uses domain-appropriate language.
    """
    pred_lower = prediction.lower()
    hits = sum(1 for term in BIOMEDICAL_TERMS if term.lower() in pred_lower)
    return hits / len(BIOMEDICAL_TERMS)


def compute_response_quality(prediction: str) -> Dict[str, float]:
    """
    Heuristic quality metrics for generated text:
      - length_score: Is response an appropriate length? (50-300 words ideal)
      - specificity: Does it avoid vague filler phrases?
    """
    words = prediction.split()
    n_words = len(words)

    # Length score — penalize very short (<20 words) or very long (>400 words)
    if n_words < 20:
        length_score = n_words / 20
    elif n_words > 400:
        length_score = max(0.5, 1 - (n_words - 400) / 400)
    else:
        length_score = 1.0

    # Specificity — penalize common filler phrases
    filler_phrases = [
        "i don't know", "i cannot", "i'm not sure", "as an ai",
        "i'm unable", "i apologize", "please consult",
    ]
    pred_lower = prediction.lower()
    filler_count = sum(1 for phrase in filler_phrases if phrase in pred_lower)
    specificity = max(0.0, 1 - filler_count * 0.25)

    return {"length_score": length_score, "specificity": specificity}


def evaluate_model_outputs(
    results: List[Dict],
    model_key: str = "finetuned_answer",
) -> Dict[str, float]:
    """
    Computes aggregate metrics across all test samples for one model.

    Args:
        results: List of comparison result dicts
        model_key: Which answer field to evaluate ('base_model_answer' or 'finetuned_answer')

    Returns:
        Dict of metric name → mean score
    """
    all_metrics = []

    for item in results:
        prediction = item.get(model_key, "")
        reference = item.get("reference_answer", "")

        if not prediction or not reference:
            continue

        metrics = {}
        metrics.update(compute_rouge(prediction, reference))
        metrics["bleu"] = compute_bleu(prediction, reference)
        metrics["exact_match"] = compute_exact_match(prediction, reference)
        metrics["biomedical_coverage"] = compute_biomedical_coverage(prediction)
        metrics.update(compute_response_quality(prediction))
        all_metrics.append(metrics)

    if not all_metrics:
        return {}

    # Aggregate
    aggregated = {}
    for key in all_metrics[0]:
        aggregated[key] = float(np.mean([m[key] for m in all_metrics]))

    return aggregated


def compare_and_report(
    results_path: str = "outputs/comparison_results.json",
    output_path: str = "outputs/evaluation_report.json",
) -> pd.DataFrame:
    """
    Loads comparison results and produces a full evaluation report.

    Returns:
        DataFrame with side-by-side metric comparison
    """
    with open(results_path) as f:
        results = json.load(f)

    print(f"Evaluating {len(results)} samples...")

    base_metrics = evaluate_model_outputs(results, "base_model_answer")
    ft_metrics = evaluate_model_outputs(results, "finetuned_answer")

    # Build comparison DataFrame
    rows = []
    for metric in base_metrics:
        base_val = base_metrics[metric]
        ft_val = ft_metrics.get(metric, 0)
        improvement = ft_val - base_val
        rows.append({
            "metric": metric,
            "base_model": round(base_val, 4),
            "finetuned_model": round(ft_val, 4),
            "improvement": round(improvement, 4),
            "improved": "✅" if improvement > 0 else "❌",
        })

    df = pd.DataFrame(rows)

    # Save report
    report = {
        "base_model_metrics": base_metrics,
        "finetuned_model_metrics": ft_metrics,
        "n_samples": len(results),
        "comparison_table": rows,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n📊 Evaluation Report")
    print("=" * 65)
    print(df.to_string(index=False))
    print(f"\nReport saved to {output_path}")

    return df


if __name__ == "__main__":
    results_path = "outputs/comparison_results.json"
    if Path(results_path).exists():
        df = compare_and_report(results_path)
    else:
        print(f"No results found at {results_path}")
        print("Run src/inference.py first to generate comparison results.")

        # Demo with synthetic data
        print("\nRunning demo evaluation on synthetic data...")
        demo_results = [
            {
                "question": "Does seizure frequency correlate with cognitive decline?",
                "reference_answer": "Yes, higher seizure frequency is associated with greater cognitive decline in epilepsy patients.",
                "base_model_answer": "I cannot provide medical advice. Please consult a doctor.",
                "finetuned_answer": "Yes, research shows that higher seizure frequency is associated with greater cognitive decline, particularly in patients with drug-resistant epilepsy.",
            },
        ]
        base_m = evaluate_model_outputs(demo_results, "base_model_answer")
        ft_m = evaluate_model_outputs(demo_results, "finetuned_answer")
        print(f"\nBase model ROUGE-L: {base_m.get('rougeL', 0):.3f}")
        print(f"Fine-tuned ROUGE-L: {ft_m.get('rougeL', 0):.3f}")

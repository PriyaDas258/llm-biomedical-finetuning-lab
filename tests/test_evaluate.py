"""tests/test_evaluate.py — unit tests for evaluation metrics"""
import pytest
from src.evaluate import compute_rouge, compute_bleu, compute_exact_match, compute_biomedical_coverage, compute_response_quality, evaluate_model_outputs

def test_rouge_identical():
    s = compute_rouge("seizure detection using EEG signals", "seizure detection using EEG signals")
    assert s["rouge1"] == pytest.approx(1.0)
    assert s["rougeL"] == pytest.approx(1.0)

def test_rouge_empty():
    s = compute_rouge("", "some reference text")
    assert s["rouge1"] == 0.0

def test_bleu_identical():
    b = compute_bleu("the patient showed improvement", "the patient showed improvement")
    assert b > 0.8

def test_exact_match_yes():
    assert compute_exact_match("Yes, this is correct", "yes, studies confirm") == 1.0

def test_exact_match_no():
    assert compute_exact_match("No evidence found", "yes it works") == 0.0

def test_biomedical_coverage():
    score = compute_biomedical_coverage("The patient had epilepsy and EEG showed seizure activity")
    assert score > 0.1

def test_response_quality_short():
    q = compute_response_quality("Yes.")
    assert q["length_score"] < 1.0

def test_response_quality_filler():
    q = compute_response_quality("As an AI I cannot provide medical advice.")
    assert q["specificity"] < 1.0

def test_evaluate_model_outputs():
    results = [{"question": "Q?", "reference_answer": "Yes, seizure frequency matters.", "finetuned_answer": "Yes, higher seizure frequency leads to cognitive decline."}]
    metrics = evaluate_model_outputs(results, "finetuned_answer")
    assert "rouge1" in metrics
    assert "bleu" in metrics
    assert 0 <= metrics["rouge1"] <= 1

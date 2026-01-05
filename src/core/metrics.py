"""
This module implements standard evaluation metrics for Question Answering tasks.
It includes functions for normalizing answers and calculating Exact Match (EM)
and F1 scores, following the standard evaluation scripts for datasets like
HotpotQA.
"""

import string
import re
from collections import Counter
from pydantic import BaseModel


def normalize_answer(s: str) -> str:
    """
    Lower text and remove punctuation, articles and extra whitespace.
    Standard normalization for SQuAD and HotpotQA evaluation.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """
    Returns 1.0 if the normalized prediction matches the normalized ground truth exactly, else 0.0.
    """
    return (
        1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0
    )


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calculates the F1 score based on token overlap between prediction and ground truth.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class PerformanceMetric(BaseModel):
    em: float
    f1: float

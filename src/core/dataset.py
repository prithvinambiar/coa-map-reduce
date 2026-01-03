# src/baseline/dataset_loader.py
"""
This module contains the dataset loader for the COA(Chain of Agents) benchmark.
"""

from typing import List, Iterator
from pydantic import BaseModel
from datasets import load_dataset


class HotpotQAContext(BaseModel):
    title: List[str]
    sentences: List[List[str]]


class HotpotQAExample(BaseModel):
    id: str
    question: str
    answer: str
    context: HotpotQAContext


def load_hotpot_qa_eval() -> Iterator[HotpotQAExample]:
    """
    Loads the HotpotQA dataset for evaluation.
    We use the 'distractor' configuration and 'validation' split.
    """
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    return (HotpotQAExample(**data) for data in dataset)


if __name__ == "__main__":
    examples = load_hotpot_qa_eval()
    print(next(examples))

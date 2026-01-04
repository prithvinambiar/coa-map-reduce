# src/baseline/dataset_loader.py
"""
This module contains the dataset loader for the COA(Chain of Agents) benchmark.
"""

from typing import List, Iterator, Dict, Any, cast
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
    for item in dataset:
        data = cast(Dict[str, Any], item)
        yield HotpotQAExample(
            id=data["id"],
            question=data["question"],
            answer=data["answer"],
            context=HotpotQAContext(
                title=data["context"]["title"],
                sentences=data["context"]["sentences"],
            ),
        )


if __name__ == "__main__":
    examples = load_hotpot_qa_eval()
    print(next(examples))

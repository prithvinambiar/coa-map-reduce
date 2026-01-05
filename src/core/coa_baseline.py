"""
This module implements the Chain of Agents (CoA) baseline orchestrator.
It coordinates the Worker Agents to process context chunks sequentially and
uses the Manager Agent to generate the final answer.
"""

import argparse
from typing import List
from src.core.dataset import HotpotQAExample
from src.core.worker_agent import WorkerAgent
from src.core.manager_agent import ManagerAgent
from src.core.evaluation import evaluate


class CoABaseline:
    def __init__(
        self,
        worker_model: str = "gemini-flash-latest",
        manager_model: str = "gemini-flash-latest",
    ):
        self.worker = WorkerAgent(model_name=worker_model)
        self.manager = ManagerAgent(model_name=manager_model)

    def _get_chunks(self, example: HotpotQAExample) -> List[str]:
        """
        Splits the HotpotQA context into chunks (one paragraph per chunk).
        """
        chunks = []
        titles = example.context.title
        sentences_list = example.context.sentences

        for title, sentences in zip(titles, sentences_list):
            # Join sentences with spaces to form the paragraph text
            paragraph_text = " ".join(sentences)
            chunks.append(f"Title: {title}\n{paragraph_text}")

        return chunks

    def predict(self, example: HotpotQAExample) -> str:
        chunks = self._get_chunks(example)
        cu = ""  # Initial Communication Unit is empty

        # Chain of Workers: Process each chunk and update the CU
        for chunk in chunks:
            cu = self.worker.process(chunk, example.question, previous_cu=cu)

        # Manager: Generate final answer based on the final CU
        return self.manager.generate_answer(example.question, final_cu=cu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Chain of Agents baseline evaluation."
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to evaluate"
    )
    args = parser.parse_args()

    baseline = CoABaseline()
    evaluate(baseline, args.num_samples)

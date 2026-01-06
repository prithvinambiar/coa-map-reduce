"""
This module implements the Map-Reduce version of the Chain of Agents (CoA).
It executes Worker Agents in parallel (Map phase) to process context chunks
and then aggregates the results (Reduce phase) for the Manager Agent.
"""

import argparse
import asyncio
from typing import List
from src.core.dataset import HotpotQAExample
from src.core.map_reduce_worker_agent import MapReduceWorkerAgent
from src.core.manager_agent import ManagerAgent
from src.core.evaluation import evaluate


class CoAMapReduce:
    def __init__(
        self,
        worker_model: str = "gemini-flash-latest",
        manager_model: str = "gemini-flash-latest",
    ):
        self.worker_model = worker_model
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

        # Map Phase: Process all chunks in parallel
        async def run_map_phase():
            worker = MapReduceWorkerAgent(model_name=self.worker_model)
            tasks = [worker.process(chunk, example.question) for chunk in chunks]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_map_phase())

        # Reduce Phase: Aggregate results, filtering out "None" (irrelevant chunks)
        valid_results = [r for r in results if r and r.strip().lower() != "none"]
        cu = "\n".join(valid_results)

        # Manager: Generate final answer based on the final CU
        return self.manager.generate_answer(example.question, final_cu=cu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Chain of Agents Map-Reduce evaluation."
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to evaluate"
    )
    args = parser.parse_args()

    coa_map_reduce = CoAMapReduce()
    evaluate(coa_map_reduce, args.num_samples)

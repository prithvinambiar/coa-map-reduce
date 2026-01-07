"""
Optimized Async Chain of Agents (CoA) Baseline.
Executes multiple samples in parallel batches, utilizing independent
Worker and Manager agents.
"""

import argparse
import asyncio
from itertools import islice
from typing import List, Optional

from src.core.dataset import HotpotQAExample, load_hotpot_qa_eval
from src.core.worker_agent import WorkerAgent
from src.core.manager_agent import ManagerAgent
from src.core.evaluation import evaluate_batch


class AsyncCoABaseline:
    def __init__(
        self,
        worker_model: str = "gemini-flash-latest",
        manager_model: str = "gemini-flash-latest",
        max_concurrency: int = 5,
    ):
        self.worker_model = worker_model
        self.manager_model = manager_model
        self.max_concurrency = max_concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    def _get_chunks(self, example: HotpotQAExample) -> List[str]:
        chunks = []
        for title, sentences in zip(example.context.title, example.context.sentences):
            paragraph_text = " ".join(sentences)
            chunks.append(f"Title: {title}\n{paragraph_text}")
        return chunks

    async def predict_async(
        self, example: HotpotQAExample, worker: WorkerAgent, manager: ManagerAgent
    ) -> str:
        """
        Runs the sequential chain for a single sample.
        """
        chunks = self._get_chunks(example)
        cu = ""  # Initial Communication Unit

        async with self.semaphore:
            try:
                # 1. Sequential Chain Loop
                for chunk in chunks:
                    cu = await worker.async_process(
                        chunk, example.question, previous_cu=cu
                    )

                # 2. Manager Synthesis
                return await manager.async_generate_answer(
                    example.question, final_cu=cu
                )

            except Exception as e:
                sample_id = getattr(example, "id", "unknown")
                print(f"Error in CoA chain for ID {sample_id}: {e}")
                return ""

    async def run_batch(self, samples: List[HotpotQAExample]):
        print(f"🚀 Starting CoA processing for {len(samples)} samples...")

        # Instantiate Agents INSIDE the async loop.
        # Each agent will create its own genai.Client inside its __init__.
        # Because we are inside an async function, the clients will attach
        # to the current loop, preventing the 'Unclosed connector' error.
        worker = WorkerAgent(model_name=self.worker_model)
        manager = ManagerAgent(model_name=self.manager_model)

        tasks = [self.predict_async(sample, worker, manager) for sample in samples]

        results = await asyncio.gather(*tasks)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Async Chain of Agents baseline evaluation."
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to evaluate"
    )
    args = parser.parse_args()

    # 1. Load Data
    dataset_iterator = load_hotpot_qa_eval()
    samples = list(islice(dataset_iterator, args.num_samples))

    # 2. Initialize
    baseline = AsyncCoABaseline(max_concurrency=5)

    # 3. Run Async
    results = asyncio.run(baseline.run_batch(samples))

    # 4. Evaluate
    evaluate_batch(samples, results)

"""
Optimized Async Map-Reduce Chain of Agents (CoA) Orchestrator.
Executes the Map phase (parallel workers) and Reduce phase (manager aggregation)
asynchronously across multiple samples in parallel.
"""

import argparse
import asyncio
from itertools import islice
from typing import List, Optional

from src.core.dataset import HotpotQAExample, load_hotpot_qa_eval
from src.core.map_reduce_worker_agent import MapReduceWorkerAgent
from src.core.manager_agent import ManagerAgent
from src.core.evaluation import evaluate_batch


class AsyncCoAMapReduce:
    def __init__(
        self,
        worker_model: str = "gemini-flash-latest",
        manager_model: str = "gemini-flash-latest",
        max_concurrency: int = 5,
    ):
        self.worker_model = worker_model
        self.manager_model = manager_model
        self.max_concurrency = max_concurrency

        # Lazy load semaphore to handle rate limiting
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

    async def async_predict(
        self,
        example: HotpotQAExample,
        worker: MapReduceWorkerAgent,
        manager: ManagerAgent,
    ) -> str:
        """
        Executes Map-Reduce for a single sample.
        """
        chunks = self._get_chunks(example)

        # We use the semaphore to limit how many *questions* we process at once.
        # Note: This allows 'len(chunks)' API calls to fly instantly for this specific question.
        async with self.semaphore:
            try:
                # --- MAP PHASE ---
                # Dispatch all chunks to the worker in parallel
                # We assume MapReduceWorkerAgent has an 'async_process' method based on your renaming convention.
                # If the worker method is still named 'process', change this call to 'worker.process'.
                map_tasks = [
                    worker.async_process(chunk, example.question) for chunk in chunks
                ]
                results = await asyncio.gather(*map_tasks)

                # --- REDUCE PHASE ---
                # Filter out irrelevant chunks (e.g., "NO_INFO" or empty strings)
                valid_results = [
                    r for r in results if r and "NO_INFO" not in r and r.strip() != ""
                ]

                # Create the Communication Unit (CU)
                if not valid_results:
                    cu = "No relevant information found in any context chunk."
                else:
                    cu = "\n".join(valid_results)

                # --- MANAGER PHASE ---
                return await manager.async_generate_answer(
                    example.question, final_cu=cu
                )

            except Exception as e:
                sample_id = getattr(example, "id", "unknown")
                print(f"Error in Map-Reduce for ID {sample_id}: {e}")
                return ""

    async def run_batch(self, samples: List[HotpotQAExample]):
        print(f"🚀 Starting Map-Reduce processing for {len(samples)} samples...")

        # Instantiate Agents INSIDE the async loop.
        # This creates fresh clients attached to the current event loop.
        worker = MapReduceWorkerAgent(model_name=self.worker_model)
        manager = ManagerAgent(model_name=self.manager_model)

        tasks = [self.async_predict(sample, worker, manager) for sample in samples]

        results = await asyncio.gather(*tasks)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Async Map-Reduce Chain of Agents evaluation."
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to evaluate"
    )
    args = parser.parse_args()

    # 1. Load Data
    dataset_iterator = load_hotpot_qa_eval()
    samples = list(islice(dataset_iterator, args.num_samples))

    # 2. Initialize
    # Note: Map-Reduce is heavy on API calls (Chunks * Samples).
    # Keep concurrency low (e.g., 3-5) to avoid Rate Limit errors.
    baseline = AsyncCoAMapReduce(max_concurrency=3)

    # 3. Run Async
    results = asyncio.run(baseline.run_batch(samples))

    # 4. Evaluate
    evaluate_batch(samples, results)

"""
Optimized Async Chain of Agents (CoA) Baseline.
Executes multiple samples in parallel batches, but maintains the 
sequential 'chaining' logic within each specific sample.
"""

import argparse
import asyncio
import os
from itertools import islice
from typing import List, Optional

from google import genai
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
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

        self.worker_model = worker_model
        self.manager_model = manager_model
        self.max_concurrency = max_concurrency
        
        # Lazy load semaphore
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
        self, 
        example: HotpotQAExample, 
        worker: WorkerAgent, 
        manager: ManagerAgent
    ) -> str:
        """
        Runs the sequential chain for a single sample.
        """
        chunks = self._get_chunks(example)
        cu = ""  # Initial Communication Unit

        # Limit concurrency to avoid hitting API rate limits
        async with self.semaphore:
            try:
                # 1. Sequential Chain Loop
                # We must process chunks in order because the output of Chunk N 
                # becomes the input for Chunk N+1.
                for chunk in chunks:
                    cu = await worker.aprocess(chunk, example.question, previous_cu=cu)

                # 2. Manager Synthesis
                # Assuming ManagerAgent has an async 'agenerate_answer' or we wrap it
                if hasattr(manager, "agenerate_answer"):
                    return await manager.agenerate_answer(example.question, final_cu=cu)
                
                # Fallback if manager is sync
                return await asyncio.to_thread(
                    manager.generate_answer, example.question, final_cu=cu
                )

            except Exception as e:
                # Safe ID access
                sample_id = getattr(example, 'id', 'unknown')
                print(f"Error in CoA chain for ID {sample_id}: {e}")
                return ""

    async def run_batch(self, samples: List[HotpotQAExample]):
        print(f"🚀 Starting CoA processing for {len(samples)} samples...")
        
        # Initialize Client once per batch (Connection Pooling)
        # This client is shared across all workers in this batch
        shared_client = genai.Client(api_key=self.api_key)
        
        # Instantiate Agents with the shared client
        worker = WorkerAgent(model_name=self.worker_model, client=shared_client)
        
        # Note: If ManagerAgent doesn't accept 'client', remove that arg.
        # Assuming you update ManagerAgent similarly to WorkerAgent.
        manager = ManagerAgent(model_name=self.manager_model) 
        if hasattr(manager, 'client'): 
             manager.client = shared_client

        tasks = [
            self.predict_async(sample, worker, manager) 
            for sample in samples
        ]
        
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
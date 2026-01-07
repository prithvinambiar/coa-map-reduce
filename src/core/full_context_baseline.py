"""
Optimized Async Full Context Baseline.
"""

import argparse
import asyncio
from itertools import islice
import os
from typing import List, Optional  # Added Optional
from google import genai
from src.core.dataset import HotpotQAExample
from src.core.dataset import load_hotpot_qa_eval
from src.core.evaluation import evaluate_batch


class AsyncFullContextBaseline:
    def __init__(
        self, model_name: str = "gemini-flash-latest", max_concurrency: int = 5
    ):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_concurrency = max_concurrency

        # FIX: Initialize to None. Do NOT create Semaphore here.
        self._semaphore: Optional[asyncio.Semaphore] = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """
        Lazy loader ensures Semaphore is created inside the active event loop.
        """
        if self._semaphore is None:
            # This line will now run only when predict_async is called
            # (which is guaranteed to be inside a running loop)
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    def format_context(self, example: HotpotQAExample) -> str:
        formatted_paragraphs = []
        for title, sentences in zip(example.context.title, example.context.sentences):
            paragraph_text = " ".join(sentences)
            formatted_paragraphs.append(f"Title: {title}\n{paragraph_text}")
        return "\n\n".join(formatted_paragraphs)

    def get_prompt(self, context: str, question: str) -> str:
        return (
            "You are a helpful AI assistant. Answer the question based strictly on the provided context.\n"
            "Keep your answer concise.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    async def predict_async(self, example: HotpotQAExample) -> str:
        context_str = self.format_context(example)
        prompt = self.get_prompt(context_str, example.question)

        # FIX: Access the property self.semaphore (which triggers the lazy load)
        async with self.semaphore:
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_name, contents=prompt
                )
                if not response or not response.candidates:
                    return ""
                content = response.candidates[0].content
                if not content or not content.parts:
                    return ""
                text_result = "".join(part.text for part in content.parts if part.text)
                if text_result:
                    return text_result.strip()
            except Exception as e:
                # Safe access to ID if available
                sample_id = getattr(example, "id", "unknown")
                print(f"Error processing sample {sample_id}: {e}")
                return ""
        return ""

    async def run_batch(self, samples: List[HotpotQAExample]):
        tasks = [self.predict_async(sample) for sample in samples]
        print(f"🚀 Starting parallel processing of {len(samples)} samples...")
        results = await asyncio.gather(*tasks)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run async baseline evaluation.")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    # 1. Load Data
    dataset_iterator = load_hotpot_qa_eval()
    samples = list(islice(dataset_iterator, args.num_samples))

    # 2. Initialize
    baseline = AsyncFullContextBaseline(max_concurrency=10)

    # 3. Run Async
    results = asyncio.run(baseline.run_batch(samples))

    # 4. Evaluate Results
    evaluate_batch(samples, results)

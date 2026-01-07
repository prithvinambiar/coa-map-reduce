"""
This module implements a RAG (Retrieval-Augmented Generation) baseline.
It retrieves the top-k most relevant paragraphs using embeddings and prompts the
Gemini model to answer the question based on the retrieved context.
"""

import argparse
import asyncio
import os
from itertools import islice
from typing import List, Optional

import numpy as np
from google import genai
from src.core.dataset import HotpotQAExample, load_hotpot_qa_eval
from src.core.evaluation import evaluate_batch


class RagBaseline:
    def __init__(
        self,
        model_name: str = "gemini-flash-latest",
        embedding_model: str = "text-embedding-004",
        max_concurrency: int = 5,
    ):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

        self.model_name = model_name
        self.embedding_model = embedding_model
        self.max_concurrency = max_concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazy load semaphore to ensure it attaches to the active event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    def get_prompt(self, context: str, question: str) -> str:
        return (
            "You are a helpful AI assistant. Answer the question based strictly on the provided context.\n"
            "Keep your answer concise.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    async def retrieve_async(self, example: HotpotQAExample, top_k: int = 2) -> str:
        """
        Retrieves the top_k most relevant paragraphs based on cosine similarity of embeddings.
        """
        titles = example.context.title
        sentences_list = example.context.sentences

        docs = []
        for title, sentences in zip(titles, sentences_list):
            docs.append(f"Title: {title}\n" + " ".join(sentences))

        if not docs:
            return ""

        # Embed question and docs in a single batch request to save latency
        inputs = [example.question] + docs

        try:
            client = genai.Client(api_key=self.api_key)
            result = await client.aio.models.embed_content(
                model=self.embedding_model, contents=inputs
            )
        except Exception as e:
            print(
                f"Error embedding content for ID {getattr(example, 'id', 'unknown')}: {e}"
            )
            return ""

        if not result.embeddings:
            return ""

        # Extract embeddings
        # result.embeddings is a list of ContentEmbedding objects, which contain .values
        try:
            embeddings = np.array([e.values for e in result.embeddings])
        except AttributeError:
            # Fallback if the SDK structure varies slightly by version
            print("Error parsing embedding response structure.")
            return ""

        q_emb = embeddings[0]
        doc_embs = embeddings[1:]

        # Calculate Cosine Similarity
        # Since API embeddings are often normalized, Dot Product approx Cosine Similarity
        scores = np.dot(doc_embs, q_emb)

        # Get indices of top_k highest scores
        # argsort sorts ascending, so we slice [::-1] to reverse it
        top_indices = np.argsort(scores)[::-1][:top_k]

        selected_docs = [docs[i] for i in top_indices]
        return "\n\n".join(selected_docs)

    async def predict_async(self, example: HotpotQAExample) -> str:
        async with self.semaphore:
            # 1. Retrieve relevant context
            context_str = await self.retrieve_async(example)

            # If retrieval failed or found nothing, we might want to fallback or return empty
            if not context_str:
                return ""

            # 2. Construct Prompt
            prompt = self.get_prompt(context_str, example.question)

            # 3. Generate Answer
            try:
                client = genai.Client(api_key=self.api_key)
                response = await client.aio.models.generate_content(
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
                print(
                    f"Error generating content for ID {getattr(example, 'id', 'unknown')}: {e}"
                )
                return ""
        return ""

    async def run_batch(self, samples: List[HotpotQAExample]):
        """Runs prediction on a list of samples in parallel."""
        tasks = [self.predict_async(sample) for sample in samples]
        print(f"🚀 Starting RAG processing of {len(samples)} samples...")

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG baseline evaluation.")
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to evaluate"
    )
    args = parser.parse_args()

    # 1. Load Data
    dataset_iterator = load_hotpot_qa_eval()
    samples = list(islice(dataset_iterator, args.num_samples))

    # 2. Initialize
    baseline = RagBaseline(max_concurrency=10)

    # 3. Run Async
    results = asyncio.run(baseline.run_batch(samples))

    # 4. Evaluate Results
    evaluate_batch(samples, results)

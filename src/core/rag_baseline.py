"""
This module implements a RAG (Retrieval-Augmented Generation) baseline.
It retrieves the top-k most relevant paragraphs using embeddings and prompts the
Gemini model to answer the question based on the retrieved context.
"""

import argparse
import numpy as np
from src.core.dataset import HotpotQAExample
from src.core.full_context_baseline import AsyncFullContextBaseline
from src.core.evaluation import evaluate


class RagBaseline(AsyncFullContextBaseline):
    def __init__(
        self,
        model_name: str = "gemini-flash-latest",
        embedding_model: str = "text-embedding-004",
    ):
        super().__init__(model_name)
        self.embedding_model = embedding_model

    def retrieve(self, example: HotpotQAExample, top_k: int = 2) -> str:
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

        # Embed question and docs in a single batch request
        inputs = [example.question] + docs
        result = self.client.models.embed_content(
            model=self.embedding_model, contents=inputs
        )

        if result.embeddings is None:
            return ""

        # Extract embeddings
        embeddings = np.array([e.values for e in result.embeddings])
        q_emb = embeddings[0]
        doc_embs = embeddings[1:]

        # Calculate Cosine Similarity
        # (Assuming embeddings are normalized, dot product is cosine similarity)
        scores = np.dot(doc_embs, q_emb)

        # Get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        selected_docs = [docs[i] for i in top_indices]
        return "\n\n".join(selected_docs)

    def predict(self, example: HotpotQAExample) -> str:
        # 1. Retrieve relevant context
        context_str = self.retrieve(example)

        # 2. Construct Prompt
        prompt = self.get_prompt(context_str, example.question)

        # 3. Generate Answer
        response = self.client.models.generate_content(
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
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG baseline evaluation.")
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to evaluate"
    )
    args = parser.parse_args()

    baseline = RagBaseline()
    evaluate(baseline, args.num_samples)

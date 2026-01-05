"""
This module implements a baseline solution for HotpotQA using the full context.
It concatenates all available paragraphs and prompts the Gemini model to answer the question.
"""

import argparse
import os
from google import genai
from src.core.dataset import HotpotQAExample, load_hotpot_qa_eval
from src.core.metrics import exact_match_score, f1_score, PerformanceMetric


class FullContextBaseline:
    def __init__(self, model_name: str = "gemini-flash-latest"):
        """
        Initializes the Gemini model.
        Expects GOOGLE_API_KEY to be set in the environment variables.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def format_context(self, example: HotpotQAExample) -> str:
        """
        Formats the HotpotQA context (titles and sentences) into a single string.
        """
        formatted_paragraphs = []
        titles = example.context.title
        sentences_list = example.context.sentences

        for title, sentences in zip(titles, sentences_list):
            # Join sentences with spaces to form the paragraph text
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

    def predict(self, example: HotpotQAExample) -> str:
        context_str = self.format_context(example)
        prompt = self.get_prompt(context_str, example.question)

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        # Ensure response.text is not None before calling .strip()
        if response and response.text:
            return response.text.strip()
        return ""


def evaluate(num_samples: int) -> PerformanceMetric:
    baseline = FullContextBaseline()
    dataset = load_hotpot_qa_eval()

    total_em = 0.0
    total_f1 = 0.0

    print(f"Running evaluation on {num_samples} samples...")

    for i in range(num_samples):
        try:
            example = next(dataset)
        except StopIteration:
            print(f"Dataset exhausted at {i} samples.")
            num_samples = i
            break

        prediction = baseline.predict(example)
        em = exact_match_score(prediction, example.answer)
        f1 = f1_score(prediction, example.answer)

        total_em += em
        total_f1 += f1

        print(
            f"[{i + 1}] EM: {em} | F1: {f1:.4f} | Pred: {prediction} | Gold: {example.answer}"
        )

    avg_em = total_em / num_samples if num_samples > 0 else 0.0
    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0.0

    print(f"\nAverage EM: {avg_em:.4f}")
    print(f"Average F1: {avg_f1:.4f}")

    return PerformanceMetric(em=avg_em, f1=avg_f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline evaluation.")
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to evaluate"
    )
    args = parser.parse_args()

    evaluate(args.num_samples)

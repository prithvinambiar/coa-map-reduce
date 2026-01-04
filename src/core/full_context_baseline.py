"""
This module implements a baseline solution for HotpotQA using the full context.
It concatenates all available paragraphs and prompts the Gemini model to answer the question.
"""

import os
from google import genai
from src.core.dataset import HotpotQAExample, load_hotpot_qa_eval


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


if __name__ == "__main__":
    # Simple test run
    baseline = FullContextBaseline()
    dataset = load_hotpot_qa_eval()

    # Get first example
    example = next(dataset)
    print(f"Question: {example.question}")
    print(f"Gold Answer: {example.answer}")

    prediction = baseline.predict(example)
    print(f"Prediction: {prediction}")

"""
This module implements the Manager Agent for the Chain of Agents framework.
The Manager Agent receives the final 'Communication Unit' (CU) from the last worker
and synthesizes the final answer to the user's question.
"""

import os
from google import genai


class ManagerAgent:
    def __init__(self, model_name: str = "gemini-flash-latest"):
        """
        Initializes the Manager Agent.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_answer(self, question: str, final_cu: str) -> str:
        """
        Generates the final answer based on the accumulated Communication Unit.

        Args:
            question: The original user question.
            final_cu: The final communication unit from the worker chain.

        Returns:
            The final answer string.
        """
        prompt = self._build_prompt(question, final_cu)

        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Error in ManagerAgent: {e}")

        return ""

    def _build_prompt(self, question: str, final_cu: str) -> str:
        return (
            "You are a manager agent. You have received a summary of information (Communication Unit) "
            "from worker agents who read the source documents.\n\n"
            f"Question: {question}\n\n"
            f"Final Communication Unit:\n{final_cu}\n\n"
            "Task: Answer the question based strictly on the Communication Unit. "
            "Keep your answer concise and direct."
        )

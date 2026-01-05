"""
This module implements the Worker Agent for the Chain of Agents framework.
The Worker Agent processes a chunk of context along with the
'Communication Unit' (CU) passed from the previous agent, and generates an
updated CU.
"""

import os
from google import genai


class WorkerAgent:
    def __init__(self, model_name: str = "gemini-flash-latest"):
        """
        Initializes the Worker Agent.
        We use gemini-flash-latest by default as it is efficient for sequential worker tasks.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def process(self, chunk: str, question: str, previous_cu: str = "") -> str:
        """
        Processes a chunk of text and updates the communication unit (CU).

        Args:
            chunk: The current text chunk (e.g., a paragraph).
            question: The user's question.
            previous_cu: The communication unit from the previous worker (summary of findings so far).

        Returns:
            The updated communication unit.
        """
        prompt = self._build_prompt(chunk, question, previous_cu)

        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Error in WorkerAgent: {e}")

        return previous_cu

    def _build_prompt(self, chunk: str, question: str, previous_cu: str) -> str:
        return (
            "You are a worker agent reading a document chunk to help answer a question.\n"
            "Your task is to update the 'Communication Unit' (summary of relevant info) based on the new chunk.\n\n"
            f"Question: {question}\n\n"
            f"Previous Communication Unit:\n{previous_cu if previous_cu else 'None'}\n\n"
            f"Current Chunk:\n{chunk}\n\n"
            "Output the updated Communication Unit. If the chunk is irrelevant, output the Previous Communication Unit exactly."
        )

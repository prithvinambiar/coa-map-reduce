"""
This module implements the Worker Agent for the Chain of Agents framework.
"""

import os
from google import genai


class WorkerAgent:
    def __init__(self, model_name: str = "gemini-flash-latest"):
        """
        Initializes the Worker Agent with its own independent client.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

        # Instantiate a new client specifically for this agent
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _build_prompt(self, chunk: str, question: str, previous_cu: str) -> str:
        return (
            "You are a worker agent reading a document chunk to help answer a question.\n"
            "Your task is to update the 'Communication Unit' (summary of relevant info) based on the new chunk.\n\n"
            f"Question: {question}\n\n"
            f"Previous Communication Unit:\n{previous_cu if previous_cu else 'None'}\n\n"
            f"Current Chunk:\n{chunk}\n\n"
            "Output the updated Communication Unit. If the chunk is irrelevant, output the Previous Communication Unit exactly."
        )

    def process(self, chunk: str, question: str, previous_cu: str = "") -> str:
        """Synchronous version."""
        prompt = self._build_prompt(chunk, question, previous_cu)
        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            return self._parse_response(response, previous_cu)
        except Exception as e:
            print(f"Error in WorkerAgent (Sync): {e}")
            return previous_cu

    async def async_process(self, chunk: str, question: str, previous_cu: str = "") -> str:
        """Asynchronous version."""
        prompt = self._build_prompt(chunk, question, previous_cu)
        try:
            # Uses the agent's own client
            response = await self.client.aio.models.generate_content(
                model=self.model_name, contents=prompt
            )
            return self._parse_response(response, previous_cu)
        except Exception as e:
            print(f"Error in WorkerAgent (Async): {e}")
            return previous_cu

    def _parse_response(self, response, fallback: str) -> str:
        if not response or not response.candidates:
            return fallback
        content = response.candidates[0].content
        if not content or not content.parts:
            return fallback
        text_result = "".join(part.text for part in content.parts if part.text)
        return text_result.strip() if text_result else fallback

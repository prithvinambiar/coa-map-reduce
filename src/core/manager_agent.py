"""
This module implements the Manager Agent for the Chain of Agents framework.
"""

import os
from google import genai


class ManagerAgent:
    def __init__(self, model_name: str = "gemini-flash-latest"):
        """
        Initializes the Manager Agent with its own independent client.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

        # Instantiate a new client specifically for this agent
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _build_prompt(self, question: str, final_cu: str) -> str:
        return (
            "You are a manager agent. You have received a summary of information (Communication Unit) "
            "from worker agents who read the source documents.\n\n"
            f"Question: {question}\n\n"
            f"Final Communication Unit:\n{final_cu}\n\n"
            "Task: Answer the question based strictly on the Communication Unit. "
            "Keep your answer concise and direct."
        )

    def generate_answer(self, question: str, final_cu: str) -> str:
        """Synchronous version."""
        prompt = self._build_prompt(question, final_cu)
        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            return self._parse_response(response)
        except Exception as e:
            print(f"Error in ManagerAgent (Sync): {e}")
            return ""

    async def agenerate_answer(self, question: str, final_cu: str) -> str:
        """Asynchronous version."""
        prompt = self._build_prompt(question, final_cu)
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name, contents=prompt
            )
            return self._parse_response(response)
        except Exception as e:
            print(f"Error in ManagerAgent (Async): {e}")
            return ""

    def _parse_response(self, response) -> str:
        if not response or not response.candidates:
            return ""
        content = response.candidates[0].content
        if not content or not content.parts:
            return ""
        text_result = "".join(part.text for part in content.parts if part.text)
        return text_result.strip() if text_result else ""

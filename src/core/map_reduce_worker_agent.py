"""
This module implements the Map-Reduce Worker Agent.
It processes individual chunks in parallel to extract evidence.
"""

import os
from google import genai

NO_INFO_TOKEN = "NO_INFO"


class MapReduceWorkerAgent:
    def __init__(self, model_name: str = "gemini-flash-latest"):
        """
        Initializes the Worker Agent with its own independent client.

        NOTE: This class should be instantiated INSIDE an asyncio loop
        (e.g. inside run_batch) to avoid 'Unclosed connector' errors.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _build_prompt(self, chunk: str, question: str) -> str:
        return f"""
            You are a worker agent looking for evidence to answer a question.
            
            USER QUESTION: "{question}"
            
            CONTEXT CHUNK:
            {chunk}
            
            INSTRUCTIONS:
            - Extract any sentences from the context that are relevant to the question.
            - If the context contains no relevant information, output exactly "{NO_INFO_TOKEN}".
            - Be concise. Copy exact quotes where possible.
            """

    async def async_process(self, chunk: str, question: str) -> str:
        """
        Asynchronously processes a chunk of text to extract relevant info.
        Renamed from 'process' to match the orchestrator's convention.
        """
        prompt = self._build_prompt(chunk, question)

        try:
            # Use the .aio accessor for async operations
            response = await self.client.aio.models.generate_content(
                model=self.model_name, contents=prompt
            )
            return self._parse_response(response)
        except Exception as e:
            print(f"Error in MapReduceWorkerAgent: {e}")
            return NO_INFO_TOKEN

    def _parse_response(self, response) -> str:
        """Helper to parse the GenAI response safely."""
        if not response or not response.candidates:
            return NO_INFO_TOKEN

        content = response.candidates[0].content
        if not content or not content.parts:
            return NO_INFO_TOKEN

        text_result = "".join(part.text for part in content.parts if part.text)

        if text_result:
            clean_text = text_result.strip()
            # Handle cases where the model might be chatty e.g. "NO_INFO."
            if NO_INFO_TOKEN in clean_text:
                return NO_INFO_TOKEN
            return clean_text

        return NO_INFO_TOKEN

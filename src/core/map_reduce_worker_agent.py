""" """

import os
from google import genai

NO_INFO_TOKEN = "NO_INFO"


class MapReduceWorkerAgent:
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

    def process(self, chunk: str, question: str) -> str:
        """
        Processes a chunk of text and extracts relevant information.
        Args:
            chunk: The current text chunk (e.g., a paragraph).
            question: The user's question.
        Returns:
            The extracted relevant information or NO_INFO_TOKEN if none found.
        """
        prompt = self._build_prompt(chunk, question)

        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            if not response or not response.candidates:
                return NO_INFO_TOKEN
            content = response.candidates[0].content
            if not content or not content.parts:
                return NO_INFO_TOKEN
            text_result = "".join(part.text for part in content.parts if part.text)
            if text_result:
                return text_result.strip()
        except Exception as e:
            print(f"Error in WorkerAgent: {e}")

        return NO_INFO_TOKEN

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

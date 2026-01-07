import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import asyncio
import numpy as np
from src.core.rag_baseline import RagBaseline
from src.core.dataset import HotpotQAExample, HotpotQAContext


class TestRagBaseline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Patch the environment variable to ensure the class can be instantiated
        self.api_key_patcher = patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"})
        self.api_key_patcher.start()

    def tearDown(self):
        self.api_key_patcher.stop()

    @patch("src.core.full_context_baseline.genai.Client")
    async def test_retrieve_async(self, mock_client_class):
        # Mock the client instance
        mock_client = mock_client_class.return_value

        # Mock embedding response
        # We simulate 1 question and 3 documents
        # Question vector: [1, 0]
        # Doc 1: [1, 0] (Sim = 1.0)
        # Doc 2: [0, 1] (Sim = 0.0)
        # Doc 3: [0.707, 0.707] (Sim ~0.707)

        mock_response = MagicMock()
        mock_response.embeddings = [
            MagicMock(values=[1.0, 0.0]),  # Question
            MagicMock(values=[1.0, 0.0]),  # Doc 1
            MagicMock(values=[0.0, 1.0]),  # Doc 2
            MagicMock(values=[0.707, 0.707]),  # Doc 3
        ]
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_response)

        rag = RagBaseline()

        example = HotpotQAExample(
            id="test",
            question="Q",
            answer="A",
            context=HotpotQAContext(
                title=["Doc1", "Doc2", "Doc3"], sentences=[["S1"], ["S2"], ["S3"]]
            ),
        )

        # Retrieve top 2
        context = await rag.retrieve_async(example, top_k=2)

        # Doc1 (1.0) and Doc3 (0.707) should be selected
        self.assertIn("Title: Doc1", context)
        self.assertIn("Title: Doc3", context)
        self.assertNotIn("Title: Doc2", context)

    @patch("src.core.full_context_baseline.genai.Client")
    async def test_retrieve_async_none_embeddings(self, mock_client_class):
        # Test handling of API failure where embeddings are None
        mock_client = mock_client_class.return_value
        mock_response = MagicMock()
        mock_response.embeddings = None
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_response)

        rag = RagBaseline()
        example = HotpotQAExample(
            id="test",
            question="Q",
            answer="A",
            context=HotpotQAContext(title=["D1"], sentences=[["S1"]]),
        )

        context = await rag.retrieve_async(example)
        self.assertEqual(context, "")


if __name__ == "__main__":
    unittest.main()

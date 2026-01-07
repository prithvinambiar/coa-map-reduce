import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
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

    @patch("src.core.rag_baseline.genai.Client")
    async def test_retrieve_async(self, mock_client_class):
        """
        Tests that retrieve_async correctly calculates cosine similarity
        and selects the top-k documents.
        """
        rag = RagBaseline()

        # 1. Setup Mock Client
        mock_client = mock_client_class.return_value
        mock_response = MagicMock()

        # We simulate 1 question and 3 documents
        # Question vector: [1, 0]
        # Doc 1: [1, 0] (Sim = 1.0) -> Match
        # Doc 2: [0, 1] (Sim = 0.0) -> No Match
        # Doc 3: [0.707, 0.707] (Sim ~0.707) -> 2nd Best

        mock_response.embeddings = [
            MagicMock(values=[1.0, 0.0]),  # Question
            MagicMock(values=[1.0, 0.0]),  # Doc 1
            MagicMock(values=[0.0, 1.0]),  # Doc 2
            MagicMock(values=[0.707, 0.707]),  # Doc 3
        ]
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_response)

        example = HotpotQAExample(
            id="test",
            question="Q",
            answer="A",
            context=HotpotQAContext(
                title=["Doc1", "Doc2", "Doc3"], sentences=[["S1"], ["S2"], ["S3"]]
            ),
        )

        # 2. Retrieve top 2
        context = await rag.retrieve_async(example, top_k=2)

        # 3. Verify Logic
        # Doc1 (1.0) and Doc3 (0.707) should be selected. Doc2 (0.0) ignored.
        self.assertIn("Title: Doc1", context)
        self.assertIn("Title: Doc3", context)
        self.assertNotIn("Title: Doc2", context)

        # Verify call arguments
        call_args = mock_client.aio.models.embed_content.call_args
        # Expecting inputs = [Question, Doc1, Doc2, Doc3]
        self.assertEqual(len(call_args.kwargs["contents"]), 4)

    @patch("src.core.rag_baseline.genai.Client")
    async def test_retrieve_async_failure(self, mock_client_class):
        """Test handling of API failure where embeddings are None or error occurs."""
        rag = RagBaseline()
        mock_client = mock_client_class.return_value

        # Simulate empty response
        mock_response = MagicMock()
        mock_response.embeddings = None
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_response)

        example = HotpotQAExample(
            id="test",
            question="Q",
            answer="A",
            context=HotpotQAContext(title=["D1"], sentences=[["S1"]]),
        )

        context = await rag.retrieve_async(example)
        self.assertEqual(context, "")

    @patch("src.core.rag_baseline.genai.Client")
    async def test_predict_async(self, mock_client_class):
        """Test the full flow: Retrieval -> Prompt -> Generation."""
        rag = RagBaseline()
        mock_client = mock_client_class.return_value

        # 1. Mock Retrieval (Mocking the method directly to isolate generation logic)
        rag.retrieve_async = AsyncMock(return_value="Mocked Context")

        # 2. Mock Generation
        mock_gen_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Generated Answer"
        mock_gen_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=mock_gen_response
        )

        example = HotpotQAExample(
            id="test",
            question="Q",
            answer="A",
            context=HotpotQAContext(title=[], sentences=[]),
        )

        # 3. Execute
        result = await rag.predict_async(example)

        # 4. Verify
        self.assertEqual(result, "Generated Answer")
        rag.retrieve_async.assert_called_once()
        mock_client.aio.models.generate_content.assert_called_once()

        # Verify prompt construction
        call_args = mock_client.aio.models.generate_content.call_args
        prompt_sent = call_args.kwargs["contents"]
        self.assertIn("Mocked Context", prompt_sent)

    @patch("src.core.rag_baseline.genai.Client")
    async def test_run_batch(self, mock_client_class):
        """Test parallel execution of the batch runner."""
        mock_client_instance = mock_client_class.return_value

        # 1. Setup Mock for Embedding (Return valid embeddings for 2 docs + 1 question)
        mock_embed_response = MagicMock()
        mock_embed_response.embeddings = [
            MagicMock(values=[1, 0]),  # Q
            MagicMock(values=[1, 0]),  # D1
        ]
        mock_client_instance.aio.models.embed_content = AsyncMock(
            return_value=mock_embed_response
        )

        # 2. Setup Mock for Generation
        mock_gen_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Batch Answer"
        mock_gen_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_client_instance.aio.models.generate_content = AsyncMock(
            return_value=mock_gen_response
        )

        rag = RagBaseline()

        examples = [
            HotpotQAExample(
                id="1",
                question="Q1",
                answer="A1",
                context=HotpotQAContext(title=["D1"], sentences=[["S1"]]),
            )
        ]

        # 3. Run Batch
        results = await rag.run_batch(examples)

        # 4. Verify
        self.assertEqual(results, ["Batch Answer"])
        # Verify client was initialized inside run_batch
        mock_client_class.assert_called()


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
from src.core.rag_baseline import RagBaseline
from src.core.dataset import HotpotQAExample, HotpotQAContext


class TestRagBaseline(unittest.TestCase):
    def setUp(self):
        # Patch the environment variable to ensure the class can be instantiated
        self.api_key_patcher = patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"})
        self.api_key_patcher.start()

    def tearDown(self):
        self.api_key_patcher.stop()

    @patch("src.core.full_context_baseline.genai.Client")
    def test_retrieve(self, mock_client_class):
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
        mock_client.models.embed_content.return_value = mock_response

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
        context = rag.retrieve(example, top_k=2)

        # Doc1 (1.0) and Doc3 (0.707) should be selected
        self.assertIn("Title: Doc1", context)
        self.assertIn("Title: Doc3", context)
        self.assertNotIn("Title: Doc2", context)

    @patch("src.core.full_context_baseline.genai.Client")
    def test_retrieve_none_embeddings(self, mock_client_class):
        # Test handling of API failure where embeddings are None
        mock_client = mock_client_class.return_value
        mock_response = MagicMock()
        mock_response.embeddings = None
        mock_client.models.embed_content.return_value = mock_response

        rag = RagBaseline()
        example = HotpotQAExample(
            id="test",
            question="Q",
            answer="A",
            context=HotpotQAContext(title=["D1"], sentences=[["S1"]]),
        )

        context = rag.retrieve(example)
        self.assertEqual(context, "")

    @patch("src.core.full_context_baseline.genai.Client")
    def test_predict(self, mock_client_class):
        mock_client = mock_client_class.return_value
        rag = RagBaseline()

        # Mock retrieve to avoid complex embedding setup for this test
        with patch.object(
            rag, "retrieve", return_value="Mock Context"
        ) as mock_retrieve:
            mock_response = MagicMock()
            mock_part = MagicMock()
            mock_part.text = "Mock Answer"
            mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
            mock_client.models.generate_content.return_value = mock_response

            example = HotpotQAExample(
                id="test",
                question="Q",
                answer="A",
                context=HotpotQAContext(title=[], sentences=[]),
            )

            prediction = rag.predict(example)

            self.assertEqual(prediction, "Mock Answer")
            mock_retrieve.assert_called_once_with(example)

            # Verify generate_content called with correct prompt containing context
            call_args = mock_client.models.generate_content.call_args
            self.assertIn("Mock Context", call_args.kwargs["contents"])


if __name__ == "__main__":
    unittest.main()

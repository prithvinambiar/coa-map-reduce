import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
from src.core.full_context_baseline import AsyncFullContextBaseline
from src.core.dataset import HotpotQAExample, HotpotQAContext


class TestAsyncFullContextBaseline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Patch the environment variable to ensure the class can be instantiated
        self.api_key_patcher = patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"})
        self.api_key_patcher.start()

    def tearDown(self):
        self.api_key_patcher.stop()

    def test_init_no_key(self):
        # Verify that ValueError is raised if API key is missing
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                AsyncFullContextBaseline()

    def test_format_context(self):
        # We mock the client to avoid side effects during instantiation
        with patch("src.core.full_context_baseline.genai.Client"):
            baseline = AsyncFullContextBaseline()

        example = HotpotQAExample(
            id="test_id",
            question="Test Question",
            answer="Test Answer",
            context=HotpotQAContext(
                title=["Doc 1", "Doc 2"],
                sentences=[["Sentence 1a.", "Sentence 1b."], ["Sentence 2a."]],
            ),
        )

        expected_output = (
            "Title: Doc 1\nSentence 1a. Sentence 1b.\n\nTitle: Doc 2\nSentence 2a."
        )
        self.assertEqual(baseline.format_context(example), expected_output)

    @patch("src.core.full_context_baseline.genai.Client")
    async def test_run_batch(self, mock_client_class):
        mock_client_instance = mock_client_class.return_value
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Batch Answer"
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_client_instance.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        baseline = AsyncFullContextBaseline()

        examples = [
            HotpotQAExample(
                id="1",
                question="Q1",
                answer="A1",
                context=HotpotQAContext(title=[], sentences=[]),
            ),
            HotpotQAExample(
                id="2",
                question="Q2",
                answer="A2",
                context=HotpotQAContext(title=[], sentences=[]),
            ),
        ]

        results = await baseline.run_batch(examples)

        self.assertEqual(len(results), 2)
        self.assertEqual(results, ["Batch Answer", "Batch Answer"])
        self.assertEqual(mock_client_instance.aio.models.generate_content.call_count, 2)


if __name__ == "__main__":
    unittest.main()

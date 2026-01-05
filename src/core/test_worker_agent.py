import unittest
from unittest.mock import patch, MagicMock
import os
from src.core.worker_agent import WorkerAgent


class TestWorkerAgent(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"})
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    @patch("src.core.worker_agent.genai.Client")
    def test_process_first_chunk(self, mock_client_cls):
        # Test the first worker in the chain (empty previous_cu)
        mock_client = mock_client_cls.return_value
        mock_response = MagicMock()
        mock_response.text = "Found answer X."
        mock_client.models.generate_content.return_value = mock_response

        agent = WorkerAgent()
        result = agent.process("Chunk text", "Question?", "")

        self.assertEqual(result, "Found answer X.")
        self.assertIn(
            "Previous Communication Unit:\nNone",
            mock_client.models.generate_content.call_args.kwargs["contents"],
        )

    @patch("src.core.worker_agent.genai.Client")
    def test_process_subsequent_chunk(self, mock_client_cls):
        # Test a subsequent worker (existing previous_cu)
        mock_client = mock_client_cls.return_value
        mock_response = MagicMock()
        mock_response.text = "Found answer X and Y."
        mock_client.models.generate_content.return_value = mock_response

        agent = WorkerAgent()
        result = agent.process("Chunk text 2", "Question?", "Found answer X.")

        self.assertEqual(result, "Found answer X and Y.")
        self.assertIn(
            "Previous Communication Unit:\nFound answer X.",
            mock_client.models.generate_content.call_args.kwargs["contents"],
        )


if __name__ == "__main__":
    unittest.main()

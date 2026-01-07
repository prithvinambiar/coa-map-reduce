import unittest
from unittest.mock import patch, MagicMock
import os
from src.core.manager_agent import ManagerAgent


class TestManagerAgent(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"})
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    @patch("src.core.manager_agent.genai.Client")
    def test_generate_answer(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Final Answer"
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_client.models.generate_content.return_value = mock_response

        agent = ManagerAgent()
        result = agent.generate_answer("Question?", "Summary of info")

        self.assertEqual(result, "Final Answer")

        # Verify prompt content
        call_args = mock_client.models.generate_content.call_args
        prompt_content = call_args.kwargs["contents"]
        self.assertIn("Question: Question?", prompt_content)
        self.assertIn("Final Communication Unit:\nSummary of info", prompt_content)

    def test_init_no_key(self):
        # Verify that ValueError is raised if API key is missing
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                ManagerAgent()


if __name__ == "__main__":
    unittest.main()

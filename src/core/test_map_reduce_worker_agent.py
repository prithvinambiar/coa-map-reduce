import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
from src.core.map_reduce_worker_agent import MapReduceWorkerAgent, NO_INFO_TOKEN


class TestMapReduceWorkerAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patcher = patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"})
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    @patch("src.core.map_reduce_worker_agent.genai.Client")
    def test_init(self, mock_client_cls):
        agent = MapReduceWorkerAgent(model_name="custom-model")
        self.assertEqual(agent.model_name, "custom-model")
        mock_client_cls.assert_called_once_with(api_key="fake_key")

    def test_init_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                MapReduceWorkerAgent()

    @patch("src.core.map_reduce_worker_agent.genai.Client")
    async def test_async_process_success(self, mock_client_cls):
        # Setup Mock Response
        mock_client = mock_client_cls.return_value
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Extracted info"
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        agent = MapReduceWorkerAgent()

        # Call the new async method
        result = await agent.async_process("Chunk text", "Question?")

        self.assertEqual(result, "Extracted info")

        # Verify prompt construction
        call_args = mock_client.aio.models.generate_content.call_args
        prompt = call_args.kwargs["contents"]
        self.assertIn("Chunk text", prompt)
        self.assertIn("Question?", prompt)
        self.assertIn(NO_INFO_TOKEN, prompt)

    @patch("src.core.map_reduce_worker_agent.genai.Client")
    async def test_async_process_no_info(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = NO_INFO_TOKEN
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        agent = MapReduceWorkerAgent()
        result = await agent.async_process("Chunk text", "Question?")
        self.assertEqual(result, NO_INFO_TOKEN)

    @patch("src.core.map_reduce_worker_agent.genai.Client")
    async def test_async_process_empty_response(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.aio.models.generate_content = AsyncMock(return_value=None)

        agent = MapReduceWorkerAgent()
        result = await agent.async_process("Chunk text", "Question?")
        self.assertEqual(result, NO_INFO_TOKEN)

    @patch("src.core.map_reduce_worker_agent.genai.Client")
    async def test_async_process_exception(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("API Error")
        )

        agent = MapReduceWorkerAgent()
        result = await agent.async_process("Chunk text", "Question?")
        self.assertEqual(result, NO_INFO_TOKEN)


if __name__ == "__main__":
    unittest.main()

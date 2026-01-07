import unittest
from unittest.mock import patch, AsyncMock
import os
from src.core.coa_map_reduce import CoAMapReduce
from src.core.dataset import HotpotQAExample, HotpotQAContext


class TestCoAMapReduce(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"})
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    @patch("src.core.coa_map_reduce.ManagerAgent")
    @patch("src.core.coa_map_reduce.MapReduceWorkerAgent")
    def test_predict_flow(self, mock_worker_cls, mock_manager_cls):
        # Setup Mocks
        mock_worker = mock_worker_cls.return_value
        # process is async, so we use AsyncMock
        mock_worker.process = AsyncMock(side_effect=["Info 1", "Info 2"])

        mock_manager = mock_manager_cls.return_value
        mock_manager.generate_answer.return_value = "Final Answer"

        coa = CoAMapReduce()

        example = HotpotQAExample(
            id="test",
            question="Q",
            answer="A",
            context=HotpotQAContext(title=["D1", "D2"], sentences=[["S1"], ["S2"]]),
        )

        # Execute
        result = coa.predict(example)

        # Verify
        self.assertEqual(result, "Final Answer")

        # Verify Worker calls
        # Since it runs in asyncio.run, we just check if it was called correctly
        self.assertEqual(mock_worker.process.call_count, 2)

        # Verify Manager call
        # The results "Info 1" and "Info 2" should be joined
        expected_cu = "Info 1\nInfo 2"
        mock_manager.generate_answer.assert_called_once_with("Q", final_cu=expected_cu)

    @patch("src.core.coa_map_reduce.ManagerAgent")
    @patch("src.core.coa_map_reduce.MapReduceWorkerAgent")
    def test_predict_filtering(self, mock_worker_cls, mock_manager_cls):
        # Test filtering of "None" or empty results
        mock_worker = mock_worker_cls.return_value
        mock_worker.process = AsyncMock(side_effect=["Info 1", "None", ""])

        mock_manager = mock_manager_cls.return_value
        mock_manager.generate_answer.return_value = "Final Answer"

        coa = CoAMapReduce()
        example = HotpotQAExample(
            id="test",
            question="Q",
            answer="A",
            context=HotpotQAContext(
                title=["D1", "D2", "D3"], sentences=[["S1"], ["S2"], ["S3"]]
            ),
        )

        coa.predict(example)

        # Only "Info 1" should be passed to manager
        mock_manager.generate_answer.assert_called_once_with("Q", final_cu="Info 1")


if __name__ == "__main__":
    unittest.main()

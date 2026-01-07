import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
from src.core.coa_map_reduce import AsyncCoAMapReduce
from src.core.dataset import HotpotQAExample, HotpotQAContext


class TestAsyncCoAMapReduce(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patcher = patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"})
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    async def test_async_predict_flow(self):
        """
        Tests the Map-Reduce flow for a single sample:
        Map (Worker) -> Reduce (Filter/Join) -> Manager
        """
        # 1. Setup Mocks
        mock_worker = MagicMock()
        # async_process is called for each chunk
        mock_worker.async_process = AsyncMock(side_effect=["Info 1", "Info 2"])

        mock_manager = MagicMock()
        mock_manager.async_generate_answer = AsyncMock(return_value="Final Answer")

        coa = AsyncCoAMapReduce()

        example = HotpotQAExample(
            id="test",
            question="Q",
            answer="A",
            context=HotpotQAContext(title=["D1", "D2"], sentences=[["S1"], ["S2"]]),
        )

        # 2. Execute directly (injecting mocks)
        result = await coa.async_predict(example, mock_worker, mock_manager)

        # 3. Verify Result
        self.assertEqual(result, "Final Answer")

        # 4. Verify Worker calls
        # Should be called twice (once for each chunk)
        self.assertEqual(mock_worker.async_process.call_count, 2)

        # 5. Verify Manager call
        # The results "Info 1" and "Info 2" should be joined with newline
        expected_cu = "Info 1\nInfo 2"
        mock_manager.async_generate_answer.assert_called_once_with(
            "Q", final_cu=expected_cu
        )

    async def test_async_predict_filtering(self):
        """
        Tests that "NO_INFO" and empty strings are filtered out before reaching the manager.
        """
        mock_worker = MagicMock()
        # Returns: Valid, NO_INFO, Empty String
        mock_worker.async_process = AsyncMock(side_effect=["Info 1", "NO_INFO", ""])

        mock_manager = MagicMock()
        mock_manager.async_generate_answer = AsyncMock(return_value="Final Answer")

        coa = AsyncCoAMapReduce()

        # 3 Chunks to trigger the 3 side_effects
        example = HotpotQAExample(
            id="test",
            question="Q",
            answer="A",
            context=HotpotQAContext(
                title=["D1", "D2", "D3"], sentences=[["S1"], ["S2"], ["S3"]]
            ),
        )

        await coa.async_predict(example, mock_worker, mock_manager)

        # Only "Info 1" should be passed to manager. The others are filtered.
        mock_manager.async_generate_answer.assert_called_once_with(
            "Q", final_cu="Info 1"
        )

    @patch("src.core.coa_map_reduce.ManagerAgent")
    @patch("src.core.coa_map_reduce.MapReduceWorkerAgent")
    async def test_run_batch(self, mock_worker_cls, mock_manager_cls):
        """
        Tests that run_batch correctly instantiates agents and orchestrates the batch.
        """
        # Setup Class Mocks to return Instance Mocks
        mock_worker_instance = mock_worker_cls.return_value
        mock_worker_instance.async_process = AsyncMock(return_value="Info")

        mock_manager_instance = mock_manager_cls.return_value
        mock_manager_instance.async_generate_answer = AsyncMock(
            return_value="Batch Answer"
        )

        coa = AsyncCoAMapReduce()

        examples = [
            HotpotQAExample(
                id="1",
                question="Q1",
                answer="A1",
                context=HotpotQAContext(title=["T"], sentences=[["S"]]),
            ),
            HotpotQAExample(
                id="2",
                question="Q2",
                answer="A2",
                context=HotpotQAContext(title=["T"], sentences=[["S"]]),
            ),
        ]

        # Execute Batch
        results = await coa.run_batch(examples)

        # Verify Results
        self.assertEqual(len(results), 2)
        self.assertEqual(results, ["Batch Answer", "Batch Answer"])

        # Verify that agents were instantiated (meaning fresh clients were created)
        mock_worker_cls.assert_called()
        mock_manager_cls.assert_called()


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
from src.core.coa_baseline import AsyncCoABaseline
from src.core.dataset import HotpotQAExample, HotpotQAContext


class TestAsyncCoABaseline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patcher = patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"})
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    async def test_predict_async_flow(self):
        # Setup mocks for agents passed into predict_async
        mock_worker = MagicMock()
        mock_worker.async_process = AsyncMock(
            side_effect=["CU_after_chunk1", "CU_after_chunk2"]
        )

        mock_manager = MagicMock()
        mock_manager.async_generate_answer = AsyncMock(return_value="Final Answer")

        baseline = AsyncCoABaseline()

        example = HotpotQAExample(
            id="test_id",
            question="Question?",
            answer="Answer",
            context=HotpotQAContext(
                title=["Doc1", "Doc2"], sentences=[["Sentence 1."], ["Sentence 2."]]
            ),
        )

        # Execute
        prediction = await baseline.predict_async(example, mock_worker, mock_manager)

        # Verify
        self.assertEqual(prediction, "Final Answer")

        # Check Worker calls
        self.assertEqual(mock_worker.async_process.call_count, 2)
        # Check args for first call
        mock_worker.async_process.assert_any_call(
            "Title: Doc1\nSentence 1.", "Question?", previous_cu=""
        )
        # Check args for second call
        mock_worker.async_process.assert_any_call(
            "Title: Doc2\nSentence 2.", "Question?", previous_cu="CU_after_chunk1"
        )

        # Check Manager call
        mock_manager.async_generate_answer.assert_called_once_with(
            "Question?", final_cu="CU_after_chunk2"
        )

    async def test_predict_async_exception(self):
        mock_worker = MagicMock()
        mock_worker.async_process = AsyncMock(side_effect=Exception("Worker Error"))
        mock_manager = MagicMock()

        baseline = AsyncCoABaseline()
        example = HotpotQAExample(
            id="test_id",
            question="Q",
            answer="A",
            context=HotpotQAContext(title=["T"], sentences=[["S"]]),
        )

        prediction = await baseline.predict_async(example, mock_worker, mock_manager)
        self.assertEqual(prediction, "")

    @patch("src.core.coa_baseline.ManagerAgent")
    @patch("src.core.coa_baseline.WorkerAgent")
    async def test_run_batch(self, mock_worker_cls, mock_manager_cls):
        # Mock the agents created inside run_batch
        mock_worker_instance = mock_worker_cls.return_value
        mock_manager_instance = mock_manager_cls.return_value

        # We can mock predict_async to verify batch processing without running the chain logic
        baseline = AsyncCoABaseline()

        # Patch predict_async on the instance
        baseline.predict_async = AsyncMock(return_value="Batch Answer")

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

        self.assertEqual(results, ["Batch Answer", "Batch Answer"])
        self.assertEqual(baseline.predict_async.call_count, 2)

        # Verify agents were instantiated
        mock_worker_cls.assert_called_once()
        mock_manager_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()

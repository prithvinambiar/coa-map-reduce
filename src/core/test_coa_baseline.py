import unittest
from unittest.mock import patch
from src.core.coa_baseline import CoABaseline
from src.core.dataset import HotpotQAExample, HotpotQAContext


class TestCoABaseline(unittest.TestCase):
    @patch("src.core.coa_baseline.WorkerAgent")
    @patch("src.core.coa_baseline.ManagerAgent")
    def test_predict_flow(self, mock_manager_cls, mock_worker_cls):
        # Setup mocks
        mock_worker = mock_worker_cls.return_value
        mock_manager = mock_manager_cls.return_value

        # Simulate worker updating the CU across 2 chunks
        # 1. Input: (Chunk1, Q, "") -> Output: "CU_after_chunk1"
        # 2. Input: (Chunk2, Q, "CU_after_chunk1") -> Output: "CU_after_chunk2"
        mock_worker.process.side_effect = ["CU_after_chunk1", "CU_after_chunk2"]

        mock_manager.generate_answer.return_value = "Final Answer"

        baseline = CoABaseline()

        example = HotpotQAExample(
            id="test_id",
            question="Question?",
            answer="Answer",
            context=HotpotQAContext(
                title=["Doc1", "Doc2"], sentences=[["Sentence 1."], ["Sentence 2."]]
            ),
        )

        # Execute
        prediction = baseline.predict(example)

        # Verify
        self.assertEqual(prediction, "Final Answer")

        # Check Worker calls
        self.assertEqual(mock_worker.process.call_count, 2)
        mock_worker.process.assert_any_call(
            "Title: Doc1\nSentence 1.", "Question?", previous_cu=""
        )
        mock_worker.process.assert_any_call(
            "Title: Doc2\nSentence 2.", "Question?", previous_cu="CU_after_chunk1"
        )

        # Check Manager call
        mock_manager.generate_answer.assert_called_once_with(
            "Question?", final_cu="CU_after_chunk2"
        )


if __name__ == "__main__":
    unittest.main()

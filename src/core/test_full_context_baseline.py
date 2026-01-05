import unittest
from unittest.mock import patch, MagicMock
import os
from src.core.full_context_baseline import FullContextBaseline
import io
from src.core.full_context_baseline import FullContextBaseline, evaluate
from src.core.dataset import HotpotQAExample, HotpotQAContext


class TestFullContextBaseline(unittest.TestCase):
    def setUp(self):
        # Patch the environment variable to ensure the class can be instantiated
        self.api_key_patcher = patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"})
        self.api_key_patcher.start()

    def tearDown(self):
        self.api_key_patcher.stop()

    @patch("src.core.full_context_baseline.genai.Client")
    def test_init(self, mock_client):
        baseline = FullContextBaseline(model_name="custom-model")
        mock_client.assert_called_once_with(api_key="fake_key")
        self.assertEqual(baseline.model_name, "custom-model")

    def test_init_no_key(self):
        # Verify that ValueError is raised if API key is missing
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                FullContextBaseline()

    def test_format_context(self):
        # We mock the client to avoid side effects during instantiation
        with patch("src.core.full_context_baseline.genai.Client"):
            baseline = FullContextBaseline()

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
    def test_predict(self, mock_client_class):
        # Setup the mock client and response
        mock_client_instance = mock_client_class.return_value
        mock_response = MagicMock()
        mock_response.text = "Expected Answer"
        mock_client_instance.models.generate_content.return_value = mock_response

        baseline = FullContextBaseline()

        example = HotpotQAExample(
            id="test_id",
            question="What is X?",
            answer="Y",
            context=HotpotQAContext(title=["T"], sentences=[["S"]]),
        )

        # Execute predict
        prediction = baseline.predict(example)

        # Verify the result and that the API was called correctly
        self.assertEqual(prediction, "Expected Answer")
        mock_client_instance.models.generate_content.assert_called_once()
        call_args = mock_client_instance.models.generate_content.call_args
        self.assertIn("What is X?", call_args.kwargs["contents"])

    @patch("src.core.full_context_baseline.load_hotpot_qa_eval")
    @patch("src.core.full_context_baseline.FullContextBaseline")
    def test_evaluate(self, mock_baseline_cls, mock_load_dataset):
        # Mock dataset with 2 examples
        mock_example_1 = MagicMock()
        mock_example_1.answer = "Apple"
        mock_example_2 = MagicMock()
        mock_example_2.answer = "Banana"
        mock_load_dataset.return_value = iter([mock_example_1, mock_example_2])

        # Mock baseline instance
        mock_instance = mock_baseline_cls.return_value
        # Predict: 1st Correct (EM=1), 2nd Incorrect (EM=0)
        mock_instance.predict.side_effect = ["Apple", "Orange"]

        metrics = evaluate(num_samples=2)

        self.assertEqual(mock_instance.predict.call_count, 2)
        # Avg EM: (1.0 + 0.0) / 2 = 0.5
        self.assertEqual(metrics.em, 0.5)
        self.assertEqual(metrics.f1, 0.5)


if __name__ == "__main__":
    unittest.main()

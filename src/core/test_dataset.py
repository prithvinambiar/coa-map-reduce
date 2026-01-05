import unittest
from unittest.mock import patch
from src.core.dataset import load_hotpot_qa_eval, HotpotQAExample


class TestDataset(unittest.TestCase):
    @patch("src.core.dataset.load_dataset")
    def test_load_hotpot_qa_eval(self, mock_load_dataset):
        # Mock data simulating the structure returned by the datasets library
        # We include extra fields (like 'type', 'level') to ensure the model handles them (ignores them)
        mock_data = [
            {
                "id": "5a7a06935542990198eaf050",
                "question": "Which magazine was published first, Arthur's Magazine or First for Women?",
                "answer": "Arthur's Magazine",
                "context": {
                    "title": ["Arthur's Magazine", "First for Women"],
                    "sentences": [
                        [
                            "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century."
                        ],
                        [
                            "First for Women is a woman's magazine published by Bauer Media Group in the USA."
                        ],
                    ],
                },
            }
        ]

        # Configure the mock to return the list when iterated
        mock_load_dataset.return_value = mock_data

        # Execute the function
        iterator = load_hotpot_qa_eval()
        results = list(iterator)

        # Verify the results
        self.assertEqual(len(results), 1)
        example = results[0]
        self.assertIsInstance(example, HotpotQAExample)
        self.assertEqual(example.id, "5a7a06935542990198eaf050")
        self.assertEqual(
            example.question,
            "Which magazine was published first, Arthur's Magazine or First for Women?",
        )
        self.assertEqual(example.context.title[0], "Arthur's Magazine")

        # Verify load_dataset was called with the correct arguments
        mock_load_dataset.assert_called_once_with(
            "hotpot_qa", "distractor", split="validation"
        )


if __name__ == "__main__":
    unittest.main()

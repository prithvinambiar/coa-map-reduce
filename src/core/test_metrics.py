import unittest
from src.core.metrics import normalize_answer, exact_match_score, f1_score


class TestMetrics(unittest.TestCase):
    def test_normalize_answer(self):
        self.assertEqual(normalize_answer("The Apple Inc."), "apple inc")
        self.assertEqual(normalize_answer("apple-pie"), "applepie")
        self.assertEqual(normalize_answer("  Spaces  "), "spaces")

    def test_exact_match_score(self):
        self.assertEqual(exact_match_score("Apple", "apple"), 1.0)
        self.assertEqual(
            exact_match_score("The Apple", "apple"), 1.0
        )  # Article removal
        self.assertEqual(exact_match_score("Apple Inc", "Apple"), 0.0)

    def test_f1_score(self):
        # Perfect match
        self.assertEqual(f1_score("apple pie", "apple pie"), 1.0)
        # No overlap
        self.assertEqual(f1_score("apple", "banana"), 0.0)
        # Partial overlap
        # Pred: "apple pie", Gold: "apple"
        # Precision: 1/2, Recall: 1/1 -> F1: 2*(0.5*1)/(0.5+1) = 1/1.5 = 0.666...
        self.assertAlmostEqual(f1_score("apple pie", "apple"), 0.6666666666666666)
        # Order shouldn't matter for bag of words F1
        self.assertEqual(f1_score("pie apple", "apple pie"), 1.0)


if __name__ == "__main__":
    unittest.main()

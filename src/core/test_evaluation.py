import unittest
from src.core.evaluation import evaluate_batch
from src.core.dataset import HotpotQAExample, HotpotQAContext


class TestEvaluation(unittest.TestCase):
    def test_evaluate_batch(self):
        # Create dummy examples
        examples = [
            HotpotQAExample(
                id="1",
                question="Q1",
                answer="Apple",
                context=HotpotQAContext(title=[], sentences=[]),
            ),
            HotpotQAExample(
                id="2",
                question="Q2",
                answer="Banana",
                context=HotpotQAContext(title=[], sentences=[]),
            ),
            HotpotQAExample(
                id="3",
                question="Q3",
                answer="Cherry",
                context=HotpotQAContext(title=[], sentences=[]),
            ),
        ]

        # Predictions:
        # 1. Exact match
        # 2. Partial match (Banana Split vs Banana) -> EM=0, F1 check
        # 3. No match
        predictions = ["Apple", "Banana Split", "Date"]

        metrics = evaluate_batch(examples, predictions)

        # Check EM
        # 1: 1.0, 2: 0.0, 3: 0.0 -> Avg EM = 1/3
        self.assertAlmostEqual(metrics.em, 1 / 3)

        # Check F1
        # 1: 1.0
        # 2: "banana split" vs "banana". Common: "banana". Prec: 1/2, Rec: 1/1. F1 = 2*(0.5*1)/(1.5) = 0.6666
        # 3: 0.0
        # Avg F1 = (1.0 + 0.6666 + 0.0) / 3 = 0.5555
        self.assertAlmostEqual(metrics.f1, (1.0 + 2 / 3 + 0.0) / 3)


if __name__ == "__main__":
    unittest.main()
import time
from typing import Protocol, List
from src.core.dataset import load_hotpot_qa_eval, HotpotQAExample
from src.core.metrics import exact_match_score, f1_score, PerformanceMetric


class BaselineModel(Protocol):
    def predict(self, example: HotpotQAExample) -> str: ...


def evaluate(baseline: BaselineModel, num_samples: int) -> PerformanceMetric:
    dataset = load_hotpot_qa_eval()

    total_em = 0.0
    total_f1 = 0.0
    total_latency = 0.0

    print(f"Running evaluation on {num_samples} samples...")

    for i in range(num_samples):
        try:
            example = next(dataset)
        except StopIteration:
            print(f"Dataset exhausted at {i} samples.")
            num_samples = i
            break

        start_time = time.time()
        prediction = baseline.predict(example)
        end_time = time.time()
        latency = end_time - start_time

        em = exact_match_score(prediction, example.answer)
        f1 = f1_score(prediction, example.answer)

        total_em += em
        total_f1 += f1
        total_latency += latency

        print(
            f"[{i + 1}] EM: {em} | F1: {f1:.4f} | Latency: {latency:.2f}s | Pred: {prediction} | Gold: {example.answer}"
        )

    avg_em = total_em / num_samples if num_samples > 0 else 0.0
    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0.0
    avg_latency = total_latency / num_samples if num_samples > 0 else 0.0

    print(f"\nAverage EM: {avg_em:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
    print(f"Average Latency: {avg_latency:.4f}s")

    return PerformanceMetric(em=avg_em, f1=avg_f1, latency=avg_latency)


def evaluate_batch(
    samples: List[HotpotQAExample], predictions: List[str]
) -> PerformanceMetric:
    """
    Evaluates a batch of predictions against the ground truth samples.
    """
    total_em = 0.0
    total_f1 = 0.0
    num_samples = len(samples)

    if len(predictions) != num_samples:
        print(
            f"Warning: Number of predictions ({len(predictions)}) does not match number of samples ({num_samples})."
        )
        num_samples = min(num_samples, len(predictions))

    for i in range(num_samples):
        sample = samples[i]
        prediction = predictions[i]
        em = exact_match_score(prediction, sample.answer)
        f1 = f1_score(prediction, sample.answer)
        total_em += em
        total_f1 += f1
        print(
            f"[{i + 1}] EM: {em} | F1: {f1:.4f} | Pred: {prediction} | Gold: {sample.answer}"
        )

    avg_em = total_em / num_samples if num_samples > 0 else 0.0
    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0.0

    print(f"\nAverage EM: {avg_em:.4f}")
    print(f"Average F1: {avg_f1:.4f}")

    return PerformanceMetric(em=avg_em, f1=avg_f1, latency=0.0)

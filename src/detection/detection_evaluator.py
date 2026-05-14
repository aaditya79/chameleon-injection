"""
Utility for batch-evaluating detectors and computing detection metrics.
"""

from __future__ import annotations

from typing import List, Tuple

from src.detection.static_detector import DetectionResult, StaticDetector


class DetectionEvaluator:
    """
    Runs a detector over a batch of (context, instruction, ground_truth) triples
    and collects results.

    Args:
        detector: A StaticDetector or AugmentedDetector instance.
    """

    def __init__(self, detector: StaticDetector) -> None:
        self.detector = detector

    def evaluate_batch(
        self,
        samples: List[Tuple[str, str, bool]],
    ) -> List[DetectionResult]:
        """
        Evaluate the detector on a list of samples.

        Args:
            samples: List of (context, instruction, ground_truth) tuples.
                ground_truth=True means the context IS injected.

        Returns:
            List of DetectionResult objects with is_correct set.
        """
        results = []
        for context, instruction, ground_truth in samples:
            result = self.detector.detect(
                context=context,
                instruction=instruction,
                ground_truth=ground_truth,
            )
            results.append(result)
        return results

    def idr(self, results: List[DetectionResult]) -> float:
        """
        Injection Detection Rate: fraction of injected samples correctly flagged.

        Only counts samples where is_correct is not None and the ground truth
        indicates injection (detected by checking if is_correct implies the
        prediction was INJECTED and matched ground truth=True).
        """
        injected_correct = sum(
            1 for r in results
            if r.verdict == "INJECTED" and r.is_correct is True
        )
        total_injected = sum(1 for r in results if r.verdict in ("INJECTED", "CLEAN"))
        return injected_correct / max(total_injected, 1)

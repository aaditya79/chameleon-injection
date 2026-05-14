"""
Detector agent wrapper (convenience re-export for experiment scripts).
"""

from src.detection.static_detector import StaticDetector, DetectionResult
from src.detection.augmented_detector import AugmentedDetector

__all__ = ["StaticDetector", "AugmentedDetector", "DetectionResult"]

"""Social distance detection: person detection, bird's-eye mapping and proximity risk.

Pipeline: detect people (YOLOv3 or SSD MobileNet) -> take the bottom-centre of each box
(the point where the person stands) -> map it to a calibrated bird's-eye ground plane ->
measure pairwise distances in centimetres -> classify each person as high, low or no risk.
"""

__version__ = "2.0.0"

from .calibration import Calibration, load_calibration, save_calibration
from .detectors import Detection, ModelFilesMissing, SSDMobileNetDetector, YOLOv3Detector, create_detector
from .perspective import BirdEyeTransform
from .risk import RiskLevel, RiskThresholds, assess_risk

__all__ = [
    "BirdEyeTransform",
    "Calibration",
    "Detection",
    "ModelFilesMissing",
    "RiskLevel",
    "RiskThresholds",
    "SSDMobileNetDetector",
    "YOLOv3Detector",
    "assess_risk",
    "create_detector",
    "load_calibration",
    "save_calibration",
]

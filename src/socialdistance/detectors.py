"""Person detectors behind one interface.

Two pre-trained detectors are supported, both run through OpenCV's DNN module on CPU:

* **YOLOv3** (Darknet, COCO classes, person = class 0) at 416x416 input.
* **SSD MobileNet** (Caffe, PASCAL VOC classes, person = class 15).

Each detector returns a list of :class:`Detection` objects for people only, after
confidence filtering and non-maximum suppression (NMS). The post-processing steps are
plain functions (:func:`yolo_postprocess`, :func:`ssd_postprocess`) so they can be
unit-tested with synthetic network outputs, without the model weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

COCO_PERSON_ID = 0
VOC_PERSON_ID = 15

YOLO_FILES = ("yolov3.cfg", "yolov3.weights")
SSD_FILES = ("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")


class ModelFilesMissing(FileNotFoundError):
    """Raised when a detector's model files have not been downloaded yet."""


@dataclass(frozen=True)
class Detection:
    """One detected person. Box is (x, y, w, h) in pixels of the original frame."""

    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the person touches the ground."""
        return (self.x + self.w / 2.0, float(self.y + self.h))


def non_max_suppression(
    boxes: Sequence[Sequence[int]],
    scores: Sequence[float],
    score_threshold: float,
    nms_threshold: float,
) -> list[int]:
    """Indices of boxes kept by OpenCV's NMS (boxes are x, y, w, h)."""
    if len(boxes) == 0:
        return []
    idxs = cv2.dnn.NMSBoxes(
        [list(map(int, b)) for b in boxes], [float(s) for s in scores], score_threshold, nms_threshold
    )
    return sorted(int(i) for i in np.array(idxs).flatten())


def _clip_box(x: float, y: float, w: float, h: float, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = int(round(max(0.0, x)))
    y0 = int(round(max(0.0, y)))
    x1 = int(round(min(float(width), x + w)))
    y1 = int(round(min(float(height), y + h)))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def yolo_postprocess(
    layer_outputs: Sequence[np.ndarray],
    width: int,
    height: int,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.5,
    person_class: int = COCO_PERSON_ID,
) -> list[Detection]:
    """Turn raw YOLOv3 output rows into person detections.

    Each row is ``[cx, cy, w, h, objectness, class_score_0, ..., class_score_79]`` with
    coordinates relative to the image size. As in the original 2021 code, a row is kept
    when its best class is ``person`` and that class score exceeds ``conf_threshold``.
    """
    boxes: list[list[int]] = []
    scores: list[float] = []
    for output in layer_outputs:
        output = np.asarray(output).reshape(-1, np.asarray(output).shape[-1])
        if output.size == 0:
            continue
        class_scores = output[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(output)), class_ids]
        keep = (class_ids == person_class) & (confidences > conf_threshold)
        for row, conf in zip(output[keep], confidences[keep]):
            cx, cy, bw, bh = row[:4] * np.array([width, height, width, height])
            boxes.append(list(_clip_box(cx - bw / 2.0, cy - bh / 2.0, bw, bh, width, height)))
            scores.append(float(conf))
    kept = non_max_suppression(boxes, scores, conf_threshold, nms_threshold)
    return [Detection(*boxes[i], score=scores[i]) for i in kept if boxes[i][2] > 0 and boxes[i][3] > 0]


def ssd_postprocess(
    detections: np.ndarray,
    width: int,
    height: int,
    conf_threshold: float = 0.4,
    nms_threshold: float = 0.5,
    person_class: int = VOC_PERSON_ID,
) -> list[Detection]:
    """Turn the SSD ``DetectionOutput`` blob into person detections.

    The blob has shape ``(1, 1, N, 7)``; each row is
    ``[image_id, class_id, confidence, x_min, y_min, x_max, y_max]`` with relative
    coordinates. The original 2021 SSD script had no NMS; it is applied here so both
    detectors produce comparable, de-duplicated boxes.
    """
    rows = np.asarray(detections).reshape(-1, 7)
    boxes: list[list[int]] = []
    scores: list[float] = []
    for _, class_id, conf, x_min, y_min, x_max, y_max in rows:
        if int(class_id) != person_class or conf <= conf_threshold:
            continue
        x0, y0 = x_min * width, y_min * height
        x1, y1 = x_max * width, y_max * height
        boxes.append(list(_clip_box(x0, y0, x1 - x0, y1 - y0, width, height)))
        scores.append(float(conf))
    kept = non_max_suppression(boxes, scores, conf_threshold, nms_threshold)
    return [Detection(*boxes[i], score=scores[i]) for i in kept if boxes[i][2] > 0 and boxes[i][3] > 0]


class PersonDetector:
    """Common interface: ``detect(frame) -> list[Detection]``."""

    name = "base"

    def __init__(self) -> None:
        self.last_inference_ms = 0.0

    def detect(self, frame: np.ndarray) -> list[Detection]:  # pragma: no cover - interface
        raise NotImplementedError


def _require(paths: Sequence[Path], detector: str) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise ModelFilesMissing(
            f"{detector} model files not found: {names}. "
            "Run `python scripts/download_models.py` to fetch them into models/."
        )


class YOLOv3Detector(PersonDetector):
    """YOLOv3 (Darknet) through OpenCV DNN; same settings as the 2021 project."""

    name = "yolov3"

    def __init__(
        self,
        cfg_path: str | Path,
        weights_path: str | Path,
        input_size: int = 416,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        cfg_path, weights_path = Path(cfg_path), Path(weights_path)
        _require([cfg_path, weights_path], "YOLOv3")
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        start = time.perf_counter()
        outputs = self.net.forward(self.output_layers)
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0
        return yolo_postprocess(outputs, w, h, self.conf_threshold, self.nms_threshold)


class SSDMobileNetDetector(PersonDetector):
    """MobileNet-SSD (Caffe, VOC) through OpenCV DNN.

    The network was trained at 300x300, but it is fully convolutional, and a larger input
    (512x512 by default) finds noticeably more of the small, distant pedestrians typical
    of surveillance footage, at a higher CPU cost.
    """

    name = "ssd"

    def __init__(
        self,
        prototxt_path: str | Path,
        caffemodel_path: str | Path,
        input_size: int = 512,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        prototxt_path, caffemodel_path = Path(prototxt_path), Path(caffemodel_path)
        _require([prototxt_path, caffemodel_path], "SSD MobileNet")
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (self.input_size, self.input_size)),
            0.007843,
            (self.input_size, self.input_size),
            127.5,
        )
        self.net.setInput(blob)
        start = time.perf_counter()
        out = self.net.forward()
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0
        return ssd_postprocess(out, w, h, self.conf_threshold, self.nms_threshold)


DETECTOR_ALIASES = {
    "yolo": "yolov3",
    "yolov3": "yolov3",
    "ssd": "ssd",
    "ssd_mobilenet": "ssd",
    "mobilenet": "ssd",
}


def model_paths(name: str, models_dir: str | Path = "models") -> list[Path]:
    """Expected model file paths for a detector name."""
    key = DETECTOR_ALIASES.get(name.lower())
    if key is None:
        raise ValueError(f"Unknown detector '{name}'. Choose 'yolov3' or 'ssd'.")
    files = YOLO_FILES if key == "yolov3" else SSD_FILES
    return [Path(models_dir) / f for f in files]


def models_available(name: str, models_dir: str | Path = "models") -> bool:
    return all(p.is_file() for p in model_paths(name, models_dir))


def create_detector(
    name: str,
    models_dir: str | Path = "models",
    conf_threshold: float | None = None,
    nms_threshold: float = 0.5,
    input_size: int | None = None,
) -> PersonDetector:
    """Build a detector by name (``yolov3`` or ``ssd``) from files in ``models_dir``."""
    key = DETECTOR_ALIASES.get(name.lower())
    paths = model_paths(name, models_dir)
    if key == "yolov3":
        return YOLOv3Detector(
            *paths,
            input_size=input_size or 416,
            conf_threshold=0.5 if conf_threshold is None else conf_threshold,
            nms_threshold=nms_threshold,
        )
    return SSDMobileNetDetector(
        *paths,
        input_size=input_size or 512,
        conf_threshold=0.4 if conf_threshold is None else conf_threshold,
        nms_threshold=nms_threshold,
    )

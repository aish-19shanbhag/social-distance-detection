"""Detector post-processing (synthetic YOLOv3 and SSD outputs), config loading, pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

from socialdistance.calibration import Calibration, load_calibration, save_calibration
from socialdistance.detectors import (
    Detection,
    ModelFilesMissing,
    PersonDetector,
    create_detector,
    models_available,
    non_max_suppression,
    ssd_postprocess,
    yolo_postprocess,
)

ROOT = Path(__file__).resolve().parents[1]
W, H = 800, 600


# ---------------------------------------------------------------- YOLOv3 rows
def yolo_row(cx, cy, w, h, cls, score, n_classes=80):
    """One YOLOv3 output row: relative box, objectness, 80 class scores."""
    row = np.zeros(5 + n_classes, np.float32)
    row[:5] = [cx / W, cy / H, w / W, h / H, score]
    row[5 + cls] = score
    return row


def test_yolo_keeps_people_and_drops_other_classes():
    out = np.stack(
        [
            yolo_row(200, 300, 50, 120, cls=0, score=0.9),  # person
            yolo_row(600, 300, 80, 60, cls=2, score=0.95),  # car
        ]
    )
    dets = yolo_postprocess([out], W, H)
    assert len(dets) == 1
    d = dets[0]
    assert (d.x, d.y, d.w, d.h) == (175, 240, 50, 120)
    assert d.score == pytest.approx(0.9)
    assert d.foot_point == (200.0, 360.0)


def test_yolo_confidence_threshold_is_strict():
    out = np.stack([yolo_row(200, 300, 50, 120, 0, 0.5), yolo_row(400, 300, 50, 120, 0, 0.51)])
    dets = yolo_postprocess([out], W, H, conf_threshold=0.5)
    assert [round(d.x + d.w / 2) for d in dets] == [400]


def test_yolo_nms_merges_overlapping_boxes_across_output_scales():
    # the same person predicted at two YOLO scales, plus a separate person
    scale1 = np.stack([yolo_row(200, 300, 50, 120, 0, 0.92), yolo_row(500, 300, 50, 120, 0, 0.7)])
    scale2 = np.stack([yolo_row(204, 302, 52, 118, 0, 0.80)])
    dets = yolo_postprocess([scale1, scale2], W, H, nms_threshold=0.5)
    assert len(dets) == 2
    assert max(d.score for d in dets) == pytest.approx(0.92)


def test_yolo_boxes_clipped_to_frame():
    out = np.stack([yolo_row(10, 590, 60, 100, 0, 0.9)])
    (d,) = yolo_postprocess([out], W, H)
    assert d.x == 0 and d.y + d.h <= H


def test_yolo_empty_output():
    assert yolo_postprocess([np.zeros((0, 85), np.float32)], W, H) == []


# ------------------------------------------------------------------ SSD rows
def ssd_blob(rows):
    """SSD DetectionOutput blob: (1, 1, N, 7) of [img, class, conf, x0, y0, x1, y1]."""
    return np.array(rows, np.float32).reshape(1, 1, -1, 7)


def test_ssd_keeps_person_class_15_only():
    blob = ssd_blob(
        [
            [0, 15, 0.9, 0.25, 0.4, 0.3125, 0.6],  # person
            [0, 7, 0.95, 0.5, 0.5, 0.7, 0.7],  # VOC car
            [0, 12, 0.8, 0.1, 0.1, 0.2, 0.2],  # VOC dog
        ]
    )
    dets = ssd_postprocess(blob, W, H)
    assert len(dets) == 1
    assert (dets[0].x, dets[0].y, dets[0].w, dets[0].h) == (200, 240, 50, 120)


def test_ssd_threshold_and_nms():
    blob = ssd_blob(
        [
            [0, 15, 0.90, 0.25, 0.40, 0.3125, 0.60],
            [0, 15, 0.85, 0.26, 0.41, 0.3150, 0.60],  # duplicate of the first
            [0, 15, 0.60, 0.70, 0.40, 0.7600, 0.60],  # second person
            [0, 15, 0.30, 0.50, 0.40, 0.5600, 0.60],  # below 0.4
        ]
    )
    dets = ssd_postprocess(blob, W, H, conf_threshold=0.4, nms_threshold=0.5)
    assert sorted(round(d.score, 2) for d in dets) == [0.6, 0.9]


def test_ssd_clips_out_of_range_coordinates():
    blob = ssd_blob([[0, 15, 0.9, -0.05, 0.8, 0.05, 1.1]])
    (d,) = ssd_postprocess(blob, W, H)
    assert d.x == 0 and d.y + d.h == H


def test_nms_helper_empty():
    assert non_max_suppression([], [], 0.5, 0.5) == []


def test_missing_weights_raise_helpful_error(tmp_path):
    assert not models_available("yolov3", tmp_path)
    with pytest.raises(ModelFilesMissing, match="download_models.py"):
        create_detector("yolov3", tmp_path)
    with pytest.raises(ValueError):
        create_detector("faster-rcnn", tmp_path)


# -------------------------------------------------------------------- config
def test_shipped_vtest_calibration_loads():
    cal = load_calibration(ROOT / "config" / "vtest_calibration.json")
    assert len(cal.roi) == 4 and len(cal.unit_points) == 3
    assert cal.unit_length_cm == 180
    assert cal.frame_size == (768, 576)
    assert "Approximate" in cal.notes


def test_calibration_roundtrip(tmp_path):
    cal = Calibration.from_points(
        [(0, 600), (800, 600), (600, 100), (200, 100), (400, 400), (480, 400), (400, 360), (1, 1)],
        frame_size=(800, 600),
        name="test",
    )
    path = save_calibration(cal, tmp_path / "cal.json")
    data = json.loads(path.read_text())
    assert data["roi"][0] == [0.0, 600.0]
    again = load_calibration(path)
    assert again.roi == cal.roi and again.unit_points == cal.unit_points and again.name == "test"


@pytest.mark.parametrize(
    "bad",
    [
        {"roi": [[0, 0], [1, 0], [1, 1]], "unit_points": [[0, 0], [1, 0], [0, 1]]},  # 3 ROI points
        {"roi": [[0, 0], [1, 0], [1, 1], [0, 1]], "unit_points": [[0, 0], [1, 0]]},  # 2 unit points
        {"roi": [[0, 0], [1, 1], [2, 2], [3, 3]], "unit_points": [[0, 0], [1, 0], [0, 1]]},  # collinear
        {"unit_points": [[0, 0], [1, 0], [0, 1]]},  # missing roi
        {"roi": [[0, 0], [9, 0], [9, 9], [0, 9]], "unit_points": [[0, 0], [1, 0], [0, 1]], "unit_length_cm": 0},
    ],
)
def test_invalid_calibrations_rejected(bad):
    with pytest.raises(ValueError):
        Calibration.from_dict(bad)


# ------------------------------------------------------------------ pipeline
class FakeDetector(PersonDetector):
    """Returns fixed boxes so the pipeline can be tested without model weights."""

    name = "fake"

    def __init__(self, detections):
        super().__init__()
        self.detections = detections

    def detect(self, frame):
        self.last_inference_ms = 1.0
        return list(self.detections)


def test_process_frame_end_to_end_with_identity_calibration():
    from socialdistance.perspective import BirdEyeTransform
    from socialdistance.pipeline import process_frame, render

    cal = Calibration(
        roi=[(0, H), (W, H), (W, 0), (0, 0)],
        unit_points=[(100, 300), (280, 300), (100, 120)],  # 1 px = 1 cm both ways
        frame_size=(W, H),
    )
    t = BirdEyeTransform(cal, (W, H))
    dets = [
        Detection(100, 100, 40, 100, 0.9),  # foot (120, 200)
        Detection(200, 100, 40, 100, 0.9),  # foot (220, 200): 100 cm from #1 -> high
        Detection(600, 300, 40, 100, 0.9),  # far away
    ]
    frame = np.zeros((H, W, 3), np.uint8)
    res = process_frame(frame, FakeDetector(dets), t)
    assert res.counts == {"high": 2, "low": 0, "safe": 1, "total": 3}
    assert res.assessment.min_distance_cm == pytest.approx(100)
    annotated, bird = render(frame, res, t)
    assert annotated.shape[1] == W and annotated.shape[0] > H  # banner added
    assert bird.shape[0] == annotated.shape[0]
    row = res.csv_row(fps=10)
    assert row["high_risk"] == 2 and row["close_pairs"] == 1


def test_process_video_writes_outputs(tmp_path):
    import cv2

    from socialdistance.pipeline import process_video

    video = tmp_path / "tiny.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (W, H))
    for _ in range(4):
        vw.write(np.full((H, W, 3), 90, np.uint8))
    vw.release()
    cal = Calibration(
        roi=[(0, H), (W, H), (W, 0), (0, 0)], unit_points=[(100, 300), (280, 300), (100, 120)], frame_size=(W, H)
    )
    dets = [Detection(100, 100, 40, 100, 0.9), Detection(260, 100, 40, 100, 0.9)]  # 160 cm -> low
    summary = process_video(video, FakeDetector(dets), cal, tmp_path / "out")
    assert summary["frames_processed"] == 4
    assert summary["person_frames"] == {"high": 0, "low": 8, "safe": 0}
    for name in ("annotated.mp4", "birds_eye.mp4", "risk_counts.csv", "summary.json"):
        assert (tmp_path / "out" / name).is_file()
    lines = (tmp_path / "out" / "risk_counts.csv").read_text().strip().splitlines()
    assert len(lines) == 5 and lines[0].startswith("frame,time_s,detected")


# ------------------------------------------------- optional real-model smoke test
@pytest.mark.skipif(not models_available("ssd", ROOT / "models"), reason="SSD model not downloaded")
def test_real_ssd_detects_people_in_sample_video():
    import cv2

    cap = cv2.VideoCapture(str(ROOT / "data" / "vtest.avi"))
    ok, frame = cap.read()
    cap.release()
    assert ok
    dets = create_detector("ssd", ROOT / "models").detect(frame)
    assert len(dets) >= 1
    assert all(0 <= d.x < frame.shape[1] and 0 <= d.y < frame.shape[0] for d in dets)

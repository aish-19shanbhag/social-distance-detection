"""End-to-end processing of frames and videos.

``process_frame`` runs one frame through detection -> foot points -> bird's-eye
ground coordinates -> risk assessment. ``process_video`` loops over a video and writes:

* ``annotated.mp4``: the camera view with coloured boxes, close-pair lines and a banner
* ``birds_eye.mp4``: the top-down map, one frame per processed frame
* ``risk_counts.csv``: per-frame counts (detected, high, low, safe, closest pair, timing)
* ``summary.json``: totals and measured processing speed
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from .calibration import Calibration
from .detectors import Detection, PersonDetector
from .perspective import BirdEyeTransform
from .risk import RiskAssessment, RiskThresholds, assess_risk
from .visualize import annotate_frame, bird_eye_view

CSV_FIELDS = [
    "frame",
    "time_s",
    "detected",
    "high_risk",
    "low_risk",
    "no_risk",
    "close_pairs",
    "min_distance_cm",
    "inference_ms",
    "total_ms",
]


@dataclass
class FrameResult:
    index: int
    detections: list[Detection]
    foot_points: np.ndarray  # image pixels, N x 2
    ground_cm: np.ndarray  # bird's-eye ground plane, N x 2
    assessment: RiskAssessment
    inference_ms: float
    total_ms: float = 0.0

    @property
    def counts(self) -> dict:
        return self.assessment.counts

    def csv_row(self, fps: float) -> dict:
        md = self.assessment.min_distance_cm
        return {
            "frame": self.index,
            "time_s": round(self.index / fps, 3) if fps else "",
            "detected": len(self.detections),
            "high_risk": self.counts["high"],
            "low_risk": self.counts["low"],
            "no_risk": self.counts["safe"],
            "close_pairs": len(self.assessment.violations),
            "min_distance_cm": round(md, 1) if md is not None else "",
            "inference_ms": round(self.inference_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


def process_frame(
    frame: np.ndarray,
    detector: PersonDetector,
    transform: BirdEyeTransform,
    thresholds: RiskThresholds | None = None,
    index: int = 0,
    roi_only: bool = False,
) -> FrameResult:
    """Detect people in one frame and classify their proximity risk."""
    start = time.perf_counter()
    detections = detector.detect(frame)
    feet = np.array([d.foot_point for d in detections], dtype=np.float64).reshape(-1, 2)
    if roi_only and len(detections):
        mask = transform.inside_roi(feet)
        detections = [d for d, keep in zip(detections, mask) if keep]
        feet = feet[mask]
    ground = transform.to_ground_cm(feet)
    assessment = assess_risk(ground, thresholds)
    return FrameResult(
        index=index,
        detections=detections,
        foot_points=feet,
        ground_cm=ground,
        assessment=assessment,
        inference_ms=detector.last_inference_ms,
        total_ms=(time.perf_counter() - start) * 1000.0,
    )


def render(frame: np.ndarray, result: FrameResult, transform: BirdEyeTransform, label: str = ""):
    """Annotated camera frame and bird's-eye canvas for one result."""
    annotated = annotate_frame(frame, result.detections, result.assessment, transform, banner_extra=label)
    bird = bird_eye_view(transform, result.ground_cm, result.assessment, height=annotated.shape[0])
    return annotated, bird


def video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def iter_video(
    video_path: str | Path,
    detector: PersonDetector,
    calibration: Calibration,
    thresholds: RiskThresholds | None = None,
    max_frames: int | None = None,
    start_frame: int = 0,
    every_n: int = 1,
    roi_only: bool = False,
) -> Iterator[tuple[np.ndarray, FrameResult, BirdEyeTransform]]:
    """Yield ``(frame, result, transform)`` for each processed frame."""
    info = video_info(video_path)
    transform = BirdEyeTransform(calibration, (info["width"], info["height"]))
    cap = cv2.VideoCapture(str(video_path))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    index, processed = start_frame, 0
    try:
        while max_frames is None or processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if (index - start_frame) % every_n == 0:
                result = process_frame(frame, detector, transform, thresholds, index, roi_only)
                processed += 1
                yield frame, result, transform
            index += 1
    finally:
        cap.release()


def process_video(
    video_path: str | Path,
    detector: PersonDetector,
    calibration: Calibration,
    output_dir: str | Path,
    thresholds: RiskThresholds | None = None,
    max_frames: int | None = None,
    start_frame: int = 0,
    every_n: int = 1,
    roi_only: bool = False,
    write_videos: bool = True,
    progress: Callable[[int, FrameResult], None] | None = None,
) -> dict:
    """Process a video and write annotated + bird's-eye videos, a CSV and a summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info = video_info(video_path)
    out_fps = info["fps"] / every_n
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers: dict[str, cv2.VideoWriter] = {}
    rows: list[dict] = []
    totals = {"high": 0, "low": 0, "safe": 0, "detections": 0, "frames_with_violation": 0}
    wall_start = time.perf_counter()
    pipeline_ms: list[float] = []

    for frame, result, transform in iter_video(
        video_path, detector, calibration, thresholds, max_frames, start_frame, every_n, roi_only
    ):
        pipeline_ms.append(result.total_ms)
        rows.append(result.csv_row(info["fps"]))
        for k in ("high", "low", "safe"):
            totals[k] += result.counts[k]
        totals["detections"] += len(result.detections)
        totals["frames_with_violation"] += int(bool(result.assessment.violations))
        if write_videos:
            annotated, bird = render(frame, result, transform, f"frame {result.index}")
            if not writers:
                writers["annotated"] = cv2.VideoWriter(
                    str(output_dir / "annotated.mp4"), fourcc, out_fps, annotated.shape[1::-1]
                )
                writers["birds_eye"] = cv2.VideoWriter(
                    str(output_dir / "birds_eye.mp4"), fourcc, out_fps, bird.shape[1::-1]
                )
            writers["annotated"].write(annotated)
            writers["birds_eye"].write(bird)
        if progress:
            progress(len(rows), result)

    for w in writers.values():
        w.release()
    wall_s = time.perf_counter() - wall_start

    with open(output_dir / "risk_counts.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    summary = {
        "video": Path(video_path).name,
        "detector": detector.name,
        "frames_processed": n,
        "frame_size": [info["width"], info["height"]],
        "avg_people_per_frame": round(totals["detections"] / n, 2) if n else 0,
        "person_frames": {k: totals[k] for k in ("high", "low", "safe")},
        "frames_with_close_pair": totals["frames_with_violation"],
        "pipeline_fps": round(1000.0 / float(np.mean(pipeline_ms)), 2) if pipeline_ms else 0,
        "pipeline_ms_mean": round(float(np.mean(pipeline_ms)), 1) if pipeline_ms else 0,
        "wall_clock_s_including_rendering": round(wall_s, 2),
        "end_to_end_fps_including_rendering": round(n / wall_s, 2) if wall_s else 0,
        "thresholds_cm": {
            "high": (thresholds or RiskThresholds()).high_cm,
            "low": (thresholds or RiskThresholds()).low_cm,
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary

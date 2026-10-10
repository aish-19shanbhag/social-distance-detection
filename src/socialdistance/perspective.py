"""Perspective (homography) transform from the camera view to a bird's-eye ground plane.

Pixel distances in a camera image are not real distances: people further from the
camera look closer together. The ROI quadrilateral, which is a rectangle on the real
ground, is mapped to an axis-aligned rectangle with a 3x3 homography. In that warped
space, equal ground distances are equal pixel distances along each axis, but the
horizontal and vertical scales differ. The 3 unit points give both scales: how many
warped pixels make up ``unit_length_cm`` horizontally (``px_per_unit_w``) and
vertically (``px_per_unit_h``). Every position is then converted to centimetres, and
distances become ordinary Euclidean distances on the ground.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .calibration import Calibration


class BirdEyeTransform:
    def __init__(self, calibration: Calibration, frame_size: tuple[int, int]) -> None:
        width, height = frame_size
        self.calibration = calibration.scaled_to(width, height)
        self.frame_size = (int(width), int(height))
        src = np.float32(self.calibration.roi)
        # Same destination as the 2021 code: ROI -> full frame rectangle.
        dst = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
        self.matrix = cv2.getPerspectiveTransform(src, dst)
        self.inverse = np.linalg.inv(self.matrix)

        p5, p6, p7 = self.warp_points(self.calibration.unit_points)
        self.px_per_unit_w = float(np.linalg.norm(p6 - p5))
        self.px_per_unit_h = float(np.linalg.norm(p7 - p5))
        if self.px_per_unit_w < 1e-6 or self.px_per_unit_h < 1e-6:
            raise ValueError("Unit-length points collapse to zero length after warping")
        unit = self.calibration.unit_length_cm
        self.cm_per_px_w = unit / self.px_per_unit_w
        self.cm_per_px_h = unit / self.px_per_unit_h

    # ------------------------------------------------------------------ mapping
    def warp_points(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Image pixels -> warped bird's-eye pixels (N x 2)."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2).astype(np.float64)

    def unwarp_points(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Warped bird's-eye pixels -> image pixels (N x 2)."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return cv2.perspectiveTransform(pts, self.inverse).reshape(-1, 2).astype(np.float64)

    def to_ground_cm(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Image pixels -> ground-plane coordinates in centimetres (N x 2).

        Origin is the top-left (far-left) ROI corner; x grows to the right and y grows
        towards the camera, so the ROI spans ``[0, roi_size_cm[0]] x [0, roi_size_cm[1]]``.
        """
        warped = self.warp_points(points)
        return warped * np.array([self.cm_per_px_w, self.cm_per_px_h])

    @property
    def roi_size_cm(self) -> tuple[float, float]:
        w, h = self.frame_size
        return (w * self.cm_per_px_w, h * self.cm_per_px_h)

    def distance_cm(self, p1: Sequence[float], p2: Sequence[float]) -> float:
        """Ground distance in cm between two image points."""
        a, b = self.to_ground_cm([p1, p2])
        return float(np.linalg.norm(a - b))

    def inside_roi(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Boolean mask: which image points fall inside the ROI polygon."""
        poly = np.asarray(self.calibration.roi, dtype=np.float32).reshape(-1, 1, 2)
        return np.array(
            [cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in points], dtype=bool
        )

    def warp_image(self, frame: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
        """Warp the ROI of a frame into a top-down image (useful to check a calibration)."""
        w, h = self.frame_size
        warped = cv2.warpPerspective(frame, self.matrix, (w, h))
        if size is not None:
            warped = cv2.resize(warped, size)
        return warped

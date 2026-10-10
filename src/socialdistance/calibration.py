"""Scene calibration: the 8 reference points that turn pixels into centimetres.

A calibration has two parts, both marked on one frame of the video:

1. **Region of interest (ROI)**: 4 points on the ground, in the order
   bottom-left, bottom-right, top-right, top-left. Seen from above they must form a
   rectangle (for example the edges of a walkway or floor tiles). This quadrilateral is
   warped into the bird's-eye view.
2. **Unit length**: 3 points. Points 5 and 6 are one unit apart horizontally and points
   5 and 7 are one unit apart in depth (vertically in the bird's-eye view). The unit is
   180 cm (about 6 feet) by default, the social distancing guideline used in 2021.

Calibrations are stored as JSON (see ``config/vtest_calibration.json``) so a scene is
calibrated once and reused. :func:`interactive_calibration` reproduces the original
2021 workflow of clicking the 8 points on the first frame.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

Point = tuple[float, float]


@dataclass
class Calibration:
    roi: list[Point]
    unit_points: list[Point]
    unit_length_cm: float = 180.0
    frame_size: tuple[int, int] | None = None  # (width, height) the points were marked on
    name: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.roi = [tuple(map(float, p)) for p in self.roi]  # type: ignore[misc]
        self.unit_points = [tuple(map(float, p)) for p in self.unit_points]  # type: ignore[misc]
        if len(self.roi) != 4:
            raise ValueError(f"ROI needs exactly 4 points (got {len(self.roi)})")
        if len(self.unit_points) != 3:
            raise ValueError(f"Unit length needs exactly 3 points (got {len(self.unit_points)})")
        if self.unit_length_cm <= 0:
            raise ValueError("unit_length_cm must be positive")
        if self.frame_size is not None:
            self.frame_size = (int(self.frame_size[0]), int(self.frame_size[1]))
        area = _polygon_area(np.array(self.roi))
        if abs(area) < 1.0:
            raise ValueError("ROI points are collinear or duplicated")

    def scaled_to(self, width: int, height: int) -> "Calibration":
        """Rescale the points if the video is processed at a different resolution."""
        if self.frame_size is None or tuple(self.frame_size) == (width, height):
            return self
        sx, sy = width / self.frame_size[0], height / self.frame_size[1]
        return Calibration(
            roi=[(x * sx, y * sy) for x, y in self.roi],
            unit_points=[(x * sx, y * sy) for x, y in self.unit_points],
            unit_length_cm=self.unit_length_cm,
            frame_size=(width, height),
            name=self.name,
            notes=self.notes,
            extra=dict(self.extra),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["roi"] = [list(p) for p in self.roi]
        d["unit_points"] = [list(p) for p in self.unit_points]
        d["frame_size"] = list(self.frame_size) if self.frame_size else None
        if not d["extra"]:
            d.pop("extra")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        known = {"roi", "unit_points", "unit_length_cm", "frame_size", "name", "notes"}
        missing = {"roi", "unit_points"} - d.keys()
        if missing:
            raise ValueError(f"Calibration is missing keys: {sorted(missing)}")
        extra = {k: v for k, v in d.items() if k not in known and k != "extra"}
        extra.update(d.get("extra", {}))
        return cls(
            roi=d["roi"],
            unit_points=d["unit_points"],
            unit_length_cm=float(d.get("unit_length_cm", 180.0)),
            frame_size=tuple(d["frame_size"]) if d.get("frame_size") else None,
            name=d.get("name", ""),
            notes=d.get("notes", ""),
            extra=extra,
        )

    @classmethod
    def from_points(cls, points: Sequence[Point], **kwargs) -> "Calibration":
        """Build from the 8 clicked points (4 ROI corners, then 3 unit points)."""
        if len(points) < 7:
            raise ValueError("Need at least 7 points: 4 ROI corners + 3 unit-length points")
        return cls(roi=list(points[:4]), unit_points=list(points[4:7]), **kwargs)


def _polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def load_calibration(path: str | Path) -> Calibration:
    with open(path, encoding="utf-8") as fh:
        return Calibration.from_dict(json.load(fh))


def save_calibration(calibration: Calibration, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(calibration.to_dict(), fh, indent=2)
        fh.write("\n")
    return path


def interactive_calibration(frame: np.ndarray, unit_length_cm: float = 180.0) -> Calibration:
    """Click the 8 calibration points on ``frame`` in an OpenCV window.

    Order: bottom-left, bottom-right, top-right, top-left (ROI, drawn in red), then the
    unit points 5, 6 (horizontal) and 7 (vertical, from point 5), drawn in blue.
    Press ``r`` to restart, ``q``/Esc to cancel. Needs a desktop session and the
    GUI build of OpenCV (``opencv-python``, not ``opencv-python-headless``).
    """
    import cv2

    points: list[Point] = []
    canvas = frame.copy()
    window = "Calibration: click 4 ROI corners (BL, BR, TR, TL) then 3 unit points"

    def redraw() -> None:
        canvas[:] = frame
        for i, (x, y) in enumerate(points):
            colour = (0, 0, 255) if i < 4 else (255, 0, 0)
            cv2.circle(canvas, (int(x), int(y)), 5, colour, -1)
            cv2.putText(canvas, str(i + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        roi = [tuple(map(int, p)) for p in points[:4]]
        for a, b in zip(roi, roi[1:]):
            cv2.line(canvas, a, b, (70, 70, 70), 2)
        if len(roi) == 4:
            cv2.line(canvas, roi[3], roi[0], (70, 70, 70), 2)
        if len(points) >= 6:
            cv2.line(canvas, tuple(map(int, points[4])), tuple(map(int, points[5])), (255, 0, 0), 2)
        if len(points) >= 7:
            cv2.line(canvas, tuple(map(int, points[4])), tuple(map(int, points[6])), (255, 0, 0), 2)

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 7:
            points.append((float(x), float(y)))
            redraw()

    try:
        cv2.namedWindow(window)
    except cv2.error as exc:  # headless OpenCV build
        raise RuntimeError(
            "Interactive calibration needs a desktop session and the GUI build of OpenCV "
            "(pip install opencv-python). Alternatively, write the points into a JSON file "
            "like config/vtest_calibration.json."
        ) from exc
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if len(points) == 7:
            cv2.imshow(window, canvas)
            cv2.waitKey(400)
            break
        if key == ord("r"):
            points.clear()
            redraw()
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("Calibration cancelled")
    cv2.destroyWindow(window)
    h, w = frame.shape[:2]
    return Calibration.from_points(points, unit_length_cm=unit_length_cm, frame_size=(w, h))

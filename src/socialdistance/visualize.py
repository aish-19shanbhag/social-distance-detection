"""Drawing helpers: annotated camera frame, bird's-eye canvas and a summary banner.

All images are BGR ``uint8`` arrays (OpenCV convention).
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .detectors import Detection
from .perspective import BirdEyeTransform
from .risk import RiskAssessment, RiskLevel

# BGR colours
COLOURS = {
    RiskLevel.HIGH: (59, 59, 208),  # red    #d03b3b
    RiskLevel.LOW: (25, 178, 250),  # amber  #fab219
    RiskLevel.NONE: (12, 163, 12),  # green  #0ca30c
}
ROI_COLOUR = (200, 200, 200)
TEXT = (245, 245, 245)
BANNER_BG = (40, 36, 32)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNER_HEIGHT = 46


def _put_text(img, text, org, scale=0.55, colour=TEXT, thickness=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, colour, thickness, cv2.LINE_AA)


def risk_banner(width: int, counts: dict, extra: str = "") -> np.ndarray:
    """A dark strip with the number of people in each risk group."""
    banner = np.full((BANNER_HEIGHT, width, 3), BANNER_BG, dtype=np.uint8)
    x = 12
    items = [
        ("HIGH RISK", counts.get("high", 0), COLOURS[RiskLevel.HIGH]),
        ("LOW RISK", counts.get("low", 0), COLOURS[RiskLevel.LOW]),
        ("NO RISK", counts.get("safe", 0), COLOURS[RiskLevel.NONE]),
    ]
    for label, value, colour in items:
        cv2.rectangle(banner, (x, 14), (x + 16, 30), colour, -1)
        text = f"{label}: {value}"
        cv2.putText(banner, text, (x + 24, 28), FONT, 0.55, TEXT, 1, cv2.LINE_AA)
        x += 24 + cv2.getTextSize(text, FONT, 0.55, 1)[0][0] + 22
    if extra:
        tw = cv2.getTextSize(extra, FONT, 0.5, 1)[0][0]
        if x + tw + 12 < width:
            cv2.putText(banner, extra, (width - tw - 12, 28), FONT, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    return banner


def draw_roi(frame: np.ndarray, transform: BirdEyeTransform, alpha: float = 0.12) -> np.ndarray:
    poly = np.asarray(transform.calibration.roi, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], (255, 255, 255))
    out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.polylines(out, [poly], True, ROI_COLOUR, 1, cv2.LINE_AA)
    return out


def _draw_tags(img: np.ndarray, tags) -> None:
    """Small dark labels with the pair distance; skips a tag that would overlap another."""
    placed: list[tuple[int, int, int, int]] = []
    for text, (mx, my), colour in reversed(tags):  # high-risk pairs first
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.42, 1)
        x0, y0 = mx - tw // 2 - 4, my - th - 10
        rect = (x0, y0, x0 + tw + 8, y0 + th + 8)
        if any(not (rect[2] < r[0] or rect[0] > r[2] or rect[3] < r[1] or rect[1] > r[3]) for r in placed):
            continue
        placed.append(rect)
        cv2.rectangle(img, rect[:2], rect[2:], (30, 30, 30), -1)
        cv2.rectangle(img, rect[:2], rect[2:], colour, 1)
        cv2.putText(img, text, (x0 + 4, y0 + th + 4), FONT, 0.42, TEXT, 1, cv2.LINE_AA)


def annotate_frame(
    frame: np.ndarray,
    detections: Sequence[Detection],
    assessment: RiskAssessment,
    transform: BirdEyeTransform | None = None,
    show_distances: bool = True,
    banner: bool = True,
    banner_extra: str = "",
) -> np.ndarray:
    """Colour each box by its person's risk and join close pairs with lines."""
    out = draw_roi(frame, transform) if transform is not None else frame.copy()
    centres = [(int(d.x + d.w / 2), int(d.y + d.h / 2)) for d in detections]

    # lines first (low risk under high risk), boxes on top, distance tags last
    tags: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
    for level in (RiskLevel.LOW, RiskLevel.HIGH):
        for i, j, dist, lv in assessment.pairs:
            if lv != level:
                continue
            cv2.line(out, centres[i], centres[j], COLOURS[lv], 2, cv2.LINE_AA)
            mid = ((centres[i][0] + centres[j][0]) // 2, (centres[i][1] + centres[j][1]) // 2)
            tags.append((f"{dist / 100:.1f} m", mid, COLOURS[lv]))

    for det, level in zip(detections, assessment.person_levels):
        colour = COLOURS[level]
        cv2.rectangle(out, (det.x, det.y), (det.x + det.w, det.y + det.h), colour, 2)
        fx, fy = det.foot_point
        cv2.circle(out, (int(fx), int(fy)), 3, colour, -1, cv2.LINE_AA)

    if show_distances:
        _draw_tags(out, tags)

    if banner:
        out = np.vstack([risk_banner(out.shape[1], assessment.counts, banner_extra), out])
    return out


def bird_eye_view(
    transform: BirdEyeTransform,
    points_cm: np.ndarray,
    assessment: RiskAssessment,
    height: int = 576,
    margin_cm: tuple[float, float] = (650.0, 350.0),
    title: str = "Bird's-eye view",
) -> np.ndarray:
    """Top-down map of the ground plane, drawn to true scale in centimetres."""
    roi_w, roi_h = transform.roi_size_cm
    mx, my = margin_cm
    header = 34
    ppc = (height - header - 10) / (roi_h + 2 * my)  # pixels per cm
    width = int(round((roi_w + 2 * mx) * ppc)) + 1
    canvas = np.full((height, width, 3), (238, 236, 232), dtype=np.uint8)

    def to_px(x_cm: float, y_cm: float) -> tuple[int, int]:
        return int(round((x_cm + mx) * ppc)), int(round(header + (y_cm + my) * ppc))

    # 1 m grid
    grid = (222, 219, 214)
    for gx in np.arange(-mx - (-mx % 100), roi_w + mx, 100):
        x0, _ = to_px(gx, 0)
        cv2.line(canvas, (x0, header), (x0, height), grid, 1)
    for gy in np.arange(-my - (-my % 100), roi_h + my, 100):
        _, y0 = to_px(0, gy)
        cv2.line(canvas, (0, y0), (width, y0), grid, 1)

    # ROI rectangle
    tl, br = to_px(0, 0), to_px(roi_w, roi_h)
    cv2.rectangle(canvas, tl, br, (255, 255, 255), -1)
    cv2.rectangle(canvas, tl, br, (150, 150, 150), 1)

    # pairs and people
    pts_px = [to_px(x, y) for x, y in np.asarray(points_cm).reshape(-1, 2)]
    for level in (RiskLevel.LOW, RiskLevel.HIGH):
        for i, j, _, lv in assessment.pairs:
            if lv == level:
                cv2.line(canvas, pts_px[i], pts_px[j], COLOURS[lv], 2, cv2.LINE_AA)
    radius = max(4, int(round(30 * ppc)))  # ~30 cm footprint
    for p, level in zip(pts_px, assessment.person_levels):
        inside = radius <= p[0] < width - radius and header + radius <= p[1] < height - radius
        if inside:
            cv2.circle(canvas, p, radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, p, radius, COLOURS[level], -1, cv2.LINE_AA)
        else:  # off the map: hollow marker pinned to the border
            q = (int(np.clip(p[0], radius, width - radius - 1)), int(np.clip(p[1], header + radius, height - radius - 1)))
            cv2.circle(canvas, q, radius, COLOURS[level], 2, cv2.LINE_AA)

    # header + scale bar
    cv2.rectangle(canvas, (0, 0), (width, header), BANNER_BG, -1)
    cv2.putText(canvas, title, (10, 23), FONT, 0.55, TEXT, 1, cv2.LINE_AA)
    unit = transform.calibration.unit_length_cm
    bar = int(round(unit * ppc))
    x0, y0 = 10, height - 14
    cv2.line(canvas, (x0, y0), (x0 + bar, y0), (60, 60, 60), 2)
    cv2.line(canvas, (x0, y0 - 4), (x0, y0 + 4), (60, 60, 60), 2)
    cv2.line(canvas, (x0 + bar, y0 - 4), (x0 + bar, y0 + 4), (60, 60, 60), 2)
    cv2.putText(canvas, f"{unit / 100:.1f} m", (x0 + bar + 6, y0 + 4), FONT, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 6) -> np.ndarray:
    """Place two images next to each other, padding the shorter one."""
    h = max(left.shape[0], right.shape[0])

    def pad(img):
        if img.shape[0] == h:
            return img
        return np.vstack([img, np.full((h - img.shape[0], img.shape[1], 3), 255, np.uint8)])

    spacer = np.full((h, gap, 3), 255, dtype=np.uint8)
    return np.hstack([pad(left), spacer, pad(right)])


def draw_calibration(frame: np.ndarray, transform: BirdEyeTransform) -> np.ndarray:
    """Show the ROI corners and unit-length points used for a calibration."""
    out = draw_roi(frame, transform, alpha=0.18)
    cal = transform.calibration
    labels = ["1 BL", "2 BR", "3 TR", "4 TL"]
    w = out.shape[1]
    for (x, y), lab in zip(cal.roi, labels):
        cv2.circle(out, (int(x), int(y)), 6, (59, 59, 208), -1, cv2.LINE_AA)
        tw = cv2.getTextSize(lab, FONT, 0.5, 1)[0][0]
        tx = int(x) + 8 if int(x) + 8 + tw < w else int(x) - 8 - tw
        _put_text(out, lab, (tx, int(y) - 8), 0.5)
    p5, p6, p7 = [tuple(map(int, p)) for p in cal.unit_points]
    cv2.line(out, p5, p6, (230, 120, 30), 2, cv2.LINE_AA)
    cv2.line(out, p5, p7, (230, 120, 30), 2, cv2.LINE_AA)
    for p, lab in zip((p5, p6, p7), ("5", "6", "7")):
        cv2.circle(out, p, 5, (230, 120, 30), -1, cv2.LINE_AA)
        _put_text(out, lab, (p[0] + 6, p[1] + 16), 0.5)
    unit = cal.unit_length_cm
    msg = f"5-6 and 5-7 = {unit:.0f} cm on the ground"
    tw = cv2.getTextSize(msg, FONT, 0.45, 1)[0][0]
    _put_text(out, msg, (max(4, min(p5[0] - tw // 2, w - tw - 4)), p5[1] + 40), 0.45)
    return out

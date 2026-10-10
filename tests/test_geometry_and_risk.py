"""Perspective transform, centimetre scaling, risk thresholds and counting."""

import math

import numpy as np
import pytest

from socialdistance.calibration import Calibration
from socialdistance.perspective import BirdEyeTransform
from socialdistance.risk import RiskLevel, RiskThresholds, assess_risk, pairwise_distances

W, H = 800, 600


def rect_calibration(unit_w_px=100.0, unit_h_px=50.0):
    """ROI = whole frame (identity homography); 180 cm = unit_w_px across, unit_h_px in depth."""
    return Calibration(
        roi=[(0, H), (W, H), (W, 0), (0, 0)],
        unit_points=[(100, 300), (100 + unit_w_px, 300), (100, 300 - unit_h_px)],
        unit_length_cm=180,
        frame_size=(W, H),
    )


def make_trapezoid_scene():
    """A synthetic camera: ground rectangle 1000 cm x 2000 cm seen as a trapezoid."""
    # Ground (cm) -> image via a known homography
    ground = np.float32([[0, 2000], [1000, 2000], [1000, 0], [0, 0]])  # BL, BR, TR, TL
    image = np.float32([[50, 580], [750, 580], [520, 120], [280, 120]])
    import cv2

    g2i = cv2.getPerspectiveTransform(ground, image)

    def project(pts):
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, g2i).reshape(-1, 2)

    unit = project([[400, 1200], [580, 1200], [400, 1020]])
    cal = Calibration(roi=image.tolist(), unit_points=unit.tolist(), frame_size=(W, H))
    return cal, project


# ----------------------------------------------------------------- perspective
def test_identity_roi_maps_points_to_themselves():
    t = BirdEyeTransform(rect_calibration(), (W, H))
    pts = np.array([[0, 0], [400, 300], [799, 599]], float)
    np.testing.assert_allclose(t.warp_points(pts), pts, atol=1e-3)
    np.testing.assert_allclose(t.unwarp_points(t.warp_points(pts)), pts, atol=1e-3)


def test_roi_corners_map_to_frame_corners():
    cal, _ = make_trapezoid_scene()
    t = BirdEyeTransform(cal, (W, H))
    expected = np.array([[0, H], [W, H], [W, 0], [0, 0]], float)
    np.testing.assert_allclose(t.warp_points(cal.roi), expected, atol=1e-2)


def test_unit_scale_in_pixels_per_180cm():
    t = BirdEyeTransform(rect_calibration(unit_w_px=100, unit_h_px=50), (W, H))
    assert t.px_per_unit_w == pytest.approx(100)
    assert t.px_per_unit_h == pytest.approx(50)
    assert t.cm_per_px_w == pytest.approx(1.8)
    assert t.cm_per_px_h == pytest.approx(3.6)


def test_distance_uses_separate_horizontal_and_vertical_scales():
    t = BirdEyeTransform(rect_calibration(unit_w_px=100, unit_h_px=50), (W, H))
    assert t.distance_cm((0, 0), (100, 0)) == pytest.approx(180)  # one unit across
    assert t.distance_cm((0, 0), (0, 50)) == pytest.approx(180)  # one unit in depth
    # 3-4-5 triangle in cm: 60 px across (108 cm) and 40 px deep (144 cm) -> 180 cm
    assert t.distance_cm((0, 0), (60, 40)) == pytest.approx(math.hypot(108, 144))


def test_trapezoid_scene_recovers_true_ground_distances():
    cal, project = make_trapezoid_scene()
    t = BirdEyeTransform(cal, (W, H))
    # ROI is exactly 1000 x 2000 cm
    assert t.roi_size_cm[0] == pytest.approx(1000, rel=1e-3)
    assert t.roi_size_cm[1] == pytest.approx(2000, rel=1e-3)
    # same 150 cm gap measured near and far from the camera
    for y in (200, 1000, 1900):
        a, b = project([[300, y], [450, y]])
        assert t.distance_cm(a, b) == pytest.approx(150, abs=0.5)
    a, b = project([[500, 500], [500 + 90, 500 + 120]])
    assert t.distance_cm(a, b) == pytest.approx(150, abs=0.5)
    ground = t.to_ground_cm(project([[0, 0], [1000, 2000]]))
    np.testing.assert_allclose(ground, [[0, 0], [1000, 2000]], atol=0.5)


def test_pixel_distance_is_misleading_but_ground_distance_is_not():
    cal, project = make_trapezoid_scene()
    t = BirdEyeTransform(cal, (W, H))
    near = project([[300, 1900], [450, 1900]])
    far = project([[300, 100], [450, 100]])
    px_near = np.linalg.norm(near[0] - near[1])
    px_far = np.linalg.norm(far[0] - far[1])
    assert px_near > 2 * px_far  # perspective shrinks the far pair in pixels
    assert t.distance_cm(*near) == pytest.approx(t.distance_cm(*far), abs=0.5)


def test_calibration_rescales_with_frame_size():
    cal, project = make_trapezoid_scene()
    small = BirdEyeTransform(cal, (W // 2, H // 2))
    full = BirdEyeTransform(cal, (W, H))
    a, b = project([[300, 1000], [450, 1000]])
    assert small.distance_cm(a / 2, b / 2) == pytest.approx(full.distance_cm(a, b), rel=1e-3)


def test_inside_roi():
    cal, _ = make_trapezoid_scene()
    t = BirdEyeTransform(cal, (W, H))
    assert t.inside_roi([(400, 400), (10, 10)]).tolist() == [True, False]


# ----------------------------------------------------------------------- risk
@pytest.mark.parametrize(
    "d,expected",
    [
        (0, RiskLevel.HIGH),
        (149.9, RiskLevel.HIGH),
        (150.0, RiskLevel.HIGH),  # boundary belongs to high risk (<= 150)
        (150.1, RiskLevel.LOW),
        (180.0, RiskLevel.LOW),  # boundary belongs to low risk (<= 180)
        (180.1, RiskLevel.NONE),
        (1000, RiskLevel.NONE),
    ],
)
def test_threshold_boundaries(d, expected):
    assert RiskThresholds().classify(d) == expected


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        RiskThresholds(high_cm=200, low_cm=180)


def test_pairwise_distances_symmetric():
    d = pairwise_distances(np.array([[0, 0], [300, 400], [0, 100]]))
    assert d.shape == (3, 3)
    np.testing.assert_allclose(d, d.T)
    assert d[0, 1] == pytest.approx(500)
    assert np.all(np.diag(d) == 0)


def test_person_takes_worst_pair_and_counts_add_up():
    # A-B 100 cm apart (high), C is 170 cm from B (low for C), D far away (none)
    pts = np.array([[0, 0], [100, 0], [270, 0], [2000, 2000]], float)
    a = assess_risk(pts)
    assert a.person_levels == [RiskLevel.HIGH, RiskLevel.HIGH, RiskLevel.LOW, RiskLevel.NONE]
    assert a.counts == {"high": 2, "low": 1, "safe": 1, "total": 4}
    assert a.min_distance_cm == pytest.approx(100)
    assert len(a.pairs) == 6  # n * (n - 1) / 2
    assert {(i, j) for i, j, _, _ in a.violations} == {(0, 1), (1, 2)}


def test_single_person_and_empty_frame():
    one = assess_risk(np.array([[10, 10]], float))
    assert one.counts == {"high": 0, "low": 0, "safe": 1, "total": 1}
    assert one.min_distance_cm is None
    empty = assess_risk(np.zeros((0, 2)))
    assert empty.counts == {"high": 0, "low": 0, "safe": 0, "total": 0}


def test_custom_thresholds():
    pts = np.array([[0, 0], [190, 0]], float)
    assert assess_risk(pts).counts["safe"] == 2
    assert assess_risk(pts, RiskThresholds(high_cm=100, low_cm=200)).counts["low"] == 2

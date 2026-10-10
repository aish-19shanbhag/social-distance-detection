"""Streamlit app smoke tests with streamlit.testing.v1.AppTest."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
from socialdistance.detectors import models_available  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")


def test_app_loads_without_errors():
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Social Distance Detection using YOLOv3"


def test_missing_yolo_weights_shows_download_instructions():
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio(key="detector").set_value("YOLOv3").run()
    assert not at.exception
    if not models_available("yolov3", ROOT / "models"):
        warnings = " ".join(w.value for w in at.warning)
        assert "download_models.py" in warnings
        assert at.button[0].disabled


@pytest.mark.skipif(not models_available("ssd", ROOT / "models"), reason="SSD model not downloaded")
def test_ssd_run_on_sample_video():
    at = st_testing.AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.slider(key="n_frames").set_value(10)
    at.button[0].click().run()
    assert not at.exception
    assert at.metric[0].value == "10"
    assert len(at.get("image")) == 2  # annotated frame + bird's-eye view

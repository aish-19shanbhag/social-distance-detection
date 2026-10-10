"""Streamlit demo: pick a detector, run it on a video, inspect risk zones frame by frame.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socialdistance.calibration import Calibration, load_calibration  # noqa: E402
from socialdistance.detectors import ModelFilesMissing, create_detector, model_paths, models_available  # noqa: E402
from socialdistance.pipeline import iter_video, render, video_info  # noqa: E402
from socialdistance.risk import RiskThresholds  # noqa: E402

SAMPLE_VIDEO = ROOT / "data" / "vtest.avi"
SAMPLE_CONFIG = ROOT / "config" / "vtest_calibration.json"
MODELS_DIR = ROOT / "models"
DETECTORS = {"YOLOv3": "yolov3", "SSD MobileNet": "ssd"}

st.set_page_config(page_title="Social Distance Detection", page_icon="🚶", layout="wide")


@st.cache_resource(show_spinner=False)
def get_detector(key: str, confidence: float, input_size: int):
    return create_detector(key, MODELS_DIR, conf_threshold=confidence, input_size=input_size)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Settings")
detector_label = st.sidebar.radio("Detection model", list(DETECTORS), index=1, key="detector")
detector_key = DETECTORS[detector_label]
default_conf = 0.5 if detector_key == "yolov3" else 0.4
confidence = st.sidebar.slider("Confidence threshold", 0.2, 0.9, default_conf, 0.05, key=f"conf_{detector_key}")
if detector_key == "ssd":
    input_size = st.sidebar.select_slider("SSD input size", [300, 512], value=512, key="ssd_size")
else:
    input_size = 416

source = st.sidebar.radio("Video", ["Sample video (vtest.avi)", "Upload a video"], key="source")
uploaded_video = uploaded_cal = None
if source == "Upload a video":
    uploaded_video = st.sidebar.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
    uploaded_cal = st.sidebar.file_uploader(
        "Calibration JSON for this camera",
        type=["json"],
        help="4 ROI corners + 3 unit-length points. Create one with "
        "`python -m socialdistance calibrate --video your.mp4 --out cal.json`.",
    )

n_frames = st.sidebar.slider("Frames to process", 10, 300, 60, 10, key="n_frames")
start_frame = st.sidebar.number_input("Start frame", 0, 10_000, 0, 10, key="start")
every_n = st.sidebar.select_slider("Process every n-th frame", [1, 2, 3, 5], value=1, key="every_n")
with st.sidebar.expander("Risk thresholds (cm)"):
    high_cm = st.number_input("High risk at or below", 50, 300, 150, 10)
    low_cm = st.number_input("Low risk at or below", 50, 400, 180, 10)

# ------------------------------------------------------------------ header
st.title("Social Distance Detection using YOLOv3")
st.caption(
    "Detect people, map their positions to a calibrated bird's-eye view, and flag pairs closer than "
    "150 cm (high risk) or 180 cm (low risk). Runs on CPU with OpenCV DNN."
)

# ------------------------------------------------------------------ checks
ready = True
if not models_available(detector_key, MODELS_DIR):
    ready = False
    missing = [p.name for p in model_paths(detector_key, MODELS_DIR) if not p.is_file()]
    flag = " --ssd-only" if detector_key == "ssd" else ""
    st.warning(
        f"**{detector_label} model files are missing** ({', '.join(missing)}). Download them with:\n\n"
        f"```bash\npython scripts/download_models.py{flag}\n```\n"
        + (
            "The YOLOv3 weights are about 248 MB. In the meantime you can switch to **SSD MobileNet** in the sidebar."
            if detector_key == "yolov3"
            else ""
        )
    )

video_path: Path | None = None
calibration: Calibration | None = None
if source.startswith("Sample"):
    video_path, calibration = SAMPLE_VIDEO, load_calibration(SAMPLE_CONFIG)
else:
    if uploaded_video is not None:
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        tmp = Path(tempfile.gettempdir()) / f"sdd_upload{suffix}"
        tmp.write_bytes(uploaded_video.getbuffer())
        video_path = tmp
    if uploaded_cal is not None:
        try:
            calibration = Calibration.from_dict(json.loads(uploaded_cal.getvalue()))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            st.error(f"Invalid calibration file: {exc}")
    if video_path is None:
        ready = False
        st.info("Upload a video in the sidebar.")
    elif calibration is None:
        calibration = load_calibration(SAMPLE_CONFIG)
        st.warning(
            "No calibration uploaded: using the sample scene's calibration, so distances will not be "
            "meaningful for your camera. Upload a calibration JSON for real measurements."
        )

if low_cm < high_cm:
    ready = False
    st.error("The low-risk threshold must be at least the high-risk threshold.")

if video_path is not None:
    info = video_info(video_path)
    st.sidebar.caption(f"{info['width']}x{info['height']} · {info['fps']:.0f} fps · {info['frames']} frames")

run = st.button("Run detection", type="primary", disabled=not ready)

# ------------------------------------------------------------------ run
if run and ready:
    try:
        detector = get_detector(detector_key, confidence, int(input_size))
    except ModelFilesMissing as exc:
        st.error(str(exc))
        st.stop()
    thresholds = RiskThresholds(float(high_cm), float(low_cm))
    progress = st.progress(0.0, text="Processing ...")
    frames, rows = [], []
    t0 = time.perf_counter()
    for k, (frame, result, transform) in enumerate(
        iter_video(video_path, detector, calibration, thresholds, n_frames, int(start_frame), every_n)
    ):
        annotated, bird = render(frame, result, transform, f"frame {result.index}")
        h = 480
        annotated = cv2.resize(annotated, (int(annotated.shape[1] * h / annotated.shape[0]), h))
        bird = cv2.resize(bird, (int(bird.shape[1] * h / bird.shape[0]), h))
        frames.append((rgb(annotated), rgb(bird)))
        rows.append(result.csv_row(info["fps"]))
        progress.progress((k + 1) / n_frames, text=f"Processed {k + 1}/{n_frames} frames")
    elapsed = time.perf_counter() - t0
    progress.empty()
    st.session_state["results"] = {
        "frames": frames,
        "rows": rows,
        "elapsed": elapsed,
        "detector": detector_label,
    }

res = st.session_state.get("results")
if res and res["frames"]:
    df = pd.DataFrame(res["rows"])
    n = len(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Frames processed", n)
    c2.metric("Avg people / frame", f"{df['detected'].mean():.1f}")
    c3.metric("Frames with a close pair", f"{(df['close_pairs'] > 0).mean():.0%}")
    closest = pd.to_numeric(df["min_distance_cm"], errors="coerce").min()
    c4.metric("Closest pair", "n/a" if pd.isna(closest) else f"{closest / 100:.2f} m")
    c5.metric("Speed (this run)", f"{n / res['elapsed']:.1f} fps")

    idx = st.slider("Frame", 0, n - 1, n - 1, key="frame_idx") if n > 1 else 0
    left, right = st.columns([1.55, 1])
    annotated, bird = res["frames"][idx]
    left.image(annotated, caption=f"{res['detector']}: frame {df['frame'][idx]}", width="stretch")
    right.image(bird, caption="Bird's-eye view (true scale; white = calibrated ROI)", width="stretch")

    st.subheader("People per risk zone over time")
    chart = df.set_index("frame")[["high_risk", "low_risk", "no_risk"]].rename(
        columns={"high_risk": "High risk", "low_risk": "Low risk", "no_risk": "No risk"}
    )
    st.area_chart(chart, color=["#d03b3b", "#fab219", "#0ca30c"], stack=True, x_label="Frame", y_label="People")
    st.download_button("Download per-frame CSV", df.to_csv(index=False), "risk_counts.csv", "text/csv")
elif ready:
    st.info("Choose settings in the sidebar and press **Run detection**.")

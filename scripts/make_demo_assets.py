"""Regenerate every demo image, the GIF, the risk chart and the speed numbers in reports/.

    python scripts/download_models.py --ssd-only
    python scripts/make_demo_assets.py --detector ssd

Steps:
1. Process the whole sample video and write outputs/<run>/ (videos, CSV, summary).
2. Pick the frame with the most close pairs and save the annotated frame, the bird's-eye
   view and a side-by-side image.
3. Save a calibration figure (reference points + the ROI warped to a top-down view).
4. Save a short GIF of the annotated output and a chart of risk counts over time.
5. Benchmark processing speed on this machine's CPU and save reports/metrics/speed.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socialdistance.calibration import load_calibration  # noqa: E402
from socialdistance.detectors import create_detector  # noqa: E402
from socialdistance.perspective import BirdEyeTransform  # noqa: E402
from socialdistance.pipeline import iter_video, process_video, render, video_info  # noqa: E402
from socialdistance.visualize import draw_calibration, side_by_side  # noqa: E402

DEMO = ROOT / "reports" / "demo"
METRICS = ROOT / "reports" / "metrics"


def save_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"  wrote {path.relative_to(ROOT)}")


def calibration_figure(video: Path, calibration) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    t = BirdEyeTransform(calibration, frame.shape[1::-1])
    left = draw_calibration(frame, t)
    roi_w, roi_h = t.roi_size_cm
    h = left.shape[0]
    top = t.warp_image(frame, (max(1, int(round(h * roi_w / roi_h))), h))
    cv2.rectangle(top, (0, 0), (top.shape[1] - 1, 30), (40, 36, 32), -1)
    cv2.putText(top, "ROI warped top-down", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
    return side_by_side(left, top)


def risk_chart(csv_path: Path, out: Path, fps: float, detector_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    t = df["frame"] / fps
    smooth = lambda s: s.rolling(5, center=True, min_periods=1).mean()  # noqa: E731
    high, low, safe = smooth(df["high_risk"]), smooth(df["low_risk"]), smooth(df["no_risk"])

    ink, muted, grid = "#2b2b2b", "#6b6b6b", "#e6e4e0"
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)
    ax.stackplot(
        t,
        high,
        low,
        safe,
        colors=["#d03b3b", "#fab219", "#0ca30c"],
        labels=["High risk (≤ 150 cm)", "Low risk (150–180 cm)", "No risk (> 180 cm)"],
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(0, float((high + low + safe).max()) * 1.05)
    ax.set_xlabel("Video time (s)", color=muted)
    ax.set_ylabel("People in frame", color=muted)
    ax.set_title(
        f"People per risk zone over time, vtest.avi ({detector_label}, 5-frame rolling mean)",
        loc="left",
        color=ink,
        fontsize=11,
        pad=28,
    )
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbab4")
    ax.tick_params(colors=muted)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3, frameon=False, fontsize=9, labelcolor=ink)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def make_gif(video, detector, calibration, start, n_frames, out: Path, width: int, fps: float) -> None:
    from PIL import Image

    frames = []
    for frame, result, transform in iter_video(video, detector, calibration, start_frame=start, max_frames=n_frames):
        annotated, bird = render(frame, result, transform)
        combo = side_by_side(annotated, bird)
        h = int(round(combo.shape[0] * width / combo.shape[1]))
        rgb = cv2.cvtColor(cv2.resize(combo, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    # one shared palette (from a sample of frames) keeps colours stable and avoids flicker
    from socialdistance.visualize import COLOURS

    stack = frames[:: max(1, len(frames) // 8)]
    swatches = [np.full((60, frames[0].shape[1], 3), c[::-1], np.uint8) for c in COLOURS.values()]
    sample = Image.fromarray(np.vstack(stack + swatches))  # make sure the risk colours get exact entries
    palette = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    frames = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"  wrote {out.relative_to(ROOT)} ({len(frames)} frames, {out.stat().st_size / 1e6:.2f} MB)")


def benchmark(video, models_dir, calibration, name, input_size, n_frames, warmup=5) -> dict:
    det = create_detector(name, models_dir, input_size=input_size)
    infer, pipe, render_ms, people = [], [], [], []
    for k, (frame, result, transform) in enumerate(
        iter_video(video, det, calibration, max_frames=n_frames + warmup)
    ):
        t0 = time.perf_counter()
        render(frame, result, transform)
        r = (time.perf_counter() - t0) * 1000
        if k >= warmup:
            infer.append(result.inference_ms)
            pipe.append(result.total_ms)
            render_ms.append(r)
            people.append(len(result.detections))
    pipe_mean, render_mean = float(np.mean(pipe)), float(np.mean(render_ms))
    return {
        "detector": name,
        "input_size": input_size,
        "frames_timed": len(pipe),
        "network_forward_ms_mean": round(float(np.mean(infer)), 1),
        "pipeline_ms_mean": round(pipe_mean, 1),
        "pipeline_fps": round(1000 / pipe_mean, 1),
        "with_rendering_fps": round(1000 / (pipe_mean + render_mean), 1),
        "avg_people_detected_per_frame": round(float(np.mean(people)), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "data" / "vtest.avi"))
    ap.add_argument("--config", default=str(ROOT / "config" / "vtest_calibration.json"))
    ap.add_argument("--detector", default="ssd")
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "vtest_demo"))
    ap.add_argument("--gif-start", type=int, default=None, help="first frame of the GIF (default: busiest window)")
    ap.add_argument("--gif-frames", type=int, default=50)
    ap.add_argument("--bench-frames", type=int, default=200)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    video, cal = Path(args.video), load_calibration(args.config)
    detector = create_detector(args.detector, args.models_dir)
    label = {"ssd": "SSD MobileNet", "yolov3": "YOLOv3", "yolo": "YOLOv3"}[args.detector]

    print("1/5 processing the full video ...")
    summary = process_video(video, detector, cal, args.output)
    out = Path(args.output)
    METRICS.mkdir(parents=True, exist_ok=True)
    shutil.copy(out / "risk_counts.csv", METRICS / "vtest_risk_counts.csv")
    print(json.dumps(summary, indent=2))

    import pandas as pd

    df = pd.read_csv(out / "risk_counts.csv")
    fps = video_info(video)["fps"]

    print("2/5 showcase frame ...")
    score = df["high_risk"] * 2 + df["low_risk"] * 3 + df["detected"] * 0.1  # prefer frames with both levels
    best = int(df.loc[score.idxmax(), "frame"])
    for frame, result, transform in iter_video(video, detector, cal, start_frame=best, max_frames=1):
        annotated, bird = render(frame, result, transform, f"frame {best}")
        save_png(DEMO / "annotated_frame.png", annotated)
        save_png(DEMO / "birds_eye.png", bird)
        save_png(DEMO / "side_by_side.png", side_by_side(annotated, bird))

    print("3/5 calibration figure ...")
    save_png(DEMO / "calibration.png", calibration_figure(video, cal))

    print("4/5 GIF and chart ...")
    if args.gif_start is None:
        busy = (df["high_risk"] + df["low_risk"]).rolling(args.gif_frames).sum()
        gif_start = int(df.loc[busy.idxmax(), "frame"]) - args.gif_frames + 1
    else:
        gif_start = args.gif_start
    make_gif(video, detector, cal, max(0, gif_start), args.gif_frames, DEMO / "demo.gif", 900, fps)
    risk_chart(out / "risk_counts.csv", DEMO / "risk_over_time.png", fps, label)

    if not args.skip_bench:
        print("5/5 speed benchmark ...")
        runs = [benchmark(video, args.models_dir, cal, args.detector, s, args.bench_frames) for s in (300, 512)]
        speed = {
            "machine": {
                "cpu_threads": cv2.getNumberOfCPUs(),
                "opencv": cv2.__version__,
                "python": platform.python_version(),
                "platform": platform.system(),
            },
            "video": video.name,
            "frame_size": summary["frame_size"],
            "runs": runs,
            "full_video_run": {
                "frames": summary["frames_processed"],
                "end_to_end_fps_including_rendering_and_video_writing": summary[
                    "end_to_end_fps_including_rendering"
                ],
            },
        }
        (METRICS / "speed.json").write_text(json.dumps(speed, indent=2) + "\n")
        print(json.dumps(speed, indent=2))

    meta = {
        "detector": label,
        "showcase_frame": best,
        "gif_frames": [max(0, gif_start), max(0, gif_start) + args.gif_frames - 1],
        "summary": summary,
    }
    (METRICS / "demo_run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

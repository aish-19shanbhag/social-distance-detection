"""Command-line interface.

Examples::

    python -m socialdistance run --video data/vtest.avi --config config/vtest_calibration.json \
        --detector ssd --max-frames 200 --output outputs/vtest_ssd
    python -m socialdistance calibrate --video my_video.mp4 --out config/my_scene.json
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2

from .calibration import interactive_calibration, load_calibration, save_calibration
from .detectors import ModelFilesMissing, create_detector
from .pipeline import process_video
from .risk import RiskThresholds


def _cmd_run(args: argparse.Namespace) -> int:
    calibration = load_calibration(args.config)
    try:
        detector = create_detector(
            args.detector, args.models_dir, conf_threshold=args.confidence, input_size=args.input_size
        )
    except ModelFilesMissing as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    thresholds = RiskThresholds(high_cm=args.high_cm, low_cm=args.low_cm)

    def progress(n, result):
        if n % 25 == 0:
            c = result.counts
            print(f"  {n} frames | people {c['total']} | high {c['high']} low {c['low']} safe {c['safe']}")

    print(f"Processing {args.video} with {detector.name} ...")
    summary = process_video(
        args.video,
        detector,
        calibration,
        args.output,
        thresholds=thresholds,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        every_n=args.every_n,
        roi_only=args.roi_only,
        write_videos=not args.no_video,
        progress=progress,
    )
    print(json.dumps(summary, indent=2))
    print(f"Outputs written to {args.output}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"error: cannot read frame {args.frame} of {args.video}", file=sys.stderr)
        return 2
    try:
        calibration = interactive_calibration(frame, unit_length_cm=args.unit_cm)
    except (RuntimeError, KeyboardInterrupt) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    calibration.name = args.name or args.video
    save_calibration(calibration, args.out)
    print(f"Saved calibration to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="socialdistance", description="Social distance detection")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="process a video")
    r.add_argument("--video", required=True)
    r.add_argument("--config", required=True, help="calibration JSON")
    r.add_argument("--detector", default="yolov3", choices=["yolov3", "yolo", "ssd"])
    r.add_argument("--models-dir", default="models")
    r.add_argument("--output", default="outputs/run")
    r.add_argument("--max-frames", type=int, default=None)
    r.add_argument("--start-frame", type=int, default=0)
    r.add_argument("--every-n", type=int, default=1, help="process every n-th frame")
    r.add_argument("--confidence", type=float, default=None, help="default 0.5 (YOLOv3) / 0.4 (SSD)")
    r.add_argument("--input-size", type=int, default=None, help="default 416 (YOLOv3) / 512 (SSD)")
    r.add_argument("--high-cm", type=float, default=150.0)
    r.add_argument("--low-cm", type=float, default=180.0)
    r.add_argument("--roi-only", action="store_true", help="ignore people standing outside the ROI")
    r.add_argument("--no-video", action="store_true", help="only write the CSV and summary")
    r.set_defaults(func=_cmd_run)

    c = sub.add_parser("calibrate", help="click the calibration points on a frame")
    c.add_argument("--video", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--frame", type=int, default=0)
    c.add_argument("--unit-cm", type=float, default=180.0)
    c.add_argument("--name", default="")
    c.set_defaults(func=_cmd_calibrate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

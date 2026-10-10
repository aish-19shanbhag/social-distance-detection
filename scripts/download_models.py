"""Download the pre-trained detector files into models/.

    python scripts/download_models.py              # everything (YOLOv3 weights are ~248 MB)
    python scripts/download_models.py --ssd-only   # only SSD MobileNet (~23 MB)

Sources:
* YOLOv3 (COCO): config and weights from Joseph Redmon's Darknet project (pjreddie.com).
* MobileNet-SSD (PASCAL VOC): deploy prototxt and caffemodel from github.com/chuanqi305/MobileNet-SSD.

Model weights are large binaries and are not committed to this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

FILES = {
    "yolo": [
        {
            "name": "yolov3.cfg",
            "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            "sha256": "22489ea38575dfa36c67a90048e8759576416a79d32dc11e15d2217777b9a953",
        },
        {
            "name": "yolov3.weights",
            "url": "https://pjreddie.com/media/files/yolov3.weights",
            "sha256": None,  # checked by size instead
            "min_bytes": 240_000_000,
        },
    ],
    "ssd": [
        {
            "name": "MobileNetSSD_deploy.prototxt",
            "url": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
            "sha256": "2d180f723b3109e21f8287f6b3c691390d07b60eed998327cd3259ffa0e50608",
        },
        {
            "name": "MobileNetSSD_deploy.caffemodel",
            "url": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel",
            "sha256": "52eed8be80522c152a17fb56740de705b79881bde1a167e0e747310523685fc7",
        },
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid(path: Path, spec: dict) -> bool:
    if not path.is_file():
        return False
    if spec.get("sha256"):
        return sha256(path) == spec["sha256"]
    return path.stat().st_size >= spec.get("min_bytes", 1)


def download(spec: dict, dest_dir: Path, url_override: str | None = None, force: bool = False) -> bool:
    dest = dest_dir / spec["name"]
    if not force and is_valid(dest, spec):
        print(f"[ok]   {spec['name']} already present")
        return True
    url = url_override or spec["url"]
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get]  {spec['name']} <- {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "social-distance-detection"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total > 1_000_000:
                    print(f"\r       {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            if total > 1_000_000:
                print()
    except Exception as exc:  # network errors, 403s, timeouts
        tmp.unlink(missing_ok=True)
        print(f"[fail] {spec['name']}: {exc}")
        return False
    shutil.move(str(tmp), dest)
    if not is_valid(dest, spec):
        print(f"[fail] {spec['name']}: checksum/size check failed; delete it and retry")
        return False
    print(f"[ok]   {spec['name']}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--ssd-only", action="store_true")
    group.add_argument("--yolo-only", action="store_true")
    ap.add_argument("--models-dir", default=str(MODELS))
    ap.add_argument("--yolo-weights-url", default=None, help="alternative mirror for yolov3.weights")
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = ap.parse_args(argv)

    dest = Path(args.models_dir)
    dest.mkdir(parents=True, exist_ok=True)
    groups = ["ssd"] if args.ssd_only else ["yolo"] if args.yolo_only else ["yolo", "ssd"]
    ok = True
    for g in groups:
        for spec in FILES[g]:
            override = args.yolo_weights_url if spec["name"] == "yolov3.weights" else None
            ok &= download(spec, dest, override, args.force)
    if not ok:
        print(
            "\nSome files could not be downloaded. The SSD MobileNet detector works on its own; "
            "for YOLOv3, download yolov3.weights manually into models/ (or pass --yolo-weights-url)."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

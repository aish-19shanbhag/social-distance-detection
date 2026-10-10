# models/

Pre-trained detector files live here but are **not committed** (they are large binaries).
Download them with:

```bash
python scripts/download_models.py            # YOLOv3 (~248 MB) + SSD MobileNet (~23 MB)
python scripts/download_models.py --ssd-only # SSD MobileNet only
```

| File | Detector | Source |
|---|---|---|
| `yolov3.cfg` | YOLOv3 (COCO, 80 classes) | [pjreddie/darknet](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) |
| `yolov3.weights` | YOLOv3 (COCO, 80 classes) | [pjreddie.com](https://pjreddie.com/darknet/yolo/) |
| `MobileNetSSD_deploy.prototxt` | MobileNet-SSD (PASCAL VOC, 20 classes) | [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) |
| `MobileNetSSD_deploy.caffemodel` | MobileNet-SSD (PASCAL VOC, 20 classes) | [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) |

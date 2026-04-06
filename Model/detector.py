import time
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import supervision as sv
from Speed_Estimator.speed_est import SpeedPipeline

PALETTE = [
    (0, 200, 255),   # car       – amber
    (0, 255, 128),   # truck     – green
    (255, 80, 80),   # bus       – blue
    (200, 0, 255),   # motorcycle– magenta
    (0, 180, 255),   # bicycle   – orange
    (255, 200, 0),   # van       – cyan
    (128, 255, 0),   # SUV       – lime
    (255, 0, 128),   # pickup    – pink
]

'''VehicleDetector encapsulates the YOLO model and provides methods for detection, annotation, and encoding.'''
class VehicleDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.45):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names: dict = self.model.names
        self._color_map: dict[int, tuple] = {
            i: PALETTE[i % len(PALETTE)] for i in range(len(self.class_names))
        }
        self.speed_pipeline = SpeedPipeline(self.model)

    def detect(self, frame: np.ndarray) -> dict:
        t0 = time.perf_counter()

        detections, speeds = self.speed_pipeline.process(frame)

        inference_ms = (time.perf_counter() - t0) * 1000

        results = []
        counts = defaultdict(int)

        annotated = frame.copy()

        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            cls_id = int(detections.class_id[i])
            cls_name = self.class_names[cls_id]

            track_id = detections.tracker_id[i]
            speed = speeds.get(track_id, 0)

            label = f"{cls_name} | {speed} km/h"

            color = self._color_map[cls_id]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            results.append({
                "class": cls_name,
                "speed_kmh": speed,
                "bbox": [x1, y1, x2, y2],
                "track_id": int(track_id) if track_id is not None else None
            })

            counts[cls_name] += 1

        return {
            "frame": annotated,
            "detections": results,
            "counts": dict(counts),
            "total": len(results),
            "inference_ms": round(inference_ms, 1),
        }

    def encode(self, frame: np.ndarray, quality: int = 85) -> bytes:
        """JPEG-encode a frame."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()


    def _draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        cls_to_id = {v: k for k, v in self.class_names.items()}
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = cls_to_id.get(det["class"], 0)
            color = self._color_map[cls_id]
            label = f"{det['class']} {det['confidence']:.0%}"

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label pill
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                frame, label,
                (x1 + 3, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA,
            )

        return frame

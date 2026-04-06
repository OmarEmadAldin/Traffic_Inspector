import time
import numpy as np
import supervision as sv


class Tracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(detections)


class SpeedEstimator:
    def __init__(self, pixels_per_meter: float = 8.0):
        self.pixels_per_meter = pixels_per_meter
        self.history = {}   # track_id -> (center, time)
        self.speeds = {}    # track_id -> km/h

    def estimate(self, detections: sv.Detections) -> dict:
        results = {}

        for i in range(len(detections)):
            track_id = detections.tracker_id[i]

            if track_id is None:
                continue

            x1, y1, x2, y2 = detections.xyxy[i]
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

            now = time.time()

            if track_id in self.history:
                prev_center, prev_time = self.history[track_id]

                dist_pixels = np.linalg.norm(center - prev_center)
                dt = now - prev_time

                if dt > 0:
                    speed_mps = (dist_pixels / self.pixels_per_meter) / dt
                    speed_kmh = speed_mps * 3.6
                    self.speeds[track_id] = int(speed_kmh)

            self.history[track_id] = (center, now)

            results[track_id] = round(self.speeds.get(track_id, 0), 1)

        return results


class SpeedPipeline:
    def __init__(self, model):
        self.model = model
        self.tracker = Tracker()
        self.speed_estimator = SpeedEstimator(pixels_per_meter=8)

    def process(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update(detections)

        speeds = self.speed_estimator.estimate(detections)

        return detections, speeds
    
'''
The idea of the estimation is based on v = d/t 
so if the center of the BB moved 100 pixels in 0.5 seconds, and we know that 8 pixels = 1 meter, then:
d = 100 pixels / 8 pixels/meter = 12.5 meters   
'''
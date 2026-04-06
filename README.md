# TrafficGuard - Vehicle Detection and Speed Estimation System

TrafficGuard is a real-time computer vision system for vehicle
detection, tracking, and speed estimation using deep learning. The
system exposes a FastAPI backend with support for image, video, webcam,
and RTSP stream processing, along with a web-based user interface.

## Features

-   Real-time vehicle detection using YOLO (Ultralytics)
-   Multi-object tracking using ByteTrack
-   Speed estimation based on object displacement over time
-   REST API for image and video inference
-   WebSocket support for live webcam and RTSP streams
-   Streaming video output with annotated frames
-   Lightweight web UI for interaction

## How It Works

1.  YOLO detects vehicles in each frame
2.  ByteTrack assigns persistent IDs to objects
3.  Speed is estimated using:

    speed = distance / time
4.  Pixel distance is converted to real-world distance using a
    calibration factor (pixels_per_meter)

## API Endpoints

Health Check: GET /health\
Image Detection: POST /api/detect/image\
Video Detection: POST /api/detect/video\
WebSocket Webcam: /ws/webcam\
WebSocket RTSP: /ws/rtsp

## Installation

git clone `<your-repo>`{=html}\
cd project\
pip install -r requirements.txt

## Run the Server

uvicorn api.main_api:create_app --factory --reload

## Model

Place your trained YOLO model at:
dataset/runs/detect/.../best.pt

## Calibration Note

pixels_per_meter = pixel_distance / real_distance
Without calibration, speed values are approximate.

## UI Disclaimer

The UI/index.html file was generated using AI tools and may require
refinement for production use.


import cv2
from Model.detector import VehicleDetector

MODEL_PATH = r"F:\Omar 3amora\حنكشة projects\TrafficGuard_ComputerVision_ETE_Proj\dataset\runs\detect\train5\weights\best.pt"
VIDEO_PATH = r"./Test_videos\0.0-30.2.mp4"

detector = VehicleDetector(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect(frame)
    result["counts"]["total"] = result["total"]
    cv2.imshow("Detection", result["frame"])
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
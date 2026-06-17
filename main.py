import cv2
import time

from detector import detect_objects
from tracker import track_vehicles
from speed_estimator import estimate_speed
from plate_reader import read_plate
from violation_detector import check_violation
from database import save_violation
from alert_system import send_alert

import os

print("Starting Smart Pedestrian System...")

VIDEO_PATH = os.path.join("videos", "road_video.mp4")

if not os.path.exists(VIDEO_PATH):
    print(f"❌ ERROR: Video file not found at '{VIDEO_PATH}'")
    exit(1)

video = cv2.VideoCapture(VIDEO_PATH)
if not video.isOpened():
    print(f"❌ ERROR: Cannot open video capture for '{VIDEO_PATH}'")
    exit(1)

print("✅ Video loaded successfully")

while True:

    ret, frame = video.read()

    if not ret:
        print("Video finished or cannot read frame")
        break

    print("Running detection...")
    try:
        vehicles, pedestrians = detect_objects(frame)
        print("Vehicles:", len(vehicles), "Pedestrians:", len(pedestrians))
    except Exception as e:
        # If the model fails to load (e.g., missing torch DLLs) we keep showing the
        # video frames and keep running.
        print("⚠️ Detection failed:", e)
        vehicles, pedestrians = [], []

    tracks = track_vehicles(vehicles)

    for track in tracks:

        x1,y1,x2,y2,track_id = map(int,track)

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        speed = estimate_speed(track_id,cx,cy)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(frame,
                    f"ID:{track_id} Speed:{speed}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,255),
                    2)

    for p in pedestrians:
        x1,y1,x2,y2 = p
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

    cv2.imshow("Smart Crossing AI System",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
# Lazy-load YOLO so importing this module does not block startup.
# This makes it easier to tell whether the video capture is failing vs.
# the model download / PyTorch load taking a long time.

import threading
import cv2

_model = None
_model_loader = None
_vehicle_classes = ["car", "bus", "truck", "motorcycle"]

# Background subtractor for a lightweight fallback detection method.
_bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)


def _do_load_model():
    """Load the YOLO model in a background thread."""
    global _model
    try:
        from ultralytics import YOLO
        print("Loading YOLO model...")
        _model = YOLO("yolov8n.pt")
        print("YOLO loaded successfully!")
    except Exception as e:
        print("❌ Failed to load YOLO model:", e)


def _load_model(timeout_s=5):
    """Return the YOLO model or None if it cannot be loaded quickly."""
    global _model, _model_loader

    if _model is not None:
        return _model

    if _model_loader is None or not _model_loader.is_alive():
        _model_loader = threading.Thread(target=_do_load_model, daemon=True)
        _model_loader.start()

    # Wait a short time for the model to be ready.
    print(f"Waiting up to {timeout_s}s for YOLO to finish loading...")
    _model_loader.join(timeout=timeout_s)

    if _model is None:
        print("YOLO model not ready yet; continuing without detection.")

    return _model


def _detect_motion(frame, min_area=400):
    """A lightweight fallback detector based on motion (background subtraction)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = _bg_subtractor.apply(gray)

    # Clean up the mask and find contours.
    _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    thresh = cv2.medianBlur(thresh, 5)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vehicles = []
    pedestrians = []

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        vehicles.append((x, y, x + w, y + h))

    return vehicles, pedestrians


def detect_objects(frame):
    model = _load_model()

    if model is None:
        # Model is still loading or failed to load; fall back to simple motion-based detection.
        return _detect_motion(frame)

    results = model(frame)[0]

    vehicles = []
    pedestrians = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        if label in _vehicle_classes:
            vehicles.append((x1, y1, x2, y2))

        if label == "person":
            pedestrians.append((x1, y1, x2, y2))

    return vehicles, pedestrians

"""
Map webcam gestures to two servo angles over serial (same wire protocol as servo_test.py:
one integer angle per line, e.g. "90\\n").

Uses models/weights/gesture_recognizer.task — labels depend on training:
  - "light" (and close variants) → ANGLE_LIGHT
  - "closed" / "Closed_Fist" (and close variants) → ANGLE_CLOSED

Requires: pyserial, opencv-python, mediapipe
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import mediapipe as mp
import serial
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "weights", "gesture_recognizer.task")

SERIAL_PORT = "/dev/ttyCH341USB0"
BAUD_RATE = 9600
ARDUINO_BOOT_DELAY_SEC = 2.0

ANGLE_LIGHT = 30
ANGLE_CLOSED = 120

GESTURE_SCORE_MIN = 0.55
STABLE_FRAMES = 6

# Normalized names (lowercase, underscores); custom folders like data/gestures/light
_LIGHT_LABELS = frozenset({"light"})
_CLOSED_LABELS = frozenset({"closed", "closed_fist"})


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def classify_binary_gesture(category_name: str) -> str | None:
    if not category_name or category_name == "None":
        return None
    n = _normalize(category_name)
    if n in _LIGHT_LABELS:
        return "light"
    if n in _CLOSED_LABELS:
        return "closed"
    return None


def main() -> None:
    if not os.path.isfile(MODEL_PATH):
        print(f"Error: missing model at {MODEL_PATH}")
        sys.exit(1)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open {SERIAL_PORT}: {e}")
        sys.exit(1)

    print(
        f"Serial open ({SERIAL_PORT} @ {BAUD_RATE}). "
        f"Waiting {ARDUINO_BOOT_DELAY_SEC:.0f}s for board reset…"
    )
    time.sleep(ARDUINO_BOOT_DELAY_SEC)

    base_options = python.BaseOptions(model_asset_path=os.path.abspath(MODEL_PATH))
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(0)
    prev_time = 0.0
    start_time = time.time()
    frame_count = 0

    pending_kind: str | None = None
    streak = 0
    last_sent_angle: int | None = None

    def send_angle(angle: int) -> None:
        nonlocal last_sent_angle
        if angle == last_sent_angle:
            return
        line = f"{int(angle)}\n"
        ser.write(line.encode())
        print(f"Sent servo angle: {angle}")
        last_sent_angle = angle

    print(
        f"Light gesture → {ANGLE_LIGHT}°, closed gesture → {ANGLE_CLOSED}°. "
        "'q' quits."
    )

    try:
        while cap.isOpened():
            ok, image = cap.read()
            if not ok:
                continue

            now = time.time()
            frame_count += 1
            fps = int(1 / (now - prev_time)) if prev_time > 0 else 0
            prev_time = now
            elapsed = now - start_time
            avg_fps = int(frame_count / elapsed) if elapsed > 0 else 0

            image = cv2.flip(image, 1)
            h, w, _ = image.shape

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize(mp_image)

            label = "None"
            score = 0.0
            kind: str | None = None
            if result.gestures and result.gestures[0]:
                top = result.gestures[0][0]
                label = top.category_name
                score = top.score
                if score >= GESTURE_SCORE_MIN:
                    kind = classify_binary_gesture(label)

            if kind is not None:
                if kind == pending_kind:
                    streak += 1
                else:
                    pending_kind = kind
                    streak = 1
            else:
                pending_kind = None
                streak = 0

            target: int | None = None
            if streak >= STABLE_FRAMES and pending_kind is not None:
                target = (
                    ANGLE_LIGHT if pending_kind == "light" else ANGLE_CLOSED
                )

            if target is not None:
                send_angle(target)

            hud = f"{label} ({score:.2f})  streak={streak}"
            if kind is None and label != "None" and score >= GESTURE_SCORE_MIN:
                hud += " (unmapped gesture)"
            cv2.rectangle(image, (0, 0), (w, 72), (0, 0, 0), -1)
            cv2.putText(
                image,
                hud,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if kind else (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            hint = (
                f"light→{ANGLE_LIGHT}°  closed→{ANGLE_CLOSED}°  "
                f"FPS {fps}  avg {avg_fps}"
            )
            cv2.putText(
                image,
                hint,
                (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (180, 220, 180),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Servo gesture", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()
        ser.close()


if __name__ == "__main__":
    main()

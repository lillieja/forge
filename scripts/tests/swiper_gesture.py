"""
Desktop swiper gated on open palm: draws hand landmarks (MediaPipe Hands), classifies
gestures with models/weights/gesture_recognizer.task, and fires Ctrl+Alt+Left/Right
when open palm holds and the wrist moves consistently far enough left or right
anywhere in the frame (not edge zones).

Requires: opencv-python, mediapipe, pyautogui
"""

from collections import deque
import os
import time

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "..", "models", "weights", "gesture_recognizer.task")

# Built-in task uses "Open_Palm"; custom trained folders often use "open".
_OPEN_PALM_NORMALIZED = frozenset({"open_palm", "open"})

COOLDOWN_SEC = 2.0
GESTURE_SCORE_MIN = 0.5

# Movement detection (normalized wrist x, 0–1 across frame width)
HISTORY_MAXLEN = 25
MIN_HISTORY_SAMPLES = 12
NET_MOVE_THRESHOLD = 0.17  # min |x_last - x_first| in window to count as a swipe
STEP_CONSISTENCY = 0.58  # share of frame-to-frame steps that agree with net direction


def _normalized_gesture_label(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def is_open_palm_gesture(category_name: str) -> bool:
    if not category_name or category_name == "None":
        return False
    return _normalized_gesture_label(category_name) in _OPEN_PALM_NORMALIZED


def swipe_direction_from_history(xs: list[float]) -> int:
    """Return -1 = desktop left, +1 = desktop right, 0 = no swipe."""
    if len(xs) < MIN_HISTORY_SAMPLES:
        return 0
    net = xs[-1] - xs[0]
    if abs(net) < NET_MOVE_THRESHOLD:
        return 0
    steps = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
    if not steps:
        return 0
    if net < 0:
        agree = sum(1 for d in steps if d < 0)
        if agree / len(steps) >= STEP_CONSISTENCY:
            return -1
    elif net > 0:
        agree = sum(1 for d in steps if d > 0)
        if agree / len(steps) >= STEP_CONSISTENCY:
            return 1
    return 0


def main() -> None:
    if not os.path.isfile(MODEL_PATH):
        print(f"Error: no model at {os.path.abspath(MODEL_PATH)}")
        print("Copy your trained gesture_recognizer.task there or train with scripts/training/gesture_training.py.")
        return

    base_options = python.BaseOptions(model_asset_path=os.path.abspath(MODEL_PATH))
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    last_swipe_time = 0.0
    prev_time = 0.0
    start_time = time.time()
    frame_count = 0
    x_history: deque[float] = deque(maxlen=HISTORY_MAXLEN)

    print(
        "Open palm, then drag hand smoothly left/right across the frame to switch desktops. "
        "'q' quits."
    )

    try:
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue

                curr_time = time.time()
                frame_count += 1
                fps = int(1 / (curr_time - prev_time)) if prev_time > 0 else 0
                prev_time = curr_time
                elapsed = curr_time - start_time
                avg_fps = int(frame_count / elapsed) if elapsed > 0 else 0

                image = cv2.flip(image, 1)
                h, w, _ = image.shape

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                hand_results = hands.process(rgb)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                rec = recognizer.recognize(mp_image)

                gesture_name = "None"
                gesture_score = 0.0
                if rec.gestures and rec.gestures[0]:
                    top = rec.gestures[0][0]
                    gesture_name = top.category_name
                    gesture_score = top.score

                palm_ok = (
                    gesture_score >= GESTURE_SCORE_MIN
                    and is_open_palm_gesture(gesture_name)
                )

                in_cooldown = (curr_time - last_swipe_time) < COOLDOWN_SEC

                if (
                    palm_ok
                    and hand_results.multi_hand_landmarks
                    and not in_cooldown
                ):
                    wrist_x = hand_results.multi_hand_landmarks[0].landmark[0].x
                    x_history.append(wrist_x)
                else:
                    x_history.clear()

                if hand_results.multi_hand_landmarks:
                    for landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, landmarks, mp_hands.HAND_CONNECTIONS
                        )

                xs = list(x_history)
                net_dx = xs[-1] - xs[0] if len(xs) >= 2 else 0.0

                status = f"{gesture_name} ({gesture_score:.2f})"
                if not palm_ok:
                    status += " — open palm to arm swiper"
                elif in_cooldown:
                    status += " — cooldown"
                else:
                    status += f" | dx={net_dx:+.2f} n={len(xs)}"

                cv2.rectangle(image, (0, 0), (w, 88), (0, 0, 0), -1)
                cv2.putText(
                    image,
                    status,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if palm_ok else (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    "Move open palm smoothly left / right (anywhere in frame)",
                    (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (200, 200, 100),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"FPS: {fps}  Avg: {avg_fps}",
                    (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

                if (
                    palm_ok
                    and hand_results.multi_hand_landmarks
                    and not in_cooldown
                ):
                    direction = swipe_direction_from_history(xs)
                    if direction == -1:
                        print("Open palm + sustained move left → desktop left")
                        pyautogui.hotkey("ctrl", "alt", "left")
                        last_swipe_time = curr_time
                        x_history.clear()
                    elif direction == 1:
                        print("Open palm + sustained move right → desktop right")
                        pyautogui.hotkey("ctrl", "alt", "right")
                        last_swipe_time = curr_time
                        x_history.clear()

                if in_cooldown:
                    cv2.putText(
                        image,
                        "COOLDOWN",
                        (w // 2 - 70, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                    )

                cv2.imshow("Swiper (open palm + hand)", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

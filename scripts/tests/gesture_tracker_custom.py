import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time

model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "weights", "gesture_recognizer.task")

if not os.path.exists(model_path):
    print(f"Error: Could not find the model at {model_path}")
    print("Make sure you are running this script from the 'gesture_project' root folder!")
    exit()

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
print("Starting Custom Gesture Recognizer... Press 'q' to quit.")

# Initialize time and frame variables
prev_time = 0
start_time = time.time()
frame_count = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # FPS Calculations
    curr_time = time.time()
    frame_count += 1
    
    # Current FPS
    fps = int(1 / (curr_time - prev_time)) if prev_time > 0 else 0
    prev_time = curr_time
    
    # Average FPS
    elapsed_time = curr_time - start_time
    avg_fps = int(frame_count / elapsed_time) if elapsed_time > 0 else 0

    image = cv2.flip(image, 1)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    recognition_result = recognizer.recognize(mp_image)

    top_gesture = "None"
    
    if recognition_result.gestures and recognition_result.gestures[0]:
        gesture_category = recognition_result.gestures[0][0]
        if gesture_category.category_name != "None":
            top_gesture = f"{gesture_category.category_name} ({gesture_category.score:.2f})"

    h, w, _ = image.shape
    cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 0), -1)
    
    # Put the gesture text (Top Left)
    cv2.putText(image, f"Custom: {top_gesture}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Put the FPS text (Top Right)
    cv2.putText(image, f"FPS: {fps} | Avg: {avg_fps}", (w - 280, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Custom Gesture Recognizer', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

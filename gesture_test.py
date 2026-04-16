import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time

# 1. Automatically download the pre-trained gesture model if it doesn't exist
model_path = 'gesture_recognizer.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe gesture model (this only happens once)...")
    url = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Configure the Gesture Recognizer
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# 3. Setup Camera
cap = cv2.VideoCapture(0) 
print("Starting Gesture Recognizer... Press 'q' to quit.")

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

    # Flip the image like a mirror
    image = cv2.flip(image, 1)

    # Convert the frame from OpenCV BGR format to MediaPipe's Image format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Run gesture recognition
    recognition_result = recognizer.recognize(mp_image)

    # 4. Extract and display the gesture
    top_gesture = "None"
    
    if recognition_result.gestures and recognition_result.gestures[0]:
        gesture_category = recognition_result.gestures[0][0]
        if gesture_category.category_name != "None":
            top_gesture = f"{gesture_category.category_name} ({gesture_category.score:.2f})"

    h, w, _ = image.shape
    cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 0), -1)
    
    # Put the gesture text (Top Left)
    cv2.putText(image, f"Gesture: {top_gesture}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
    # Put the FPS text (Top Right)
    cv2.putText(image, f"FPS: {fps} | Avg: {avg_fps}", (w - 280, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Gesture Recognizer', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

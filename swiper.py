import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup Camera
cap = cv2.VideoCapture(0)

# Swipe mechanics
cooldown_time = 2.0  # Wait 2 seconds between swipes
last_swipe_time = 0

print("Starting Desktop Swiper... Press 'q' to quit.")

# Initialize time and frame variables
prev_time = 0
start_time = time.time()
frame_count = 0

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

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

        # FLIP THE IMAGE
        image = cv2.flip(image, 1)
        h, w, c = image.shape

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Draw Trigger Zones
        cv2.rectangle(image, (0, 0), (int(w * 0.2), h), (255, 0, 0), 2)
        cv2.rectangle(image, (int(w * 0.8), 0), (w, h), (0, 0, 255), 2)

        # Display FPS text (Top Middle)
        cv2.putText(image, f"FPS: {fps} | Avg: {avg_fps}", (int(w/2) - 100, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                wrist_x = hand_landmarks.landmark[0].x

                if (curr_time - last_swipe_time) > cooldown_time:
                    if wrist_x < 0.2:
                        print("Swiped LEFT!")
                        pyautogui.hotkey('ctrl', 'alt', 'left') 
                        last_swipe_time = curr_time
                        
                    elif wrist_x > 0.8:
                        print("Swiped RIGHT!")
                        pyautogui.hotkey('ctrl', 'alt', 'right') 
                        last_swipe_time = curr_time

        if (curr_time - last_swipe_time) < cooldown_time:
            cv2.putText(image, "COOLDOWN...", (int(w/2)-80, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Gesture Swiper', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

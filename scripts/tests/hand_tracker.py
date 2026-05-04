import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open the webcam. 
# Try changing 0 to 1 or 2 if your Logitech camera isn't picked up immediately.
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0, # Fast model for Jetson CPU
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    print("Starting camera... Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # OpenCV captures in BGR, but MediaPipe needs RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Draw the hand annotations on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Draw the skeleton lines and dots
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS)
                
                # Extract and print the exact pixel coordinates of the Index Finger Tip
                index_tip = hand_landmarks.landmark[8]
                h, w, _ = image.shape
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                print(f"Index Finger Tip: X={cx}, Y={cy}")

        # Display the video feed
        cv2.imshow('MediaPipe Hand Tracking', image)
        
        # Press 'q' to close the window and exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

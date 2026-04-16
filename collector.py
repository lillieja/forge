import cv2
import os

# 1. Ask the user what gesture they are recording
print("--- MediaPipe Data Collector ---")
gesture_name = input("Enter the name of your custom gesture (e.g., spock): ").strip().lower()

# 2. Create the directory inside the existing 'dataset' folder
save_dir = os.path.join("dataset", gesture_name)
os.makedirs(save_dir, exist_ok=True)

# Count existing images so we don't overwrite them
count = len(os.listdir(save_dir))

# 3. Start Camera
cap = cv2.VideoCapture(0) # Change to 1 if camera doesn't open

print(f"\nSaving images to: {save_dir}/")
print("INSTRUCTIONS:")
print(" - Press SPACEBAR to take a picture.")
print(" - Press 'q' to quit and save.")
print(" - Remember to move your hand around and change angles between shots!\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip like a mirror
    frame = cv2.flip(frame, 1)
    
    # Create a copy of the frame to draw text on, so we don't save the text in the image
    display_frame = frame.copy()

    # Show the image counter on the screen
    cv2.putText(display_frame, f"Images collected: {count}/100+", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Data Collector', display_frame)

    # 4. Handle Keyboard Inputs
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '): # Spacebar pressed
        # Save the CLEAN frame (no text) to the folder
        img_path = os.path.join(save_dir, f"{gesture_name}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")
        count += 1
        
    elif key == ord('q'): # 'q' pressed
        break

cap.release()
cv2.destroyAllWindows()

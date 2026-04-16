import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

# 1. Download the DEFAULT Google model for the baseline test
default_model_path = 'default_gesture_recognizer.task'
if not os.path.exists(default_model_path):
    print("Downloading Google's default gesture model...")
    url = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
    urllib.request.urlretrieve(url, default_model_path)
    print("Download complete!")

# 2. Configure for IMAGE mode (Not Live Stream)
base_options = python.BaseOptions(model_asset_path=default_model_path)
# We use IMAGE mode here because we are processing static files, not a webcam feed
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE 
)

# 3. Setup the paths and counters
folder_path = "dataset/iloveyou"  # Update this if your path is slightly different!
correct_predictions = 0
total_images = 0

print(f"Starting evaluation on folder: {folder_path}...")

# Load the recognizer
with vision.GestureRecognizer.create_from_options(options) as recognizer:
    
    # Iterate through every file in your folder
    for filename in os.listdir(folder_path):
        # Only process image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        file_path = os.path.join(folder_path, filename)
        total_images += 1
        
        # Load the image using MediaPipe's built-in image loader
        try:
            mp_image = mp.Image.create_from_file(file_path)
            recognition_result = recognizer.recognize(mp_image)
        except Exception as e:
            print(f"Could not process {filename}: {e}")
            continue

        # Check if the model saw a hand AND guessed a gesture
        if recognition_result.gestures and recognition_result.gestures[0]:
            top_guess = recognition_result.gestures[0][0].category_name
            
            # Google's default model outputs exactly "ILoveYou" for this sign
            if top_guess == "ILoveYou":
                correct_predictions += 1
        
        # Optional: Print progress every 10 images so you know it's not frozen
        if total_images % 10 == 0:
            print(f"Processed {total_images} images...")

# 4. Calculate and display the final score
if total_images > 0:
    accuracy = (correct_predictions / total_images) * 100
    print("-" * 30)
    print(f"Total Images Evaluated: {total_images}")
    print(f"Correctly Identified:   {correct_predictions}")
    print(f"Default Model Accuracy: {accuracy:.2f}%")
else:
    print("No images found! Check your folder path.")

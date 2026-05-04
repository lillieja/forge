import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

# --- PATH CONFIGURATION ---
# Get the directory where THIS script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Model Path: Move Google's default model to the common weights folder
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "..", "models", "weights", "default_gesture_recognizer.task")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 2. Dataset Path: Point to the 'iloveyou' folder in the new data structure
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "gestures", "iloveyou")

# --- BASELINE MODEL SETUP ---
if not os.path.exists(MODEL_PATH):
    print(f"Downloading Google's default gesture model to {MODEL_PATH}...")
    url = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete!")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE 
)

# --- EVALUATION LOGIC ---
correct_predictions = 0
total_images = 0

if not os.path.exists(DATASET_PATH):
    print(f"Error: Folder '{DATASET_PATH}' not found. Check your structure!")
    exit(1)

print(f"Starting evaluation on: {DATASET_PATH}...")

with vision.GestureRecognizer.create_from_options(options) as recognizer:
    for filename in os.listdir(DATASET_PATH):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        file_path = os.path.join(DATASET_PATH, filename)
        total_images += 1
        
        try:
            mp_image = mp.Image.create_from_file(file_path)
            recognition_result = recognizer.recognize(mp_image)
            
            if recognition_result.gestures and recognition_result.gestures[0]:
                top_guess = recognition_result.gestures[0][0].category_name
                if top_guess == "ILoveYou":
                    correct_predictions += 1
        except Exception as e:
            print(f"Could not process {filename}: {e}")

# --- RESULTS ---
if total_images > 0:
    accuracy = (correct_predictions / total_images) * 100
    print("-" * 30)
    print(f"Total Images: {total_images} | Correct: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

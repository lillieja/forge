# F.O.R.G.E.: Fine-tuned Optical Reconstruction for Gesture Extraction

F.O.R.G.E. is a computer vision project that leverages OpenCV and Google's MediaPipe to perform real-time hand tracking, gesture recognition, and system control. It includes tools to build custom gesture datasets, train custom recognition models, and map physical hand movements to desktop keyboard actions.

## Project Structure & File Descriptions

### 1. Data Collection & Training
* **`collector.py`**
  A webcam-based tool to build your custom dataset. It prompts you for a gesture name, creates the necessary directory in the `dataset/` folder, and captures frames when you press the Spacebar.
* **`train_model.py`**
  Uses MediaPipe Model Maker to train a custom gesture recognition model on the images inside the `dataset/` directory. It splits the data for training, validation, and testing, and exports the final model to `custom_model/gesture_recognizer.task`.

### 2. Model Evaluation & Testing
* **`evaluate_default.py`**
  Downloads Google's default gesture recognition model and evaluates its accuracy against the static images stored in your local dataset (specifically looking for the "ILoveYou" sign).
* **`gesture_test.py`**
  Runs a live webcam feed to test the **default** MediaPipe gesture recognizer in real-time. It draws the current gesture and average FPS on the screen.
* **`gesture_test_custom.py`**
  Runs a live webcam feed to test your **custom-trained** model (`custom_model/gesture_recognizer.task`) in real-time, displaying the top detected gesture and FPS.

### 3. Applications & Utilities
* **`hand_tracker.py`**
  A fundamental hand-tracking script. It identifies hands in the webcam feed, draws the skeleton connections, and prints the exact real-time X/Y pixel coordinates of the index finger tip to the console.
* **`swiper.py`**
  A practical application that maps physical gestures to keyboard shortcuts. It tracks the X-coordinate of your wrist to detect left or right swipes and uses `pyautogui` to trigger desktop switching shortcuts (`Ctrl` + `Alt` + `Left/Right`).

## Prerequisites
To run these scripts, you will need the following Python libraries installed:
* `opencv-python` (cv2)
* `mediapipe`
* `mediapipe-model-maker` (for training)
* `pyautogui` (for the swiper tool)

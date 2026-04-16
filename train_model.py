import os
from mediapipe_model_maker import gesture_recognizer

# 1. Point to your dataset
dataset_path = "dataset"
print(f"Loading images from {dataset_path}...")

# Load the data and format it for MediaPipe
data = gesture_recognizer.Dataset.from_folder(
    dirname=dataset_path,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)

# Split the dataset: 80% for training, 10% for validation, 10% for testing
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# 2. Configure the training parameters
print("Starting the training process... This will take a few minutes.")
# We set epochs to 10 (it will pass over your data 10 times to learn)
hparams = gesture_recognizer.HParams(export_dir="custom_model", epochs=10)
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)

# 3. Train the model
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

# 4. Evaluate the model on the 10% test data it hasn't seen yet
loss, acc = model.evaluate(test_data)
print(f"\nFinal Test Accuracy: {acc * 100:.2f}%")

# 5. Export the final .task file
model.export_model()
print("Success! Your custom 'gesture_recognizer.task' is saved in the 'custom_model' folder.")

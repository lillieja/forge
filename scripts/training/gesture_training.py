import sys
from types import ModuleType
import tensorflow as tf

# Define the ghosts
ghosts = ['tensorflow_text', 'sentencepiece', 'tensorflow_addons']

for module_name in ghosts:
    if module_name not in sys.modules:
        mock = ModuleType(module_name)
        sys.modules[module_name] = mock
        
        # SPECIAL FIX: Redirect tensorflow_addons.optimizers to the real tf.keras.optimizers
        if module_name == 'tensorflow_addons':
            # Create a fake 'optimizers' submodule
            mock_optimizers = ModuleType('optimizers')
            # Point the fake's content to the real TF optimizers
            mock_optimizers.__dict__.update(tf.keras.optimizers.__dict__)
            sys.modules['tensorflow_addons.optimizers'] = mock_optimizers
            mock.optimizers = mock_optimizers
            
        print(f"Bypassed and patched {module_name}")
        
import tensorflow.python.tpu as tpu
if not hasattr(tpu, 'embedding_context_utils'):
    mock_tpu_utils = ModuleType('embedding_context_utils')
    # We add a dummy function to prevent attribute errors if called
    mock_tpu_utils.get_context = lambda: None 
    tpu.embedding_context_utils = mock_tpu_utils
    print("Patched tensorflow.python.tpu.embedding_context_utils")
        
import os
import shutil

import site
sys.path.append(site.getusersitepackages())

# Set environment variable for Keras to avoid compatibility issues
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Check dependencies
try:
    import numpy
    from mediapipe_model_maker import gesture_recognizer
    
    if numpy.version.version.startswith('2.'):
        raise ImportError("NumPy 2.x detected. MediaPipe Model Maker requires NumPy < 2.0.0")
    print("MediaPipe and NumPy loaded successfully!")
    
except (ImportError, RecursionError) as e:
    print(f"Dependency error: {e}")
    print("\nRun this command to fix dependencies:")
    print(f"{sys.executable} -m pip install 'numpy<2.0.0' mediapipe==0.10.11 mediapipe-model-maker")
    sys.exit(1)

# --- CONFIGURATION ---
# Points to the new data location
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "gestures")

# Points to the new models folder
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "custom")
# ---------------------

# 0. Clear previous model artifacts
if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)
    print(f'Cleared previous {EXPORT_DIR} directory.')

# 1. Verify Dataset Path
if not os.path.exists(DATASET_PATH):
    print(f"Error: Folder '{DATASET_PATH}' not found.")
    print("Ensure your images are organized in subfolders by label, e.g.:")
    print(f"  {DATASET_PATH}/rock/...")
    print(f"  {DATASET_PATH}/paper/...")
    sys.exit(1)

# 2. Load Data and Split
print(f"Loading dataset from: {os.path.abspath(DATASET_PATH)}...")
try:
    data = gesture_recognizer.Dataset.from_folder(
        dirname=DATASET_PATH,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
except Exception as e:
    print(f"Failed to load dataset: {e}")
    sys.exit(1)

train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# 3. Train the Model
print("Starting training (this may take a few minutes)...")
hparams = gesture_recognizer.HParams(export_dir=EXPORT_DIR, epochs=10)
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)

model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data, 
    validation_data=validation_data, 
    options=options
)

# 4. Export
print("Exporting model...")
model.export_model()

full_export_path = os.path.abspath(os.path.join(EXPORT_DIR, 'gesture_recognizer.task'))
print(f"\nSuccess! Training complete.")
print(f"Your model is ready at: {full_export_path}")

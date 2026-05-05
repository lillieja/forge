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
import random
from pathlib import Path
from typing import Iterable
import cv2

import site
sys.path.append(site.getusersitepackages())

# Set environment variable for Keras to avoid compatibility issues
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Check dependencies
try:
    import numpy
    from mediapipe_model_maker import gesture_recognizer
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    
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
OUR_TASK_PATH = os.path.join(EXPORT_DIR, "gesture_recognizer.task")
GOOGLE_TASK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "weights", "google_gesture_recognizer.task"
)
# ---------------------


def _norm_label(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum() or ch == "_")


def _make_eval_splits(
    dataset_root: str, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[tuple[Path, str]]]:
    root = Path(dataset_root)
    rng = random.Random(seed)
    train_samples: list[tuple[Path, str]] = []
    val_samples: list[tuple[Path, str]] = []
    test_samples: list[tuple[Path, str]] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        files = [p for p in sorted(label_dir.iterdir()) if p.suffix.lower() in exts]
        if not files:
            continue
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n_train <= 0:
            n_train = 1
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train = files[:n_train]
        val = files[n_train : n_train + n_val]
        test = files[n_train + n_val :]
        if not test and val:
            test = [val.pop()]
        label = label_dir.name
        train_samples.extend((p, label) for p in train)
        val_samples.extend((p, label) for p in val)
        test_samples.extend((p, label) for p in test)
    return train_samples, val_samples, test_samples


def _iter_top_predictions(task_path: str, samples: Iterable[tuple[Path, str]]) -> tuple[int, int]:
    if not os.path.isfile(task_path):
        print(f"Task not found, skipping eval: {task_path}")
        return 0, 0
    options = mp_vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task_path)
    )
    recognizer = mp_vision.GestureRecognizer.create_from_options(options)
    correct = 0
    total = 0
    try:
        for img_path, expected in samples:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize(mp_image)
            pred = "none"
            if result.gestures and result.gestures[0]:
                top = max(result.gestures[0], key=lambda c: float(c.score))
                pred = getattr(top, "category_name", "") or getattr(top, "display_name", "") or "none"
            if _norm_label(pred) == _norm_label(expected):
                correct += 1
            total += 1
    finally:
        recognizer.close()
    return correct, total


def _print_acc(prefix: str, correct: int, total: int) -> None:
    if total == 0:
        print(f"{prefix}: N/A (no samples)")
        return
    print(f"{prefix}: {correct}/{total} ({(100.0 * correct / total):.2f}%)")


def _without_label(
    samples: list[tuple[Path, str]], label_to_skip: str
) -> list[tuple[Path, str]]:
    skip_norm = _norm_label(label_to_skip)
    return [(p, y) for (p, y) in samples if _norm_label(y) != skip_norm]

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

full_export_path = os.path.abspath(OUR_TASK_PATH)

print("\nRunning validation/test comparison...")
_, val_samples, test_samples = _make_eval_splits(DATASET_PATH)
our_val_c, our_val_n = _iter_top_predictions(full_export_path, val_samples)
our_test_c, our_test_n = _iter_top_predictions(full_export_path, test_samples)
val_samples_google = _without_label(val_samples, "light")
test_samples_google = _without_label(test_samples, "light")
goog_val_c, goog_val_n = _iter_top_predictions(
    os.path.abspath(GOOGLE_TASK_PATH), val_samples_google
)
goog_test_c, goog_test_n = _iter_top_predictions(
    os.path.abspath(GOOGLE_TASK_PATH), test_samples_google
)

print(f"\nSuccess! Training complete.")
print(f"Your model is ready at: {full_export_path}")
print("\nAccuracy comparison on held-out splits:")
_print_acc("Our model   | Validation", our_val_c, our_val_n)
_print_acc("Our model   | Test      ", our_test_c, our_test_n)
_print_acc("Google model| Validation", goog_val_c, goog_val_n)
_print_acc("Google model| Test      ", goog_test_c, goog_test_n)

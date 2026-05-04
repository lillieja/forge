# F.O.R.G.E.

**Fine-tuned Optical Reconstruction for Gesture Extraction** — a computer-vision project using OpenCV and [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) for hand tracking, gesture recognition, dataset capture, optional custom training, and small hardware/utility demos.

Run commands from the **repository root** (`forge/`) unless noted otherwise.

---

## Repository layout

High-level structure (depth 3):

```text
forge/
├── data/gestures/          # Image dataset: one subfolder per class label
│   ├── closed/
│   ├── iloveyou/
│   ├── light/
│   ├── none/
│   └── open/
├── models/
│   ├── weights/            # Shipped or downloaded .task models for inference
│   │   ├── default_gesture_recognizer.task   # Google’s default bundle (or auto-downloaded)
│   │   └── gesture_recognizer.task           # Copy used by some test scripts
│   └── custom/             # Output of training (checkpoints, logs, gesture_recognizer.task, …)
├── scripts/
│   ├── collection/         # Capture frames into data/gestures/
│   ├── tests/              # Offline eval + live webcam demos
│   └── training/           # MediaPipe Model Maker training
├── hardware/               # Serial / servo demos (some pair with gesture + camera)
├── .tests/                 # uv — inference, collection, evaluation (see below)
├── .training/              # uv — MediaPipe Model Maker / TensorFlow training stack
└── README.md
```

### `data/gestures/`

Folders named after gesture **labels**; each contains `.jpg`/`.png` images. Used by `scripts/collection/collector.py`, `scripts/training/*.py`, and `scripts/tests/evaluate_default.py` (expects `iloveyou/` for the packaged baseline check).

### `models/`

| Path | Role |
|------|------|
| `models/weights/` | Static or one-time–downloaded `.task` files for **GestureRecognizer** (default and shared weights). |
| `models/custom/` | **Training output**: `gesture_recognizer.task`, checkpoints, TensorBoard logs, `metadata.json`, etc. `scripts/training/gesture_training.py` clears this directory before each run. |

Live demos that use a **custom** model read `models/weights/gesture_recognizer.task` — copy or symlink your export there for **`gesture_tracker_custom.py`**, **`swiper_gesture.py`**, and **`hardware/servo_gesture.py`**.

### `scripts/collection/`

| Script | Purpose |
|--------|---------|
| `collector.py` | Webcam UI: prompt for a label, save mirrored frames with **Space**, quit with **q**. Writes under `data/gestures/<label>/`. |

### `scripts/tests/`

| Script | Purpose |
|--------|---------|
| `evaluate_default.py` | Loads `models/weights/default_gesture_recognizer.task` (downloads if missing), runs **image** mode over `data/gestures/iloveyou/`, reports accuracy vs label **ILoveYou**. |
| `gesture_tracker_default.py` | Webcam: real-time **default** gesture model; may download `models/weights/gesture_recognizer.task`. |
| `gesture_tracker_custom.py` | Webcam: real-time recognizer using `models/weights/gesture_recognizer.task` (your trained export). |
| `hand_tracker.py` | Webcam: classic MediaPipe **Hands** landmarks + skeleton; prints index-tip coordinates. |
| `swiper.py` | Webcam: wrist motion in screen **edge zones** → **pyautogui** desktop switch (`Ctrl+Alt+Left/Right`) with cooldown. |
| `swiper_gesture.py` | Webcam: **open palm** + sustained left/right **drag** (anywhere in frame) + same desktop hotkeys; MediaPipe **Hands** + `models/weights/gesture_recognizer.task`. |

### `scripts/training/`

| Script | Purpose |
|--------|---------|
| `gesture_training.py` | **Primary** trainer: applies TF/Keras compatibility shims, reads `data/gestures`, trains with MediaPipe Model Maker, exports to `models/custom/gesture_recognizer.task`. **Deletes** existing `models/custom/` first. |
| `train_model.py` | Simpler trainer that exports into a `custom_model/` folder relative to the process working directory; paths are easy to misalign with the rest of the repo — prefer `gesture_training.py` for the layout above. |

### `hardware/`

| File | Purpose |
|------|---------|
| `arduino.ino` | **Arduino sketch** (upload to the board connected to the servo): listens on **Serial @ 9600 baud** for newline-terminated integers **0–180** (`Serial.parseInt`), drives a **servo on pin 9**. Load this onto the Arduino so the **main computer** (e.g. Jetson running the Python scripts below) can command the servo over USB serial. |
| `servo_test.py` | Writes angles over **pyserial** to a fixed port (`/dev/ttyCH341USB0` @ 9600 baud); sweep example for an attached MCU. |
| `servo_gesture.py` | Webcam + `models/weights/gesture_recognizer.task`: **light** vs **closed** gesture (debounced) → two servo angles; same serial line protocol as `servo_test.py`. |

---

## Python environments

This repo uses **two** local virtual environments:

| Directory | Typical use | Notes |
|-----------|-------------|--------|
| **`.tests`** | Collection, all `scripts/tests`, offline eval | Created with **uv**. Includes MediaPipe Tasks, OpenCV, pyautogui, and a large TensorFlow-related dependency tree used elsewhere — but **not** `mediapipe-model-maker` (see below). |
| **`.training`** | `scripts/training` (Model Maker) | Standard **venv** with `mediapipe-model-maker`, TensorFlow 2.15 stack, etc. Training has heavy and fragile dependency pins; keep it separate from `.tests` if that works for your machine. |

Activate from the repo root:

```bash
source .tests/bin/activate      # or: source .training/bin/activate
```

Example runs:

```bash
source .tests/bin/activate
python scripts/tests/hand_tracker.py

source .training/bin/activate   # when training — resolve any missing extras (e.g. tensorflow_text) per upstream Model Maker docs
python scripts/training/gesture_training.py
```

---

## Dependencies

Most scripts need **Python 3.10+**, **MediaPipe**, **OpenCV** (`opencv-python`), and a **NumPy** pin compatible with MediaPipe (commonly `numpy<2`). **Swiper** scripts also need **pyautogui**. **Hardware** scripts need **pyserial**.

Example minimal install:

```bash
uv venv .tests && source .tests/bin/activate
uv pip install "numpy<2" mediapipe opencv-python pyautogui pyserial
```

**Training** (`scripts/training/gesture_training.py`): use the **`.training`** environment and install **mediapipe-model-maker** plus its TensorFlow stack per [MediaPipe Model Maker](https://developers.google.com/mediapipe/solutions/model_maker) and your OS/GPU notes.

List packages in an existing venv: `uv pip list --python .tests/bin/python` (or `.training/bin/python`).

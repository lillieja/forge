#!/usr/bin/env bash
# Extract gesture_recognizer.task -> TFLite -> ONNX -> TensorRT FP16 engines.
# Produces four TRT engines: hand_detector, landmarks, embedder, classifier.
# The detector outputs raw SSD tensors; host code must decode/NMS/ROI (see mp_palm.py / swiper_opt).
# Run from anywhere; paths are relative to repo root.
#
# This is a bash script (not Python):  ./scripts/tests/build_gesture_engines.sh
#
# Install converter deps into .tests (uv; this venv may not have pip):
#   cd /path/to/forge && uv pip install tf2onnx --python .tests/bin/python3
# TensorFlow is only needed for TFLite->ONNX; if missing, add: tensorflow

set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${PROJECT}/models/weights/extracted_tflite"
TRTEXEC="$(command -v trtexec 2>/dev/null || echo /usr/src/tensorrt/bin/trtexec)"

# Prefer repo .tests Python so tf2onnx lands in the same env swiper_opt uses.
if [[ -x "${PROJECT}/.tests/bin/python3" ]]; then
  PYTHON3="${PROJECT}/.tests/bin/python3"
else
  PYTHON3="$(command -v python3)"
fi

if ! "${PYTHON3}" -c "import tf2onnx" 2>/dev/null; then
  echo "Missing Python package 'tf2onnx'. From repo root run:" >&2
  echo "  uv pip install tf2onnx --python .tests/bin/python3" >&2
  exit 1
fi

echo "PROJECT=${PROJECT}"
echo "Using python: ${PYTHON3}"
echo "Using trtexec: ${TRTEXEC}"

mkdir -p "${OUT}/outer" "${OUT}/landmarker" "${OUT}/gesture"

unzip -o "${PROJECT}/models/weights/gesture_recognizer.task" -d "${OUT}/outer"
unzip -o "${OUT}/outer/hand_landmarker.task" -d "${OUT}/landmarker"
unzip -o "${OUT}/outer/hand_gesture_recognizer.task" -d "${OUT}/gesture"

cp "${OUT}/landmarker/hand_detector.tflite" "${OUT}/hand_detector.tflite"
cp "${OUT}/landmarker/hand_landmarks_detector.tflite" "${OUT}/landmarks.tflite"
cp "${OUT}/gesture/gesture_embedder.tflite" "${OUT}/embedder.tflite"
cp "${OUT}/gesture/custom_gesture_classifier.tflite" "${OUT}/classifier.tflite"

"${PYTHON3}" -m tf2onnx.convert --tflite "${OUT}/hand_detector.tflite" \
  --output "${OUT}/hand_detector.onnx" --opset 13
"${PYTHON3}" -m tf2onnx.convert --tflite "${OUT}/landmarks.tflite" \
  --output "${OUT}/landmarks.onnx" --opset 13
"${PYTHON3}" -m tf2onnx.convert --tflite "${OUT}/embedder.tflite" \
  --output "${OUT}/embedder.onnx" --opset 13
"${PYTHON3}" -m tf2onnx.convert --tflite "${OUT}/classifier.tflite" \
  --output "${OUT}/classifier.onnx" --opset 13

"${TRTEXEC}" --onnx="${OUT}/hand_detector.onnx" --saveEngine="${OUT}/hand_detector.engine" --fp16
"${TRTEXEC}" --onnx="${OUT}/landmarks.onnx" --saveEngine="${OUT}/landmarks.engine" --fp16
"${TRTEXEC}" --onnx="${OUT}/embedder.onnx" --saveEngine="${OUT}/embedder.engine" --fp16
"${TRTEXEC}" --onnx="${OUT}/classifier.onnx" --saveEngine="${OUT}/classifier.engine" --fp16

cp -f "${OUT}/hand_detector.engine" "${OUT}/landmarks.engine" "${OUT}/embedder.engine" \
  "${OUT}/classifier.engine" "${PROJECT}/models/weights/"

echo "Done. Engines copied to ${PROJECT}/models/weights/"

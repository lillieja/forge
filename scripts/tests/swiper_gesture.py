import sys
import os

# --- 1. Environment Path Setup ---
sys.path.append(os.path.expanduser('~/.local/lib/python3.10/site-packages'))
sys.path.append('/usr/lib/python3.10/dist-packages')


import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pyautogui
from collections import deque



# --- 2. Configuration ---
PROJECT_ROOT = "/home/group3/projects/forge"
LANDMARK_ENGINE_PATH = os.path.join(PROJECT_ROOT, "models/weights/default_landmarker.engine")
EMBEDDER_ENGINE_PATH = os.path.join(PROJECT_ROOT, "models/weights/default_gestures.engine")
CLASSIFIER_ENGINE_PATH = os.path.join(PROJECT_ROOT, "models/weights/default_classifier.engine")

LM_INPUT_SHAPE = (224, 224) 
OPEN_PALM_ID = 2          # Canned Classifier ID for Open Palm
GESTURE_SCORE_MIN = 0.60  

# Swipe Sensitivity
COOLDOWN_SEC = 2.0
HISTORY_MAXLEN = 20
MIN_HISTORY_SAMPLES = 10
NET_MOVE_THRESHOLD = 0.17 
STEP_CONSISTENCY = 0.60   

# --- 3. TRT 10.x Engine Wrapper ---
class TRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.stream = [], [], cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            tensor_info = {'name': name, 'host': host_mem, 'device': device_mem}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor_info)
            else:
                self.outputs.append(tensor_info)

    def run(self, input_data):
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        for t in self.inputs + self.outputs:
            self.context.set_tensor_address(t['name'], int(t['device']))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host']

def swipe_direction_from_history(xs: list[float]) -> int:
    if len(xs) < MIN_HISTORY_SAMPLES: return 0
    net = xs[-1] - xs[0]
    if abs(net) < NET_MOVE_THRESHOLD: return 0
    steps = [xs[i] - xs[i-1] for i in range(1, len(xs))]
    if net < 0:
        agree = sum(1 for d in steps if d < 0)
        if agree / len(steps) >= STEP_CONSISTENCY: return -1
    elif net > 0:
        agree = sum(1 for d in steps if d > 0)
        if agree / len(steps) >= STEP_CONSISTENCY: return 1
    return 0

# --- 4. Main Execution ---
def main():
    # Load all three engines
    lm_det = TRTEngine(LANDMARK_ENGINE_PATH)
    gs_emb = TRTEngine(EMBEDDER_ENGINE_PATH)
    gs_cls = TRTEngine(CLASSIFIER_ENGINE_PATH)

    cap = cv2.VideoCapture(0)
    last_swipe_time = 0.0
    prev_time = 0.0
    x_history = deque(maxlen=HISTORY_MAXLEN)

    print("Triple-Engine Official Canned Swiper Active.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        # Step 1: Landmarks (Pixels -> 63)
        img_in = cv2.resize(frame, LM_INPUT_SHAPE).astype(np.float32) / 255.0
        img_in = np.transpose(img_in, (2, 0, 1))
        lms = lm_det.run(img_in)
        
        # Normalize Wrist X (using your calibrated range)
        wrist_x = np.clip((lms[0] - 120) / -25.0, 0.0, 1.0)

        # Step 2: Embedder (63 -> 128) - This fixes the ValueError
        embedding = gs_emb.run(lms)

        # Step 3: Classifier (128 -> Probability IDs)
        probs = gs_cls.run(embedding)
        gid = np.argmax(probs)
        score = probs[gid]

        palm_ok = (gid == OPEN_PALM_ID and score >= GESTURE_SCORE_MIN)
        in_cooldown = (now - last_swipe_time) < COOLDOWN_SEC

        if palm_ok and not in_cooldown:
            x_history.append(wrist_x)
            direction = swipe_direction_from_history(list(x_history))
            if direction == -1:
                print("SWIPE LEFT")
                pyautogui.hotkey("ctrl", "alt", "left")
                last_swipe_time, x_history.clear()
            elif direction == 1:
                print("SWIPE RIGHT")
                pyautogui.hotkey("ctrl", "alt", "right")
                last_swipe_time, x_history.clear()
        else:
            x_history.clear()

        # UI Visuals
        fps = int(1 / (now - prev_time)) if prev_time > 0 else 0
        prev_time = now
        color = (0, 255, 0) if palm_ok else (0, 0, 255)
        status = f"ID: {gid} | Score: {score:.2f} | WristX: {wrist_x:.2f} | FPS: {fps}"
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("TRT Triple Engine", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
